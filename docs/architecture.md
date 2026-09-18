# CalMetricsEngine Architecture

## 1. 定位

CalMetricsEngine 是基金投研平台的独立计算引擎。长期目标是统一承载：

```text
DSL / AST
    ↓
Typed DAG
    ↓
Type & Shape Inference
    ↓
Operator Lowering
    ↓
Native Execution Plan
    ↓
PyBind11 / C++ AOT Runtime
```

基金投研平台负责业务页面、数据来源、业务模板与领域编排；CalMetricsEngine 负责可复用的计算语义、执行计划和原生数值执行。

当前生产级 DSL / AST / Typed DAG 的事实来源仍在 FundInvestmentResearchPlatform。迁移时应抽取并保持契约一致，不能在本仓库重新设计一套不兼容的 DSL。

## 2. 架构边界

### Python 层

未来负责：

- DSL 解析与受限 AST。
- Typed DAG 构建。
- operator contract、dtype、axis、shape、NaN、causality 元数据。
- DAG 校验、cost model、execution-plan lowering。
- 调用一次 native plan，而不是每个节点反复跨 Python/C++ 边界。

Python 不负责大规模数值循环，不做节点内部数据复制，也不生成运行时 JIT 代码。

### C++ 层

负责：

- AOT 编译的 primitive / statistics / rolling / matrix / finance kernels。
- strided readonly ArrayView。
- NativeExecutionPlan 执行。
- workspace 生命周期与 buffer 复用。
- 有边界的 CPU 并行。
- GIL 释放期间的完整数值路径。

当前仓库已有 finance kernels，后续 primitive operator 迁移时必须复用同一 ArrayView、线程和内存契约。

## 3. 零拷贝契约

计算输入必须在平台数据边界完成 dtype 规范化。进入 CalMetricsEngine 后：

- 只接受 NumPy ndarray。
- dtype 必须精确匹配 native contract。
- 不使用 `np.asarray(..., dtype=...)`、`np.require`、`np.ascontiguousarray` 做隐式修复。
- C/F contiguous、普通 slice、负 stride、列视图均通过 `data + shape + strides` 直接读取。
- readonly 输入合法。
- 输入生命周期由 Python owner 保持到 native 调用结束。
- 输入复制目标为 0。
- 输出数组和必要 workspace 可以分配。
- workspace 应由执行计划基于 liveness 复用，避免每个 DAG node 独立分配长期中间数组。

无法满足 dtype/alignment 的输入直接失败，由调用方在统一数据入口显式转换一次。

### SIMD 高性能输入布局

通用 API 继续支持 strided zero-copy view；重型 SIMD batch 的首选布局更严格：

```text
values  = [product_0][product_1]...[product_n]
dates   = [product_0][product_1]...[product_n]
offsets = [0, p0_end, p1_end, ..., total_observations]
```

C++ 通过 `data + offsets[product] + interval_start` 直接定位数据。不同区间通过
`product_id/start/end` 元数据描述，不复制窗口数组。多字段采用 SoA。

NaN 不因性能原因被删除或填零。执行层按 operator missing policy 选择 dense SIMD、
masked SIMD、valid-span 或 scalar fallback。

## 4. 目录结构

```text
CalMetricsEngine/
├── src/calmetrics_engine/
│   ├── __init__.py
│   ├── _api.py
│   └── _native.*            # 唯一 PyBind11 扩展
├── cpp/
│   ├── bindings.cpp
│   ├── include/calmetrics_engine/
│   │   ├── array_view.hpp   # strided zero-copy views
│   │   ├── parallel.hpp
│   │   ├── numeric.hpp
│   │   ├── calendar.hpp
│   │   └── finance.hpp
│   └── finance/
│       ├── statistics.cpp
│       ├── drawdown.cpp
│       ├── streaks.cpp
│       └── rolling.cpp
├── tests/
├── tools/
└── docs/
```

后续抽取 Typed DAG 后再增加 `compiler/`、`operators/`、`runtime/`，不提前创建空壳模块。

## 5. NativeExecutionPlan 目标

不采用：

```text
Python node → PyBind → C++ → Python node → PyBind → C++
```

也不采用：

```text
AST → generated Python source → exec → numba.njit
```

目标：

```text
Typed DAG
    ↓ lower once
NativeExecutionPlan
    ↓ one native call
C++ DAG executor
    ├── operator dispatch
    ├── readonly input views
    ├── workspace reuse
    └── output materialization
```

执行计划应包含稳定 opcode、输入槽位、参数槽位、输出槽位、shape/dtype contract 和 workspace layout，不包含 Python callback。

## 6. Operator 分层

- Primitive：add/subtract/multiply/divide、comparison、logical、where。
- Reduction：sum/mean/std/variance/min/max/quantile。
- Time series：lag/difference/rolling/scan/drawdown。
- Matrix：dot/matmul/matvec/solve/covariance/correlation。
- Finance：portfolio、risk、drawdown、fund analytics。
- Coupled kernel：只有不可拆的递推、拟合、联合约束才允许成为黑盒 native kernel。

通用算子只有一份数值实现，业务中心不得复制。

## 7. SIMD 与统一执行调度

SIMD 采用 portable baseline + runtime dispatch，不用全局 host-specific ISA：

```text
x86_64: baseline → AVX2 → AVX-512
arm64:  NEON baseline
fallback: scalar
```

是否增加某个 ISA 路径必须由 benchmark 和数值等价测试证明。

重型任务由一个 ExecutionScheduler 决定并行层级，业务模块不能独立创建互相竞争的
process/thread pools。Scheduler 至少基于：

- 产品数
- 区间数
- DAG cost
- 时间长度
- scenario 数
- 输入与 workspace bytes
- CPU / 内存预算
- hard-stop / 隔离要求

调度层级：

```text
small:
    single thread + SIMD

medium:
    one process + native thread pool + SIMD

large shared dataset:
    process pool + shared memory/mmap
        ↓
    per-process native thread budget
        ↓
    SIMD
```

产品块通常优先于指标维度并行，因为多个指标会共享 DAG 上游。单产品大量区间时，
区间块可以成为线程调度维度。

ProcessPool 场景下，多 worker 共同读取的大数组必须优先使用 SharedMemory/mmap，
worker 只接收描述符、offsets 和任务区间，禁止 pickle 整块行情数组。

`process_count × threads_per_process` 受统一 CPU budget 约束，禁止过度订阅。

## 8. 构建与平台

- C++17 + pybind11 + scikit-build-core + CMake。
- AOT wheel，不允许运行时编译。
- CPython 3.10–3.14。
- Linux x86_64/aarch64、macOS x86_64/arm64、Windows AMD64。
- 发布 wheel 不使用 `-march=native` 或强制 AVX。
- ISA 优化只能通过安全 baseline 或经过测试的 runtime dispatch 增加。

## 9. 当前版本边界

0.3.0 完成计算引擎身份和 native runtime 基础：

- 项目、distribution、Python package、C++ namespace 统一为 CalMetricsEngine。
- 原 `my_ctools` import 不再作为当前 API。
- native extension 改为 `calmetrics_engine._native`。
- 输入从“规范化后再计算”升级为 exact-dtype strided zero-copy。
- 现有 finance kernels 保持原金融计算口径。

0.3.0 不复制 FundInvestmentResearchPlatform 的 AST/DAG 实现。后续迁移必须先做契约映射与等价测试，再把生产事实来源逐步下沉到本引擎。
