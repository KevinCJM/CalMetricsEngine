# Phase 2 Execution Graph 验收记录

> 历史验收快照：本文对应 Python 控制层 + C++ 计算图的阶段二实现。当前编译、规划、线程/进程和共享内存均已下沉 C++；最新结果见 `cpp-first-acceptance.md`。本文保留原始测试口径和数据，不代表当前运行时分层。

## 1. 已完成范围

Phase 2 已把 CalMetricsEngine 从原生算子库扩展为可规划、可调度的计算图运行时：

```text
restricted AST
  ↓
shared multi-root DAG
  ↓ CSE + liveness
ExecutionPlan
  ↓
AdaptiveScheduler
  ├─ single native call
  ├─ persistent thread pool
  ├─ process pool + SharedMemory
  ├─ isolated hard-stop process
  └─ asyncio coroutine orchestration
        ↓
one C++ graph call per chunk
        ↓
118 canonical operators + SIMD
```

本阶段没有修改 BetterSaaTaa，也没有迁移其生产 Typed DSL、semantic axis、price basis 或 causality/knowledge-time 契约。Phase-2 parser 是受限数学 compiler，不是第二套业务 DSL。

## 2. AST / DAG / 内存规划

- 只允许数值常量、已声明变量、canonical operator、基础算术和单一比较。
- attribute/subscript/lambda/comprehension/任意 Python callable 等均 fail closed。
- 多个指标 root 一次构图并做结构 CSE。
- scalar runtime parameter 使用参数槽，不因参数值变化重编 graph。
- series/mask 中间结果做 last-use liveness，复用 worker-local arena slot。
- `lag` 等 borrowed view 不占 arena。
- graph chunk 内没有 Python operator callback。

最终 16 指标图在 composite/regression lowering 后：

```text
raw node attempts     147
unique DAG nodes       42
CSE eliminated        105
numeric arena slots     2
mask arena slots        1
SIMD-eligible nodes    10
```

`total_return` 保留高效 canonical 单次扫描，`annualized_return` 复用其结果；
`linear_slope / linear_r_squared` 等共享一个 `linear_fit` state，避免重复拟合。

## 3. 自动调度策略

`AdaptivePlanner` 同时考虑 graph cost、产品数、区间数、最大窗口、输入/输出字节、worker scratch、CPU budget、memory budget、Hard Stop 和输入是否已共享。

默认策略经本机 benchmark 校准：

- 小图：single native chunk。
- CPU 图：优先持久有界 thread pool；C++ graph chunk 全程释放 GIL。
- 超大 work/input 或隔离需求：process pool。
- process + 大输入：SharedMemory descriptors，不 pickle 大数组。
- Hard Stop：独立 disposable process + SharedMemory，可 timeout 后 terminate。
- coroutine：只负责编排和等待，不执行 Python 数值循环。
- 有足够产品时按 product 分块；产品少、区间多时按 interval 分块。
- metric 不作为默认并行维度，保持 shared DAG 上游复用。
- Phase 2 默认 process × thread = N × 1，不启用嵌套 hybrid。

Scheduler 使用全局 CPU token budget。同步、线程、进程和多个 coroutine job 共用同一预算，不允许每个异步任务各自把核心开满。

## 4. SharedMemory 契约

- Parent 创建、持有并 unlink 自动创建的共享内存。
- Worker 只 attach/close，不负责 unlink。
- 原始 NumPy → SharedMemory 的一次边界复制显式计入 `boundary_copy_bytes`。
- `SharedInputBundle` 可提前创建并跨多次 process execution 复用，避免重复输入复制。
- process worker 接收 descriptor + row range，不传 raw pointer。
- shared output 每个 worker 写互不重叠的行；最终复制回调用方结果一次，并计入 `output_copy_bytes`。

## 5. 正确性与安全验证

当前 Python 全量回归：**4034 passed**。发布候选 sdist/wheel 实际安装到 CPython 3.10 / 3.11 / 3.12 / 3.13 / 3.14，以及 CPython 3.12 + NumPy 1.26.4，均为 **4034 passed**。

新增 Phase-2 测试覆盖：

- restricted AST malicious syntax rejection；
- multi-root CSE / liveness slot reuse；
- native graph vs NumPy；
- single / thread / process + SharedMemory 数值一致；
- product / interval 自动分块；
- memory budget 降低并行度；
- Hard Stop isolated process 正常路径与 timeout terminate；
- coroutine 多任务；
- scheduler-wide CPU token budget；
- SharedMemory owner unlink。

Native C++：ARM64 Release CTest **3/3 passed**；ARM64 ASan/UBSan CTest **3/3 passed**；x86_64 交叉编译并在 Rosetta 执行 Release CTest **3/3 passed**，其中均包含独立 graph executor 测试。

## 6. 数值一致性

Phase-2 benchmark 使用 BetterSaaTaa 真实 `CompiledNumbaBatchPlan` 与 CalMetricsEngine Native Graph 计算同一组 16 个公式。

500 产品 × 12 区间 × 16 指标、1000 产品 × 12 区间 × 16 指标中：

```text
max absolute error = 0
```

所有 single/thread/process/shared-memory 路线均与 NJIT reference 一致。

## 7. 完整 workload 性能

环境：macOS ARM64、10 CPU cores、Python 3.12、NumPy 1.26.4、Numba 0.60.0。输入和输出在计时前准备；NJIT warm compute 与 Native warm compute 对比。

### 性能硬门槛

按项目要求，warm C++ 性能**不得低于**生产 NJIT。仓库进一步将验收阈值设为：

```text
Prepared Native / NJIT <= 0.90
ordinary Scheduler.execute / NJIT <= 1.00   # required micro workloads
```

冷启动无 JIT、低内存不能用于掩盖 warm compute 变慢。门槛同时覆盖“大型 shared-DAG batch”和“1产品×1区间×5指标”的极小任务。

微型任务必须同口径比较：BetterSaaTaa 使用已经 compile/warm 的 `CompiledNumbaBatchPlan.compute()`，CalMetricsEngine 的绝对热路径使用已经 plan/bind 的 `PreparedGraphExecution.run()`。同时普通 `Scheduler.execute()` 也必须不慢于 NJIT，防止公共自适应入口因 Python 控制面开销发生性能回退。

最终 Planner 使用真实区间 observation 总量估算 cost，并按产品总 observation workload 加权分块。paired benchmark 交替 NJIT / Native 的先执行顺序，降低热状态偏差。

| Workload | NJIT paired median | Native paired median | Native/NJIT | 结果 |
| --- | ---: | ---: | ---: | --- |
| 500 × 2520 × 12 × 16 | 38.08 ms | **27.22 ms** | **0.715×** | Native 快约 28.5% |
| 1000 × 2520 × 12 × 16 | 80.69 ms | **51.93 ms** | **0.644×** | Native 快约 35.6% |
| 2000 × 2520 × 12 × 16 | 152.91 ms | **100.25 ms** | **0.656×** | Native 快约 34.4% |

三档规模均明显低于 0.90 硬门槛。500/1000 为必测 gate，2000 为扩展 scale check。

单独 lane 的同轮 warm 中位数也显示 thread-first 合理：500 产品 thread 31.24 ms、process+SharedMemory 35.54 ms；1000 产品 thread 50.28 ms、process 51.22 ms；2000 产品 thread 100.12 ms、process 111.96 ms。

### 1产品 × 1区间 × 5指标

指标为平均收益、样本波动率、中位数收益、最大回撤、NAV R²。这个 workload 用来专门检查固定调用开销。

初始拆分测量发现：

```text
63 observations:
NJIT compute              ≈ 4.4 µs
native Program.execute    ≈ 3.6 µs
old Scheduler.execute     ≈ 8.1 µs
```

说明原先短区间“C++慢”不是数值核问题，而是 Python 控制面固定开销。后续完成四层优化：

1. `prepare_execution()` 预绑定 array/owner/dtype/shape/parameter/native program/CPU budget；
2. ordinary single-lane `execute()` 自动缓存固定 native batch；
3. static audit 缓存，动态调用只返回 queue/compute 时间；
4. C++ 直接创建本次独立 output ndarray，避免 Python 结果 copy。

因此调用方既可以用 `PreparedGraphExecution.run()` 取得最低热循环开销，也可以继续使用普通 `Scheduler.execute()` 而不低于 NJIT。

| 单区间长度 | NJIT | Prepared Native | Native/NJIT | 提升 |
| ---: | ---: | ---: | ---: | ---: |
| 63 | 4.46 µs | **2.33 µs** | **0.523×** | 约47.7% |
| 252 | 9.25 µs | **6.17 µs** | **0.667×** | 约33.3% |
| 504 | 16.58 µs | **11.21 µs** | **0.676×** | 约32.4% |
| 2520 | 97.83 µs | **68.67 µs** | **0.702×** | 约29.8% |

最新微型实测中，普通 `Scheduler.execute()` 同样全部领先：63/252/504/2520 observations 的 Native/NJIT ratio 约为 **0.916 / 0.861 / 0.784 / 0.747**。

性能门槛工具：`tools/check_phase2_performance.py`。默认必测包括 1×1×5 的63/252 observations，以及500/1000产品完整 workload。Prepared 必须 `<=0.90`；普通 Scheduler 微任务必须 `<=1.00`。任何必测 workload 超过对应 ceiling 都视为性能验收失败。

## 8. AOT / 内存优势

最终 500/1000/2000 产品 benchmark 中 BetterSaaTaa runtime graph compile 约 **11.0–11.3s**；对应 RSS high-water 约为 **182MB→985MB**、**224MB→1.12GB**、**304MB→1.15GB**。不同运行的 JIT cache/allocator 状态会影响绝对值，因此这里只记录数量级，不把 RSS high-water 当作精确 live allocation。

CalMetricsEngine graph/operator 代码为 AOT：

- runtime graph JIT compile = 0；
- thread graph 输入边界 `input_copy_bytes = 0`；
- 16指标图每个 native worker 仅复用 2 个 numeric slot + 1 个 mask slot，以及 operator workspace；
- 500产品10线程 benchmark 的 arena/workspace 合计约 418KB；
- process 路线共享约 10.94MB input/interval/output storage，已有 reusable input bundle 时本次边界复制只有 starts/ends 约96KB，最终 output copy 768KB。

这些 RSS high-water 数字跨 Python/Numba/C++ standalone 不等同于精确 live allocation；它们只用于说明 JIT runtime 与 AOT runtime 的数量级差异。精确输入/算法复制以 native audit 为准。

## 9. quantile / median 优化

原 Phase 1 使用索引排序并间接读取 source values，cache locality 较差。Phase 2 改为：

```text
readonly source
  ↓ one controlled algorithm copy
contiguous double scratch
  ↓ sort
quantile / median
```

1,000,000 元素 quantile 本机由约106ms降至约72ms。该复制计入 `algorithm_copy_bytes`；绑定层仍为 `input_copy_bytes=0`。

## 10. Graph-level Reduction Fusion

Phase 2 已加入透明的 graph-level reduction fusion：

- 同一 numeric series 的 `sum/product/mean/min/max/variance/std/RMS/MAD/total_return` 共用顺序 summary scan；
- `median + quantile` 共用一次连续排序；
- 只有同一 source 存在多个可融合消费者时才启用，单独一个 reduction 仍走 canonical kernel；
- canonical operator 单独调用的行为不变；
- audit 暴露 `fused_scalar_calls / summary_source_scans / order_stat_sorts`。

例如500产品 benchmark 的一个600-row native chunk 中，多统计根合计产生5400次 fused scalar result，但只需要 **600次 summary source scan + 600次 order sort**，不再按指标反复扫描/排序。这一层是 C++ 从“接近 NJIT”提升到稳定通过性能硬门槛的主要原因。

## 11. 已知边界 / 下一优化层

Phase 2 的 interval DAG 主要服务多产品 × 多区间 × 多标量指标，不在 graph 内物化随区间增长的 matrix/portfolio matrix 节点；这些算子仍可直接通过 canonical registry 调用。

当前尚未完成：

- 研究平台 production Typed DSL / semantic / causality 契约迁移；
- 更广 reduction/statistics SIMD；
- matrix DAG / BLAS backend dispatch；
- benchmark 证明有价值前不启用 process×thread hybrid。

后续性能目标是 **broader SIMD + matrix/BLAS + production Typed-DAG lowering**，同时必须持续满足 NJIT 性能硬门槛。

详细设计见 `phase2-execution-graph-design.md`。
