# C++ Typed DSL / Rolling Scope 详细设计（2026-09-20）

## 1. 目标

本次仅完成两个明确需求：

1. 将 BetterSaaTaa 指标执行所依赖的 Typed DSL / Typed IR 核心能力迁入 CalMetricsEngine C++。
2. 将 `rolling_window` / `rolling_apply` 从 BetterSaaTaa 的 NJIT compiler scope 迁入 C++ compiler / graph executor。

最终计算链路：

```text
Python expression + variable type descriptors
    ↓ one PyBind boundary
C++ restricted parser
    ↓
C++ Typed Logical IR
    ├── dtype
    ├── named axes
    ├── symbolic shape
    ├── semantic_dimension
    ├── price_basis
    └── record/window intermediate types
    ↓
C++ canonicalization / lowering
    ├── aliases → 118 canonical operator ids
    ├── composite lowering
    ├── rolling_window → compiler-owned rolling scope
    └── rolling_apply → compiler-owned interval sub-program
    ↓
Physical Program + CSE + liveness + arena
    ↓
C++ Graph Executor
    ├── scalar roots
    └── aligned time-series roots
```

Python 不实现 AST、类型推导、rolling 语义或数值 fallback。

## 2. 四问原则

1. **是否属于当前需求？**
   - Typed type inference、alias canonicalization、rolling scope、series root execution：属于。
   - BetterSaaTaa 的变量目录、数据权限/可得性、指标 CRUD、展示格式、研究版本治理：不属于。

2. **不改是否阻塞当前需求？**
   - 当前 compiler 只有 `scalar/series/mask` 粗类型且 root 强制 scalar，会直接阻塞 Typed DSL 与 rolling series 输出，因此必须改。
   - Planner/Scheduler 的输出字节数与 process transport 默认按 `rows × roots`，会阻塞 series root，因此需要最小适配。

3. **是否可以最小改造？**
   - 保留现有 118 canonical registry、现有 operator kernels、现有 Scheduler/ProcessPool。
   - Typed IR 作为 compiler metadata，不复制数值实现。
   - rolling scope 复用同一个 `graph::Program` executor，不新增 Python/JIT/第二套 operator runtime。

4. **是否改变原契约？**
   - 旧 `GraphCompiler({"x": "series"})`、scalar-root `result.values.shape == (rows, roots)` 保持不变。
   - 新 typed declaration 与 series-root 是增量能力。
   - 旧 program serialization 继续可 decode；新程序使用新版本编码。

## 3. Typed IR

### 3.1 ValueType

C++ 新增稳定类型：

- kind: `scalar | series | vector | matrix | window | record`
- dtype: `float64 | bool`
- axes: `time | asset | window`
- symbolic shape: 如 `T`、`N`、`W`、`T-1`
- `semantic_dimension`
- optional `price_basis`
- record fields

约束遵循 BetterSaaTaa：

- series 只能是 `time` 轴；
- vector 只能是 `asset` 轴；
- window 只能是 `(time, window)`；
- bool 必须是 mask semantic；
- elementwise 非标量输入必须 axes/shape 一致；
- window 是 compiler-only 中间态，不能作为公开 root；
- record 必须先通过 field extractor 变成普通数值。

### 3.2 Variable declaration

继续兼容：

```python
GraphCompiler({"nav": "series", "periods": "scalar"})
```

新增：

```python
GraphCompiler({
    "adjusted_nav": {
        "kind": "series",
        "dtype": "float64",
        "axes": ["time"],
        "shape": ["T"],
        "semantic_dimension": "adjusted_nav",
        "price_basis": "hfq",
    }
})
```

Python 只负责 dict → C++ struct 的边界转换；所有校验和推导在 C++。

### 3.3 Alias

Alias 不进入 canonical registry，不新增数值 kernel。

C++ compiler 在 AST/Logical IR 阶段 canonicalize，例如：

```text
sub → subtract
mul → multiply
abs → absolute
var → variance
cov → covariance
corr → correlation
masked_mean → mean_where
sequence_mean → mean
...
```

因此 BetterSaaTaa 后续只需要发送兼容表达式，不需要保存 Alias NJIT 实现。

## 4. Logical IR → Physical Program

Logical IR 与 Physical Program 分离。

Logical node 保存：

- variable / constant / canonical call / rolling_window / rolling_apply
- inferred ValueType
- parents
- canonical operator id
- source expression identity

普通节点 lowering 到当前 `graph::Node`。

`rolling_window` 不进入 canonical registry，也不物化 `T × W`：

```text
mean(rolling_window(x, 20))
        ↓ compiler lowering
rolling scope {
    body = mean(x)
    width = 20
}
```

历史 `rolling_mean/std/min/max` 保持兼容，可以继续使用原 canonical kernel。

## 5. rolling_apply

### 5.1 语义

遵循 BetterSaaTaa `rolling_scope.py`：

```text
rolling_apply(<完整区间标量子图>, width [, min_periods])
rolling_apply(<完整区间标量子图>, width, dates, annual [, min_periods])
```

要求：

- body 必须输出 numeric scalar；
- body 至少依赖一个 time numeric series；
- 不允许嵌套 rolling scope；
- window 默认要求 complete finite window；
- min_periods 只控制联合有限观察数，不压缩原始窗口；
- 每个窗口重新执行完整 body，scan/state 从窗口起点重置；
- explicit dates 必须有限且严格递增；
- window 1..5000；
- 保留 work / scratch budget，超限 fail closed。

### 5.2 Physical RollingScope

`graph::Program` 新增 rolling scope metadata：

- scalar body sub-program；
- captured outer series bindings；
- captured scalar parameter bindings；
- width / min_periods；
- optional date / annual context；
- context substitutions：
  - observation_count
  - window_elapsed_days
  - risk_free_return_window
- output semantic metadata。

Scope node只生成一个 aligned series，长度等于当前 interval。

### 5.3 执行

Executor 对每个 row：

1. 取 outer interval view；
2. 校验 width/min/date；
3. 建 joint finite prefix count；
4. 对每个 trailing window 构造**只读 slice view**；
5. 使用 scope 独立 Scratch 执行 body sub-program；
6. 写入当前 scope series arena；
7. 无效窗口写 NaN。

禁止：

- materialize T×W；
- Python callback；
- 每个窗口重新分配完整输入；
- 每个窗口重新编译。

## 6. Series root 输出

本次增加 BetterSaaTaa time-series indicator 所需 root contract：

- 一个 graph 的公开 roots 必须全部是 scalar，或全部是 aligned numeric series；
- 不允许 scalar/series 混合 roots；
- window / record / mask 不允许作为公开结果；
- series root 必须与当前 interval 长度一致。

输出：

### scalar graph

保持原样：

```text
values.shape = (rows, roots)
```

### series graph

```text
values.shape = (sum(end-start), roots)
offsets.shape = (rows + 1,)
```

其中：

```text
values[offsets[i]:offsets[i+1]]
```

就是第 i 个 interval 的所有 series roots。

这样避免 Python object/ragged list，也适合多产品 product-major 批量计算。

## 7. Scheduler / Process 最小适配

Planner：

- scalar output bytes = `rows × roots × 8`
- series output bytes = `interval_observations × roots × 8`

Thread chunk：

- scalar offset = `chunk.begin × roots`
- series offset = 前序 interval observation 数 × roots

Process shared output：

- 仍为一个连续 SharedMemory region；
- worker 根据全局 starts/ends 计算自己的 series output offset；
- 不新增 pickle / Python worker；
- final shared result仍可直接映射 NumPy，无额外返回复制。

DAG-branch fork/join 本轮只用于 scalar-root graph。Series-root graph 使用 row/product/interval parallelism，避免多个 branch 并发写同一个 series segment。

## 8. 序列化

Program 编码版本升级。

新版本只写 worker 执行必需的信息：

- root output kind；
- rolling scopes；
- rolling body sub-program。

Typed node/root/semantic metadata保留在 `CompiledGraph` 与 fingerprint 中，不重复塞进 worker
`Program`，避免扩大 IPC 和 worker 执行契约。decode 保留旧版本支持，旧 scalar graph 不失效。

## 9. Python API

`CompiledGraph.metadata()` 增加：

- typed_ir_version
- output_kind
- root_types
- variable_types
- rolling_scope_count

Alias canonicalization 直接发生在 C++ compiler 内，不新增第二份 Alias operator registry。

`GraphExecutionResult`：

- `values`：scalar 或 concatenated series matrix；
- `offsets`：scalar graph 为 None，series graph 为 int64 prefix offsets；
- `output_kind`。

不在 Python 增加任何类型推导或 rolling 计算。

## 10. 测试

### Typed DSL

- legacy string variable declarations；
- typed dict declarations；
- series vs asset vector axis mismatch；
- shape mismatch；
- semantic dimension / price basis mismatch；
- mask misuse；
- record/window root 拒绝；
- Alias 与 canonical graph fingerprint/结果一致。

### rolling_window

- mean/std/variance/min/max；
- `std/variance(rolling_window(...))` 沿用 reduction 默认 `ddof=1`；历史兼容 `rolling_std(...)` 仍默认 `ddof=0`；
- min_periods；
- NaN；
- 不物化 T×W；
- legacy rolling_* parity。

### rolling_apply

- `rolling_apply(mean(returns), W)`
- 组合 body：`mean(divide(difference(nav,1),lag(nav,1)))`
- state reset：drawdown / cumulative path 每个窗口重置；
- explicit dates/context；
- min_periods；
- nested rolling 拒绝；
- invalid width/date/budget fail closed。

### executor

- single row series output；
- multi-row offsets；
- thread lane；
- process/shared-memory lane；
- prepared execution；
- scalar regression 100% 保持。

### 验收

- C++ CTest；
- Python full pytest；
- `git diff --check`；
- CodeGraph sync；
- 重点审核 Python 无 AST/numeric fallback、118 registry 无重复实现。

## 11. 实施与验收结果

本设计已完成实现。

- 新增 `cpp/include/calmetrics_engine/typed_ir.hpp`、`cpp/typed_ir.cpp`：C++ ValueType、命名轴、symbolic shape、semantic dimension、price basis 与 operator type inference。
- BetterSaaTaa 当前 **37/37 historical aliases** 与 C++ canonicalization 表机械对账完全一致；未新增重复数值 kernel。
- `rolling_window` 为 compiler-only logical intermediate，不物化 `T×W`。
- `rolling_apply` 编译完整 scalar body sub-program；每个窗口只传 slice view，并重置 body state；支持 BetterSaaTaa 的日期/年度上下文、`observation_count`、`L` 水平序列前置观察语义。
- series root 返回连续 `values + offsets`，single/thread/process/shared-memory/prepared 路径一致；低层 `Program.execute` 同样做 series 输出边界检查。
- BetterSaaTaa 当前 **35 个 Typed 标量指标 + 9 个时序输出 = 44 条生产公式**全部由 C++ GraphCompiler 编译成功，输出 kind 0 差异。
- 专项 Typed/rolling 测试：**17 passed**。
- 全量 Python：**4051 passed**。
- Native CTest：**5/5 passed**。
- ASan/UBSan Native CTest：**5/5 passed**（LeakSanitizer 未作为验收依据）。
- `git diff --check`：通过。
- Release 性能门：500 产品 Native/NJIT **0.637**；1000 产品 **0.612**；63/252 obs Prepared **0.513/0.674**；普通 Scheduler **0.873/0.831**，四项均通过既有阈值。
