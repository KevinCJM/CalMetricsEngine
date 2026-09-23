# CalMetricsEngine

AOT calculation engine for quantitative finance and fund research.

CalMetricsEngine is being built as the reusable execution layer behind the fund
investment research platform:

```text
DSL / AST
    ↓
Typed DAG
    ↓
Operator Lowering
    ↓
Native Execution Plan
    ↓
PyBind11 / C++17
```

The current 0.3.0 code line is **C++ first**: restricted AST parsing, native Typed IR for
scalar/time-series indicator execution, shared-DAG compilation, CSE, alias-aware liveness,
rolling scopes, cost planning, CPU admission, thread/process pools, shared-memory ownership
and all 146 canonical operators execute in C++. Python retains public import names, NumPy
ownership adaptation and an asyncio await bridge. A complete synchronous request crosses
PyBind once, not once per worker. The eight existing finance APIs retain their numerical contracts.

The native Typed IR now owns dtype, named time axes, symbolic shape, semantic dimension,
price basis, alias canonicalization, `rolling_window` lowering, compiler-owned `rolling_apply`
and aligned time-series roots. Typed matrix/vector and exact int64 category/index inputs,
native block/filter/group scopes and bounded scalar root finding extend the same execution
chain. Research-platform causality/knowledge-time and business governance remain upstream.
No PyPI publication is implied by local builds.

## 使用文档

首次接入从 [用户使用手册](docs/user-guide.md) 开始；完整导航见 [docs/README.md](docs/README.md)。

| 内容 | 文档 |
| --- | --- |
| 标量、时序、向量、矩阵、输入输出、DAG 和可运行例子 | [用户使用手册](docs/user-guide.md) |
| 全部 146 个算子的签名、形状、数学逻辑及边界 | [数学算子参考](docs/operator-reference.md) |
| SIMD、线程、进程、协程、共享内存的实际选择条件 | [执行与性能指南](docs/execution-guide.md) |
| out、Workspace、借用视图及直接调用约束 | [算子接口与内存契约](docs/canonical-operators.md) |

金融数据选择与业务解释由调用方负责；当前 Typed IR 已有的语义校验仍按代码执行，
具体边界见使用手册。阶段设计与验收记录保留当时范围，不作为最新能力目录。

## Install

After publication:

```bash
python -m pip install --only-binary=:all: "calmetrics-engine==0.3.0"
```

From the current checkout:

```bash
python -m pip install .
```

Import:

```python
import calmetrics_engine as engine
```

## Why this engine

- No runtime JIT compilation or service-start warmup.
- PyBind11/C++ AOT backend.
- Exact-dtype, strided, zero-copy NumPy inputs.
- Direct operators support compatible C/F-order, sliced and readonly arrays; graph float64 series require C-contiguous inputs.
- GIL released during native numerical work.
- Portable baseline wheels instead of mandatory AVX.
- One numerical implementation per reusable operator.
- Whole synchronous requests cross Python/C++ once; C++ workers reuse liveness-planned arenas.
- Native worker processes use the packaged `calmetrics_worker`, without a Python interpreter.
- Shared process outputs return as native-owned NumPy views without a final result copy.
- Adaptive product/interval scheduling with one CPU budget across threads, processes and async jobs.
- Process execution can use parent-owned shared memory instead of pickling large inputs.

## Canonical operators

```python
import numpy as np
from calmetrics_engine import operators as op

values = np.array([0.01, -0.02, 0.03, 0.005], dtype=np.float64)
assert len(op.catalog()) == 146
volatility = op.std(values, ddof=1)

output = np.empty_like(values)
result, audit = op.add(values, 1.0, out=output, simd="auto", audit=True)
assert result is output
assert audit["input_copy_bytes"] == 0

workspace = op.Workspace()
q95 = op.quantile(values, 0.95, workspace=workspace)
```

`operators` provides elementwise, masks, reductions, rolling/scan, matrix,
regression, state projections and compatible composite names. All calculations
execute in C++; Python only exports the native registry. Ordinary `std` defaults
to `ddof=1`, while `rolling_std` defaults to `ddof=0`. Missing values are handled by
each operator's contract, never silently dropped to enable SIMD.

Named functions, `op.call(name, ...)` and cached `op.get(name)` handles use the
same implementation. `out=` must not overlap input arrays. `Workspace` is
exclusive to one running call. `lag`, `transpose` and matrix `diag` default to
read-only views that retain their input owner.

See [the API and memory contract](docs/canonical-operators.md),
[the detailed design](docs/canonical-operators-design.md), and
[verification and performance evidence](docs/canonical-operators-acceptance.md).

## Native graph execution

```python
import numpy as np
from calmetrics_engine import AdaptiveScheduler, GraphCompiler

graph = GraphCompiler({"nav": "series"}).compile([
    "mean(divide(difference(nav,1),lag(nav,1)))",
    "std(divide(difference(nav,1),lag(nav,1)),1)",
])

# Product-major storage: two products, five observations each.
nav = np.array([1.0, 1.1, 1.05, 1.2, 1.3,
                2.0, 2.1, 2.0, 2.2, 2.3], dtype=np.float64)
starts = np.array([0, 5], dtype=np.int64)
ends = np.array([5, 10], dtype=np.int64)
product_ids = np.array([0, 1], dtype=np.int64)

with AdaptiveScheduler(cpu_budget=4) as scheduler:
    plan = scheduler.plan(
        graph, {"nav": nav}, starts, ends, product_ids=product_ids
    )
    result = scheduler.execute(
        graph,
        {"nav": nav},
        starts,
        ends,
        plan=plan,
        product_ids=product_ids,
    )

print(plan.metadata())
print(result.values)
```

Typed declarations can be supplied directly to the same C++ compiler. A time-series root returns
one contiguous value matrix plus interval offsets instead of Python ragged objects:

```python
graph = GraphCompiler({
    "returns": {
        "kind": "series",
        "dtype": "float64",
        "axes": ["time"],
        "shape": ["T"],
        "semantic_dimension": "return_decimal",
    }
}).compile(["rolling_apply(mean(returns), 20)"])

result = scheduler.execute(graph, {"returns": returns}, starts, ends)
# result.values.shape == (sum(ends - starts), 1)
# result.offsets[i]:result.offsets[i + 1] selects interval i.
```

`rolling_window` is a compiler-only logical window and is never materialized as a `T×W` matrix.
`rolling_apply` compiles its scalar body into a native sub-program and reruns that body on trailing
read-only window views with state reset at each window start. Historical aliases are canonicalized
inside C++ before DAG construction; they are not separate numerical implementations.

The planner chooses single/thread/process execution from the **post-fusion physical DAG cost**,
product/interval shape, input bytes and resource budgets. Product/interval chunks are balanced by
estimated physical row work, and a one/few-row request may fork dependency-closed heavy DAG
branches when ordinary row parallelism cannot fill the CPU budget. Process jobs use SharedMemory when required;
`execute_async` / `execute_many_async` provide coroutine orchestration while numerical
loops remain in C++.

The native graph executor fuses compatible reductions across roots so one source is not rescanned
for every statistic. Ordinary `scheduler.execute()` reuses compatible bound inputs while returning
independent results. For repeated tasks with the same graph and input geometry, prepare once:

```python
with AdaptiveScheduler(cpu_budget=4) as scheduler:
    prepared = scheduler.prepare_execution(graph, {"returns": returns}, starts, ends)
    retained = prepared.run_snapshot()  # Independent values, statuses and audit; safe to keep.
    current = prepared.run_audit()      # Reuses values; consume before the next run.
    values, statuses, evidence = current.values, current.statuses, current.audit
    # Input contents may be updated between completed runs; keep the same owners and geometry.
    next_result = prepared.run_snapshot()
    # retained remains unchanged; current.values may have been overwritten.
```

Use `run_snapshot()` for stored/history results and `run_audit()` for immediate consumption with
statuses and execution evidence. `run()` is the minimal values-only API when its validity contract
is sufficient. Prepared output from `run()`/`run_audit()` is overwritten on the next run; a readonly
view is not an independent snapshot. Changing bound dtype, pointer, shape or interval geometry requires rebinding/replanning;
do not mutate buffers concurrently with execution.

Performance is tested against the real BetterSaaTaa compiled NJIT batch on the same data/formulas,
with alternating paired execution. Required Native/NJIT thresholds are `<= 0.90` for prepared/batch
execution and `<= 1.00` for ordinary scheduler micro workloads. These are measured workload gates,
not a guarantee for every formula or CPU, and require explicitly running the gate tool.

See [the C++-first design](docs/cpp-first-design.md) and
[current acceptance evidence](docs/cpp-first-acceptance.md).

## Zero-copy input contract

CalMetricsEngine does not silently normalize calculation inputs.

```python
import numpy as np
import calmetrics_engine as engine

values = np.arange(40, dtype=np.float64).reshape(10, 4)
view = values[:, ::2]                 # non-contiguous view, no copy

result = engine.cal_std_mean(view)    # C++ reads the original strides directly
assert np.shares_memory(view, values)
```

Input rules for the existing top-level `cal_*` finance APIs:

- Values: NumPy `float64` ndarray.
- Group ids: NumPy `int32` ndarray.
- Dates / indices: NumPy `int64` ndarray.
- Dates may also be exact `datetime64[ns]`; they are viewed as int64 without copying.
- Arrays must be aligned.
- Readonly arrays are supported.
- Wrong dtype, Python lists, or incompatible objects fail instead of being copied.

Outputs and required scratch/workspace memory may be allocated. The contract is
**zero input copies and zero unnecessary intermediate copies**, not “no allocation”.

Canonical operators separately accept exact native float64 arrays and bool/uint8
masks in their declared scalar/vector/matrix signatures. Their `days_between`
operator consumes calendar-day numbers, not raw nanosecond timestamps. See the
operator contract before reusing a finance API's date or shape conventions.

## High-performance data preparation

The generic native operator APIs can read strided NumPy views without copying. The Phase-2
graph/SIMD batch path intentionally has a stricter layout: each product's observations
must be stored contiguously with unit stride.

Recommended product-major representation:

```text
values  = [ product_0 ][ product_1 ][ product_2 ] ... [ product_n ]
dates   = [ product_0 ][ product_1 ][ product_2 ] ... [ product_n ]
offsets = [0, p0_end, p1_end, ..., total_observations]
```

C++ locates one product by pointer arithmetic:

```text
begin = offsets[product]
end   = offsets[product + 1]

product_values = values + begin
length         = end - begin
```

Different intervals of the same product should be represented as
`product_id + start_offset + end_offset`, not materialized as new NumPy arrays.

For multiple fields, prefer Structure of Arrays (SoA):

```text
returns[total_observations]
close[total_observations]
volume[total_observations]
dates[total_observations]
offsets[product_count + 1]
```

The calling application should perform dtype normalization, sorting/alignment,
and any unavoidable data compaction once at the ingestion boundary. After data
enters CalMetricsEngine, hot-path code must not silently call
`astype`, `copy`, `np.ascontiguousarray`, advanced indexing, or equivalent
operations that materialize another input array.

### NaN and SIMD

NaN does not disable SIMD by itself, but per-element missing-value branches and
masks can reduce SIMD throughput. Missing-data semantics must never be changed
only for speed.

The intended execution lanes are:

1. **Dense SIMD lane** — metadata proves the block contains no NaN.
2. **Masked SIMD lane** — vector masks preserve the operator's NaN contract.
3. **Valid-span lane** — use contiguous valid spans when the operator semantics allow it.
4. **Scalar fallback** — irregular sparse missingness where SIMD is not beneficial.
5. **Reject** — operators whose contract forbids NaN fail explicitly.

Do not delete observations or fill NaN unless the operator's documented financial
semantics explicitly require that behavior.

## Adaptive execution model

One C++ `AdaptiveScheduler` owns admission and native pools; business modules must not create
competing thread/process pools around it. Python `asyncio.to_thread` is only a blocking-request
await adapter, not the engine's numerical worker pool.

The planner considers:

```text
products × intervals × DAG cost × observations
+ scenario count
+ input/workspace bytes
+ CPU budget
+ memory budget
+ hard-stop requirements
```

Current hierarchy:

- small CPU graph: one native call + SIMD
- medium/large in-process graph: bounded persistent thread pool + SIMD
- very large input/work or isolation requirement: process pool; large shared inputs use SharedMemory
- hard-stop/fault-isolated graph: disposable isolated process + SharedMemory
- coroutine API: orchestration/waiting only; numerical loops remain native

Product blocks are preferred when `product_ids` provide enough independent products;
otherwise the scheduler partitions interval rows. Both modes use physical-work weighting. If there
are too few rows to use the CPU budget, the compiler can expose independent root components as
precompiled DAG branches; shared operation prefixes and fusion groups remain together. Metrics are
never blindly split into independent jobs.

Native processes reading a large dataset attach to shared mappings using descriptors/offsets;
they do not pickle NumPy arrays or start Python. A `SharedInputBundle` can reuse shared inputs
across calls. Workers write disjoint result rows into shared output; the returned NumPy view pins
the C++ owner and remains valid after engine close, with no final copy. Preparing shared storage
from a normal array still requires one explicitly counted boundary copy.

All numerical entry points share one **process-wide native C++ CPU-admission budget and
persistent ThreadPool**. Each AdaptiveScheduler/Engine may impose a smaller local CPU cap, but Graph
requests and legacy finance APIs still acquire from the same NativeScheduler, so independent callers
cannot silently oversubscribe each other. Legacy `n_threads` remains a compatibility request ceiling,
not permission to create private threads.

Phase 2 intentionally uses either `1 process × N threads` or `N processes × 1 native graph
thread`; hybrid process×thread execution is disabled until benchmarks justify it.

Metric-level parallelism is not the first choice because many indicators share upstream DAG
work. Product and interval blocks remain preferred; DAG branch fork/join is a coarse fallback only
for independent heavy root components after shared prefixes/fusion relationships have been
collapsed.

The detailed implementation rules for future changes are in [AGENTS.md](AGENTS.md).

## Current finance kernels

| Function | Result |
| --- | --- |
| `cal_std_mean` | Column sample standard deviation |
| `cal_std_mean_simd` | Column means and sample standard deviations |
| `cal_cpr` | Persistence ratio by peer group |
| `cal_max_dd` | Maximum drawdown, date and recovery period |
| `cal_longest_dd_recover` | Longest drawdown-recovery duration |
| `cal_all_largest_indicators` | Largest streak statistics |
| `cal_all_longest_indicators` | Longest streak statistics |
| `cal_rolling_gain_loss` | Rolling return distribution statistics |

These kernels retain their existing financial calculation semantics. Their historical
`n_threads` argument is now only a per-request upper bound routed through the same process-wide
native Scheduler used by the Graph runtime; the finance kernels no longer create private thread
groups. The rename and memory architecture change do not silently redefine formulas.

## Architecture

```text
Python import aliases / NumPy owner adaptation / async await bridge
          ↓ one native request boundary
cpp/compiler.cpp       restricted AST / shared DAG / CSE / alias liveness
cpp/planner.cpp        physical cost / weighted partition / DAG branch strategy
cpp/scheduler.cpp      process-wide CPU admission / lazy persistent ThreadPool
cpp/native_runtime.cpp Engine-local caps / Graph + process execution / draining
cpp/native_process.cpp standalone native workers / framed IPC / hard stop
cpp/shared_memory.cpp  native RAII shared regions
          ↓
cpp/graph.cpp          numerical graph + fusion + reusable arenas
cpp/operators/         canonical kernels + SIMD
cpp/finance/           existing financial calculation contracts
```

The Python package contains one native extension and a standalone worker executable:

```text
calmetrics_engine._native
calmetrics_engine/calmetrics_worker       # calmetrics_worker.exe on Windows
```

`graph.py`, `planner.py` and `shared.py` retain compatibility aliases only. `runtime.py`
retains the asyncio adapter and compatibility names. Execution-side Typed IR and rolling scope
semantics are native C++; business causality/knowledge-time governance and research workflow
contracts remain in the research platform and lower into this native boundary.

See [docs/architecture.md](docs/architecture.md).

## Platform matrix

Configured CI targets:

| OS | Architectures |
| --- | --- |
| Linux glibc | x86_64, aarch64 |
| Linux musl | x86_64, aarch64 |
| macOS 11+ | x86_64, arm64 |
| Windows | AMD64 |

CPython 3.10–3.14 and NumPy 1.26–2.x are covered by the configured matrix.

Published wheels must not globally use `-march=native`, mandatory `-mavx*`, or
host-only CPU assumptions. AVX2 flags apply only to a separate translation unit,
reached after CPU/OS capability checks. Baseline initialization stays portable.

Eighteen canonical operators have explicit SIMD paths: 17 elementwise/mask
operators plus register-blocked matrix multiplication. ARM64 uses NEON; x86_64
provides SSE2 and separately compiled AVX2. No AVX-512 implementation is claimed.
Strided inputs and order-sensitive reductions/recurrences retain safe C++ scalar
paths. `op.available_simd()` reports usable compiled paths on the current machine.

Configuration is not platform acceptance: see the verification record for actual
runs. SIMD improves the measured native baselines, but the current matrix kernel
is still slower than system BLAS; it is not a claim to outperform every NumPy or
Numba operation.

## Development

```bash
python -m pip install ".[dev]"
python -m build
python tools/check_dist.py
python -m twine check --strict dist/*
python -I -m pytest tests -q --import-mode=importlib
ruff check src tests tools
```

Native-only tests:

```bash
cmake -S . -B .build-native \
  -DCALMETRICS_ENGINE_BUILD_PYTHON=OFF \
  -DCALMETRICS_ENGINE_BUILD_TESTS=ON
cmake --build .build-native --config Release
ctest --test-dir .build-native -C Release --output-on-failure
```

Sanitizers on GCC/Clang:

```bash
cmake -S . -B .build-sanitized \
  -DCALMETRICS_ENGINE_BUILD_PYTHON=OFF \
  -DCALMETRICS_ENGINE_BUILD_TESTS=ON \
  -DCALMETRICS_ENGINE_SANITIZE=ON
```

## Design rules

1. Python is the interface adapter; C++ compiles, plans, schedules and calculates.
2. Reusable mathematics belongs in CalMetricsEngine, not duplicated across business centers.
3. Primitive DAG execution should cross Python/C++ once per plan, not once per node.
4. Direct operators support strided views. Graph float64 time-series inputs retain the contiguous product-major contract; newly supported typed matrices, vectors, integer and mask inputs preserve validated strides.
5. No runtime Numba/JIT dependency in CalMetricsEngine.
6. Business-specific orchestration remains in the research platform.
7. Coupled black-box kernels are allowed only for genuinely inseparable recursive,
   fitting or jointly constrained algorithms.

Build and release details: [docs/publishing.md](docs/publishing.md).

## Platform execution contracts

Graphs can opt into per-root numerical error isolation with `error_policy="isolate"`.
Use `result.statuses` alongside `result.values`; structural errors still raise.
Prepared calls reuse output, while `prepared.run_snapshot()` produces independent
read-only output for retained results. Native interval bindings preserve N NAV
versus N−1 returns and per-root parameters. See [execution contracts](docs/platform-execution-contracts.md).

## Mathematical composition

The original 118 operator IDs retain their behavior. IDs 119–125 add `normal_cdf`,
`aligned_shift`, `recursive_filter`, `argsort`, `gather`, `distinct_count` and `floor`.
`lag` still returns a shorter view; `aligned_shift` instead preserves length and fills
the prefix. `recursive_filter` exposes real-valued alpha, seeding, update masks and
gap emission policies; it does not change `recursive_smooth`.

```python
from calmetrics_engine import GraphCompiler

# Even-span EMA with first-valid initialization and held output across missing rows.
ema = GraphCompiler({"x": "series"}).compile([
    "recursive_filter(x,2/(12+1),0,finite_mask(x),2,0)"
])

# Disjoint complete blocks, then a visible reduction of their statistics.
blocks = GraphCompiler({"x": "series"}).compile([
    "mean(block_apply(std(x,0),20))"
])

# Exact int64 group identity, native group reduction and aligned broadcast.
groups = GraphCompiler({"x": "series", "key": {"kind": "series", "dtype": "int64"}}).compile([
    "group_apply(mean(x),key)"
])

# solve_x is local to the bounded native solver, never a Python callback.
root = GraphCompiler({"x": "series"}).compile([
    "bisect(solve_x*solve_x-mean(x),0,10,1e-12,100)"
])
```

`filter_apply(body,mask[,empty_default])` preserves selected order and returns a scalar;
an empty selection defaults to NaN. `block_apply` drops incomplete tails and produces
a compact vector, returned through the typed result protocol when selected as a root.
`group_apply` uses complete groups and is retrospective within each group. Sorting
indices likewise do not establish chronological or causal order.

Integer indices/categories stay int64 through the DAG and worker transport. `distinct_count`
compares IDs exactly and returns an exactly representable float64 count, consistent with
other count reductions. Missing/unknown IDs require an explicit validity mask; negative
codes are not silently discarded. Sorting, discrete selection and gather use necessary
native workspace/output allocations, not a blanket zero-allocation promise.

See the [detailed design and contracts](docs/mathematical-composition-design.md) and
[validation record](docs/mathematical-composition-acceptance.md). These generic capabilities
do not mean every MetricsFactory definition has been migrated or certified.

## Stateful and typed time series

IDs 126–146 add reusable adaptive and second-order recurrences, cosine, shared scalar
Kalman state, separate classification/confirmation, drawdown state, turning events,
joint PS filtering and complete-segment boundaries. KAMA and Super Smoother remain
visible compositions of these mathematical steps. Existing `recursive_filter` still
uses its original fixed-alpha and gap-hold contract.

```python
# One Kalman solve, two borrowed intermediate projections, independent public output.
kalman = GraphCompiler({"x": "series"}).compile([
    "state_estimate(scalar_kalman(x,0.01,0.1))",
    "state_variance(scalar_kalman(x,0.01,0.1))",
])

# Conditions, classification and consecutive confirmation remain separate nodes.
states = GraphCompiler({"x": "series"}).compile([
    "state_confirm(state_select(x>=0,0,1,finite_mask(x)),2,3)"
])
```

Homogeneous aligned series preserve **float64, bool and exact int64** in the compatible
`values`/`offsets` interface. Vectors, matrices, shortened sequences and mixed shapes/dtypes
use `result.outputs[root].values[interval]`, with readonly typed arrays and per-output statuses.
Use `compile(..., result_format="typed")` to select that interface explicitly. Records and
internal state bundles still require field projections. See the [multi-output examples](docs/user-guide.md).

Always read isolation statuses: invalid bool/int64 entries use False/0 placeholders.
Typed `prepared.run()` returns a Result with statuses; compatible bool/int64 bare-array runs
reject failed results. Prepared values are borrowed until the next run; `run_snapshot()`
returns independent results.

`segment_apply(body,between_events(events))` evaluates a scalar body on both complete
segment endpoints `[left,right]` and broadcasts to `[left,right)`. Open tails and missing
gaps stay missing. For example, `last(price)/first(price)-1` remains a visible body.
Peak/trough revisions, PS filtering and complete waves are retrospective; adding an
ordinary comparison does not make their dependent outputs causal.

In isolate mode, actual segment exceptions retain per-position failure statuses through
comparisons, masks and state selection; a resulting False/0 is still invalid. Unaffected
segments and independent roots remain available. Nonlocal operators without an exact
position-dependency mapping conservatively invalidate their dependent root; ordinary
missing/warmup values keep their existing numerical semantics.

See [stateful series design](docs/stateful-series-design.md),
[state/event contracts](docs/state-event-contracts.md) and
[validation evidence](docs/stateful-series-acceptance.md). These changes extend the engine;
they do not switch research-platform services or remove platform NJIT warmup.
