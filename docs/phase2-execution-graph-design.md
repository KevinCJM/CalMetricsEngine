# Phase 2: AST/DAG Execution Planner and Adaptive Parallel Runtime

> Historical Phase-2 design. Compiler/planner/scheduler and process/shared-memory ownership now execute in C++; current responsibilities and lifecycle rules are documented in `cpp-first-design.md` and `architecture.md`.

## 1. Goal

Phase 2 turns CalMetricsEngine from a collection of native operators into a reusable execution runtime.

The target chain is:

```text
formula(s)
  ↓
restricted AST
  ↓
shared typed-ish DAG
  ↓
CSE + liveness + cost model
  ↓
ExecutionPlan
  ↓
Adaptive Scheduler
  ├── coroutine orchestration
  ├── single-process native execution
  ├── persistent thread pool
  └── process pool + shared memory
           ↓
      C++ graph executor
           ↓
      canonical operators
           ↓
      SIMD where eligible
```

Phase 2 does **not** copy BetterSaaTaa business definitions or replace its production path yet. It creates the engine capability needed for a later controlled migration.

## 2. Scope discipline

Four questions:

1. Is AST/DAG planning, threading, processes, coroutines and shared memory required? Yes.
2. Does Phase 2 require modifying BetterSaaTaa? No. It is read-only reference/benchmark input.
3. Can this be implemented without introducing a second business DSL? Yes. The parser is a narrow mathematical expression compiler over canonical operator names and caller-provided variables.
4. Can semantics change for speed? No. Canonical operator behavior remains the source of numerical truth.

Out of scope:

- service/API integration in BetterSaaTaa;
- database/network fetching logic;
- automatic rewriting of persisted BetterSaaTaa definitions;
- GPU execution;
- distributed execution across hosts;
- arbitrary Python callbacks inside a graph;
- matrix-valued interval DAG nodes whose size grows quadratically with interval length.

## 3. Restricted AST

Accepted syntax:

- numeric constants;
- caller-declared variable names;
- canonical operator calls;
- arithmetic: `+ - * / **`;
- unary `-`;
- comparisons: `== != < <= > >=`.

Rejected:

- attribute access;
- subscripting;
- comprehensions;
- lambda;
- assignment;
- imports;
- arbitrary callables;
- keyword splats;
- unknown variables/operators.

Arithmetic syntax lowers to canonical operators. Example:

```text
mean(returns) / std(returns, 1)
```

becomes ordinary `divide(mean(...), std(...))` nodes.

The parser is only a safe compiler front end. Operator mathematics stay in C++.

## 4. Shared DAG

Multiple roots compile together.

CSE key:

```text
(kind, operator/opcode, ordered parents, constant value, variable binding)
```

This means:

```text
mean(returns)
std(returns, 1)
mean(returns) / std(returns, 1)
```

share one `returns` input and one `mean` node.

The graph stores:

- stable node id;
- node kind;
- opcode;
- parent ids;
- inferred value class: scalar / series / mask / fit / interval;
- estimated cost;
- last consumer;
- reusable arena slot for series/mask intermediates;
- root membership.

## 5. Liveness and memory plan

Series intermediates use reusable worker-local arenas.

For every node:

```text
birth = node index
death = last downstream consumer
```

A free-list allocator reuses slots whose previous values are dead.

Scalar and record results live inline and do not consume array slots.

For interval graphs supported in Phase 2:

- numeric series output length may not exceed interval length;
- mask output length may not exceed interval length;
- borrowed views such as `lag` do not allocate;
- scalar/record roots require no large output arena;
- unsupported matrix-growing nodes fail at compile time.

This avoids one allocation per node/window.

### Cross-root reduction fusion

Structural CSE alone is not enough when many scalar roots independently scan the same series.
The C++ graph executor therefore fuses compatible reductions per source node:

- `sum/product/mean/min/max/variance/std/root_mean_square/mean_absolute_deviation/total_return`
  share one ordered summary scan when at least two compatible consumers exist;
- `median` and `quantile` share one contiguous copy/sort;
- a single isolated reduction stays on the canonical operator path to avoid doing unnecessary work;
- `mean_absolute_deviation` reuses the cached mean and performs only its required second pass;
- the canonical 118-operator standalone behavior is unchanged.

Fusion is transparent to graph roots and is auditable through
`fused_scalar_calls`, `summary_source_scans` and `order_stat_sorts`.

## 6. Native graph execution

The C++ graph executor receives one immutable program and batch descriptors:

```text
inputs[]
starts[]
ends[]
output[row, root]
```

Each input is a product-major contiguous `float64` array. `start/end` are absolute half-open interval offsets.

For each row:

1. input nodes become zero-copy `Value` views;
2. constants stay inline;
3. operator nodes use the same canonical `prepare` contract; ordinary nodes call canonical `execute`, while eligible multi-root reductions use the graph fusion cache with equivalent numerical ordering;
4. array outputs use liveness-planned worker-local slots;
5. scalar roots are written to preallocated output.

There is no Python callback inside a row and no Python→C++ crossing per DAG node.

## 7. Adaptive cost model

Planner inputs:

- `R`: interval rows;
- each interval's exact observation length `T_i`;
- `M`: graph node/operator weighted cost;
- input bytes;
- output bytes;
- CPU budget;
- memory budget;
- hard-stop/fault-isolation requirement;
- whether inputs already reside in shared memory;
- whether there are asynchronous I/O producers.

Conceptual score:

```text
linear node work ≈ factor × Σ T_i
sort node work   ≈ factor × Σ (T_i × log2(T_i))
constant node    ≈ factor × interval_rows
```

The planner uses exact interval lengths rather than `row_count × max_window`, so a workload with many short windows is not overestimated merely because one long interval exists. Product-mode execution keeps one product intact and balances workers by the sum of that product's interval observation counts.

Cost classes:

- constant/projection: O(1)
- elementwise/scan/reduction/regression: O(T)
- rolling: O(T)
- median/quantile: O(T log T)
- matrix dot/pair covariance: O(T)
- matrix multiplication/solve: higher-order and not interval-DAG scheduled in Phase 2

Thresholds are configuration, not financial semantics. Defaults are conservative and testable.

## 8. Execution level selection

### Coroutine

Coroutine is **orchestration only**.

Use when:

- waiting for async input producers;
- concurrently coordinating independent graph jobs;
- waiting on process/thread futures.

Never run the numerical loop itself as Python coroutine work.

### Single native call

Use when work is too small for pool overhead.

```text
processes = 1
threads = 1
shared_memory = false
```

### Thread pool

Preferred medium CPU lane because native graph calls release the GIL.

Use when:

- data already lives in this process;
- row count gives independent chunks;
- work amortizes task dispatch;
- hard termination/fault isolation is not required.

The Scheduler owns one persistent bounded `ThreadPoolExecutor`.

### Process pool

Use when:

- hard-stop/fault isolation is requested; or
- workload/input size crosses benchmark-calibrated process isolation thresholds; or
- service isolation is more important than thread-only latency.

Phase-2 defaults are deliberately thread-first because current C++ graph chunks release the GIL and avoid IPC. The initial calibrated defaults are `thread_work_units=250,000`, `process_work_units=4,000,000,000`, and `process_input_threshold_bytes=256 MiB`; these are tunable execution-policy values, not numerical semantics.

Workers receive a serialized plan plus array descriptors, never Python callback functions.

### Shared memory

Shared memory is selected only with process execution when shared input size crosses the configured threshold, or when the caller already provides shared arrays.

Automatic path:

```text
NumPy contiguous input
  ↓ one explicit boundary copy
SharedMemory
  ↓ descriptors only
process workers
```

Repeated callers can construct a reusable shared input owner up front and avoid even that repeated boundary copy.

## 9. CPU budget and oversubscription

One Scheduler owns total CPU budget. A scheduler-wide atomic CPU-token lease is shared by synchronous calls, thread jobs, process jobs and coroutine-submitted jobs, so concurrent asynchronous requests cannot each reserve the full host.

Phase 2 policy:

- thread lane: 1 process × N threads;
- process lane: N processes × 1 native graph thread.

Hybrid process×thread execution is deliberately not enabled by default in Phase 2 because it easily oversubscribes the host. The planner metadata reserves `threads_per_process` for later benchmark-driven hybrid support.

This is safer than independently maximizing both levels.

## 10. SIMD planning

The graph planner annotates every canonical operator node:

- `simd_eligible`;
- expected unit-stride fast path;
- scalar/stateful path;
- expected minimum useful length.

The actual ISA decision remains inside C++ runtime dispatch.

Planner uses SIMD metadata for cost estimation, but never forces an unsupported ISA.

## 11. Process/shared-memory lifecycle

Rules:

- parent owns and unlinks automatically created SharedMemory;
- worker only attaches/closes;
- parent waits for all worker futures before unlink;
- no raw pointers cross processes;
- descriptors include name, dtype, shape;
- shared inputs are read-only;
- process workers write disjoint row ranges into one parent-owned shared output array; the parent materializes the final caller-owned NumPy result once;
- executor shutdown is explicit/context-managed.

## 12. Public API

Target:

```python
from calmetrics_engine import GraphCompiler, AdaptiveScheduler

graph = GraphCompiler(
    variables={"adjusted_nav": "series"}
).compile([
    "mean(adjusted_nav)",
    "std(adjusted_nav, 1)",
])

scheduler = AdaptiveScheduler(cpu_budget=8)

plan = scheduler.plan(
    graph,
    inputs={"adjusted_nav": values},
    starts=starts,
    ends=ends,
)

result = scheduler.execute(plan, ...)

# Repeated tiny/single-lane hot loops bind once, then execute with one PyBind call.
prepared = scheduler.prepare_execution(
    graph,
    inputs={"adjusted_nav": values},
    starts=starts,
    ends=ends,
    plan=plan,
)
values = prepared.run()

result = await scheduler.execute_async(plan, ...)
results = await scheduler.execute_many_async([...])
```

Planner output is inspectable:

```text
lane
process_count
thread_count
shared_memory
estimated_work_units
estimated_input_bytes
simd_nodes
array_slots
reason_codes
```

No opaque “AI decides” behavior.

## 13. Validation

Required:

- malicious AST rejection;
- 118 operator name boundary;
- multi-root CSE;
- liveness slot reuse;
- scalar/thread/process parity;
- shared-memory parity;
- coroutine orchestration parity;
- deterministic planner choices around thresholds;
- CPU budget never exceeded;
- no process payload contains full large arrays when shared mode is selected;
- owner cleanup after success/error;
- zero-copy native interval input;
- the full current regression suite stays green across supported CPython/NumPy targets;
- ASan/UBSan native graph tests;
- multi-product/multi-interval/multi-metric benchmark against BetterSaaTaa NJIT.

## 14. Acceptance criterion

Phase 2 is complete when:

1. AST→DAG→ExecutionPlan exists as a public engine capability.
2. One native call executes a whole graph chunk.
3. Planner automatically selects single/thread/process and shared-memory policy.
4. Coroutine APIs orchestrate independent jobs without moving numerical loops into Python.
5. SIMD eligibility is part of plan metadata and native dispatch.
6. Results are identical across execution lanes.
7. Resource ownership and CPU budgets are tested.
8. Required paired workload benchmarks satisfy `Prepared Native/NJIT <= 0.90`; required micro workloads additionally require ordinary `Scheduler.execute/NJIT <= 1.00`. This covers both large shared-DAG batches and 1-product × 1-interval × 5-metric workloads. A slower warm C++ public scheduler path is a failed performance acceptance even if AOT startup or memory is better.
