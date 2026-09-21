# CalMetricsEngine Repository Guidelines

## 1. Project Mission

CalMetricsEngine is the reusable **C++-first, AOT, high-performance calculation engine** for the
fund investment research platform.

Its core architectural objective is:

> **Python is only the public/interface layer. Everything between API ingress and result egress —
> parsing, IR/DAG construction, lowering, optimization, planning, scheduling, concurrency, shared
> memory, numerical operators and execution — must be implemented in pure C++.**

The intended end-to-end chain is:

```text
Python API
    ↓
PyBind11 boundary                     ← Python responsibility stops here
    ↓
C++ Restricted AST / future Typed IR
    ↓
Logical DAG
    ↓
Operator Lowering
    ↓
Structural CSE
    ↓
Borrow-aware Liveness / Arena Planning
    ↓
Physical Execution DAG
    ├── reduction/order fusion
    ├── reusable fit/state nodes
    └── independent heavy root branches
    ↓
C++ Adaptive Planner
    ├── physical cost model
    ├── product/interval weighted partition
    ├── DAG-branch planning
    ├── CPU / memory admission
    └── single / thread / process / shared-memory strategy
    ↓
C++ Native Scheduler
    ├── persistent ThreadPool
    ├── persistent / isolated ProcessPool
    ├── SharedMemory / mmap
    └── Hard Stop / timeout lifecycle
    ↓
C++ Graph Executor
    ↓
Canonical C++ Operators
    ↓
SIMD / BLAS / native kernels
    ↓
PyBind11 result boundary
    ↓
Python-facing NumPy/result objects
```

Python may provide only thin interface adaptation:

- accept Python/NumPy objects;
- pin/retain owners while native work is active;
- translate Python-friendly parameters into native bindings;
- translate native exceptions/results back to Python;
- expose lightweight asyncio/await bridging;
- expose compatibility aliases and ergonomic public APIs.

Python **must not** implement or duplicate:

- AST parsing or expression semantics;
- DAG construction, lowering, CSE or liveness;
- physical execution planning or fusion decisions;
- cost models, partitioning or worker-count decisions;
- thread/process scheduling or CPU admission;
- SharedMemory creation policy, worker transport or lifetime rules;
- numerical/financial operators;
- rolling/reduction/regression/state algorithms;
- SIMD/BLAS dispatch;
- process worker execution;
- native output/workspace planning.

If a new feature requires logic in any of those categories, implement it in C++ and expose only the
minimal PyBind/Python interface needed to call it.

Do not reintroduce runtime Numba/JIT compilation, Python numerical fallbacks, Python worker pools,
or Python-side duplicate planner/operator implementations into CalMetricsEngine.

The research platform may retain separate **business semantics** such as production financial DSL,
semantic axes, causality/knowledge-time and research contracts. Those semantics should lower into a
stable native IR/DAG boundary rather than moving CalMetricsEngine runtime logic back into Python.

## 2. Current vs Target Architecture

### Current

The current repository already provides:

- `calmetrics_engine` as a thin Python-facing package backed by one PyBind11 native extension,
  `calmetrics_engine._native`.
- C++17 AOT finance/numerical kernels; runtime JIT is not part of the architecture.
- Exact-dtype zero-copy NumPy input binding, strided `ArrayView` support and explicit owner pinning.
- A 125-entry canonical **C++** operator registry with stable native opcodes; original IDs 1–118 remain unchanged.
- Exact-shape output/workspace contracts and reusable native `Workspace`.
- Explicit NEON/SSE2/optional AVX2 runtime dispatch for eligible kernels.
- A pure-C++ restricted mathematical parser/compiler.
- Native C++ Typed IR for scalar/time-series indicator execution, including dtype, named time axis,
  symbolic shape, semantic dimension, optional price basis and record/window intermediates.
- C++ historical-alias canonicalization into the single native operator registry.
- Compiler-owned `rolling_window` lowering with no materialized T×W production array.
- Compiler-owned `rolling_apply` sub-programs with trailing window views and state reset at each window start.
- Native `block_apply`, `filter_apply`, `group_apply` and bounded `bisect` sub-programs, using the same compiler and executor rather than Python numerical callbacks.
- Exact int64 category/index intermediates and typed matrix/vector inputs, including native worker transport and geometry validation.
- Generic real-alpha masked recurrence, aligned shift, Gaussian CDF, stable index sorting, gather, integer distinct counts and floor primitives. Financial indicators remain explicit compositions.
- Aligned time-series root outputs as contiguous values plus interval prefix offsets.
- A shared multi-root logical DAG with operator lowering and structural CSE.
- Borrow-aware C++ liveness analysis, including backing buffers of borrowed views such as `lag`.
- Reusable numeric/mask arenas with slot reuse driven by DAG lifetime.
- Compiler-generated **PhysicalExecutionMetadata**:
  - fusion-aware physical cost;
  - dependency-closed independent root branches;
  - branch-local compact numeric/mask arenas.
- A C++ Planner using physical work rather than blind logical-node count.
- Physical-work-weighted product and interval partitioning.
- Coarse DAG heavy-branch fork/join when row/product/interval parallelism cannot fill the CPU budget.
- Persistent bounded C++ ThreadPool and ProcessPool.
- One native CPU-token admission authority shared by synchronous and async-interface requests.
- Parent-owned SharedMemory/mmap transport; workers attach the same large input region read-only.
- Pre-shared input reuse through native descriptors.
- Native-owned shared output with no final result copy when the shared process path is selected.
- A packaged standalone `calmetrics_worker` that never imports or links Python.
- Isolated disposable worker execution for Hard Stop / hard process termination.
- Entire synchronous requests crossing Python→C++ once, not once per DAG node or worker chunk.
- `graph.py`, `planner.py` and `shared.py` as compatibility/interface aliases only;
  `runtime.py` contains only thin asyncio/interface adaptation.
- Performance and memory acceptance against the real BetterSaaTaa fused NJIT baseline.

### Target

The following remain architectural targets and must not be claimed as already implemented until
code and tests exist:

- Broader production business DSL governance that is intentionally above the generic engine boundary,
  especially causality/knowledge-time/research-workflow contracts.
- Broader public matrix-output contracts and portfolio business semantics beyond the current scalar/aligned-series root boundary.
- Additional masked/path/matrix fusion beyond the current summary/order-stat graph fusion.
- Broader SIMD coverage and benchmark-justified ISA extensions beyond current kernels.
- BLAS/backend dispatch where matrix workloads justify it.
- Additional hardware-specific tuning justified by full-workload benchmarks.
- Hybrid process × thread execution only if reproducible benchmarks beat the current one-level
  CPU-budget policy without oversubscription.

### Core Architecture Principles

These principles are project-level constraints, not optional coding preferences.

#### 1. C++ is the single execution source of truth

There must be one implementation of every runtime calculation concept.

```text
Python API
    ↓
C++ implementation
```

Never maintain:

```text
Python planner + C++ planner
Python operator + C++ operator
Python DAG executor + C++ DAG executor
```

for the same current behavior.

Compatibility Python modules may forward calls, but must not duplicate execution semantics.

#### 2. One whole-request language crossing

Prefer:

```text
Python
    ↓ one PyBind call
C++ compile/plan/schedule/execute
    ↓
Python result
```

Reject architectures such as:

```text
Python node → C++ node → Python node → C++ node
Python worker → PyBind → C++ worker repeated per chunk
```

The hot path must never depend on repeated Python callbacks.

#### 3. Separate Logical DAG from Physical Execution DAG

The logical graph expresses calculation semantics.

The physical graph expresses how those semantics are executed efficiently after:

- lowering;
- CSE;
- liveness analysis;
- reduction/order fusion;
- reusable fit/state extraction;
- branch construction;
- arena planning.

Planner decisions must use the **physical execution cost**, while preserving logical cost metadata
for auditability.

Do not change mathematical semantics merely to make the physical graph cheaper.

#### 4. Preserve shared work before introducing parallel work

Parallelism comes **after** CSE/fusion.

Never split metrics into independent jobs if doing so duplicates shared upstream work.

Preferred order:

```text
share computation
    ↓
fuse compatible work
    ↓
identify truly independent units
    ↓
parallelize those units
```

#### 5. Use hierarchical parallelism, not maximum parallelism

Preferred hierarchy:

1. single native thread + SIMD for small work;
2. product-level parallelism;
3. interval-level parallelism;
4. coarse independent DAG-branch fork/join when row-level work cannot use the CPU budget;
5. process isolation/shared-memory execution when justified;
6. hybrid process × thread only with benchmark evidence.

The goal is lowest wall time under CPU/memory constraints, not the highest thread/process count.

#### 6. One Scheduler owns all execution resources

Only the native C++ Scheduler may decide:

- single/thread/process lane;
- worker count;
- partition dimension;
- branch task count;
- CPU-token admission;
- memory-budget reduction;
- SharedMemory policy;
- Hard Stop isolation.

Business code, Python wrappers and individual operators must not create competing pools or hidden
parallel regions.

There is one **process-wide native C++ Scheduler resource layer** for CPU admission and the persistent
ThreadPool. Graph Engines may impose smaller local CPU caps, but every in-process numerical entry point
—including compatibility/legacy finance APIs—must also acquire from the same process-wide budget.
A compatibility parameter such as `n_threads` may only be a per-request upper bound; it must never
create private threads or bypass global admission.

#### 7. Memory movement is part of the algorithm

Avoid unnecessary copying by design.

Preferred order:

```text
existing contiguous native/shared input
    ↓
views / offsets / descriptors
    ↓
arena-reused intermediates
    ↓
native output
```

For process execution, one large input block should be shared by descriptor across workers instead
of copied once per worker. Small inline IPC remains allowed when measured to be cheaper.

Zero-copy claims must distinguish:

- Python→native input binding;
- one-time copy into shared storage;
- algorithm-required scratch copy;
- inter-process copies;
- final result copy.

#### 8. Ownership and lifetime must be explicit

Every pointer/view must have a clear owner.

This includes:

- NumPy input owners;
- borrowed DAG views;
- branch-local arenas;
- SharedMemory regions;
- output mappings;
- worker task captures.

No task may outlive memory it can access. Error/timeout paths must drain or terminate-and-wait
before releasing owners, mappings or CPU leases.

#### 9. Numerical and financial contracts outrank performance

Optimization must preserve:

- floating-point operation ordering where contractually relevant;
- NaN/Inf semantics;
- ddof and quantile conventions;
- temporal/interval semantics;
- causality/business semantics supplied by upstream research layers;
- stable operator IDs and public contracts.

Never use fast-math, NaN removal, approximation, reordering or data filling merely because it is
faster unless the public numerical contract explicitly allows it.

#### 10. Fail closed at boundaries

Validate before entering unchecked/native hot paths:

- dtype;
- shape;
- strides;
- offsets;
- interval bounds;
- plan identity/geometry;
- parameter count/type;
- SharedMemory descriptor bounds;
- process output geometry.

Stale plans, mutated prepared arrays or invalid descriptors must raise rather than silently run.

#### 11. Performance decisions require evidence

Thresholds and optimized paths must come from reproducible full-workload measurements.

Microbenchmarks may identify kernel bottlenecks but cannot justify whole-engine architecture alone.

Every major optimization should answer:

- Is wall time lower?
- Is memory peak acceptable?
- Are input/output copies reduced or at least explicit?
- Does it still beat the production NJIT baseline where required?
- Does it remain correct under sanitizer, native and cross-version tests?

#### 12. Keep business semantics above the engine boundary

CalMetricsEngine should understand stable mathematical/native execution semantics.

The investment-research platform may own richer concepts such as:

- financial/business DSL;
- semantic axes;
- price basis;
- causality;
- knowledge time;
- research workflow contracts.

Those should lower into CalMetricsEngine's stable native IR instead of embedding research-platform
business logic into the generic engine.

## 3. Scope Discipline

Before changing code, answer these four questions:

1. Is the change required by the current request?
2. If it is not changed, does it block the current request?
3. Can the requirement be solved with a smaller change?
4. Does the change alter numerical, financial, temporal, memory, or concurrency semantics?

Do not refactor unrelated code merely for elegance, technical-debt cleanup, or speculative future
needs. Do not fix unrelated bugs unless the current task requires it.

### Authorization boundary

- Do not remove, replace, disable or change existing capabilities, workflows, models, algorithms or
  behavior contracts without an explicit user requirement or authorization covering that change.
- Refactoring, optimization, deduplication and compliance with these guidelines do not independently
  authorize a behavior change. Explain effects outside the authorized scope and obtain permission
  before making them; do not ask again for work already authorized in the conversation.
- The cleanup and operator-design rules below apply within the authorized scope. Updating these
  guidelines does not itself authorize a migration, service switch or removal of existing APIs.

### Source history and replacement discipline

- Use Git commits, tags and branches to preserve source history. Do not retain historical source
  copies, commented-out implementations or unreachable version-selection branches merely for
  comparison or rollback.
- Within an authorized replacement, remove the superseded implementation together with obsolete
  imports, exports, registrations, configuration, tests, documentation and dependencies. Preserve
  still-relevant behavioral tests by directing them at the current implementation.
- Roll back through Git, not through dormant old functions, files or runtime switches embedded in
  the current source tree.
- Retain compatibility entry points only for an explicit, still-supported public contract with
  tests; they must delegate to the single current native implementation. Remove them when that
  compatibility contract is explicitly retired within the authorized scope.
- External API/data-format versions, immutable definitions, snapshots and model artifacts may be
  retained for compatibility or audit. They do not justify parallel historical numerical runtimes.
- Portable scalar/SIMD variants implementing the same current contract are active execution paths,
  not historical copies. Controlled test references and benchmark baselines are permitted evidence,
  but must never become production fallbacks or a dependency of standalone engine execution.

## 4. Canonical Data Layout

High-performance batch execution should use **product-major contiguous storage**.

For one numeric field:

```text
values = [ product_0 observations ][ product_1 observations ] ... [ product_n observations ]
offsets = [0, p0_end, p1_end, ..., total_observations]
```

For product `p`:

```text
begin = offsets[p]
end   = offsets[p + 1]
value(t) = values[begin + t]
```

Dates should use the same observation layout when products do not share one calendar:

```text
dates[begin:end]
```

For multiple fields, prefer **Structure of Arrays (SoA)**:

```text
close_values[total_observations]
return_values[total_observations]
volume_values[total_observations]
dates[total_observations]
offsets[product_count + 1]
```

Do not default to row objects, Python object arrays, per-product Python lists, or AoS layouts for
the numerical hot path.

### Intervals

Different date ranges for the same product should be represented by metadata, not copied arrays:

```text
product_id[k]
start_offset[k]
end_offset[k]
```

C++ computes the effective pointer/range from the product base offset and interval offsets.

Do not materialize a new NumPy array for every product, interval, window, group, or DAG node.

### Stable numeric and categorical boundaries

- Keep `object` arrays and mixed Python containers outside native numerical execution. Numeric
  values stored as objects must be converted according to their numeric meaning at ingestion;
  do not reinterpret them as category codes or silently turn invalid/missing values into zero.
- For finite categories, enums or identifiers, the ingestion/schema owner must define a stable,
  reversible mapping into an explicitly supported fixed-width integer dtype. Preserve the mapping
  and its version with persistent results and use the same meaning across nodes, batches and workers.
  Never independently re-encode each batch or process.
- Missing and unknown categories need distinct, documented reserved codes or validity masks that
  cannot collide with valid values. Integer category codes do not imply order or numeric distance.
- Respect each public API's existing dtype contract. Unsupported categorical inputs must fail closed;
  this policy does not add integer/category bindings to a float64-only execution entry point or
  permit disguising unsupported types as floating-point values.
- For affected category interfaces, test consistent mappings, decoding, missing/unknown values,
  integer-range boundaries and cross-worker transport. Pure presentation and I/O metadata are not
  numerical inputs and need not be encoded merely to pass through an interface.

## 5. Zero-Copy Contract

Once data enters the calculation layer:

- Inputs are NumPy arrays with explicit stable dtype.
- C++ receives pointer + shape + strides or pointer + offsets.
- Read-only input is the default.
- Basic slices and views must remain views.
- Advanced indexing, boolean indexing, `take`, `astype`, `copy`,
  `np.ascontiguousarray`, or `np.require` must not be inserted into the hot path silently.
- If conversion is unavoidable, perform it once at the ingestion boundary and reuse the result.
- Output arrays and necessary workspace buffers may be allocated.
- Workspace buffers should be planned and reused according to DAG liveness.

Zero-copy means **zero unnecessary input/intermediate copies**, not zero allocations.

## 6. SIMD Policy

SIMD is a first-class optimization target.

### Fast path layout

The highest-throughput SIMD path expects:

- Exact native dtype.
- Aligned memory.
- Unit-stride contiguous data inside each product/block.
- Sufficient block length to amortize dispatch overhead.

The engine may retain a generic strided zero-copy path, but the Scheduler should prefer prepared
unit-stride blocks for heavy batch execution.

### ISA policy

Published wheels must remain portable.

Never globally require host-specific ISA flags such as:

- `-march=native`
- unconditional `-mavx2`
- unconditional `-mavx512*`
- equivalent MSVC host-only assumptions

The intended design is runtime dispatch:

```text
x86_64: baseline → AVX2 → AVX-512 when available and benchmarked
arm64:  NEON baseline, optional newer extensions when safely detected
fallback: scalar/reference path
```

An optimized ISA path must always have an equivalent baseline path.

### What to vectorize

Good SIMD candidates:

- add/subtract/multiply/divide
- comparison/masks
- reductions
- mean/std/variance
- covariance/correlation
- matrix/vector operations
- independent rolling-window arithmetic
- batch evaluation across independent products

Stateful recurrence along time, such as drawdown or cumulative path algorithms, may not vectorize
well across observations. For these, consider vectorizing **across independent products** or use
thread-level parallelism instead of forcing unsafe SIMD.

## 7. NaN and Missing-Data Policy

NaN does not inherently disable SIMD, but per-element NaN handling can add masks, branches, and
reduce throughput.

Never remove/fill NaN merely to improve SIMD if doing so changes financial semantics.

Use one of these explicit strategies:

1. **Dense fast lane**: metadata proves a block contains no NaN; use branch-free SIMD.
2. **Masked SIMD lane**: preserve NaN semantics with vector masks.
3. **Valid-span execution**: when the operator contract permits it, precompute contiguous valid
   spans and execute dense kernels on those spans.
4. **Scalar fallback**: for irregular sparse missingness where masked SIMD is not beneficial.
5. **Fail closed**: operators whose contract forbids NaN should reject the input.

The operator contract must state its missing-value policy. Do not infer it from implementation
convenience.

## 8. Pointer Arithmetic and Bounds

C++ may use pointer arithmetic for performance only after validating:

- dtype
- alignment
- product offsets
- interval offsets
- shape
- strides
- owner lifetime

Preferred product/interval addressing:

```text
product_base = data + offsets[product]
interval_ptr = product_base + start
length       = end - start
```

Do not construct unsafe views or dereference unchecked offsets.

All pointer arithmetic must be covered by empty-input, first/last element, negative/invalid offset,
and boundary tests.

## 9. Unified Scheduler

There must be one scheduling authority for heavy calculation.

Business modules must not independently create their own process pools, thread pools, or nested
parallel regions.

The Scheduler should reason from at least:

- product count `P`
- interval/window count `W`
- DAG/operator cost `M`
- observations `T`
- scenario count when applicable
- input bytes
- estimated workspace bytes
- available physical/logical CPU budget
- available memory
- operator/kernel concurrency capability
- cancellation / hard-stop requirement

Conceptual work size is based on the **post-lowering/post-fusion physical plan**, not a blind
sum of every logical DAG node:

```text
Work ≈ constant_per_row × rows
     + linear_per_observation × Σ T_i
     + sort_nlogn × Σ (T_i × log2(T_i))
```

Use each interval's actual observation count, not `row_count × max_window`. Product and interval
chunks must be balanced by estimated physical row work. Preserve both logical and physical work
estimates in metadata so Planner decisions remain auditable. Thresholds must come from reproducible
benchmarks.

## 10. Parallelism Hierarchy

### Single-thread + SIMD

Use for small workloads where scheduler/parallel overhead dominates.

### Thread pool

The native Scheduler owns a persistent bounded C++ thread pool. Tasks contain only native values and owners, never `py::object` or Python callbacks. Binding-layer input pins stay alive until all tasks settle.

Preferred for CPU-bound work inside one process when:

- data already lives in the process address space
- tasks are independent
- GIL is released
- failure isolation/hard-stop is not required

Priority dimensions are generally:

1. product blocks
2. interval blocks when one/few products have many independent intervals
3. dependency-closed heavy DAG branches when row-level parallelism cannot fill the CPU budget
4. scenario blocks
5. independent heavy native kernels

Do **not** default to metric-level parallelism because DAG outputs often share upstream work. DAG
branch fork/join is allowed only after shared operation ancestors and fusion relationships have been
collapsed into one branch, and only when each branch is coarse enough to amortize task overhead.

### Process pool

Use for coarse heavy jobs when at least one is true:

- many independent products/chunks provide enough work per process
- hard termination / fault isolation is required
- one native process would otherwise monopolize the service
- memory ownership or long-running job isolation requires process boundaries

Process tasks must be coarse enough to amortize startup, IPC, and scheduling overhead.

### Shared memory with process pool

When multiple processes read the same large input dataset, use shared memory or memory mapping
instead of serializing/copying the dataset per process.

Pass descriptors:

```text
shared_memory_id / mmap descriptor
dtype
shape
strides or offsets
product range
interval metadata
```

Each worker attaches and creates read-only views. Large arrays must not be sent through
`ProcessPoolExecutor` payload pickling.

If a process owns unique small data, shared memory is not automatically required.

## 11. Avoid Oversubscription

Never independently maximize process count and native thread count.

The process owns one global native C++ CPU-admission budget and one persistent native ThreadPool.
Every AdaptiveScheduler/Engine adds its own local CPU cap, but all Graph requests and legacy finance
APIs also acquire from the same process-wide NativeScheduler before running numerical work. This
prevents different Engine instances, synchronous/async callers and compatibility APIs from silently
oversubscribing each other.

Bad:

```text
8 processes × 16 native threads = 128 runnable threads on a 16-core machine
```

Phase-2 default examples:

```text
1 process × 16 threads
16 processes × 1 native graph thread
```

Hybrid process × thread layouts such as `4 × 4` remain disabled by default until benchmarks
demonstrate a real advantage without oversubscription. Actual choices may reserve cores for the
API/service process.

Native kernels must accept a scheduler-provided thread budget and must not silently spawn an
unbounded number of threads. Legacy `n_threads` parameters are compatibility request ceilings only;
they must route through the process-wide NativeScheduler and must not create private `std::thread`
groups or increase concurrency beyond detected machine capacity.

## 12. Shared-Memory Lifecycle

Shared memory is an execution transport, not a business-data store.

Rules:

- Prefer one ingestion/copy into shared storage over one copy per worker.
- When possible, load data directly into mmap/shared storage and remove even the initial copy.
- Inputs shared across workers are physically read-only.
- After initial ingestion/copy, the creator/parent input mapping must be sealed read-only before it is
  exposed for reuse; a reversible NumPy `writeable=False` flag alone is insufficient.
- A readonly input descriptor is immutable policy: asking to attach it writable must still produce a
  physically read-only mapping.
- The owner keeps shared memory alive until all workers finish.
- Cleanup/unlink is explicit and exception-safe.
- Never pass raw process-local pointers across processes.
- Workers reconstruct views from descriptors.
- Output ownership must be explicit: local result, shared output, or reduction result.
- A returned NumPy shared-output view must retain its native region owner after scheduler/owner close. Distinct normal calls return independent storage; Prepared.run intentionally reuses output.
- A physically read-only mapping must export a read-only buffer, not only set a reversible ndarray flag.
- Child failures/timeouts must drain or terminate-and-wait before parent mappings or CPU leases are released.

## 13. Thread-Pool Runtime

The C++-first Engine dispatches graph chunks entirely in C++. Each whole synchronous request crosses Python/C++ once; numeric workers do not cross back into Python. The caller's GIL is released during native planning/execution where no Python state is accessed.

Requirements:

- bounded workers and scheduler-wide CPU tokens;
- no nested unbounded pools;
- exception propagation;
- deterministic completion;
- clean shutdown;
- physical-work-weighted product/interval partitioning;
- coarse dependency-closed DAG-branch fork/join when justified;
- thread-local native arena/workspace reuse;
- memory-budget-aware concurrency reduction.

Do not create a pool per operator, per DAG node, or per business module. Do not reintroduce Python
ThreadPoolExecutor/ProcessPoolExecutor for numerical dispatch. A thin asyncio bridge may await a
blocking native request; **all** worker selection, partitioning, branch construction, admission and
execution remain native C++.

## 14. DAG Execution and Memory Planning

The current C++-first path is one Python→C++ transition per whole synchronous request:

```text
Restricted AST / future Typed IR
    ↓
Logical shared DAG
    ↓
Operator Lowering
    ↓
Structural CSE
    ↓
Borrow-aware Liveness / Arena Planning
    ↓
Physical Execution Metadata
    ├── fusion-aware cost
    ├── weighted row/product work
    └── dependency-closed heavy branches
    ↓
C++ Adaptive Planner
    ↓
C++ Native Scheduler
    ├── single
    ├── product/interval thread chunks
    ├── DAG-branch fork/join
    └── process/shared-memory isolation
    ↓
C++ Graph / Branch Executor
    ↓
Canonical C++ Operators
    ↓
SIMD / native kernels
```

Avoid:

```text
Python node → PyBind → C++ → Python node → PyBind → C++
```

Compilation/planning should identify at least:

- operator opcode;
- input/output slots;
- immutable parameters;
- dtype/shape contracts;
- product/date axes;
- workspace bytes;
- liveness interval;
- physical fusion groups;
- logical vs physical work estimates;
- dependency-closed root branches;
- branch-local compact arena slots;
- SIMD eligibility;
- product/interval/branch/process eligibility;
- CPU and memory-budget requirements;
- missing-value policy.

Buffers whose values are dead must be eligible for workspace reuse. Branch sub-programs must not
inherit full-graph arena capacity when their dependency closure can use fewer slots.

## 15. Operator Design

### Choose the smallest independent calculation semantics

Divide operators by **smallest independent calculation semantics**, not by function count, code
length, formula length, output count or execution cost. Each operator must have a one-sentence
responsibility, explicit inputs/outputs and independent tests. This does not require exposing every
arithmetic instruction or internal loop as a logical node.

Distinguish these granularity classes from domain families such as elementwise math, statistics,
time series, linear algebra and finance:

- **Primitive operator**: one independently meaningful, reusable calculation.
- **Composite template**: an explicit logical graph of existing operators whose intermediate
  calculations can be independently used or replaced. Complete business algorithms belong in
  upstream templates, lowered into the native graph, rather than new opaque C++ operators.
- **Coupled kernel**: an inseparable recurrence, joint constrained iteration or model fit whose
  decomposition into ordinary one-way edges would change numerical, state or temporal semantics.
  Document the justification, state initialization/reset, termination conditions and independently
  separable surrounding calculations. Complexity or speed alone does not justify this class.

Independently replaceable feature calculation, condition evaluation, classification, confirmation
and interval statistics must remain separable. Presentation belongs to the calling application.

For every new or changed operator, answer in the task's design and acceptance document:

1. What single problem does it solve?
2. Can an intermediate result be independently used or replaced?
3. Can existing canonical operators express the same calculation?
4. Would further splitting change numerical, state or temporal semantics?

### Keep logical granularity separate from physical execution

Multiple outputs may share one coupled kernel when they describe the same inseparable solve.
Expose separately addressable projections of the shared result where supported. Independent
business metrics must remain independent logical roots even when one native request computes them.

Lowering or template expansion must create real dependencies and preserve every used output and
downstream reference. Do not substitute a single primary output for a multi-output contract or
expose decorative child nodes that do not participate in execution. Preserve parameter bindings,
defaults, required inputs and meaningful diagnostics so callers can inspect or edit supported
graphs. Authoring, preview UI and undo remain responsibilities of the upstream application.

The native compiler and Planner may perform CSE, fusion, shared-state reuse and coarse parallelism
without changing public operator granularity. Preserve logical output provenance and the ability
to evaluate supported selected roots independently; do not force all independent results to be
computed or retained. A faster fused kernel or ISA variant does not by itself justify a new public
operator. Equivalence and reproducible performance evidence are required for such optimizations.

### Keep one explicit operator contract

Reuse the canonical C++ registry and kernels for common mathematical, statistical and sequence
semantics. Do not duplicate numerical implementations in Python, adapters or separate business
modules. New domain state/event/interval capabilities require explicit contracts.

For new or changed operators, the registry and its associated contract documentation must declare
responsibility, granularity class, input/output types, dtype/shape/axes, parameter defaults and
constraints, missing-value behavior, equality boundaries, window initialization/reset and output
semantics. Parser validation, Typed IR inference, lowering, public metadata and execution must agree
with that contract; display formulas or descriptions are not an alternative calculation authority.

Numeric values, conditions, states, events and interval boundaries require distinct declared types.
Missing data, false conditions, no new candidate, unclassified and neutral states must not be
silently interchanged or filled with zero merely to connect nodes.

Declare each operator's temporal dependency behavior. Preserve all actual dependencies and the
upstream semantic context through lowering and optimization. A future-dependent or full-sample
result remains retrospective after downstream comparisons; splitting, renaming or adding a lag
does not by itself prove causality. Business knowledge-time policy and research/backtest/release
gates remain upstream; the engine must not claim to certify them.

### Preserve compatibility and prove equivalence

Splitting must preserve data axes and lengths, dtype, NaN/Inf, window warmup/reset, endpoint
inclusion, state codes, threshold equality, confirmation counts, parameter bindings and all outputs.
It must also preserve the selected error policy, per-root result/status behavior and borrowed versus
independent output ownership. A mathematical-policy change requires an explicit algorithm/contract
version rather than being presented as a structural refactor.

Do not automatically rewrite saved definitions before equivalence is established. Historical
definitions, run snapshots, published versions and downstream references remain immutable; an
authorized definition transformation produces a new definition with explicit output mappings.
Historical aliases must use explicit version mappings into the current native implementation,
with compatibility tests, rather than guessed names or duplicate historical numerical runtimes.

Acceptance for affected paths must cover original-versus-composed values and statuses, boundary and
missing-data cases, temporal dependencies, historical contracts, expanded dependencies and output
references, and independently selected supported roots. Verify formula/graph round trips where the
public API supports them, and execution through the actual AOT package, not merely the presence of
C++ source. Run affected native and binding regressions; consuming-application UI and gate acceptance
belongs to an integration change. Apply the memory/performance checks in Sections 16–17 as relevant.

These rules govern authorized changes; they do not authorize redesigning existing operators or
claim that every future operator type, template or authoring capability is already implemented.

## 16. Performance Evidence

Do not claim an optimization is faster without measurement.

For SIMD/thread/process/shared-memory changes, benchmark representative shapes including:

- few products × long history
- many products × short history
- many products × many intervals
- many metrics sharing one DAG
- NaN-free dense input
- masked/NaN-heavy input
- contiguous vs strided input
- single process vs process pool
- thread scaling
- memory peak/RSS

Measure at least:

- wall time
- CPU utilization
- throughput
- peak memory
- input copies / bytes copied
- scheduler overhead
- process IPC/shared-memory overhead

### Phase-2 NJIT performance gate

For the standard multi-product × multi-interval × multi-metric workload, **warm native graph
execution is not allowed to be slower than the production BetterSaaTaa fused NJIT baseline**.
Cold-start/JIT avoidance and lower memory do not excuse slower warm numerical execution.

The current hard acceptance gate is deliberately stronger than parity:

```text
prepared/native paired median / NJIT paired median <= 0.90
ordinary Scheduler.execute paired median / NJIT paired median <= 1.00
```

The second condition applies to the required micro workloads so the public adaptive entry point
itself is never allowed to regress below NJIT, while the pre-bound hot path must retain at least a
10% margin.

Reference gate workloads are at least:

- 1 product × 1 interval × 5 metrics × 63 observations, using a pre-bound `PreparedGraphExecution`
- 1 product × 1 interval × 5 metrics × 252 observations, using a pre-bound `PreparedGraphExecution`
- 500 products × 2520 observations × 12 intervals × 16 shared-DAG metrics
- 1000 products × 2520 observations × 12 intervals × 16 shared-DAG metrics

2000 products is an extended scale check. Use alternating-first paired runs to reduce thermal and
execution-order bias. The micro gate compares NJIT's already-compiled `compute()` against the
equivalent already-bound native `PreparedGraphExecution.run()`; do not charge one side for
planning/binding work that the other side already completed. Run `tools/check_phase2_performance.py`
in a benchmark-capable environment.
If any required workload exceeds the ratio ceiling, performance acceptance fails; do not hide the
failure behind single-operator microbenchmarks or JIT compile-time comparisons.

## 17. Testing Requirements

Any high-performance change must preserve numerical and financial semantics.

Minimum tests for affected paths:

- reference parity
- scalar vs SIMD parity
- baseline ISA vs optimized ISA parity
- NaN/Inf policy
- empty input
- one element
- product offset boundaries
- interval boundaries
- contiguous/strided/negative-stride inputs where supported
- readonly inputs
- shared-memory attach/lifetime/cleanup
- thread-count determinism
- process-count determinism
- oversubscription guard
- exception propagation
- no input mutation
- no hidden input copy where zero-copy is promised

Use `np.shares_memory`, pointer/stride inspection, allocation counters, or equivalent evidence for
zero-copy claims.

## 18. Build and Portability

- C++17.
- PyBind11.
- scikit-build-core + CMake.
- AOT wheels only.
- CPython versions follow `pyproject.toml`.
- Keep Linux, macOS, and Windows build portability.
- No mandatory OpenMP dependency unless separately justified and benchmarked.
- Avoid global fast-math when IEEE/NaN behavior is part of a financial contract.
- SIMD runtime dispatch must retain portable baseline behavior.

### AOT execution admission and evidence

- Build native machine code before deployment, not during package import, service startup or formal
  requests. Parsing formulas and constructing native IR/DAGs or execution plans is permitted runtime
  work; it is not machine-code JIT compilation.
- Missing or incompatible native packages must fail closed. Never compile on demand, invoke a
  Python/Numba numerical fallback or fabricate NJIT signatures to satisfy a consuming application's
  execution gate.
- Graph execution evidence must identify the actual native backend, engine version, build identity,
  operator-registry and Typed IR contracts, plan fingerprint, CPU budget/usage and result ownership.
  Report request-time compilation, Python numerical callbacks and fallback counts explicitly; these
  must remain zero. Invalid or incomplete evidence required by an integration contract must be
  rejected by that integration, not inferred from a backend label.
- Admission must honor the input/output dtype and axis contracts, per-metric parameters, historical
  source versions and selected error policy. Numerical failure isolation must preserve unaffected
  roots where promised; malformed inputs, unsafe geometry and resource/worker failures must not be
  concealed as successful partial computation. Preserve strict-mode behavior where selected.
- Distinguish reusable borrowed output from independently owned results. Retained results must use
  the supported independent-output contract; an ndarray wrapper or read-only flag alone does not
  establish independence from a buffer reused by a later call.
- Validate the actual installed wheel and native execution path, including missing-package failure,
  zero Python numerical callbacks/fallbacks, result lifetime and applicable execution evidence.
  C++ source, a successful build or a declared backend string alone is not runtime acceptance.
- Keep standalone wheel execution independent of the research platform's source tree and runtime.
  Platform-side gates and business acceptance are separate integration checks; passing native tests
  does not certify those checks or prove that a platform service has stopped NJIT prewarming.

## 19. Source of Truth

- **Runtime execution source of truth: C++ only.**
- Numerical/operator implementation: CalMetricsEngine native C++ code.
- DAG/compiler/lowering/liveness/physical-plan implementation: CalMetricsEngine native C++ code.
- Planner/scheduler/CPU-memory admission/partitioning implementation: CalMetricsEngine native C++ code.
- Process/SharedMemory/worker lifecycle implementation: CalMetricsEngine native C++ code.
- Python package responsibility: interface adaptation only; Python must not become a second runtime source of truth.
- Execution-side Typed IR, alias lowering and rolling-scope semantics: CalMetricsEngine native C++ code.
- Business causality/knowledge-time and research-workflow semantics: FundInvestmentResearchPlatform.
- Historical behavior is preserved by tests and Git history, not duplicate current implementations.
- README describes public/user-facing contracts.
- `docs/architecture.md` describes architecture.
- This `AGENTS.md` describes implementation discipline for future AI/code changes.

Any change that introduces planner logic, numerical logic, execution scheduling, worker behavior,
memory-transport policy or operator semantics into Python is an **architecture violation** unless it
is strictly temporary migration code explicitly documented with a removal plan.

Canonical operator changes must preserve the single native registry and the pinned source
reference cases. A new ISA variant is an implementation of the same contract, not a new public
operator. Do not silently remap native opcodes to an external compiler's opcode numbering.

Distinguish zero-copy input binding from algorithm scratch initialization: report necessary
solver workspace copies separately. `Workspace` is exclusive to a running call; `out` cannot
alias shared inputs. A native scalar lane is legitimate and is not a Python fallback.

When these sources conflict, do not guess. Verify current code and tests before changing behavior.
