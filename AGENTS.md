# CalMetricsEngine Repository Guidelines

## 1. Project Mission

CalMetricsEngine is the reusable high-performance calculation engine for the fund investment
research platform.

The long-term execution chain is:

```text
DSL / AST
    ↓
Typed DAG
    ↓
Operator Lowering
    ↓
Native Execution Plan
    ↓
Single Scheduler
    ↓
Shared Memory / Process Pool / Native Thread Pool
    ↓
PyBind11 / C++ AOT Runtime
    ↓
SIMD Kernels
```

Python owns **what to calculate**. C++ owns **how to calculate it efficiently**.

Do not reintroduce runtime Numba/JIT compilation into CalMetricsEngine.

## 2. Current vs Target Architecture

### Current

The current repository already provides:

- `calmetrics_engine` Python package.
- One PyBind11 native extension: `calmetrics_engine._native`.
- C++17 AOT finance kernels.
- Exact-dtype zero-copy NumPy input boundary.
- Strided `ArrayView` support.
- Per-call native threading.
- Cross-platform wheel build configuration.

### Target

The following are architectural targets and must not be claimed as already implemented until
code and tests exist:

- Production Typed DSL / AST / DAG migrated from FundInvestmentResearchPlatform.
- NativeExecutionPlan.
- C++ DAG executor.
- Runtime SIMD ISA dispatch.
- Persistent native thread pool.
- Unified ExecutionScheduler.
- Managed process pool and shared-memory execution.

## 3. Scope Discipline

Before changing code, answer these four questions:

1. Is the change required by the current request?
2. If it is not changed, does it block the current request?
3. Can the requirement be solved with a smaller change?
4. Does the change alter numerical, financial, temporal, memory, or concurrency semantics?

Do not refactor unrelated code merely for elegance, technical-debt cleanup, or speculative future
needs. Do not fix unrelated bugs unless the current task requires it.

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

Conceptual work size:

```text
Work ≈ Product × Interval × DAG Cost × Observation
```

This is a cost model, not a fixed threshold formula. Thresholds must come from reproducible
benchmarks.

## 10. Parallelism Hierarchy

### Single-thread + SIMD

Use for small workloads where scheduler/parallel overhead dominates.

### Native thread pool

Preferred for CPU-bound work inside one process when:

- data already lives in the process address space
- tasks are independent
- GIL is released
- failure isolation/hard-stop is not required

Priority dimensions are generally:

1. product blocks
2. interval blocks when one/few products have many independent intervals
3. scenario blocks
4. independent heavy native kernels

Do **not** default to metric-level parallelism because DAG outputs often share upstream work.

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

The Scheduler owns a total CPU budget.

Bad:

```text
8 processes × 16 native threads = 128 runnable threads on a 16-core machine
```

Acceptable examples:

```text
1 process  × 16 threads
4 processes × 4 threads
8 processes × 2 threads
```

Actual choices must be benchmark-driven and may reserve cores for the API/service process.

Native kernels must accept a scheduler-provided thread budget and must not silently spawn an
unbounded number of threads.

## 12. Shared-Memory Lifecycle

Shared memory is an execution transport, not a business-data store.

Rules:

- Prefer one ingestion/copy into shared storage over one copy per worker.
- When possible, load data directly into mmap/shared storage and remove even the initial copy.
- Inputs shared across workers are read-only.
- The owner keeps shared memory alive until all workers finish.
- Cleanup/unlink is explicit and exception-safe.
- Never pass raw process-local pointers across processes.
- Workers reconstruct views from descriptors.
- Output ownership must be explicit: local result, shared output, or reduction result.

## 13. Native Thread Pool

The target is a persistent native thread pool owned by the execution runtime, not
`std::thread` creation for every small kernel call.

Requirements:

- bounded workers
- scheduler-controlled concurrency
- no nested unbounded pools
- exception propagation
- deterministic completion
- clean shutdown
- ability to execute product/interval/scenario blocks
- reusable per-worker scratch buffers where safe

Until that pool exists, existing per-call threading must not be described as the final scheduler.

## 14. DAG Execution and Memory Planning

The target is one Python→C++ transition per execution plan:

```text
Typed DAG
    ↓
NativeExecutionPlan
    ↓
C++ DAG Executor
    ↓
SIMD/native kernels
```

Avoid:

```text
Python node → PyBind → C++ → Python node → PyBind → C++
```

Execution-plan lowering should identify:

- operator opcode
- input/output slots
- immutable parameters
- dtype/shape contracts
- product/date axes
- workspace bytes
- liveness interval
- SIMD eligibility
- thread/process eligibility
- missing-value policy

Buffers whose values are dead must be eligible for workspace reuse.

## 15. Operator Design

Operators are divided by smallest independent numerical semantics:

- primitive elementwise
- reductions/statistics
- time-series/rolling
- matrix/linear algebra
- finance
- coupled state/model kernels

Do not create one black-box C++ operator merely because a business algorithm is complex.

A coupled kernel is justified only when splitting it would break recursive state, joint
optimization, model fitting, or exact temporal semantics.

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

## 19. Source of Truth

- Numerical implementation: CalMetricsEngine native/runtime code.
- Production DSL/Typed-DAG semantics until migration: FundInvestmentResearchPlatform.
- Historical behavior is preserved by tests and Git history, not duplicate current implementations.
- README describes public/user-facing contracts.
- `docs/architecture.md` describes architecture.
- This `AGENTS.md` describes implementation discipline for future AI/code changes.

When these sources conflict, do not guess. Verify current code and tests before changing behavior.
