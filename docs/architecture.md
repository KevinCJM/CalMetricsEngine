# CalMetricsEngine architecture — C++ first

## 1. Current execution boundary

```text
Python public names / NumPy pins / exception conversion / asyncio await adapter
    |
    | one binding transition per synchronous request
    v
C++ restricted expression compiler (compiler.cpp + typed_ir.cpp)
    AST -> Typed IR -> alias canonicalization -> lowering -> structural CSE
    -> borrowed-view-aware liveness -> immutable graph program
    -> compiler-owned rolling/block/filter/group/root-solve scopes
    v
C++ planner (planner.cpp)
    actual interval lengths -> product/interval cost -> storage estimate
    -> CPU budget -> single/thread/process policy
    v
Process-wide C++ NativeScheduler (scheduler.cpp)
    shared CPU admission + one persistent native ThreadPool
    v
C++ Engine (native_runtime.cpp)
    per-Engine local CPU cap + persistent/disposable native ProcessPool
    v
C++ graph executor (graph.cpp)
    shared reductions / sorting / regression state
    -> rolling-window sub-program execution with state reset
    -> scalar or aligned time-series roots
    -> reusable arenas and Workspace
    -> canonical operators and existing SIMD dispatch
```

Execution-side Typed IR for scalar/time-series indicators is now native C++: dtype, time-axis identity,
symbolic shape, semantic dimension, price basis, historical alias canonicalization, logical
`rolling_window`, compiler-owned `rolling_apply`, and aligned time-series roots all live below the
PyBind boundary. Business causality/knowledge-time governance remains in the research platform.
The same 125-entry registry now supports typed matrix/vector bindings and matrix intermediates
inside the interval DAG, together with exact int64 category/index storage. Public graph roots remain
scalar or aligned float64 series; a general public matrix-output API is a separate contract.
Native block/filter/group/bisection sub-programs make segmented statistics and iterative scalar solves
composable without Python callbacks. See [the mathematical composition design](mathematical-composition-design.md).

## 2. Module responsibilities

| Native module | Responsibility |
| --- | --- |
| `compiler.cpp` | Restricted parsing, precedence, Typed Logical IR construction, alias/rolling lowering, CSE, liveness, native plan encoding |
| `typed_ir.cpp` | ValueType validation and operator type/axis/semantic/price-basis inference |
| `planner.cpp` | Exact interval geometry, weighted product partitioning, physical cost/storage estimates and strategy |
| `scheduler.cpp` | Process-wide C++ CPU admission and lazily grown persistent ThreadPool shared by all numerical entry points |
| `native_runtime.cpp` | Engine-local CPU cap, Graph/process execution, task draining and shutdown on top of the shared scheduler |
| `native_process.cpp` | Framed/versioned native IPC, persistent worker reuse, disposable hard-stop workers |
| `native_worker_main.cpp` | Standalone executable entry, no Python interpreter |
| `shared_memory.cpp` | POSIX/Windows shared regions with native RAII ownership |
| `graph.cpp` | Scalar/series numerical graph execution, fusion, rolling-scope sub-programs, arenas and scratch reuse |
| `graph_bindings.cpp`, `native_api_bindings.cpp` | Python type adaptation, pinned arrays, exception/result conversion |

Python `graph.py`, `planner.py`, and `shared.py` export compatibility aliases. `runtime.py` only bridges
blocking native requests into `asyncio` and exports compatibility names. No Python CSE, liveness, cost
algorithm, numerical chunking, CPU-token loop, or numerical process/thread pool remains.

All public numerical entry points participate in process-wide native CPU admission: Graph Engine calls,
legacy finance compatibility APIs, direct canonical-operator calls and low-level Python
`Program.execute()`. Direct single-thread APIs consume one token; legacy `n_threads` is only a
request ceiling and uses the shared persistent pool.

`asyncio.to_thread` is an await adapter, not the engine's numerical worker pool. Cancelling an awaiter
does not forcibly kill a running native calculation; isolated processes and execution deadlines provide
that hard-stop boundary.

## 3. Compiler and plan

Accepted expressions are numeric constants, declared scalar/time-series/vector/matrix variables, canonical or
historical-alias calls, arithmetic, unary signs and one comparison. Typed declarations additionally
carry dtype, named axes, symbolic shape, semantic dimension and optional price basis. Arbitrary Python calls, attributes, subscripts, imports,
lambdas and comprehensions are rejected. Source bytes, nesting, node counts and arities are bounded.

Operator names/opcodes and mathematics still come from the canonical registry. Alias names are
canonicalized in C++ before the physical DAG is built; they do not add numerical kernels. Logical
`rolling_window` nodes lower without materializing a `T×W` matrix. `rolling_apply` owns a compiled
scalar body sub-program and executes it on trailing window views with independent state reset.
Lowering also shares `linear_fit` results and total-return results without changing evaluation order. Borrowed `lag`, `transpose` and matrix `diag` views
extend the lifetime of their backing arena transitively. Plans sent to workers contain no pointers or
Python objects; the versioned decoder bounds counts and validates topology/opcodes.

A native graph fingerprint is a cache label, not an authentication hash. Graph compatibility also
checks the encoded program. Supplied plans validate input sizes and interval/product geometry before
execution; changing intervals requires replanning.

## 4. Arrays and memory

Current interval-DAG input storage is exact native `float64`, one-dimensional time-series data,
aligned and contiguous within each product; the Typed IR can describe broader vector/matrix types,
but full matrix graph binding remains out of this execution path. Interval arrays are exact `int64`
and half-open `[start, end)`. No implicit data coercion or
compaction occurs. Generic direct operator APIs retain their strided/readonly support.

The binding layer pins Python owners until every native task settles. Prepared calls reject a changed
pointer, dtype, shape, stride or output geometry. Callers must not resize/retype or concurrently mutate
arrays while a native call uses them. Parameters can change through an unchanged parameter array;
ordinary mapping parameters are rebound when values/keys change.

Scalar graphs return `(interval_rows, roots)`. Aligned time-series graphs return one contiguous
`(sum(end-start), roots)` matrix plus `int64` prefix offsets so each interval is a zero-object-overhead
slice. Outputs and algorithm workspace may be allocated. Zero-copy input binding is not a claim of zero
allocation or zero algorithm scratch initialization. Quantile/median sorting and solvers report their
necessary algorithm copies separately. Native graph audit includes order-stat scratch as well as the
main arenas and operator workspace.

Memory budgets constrain estimated execution storage, not exact resident memory. Allocator high-water
retention, interpreter memory, operating-system page accounting and unrelated requests are not a hard
RSS guarantee. Current native estimates are conservative, including legacy copied-output capacity even
when the Python binding returns the new zero-copy shared output view.

## 5. Scheduling

Work depends on each interval's actual observations and on the **physical execution DAG after
lowering/CSE/fusion**, not row count times the longest window and not a blind logical-node sum.
Compiled graphs expose both logical and physical work estimates.

Many products use physical-work-weighted product blocks; a few products with many intervals use
physical-work-weighted interval blocks. Metrics remain roots in the same shared DAG rather than
separate jobs that duplicate their inputs.

When row/product/interval parallelism cannot fill the CPU budget, Compiler-provided independent root
components may be scheduled as dependency-closed DAG branches. Shared operation prefixes and
summary/order fusion groups are collapsed before branches are formed, so branch fork/join does not
duplicate shared computation. Branch tasks are coarse-grained and execute precompiled sub-programs,
not individual DAG nodes.

Small requests execute on the caller's native thread. Larger in-process requests use a persistent C++
thread pool; one caller can participate while other chunks run on native workers. Very large or isolated
requests can use persistent native processes. Default policy is thread-first; large work alone does not
prove processes faster. Thresholds are tunable and must be supported by workload benchmarks.

One process-wide native C++ Scheduler owns the shared CPU-admission budget and persistent ThreadPool.
Each Engine keeps a local CPU cap, but Graph jobs and legacy finance APIs also acquire from the same
process-wide budget, preventing multiple Engine instances or compatibility calls from silently
oversubscribing one another. Legacy `n_threads` is a request ceiling and never creates private
threads.

Process workers run one numerical thread each; nested process × thread hybrid execution is not enabled.
Exceptions drain submitted work before returning or releasing leases. Thread timeouts are cooperative at
chunk boundaries; an in-flight kernel is not forcibly stopped. Hard-stop requests use disposable native
workers and terminate-and-wait before releasing shared storage.

## 6. Native processes and shared regions

On POSIX, workers use `posix_spawn`; on Windows the implementation uses `CreateProcess`. Workers are
packaged beside the extension as `calmetrics_worker` (or `.exe`) and link only the native core and system
runtime. They neither import Numba nor start a Python interpreter. Runtime code is AOT; dynamic formulas
compile to native IR, not machine code via JIT.

Large inputs use shared region descriptors, while small-input inline IPC is explicit and counted.
Existing `SharedInputBundle` regions can be reused across requests. Creating a shared region from an
ordinary NumPy array is one explicit boundary copy, not falsely described as zero-copy ingestion.

Workers write disjoint result rows into a parent-owned mapping. The Python result is a NumPy view of
that region, pinned by a C++ `shared_ptr` owner: **no final shared-output-to-NumPy copy**. Every normal
request owns distinct output storage. Closing the engine/owner does not unmap an exported live view.
`PreparedGraphExecution.run()` is intentionally different: it reuses output and overwrites the previous
contents on the next run.

Creator regions unlink on final lifetime release; attached workers only unmap/close. Shared input
regions are sealed read-only after their one-time ingestion copy. Their public descriptors retain that
readonly contract, so even an explicit writable attach request cannot reopen an input mapping for writes.
Physical read-only mappings export read-only buffers, so `setflags(write=True)` cannot enable writes to
protected pages. Parent cleanup is required even when a child is killed; RAII inside a terminated worker
cannot run.

## 7. Public API and validation

Existing names remain: `GraphCompiler`, `CompiledGraph`, `AdaptivePlanner`, `PlannerConfig`,
`ExecutionPlan`, `AdaptiveScheduler`, `PreparedGraphExecution`, `GraphExecutionResult` and shared owners.
The object implementation is native; Python dataclass internals are not an execution contract.

Regression covers canonical parity, parser restrictions/precedence, alias liveness, rebinds, stale plans,
ordinary output independence, shared output lifetime, worker failures, timeouts, engine reuse/close,
concurrent CPU admission, and proof that synchronous requests do not use Python parsing or numerical
pools. Native compiler/planner/process tests build with Python disabled.

Performance acceptance compares the complete workload to the real BetterSaaTaa NJIT baseline using
paired alternating order. Prepared/batch ratio must be <= 0.90; ordinary scheduler micro ratio <= 1.00.
This is a runnable gate, not a guarantee about every formula or machine. Cross-platform results must
be labelled actually executed vs configured only.

Detailed design: `cpp-first-design.md`. Current verification: `cpp-first-acceptance.md`.
Historical Phase-1/2 reports are retained as measurements of earlier implementations, not the current
Python/C++ division of responsibility.
