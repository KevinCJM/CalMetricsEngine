# CalMetricsEngine architecture — C++ first

User-facing entry points: [user guide](user-guide.md), [operator reference](operator-reference.md),
and [execution guide](execution-guide.md). This page focuses on implementation responsibilities.

Before changing these mechanisms, apply the [computation design rules](computation-design-rules.md).
They govern dependency proofs, reuse, mathematical equivalence and resource evidence; conditional
requirements for future capabilities do not extend the implemented scope described by code and tests.

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
    -> scalar/aligned-series compatibility output or heterogeneous typed roots
    -> reusable arenas and Workspace
    -> canonical operators and existing SIMD dispatch
```

Execution-side Typed IR for scalar/time-series indicators is now native C++: dtype, time-axis identity,
symbolic shape, semantic dimension, price basis, historical alias canonicalization, logical
`rolling_window`, compiler-owned `rolling_apply`, and aligned time-series roots all live below the
PyBind boundary. Business causality/knowledge-time governance remains in the research platform.
The same 190-entry registry now supports typed matrix/vector bindings and matrix intermediates
inside the interval DAG, together with exact int64 category/index storage. Public roots include scalar, series, vector and matrix. Heterogeneous typed results retain each root's dtype, named axes, actual per-interval shape and statuses; legacy scalar/aligned-series arrays remain compatible.
Native block/filter/group/segment/bisection sub-programs make segmented statistics and iterative scalar solves
composable without Python callbacks. See [the mathematical composition design](mathematical-composition-design.md).

Stateful native records retain nominal tags and named field axes while sharing ordinary
numeric/integer matrix storage. Projection views extend their backing arena lifetime;
matrix widths are included in memory admission. Adaptive/second-order recurrence and
scalar Kalman filters retain explicit seed, gap and warmup contracts. State, event and
complete-wave semantics are described in [the stateful series design](stateful-series-design.md).
Program serialization supports v7/v8 (v8 retains layout policies; older formats retain their supported schema); worker IPC v21 carries exact output
dtype/byte width, Eigen rotation/covariance/matvec, guarded Accelerate pair/MatMul,
covariance attempts/success/replay/fallback and panel materialization, exact-zero
covariance/MatMul counters, actual guarded vForce, compact scan group/batch/step counts and completed Highway numeric-line sort calls, plus the maximum per-chunk library workspace admission
estimate; it rejects incompatible peers. Typed failure placeholders require the
parallel status array, never implicit numeric conversion or False/0 success.

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
| `operators/cholesky_library.hpp` | Ordered Eigen Array row panels borrowing bounded Workspace; original row-finalization and numeric-failure replay remain in the canonical caller |
| `operators/covchol_accelerate.cpp` | macOS arm64 单根协方差→Cholesky 的有界 Accelerate 面板、数值守卫及原路径回退准入；Planner 预算和独立结果由图层负责 |
| `operators/matmul_accelerate.cpp` | Independent nonnegative final float MatMul root: per-call input-domain/stride admission, borrowed inputs, direct output, scoped single-thread BLAS and exact finite-zero fill; Planner reserves are estimates, not measured allocations |
| `operators/matrix_backend.cpp` | Pinned Eigen rotations/covariance and ordered matrix-vector feature lanes; borrowed inputs, independent output accumulation; no private pool, BLAS state or Python callback |
| `operators/transcendental_accelerate.cpp` | Guarded independent exp/log output; same-order/continuous spans without new scratch, cross-layout rank-2/3 vForce/vDSP tiles bounded to 64KiB and covered by Planner; canonical rejection before writes |
| `operators/numeric_line_sort.cpp` | Shared numeric-line sorting for `quantiles_axis` and compact consumers: guarded Highway VQSort on caller-owned scratch, otherwise canonical NaNLast standard-library sorting; no private pool or borrowed-input writes |
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

A complete compact branch may share the main Program's `const ExecutionMetadata`, including its
immutable compact proof, only after every root/node mapping, slot, Program field, root schema and
constant bit pattern matches. Scope-bearing and nonidentity branches retain separately finalized
metadata. Structural validation still runs; public definitions and serialization are unchanged.
This avoids rebuilding the same physical proof, not numerical execution: request Scratch, state,
input ownership and output storage are not shared by this mechanism. Worker scratch/work budgets
remain unchanged.

An adjacent single-consumer `symmetric_eigh → eigenvalues` may execute as a
values-only physical kernel in a healthy typed graph without an observer. It
preserves bounded Jacobi arithmetic and errors while dropping unused vector
rotations and the full record. Copying the source before writes permits dead
arena reuse; independent roots write their own storage. Logical view lifetimes
remain valid for shared vectors, isolation and observation. Full records use
standard-library index sorting and cycle copies at n≥128, with N indices in the
workspace budget. No operator, nominal type, opcode or worker protocol changes.

A native graph fingerprint is a cache label, not an authentication hash. Graph compatibility also
checks the encoded program. Supplied plans validate input sizes and interval/product geometry before
execution; changing intervals requires replanning.

## 4. Arrays and memory

Float64 series inputs are aligned and C-contiguous. Typed vector/matrix/tensor, int64 and mask
inputs also support validated signed strides; dense C/F inputs retain their physical layout.
Interval arrays are exact contiguous one-dimensional `int64` and half-open `[start, end)`.
Input binding does not silently coerce dtype or compact arrays. Process transport may pack
non-dense inputs, with the copy included in its budget and audit. Generic direct operator APIs
retain their own strided/readonly contracts.

The binding layer pins Python owners until every native task settles. Prepared calls reject a changed
pointer, dtype, shape, stride or output geometry. Callers must not resize/retype or concurrently mutate
arrays while a native call uses them. Parameters can change through an unchanged parameter array;
ordinary mapping parameters are rebound when values/keys change.

Scalar graphs return `(interval_rows, roots)`. Aligned time-series graphs return one contiguous
`(sum(end-start), roots)` matrix plus `int64` prefix offsets so each interval is a zero-object-overhead
slice. Other root combinations, or explicit `result_format="typed"`, use `ResultLayout`:
interval/root-ordered capacity slots aligned to eight bytes, one numeric payload, per-root actual
shapes/statuses, and readonly output views retaining the payload owner. Capacity is planned without
reading numerical values; actual dynamic lengths are recorded during the same DAG execution.
Metadata/status storage participates in memory admission. Current worker requests use protocol 26;
response layout is 25. They carry geometry, resolved output order and completed kernel audits.
Row-geometry optimization flags are validated, old/unknown requests fail before result writes, and
parent/worker identities must match.
Program encoding v8 records root order policies; all-C definitions may retain v7 encoding.
New graph compilation accepts `output_order="auto"|"C"|"F"`. Supported roots write directly
to the selected independent layout; other explicit-F roots use budgeted C scratch followed by
materialization. NumPy views expose the true strides without a boundary transpose or copy.
Typed auto/F results use `typed-results-2`; explicit C/legacy results retain `typed-results-1`.
Positional status arrays remain logical C. General DAG intermediate/state layout optimization
remains pending; see [current layout design](physical-layout-design.md) and
[historical typed result design](typed-results-design.md).

Outputs and algorithm workspace may be allocated. Zero-copy input binding is not a claim of zero
allocation or zero algorithm scratch initialization. Quantile/median sorting, solvers and admitted
large-matrix panel packing report their necessary algorithm copies separately. Native graph audit includes order-stat scratch as well as the
main arenas and operator workspace.

On Apple arm64 builds with Highway enabled, numeric-line sorting uses VQSort only for automatic ISA,
at least 1024 values and runtime float64 support. NaN, any negative zero, or any nonzero subnormal value
keeps the canonical NaNLast sort; infinities alone do not exclude the library. Only mutable sorting
scratch is written. `highway_vqsort_calls` reports actual completed library calls. Planner reserves
an additional estimated 16KiB per active worker for graphs that may execute `quantiles_axis`,
including nested scope bodies. This conservative library stack/TLS allowance is separate from the
N-double sort scratch; it is not an allocated buffer or a strict bound on all OS/compiler stack
frames. The audit's library reserve records the maximum estimate used by a chunk, not measured RSS.

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

Performance acceptance compares complete workloads against warmed NJIT with equal CPU budgets
and paired alternating order. The historical BetterSaaTaa script retains prepared/batch <= 0.90
and ordinary micro <= 1.00 as diagnostics; a few KB or milliseconds of regression alone do not
block adoption under the current [D8 policy](computation-design-rules.md#10-d8用对等的完整负载证明收益).
Actual batch effects, correctness, ownership and resource budgets still apply. Cross-platform
results must be labelled actually executed vs configured only.

Detailed design: `cpp-first-design.md`. Current verification: `cpp-first-acceptance.md`.
Historical Phase-1/2 reports are retained as measurements of earlier implementations, not the current
Python/C++ division of responsibility.

The latent-model extension (opcodes 161–185) follows the same registry/Typed IR/Planner/worker protocol. `latent.cpp` owns numeric kernels; `typed_latent.cpp` owns nominal model projections and axis/shape contracts; `excel_latent.cpp` owns formula expansion. EM returns one shared internal model; public roots project independent scalar/vector/matrix results. Training, initialization and inference remain separately editable DAG nodes. See [design and acceptance](latent-model-operators.md).

### Qualified row and result storage reuse (2026-10-09)

The physical executor can borrow validated consecutive C rows from immutable inputs or already
completed independent roots, including matching private pointwise training trees. Logical nodes
still run their own preparation and preserve validation/error order. Admission retains original
geometry whenever proof storage would reject a previously admissible budget.

A private int64 argmax used only by a proven pointwise state-decision region can use the current
independent terminal root when both length and actual slot saving justify it. Bounded original-label
tiles preserve comparisons during in-place writes; cancellation preserves the completed prefix and
restores the unpublished suffix. Latent history borrowing also restores an unwritten future float64
root before later nodes or exceptions can expose it. Inputs and previous results are never loaned.

These are physical changes, with no new operators, numerical models or Excel formulas. Frozen full/
SAN/library-OFF qualification and three paired groups precede actual-source rebuild and six-chain
replay. [Evidence and remaining workloads](physical-execution-evidence/2026-10-09-latent-current-v69/future-result-screen-v74/qualified-adoption/row-state-reuse-v78/README.md) distinguish stage adoption from the still-open
whole-engine time/memory objective.

### Gaussian/filter tail replay (V85, 2026-10-09)

Eligible private Gaussian-density/filter/last-row chains use bounded128-row
density tiles, a full validation pass and the original ordered filter recurrence.
Static backing/liveness and runtime geometry proofs precede omission of NK
histories. Planner includes the extra density pass at top level and recursively
weighted healthy child scopes. The131K-double workspace is admitted normally.
Canonical replay preserves errors and independent outputs; public Prepared,
observation and unsupported geometry retain the original route.

Completed `gaussian_filter_streams` audits use worker request26/response25.
Older epochs fail before decoding; this is not a general build-ID handshake,
so deployments must still pair extension and worker from the same wheel.
Logical operators, numerical contracts and Excel formulas are unchanged.
[R4 and actual-source qualification](physical-execution-evidence/2026-10-09-latent-current-v69/future-result-screen-v74/qualified-adoption/row-state-reuse-v78/README.md#gaussian-filter-stream-v85)
separates reused R3 evidence, fresh R4 tests and actual complete paired replay.
