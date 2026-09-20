# C++-first engine: design and acceptance plan

## Requirement and scope

Retain the existing Python API while moving expression compilation, DAG lowering/CSE/liveness,
cost planning, partitioning, CPU admission, thread/process scheduling and shared-memory ownership
into an independently buildable C++17 core. Python retains import aliases, object adaptation and
an asyncio await bridge. This is not a rewrite of the platform's financial DSL or its data layer.

Four-question check:

1. Compiler/runtime migration is the current requirement; operator mathematics and platform code are not.
2. A native request executor is required to remove per-chunk Python dispatch; new finance formulas are not.
3. Reuse the 118 canonical operators and existing graph evaluator/fusion. Add native compiler/planner/runtime
   modules rather than another numeric implementation.
4. Preserve input immutability, exact dtype, scalar-root semantics, numerical order, public method names,
   normal independent outputs, and explicit Prepared output reuse. Resource-safety failures must fail closed.

## Target path

```text
Python arguments / NumPy owner pins / async await adapter
    -> one native request boundary
C++ restricted expression parser
    -> typed numeric IR -> composite lowering -> shared DAG/CSE
    -> alias-aware liveness + storage slots + compiled fusion eligibility
C++ cost planner
    -> exact interval observations / product groups / CPU+memory reservations
C++ Engine
    -> single or persistent native threads
    -> native worker executable via OS spawn (never embedded Python)
    -> native shared mappings and descriptor-only IPC for large inputs
C++ canonical operators / fusion / existing SIMD dispatch
    -> result owner and lazily materialized audit at the Python boundary
```

## Compatibility

`GraphCompiler`, `CompiledGraph`, `GraphNode`, `AdaptivePlanner`, `PlannerConfig`, `ExecutionPlan`,
`AdaptiveScheduler`, `PreparedGraphExecution`, `GraphExecutionResult` and shared-array wrappers remain
importable. Compatibility adapters must contain no math loops, graph optimization, partition algorithms,
CPU token logic, multiprocessing or thread-pool numerical dispatch.

The expression grammar retains numeric literals, declared series/scalar variables, direct canonical
calls, + - * / **, unary +/- and one comparison. Python evaluation is never used. Precedence is preserved
(e.g. -2**2 and 2**-2). Parser resource limits bound nesting, source bytes, nodes and arguments.
Unicode names declared by callers remain supported; arbitrary attribute/subscript/code syntax is rejected.

Only interval graphs with scalar roots are supported, as before. Matrix-growing nodes remain direct
operators, not silently treated as interval-series nodes.

## Native compiler and plan

The canonical opcode registry remains the source of operator contracts. Native IR records value class,
parents, binding index and exact floating-point constant bits. Composite lowering preserves evaluation
order. Identical nodes share one id. Liveness propagates through borrowed views (e.g. lag of an arena
intermediate); buffers cannot be reclaimed while any derived view is live.

Fusion eligibility is compiled once, not rebuilt on every execution. A versioned binary native plan
contains no pointers or Python objects and can be sent to native workers. Decoders bound every count,
validate opcodes/topology and reject incompatible versions. Plans have stable content fingerprints;
cache lookup must not assume hash equality alone proves arbitrary serialized plans equivalent.

## Native scheduling

A persistent bounded C++ thread pool owns tasks that contain only native values. A single engine-wide
CPU admission mechanism covers synchronous, concurrent and async-submitted jobs. A thread job uses one
process with N native workers; a process job uses N native processes with one numerical thread each.
No nested pools. Small jobs execute on the caller thread without queueing.

Cost estimation uses actual interval lengths, shared graph nodes, O(T)/sorting costs, product groups,
inputs, outputs, arenas, order-stat scratch, operator scratch and transport bytes. Estimates are disclosed
as execution-storage estimates, not exact process RSS guarantees. Operator constants and dynamic windows
retain canonical validation. A supplied plan must be checked against the current graph/workload and CPU
budget; changing data/interval geometry cannot reuse stale unsafe metadata.

Exceptions and timeouts drain/join all submitted work before releasing input pins, CPU tokens or mappings.
Thread cancellation is cooperative; only isolated native processes provide a hard termination boundary.
A single request deadline covers queueing and execution, not one fresh timeout per future. Close is
idempotent, stops admission and drains live work before joining workers.

## Native shared memory and processes

Shared regions use POSIX shm_open/mmap on macOS/Linux and file mapping on Windows. Ownership is RAII:
only the creator unlinks, attached workers unmap/close, and NumPy views retain a shared native owner.
Explicit release invalidates the Python owner interface but must not unmap memory underneath a live view.
Descriptors include mapping name, logical bytes, shape/type/offset. Validate sizes and offsets before
pointer arithmetic. No object arrays or raw process-local pointers are transported.

Use posix_spawn on POSIX or CreateProcess on Windows to start the packaged `calmetrics_worker` executable.
Do not fork an active threaded Python runtime, call Python from workers, or import Numba. Persistent workers
use bounded/versioned framed IPC. Hard-stop jobs use disposable workers and always terminate+wait on expiry.
Large inputs are copied to shared storage once unless already shared. Small-input inline IPC is measured
explicitly. Shared output may be returned by a NumPy view with a native owner, avoiding an unnecessary
final copy while preserving independent results between normal calls.

## Binding and owner rules

All pybind11 object creation, refcounts, dtype checks and Python exceptions stay under the GIL. Tasks
submitted to C++ threads carry raw checked views plus C++ owners only, never py::object. The calling
boundary pins input owners until every task has settled. Prepared calls verify binding geometry before
using cached pointers and reject resizing/retyping. Callers must not concurrently mutate input buffers.
Normal execute returns a new result. Prepared.run deliberately reuses its output (overwritten next run).
Audit is a per-call native snapshot, never stale first-call counters reused after interval changes.

## Validation and release gate

Keep pre-change source and benchmark evidence under ignored `.build-native-first/`.

- Existing numerical/contract suite plus direct C++ compiler/planner/thread/process tests.
- AST injection/precedence/resource limits; CSE and borrowed-view liveness.
- Serial/thread/native-process/shared/async parity; owner lifetime and output independence.
- Input geometry mutations, supplied stale plans, deadlines, worker crashes, close/admission and cleanup.
- Standalone native build without Python; worker binary must not link Python.
- Release, ASan/UBSan and local x86_64 execution where available.
- Actual installed wheel/sdist; CPython 3.10-3.14 and NumPy 1.26 compatibility.
- Micro: 1 product x 1 interval x 5 indicators at 63/252/504/2520 observations.
- Batch: 500/1000 products x 12 intervals x 16 indicators, same real NJIT baseline and paired order.
- Prepared Native/NJIT <=0.90; ordinary public micro execute/NJIT <=1.00.
- Also measure plan+execute (not just preplanned), process startup vs reuse, scratch and copy accounting.

Performance gates are runnable checks, not automatically enforced by merely adding a script. Cross-platform
CI, release upload and production integration must not be claimed without actual evidence.

## Technical references

The implementation follows the pybind11 GIL/lifetime guidance (official advanced/misc documentation),
POSIX shared-memory/spawn contracts, and Microsoft file-mapping/process APIs. RAII cannot clean up an
object in a killed worker; parent-owned cleanup and wait-for-exit are therefore required, not optional.
