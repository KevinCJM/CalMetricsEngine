# Independent covariance-to-Cholesky F root: detailed design

Baseline native build: f90ddcc0b42dddba625903abf18e8860303b8e747548dd14e8b2e2d1ed246a57. This stage is a candidate until the root agent completes correctness, full-request time, and allocation/RSS validation. No commit or push.

## Admission and graph planning

Extend only the existing private adjacent covariance-to-Cholesky physical pair when it is the sole typed float64 matrix root, explicitly requested F order, with no isolate policy, root consumer, apply scope, or rolling scope. The covariance already has exactly one consumer, is not public, and has persistent input backing under the existing fusion proof. Do not admit unfused covariance inputs or alter the auto/C layout policy.

The existing independent-Cholesky-F predicate will accept a covariance parent only when the active execution metadata proves that exact fused pair. Finalization must pass its newly constructed metadata explicitly; it must not consult stale Program metadata. The same predicate already controls direct root admission, prebound Planner arena reservation, and executor downgrade for observer/prebound/input-failure routes. Those exceptional routes still remove the direct root destination and restore the original C arena before Scratch planning; they never use the new F writer.

## Numerical execution and ownership

Keep the current pair mathematics and both current physical backends. The pair receives the independent F result allocation but uses a local Output descriptor with C order during covariance, guarded Accelerate DSYRK/DPOTRF, canonical Cholesky, and canonical error replay. DPOTRF keeps its original U triangle. Source inputs remain read-only; the local descriptor does not mutate the published output order.

Only after successful completion, transpose the square matrix in place with standard-library swaps, bounded cache tiles, and no auxiliary numerical allocation. Both the successful Accelerate path and the successful canonical/replay path reach this single finalization. No transpose is performed when computation throws. Full swaps preserve the computed lower factor, cleared upper +0 values, diagonal, and IEEE bits. Publishing the existing F strides then yields exactly the original logical lower factor. There is no new public operator, algorithm, schema, protocol, dependency, thread pool, or cross-request cache.

The existing independent result owns its entire storage for this call. Previous results, distinct intervals, and worker/shared-memory outputs remain independent. Reject unsupported output order/private-pair combinations before touching output. Empty and 1x1 factors need no movement.

## Memory and audit

Planner/Scratch omit the additional D*D C arena only under the shared direct-root predicate; existing panel/vendor reserve remains. At D=4096, the target is removal of one 128 MiB matrix, not a promise about every RSS metric. Root copy bytes become zero; direct output bytes still describe the final allocation. The in-place transpose writes D*(D-1) double elements (two destinations per off-diagonal swap); count those bytes in algorithm_copy_bytes. This is an in-place layout conversion, not zero total copying.

## Regression requirements

Update the two existing Python tests that explicitly required the former F arena. Retain original bitwise numerical, readonly/ownership, zero upper triangle, failure and recovery assertions. Extend C/F/strided/padded input tests and C/F outputs to assert zero arena/root copy for qualified pairs. Compare C/F Planner budgets and preserve panel reserve. Cover actual single/thread/process-inline/process-shared execution, prepared snapshots across failures and input mutation, private-pair exclusion for public/shared covariance, temporary inputs, isolate, multi-root and nested graphs, plus Excel observer canonical values. The root agent adds native canary/replay and explicit F observer/prebound/input-failure downgrade tests and runs all numerical/build/performance checks serially.

No performance acceptance is inferred from source changes or the eliminated logical arena alone. Use the frozen old/new complete Scheduler harness, same CPU and result layout, independent retained outputs, and fresh measured setup/cold/warm heap and RSS. Include Accelerate success and canonical fallback, failure/recovery and exclusions.
