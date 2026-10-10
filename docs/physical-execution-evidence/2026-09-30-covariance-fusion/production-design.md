# F covariance native-library fusion design (2026-09-30)

Baseline actual installed wheel 63090c5a7971acd8c947ef5b0e20a46bf207f88772e8c322ab94a3537ecd96bb. Current task extends the previous C-layout optimization and must keep its frozen evidence.

Candidate screening separates mean8, pair2x2, their combination, and cross-output gather4x4. The combined mean8 + pair2x2 is the initial preferred candidate: 229 fixtures/1832 records pass original 1e-12, IEEE, discrete and integrity checks; pair-combination preserves the pinned Eigen two-packet reduction tree. Gather4x4 has 124 diagnostic bit differences but no test failure; it is slower than the combined candidate in the measured long/wide grid, so no reason to change the reduction tree. One fixed pair2x4 extension will be assessed before final selection; no automatic tuning or per-case dispatch search.

## Production integration

Only single-matrix covariance/correlation backend on existing macOS15+arm64 Eigen route. Retain N>=32,D>=8, row_stride=1, column_stride>=N, finite and existing overflow bound admission; unsafe/nonfinite/special stride and smaller dimensions retain the existing full-N expression/mean reduction. Scalar/Eigen-OFF/other-platform and two-vector overload unchanged. No canonical registry, public math, AST/DAG, planner, IPC or Excel formula/budget change.

Fixed public Eigen Array mean8 gathers independent columns and runs the same per-column Welford recurrence. Products share centered source packets across independent outputs; pinned Eigen3.4 public head/tail/sum reproduce the prior packet grouping. Last output row holds means until the final diagonal; output normalization/mirroring remains in matrix.cpp. No input writes, numeric pack, dynamic workspace, persistent cache, private threads or reciprocal/FMA substitutions. Each request computes current input and produces independently owned outputs.

Split finite admission, mean setup, and product execution with the same four-reference bool ABI as library_covariance_upper. Tail calls release the completed admission/driver frame before products, preserving the existing known numeric stack bound. Inspect actual arm64 objects: source fixed Arrays do not prove register residence or stack neutrality. Audit count increments exactly once on successful native library path; rejected paths do not write output or increment.

## Evidence and governance

D1-D5: dependency closure, feature axes, per-observation Welford recurrence and per-pair product tree remain explicit; no cross-request reuse, hidden value-based cache or omitted mean consumer. D6: single budget-controlled thread, no shared-memory protocol change. D7: no random/state algorithm. D8: serial correctness/build/timing, actual baseline archive/wheel identity, small/large and C/F/padded coverage, original strict phase2 gates, complete public request timing and separate heap/known stack/RSS. X1-X10: same sample divisor/errors/formula expansion and cell budget; existing real Excel recalculated snapshot may be replayed with explicit historical limitation, ordinary requests never run Excel.

Production adoption needs focused remainder/ownership/error regressions, full Python and native/ASan/OFF, primitive replay, frozen Excel comparison, source/wheel identity; then fixed old/new/NJIT public timing and memory. Existing failures and untracked frees/RSS limitations remain. New prototype speedups are not full-request acceptance. No commit/push or FIRP change authorized this round.

## Selection after fixed V4 screening

Adopt mean8 + pair2x2. All229 fixtures/1832 records for the V3 and V4 grids pass original tolerance/IEEE/discrete checks. The124 bit differences in each grid belong to the gather candidate and are diagnostic only. Gather is slower than combined on long/wide cases. Pair2x4 cuts hot instruction loads another25%, but its actual prototype stack is560 B vs combined416 B (including mean helper); its wider-case extra time gain is only about8%. It is not adopted because its extra56 B numeric spill and larger frame do not establish memory neutrality. Preserve candidate source/build/timing/negative results; do not retain them as parallel production implementations.

Actual integration uses namespace covariance_f_detail, fused mean/product helpers, tail-called admission and fast driver. Existing fallback full-N expression remains solely for current unsupported feature/geometry/nonfinite contracts. Formal object and public request measurements follow; prototype results do not substitute. New regression additions cover N%4=2/3 and D%4=2/3, full final blocks, mean8 seven-column tail, four IEEE/discrete patterns and34x12 prepared snapshots; native expands original4shapes to6shapes×C/F/paddedF. Auto-path backend counts are checked on eligible bounded fixtures.
