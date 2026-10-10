# Softmax output reuse with guarded public Eigen exponential (2026-09-30)

Baseline f0e62a925728d638bcadfce326bdc054b08fcdf80251cee5fc0a23f55fe0b72e.
Scope: implementation of existing softmax_axis only; no new logical operator,
AST/DAG, Planner, IPC, dtype/shape, ownership, Excel formula or budget changes.
One input axis normalizes independently; no cross-row or cross-request reuse.

MacOS 15+ arm64 with pinned Eigen 3.4, non-scalar requested ISA, total size >=256,
axis count >=8, input axis stride=1 and output inner=1 are eligible. Other cases
retain the original scalar implementation. Every row performs original finite
validation before output writes. Min/max guard requires max-min <=700-log(count).
Overflowing finite span is ineligible. This conservative domain avoids Eigen's
pexp_double clamp at -709.784, exp underflow and normalized subnormal values:
exp(-span)/sum >= exp(-span)/count >= exp(-700), with a large margin above DBL_MIN.
Guard is implementation selection, never a user parameter or mathematical policy.

Eligible row maps borrowed input and independent output using public unaligned
Eigen Array Maps: shifted exp writes output, scalar ascending sum retains original
reduction order, then public Array /= performs true division. Eigen's polynomial
may use explicit FMA; original numerical tolerances remain mandatory, not bitwise
identity. Tail, zero/equal maxima and downstream known discrete classifications
must be checked. No dynamic mathematical workspace, packing, input copy, private
threads, extra output, or retained state. Output is not exposed until completion.
Nonfinite row raises original NON_FINITE_RESULT before writing that row; preceding
rows and isolated downstream error propagation remain governed by existing engine.

Initial unguarded prototype had 198/2448 classification failures (extreme exp was
nonzero instead of zero), preserved in softmax/screen.json. Guarded span700
prototype passed 2736 cases and shows candidate gains, not production acceptance.
Production uses the stricter count-adjusted domain. Require fresh installed-wheel
regressions, complete-request old/new/NJIT balanced time, independent heap/RSS,
native/Eigen-OFF/sanitizer and source/build identity. Do not claim all remaining
operators meet performance goals. Historical Microsoft Excel recalculation can
validate unchanged fixtures but cannot be called a new live Excel run.
