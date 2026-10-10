# V62 B: existing mature DGEMM for terminal scalar arithmetic

Before implementation, A whole1024^3 matrix/scalar chain all-output parity passed
but CPU1C/F0.104/0.099s vs warmed NJIT0.011/0.011s. Storage alone is not adopted.

Reuse the existing native root DGEMM contract, not a new math kernel. Its input
domain is finite normal zeros/[2^-128,2^128], dense C/F positive pitches, and
its existing gamma(2K) absolute dot certificate. Reuse mature Eigen input bounds
in bounded64Ki-cell slices; no manual numeric matrix implementation or new
library. Existing root preflight repeats its checks, all cost is measured.

For <=8 surrounding scalar steps, admit add/subtract with zero or absolute
scalar[2^-128,16], multiply with absolute scalar[2^-24,1]. Division/other values
keep A's exact existing math. Inputs/scalars must be numeric rank0/finite and
RN. This includes general scales, not just dyadic constants. Read later leaf
values only for nonthrowing admission, never suppress or reorder their actual
logical preparation. Wrong data declines to original node/error order.

Compute M=outward(K*maxabsA*maxabsB), E0=existing outward2*gamma(2K)*M and
require E0<=0.000049, so the old library cannot enter a looser partitioned
budget. Propagate each scalar's error amplification and BOTH arithmetic
roundings outward, carrying a magnitude bound. Require finalE<=0.000049.
Multiplication: E'=|s|E+2u*(|s|*(M+E)); magnitude=(1+u)|s|M.
Addition/subtraction: E'=E+2u*(M+|s|+E); magnitude=(1+u)(M+|s|).
All positive operations outward rounded, u=2^-53. Per-step minima keep binary
dyadic grids above underflow (including FMA, initial product grid>=2^-360;
<=8 multiply grids each extend at most76 bits, so >=2^-968). Existing input/exact zero paths and
final magnitude/error admission prevent overflow and nonfinite-class changes.

After the last actual arithmetic node, only |root|<=finalE (or nonfinite) is
ambiguous for zero/sign. Replay that coordinate through existing dot_value and
the existing canonical scalar ops in original operand order, no hand-written
second operator body. This fixes final cancellation, including signed zero;
base exact zero needs no replay. No intermediate comparisons/state/consumer
exists. Keep deadline/cancellation checks on input blocks and final replay.

The existing root helper enforces/restores CPU1BLAS and root layout. Model no
new IPC fields/threads. Graph planner must include existing vendor reserve for
the producer even though its arena is elided. Actual numerical scratch remains
lazy. Candidate A frozen text/source identity/failure retained. Requalify full
arrays/classes/errors/owners, then complete time/heap/RSS before adoption.
