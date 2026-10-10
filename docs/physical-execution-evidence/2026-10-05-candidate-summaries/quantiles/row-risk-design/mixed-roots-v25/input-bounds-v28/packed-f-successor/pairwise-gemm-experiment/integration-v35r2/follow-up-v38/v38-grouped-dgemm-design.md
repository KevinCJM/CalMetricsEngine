# V38: deterministic grouped native-library accumulation

Scope: isolated candidates only. Live V35r2 numerical source, public operators,
AST/DAG, error policy, output layout, Excel formulas and ownership stay unchanged.
Apply D1/D5/D6/D8 and X1/X9/X10. Original 152 controls, input hashes and budget
0.000049 remain unchanged. Do not use probabilistic bounds or low precision.

## Motivation and library facts

V37 measured significant extra explicit-merge traffic. Unconditional tile geometry
and output-only geometry caused material CC regressions and were rejected. This
experiment retains the original geometry, original chunk admission and scratch
cap. It changes a group of four/eight DGEMM calls from beta=0 pieces plus explicit
adds to a beta=0 first piece followed by beta=1 calls in the same private buffer.
Only completed groups are explicitly merged by the existing mature Eigen maps.

[Reference LAPACK DGEMM](https://raw.githubusercontent.com/Reference-LAPACK/lapack/master/BLAS/SRC/dgemm.f)
specifies C=alpha*op(A)*op(B)+beta*C; beta=0 does not consume old C. Accelerate
provides the already qualified FP64 backend; alpha/beta are exactly zero or one.
[Higham/Mary Lemma 2.2](https://eprints.maths.manchester.ac.uk/2731/1/paper.pdf)
provides the deterministic gamma_n bound; the paper's probabilistic bounds are
not admission rules. The following application and derivation are ours, not a
vendor-certified bound or a published benchmark of this implementation.

## Deterministic certificate

Let u=2^-53, gamma_n=n*u/(1-n*u), and S be the existing outward-rounded bound
on sum_k |a_ik*b_kj| for every output cell. b is the original admitted chunk.
Under the same classical binary64 multiply/add model as the existing strategy,
a DGEMM piece with a known computed addend C_hat has local absolute error at most
gamma_(b+2)*(piece_absolute_product_sum+|C_hat|). A product has one multiply plus
at most b+1 addition/scaling rounding stages; exact alpha/beta=1 do not increase
the bound. This does not assume FMA, pairwise vendor reduction, or random errors.
Already accumulated error is added separately: C_hat is a known floating-point
input, not the true prior sum, so no previous error term is lost or double counted.

Before every beta=1 call, measure max_abs(C_hat) with Eigen over only the initialized
tile (honor its C/F leading stride). Sum these maxima outward into T for the tile.
Then all group errors combined are bounded by gamma_(b+2)*(S+T). For g completed
groups, d=g-1 for sequential merges, or ceil(log2(g)) for the existing balanced
carry tree including final active-level merges. The final error bound is

E = gamma_d*S + (1+gamma_d)*gamma_(b+2)*(S+T).

Every positive operation is rounded outward; all quantities must stay finite.
The input exponent guard bounds exact products/partial sums far inside binary64
normal/overflow limits; cancellation yields zero or an integer multiple of at
least 2^-360. This uses the existing classical-library model, not a proof of every
opaque vendor instruction. Preserve that evidence limitation explicitly.

If the bound exceeds the unchanged budget at any prefix, discard the tentative
private tile and recompute it with group width one, using the original admitted
beta=0/explicit-merge plan. The same tile implementation handles both choices;
there is no second custom multiplication kernel. Attempted library calls and
actual copied/merged/scanned elements remain in audit counters. No partial tile
is published, and failure/cancellation never restores an old result buffer.

For successful groups use max(original_plan_error,E) as the tile error. This
retains original replay triggers. Widen near-zero ambiguity to the maximum of
the original ambiguity and gamma_(2K)*S+tile_error. Replay candidate near-zero
cells through the original ordered native dot and certify
tile_error+|canonical-candidate|. Over-budget/nonfinite replay still rejects the
whole private fast root before publication, exactly as existing callers require.

## Allocation, dependencies and ownership

Use the original allocated scratch cap and original tree slots. Four/eight-piece
grouping reduces required tree depth; it never needs more slots than the original
plan. Sequential first group writes the independent root, subsequent groups use
the exclusive one-tile buffer; tree groups use exclusive initialized slots and
copy the completed tile into the requested final C/F layout. No input pack,
input mutation, cross-request pointer retention, shared caller workspace, Python
callback, new thread pool or process is added. Calling-thread BLAS control and
exception restoration remain in the existing private C ABI for prescreening.

Do not claim peak memory savings from fewer merges. This first screen deliberately
keeps allocation unchanged to isolate the time hypothesis. Any later root/graph
integration needs actual wheel, Planner/worker/lifetime tests and fresh peaks.

## Qualification and decision

Compile AOT C++17/O3 with no fast-math or contraction changes. Original 152 controls
must retain 88 admission/64 rejection and all 304 input hashes. Repeat 24 changed
geometry cases with whole-matrix exact integer controls and Decimal80 points.
Add forced successful grouping, numerical bound fallback, odd-tail tree, padded
layouts, canaries, callbacks/cancellation, rounding/BLAS restoration and insufficient
scratch checks; normal and ASan/UBSan. Test returned bounds independently.

Only then run six rotating-first paired complete owned-input/two-independent-output
requests against actual V35r2 and warmed explicitly CPU1 NJIT/Accelerate, across
large CC/FF and both result layouts. Private C ABI is screening evidence only.
Do not adopt a slower candidate or silently increase budgets. Full actual-engine
qualification and stable time/memory gains precede any live-source application.
