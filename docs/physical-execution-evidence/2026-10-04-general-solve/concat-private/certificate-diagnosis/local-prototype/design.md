# Existing solve_multi: local LU admission prototype (2026-10-04)

Current actual CalMetricsEngine source/wheel/worker remains fafa. The prototype is isolated and has not changed the runtime, registry, AST, Excel or public error contract. It is a native admission guard around existing mature DGETRF2/DLASWP/DTRSM/DGEMM/DGETRS, not a new solver or public operator. The ordinary current-native solve_multi is linked as a separate test oracle only.

## Why the prior guard rejected KKT

The old absolute triangular inverse recurrences and global Schur amplification are much too loose on the 1040/2064/4112-order KKT examples. Dropping the identity-permutation check or substituting DGECON's estimate did not repair them. Keep those negative measurements. Do not delete the 1e-14 original threshold to obtain speed.

## Local coefficient induction

Use the final library row-permutation coordinates and the exact mathematical shadow LU represented by stored float64 factors. At stage p its pivot row is U[p,:] and its active pivot column is L[:,p]*U[p,p]. A strict gap proves the canonical partial-pivot path chooses the same mathematical row; future row permutations can be treated as coordinate relabeling. Ties, uncertain pivots and unsafe intermediate bounds reject admission and require the original caller-owned replay.

For column tiles J (initial candidate width 512), define V[k,J]=max_j_in_J |U[k,j]| and T[k,J]=max_j_in_J,j>k |U[k,j]|. Compute Q[i,J]=sum_k |L[i,k]| V[k,J], taking the unit diagonal of L. Q bounds all active Schur entries in that row/tile. Build Q with positive DGEMM in 64-row panels, with roundoff inflation. Initial E[i,J]=gamma(8N)*Q[i,J] bounds the library's componentwise factor backward error, subject to the conventional double-precision BLAS error assumption described below.

With d=|U[p,p]| and e_p=E[p,tile(p)], use a downward-rounded h=d-e_p. Reject h<=1e-10 (a conservative admission margin, never a replacement for the original 1e-14 threshold). Require h strictly larger than every |L[i,p]|*d+E[i,tile(p)]. The multiplier difference is bounded by

    D_i = (E[i,tile(p)] + |L[i,p]|*e_p)/h + division_roundoff
    M_i = |L[i,p]| + D_i

Each remaining row/tile is advanced by the triangle inequality for the original ordered multiply/subtract, using D_i*T[p,J], M_i*E[p,J] and product/subtraction roundoff bounded by Q[i,J]. Positive expressions are inflated and underflow gets a conservative 1e-270 allowance. The helper admits only finite safe bounds; it does not report new numerical exceptions. No inverse-norm estimate or full N² radius matrix is used.

Small-matrix diagnostics separately reconstruct the shadow Schur complement at every step and compare actual ordered updates against the predicted radius plus reconstruction roundoff. They also check the actual selected pivot. This is evidence for the implementation, not a universal proof from passing fixtures.

## RHS and back substitution

Assume finite RHS with max absolute entry <=1e6 (validate B before A and replay original errors on failed admission). Start each RHS-row magnitude bound at 1e6, update with the same multiplier upper bound and rounded subtraction allowance, then bound reverse substitution using |U[i,j]|+E[i,tile(j)] and the lower diagonal h_i. Bound all partial products/sums/quotients far below overflow. An accepted coefficient guard alone is insufficient; the RHS and substitution induction must also pass. Zero RHS is mathematically safe in the prototype but the integration must preserve existing zero/signed-zero diagnostics rather than automatically widening this case.

The explicit guard arrays require 4*N*ceil(N/512)+67*N doubles, about 3.39 MB for N=4112. This is a scratch formula, not complete-request heap/RSS evidence. The original producer/ownership proof must still avoid a second full coefficient matrix.

## Evidence and scope

- Coefficient-only prototype: nine shape/width screens passed. Fine tiles cost substantially more; this is diagnostic, not an adoption decision from micro timings.
- Full coefficient/RHS/finite prototype: 1040/2064/4112 KKT passed; current-native oracle status OK, maximum solution difference 3.42e-14, normalized residual <=2.40e-17. At4112 the one-shot factor/guard times were0.227/0.230s. These timers exclude graph/input/output and are not a full-chain or stable NJIT advantage claim.
- Checked prototype normal:145 cases,73 admitted/72 rejected. Covers dense/permuted/DD/SPD, original threshold below/equal/above, NaN/Inf, zero/max/invalid RHS, Wilkinson ties, growth, small KKT and four real near-threshold counterexamples. The four historical examples all reject admission, including the one where canonical succeeds. No accepted case had a canonical error or a small-matrix radius/pivot violation. Largest observed difference/radius ratio0.0399.
- Same source actual ASan/UBSan plus sanitized current-native oracle:145 cases, same inputs and73/72 outcomes; leak detection disabled, no LSan claim.
- Direct mature DGETRF alternative:145 cases and six large KKT/width screens passed, but one-shot factor timing did not establish a material speed advantage over existing bounded-panel orchestration. Keep the established bounded-panel entry pending full-task comparison.
- The first screen terminated because all-zero RHS produced0/0 in the diagnostic residual normalization. Corrected the reporting formula to0 for a zero numerator/denominator. Certificate mathematics unchanged; preserve the original failed record.

## Remaining mathematical review

The initial gamma(8N)*|L||U| envelope must be grounded for the actual bounded-panel/vendor implementation and floating environment, not inferred from library INFO alone. The positive DGEMM upper bound likewise assumes a conventional double-precision product/sum error model; fast BLAS alternatives may require a different bound. The current source uses a related conventional bound already, but that is not independent verification of this extension. [LAPACK Working Note22](https://www.netlib.org/lapack/lawnspdf/lawn22.pdf) discusses loss of componentwise stability under fast BLAS. The present prototype must not be described as a universally proved vendor-independent certificate. Review before production adoption; add independent factor-residual checks if needed and measure their full cost.

A small backward residual is not a universal forward-error bound for an ill-conditioned system. The current user precision policy allows floating tails beyond four decimals and prioritizes correct mathematical logic/invariants. Terminal float64 roots must retain residual/conditioning qualification, explicit tolerance tests and all original errors. DGECON estimates must not be advertised as rigorous bounds. Preserve exact downstream discrete behavior by keeping the new entry limited to independent terminal floating roots. The former DD path and its stricter residual/Varah admission should remain unchanged.

## Actual integration plan

1. Extend only existing native solve_multi private-coefficient dispatch. A concat tree must already satisfy fafa's input-backed/private/constant-axis proof and existing exact arena/RHS/output-overlap checks. C/F public result behavior stays unchanged.
2. Reuse the same concat leaf-copy routine to regenerate bounded coefficient row spans for residuals and the full C coefficient on replay. Use a synchronous native function-pointer+context descriptor, no Python callback and no retained pointer after the call. Clip rank2 rows with checked geometry; normal rank1/3 concat behavior remains unchanged. Do not preserve a second N² matrix.
3. In solve_accelerate, keep the existing DD certificate and forward bound. Add the local guard only for the explicitly admitted non-DD/private concat route. Keep all B-before-A validation, rounding-mode, finite/scaled input, CPU-budget and result-publication rules. Prove the new result's original-equation residual with caller-regenerated rows; do not publish output before validation/replay decisions finish.
4. Planner/Scratch/library estimates must cover the guard's actual bounded arrays for every worker. Use current exclusive Workspace and stable input owners; no hidden pool, cross-request factor cache or new opcode. Audit actual calls/fallbacks/scratch/copied bytes using current fields.
5. Tests: raw native first-error/buffer cases; clipping/empties/C/F/negative/zero strides; all old DD acceptance/replay cases; original four counterexamples; non-DD KKT, resource/FE modes and unavailable libraries; readonly inputs and two independent output generations; single/thread/process-inline/shared; prepared/isolate/observer exclusions; actual normal, sanitized and OFF packages.
6. Benchmark old fafa/new/warmed NJIT on the actual ordinary graph, identical inputs/CPU/output retention. Include preparation, copy, dispatch and all output materialization. Repeat C/F full KKT, major direct/private solve controls and negative/replay cases, continuous requested heap and three fresh RSS processes. No complete-goal claim until actual adoption and all important medium/large workloads meet the overall objective.
