# Wide EM fixed-size Eigen maps

Baseline production softmax V3 02aa13cb. Extend the already validated
weighted_feature_sums non-owning Eigen Map route to measured widths 32/64/128/256.
Per-feature observation reductions, multiplication parenthesization and all
EM steps/likelihood/floors/stopping are unchanged; no new operator or formula.
The independent output model fields remain accumulators after the E-step.
No numerical buffer, packing, extra heap, thread, global state or user-input copy.
Admission remains macOS15+arm64, Eigen3.4, requested non-scalar, n>=32,
input feature stride1. Other paths keep original scalar code.
Feature/tail/stride/readonly/NaN/error/oneULP stopping boundaries must remain
exact, including scalar vs optimized. Use complete public projections for
ordinary/prepared/direct old/new/NJIT fair single-CPU timing and memory.
Only select widths with meaningful whole-model improvements and no added
algorithmic memory; report absolute RSS/stack separately with all regressions.
No Excel formula/budget change; prior real Excel snapshot is historical only.

The first native ABI screening (not public acceptance) showed useful gains for all four new widths; 148 ordinary and 204 stopping-boundary records were bitwise unchanged. The scope is width dispatch and call-frame isolation; the existing mathematical helper is reused. Exponential candidates failed 58 stopping checks and density block candidates lacked meaningful whole-model benefit; neither enters production.

V1 direct inlining increased the common latent frame from 1088 to1200 bytes.
V2 isolates only new widths in a noinline bool-template driver; original widths
and all guards stay in the prior dispatcher. Object screening restores the
1088-byte common frame; wide mean/variance helpers have224/160-byte frames.
The existing density subcall already has224 bytes, so the inspected numerical
chain maximum remains1312 bytes. System library/exception/outer frames are
not fully bounded; no claim of a complete request stack bound.
