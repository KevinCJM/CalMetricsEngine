# Integration decision (before production edits)

Screen: 334 unchanged+expanded fixtures, all nine candidates passed original numerical/discrete checks. 51 cases, 11 paired rounds, single core, immutable 1aec control; preallocated native-only screening is not complete-request acceptance.

Admit rows4x8 (default Eigen settings) only for C-compatible B, stride=(pitch,1), pitch>=N, auto ISA, M/K/N>=32, macOS15+arm64/Eigen. This extends the existing library dispatch after measured C-right gains (1024 CC/FC 27%/28% time reductions). Keep existing F-right4x4: CF long-K wide variant regressed 18%. No isolated unrolling macro/ODR risk. Explicit ISA remains unchanged.

For eligible C/F B and A row stride0, all input rows depend on precisely the same immutable coordinates. Evaluate one original ascending-k reduction per output column, then copy the completed first row to the remaining independent output rows. No numeric cache crosses calls; changing bound values must refresh prepared results. Record actual (M-1)*N*8 algorithm-copy bytes, not zero-copy. Output shape/product has already been validated. Scalar arithmetic preserves NaN/Inf/zero and error propagation. Three void entry points permit tail dispatch without charging unrelated function frames.

Logical operator, AST/DAG, planner output size and formula expansion remain unchanged. Excel retains full mathematical expansion and budget accounting; optimization does not reduce logical workbook budget. No private pool/global BLAS state, input packing, heap scratch or changed tolerance. Native object frame/calls must be inspected before memory claims; memcpy implementation stack and OS residency remain distinct uncertainty.

Validation: expand numerical/tail/layout/discrete/ownership/audit/worker tests; Python full regression, native+ASan/UBSan+EigenOFF;185 operator replay and previous real Excel snapshot parity (not fresh recalculation). Full original90 timing and31 memory cases retained, add affected shapes if missing; old1aec/new/NJIT same core, complete result retained, cold/warm epochs and requested heap/RSS reported separately. No claim of full NJIT closure from screened kernels.

## Evidence-driven admission correction

Complete 114-case candidate timing (529294f4) found padded-C 512³ ordinary 12.473625→16.836750 ms. Reject broad C pitch>=N for the wider tile. Require B exact C strides=(N,1), while repeat-row dependency reuse still accepts padded C/F. Preserve all candidate timing/partial memory, stop the now-rejected candidate memory before rebuilding; final verification repeats all114 timing and51 memory cases, including padded-C as an unchanged-path control. No cases or original thresholds removed.

## Whole-column-tile correction

Second full114 timing (16d2) showed ordered C4x8 partial-column tails regress:127³CC paired median1.1327 (11/11 slower),33x33x39 CC/FC1.1410/1.1289 (11/11),512x512x513CC1.0234 (9/11). Narrow regularC to exact contiguous B and N divisible8, the natural full column-tile boundary. Repeated-row and F4x4 admissions unchanged. Preserve all prior failures and repeat original full scopes; C-path functional fixtures explicitly reach N40 while remaining odd-column fixtures exercise fallback. No parameter sweep or threshold weakening.

## Installation-path memory attribution

16d2 full51 campaign showed +8 B cold/warm live peaks in45 Scheduler/prepared cases and +72 B cold cumulative allocation; direct cases0. Preserve those results. Six fresh heap children copied byte-identical1aec/16d2 packages to equal-length real paths: ordinary/prepared/direct differences all0 for setup/cold/warm live/peak/allocated/count. This supports installation-path overhead, not new numeric scratch; allocation call-site trace is not claimed. Final114/51 campaign uses separate equal-length real final-install-old/new, with original binaries hashed and no input/output/timing/memory scope reduction. RSS remains independently measured, not inferred from this attribution.

## Rectangular stability correction

The complete d2e campaign (114 timings and51 memory cases/306 children) shows sustained C4x8 regressions:32768x64x32 FC ordinary/prepared paired2.4911/2.4540 and64x64x32768 CC ordinary1.5902,11/11 rounds each, with matching CPU increases. The unchanged F control also varies across campaigns. This is not sufficient to dismiss regression as noise or identify its cause. Preserve all d2e samples/source/identity in candidate-d2e. Conservatively admit regular C4x8 only for square M=K=N and N divisible8, with exact C B as before. Rectangles retain existing SIMD, including previously faster rectangular cases; do not cherry-pick individual aspect ratios. Repeated-row exact dependency optimization still applies first to square/rectangular C/F-compatible B. This is a measured-scope boundary, not a universal theorem that squares are faster. All114 timing and51 memory cases remain in final acceptance, and native/Python tests add true square C/FC positives while keeping rectangular controls. The source change justifies a fresh full final run; retain the prior strict phase2 failure rather than overwrite its evidence.
