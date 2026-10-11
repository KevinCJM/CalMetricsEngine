# V56 bounded posterior GMM design (not adopted yet)

Scope: existing diagonal Gaussian mixture EM physical storage and Eigen kernels only. No operator, initialization, statistical algorithm, convergence condition, public protocol, or Excel formula change. HMM behavior remains unchanged. R source is preserved until qualification and workload measurements finish.

## Measured selection

The ungated D prototype removed an uncounted component-tile-to-posterior-tile copy of 1,610,612,736 bytes in the 1,048,576 x 8, K=24, four-iteration request. Existing Eigen density rows now write direct output columns using an internal output stride. Public density output remains canonical C order and retains the previous finite-failure publication prefix.

Runtime selection is Apple ARM macOS >=15 with Eigen built in, K>=16, 8<=D<=32, N*K>=16,777,216 elements, and bounded replay storage <N*K. K=12 regressed 17-33% even with 192 MiB posterior and therefore retains its current path. D>32 has no new qualification, so remains current. Cutoff is conservative dispatch, not a claim of optimality at every adjacent shape.

Workspace in doubles: S=K*(128+2D+3)+128D. Exclusive storage: posterior tile, old means, carried per-feature sums, counts, old variance logs, old weight logs, optional column input tile. First pass computes old E, exact sample-order likelihood and effective counts and mean sums. Publish new means only after full E finite validation; copy all feature sums before feature-by-feature division/finite checks to preserve the canonical public failure prefix. Second pass replays the SAME old means/variances/weights, accumulating centered squared differences to NEW means in the original per-feature observation order. Do not use raw second moments or subtract mean squared. Variances are published only after all old-parameter replay completes. Floors, stopping strict inequality, iteration count and projection metadata stay unchanged.

No global fast math, float32, extra kernel threads, Python callbacks, value cache or cross-call ownership change. Scalar ISA still provides a native equivalent oracle. Library-off/other-platform builds retain canonical full posterior. Logical graph/Excel expansion unchanged; C++ requests never invoke Excel.

## Graph and admission

A replay fit cannot borrow an NK span using the existing full-posterior loan geometry. Skip the loan before unsigned small_count subtraction; preserve all previous loans on unselected shapes. Planner shapes are maxima: at a replay maximum, shorter rows may still allocate NK. For K>0, last original length is max((cutoff-1)//K,S//K). Compute its geometry through the same latent_scratch function, then take max(S, original geometry). This also covers the unusual S>=NK crossing without duplicate formulas. CPU/IPC use the same build and deterministic shape selection. The admission envelope may remain near 128 MiB; actual selected execution does not reserve or allocate that unused posterior.

The pre-existing Planner work-unit coefficients and iteration bound remain unchanged. They are heuristic routing units, not measured FLOPs: an extra replay does not justify doubling a calibrated latency estimate when the selected route is faster. Actual replay calls and input packing bytes are separately audited. Scheduler CPU tokens remain hard admission limits; complete thread/process transport is tested. No claim of full-engine allocation/copy or physical cost-model closure is made.

## Required proof

Qualify cutoff-1/equal/+1, K=15/16, D=7/8/32/33, C/F/signed/feature strides, short final tile, automatic/scalar exact results, old build error-prefix arrays, tolerance nextafter neighbors, readonly inputs, exclusive Workspace reuse, two owned generations, single/thread/inline/shared process transport, CPU budget, graph input binding copy and full algorithm copy audit. Preserve prior failed test attempts. Existing tests remain unchanged; do not weaken workspace-loan assertions to fit a new path.

Complete ordinary requests include identical readonly caller input copies, all six fit outputs, two retained generations and warmed same-core NJIT. Use six alternating-first fresh-process pairs, and fresh startup malloc/RSS epochs on the final gated build. Also inspect the actual thirteen-output standardize/train/initialize/fit/classify/rank/scatter/state chain; local fit wins do not establish the overall active platform goal. No adoption decision from isolated prototype data or compile time.
