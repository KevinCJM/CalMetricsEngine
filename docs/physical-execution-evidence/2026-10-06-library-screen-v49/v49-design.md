# V49 isolated saved-mean candidate

Baseline: actual V48 source and installed wheel. No actual repository changes yet.

Finalize reconstructs private metadata only: a variance_axis(x,0,ddof) may read the saved short mean_axis(x,0) broadcast snapshot if its producer already executed and its proven sole arithmetic consumer has not executed. Match exact input node and literal axis 0. Never extend an arena lifetime or retain cross-request numerical caches.

Run the original variance canonical prepare first. Admit C-contiguous numeric rank-2 N>=32, 8<=D<=256 only; cached producer must have actually saved the current D values. Reuse the same Eigen ordered squared-deviation kernel with an optional known-mean view. Nonfinite means, unsupported geometry, isolate, inherited status, deferred-error and observed execution retain the existing path. No change to ddof, accumulation order, error position, root ownership, AST semantics or public operators.

Add an audit count solely to prove successful saved-mean reuse; carry it through child execution, threaded and process audit aggregation. No new full-sized allocation or copy.

Before adoption: source/build identity, regression over valid and excluded cases plus four lanes, balanced complete requests against V48 and warm CPU1 NJIT, and fresh peak-memory trials. If only a tiny change or no material large-workload benefit, retain the evidence and reject the candidate. Full normal, ASan/UBSan and disabled-library qualification is required before any actual adoption.

Research: Eigen official lazy evaluation and broadcasting documentation support expression fusion and reuse of costly repeated subexpressions; they do not establish local performance. No new external package required for this candidate.
