# Fixed-tape MC projection benchmark — static audit

Date: 2026-10-03. Files inspected: `bench.py`, `reference.py`, `memory.py`, the existing `tools/benchmark_memory.py` helpers and complete source/wheel identity helper. This is source review only; no measurement was run by this reviewer.

## Source mathematics and workload scope

CodeGraph located `backend/scenario_stress/numba_kernels.py::monte_carlo_projection_kernel`; its source SHA-256 was independently checked against the frozen literal: `bb53056653d697c698dab4ac5e716d96dbaec9071157d755c3fbef32e9831b79`.

For the declared healthy, already-transformed, all-selected fixture, `frozen_one` preserves initial/intercept first, ascending factors, coefficient-first multiplication, `expm1`, weight-first multiplication, ascending asset period sum, step-then-path contribution sum and final division by path count only. The existing finite and return<=-1 checks remain in the source-shaped service reference. The fixture intentionally excludes unavailable assets and the complete ordered first-fault prefix output protocol.

The platform service genuinely creates the transformed factor tape once and invokes projection again for each portfolio. `service_chain` preserves those repeated numerical projections and returns independent arrays without a packed-output copy, while excluding Python business integration overhead. `shared_chain` is a stronger additional optimized NJIT comparator: one asset-sized temporary, a shared affine/expm1 row, and the same ordered folds for every portfolio. It has no full N×A log/simple-return matrix. Its omission of the service's minimum tracking is disclosed by the narrower selected-output scope.

The source-shaped reference uses lazy readonly specialization and an int64 all-available fixture mask, whereas the production signature is fixed mutable C-array inputs and uint8 availability. This is a derived mathematical comparator rather than the exact production binary. That small signature/fixture difference is not a meaningful memory gate, but should remain clear in claims.

## Fairness and measurement ranges

- Both paths return every selected period matrix and every asset contribution. Native uses independent scalar roots for contributions; NJIT uses one contribution vector per portfolio. Total mathematical data bytes agree; wrapper metadata differs. F/C output layout differences must be reported rather than forcibly copied for equality.
- Prepared uses `run_snapshot`, and comparison requires independent output lifetime. A timed assignment retains the old result until the new result returns, on both paths. The paired process also retains the other backend's result, so it transiently holds three full mathematical result sets; it is not a two-result total-RSS experiment. Existing report notes disclose this.
- Default CPU=1 is the same-effective-core comparison; references are serial without `prange`. CPU>1 is correctly labeled exploratory. Tiny signature compilation and input preparation are excluded from warmed execution timing and reported separately.
- Heap children run an existing continuous epoch spanning graph/call construction, cold result, release, five warmups, retained-result replacements and release. Stage peaks overlap and cannot be added. Shared input creation and tiny NJIT compilation precede the epoch.
- RSS constructs owner/graph before baseline, then includes the first full-size execution and replacements. Page sampling and idle page control are reported separately without subtraction. Heap and RSS are complementary ranges, not interchangeable totals.
- Untracked frees cannot be interpreted as illegal frees. Interposed tracking can miss Python arena suballocations, mmap and other allocation paths; zero table overflow does not prove complete allocation coverage. Do not present the heap result as an exact universal memory total.
- `memory.py` now explicitly requires positive fusion and tile counters, zero numeric arena, direct output bytes equal to complete output bytes, successful roots and no Python/request compilation. This closes the earlier memory-driver physical-path proof gap. Runtime parity belongs to the separate comparison driver; memory observation only checks schema/status/path and identity.

## Remaining concrete evidence gaps

1. `bench.py::compare` asserts correctness, independence and no fallback/JIT, but does not assert that the new whole-projection physical path actually ran. A mathematically correct ordinary canonical executor can pass it. Require the same positive-fixture fusion/tile/zero-arena/direct-output checks as the updated memory driver when reporting fusion timing.
2. The time driver records native binary and driver hashes only at the end. It does not verify the complete root-generated source/wheel manifest before and after timing, unlike `memory.py`. The root's frozen install discipline can supply external identity proof; the JSON alone currently cannot establish source/build identity throughout its observation.
3. Memory `source_unchanged=True` verifies the engine manifest and binary before/after, but its own driver/reference/helper hashes are captured only after. This proves the engine source/build stayed unchanged, not that every measurement script stayed unchanged. Freeze scripts externally or capture those few existing hashes before and after as well before making the broader claim.

Minor report wording: `output_order=args.order` in an NJIT memory report denotes the requested native layout, while NJIT arrays actually remain C layout; report observed strides/flags or distinguish the two fields. The invariant comment's “mean cumulative period return” means the mean sum of simple period returns, not a compounded NAV return.

## Interpretation

No false claim of a full N×A NJIT materialization or a 1GiB NJIT saving was found in these drivers. Native zero arena establishes removal of its unfused intermediate slots. Both NJIT references already use bounded internal storage, so native-vs-NJIT memory conclusions require the actual complete-output lifetime measurements.

Healthy fixed-tape projection speed and allocation evidence cannot close primitive random generation, old-seed equivalence, covariance-root semantics, ordered first-fault diagnostics, NAV/drawdown, fan/tail statistics or HTTP/business integration. The existing scope and gap fields appropriately retain these limitations.
