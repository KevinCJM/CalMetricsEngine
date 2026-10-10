# MC period/contribution shared product — static implementation review

Date: 2026-10-03. Scope: only `cpp/affine_projection.cpp` and its private header. No numerical execution, compile, test, benchmark or RSS probe was performed by this reviewer. Root owns all execution and adoption decisions.

## Baseline and implementation

The two previously frozen production files were copied before edits into `baseline-source/affine_projection.cpp` and `baseline-source/affine_projection.hpp`, preserving the implementation associated with the root's existing 2bbb build. These are outside the production checkout and are solely an audit baseline.

The previous producer computed one bounded affine/expm1 tile, then each period root performed an asset fold. Every contribution root separately scanned all tile rows and recomputed the same weighted product. For an all-assets, all-combinations consumer selection, this meant a repeated product and repeated tile traversal for every contribution. The implementation now shares only products whose graph identity proves equality.

Planning selects the first period consumer for each exact weights node ID as the sole carrier. Each contribution becomes either one sparse link `(asset, contribution_index)` in that carrier, or one entry in the unmatched list. Links are ordered by period, asset, contribution index; each period stores a half-open link range. Matching never compares floating values, backing addresses or input aliases. Contribution indices are assigned after the existing topological root sort.

At execution, the single `ordered_weighted_sum` body retains weight-first multiplication and the asset-ascending sum starting at positive zero. After each asset sum step, linked contribution accumulators receive that same computed product. Every contribution accumulator still receives exactly one product per logical row, in step-then-path order, across all tiles. There is no subtotal or cross-row reduction tree. Unmatched contributions retain the existing independent scan and scalar state. The final scalar division loop is unchanged.

## Boundary review

| Case | Static result |
| --- | --- |
| All period and contribution roots selected | One carrier per weights node; contribution scan removed for every matched root. |
| Period-only roots | No links; the same weighted mathematical body uses a no-op product consumer. |
| Contribution-only roots | No periods or carriers; every contribution is unmatched and keeps its original scan. |
| Partial contribution selection | Sparse links contain only selected roots; no implicit extra observable result is added. |
| Duplicate period roots, C/F or swapped | Only the first period is a carrier; other roots continue independent writes without adding a contribution again. |
| Duplicate contribution roots | Each independent accumulator has exactly one link; identical numerators with distinct denominators retain distinct division and output states. |
| Equal values or aliased inputs under distinct weights node IDs | No carrier match; independent scan remains. |
| Different weight nodes | Every contribution links only to its own exact weight identity or remains unmatched. |
| Empty asset axis | Period fold returns positive zero without forming a pointer into the absent tile; contributions still cannot pass their original asset geometry validation. |
| Empty outer axis with contributions | Existing `INSUFFICIENT_SAMPLE` check remains before final division. |
| Empty factor axis | Unchanged initial-only affine/expm1 producer; carrier logic is independent of factor count. |
| NaN/Inf/overflow and rounding modes | Same operand order and same stored double product; no reassociation, FMA, zero skipping, clipping or new payload failure rule. Incidental repeated floating exception flag counts are not exposed by the public result protocol. |
| Zero divisor / scalar failure | Divisions still follow original sorted divide node IDs, after numerator folds; no new eager check changes canonical error order. |
| Negative/padded input strides | Uses the existing `Value::f(asset)` for period weight operands; the link proof points to the identical underlying weight node and asset. |
| Multiple intervals and tile boundaries | Totals reset once per interval and remain live across its tiles; the link cursor resets per period row. |

The existing strict typed whole-graph admission, real last-axis geometry, observer/scope/isolate fallback, independent output ownership, layout writer, output-capacity checks, cancellation and deadline checks are unchanged.

## Memory and budget

No product matrix, second tile, private pool, new numeric buffer, worker, scheduler or borrowed result was added. The bounded one-MiB tile and existing three doubles per contribution remain unchanged. Added persistent metadata is O(selected contribution roots + selected period roots): sparse links, unmatched indices and link ranges.

`projection_metadata_bytes` counts the proof object and the actual capacities of its period, contribution, link and unmatched vectors with checked size arithmetic. Both the Planner workspace helper and runtime Audit include that same amount. Shared proof metadata is conservatively charged per worker; this can overcount shared storage rather than omit it. Transient graph-planning containers are released before execution.

## Required runtime evidence

The root should verify the frozen baseline and new source/wheel identities independently. Representative tests should cover duplicate roots, different divisors and output order, partial selections, different/aliased weight-node identities, empty axes, nonfinite values, four rounding modes, cancellation and crossing the 4096-row tile boundary. Full selected period and contribution results, schema, statuses, ownership and root error order must remain equivalent to the canonical graph.

Performance adoption needs root-owned paired service NJIT and stronger shared NJIT comparisons on the same realistic shape and CPU budget, plus fresh-child incremental heap and RSS evidence with independent previous/new outputs. This change removes an established repeated scan; no speedup or memory acceptance is claimed until those measurements complete. Neither NJIT reference materializes a full N×A log/simple-return tensor, so removing native DAG materialization must not be described as an N×A memory saving relative to NJIT.

Status: implementation frozen after static self-review; `mc_closure_design` independently reviewed the delta and reported no confirmed defect. Numerical, compiler and performance evidence remain pending root execution.
