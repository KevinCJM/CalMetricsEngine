# Native state, event and segment contracts

Contract version: `state-events/1`. These native operators add reusable mathematical
state and event capabilities. They do not create, publish or migrate a research
platform model, and do not certify point-in-time trading or backtest eligibility.
The engine remains independently installable; the platform is a reference for
specified semantics, not a runtime dependency.

## Types and shared work

All temporal inputs are rank-one arrays on the same observation axis. Prices and
features are exact `float64`; state/event codes are exact `int64`; validity uses
`bool`/validated `uint8` masks. Native scalar configuration parameters are finite
numeric scalars; integer parameters must be integral. Parameters in this document
are all required; defaults belong to explicitly defined upstream templates.

- States: `-1` unclassified; valid category codes are nonnegative. Hysteresis and
  drawdown use the mappings below, which are distinct algorithms.
- Phase directions have separate `phase` semantics: up `0`, down `1`, unknown `-1`.
  They are not ordinary `state` codes; neutral/recovery labels cannot substitute for direction.
- Events: `+1` peak, `-1` trough, `0` no event, `-2` unavailable observation.
  The platform's floating event `NaN` maps explicitly to `-2`; it must never be
  confused with the valid trough code. Other event codes fail closed.
- Segments: paired int64 start/end boundaries. `(-1,-1)` means no complete segment.
  Complete membership is `[left,right)` with `0 <= left < right < T`; the right
  endpoint is an observed price, but belongs to the next phase as a membership row.
- Continuous-state result: one shared `T×3` int64 storage block, columns
  `state`, `evidence`, `pending_count`.
- Segment result: one shared `T×2` int64 block, columns `start`, `end`.

The physical bundles use existing integer matrix storage. Typed IR uses explicit
`continuous_state_bundle` / `event_segments` tags and named field axes so a
numerical matrix cannot impersonate a state or boundary object. Graph event
outputs have `event` semantics and state outputs have `state` semantics. Direct
operator calls validate concrete dtype, shape, values and boundary geometry;
nominal source identity is additionally enforced by the graph compiler.

Bundle projections borrow strided columns and retain the owner. Explicit `out=`
materializes only the requested column into caller-owned int64 output. Shared
bundles are computed once through ordinary DAG CSE, not independently per field.
The executor must extend the source lifetime through every borrowed projection.

## Operators

| ID | Operator and argument order | Result | Calculation class |
|---|---|---|---|
| 132 | `state_hysteresis(values, upper_enter, upper_exit, lower_enter, lower_exit)` | state T | Coupled recurrence |
| 133 | `state_confirm(codes, confirmation, min_hold)` | state T | Coupled recurrence |
| 134 | `state_continuous(candidate, initial, observed, confirmation, state_count)` | shared state T×3 | Coupled recurrence |
| 135 | `continuous_state_values(state)` | state T view | Projection |
| 136 | `continuous_state_evidence(state)` | category T view | Projection |
| 137 | `continuous_state_pending(state)` | count T view | Projection |
| 138 | `drawdown_cycle_state(price, drawdown, valid, stress, rebound, exit)` | state T | Coupled recurrence |
| 139 | `local_extrema(price, left, right, head, tail)` | event T | Coupled event selection |
| 140 | `ps_filter(price, events, min_phase, min_cycle, amplitude)` | event T | Coupled joint constraints |
| 141 | `between_events(events)` | shared segments T×2 | Boundary construction |
| 142 | `segment_starts(segments)` | index T view | Projection |
| 143 | `segment_ends(segments)` | index T view | Projection |
| 144 | `phase_direction(events, segments)` | phase T | Independent event interpretation |
| 145 | `drawdown_cycle_reference(phases, changes, segments, stress)` | state T | Coupled cross-segment recurrence |
| 146 | `state_select(condition, when_true, when_false, valid)` | state T | Independent condition selection |

### Explicit condition selection

`condition` and `valid` are same-axis bool/validated-uint8 masks. False validity
emits state `-1`; it never substitutes neutral. Each branch is either an aligned
int64 state sequence with codes `>= -1`, or a finite integral scalar configuration
in `[-1, 2^53-1]`. Int64 array identities above `2^53` remain exact. Float arrays
are not silently converted. Both branch contracts are validated before writing;
the selected branch supplies the result. Existing whole-graph error propagation
still applies to errors while computing either dependency.

The two masks explicitly separate condition truth from data availability. Existing
comparisons and `finite_mask` construct them; logical composition must combine
validity from every required operand. Numeric missing data is not interpreted as a
false condition. For example, three-state threshold candidates remain a visible
composition feeding a separate confirmation kernel:

```text
state_confirm(
  state_select(x >= upper, 0,
    state_select(x <= lower, 2, 1, finite_mask(x)), finite_mask(x)),
  confirmation, min_hold)
```

Dynamic thresholds additionally require their validity in the composed mask.
The selector adds no recurrence, confirmation, knowledge-time reset or implicit
financial classification beyond its explicit branch inputs.

### Hysteresis

Initial active state is neutral `1`; upper is `0`, lower is `2`. All four thresholds
are finite. Their order is not silently sorted or adjusted. From upper, an
observation `<= lower_enter` enters lower, otherwise `<= upper_exit` enters neutral.
From lower, `>= upper_enter` enters upper, otherwise `>= lower_exit` enters neutral.
From neutral, upper entry has precedence over lower entry. Equality is inclusive.
A nonfinite observation emits `-1` and retains internal active state. Initialization
restarts per execution interval. Feature construction stays outside this kernel.

### Confirmation and minimum hold

`confirmation` and `min_hold` are integers in `[1, 2^31-1]`. Confirmation tracks
one candidate, its uninterrupted count, the active category, and active duration
jointly. Splitting confirmation and holding into sequential ordinary nodes changes
switch dates and is not equivalent.

Before first confirmation, output is `-1`. Once active, duration increments before
processing each row, including missing rows. Code `-1` emits unknown, clears the
pending candidate/count, and retains active state/duration. Any code below `-1`
is invalid. Equal active observations clear pending confirmation. A different
candidate switches on the current row only when its count meets confirmation and
`active_duration > min_hold`; a switch resets duration to one. There is no
backdating. Exact int64 identities are preserved, including values above `2^53`.

### Continuous candidates

`confirmation` is `[1,252]`; `state_count` is `[2,12]`. Every candidate and initial
code must be in `[-1,state_count-1]`; initial arrays have the full aligned length,
but only `initial[0]` establishes state and it must be nonnegative. All observed
prices must be positive and finite. Invalid observations are errors, not absent
candidates; validation occurs before output writes.

Here candidate `-1` means **no new proposal**, not missing market data. It resets
pending confirmation and holds the state. Evidence is `0` initial estimate,
`1` same active candidate or confirmed switch, `2` continuation after any accepted
candidate, `3` pending different candidate. A candidate matching active state
counts as accepted immediately. A switch occurs on the confirmation row, resets
pending count to zero, and never changes earlier outputs. All three fields share
one solve. This differs intentionally from `state_confirm`'s unknown-row policy.

### Real-time drawdown recurrence

Codes: Normal `0`, Recovery `1`, Stress `2`, unknown `-1`. Parameters satisfy
`0 <= exit < stress < 1` and `0 < rebound < 1`. Callers provide the numeric drawdown
and a mask representing validity of the **whole chosen trailing price window**.
Rolling maximum, window validation and `price/rolling_high-1` are independent
compositions; they are not recomputed inside the state kernel. Valid finite
drawdowns must lie in `[-1,0]`.

The initial state is Normal. Invalid mask, missing/nonpositive price or nonfinite
drawdown emits unknown and resets state/trough. After reset, the dead band stays
unknown until drawdown reaches stress or exit. Normal enters Stress at
`drawdown <= -stress`. Stress tracks a strictly lower trough and enters Recovery
at `price/trough-1 >= rebound`. Recovery falls back to Stress on a strictly lower
price, or exits to Normal at `drawdown >= -exit`. Each row performs at most one
transition: a jump can enter Recovery today and Normal on a later row. Temporal
causality depends on all supplied inputs, including the validity/drawdown graph.

### Local extrema and joint PS selection

`left,right` are integers `[1,5000]`; `head,tail` are `[0,5000]`. Invalid/nonpositive
prices split valid spans and emit event `-2`. Search excludes the first
`max(left,head)` and last `max(right,tail)` observations of each valid span. Peaks
are strictly greater than preceding prices and greater-or-equal to following
prices; troughs use the opposite inequalities. This chooses the earliest member
of a plateau. A later, strictly stronger same-kind event removes the earlier
one; alternating candidates must move in their stated direction.

Because later candidates can remove earlier events, local-extrema selection is
retrospective, not merely a fixed `right`-row confirmation lag. No online guarantee
is inferred by renaming or shifting the result.

PS receives the candidate events separately. `min_phase` is `[1,10000]`,
`min_cycle` is `[2,20000]`, and amplitude is finite/nonnegative. Invalid price or
missing event splits the sample. It repeatedly applies the same ordered rules:

1. Enforce alternation; same-kind ties retain the earlier extreme.
2. Remove a first/last event censored by a more extreme sample endpoint region.
3. For a short full cycle, retain the stronger same-kind endpoint and remove its
   weaker counterpart plus the intervening opposite event; equality keeps earlier.
4. For a short phase, amplitude exempts it only when strictly **greater than** the
   threshold. Edge phases discard the edge event. Interior phases compare adjacent
   log-price excursions, with deterministic earlier/later tie handling.
5. Restart constraints after each deletion, including alternation where required.

Each changed pass removes at least one event, giving finite termination. A fixed
number of independent filtering passes is not equivalent. PS uses full-input
knowledge and can redraw earlier history when future data is appended.

### Segments, phase direction and retrospective drawdown

`between_events` emits only complete intervals bounded by opposing events. A
missing event resets the pending boundary; a repeated same-kind event replaces
it without inventing an interval. Unknown heads and tails remain unknown.
`phase_direction` interprets trough→peak as `0`, peak→trough as `1`, otherwise `-1`;
its Typed IR semantic is `phase`, not `state`. `drawdown_cycle_reference` requires
this declared phase type and rejects ordinary category/state codes even when the
integer values coincide. This is wave direction, not a bull/bear classification. Complete boundary
geometry is validated before index access.

Numeric interval statistics remain explicit `segment_apply` bodies. The scope
reads endpoint-inclusive prices `[left,right]`, computes a scalar once, and
broadcasts to half-open membership `[left,right)`. Thus endpoint return, amplitude,
return volatility and path efficiency reuse ordinary numerical operators. Grouping
on start indices alone omits the right endpoint and is not an equivalent substitute.

`drawdown_cycle_reference` carries whether the preceding completed decline had
`change <= -stress`, with `0 < stress < 1`. Such a decline labels `(left,right]`
Stress and its following rise labels `(left,right]` Recovery, retaining Stress
on the shared trough. Other completed observations are Normal. Shared peaks retain
the preceding wave's label. Missing/incoherent phase or nonfinite change clears
cross-wave pressure. To preserve the existing platform algorithm, positions with
no complete segment merely advance the scan and **do not clear** this pressure;
this differs from the real-time invalid-window reset. Malformed native boundary
geometry fails closed. Open head/tail positions remain unknown except the observed
terminal endpoint explicitly completed by the preceding wave.

## Execution, cost and acceptance boundary

All recurrences reset per graph interval and execute entirely in the AOT runtime.
Input views, including supported negative strides/read-only arrays, are read only.
PS uses one reusable T-element native index workspace; its event kinds are read
from the original event array. No input array is copied. Other state/event kernels
need only their output arrays and constant internal state. Bundle projections
borrow, while normal independent operator calls own distinct result storage.

State scans and boundary construction are O(T); extrema is O(T×(left+right)); PS
has a conservative O(T²) bound from bounded deletion plus scans/compaction. The
Planner must account for the shared bundle width and nonlinear event work; no
private thread pools or callbacks are introduced.

Known-at time follows every actual dependency. States from causal inputs may be
prefix-stable, but their upstream observations still carry their own availability.
Events, complete segments and PS/reference outputs are retrospective. Their
occurrence index is not a confirmation or knowledge time. The consuming platform
must retain full-input recognition/available-at data and enforce its own live,
publication and backtest gates. Lagging a retrospective result does not qualify it.

Tests cover exact codes, thresholds, gaps, first/last membership, shared output
projections, explicit copy-out, parameter/geometry rejection, independent ownership,
read-only strided input, bounded PS fixtures and platform-oracle parity. Native
source or direct operator tests alone do not establish platform integration,
production performance or point-in-time investment acceptance.

Under error isolation, a caught segment exception carries separate per-position
failure provenance. Comparisons, boolean masks and state selection cannot erase it
by turning NaN into False/0. Precise slice and scope membership propagation, conservative
nonlocal propagation and workspace ownership are specified in
[the stateful-series design](stateful-series-design.md#分段异常向下游传播).
