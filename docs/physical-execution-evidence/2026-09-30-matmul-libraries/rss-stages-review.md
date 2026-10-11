# MatMul staged RSS attribution review

2026-09-30. Read-only source audit of `rss_stages_matmul.py` and its reused
`tools/check_matmul_performance.py`, `tools/check_large_matrix_performance.py`
and `tools/benchmark_memory.py`. The parent was running the script. This reviewer
did not execute numerical work, build, benchmark, import the adapter, alter the
script or inspect incomplete results as if they were final.

## Conclusion

**No blocking sampling/lifetime/fairness defect was identified.** The script can
support paired, phase-specific RSS/VM/allocator attribution for its three frozen
cases and two frozen builds. Its own `attribution_only`, sampled-lower-bound and
`memory_acceptance=not_evaluated` labels are appropriate.

It cannot alone establish full memory nonincrease, an exact transient peak,
kernel-only causation, or a victory against NJIT. The following interpretation
limits matter; treating these as a complete acceptance gate would mislead.

## Correctly controlled aspects

- Fresh child processes, fixed case construction and complete build identities:
  old `4e12b6bb2276be316adc975be2e417d092b961b15d0d83514fed7030cde86d0a`,
  new `1aec1a6b0d4a07f0caee43b58cd4b339bdcc165f5e615f2190b15da0ba66e86b`.
  Imported package path and observed audit build are checked; failed children
  prevent a completed status. Input identities and complete outputs are checked
  after observation, including the ordered candidate's stronger finite-bit
  preservation claim.
- Eleven paired old/new runs alternate AB/BA (6/5 orders). Within each pair,
  both backends receive the same shape/layout/data and one-CPU environment. The
  two affected F_B cases and one unchanged C_B control are kept distinct.
- RSS sampling is in the parent. The child has no sampling Python thread or
  allocation interposer; its native observer is compiled against actual SDK
  structures. `ctypes` receives a defined 19-word wire structure, and task-info
  fields are read only when the kernel returned sufficient bytes. Missing VM
  fields become null, not fabricated zeros.
- Observation buffers/protocol messages and repeated native observer warmup
  precede the first baseline. Inputs, input identity, backend import, setup,
  cold result, releases, warm calls, two live results, close and explicit GC are
  separate stages. No hidden numerical warmup occurs before the request baseline.
- Ordinary calls use `MaterializedResult`, which retains the real raw result and
  projected public matrix inside each measured call. Direct calls return the
  independent ndarray. The warm-one and warm-two stages retain exactly one and
  two complete results; neither creates a numeric snapshot copy during sampling.
- The first result is released before five discarded warm calls. Both retained
  results and the callable are released before engine close; GC is an explicit
  later diagnostic stage. Input arrays, imported modules and interpreter state
  intentionally remain alive after engine close.
- Numerical parity checks, copied verification snapshots, sampled `fsum`, build
  audit and NPZ serialization all occur after `END`, beyond the sampled lifecycle.
  They cannot inflate the recorded stage/request peaks. The post-END identity
  call is a tiny mean, not an extra unreported measured MatMul.
- Lifetime VM high-water counters and summed malloc-zone high-water values are
  preserved as raw diagnostics, explicitly not subtracted into a fictitious
  request peak. Current-state deltas use current fields instead. VM categories
  are not incorrectly summed with allocator totals.

## Limits that must remain explicit

### 1. “Cold” is conditional on the measured preparation sequence

Before `backend_import`, `bench.inputs` allocates/touches matrices and
`bench.input_identity` hashes `array.tobytes(order="C")`. That identity step can
allocate and release an entire logical input copy, especially for F layouts.
The allocator may retain its pages for the later output. Imports/observer work
also precede the request baseline.

Consequently, `cold_result` means **first MatMul after these stated preparations
in a fresh child**. It is not an untouched allocator, an absolute process-cold
start, or proof of the first residency cost of every heap page. The old/new
preparation is equal and the script exposes `inputs`/`input_identity`/import
snapshots, so this is a scope boundary, not a reason to discard the comparison.
The earlier single-child harness also hashes inputs before its epoch; do not
describe the new diagnostic as eliminating that preconditioning.

### 2. Summarized phase differences are absolute state differences

`paired_summary.phase_state_differences` summarizes `new_snapshot-old_snapshot`
at each phase. It includes any difference already present at backend import.
It is **not** the incremental memory caused by that phase or request.

For request attribution, use the per-pair
`phase_differences[phase].native_change_from_backend_import`; for one transition,
use `native_change_from_previous_phase`. Show baseline, total peak and request
delta together. For example, a +32 KiB total difference with +32 KiB already at
import does not establish +32 KiB numerical scratch. Conversely, stable positive
post-baseline internal/external deltas must not be dismissed as noise simply
because malloc live bytes match.

### 3. Sampling gaps include checkpoint handoff gaps

The script reports sample counts, observer call duration and maximum gaps
**within each phase's sample sequence**. The parent acknowledges the previous
checkpoint before taking the next phase's first sample. A fast transition can
complete in this interval; the within-phase maximum does not include that
cross-phase handoff gap. Raw global elapsed timestamps permit deriving these
gaps offline if a maximum request-wide interval is needed.

Therefore there is no upper bound on a transient peak merely from the configured
0.5 ms interval or a phase's reported maximum gap. Retaining outputs through the
20 ms hold makes their resident state observable, but does not recover transient
packing, stack or wrapper peaks that have already disappeared. The script
correctly calls the observed peak a lower bound; keep that wording. Malloc
tracking, final active-stack inspection and the original uninterrupted workload
remain separate evidence.

### 4. Pauses and native snapshots change the observed workload

The native snapshot itself reads malloc-zone/task state in the child, then the
child waits for parent acknowledgment. Holds permit reclaim/allocator/OS changes
that need not occur in an uninterrupted call sequence. Cold/warm/one/two-result
phase endpoints are useful diagnostics, but their durations are not performance
timings and their peak is not interchangeable with the prior continuous sampler.
`request_observed_peak_rss` covers setup through warm-two, while later release/
close/GC states remain in the raw phase reports rather than that peak summary.

### 5. Residency categories suggest causes, not exact attribution

- `external` growth supports a file-backed residency explanation, which can
  include code, images or mapped data. It does not identify a specific function
  or prove all changed bytes are the new MatMul kernel.
- `internal` includes anonymous heap and stack; it is not a live-allocation
  counter. `malloc_size_in_use` includes full Python arenas, whereas malloc
  reserved capacity is not identical to resident pages.
- Released-result memory can stay reserved/resident without remaining live; a
  retained increment is neither automatically a leak nor automatically harmless
  noise. Compression and physical footprint must be considered alongside RSS.
- The C_B fallback control helps distinguish common build/import/runtime effects
  from F_B-associated changes, but cannot by itself prove causation. Shared
  machine file-cache/OS state is not reset by starting a fresh process; AB/BA
  pairing reduces order confounding without proving statistical equivalence.

## Supported final statements

- “For these three cases and frozen builds, paired observations showed this
  baseline difference, this sampled total peak difference and this request-delta
  difference; the change persisted/disappeared at these lifecycle phases.”
- “The increase correlated with external/internal/allocator capacity fields,
  with these per-pair ranges and positive-pair counts.”
- “The selected result paths were fully materialized and independently retained;
  release and engine close were observed separately.”

## Unsupported final statements

- “Zero malloc growth proves RSS/stack/total memory did not increase.”
- “No sampled increase proves the true transient peak is unchanged.”
- “The new build's entire RSS difference is numeric workspace/the new kernel.”
- “A three-case diagnostic establishes memory acceptance for every MatMul layout
  or all operators, or superiority to NJIT.”
- “Twenty-millisecond holds and first-call stage durations measure production
  latency.”
- “Stable positive RSS differences are noise without further evidence.”

Read the completed `status.json`, all 11 pairs and the matching manifest before
reporting results. Also inspect each snapshot's `mach_status`/`valid_mask` and
the summary's `available_pairs`: a task-info failure leaves VM fields null while
RSS/parity can still complete, so `completed=true` alone is not proof that every
VM attribution field was obtained. This audit does not itself assert that the ongoing run
completed or that any measured difference passed a gate.
