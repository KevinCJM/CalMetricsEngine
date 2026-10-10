# Supported complete latent-chain benchmark

This is a controlled benchmark adapter, not engine/runtime or platform-service
integration. It imports only the installed AOT package and the engine's frozen
`tests.latent_reference`; no platform source/runtime dependency is introduced.
The parent agent alone runs numeric, timing and memory tasks serially.

## Scope and dependency identity

- Complete finite raw matrix, fixed realtime train/classify positions, random
  Park–Miller initialization, `state_count == components <= 12`.
- The full original matrix is standardized with training-only mean and scale;
  classification consumes only the OOS rows. First raw feature remains score.
- GMM or GMM-pretrained HMM fits are shared across all parameter projections,
  probabilities, states, confidence and audit outputs by native structural CSE.
- HMM pretrain maximum is `min(max_iterations,30)`. Its initial distribution,
  transition normalization, forward training state and OOS first transition are
  separate visible mathematical expressions. No Python numerical callbacks.
- Every call refits the current bound data. No numerical cache or immutable
  content identity is inferred from the addresses. The input fixture is readonly.

## Layout and ownership

Raw uses time/factor axes. One complete N-row int64 position sequence shares the
same observation axis. Training/classification selectors and the K-row identity
order are sliced inside the DAG, avoiding falsely labelling shorter external
lookups as N-row time inputs. The graph's `take_axis` preserves the selected
matrix's time axis, and `partition_mean` of one-row partitions creates
component/factor starting means. Execution receives only graph-consumed inputs
and the actual [0,N) observation interval.

Both warmed benchmark paths return independent results. Prepared native mode
uses `run_snapshot`, never borrowed `run`. Existing baseline EM scratch and native
scratch are included in the execution whose peak memory the parent measures.
Input construction and AST compilation/binding preparation are separately
reported. All service outputs and audit fields are retained on both sides.

## Equivalence and discrete behavior

Standardization is two-pass ascending observation order, ddof 0, scale fallback
1 below 1e-10. GMM starting variance is at least 1e-4 and uniform starting weights.
Fitted contracts preserve the frozen EM/HMM stop tolerances and maximum iteration
counts; explicit completed counts are compared exactly. The frozen service does
not expose whether a fit converged on the final permitted iteration, so the
adapter does not invent that boolean from `completed < maximum`.

Component ordering uses finite first-column means. Current Numba's cutoff 15
makes K<=12 component sort insertion-stable, matching the native stable argsort.
Classification decides the first original-component winner before reordering:
one_hot(original argmax) -> reverse ordered component columns -> row scatter ->
state index -> explicit scalar `state_select` decisions. This preserves original
posterior ties instead of selecting the first reordered-state tie. Probability
rows and confidence remain separately derived. Training state is -1, confidence
NaN and probability zero, controlled by an independent explicit validity mask.
The one-column score is reduced with max rather than sum to preserve negative 0.

Float outputs use the existing latent parity tolerance (rtol/atol 1e-9);
dtype/shape, state, order and iteration counts are exact. Probability sums and
unclassified training rows are independent invariants. Timing alternates first
backend each round, warms NJIT outside measurement, uses a single CPU budget and
returns the same outputs. A micro result does not establish whole-platform gains.

## Remaining functional and evidence gaps

Dynamic finite-row filtering, availability/knowledge-time split, retrospective
mode, default quantile-init long-array ties, Markov and state_count>components
remain outside this adapter. These are not declared generally unsupported by the
engine; they require separate closure/equivalence work. No target Excel recalculation
or new backend behavior is asserted: this adapter uses unchanged canonical operators
and does not execute export in the production numerical path. Actual full-workload
latency and heap/RSS evidence is still required, and belongs to the parent run.

## Memory driver

`memory.py` delegates unchanged accounting to `tools/benchmark_memory.py` and
launches each backend/metric in a fresh sequential child. Both backends warm the
same tiny readonly NJIT signature before the epoch; compilation is not charged
as native setup. The heap setup epoch includes graph/callable construction,
first result, five warmups and at least two retained-result replacements. An
optional execution epoch instead excludes graph construction and reports that
scope explicitly; those two peaks cannot be added. RSS uses a separate fresh
child, a warmed sampler, first full-size execution and retained replacement.

Common input bytes and runtime imports are outside the incremental observations.
Heap interposition does not prove coverage of Python arena suballocations, mmap
or APIs bypassing intercepted malloc. RSS is supplementary page-level sampling,
with an un-subtracted idle control; its baseline contains the already-created
graph/owner. All thirteen outputs and output ownership match the latency chain.
The actual native binary and all source hashes are verified against a selectable
build manifest, plus benchmark/helper/probe SHA and post-run source stability.
