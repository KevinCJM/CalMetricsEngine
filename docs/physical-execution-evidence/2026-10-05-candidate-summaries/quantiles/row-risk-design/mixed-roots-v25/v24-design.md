# V24 mixed matrix/vector row outputs

Status: developed candidate; not built, tested, adopted or performance-qualified.

The existing derived-row proof admits a single static N x K input and row-local
transformations, one private MatMul trajectory, and reductions/path operations
that preserve the candidate axis. Its rank-one-only root check forces a complete
N x T arena when callers also request normalized weights N x K. Retain that
required matrix and all risk/count vectors while bounding private trajectories.

## Dependency and mathematical proof

Extend only root geometry to rank one/two with candidate axis zero and float64.
The existing full-versus-blocked shape proof remains mandatory. No new opcode,
formula, random source, approximation, public interface or numerical algorithm.
Every child executes the same canonical logical nodes and ascending recurrences.
Cross-row reductions, multiple varying inputs, scopes, isolation and observed or
prebound execution retain their supported canonical route. A public trajectory
still disables the private producer optimization. RNG, constraint repair,
multi-round exploration and frontier selection are outside this fixture's claim.

## Layout, allocations and ownership

C roots write their independent final row fragments directly. If any matrix root
requires F order, use one bounded C block for all roots, then scatter into each
full C/F final destination. Full F leading dimension is N, never block length or
worker range length. Empty N x 0 roots do not advance pointers or dereference.
This is an explicit bounded output copy, not a zero-copy claim. The local block
is reused within one interval and released before the next. Count its capacity
in operator workspace audit. Planner reserves each root's maximum block extent
min(64 * root_width, max(163840, root_width)), plus shape/layout descriptors and
existing child scratch, including wider batches possible in shorter windows.
Returned shape, order, status, dtype, root identity and independent ownership are
unchanged. Process encoding and shared outputs retain their current protocol.

## Scheduler and failures

Use the same native row-batch executor for single/interval/process execution and
Scheduler-owned array-row ranges. Workers use full immutable input geometry and
write disjoint rows of the same final buffers, including F column fragments.
No new pool, BLAS threading change, Python runtime math or hidden parallelism.
Preserve earliest logical-node numerical failure across all blocks and workers;
resource/timeouts drain according to existing runtime. Only one worker publishes
full shapes/statuses. Normal execution clears its own interval once; parallel
ranges never clear another range's destination.

## Validation and complete-request measurement

Test C/F/negative/padded inputs, C/F/auto roots, tail blocks, K=0/1/7, CPU=1/4,
different root widths/positions, exact discrete counts, independent retained
generations, unchanged input, current prepared content, multiple windows,
single/thread/inline-process/shared-process, memory admission and error priority.
Native canaries include F full-stride ranges and empty extreme-stride storage.
Run full/binding/native/sanitized/library-OFF quality before measurements.

Compare complete owned input preparation, log/normalization, shared trajectory,
three risks, required normalized weight matrix and result return with warmed
prange NJIT blocks 64 and 512 under equal CPU and BLAS budgets, C/F outputs and
two retained result generations. Keep a full materializing reference and bounded
reference; report all outputs, actual plans, copies, repeated paired time,
malloc requested peak and fresh process RSS increments. A lower C++ arena alone
is not proof of lower total peak. No heavy build or benchmark overlaps an
existing time/memory campaign. Excel logical formulas and budgets are unchanged;
physical-only change does not require ordinary calculation to run Excel.
