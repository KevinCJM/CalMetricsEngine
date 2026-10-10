# V41: measured HMM hotspots and mature vector-library screening

Target remains the frozen current engine, complete ordinary Scheduler requests,
one CPU, owned immutable input snapshots and two independently retained readonly
output generations. The isolated copies below do not alter the current worktree.
This is an experiment design, not an acceptance or production-adoption claim.

## Measured prerequisite

The full current fit, including the GMM pretrain, is instrumented only at iteration
phase boundaries in `source-v41-profile`. All 8 training and 13 realtime-chain
outputs are byte-identical to independently frozen actual-wheel snapshots.
Timers start after fit allocation, finite validation and copying initial model
parameters; complete request costs must therefore still be measured separately.
CPU1, 500,000 input rows, 375,000 training rows, 4 features, 12 states, 5 pretrain
and 5 HMM iterations remain unchanged. The HMM phase is about 5.10 seconds:
XI ~2.35 seconds, forward ~1.15, backward ~1.19, density ~0.078, gamma ~0.190,
mean/parameter updates ~0.093 and variance ~0.054. XI is about 46%, not 80%.
Instrumentation timings are diagnostic, not performance acceptance.

## Primary sources and boundaries

- Apple documents native double-array `vvexp` and `vvlog1p` APIs:
  https://developer.apple.com/documentation/accelerate/1470381-vvexp?language=objc
  https://developer.apple.com/documentation/accelerate/vvlog1p(_:_:_:)?language=objc
  Search-index primary-source API text was readable; normal pages require JS and
  the Markdown endpoint failed retrieval. No undocumented ULP or speed guarantee
  is inferred from these API descriptions.
- The maintained HiddenMarkovModels.jl formulas document the forward/backward and
  one/two-state posterior equations, with Rabiner's paper as primary bibliography:
  https://juliastats.org/HiddenMarkovModels.jl/stable/formulas/
  https://web.mit.edu/6.435/www/Rabiner89.pdf
  In exact arithmetic, local gamma and XI denominators coincide by the backward
  recurrence. Reusing that denominator changes floating grouping and can affect
  finite checks, convergence or labels. This experiment deliberately does not
  regroup it, substitute scaled recurrences, or use a global likelihood divisor.

## Isolated original-sequence vector candidate

Only `cpp/operators/latent.cpp` in `source-v41-library` changes:

1. For N>=4096,K>=3, macOS arm64 with the already-linked native-library build flag,
   default/non-scalar dispatch and FE_TONEAREST, batch 128 independent time rows.
2. For each row retain the original s-then-t online logaddexp normalizer sequence,
   conditional maximum, and every XI cell's ascending time reduction. Only the
   independent calls `exp`/`log1p` become `vvexp`/`vvlog1p`. Gamma, density, complete
   forward/backward validation, parameter floors, M step and stopping remain the
   existing code. Scalar, non-nearest, short and unavailable-library lanes remain
   actual supported portable computation, not Python fallback.
3. Nonfinite operands/differences use the original scalar expressions and are
   excluded from vector-library input with harmless dummy zeros. Short final tiles
   call the API with their exact length. No tile consumes prior-workspace contents.
4. Two extra 128-double buffers (2048 bytes) are included in `latent_scratch`
   before any workspace pointer is acquired. The registry and Graph loan already
   derive their small-tail capacity from this same total, so no side allocation
   or fit-time resize is introduced. This minor increase is not a performance gate.
5. The existing `accelerate_vforce_calls` counter records actual calls. No hidden
   user option, Python numerical callback, global cache or thread-setting mutation
   is added. Library batch length stays bounded; elapsed/process-CPU costs and
   Scheduler transport tests are used to verify the one-CPU execution scope.

## Qualification and go/no-go

The 2026-10-04 computation discipline permits only explained floating tails after
four decimals; discrete stopping, errors, dimensions, ownership and finite behavior
remain exact. New controls check frozen independent actual models across C/F/negative
strides, multi-iteration/early-stop/scalar, all four rounding modes, output-prefix
errors/recovery, reused/poisoned workspace, ordinary/prepared Graph budget rejection,
retained readonly generations and neighboring stopping tolerances. Existing tests
remain untouched. New tolerant floating checks do not conceal any exact-test failure.

Any changed iteration/converged/classification or material error rejects adoption.
Independent probability normalization and downstream-state tests are also required.
Only after correctness qualifies, run interleaved complete original/current/candidate
and warmed strong NJIT comparisons with identical workloads and inputs, then fresh
malloc-peak/RSS tests. A profile, leaf screen or small allocation reduction does not
replace full qualification. No production adoption occurs until full regression,
sanitizer/library-off and applicable Excel evidence close for the exact adopted HEAD.

## Original-sequence successor after the failed first screen

The first vector candidate completed six alternating rounds for each complete
workload. Training was 6.23109 seconds versus current 5.54564 (NI 10.52294), and
the 13-output realtime chain was 6.39809 versus current 5.71239 (NI 10.64743).
The ~0.686-second regression rejects that candidate. Twenty-four new controls,
651 existing related tests and native 9/9 passed; those are not speed evidence.
Its training transition-root difference was only 3.47e-18, with all other training
fields identical, but numerical compatibility alone is insufficient for adoption.
No fresh-memory campaign is justified for a slower candidate with added buffers.

The isolated `source-v41-bound` successor retains the same original arithmetic
sequence and library functions. It replaces repeated per-transition finite checks
and duplicate expression reads with one preflight and a remembered maximum:

- Check all contiguous emission/alpha/beta entries and log-transition entries
  against `abs(value)<=DBL_MAX/32` once after the full original validation sweeps.
  NaN/Inf fails this comparison. If any fail, execute the original portable path
  before the vector candidate makes a write.
- Each raw four-term transition sum is bounded by DBL_MAX/8. The checked K-squared
  scratch geometry bounds pair count by PTRDIFF_MAX/8. Each normalizer increment
  is log1p(exp(nonpositive)) <= log(2); even the geometry maximum number of these
  increments is negligible against the reserved DBL_MAX margin. Thus the following
  additions/subtractions cannot overflow. This does not assert a vForce ULP bound.
- Store each row's original conditional maximum beside the exponential argument,
  then add the library log1p result once; no transition expression is reread. XI
  consumes the already-computed library exponential directly in ascending time.
  The original scalar, unsupported-library, non-nearest and exceptional-content
  lanes remain real supported contracts, not obsolete parallel implementations.
- The third 128-double array adds 1024 bytes to the first candidate, 3072 bytes
  above current. All three extra arrays are in the Planner/operator small scratch
  tail before pointer acquisition. No whole matrix is copied and no new thread,
  cache, model reuse or Python numerical path is introduced.

Qualification additionally exercises finite extreme inputs on both sides of the
content guard, and large underflow/zero-probability/constant cases with probability
invariants and exact downstream state labels. The current production source stays
unchanged until complete timing, memory and full adoption checks qualify.

## Additional research and next measured target

SLEEF's maintained upstream repository documents vectorized libm, AArch64 AdvSIMD
and tested Mac/Clang support, under the Boost Software License 1.0:
https://github.com/shibatch/sleef
Its upstream AArch64 API reference specifies a 1.0-ULP version of double vector
exp and log1p:
https://raw.githubusercontent.com/shibatch/sleef/master/docs/aarch64.xhtml
https://raw.githubusercontent.com/shibatch/sleef/master/docs/additional.xhtml
The website XHTML pages failed web retrieval; upstream raw docs were readable.
No SLEEF dependency was downloaded, built, linked or benchmarked in this round.
Its advertised function accuracy is not a full EM error/stop guarantee. Any later
screen retains gradual-underflow/fenv validation and the engine's no-fast-math
build policy; no global FP mode is changed for speed.

The measured forward and backward phases remain ~2.34 seconds in the current
5.10-second HMM fit. A next hypothesis is batching all independent target/source
states at each time into one native vector call, retaining the current sequential
time dependency and each state's ordered max/exponential reduction. The graph must
plan the bounded K-squared buffers; no beta/alpha span is casually overwritten.
This is not implemented or qualified here. It can be compared with SLEEF packet
functions for portability and call overhead. Scalar, exceptional-content and
non-nearest lanes must stay supported, and exact threshold-neighbor, state-label,
error-prefix and retained-output tests still gate adoption.

Reusing gamma's local denominator, as implied by the primary HMM formulas, remains
an independent mathematical regrouping hypothesis rather than a library swap.
It requires its own stopping/finite-behavior evidence. The maintained hmmlearn
native implementation also demonstrates log/scaling methods and a sequence
likelihood divisor:
https://raw.githubusercontent.com/hmmlearn/hmmlearn/main/ext/_hmmc.cpp
We did not copy its code or adopt its scaling, threshold, global-divisor or
log-accumulation semantics. The existing actual engine contracts remain authoritative.
