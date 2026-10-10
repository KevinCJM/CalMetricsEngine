# Ordered HMM transition accumulation

The actual narrow-feature fit uses most time in the sequential logaddexp xi
normalizers. The initial single-core 10000x4, K12, 5-iteration triage gave native
284ms vs warmed NJIT304ms. It does not prove complete-chain acceptance.

The vForce exp/log1p screen reduces isolated xi roughly 4.3x but is not adopted:
tail errors may affect strict convergence and later discrete decisions. The
exact scalar-libm scheduling control instead gives roughly 3x at 10000/20000,
and all four initial controls have bit-identical xi. No new numerical library
or mathematical contract is needed; std::exp and std::log1p remain unchanged.

## Proposed production change

- Keep each row normalizer's source then target sequence and arithmetic tree.
- Process 128 independent time rows before accumulating each xi cell in its
  original increasing-time sequence. No forward/backward recurrence is split.
- One common helper owns the mathematics; block=1 serves small/scalar inputs.
  Automatic n>=128 and K>=3 uses block128, with min(n-1,128) normalizer doubles
  reserved in latent_scratch before any Workspace pointer is acquired.
- Normalizers do not alias any em/alpha/beta/log_a/xi/model/input region. Only
  xi is updated; beta/alpha remain available for subsequent gamma calculation.
- No cache, input pack, additional full matrix or copy. Scratch is at most1KiB;
  this is a diagnostic increment, never a standalone performance veto.
- No new public operator, AST/DAG/IR/IPC/schema or default change. Model fit is
  still one coupled canonical kernel; existing projections and result owners
  remain unchanged. All libm exceptions/checks occur at the existing boundaries;
  the xi loop contains no C++ throwing check and publishes no model state midway.
- No additional threads/processes. Full-request CPU1 timing compares old/new
  AOT wheels and warmed identical complete NJIT outputs. Heap/RSS is separate.
- Existing Excel logical formulas remain valid because only the private physical
  scheduling changes. Check unchanged snapshot parity, not a new export claim.

## Validation required before adoption

Exact models/iterations/convergence against scalar, boundaries127/128/129 and
255/256/257, all layouts, requested scalar, rounding modes, degenerate/tied
models, input/numerical failures, retained independent results, workspace/out
canaries; likelihood-delta nextafter stop neighborhoods. Full supported-chain
all outputs, complete ordinary/prepared pairs, incremental heap/RSS and source
identity. No conclusions about dynamic finite rows, availability-time splits,
default quantile long-sort ties, other platform models or whole-engine acceptance.

## Measurement correction

The first bounded HMM full-chain timing overlapped the initial exact control
screen for a few seconds by orchestration mistake. Its parity remains valid,
but that timing and the overlapping exact timing are diagnostic only. The exact
control was rerun alone (screen-exact-clean.log); complete old/new pairs must
be run serially again before adoption. No numerical process remained live when
the clean screen started; session47506 was observed terminal first.
