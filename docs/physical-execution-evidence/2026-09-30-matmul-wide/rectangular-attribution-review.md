# Read-only attribution of rectangular MatMul timing reversal

Scope: static source/manifest/sample inspection only. No build, numerical run or timing was started for this review. These are unresolved attribution findings, not permission to discard a failing case or select the fastest campaign.

## What the completed records actually show

`timing-final/timing.json` now contains all 114 cases. The target is M=32768,K=64,N=32, A Fortran / B C. All native campaigns use the same input-value SHA, shapes, strides and read-only flags; scalar canonical comparisons passed.

| Campaign | Entry | Old median ms | New median ms |
|---|---|---:|---:|
| 529 | ordinary | 31.019292 | 28.750250 |
| 16d2 | ordinary | 32.074416 | 13.148375 |
| aligned d2e734 | ordinary | 11.321625 | 28.177542 |
| 529 | prepared | 31.156500 | 28.912292 |
| 16d2 | prepared | 32.185709 | 12.922958 |
| aligned d2e734 | prepared | 11.828959 | 28.430583 |
| 529 | direct | 30.566542 | 27.777375 |
| 16d2 | direct | 30.228000 | 28.278125 |
| aligned d2e734 | direct | 30.908125 | 28.374167 |

This is a stable within-case regime change, not a lone outlier: current ordinary old has 11 samples in 11.179–12.217 ms, new in 28.046–28.379 ms. In 16d2 old is 31.931–32.703 ms, new 12.912–13.503 ms. Rotating old/new/NJIT order does not remove the regimes.

`process_time_ns` was recorded. Current old/new median CPU/wall ratios are 1.00022/1.00006; previous campaign ratios are likewise about one. Graph audit attributes nearly the entire elapsed duration to compute, with queue wait about 0.001–0.005 ms. This excludes scheduler queueing/descheduling as the 17 ms difference; it does not distinguish core type/frequency from memory stalls.

The second failure shares a long power-of-two leading stride: M=64,K=64,N=32768 CC ordinary is old32.913458/new50.513208 ms in current records, versus old60.694375/new33.489750 in16d2. CPU medians track wall medians. No conclusion that its reversal has exactly the same cause is warranted yet.

## What is and is not controlled

- Old installed `.py`/`.so` hashes match between16d2/current; all four benchmark source hashes match too.
- New `matrix_backend.cpp.o` SHA remains `749654023ecccee37619a541149e4169c05b0d8ff4d9a7a2e3972f7700691118` across529,16d2,aligned. Only dispatcher admission narrowed. These two rectangular cases enter the same C helper in every candidate.
- One long-lived child per backend processes the whole case sequence. Each case allocates new inputs and warms two independent outputs. Each of11 rounds then executes once for M>=256, retaining the preceding output until its successor returns. All complete output wrapping is inside the timer.
- The harness rotates request order, limits engine/thread-library concurrency to1 and records process CPU and wall times. It neither pins nor records an actual OS CPU/core, core frequency, QoS or hardware cache/TLB counters. `cpu_budget=1` is a concurrency budget, not affinity.
- Input identity excludes addresses. Target direct audit is deliberately absent to avoid unmatched extra numerical work. Graph audits do not carry source/result addresses. Scalar-reference manifest addresses are from a separate, untimed process, so cannot identify the timed allocations.
- The equal-length old/new installation paths now remove a known worker string-size memory confound, but change process/environment/allocator history relative to the previous campaign. There is no evidence that their string lengths themselves caused these numerical timings.

## Plausible mechanism and evidence against overclaiming

The FC target has A byte strides(8,262144), A16MiB, B16KiB, output8MiB. Both old4x4 and new4x8 kernels traverse increasing k inside a tile, so one k sweep touches64 A locations separated by262144 bytes before advancing the output tile. The wide CC failure has B byte strides(262144,8), with the same64-way long-stride access pattern. Neither kernel packs these inputs. An interaction among allocation placement, translation/cache indexing and this access pattern is therefore a concrete hypothesis worth testing. Cache geometry and misses were not captured: do not label this a proven cache-conflict root cause.

P/E-core or frequency differences remain possible, because CPU time includes stalls and provides no core identity. However, a whole-process/whole-campaign explanation alone is weak: direct timings stay near30/28 ms across all campaigns while the graph entries reverse; neighbouring32768 CC/CF/FF cases remain near6.3/7.4/32 ms across campaigns. These observations make allocation/path-specific state a stronger initial target than asserting universal thermal noise. The full graph's reported numerical compute time shifts too; Python result wrapping or IPC is not the dominant difference.

There is no demonstrated mathematical/ownership/dispatch defect. There **is** a demonstrated performance-acceptance failure for the current rectangular C admission, and prior favourable runs do not cancel it.

## Minimal attribution experiment after the serialized final campaign

1. Keep the frozen aligned candidate install. In one process use identical read-only a/b and explicit preallocated independent output buffers. Alternate `operators.call('matmul',...,simd='auto',out=...)` and `simd='neon'` in ABBA order, checking exact output agreement outside timers. Explicit NEON reaches the unchanged old SIMD kernel in that same binary. This controls input addresses, process, binary/load environment and arithmetic boundary simultaneously. It is an attribution measurement only; caller-owned output timings must not replace the independent-result acceptance numbers.
2. Record a/b/out addresses and their offsets modulo64/4096/16384/262144, wall+process/thread CPU, page faults and context switches around calls, plus actual core/frequency only if genuinely observable. Do not represent QoS or the engine budget as a core pin.
3. Repeat only the two problematic shapes with a small predetermined set of backing-storage offsets (e.g.0/64/4096 bytes) while retaining exact logical values/strides, then two fresh processes. This changes address placement while keeping the numerical algorithm and dimensions constant. A timing regime that follows offsets with a stable same-process control implicates placement; one that follows observed core/frequency instead implicates scheduling. If neither is observed, retain the uncertainty rather than tune an address-dependent production rule.

If the candidate has already narrowed to square-only, use the preserved aligned install for this diagnostic; current auto would intentionally use the old rectangular path and would not compare two kernels.

## Conservative production scope

Restricting nonrepeated C4x8 to M==K==N plus exact-C B and N divisible by8 is mathematically valid: only backend selection changes; the established rectangular kernel remains available. It is an empirical acceptance boundary supported by the measured square cases, not a mathematical claim that every square is faster or every rectangle is slower. Keep repeated-row exact dependency reuse before that predicate; its independent benefit and correctness do not depend on a square. Preserve all rectangular failing records and fixtures, with this factor hypothesis explicitly open.
