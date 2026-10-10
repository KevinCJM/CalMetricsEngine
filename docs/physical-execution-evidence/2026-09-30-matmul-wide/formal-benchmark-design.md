# Ordered C MatMul: formal full-request benchmark plan

The driver files were prepared without running numerical tests, benchmarks or builds. Syntax and metadata-only AST checks passed. They change no repository source or shared harness. The parent controls all execution serially.

## Why these candidates proceed to full-request testing

The completed 51-case caller-output screen suggests that rows4x8 can help C-right cases: candidate/current paired median ratios were 0.941/0.725 for 512³ C/C and F/C, 0.744/0.720 for 1024³, and 0.803/0.715 for 64×32768×64. The zero-A-row-stride candidate ratios were 0.0182 for 512³ and 0.0280 for 32768×64×32 with F B. These are kernel-only screening results from `screen-final/timing-wide.json`, not public request performance or memory acceptance.

Final regular C-right admission requires square M==K==N and exact C-contiguous B (`bc=1`, `br=N` in element strides), `N % 8 == 0`, M/K/N at least 32, automatic ISA and the existing Eigen/macOS/arm64 guards. General F B retains the old 4×4 path. Repeated-row reuse is separately admitted when A row stride is zero and B is C-compatible (`bc=1, br>=N`) or F-compatible (`br=1, bc>=K`); it retains padded C/F support and does not require square dimensions or N divisible by 8. Padded nonrepeated C and nonmultiple-of-eight C-tail cases remain in the full original-plus-added measurement set and use fallback. Their unsuccessful candidate results are preserved under `timing-candidate` and `candidate-16d2`, rather than removed after seeing performance.

## Reuse and provenance

`check_matmul_wide_performance.py` imports `tools/check_matmul_performance.py` and reuses its setup, typed materialized ordinary/prepared outputs, direct result path, canonical comparison, audit policy and shared timing/memory code. It only adapts baseline identity, input-layout generation, case selection and child routing. Both `base.__file__` (canonical reference workers) and `bench.__file__` (timing/heap/RSS workers) point to this temporary adapter. This is essential: launching the repository adapter as a worker would restore its prior 4e12 baseline.

Final measurement uses equal-length real directories `/private/tmp/calmetrics-matmul-wide/square-install-old` and `/private/tmp/calmetrics-matmul-wide/square-install-new`. The old bytes remain build `1aec1a6b0d4a07f0caee43b58cd4b339bdcc165f5e615f2190b15da0ba66e86b`; the new directory receives the final square-admission build. Copies must match their source file hashes and must not be symlinks resolved back to unequal paths. This removes an installation-path allocation confound found in the 16d2 campaign; the original +8 B retained / +72 B allocated evidence and six-child attribution remain preserved, with details in `install-path-memory-attribution.md`.

Before each independent campaign, separate old/new processes obtain source identity through the existing `mean` audit diagnostic; no MatMul call occurs. The new identity is then frozen in an inherited environment variable and verified by every new child. This supports a newly built wheel without inventing its SHA in advance. The initial manifests record both build identities and installed Python/native file hashes; every report also contains the actual identity. Timing and memory campaigns must be compared to confirm the same new build was used.

Ordinary and prepared results materialize every public output field inside the measured callable; prepared uses `run_snapshot()`. The previous result remains alive until the next call returns. NJIT is `a @ b`, compiled for the actual readonly/layout signature before the memory epoch, without executing a numerical warmup there. Its first BLAS call and initialization occur inside cold memory measurement. Input arrays, imports and input hashing occur before the epoch; hashing may allocate/free a copy, so this is not pristine allocator startup. The graph/setup and retained native workspace are included in cold request memory. Direct audit remains explicitly not collected, avoiding an extra native MatMul call outside the measured request. Ordinary/prepared return their actual audit.

## Frozen timing selection: 114 cases

All original 90 cases remain unchanged in order, shape, layout, entry, seed and tolerance. Add 24:

- 512³ zero A row stride with C B: ordinary/direct/prepared (3).
- 512³ zero A row stride with F B: direct/prepared (2); ordinary already belongs to the original 90.
- 32768×64×32 and 64×32768×64, zero A row stride with C/F B, ordinary (4).
- 512³, negative-inner-stride A / sliced A / padded C B / zero-row A plus padded C B, ordinary (4).
- 64×32768×64 F/C, direct/prepared (2); ordinary and the existing large F-left square cases are already covered.
- 129×257×65, 32×32×33, 33×33×39 and 512×512×513, C/C and F/C, ordinary (8).
- 513×512×520 C/C ordinary (1).

The same data generators feed old/new/NJIT. The original three-process timing scheduler activates only one backend at a time, rotates its first backend across 11 paired rounds, and records all wall/CPU nanoseconds. Cases with M<256 have five calls per round, including large rectangles such as 64×32768×64; other cases have one. Both remain 11 statistical clusters, not 55 independent rounds. Inputs, independent-result checks, scalar reference reads and sampled high-accuracy arithmetic are outside timing. Existing canonical native bit comparison and NJIT `rtol=atol=1e-11` are unchanged; the bit comparison is implementation preservation evidence, not a new public tolerance contract.

## Frozen memory selection: 51 cases, 306 children

`run_wide_memory.py` imports the previous selector/validator and preserves its complete 31-case set. It adds 20 cases, all present in the timing selection:

- 512³ C/C and F/C direct/prepared (4).
- 1024³ F/C ordinary (1).
- 64×32768×64 C/C and F/C ordinary (2).
- 512³ zero-row A with C B ordinary/direct/prepared, and F B direct/prepared (5).
- 32768×64×32 zero-row A with C/F B ordinary (2).
- 512³ negative-inner-stride A / sliced A / padded C B / zero-row A plus padded C B (4).
- 512×512×513 C/C and F/C ordinary (2).

The old/new/NJIT × heap/RSS children execute sequentially, using the unchanged continuous-live allocation probe and independent RSS worker. All 306 expected report identities and case combinations must be present exactly once. This is 51/114 memory coverage; it must not be described as complete 114-case memory acceptance. Heap, RSS and static stack evidence stay separate. Record positive RSS differences and fixed byte differences exactly; neither the advisory lock nor the one-thread budget establishes CPU affinity or an idle system. The parent must pause unrelated numerical/build work.

## Parent execution commands

Run timing after correctness succeeds, then memory after timing completes. Both commands refuse to overwrite their existing final report; choose a fresh directory for any rerun and retain old results.

```sh
CME_MATMUL_REPO_ROOT=/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine /Users/chenjunming/Desktop/myenv_312/bin/python3.12 /private/tmp/calmetrics-matmul-wide/check_matmul_wide_performance.py --old-install /private/tmp/calmetrics-matmul-wide/square-install-old --new-install /private/tmp/calmetrics-matmul-wide/square-install-new --output /private/tmp/calmetrics-matmul-wide/timing-final --mode timing
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 /private/tmp/calmetrics-matmul-wide/run_wide_memory.py --repo /Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine --old-install /private/tmp/calmetrics-matmul-wide/square-install-old --new-install /private/tmp/calmetrics-matmul-wide/square-install-new --heap-probe /private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib --output /private/tmp/calmetrics-matmul-wide/memory-final
```

Do not invoke the adapter with `--mode memory` unless intentionally measuring all 114 cases; the fixed 51-case selection belongs to the dedicated memory orchestrator. Both campaigns record source hashes and build ids. A completed campaign is measurement evidence, not an automatic assertion that time, heap and RSS jointly pass.

The completed d2e 114/51 campaign is preserved under `candidate-d2e`: rectangular C regressions prompted square-only regular-C admission. The 114/51 case sets remain intact; excluded rectangles still execute their existing fallback and are measured. Repeated-row admission is unchanged. Final comparison uses real equal-length `square-install-old/new` directories (54 path bytes), independent of prior final/equal install trees.
