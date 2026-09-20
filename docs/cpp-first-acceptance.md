# C++-first migration acceptance

## Scope and implementation

The computation engine's restricted expression parser, type checks, composite lowering, CSE,
borrowed-view-aware liveness, cost model, product/interval partitioning, CPU admission, thread pool,
process pool, shared mappings and execution now live in C++17. Existing canonical operator
mathematics and public method names are retained.

The four former Python core modules are now interface adapters only:

| File | Lines, including comments |
| --- | ---: |
| `graph.py` | 9 |
| `planner.py` | 5 |
| `runtime.py` | 48 |
| `shared.py` | 19 |

These files have no Python AST compiler, liveness optimizer, cost calculation, numerical chunk
partitioning, multiprocessing worker or numerical thread pool. The runtime's `asyncio.to_thread`
is only an await bridge around a complete native request. Other Python files still provide the
public finance/registry interface. This is not a claim of literally zero Python instructions.

A normal synchronous request has one Python-to-native boundary. Worker tasks contain no Python
objects or callbacks. `calmetrics_worker` is a standalone native executable, built and tested with
`CALMETRICS_ENGINE_BUILD_PYTHON=OFF`. Local `otool -L` shows only libc++ and libSystem, not Python.

## Improvements beyond moving files

- Shared-process output is returned as a NumPy view pinning a native shared-region owner. No final
  copy from shared output to a second NumPy output is required.
- Normal results remain independent across calls and valid after engine close. Prepared output
  deliberately reuses storage and is overwritten on the next run.
- The compiler extends backing-buffer lifetime transitively through borrowed `lag` views.
- Prepared bindings recheck pointer/dtype/shape/stride geometry. Supplied plans reject changed
  interval/product geometry instead of reusing stale memory and scheduling estimates.
- Graph, canonical-operator and legacy finance PyBind array entry points validate that every strided
  view remains inside its real owner allocation; forged/out-of-bounds views fail closed.
- Audit is a per-execution native snapshot rather than counters copied from a previous call.
- Physical read-only mappings expose read-only buffers; Python cannot enable writes with
  `setflags(write=True)` and then fault on a protected OS page.
- Native failures drain submitted tasks before releasing input owners, mappings or CPU tokens.
  Process deadlines terminate and wait; thread timeouts cannot forcibly interrupt an active kernel.
- One process-wide native CPU-admission budget and persistent ThreadPool cover Graph requests and legacy finance APIs; each Engine may impose a smaller local cap, but independent callers cannot bypass global admission.
- POSIX workers use spawn, not fork of an active threaded Python interpreter; Windows has a native
  CreateProcess/file-mapping implementation whose remote CI remains to be run.

## Regression and native tests

Current installed-source Python suite: **4034 passed**. This includes the existing C++-first acceptance coverage for parser precedence/limits and Unicode names, no Python parser/pool use on synchronous paths, borrowed-buffer liveness, rebind safety, stale plans, changed parameters, output independence, physically read-only shared inputs/descriptors, timeout/error recovery, process-wide CPU admission, async adaptation, legacy-finance scheduler routing, and installed worker availability.

Local native results:

| Configuration | Result |
| --- | --- |
| macOS ARM64, Release, Python disabled | 5/5 CTest passed |
| macOS ARM64, ASan/UBSan, Python disabled | 5/5 CTest passed |
| macOS x86_64 cross-build executed under Rosetta | 5/5 CTest passed |

These include standalone finance, operator, graph, compiler and runtime/process/shared-memory tests.

Installed artifact verification (same native source, current regression suite):

| Environment | Installed artifact | Result |
| --- | --- | --- |
| CPython 3.10.20 | sdist, built locally | 4034 passed |
| CPython 3.11.13 | sdist, built locally | 4034 passed |
| CPython 3.12.11 | wheel built from sdist | 4034 passed |
| CPython 3.13.11 | sdist, built locally | 4034 passed |
| CPython 3.14.6 | sdist, built locally | 4034 passed |
| CPython 3.12.11 + NumPy 1.26.4 | same CPython 3.12 wheel | 4034 passed |

A NumPy 2.5 deprecation warning was exposed by the negative test that deliberately changes an
existing ndarray's dtype/shape. Only that intentional-mutation test suppresses the matching
warning; production validation is unchanged. It would be incorrect to use a new view in this test,
because that would not exercise a stale bound pointer on the same object.

Ruff lint/format, `git diff --check`, wheel/sdist contents and Twine strict metadata checks passed.
The archive checker now requires the native worker and verifies its POSIX executable mode.

## Full-workload performance gate

Environment: local macOS ARM64, ten CPU cores, CPython 3.12.11, NumPy 1.26.4 and Numba 0.60.0.
The NJIT baseline is the actual BetterSaaTaa `CompiledNumbaBatchPlan`. Both implementations use
identical data/formulas, warmed execution, equal CPU budgets and alternating-first paired timing.

Executed command:

```bash
.venv-cpp-first/bin/python tools/check_phase2_performance.py \
  --products 500 1000 --micro-histories 63 252 \
  --cpu-budget 10 --repeats 5 \
  --output-dir .build-cpp-first-review/final-gate
```

The actual gate returned PASS for all four workloads:

| Workload | NJIT paired median | Native paired median | Native/NJIT |
| --- | ---: | ---: | ---: |
| 500 products × 2520 observations × 12 intervals × 16 roots | 37.574 ms | 24.213 ms | 0.644 |
| 1000 products × 2520 observations × 12 intervals × 16 roots | 73.830 ms | 46.194 ms | 0.626 |
| 1 product × 1 interval × 5 roots × 63 observations, prepared | 4.375 µs | 2.250 µs | 0.514 |
| 1 product × 1 interval × 5 roots × 252 observations, prepared | 9.083 µs | 6.083 µs | 0.670 |

The ordinary public scheduler micro path is also tested independently in paired runs:

| Observations | Ordinary scheduler median | Scheduler/NJIT |
| ---: | ---: | ---: |
| 63 | 3.791 µs | 0.858 |
| 252 | 7.625 µs | 0.836 |

Each micro comparison has its own paired NJIT samples; do not derive the scheduler ratio from
the prepared pair's displayed NJIT median. All checked full-workload values match the NJIT
reference, with maximum absolute error 0 in the recorded runs. This does not imply bitwise
identity for every possible formula or hardware target.

Prepared/batch ratios must remain <=0.90 and ordinary scheduler micro ratios <=1.00. The gate
is an explicitly runnable check; this work does not claim it automatically runs on every commit.
No cold-start advantage is used to excuse a slower warmed calculation.

## Compilation/planning overhead against the previous Python control layer

`tools/benchmark_cpp_first_control.py` runs the same workload with the old installed engine and
the new installed engine; each named operation has the same timing boundary. Workload is one
product, one interval, five roots, 252 observations, 301 timing samples.

| Operation | Previous Python control layer | Native control layer, repeat |
| --- | ---: | ---: |
| Compile five expressions | 308.958 µs | 30.292 µs |
| Plan execution | 25.334 µs | 1.708 µs |
| Bind prepared execution | 2.209 µs | 1.667 µs |
| Execute already-planned request | 8.125 µs | 8.083 µs |
| Plan and execute together | 34.084 µs | 8.458 µs |
| Prepared.run | 6.292 µs | 6.291 µs |

The principal gain here is compilation/planning, not a claim of making an already-native kernel
10× faster. One initial candidate control run had elevated execute timings (19.333 µs); repeat
and paired-NJIT runs did not reproduce that slowdown. All original logs are retained rather than
replaced. Treat this table as measured stage evidence, not a deterministic timing promise.

Evidence locations:

- `.build-cpp-first-review/control-python.json`
- `.build-cpp-first-review/control-native.json`
- `.build-cpp-first-review/control-native-repeat.json`
- `.build-cpp-first-review/final-gate/`

## Memory and transport

The final 500/1000-product full-workload audit reports:

| Item | 500 products | 1000 products |
| --- | ---: | ---: |
| Thread lane input boundary copies | 0 | 0 |
| Ten workers' arena + operator + order scratch capacities | 418,000 B | 418,000 B |
| Process shared input/interval/output logical bytes | 10,944,000 B | 21,888,000 B |
| Reusable-input process call's starts/ends boundary copy | 96,000 B | 192,000 B |
| Final process result copy | **0** | **0** |

The process benchmark reuses a pre-created SharedInputBundle. The initial copy into that bundle
is deliberately outside repeated compute timing and is separately counted when constructed.
Algorithm scratch, including the copy used for sorting, is not input-binding copying. The memory
figures above are explicit native audit counters/capacities, not process RSS measurements or
proof of a global hard memory cap.

## Extended mixed-shape benchmark

A separate 15-scenario matrix now covers single/small/large product, interval and metric counts, including a single-row heavy-DAG branch case. All 15 scenarios have Native/NJIT < 1.0. The slowest relative case is the single-product/single-interval/single-metric long-history call at about 0.969× NJIT latency; large 500/1000-product × 12-interval × 16-root workloads are about 0.638× / 0.580×, while the new 1-product × 1-interval × 16-root DAG-branch scenario is about 0.397×.

The matrix also exposed and fixed two scheduling errors: the default thread threshold was too high for interval-heavy jobs, and the thread lane was unnecessarily capped by `min_rows_per_worker`. The default threshold is now 250,000 work units and justified thread-lane jobs use `min(cpu_budget, independent_rows)`, while memory budget can still reduce concurrency and ProcessPool retains coarse-grained row limits.

Fresh-process memory runs show representative NJIT compile/warm RSS growth of roughly 76–989 MB versus roughly 0–2 MB C++ warm growth. Large repeated C++ calls do retain allocator/thread/output high-water (about 18 MB around 50–200 repeats and about 28 MB at 1000 repeats in the observed run); this is sublinear rather than evidence of a linear leak, but remains a memory-tuning opportunity. Full methodology and tables are in `docs/cpp-vs-njit-benchmark-matrix-2026-09-20.md`.

## Remaining boundaries

- The research platform's production financial DSL, semantic axes and causality contracts remain
  external. Its formulas are not silently migrated by replacing the runtime.
- Matrix-growing/portfolio-matrix nodes inside the interval DAG and a BLAS backend are not added.
- Coroutine cancellation alone does not hard-kill native work; use isolated execution deadlines.
- Linux and Windows runtime/packaging acceptance still needs actual platform CI. macOS x86_64
  here is Rosetta, not a physical Intel performance benchmark.
- No Git commit, push, PyPI publication or production platform integration is part of this run.
