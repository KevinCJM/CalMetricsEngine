# Multi-product / Multi-interval / Multi-metric Benchmark

## Purpose

Single-operator microbenchmarks are not sufficient to evaluate CalMetricsEngine.

The relevant workload is:

```text
products × intervals × shared indicator DAG × observations
```

This benchmark compares:

- BetterSaaTaa production `CompiledNumbaBatchPlan`
- CalMetricsEngine C++ canonical Operator Registry

using the **same product-major NAV data, the same interval rows, the same 16 metric definitions and the same output matrix**.

The C++ side is a standalone benchmark harness linked directly to the production Operator Registry. It does not call one operator at a time through Python. Its row-parallel mode is benchmark-only static partitioning and is not the future persistent Scheduler/ThreadPool.

## Metric set

The 16 scalar roots are:

1. total return
2. annualized return
3. mean return
4. sample standard deviation
5. median return
6. minimum return
7. maximum return
8. 5% quantile
9. mean absolute deviation
10. root mean square
11. maximum drawdown
12. new-high ratio
13. NAV linear slope
14. NAV R²
15. mean positive return
16. standard deviation of negative returns

BetterSaaTaa compiled these roots into one shared graph:

```text
input nodes:      62
shared nodes:     40
eliminated nodes: 22
```

Returns are generated once per interval and shared across roots.

## Correctness

For both benchmark sizes and all tested thread counts:

```text
max absolute error = 0
max relative error = 0
```

C++ serial, C++ row-parallel and BetterSaaTaa NJIT produce the same 16 values for every interval row.

## Benchmark environment

- macOS 15.6.1
- Apple ARM64
- 10 physical / logical CPU cores
- Python 3.12.11
- NumPy 1.26.4
- Numba 0.60.0
- warm compute timings
- inputs and outputs allocated before timing
- BetterSaaTaa compilation/warmup excluded from warm compute timing and reported separately
- C++ input-file loading excluded from its internal timing

## Workload A: 500 × 12 × 16

```text
products              500
history/product       2520
intervals/product       12
interval rows         6000
metrics                 16
result cells         96000
interval observations 3,213,000
NAV input            10.08 MB
```

### Warm compute

| Threads | BetterSaaTaa NJIT | CalMetricsEngine C++ | C++ / NJIT |
| ---: | ---: | ---: | ---: |
| 1 | 286.150 ms | 356.621 ms | 1.246× |
| 2 | 146.292 ms | 182.664 ms | 1.249× |
| 4 | 73.384 ms | 91.044 ms | 1.241× |
| 8 | **45.181 ms** | 56.170 ms | 1.243× |
| 10 | 46.478 ms | **55.495 ms** | 1.194× |

Current C++ is approximately **19%–25% slower** than the fused NJIT batch on this workload.

### Memory and compile behavior

BetterSaaTaa:

```text
runtime plan compile       ≈ 10.04 s
RSS high-water before JIT   324.8 MB
RSS high-water after JIT  1,107.9 MB
warm benchmark growth         ~32 KB
```

The RSS delta is a process high-water measurement and includes Numba/compiler/runtime allocator effects; it is not an exact live-allocation figure.

CalMetricsEngine standalone C++ process at 10 threads:

```text
peak process RSS             ~13.5 MB
input NAV                     10.08 MB
result output                  0.77 MB
worker buffers                 0.43 MB
operator workspace             0.16 MB
warm repeated-run RSS growth  ~32 KB
runtime compilation               0
```

Standalone C++ RSS is not directly equivalent to embedding the extension in a Python service process, but it demonstrates that the AOT numerical runtime itself is small and does not require JIT compiler state.

## Workload B: 1000 × 12 × 16

```text
products             1000
history/product       2520
interval rows        12000
result cells        192000
interval observations 6,426,000
NAV input             20.16 MB
```

| Threads | BetterSaaTaa NJIT | CalMetricsEngine C++ | C++ / NJIT |
| ---: | ---: | ---: | ---: |
| 4 | 144.428 ms | 181.897 ms | 1.259× |
| 8 | 83.902 ms | 108.890 ms | 1.298× |
| 10 | **82.746 ms** | **107.748 ms** | 1.302× |

At the larger workload, the current C++ registry remains roughly **26%–30% slower**.

## Why Phase 1 C++ is not faster yet

This result is expected from the current architecture.

BetterSaaTaa already executes:

```text
16 roots
  ↓
shared Typed DAG
  ↓
CSE: 62 nodes → 40 nodes
  ↓
one compiled row kernel
  ↓
prange over interval rows
```

Phase 1 CalMetricsEngine executes:

```text
interval
  ↓
returns once
  ↓
metric 1 prepare/execute
metric 2 prepare/execute
...
metric 16 prepare/execute
```

The benchmark already removes Python-per-operator calls and pre-resolves opcodes, but C++ still lacks:

- NativeExecutionPlan
- graph-wide CSE
- liveness / buffer planning
- compatible-reduction fusion
- persistent native thread pool
- Scheduler
- SIMD reductions/statistics
- BLAS dispatch for matrix-heavy plans

Most of this 16-metric workload is reductions, sorting, regression and path-state calculation. Only a small fraction currently benefits from the 18 Phase-1 SIMD operators.

## Architectural conclusion

The next performance milestone should **not** be “rewrite more individual NumPy operations in C++”.

The next milestone should be:

```text
Typed DAG
  ↓
NativeExecutionPlan
  ↓
C++ graph-wide CSE
  ↓
one-pass compatible reduction fusion
  ↓
liveness-based workspace reuse
  ↓
persistent ThreadPool / Scheduler
  ↓
SIMD reductions + native kernels
```

A key optimization opportunity is to combine metrics that scan the same return interval. For example, growth, mean, min/max, positive/negative counts and sums, moments and RMS can share passes instead of independently reading the same window.

Only after this native fused-plan layer exists is it meaningful to make a final “C++ engine vs NJIT engine” performance judgment.

## Reproduce

```bash
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 \
  tools/benchmark_multiworkload.py \
  --products 500 \
  --history 2520 \
  --intervals 12 \
  --thread-grid 1 2 4 8 10 \
  --threads 10 \
  --repeats 5 \
  --output .build-multiworkload/full-500x12x16.json
```

The benchmark creates `calmetrics_engine_multiworkload_benchmark` with:

```text
CALMETRICS_ENGINE_BUILD_BENCHMARKS=ON
```

This target is disabled by default and is not part of the production runtime.
