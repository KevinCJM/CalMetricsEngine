# C++ Native vs BetterSaaTaa NJIT — Physical-DAG Benchmark Matrix (2026-09-20)

## Method

- Same deterministic product-major NAV data and same formulas for both backends.
- NJIT baseline is BetterSaaTaa CompiledNumbaBatchPlan; warm compute excludes compile/JIT.
- Native path is C++ Compiler → Physical DAG metadata → Planner → Scheduler → graph/branch executor.
- Paired timing alternates which backend runs first.
- CPU budget is 10.

## Time matrix

| Scenario | P | I | M | History | Native lane | NJIT | C++ | C++/NJIT |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| micro_1p_1i_1m_63 | 1 | 1 | 1 | 63 | single/product | 3.250 µs | **2.833 µs** | **0.872×** |
| micro_1p_1i_1m_2520 | 1 | 1 | 1 | 2520 | single/product | 19.708 µs | **19.041 µs** | **0.966×** |
| micro_1p_1i_5m_63 | 1 | 1 | 5 | 63 | single/product | 4.709 µs | **4.125 µs** | **0.876×** |
| small_2p_2i_3m | 2 | 2 | 3 | 252 | single/interval | 0.113 ms | **13.292 µs** | **0.118×** |
| small_5p_2i_5m | 5 | 2 | 5 | 504 | single/interval | 0.133 ms | **0.102 ms** | **0.763×** |
| product_heavy_100p_1i_3m | 100 | 1 | 3 | 504 | thread/product | 0.274 ms | **0.164 ms** | **0.599×** |
| interval_heavy_1p_16i_5m | 1 | 16 | 5 | 2520 | thread/interval | 0.339 ms | **0.172 ms** | **0.507×** |
| metric_heavy_10p_2i_16m | 10 | 2 | 16 | 756 | thread/product | 0.241 ms | **0.138 ms** | **0.573×** |
| dag_branch_1p_1i_16m | 1 | 1 | 16 | 200000 | thread/dag_branch | 34.153 ms | **13.899 ms** | **0.407×** |
| mixed_10p_12i_3m | 10 | 12 | 3 | 1512 | thread/product | 0.402 ms | **0.246 ms** | **0.612×** |
| mixed_50p_4i_8m | 50 | 4 | 8 | 1008 | thread/product | 1.152 ms | **0.678 ms** | **0.588×** |
| mixed_100p_8i_8m | 100 | 8 | 8 | 1512 | thread/product | 8.266 ms | **4.602 ms** | **0.557×** |
| many_products_500p_1i_5m | 500 | 1 | 5 | 2520 | thread/product | 14.096 ms | **11.395 ms** | **0.808×** |
| large_500p_12i_16m | 500 | 12 | 16 | 2520 | thread/product | 84.343 ms | **52.357 ms** | **0.621×** |
| large_1000p_12i_16m | 1000 | 12 | 16 | 2520 | thread/product | 203.236 ms | **123.619 ms** | **0.608×** |

All 15 final scenarios have Native/NJIT < 1.0.

Notable physical-DAG case: 1 product × 1 interval × 16 metrics × 200,000 observations is planned as four independent DAG branches and measures about 0.397× NJIT latency on the current ARM64 host.

## Planner metadata effect

Fusion-aware physical estimates are materially lower than logical-node estimates for shared-DAG workloads. For example, the 500/1000-product × 12-interval × 16-root cases retain the same numerical graph but Planner decisions use post-fusion physical work instead of charging every logical reduction independently.

Product and interval chunks are now weighted by physical row work. Single/few-row requests may use dag_branch only when multiple independent compiled branches are heavy enough and row parallelism cannot fill the CPU budget.

## Memory reference

Memory transport/runtime behavior is unchanged by this Planner optimization. Fresh-process representative measurements from the prior memory matrix remain valid for the same execution paths:

| Scenario | NJIT warm RSS growth | C++ warm RSS growth | C++ post-warm high-water growth |
| --- | ---: | ---: | ---: |
| micro_1p_1i_1m_63 | 1 | 1 | 1 | 63 | single/product | 3.250 µs | **2.833 µs** | **0.872×** |
| small_5p_2i_5m | 5 | 2 | 5 | 504 | single/interval | 0.133 ms | **0.102 ms** | **0.763×** |
| interval_heavy_1p_16i_5m | 1 | 16 | 5 | 2520 | thread/interval | 0.339 ms | **0.172 ms** | **0.507×** |
| mixed_100p_8i_8m | 100 | 8 | 8 | 1512 | thread/product | 8.266 ms | **4.602 ms** | **0.557×** |
| large_1000p_12i_16m | 1000 | 12 | 16 | 2520 | thread/product | 203.236 ms | **123.619 ms** | **0.608×** |

## Planner optimization

- physical work replaces logical work for lane selection;
- interval/product partitions use physical row weights;
- independent root components are compiled into dependency-closed branch Programs;
- dag_branch fork/join is enabled only for heavy coarse branches;
- branch tasks are ordered by work and executed in CPU-token-bounded waves;
- per-branch arena slots are compacted;
- process SharedMemory policy and output ownership are unchanged.

## Evidence

- .build-benchmark-matrix/time-physical-dag-final.json
- tools/benchmark_cpp_vs_njit_matrix.py
- docs/planner-physical-dag-optimization-design-2026-09-20.md

