# MatMul full-request summary

Validated 114/114 timing cases and 51/51 memory cases (306 children). Memory is 51/114 coverage.

Original strict Phase-2 gate: 4/4. Its raw thresholds and failures remain unchanged; formal collection does not override this separate gate.

Ratios below one favor new. Median ratios use 11 paired rounds; five inner calls when M<256 (including large rectangles) are one cluster. Bootstrap bounds are descriptive and unadjusted for multiple comparisons.

| Path group | Cases | New/old ratio range | New/NJIT ratio range |
|---|---:|---:|---:|
| all | 114 | 0.03268–1.569 | 0.2548–29.02 |
| Cright | 24 | 0.6501–1.035 | 2.889–21.16 |
| repeated | 11 | 0.03268–0.1521 | 0.2548–1.484 |
| unchanged | 79 | 0.7797–1.569 | 2.876–29.02 |

## Selected cases

| Case | Old ms | New ms | NJIT ms | Paired new/old | Paired new/NJIT |
|---|---:|---:|---:|---:|---:|
| matmul_1024x1024x1024_cc_ordinary | 201.798 | 164.584 | 8.09042 | 0.8205 | 20.4 |
| matmul_1024x1024x1024_fc_ordinary | 221.46 | 143.444 | 8.13742 | 0.6501 | 17.61 |
| matmul_64x32768x64_fc_ordinary | 61.3538 | 60.3338 | 3.96996 | 0.9835 | 15.28 |
| matmul_512x512x512_zero_a_f_b_ordinary | 13.3325 | 0.442042 | 1.65796 | 0.03363 | 0.2548 |
| matmul_1024x1024x1024_cc_direct | 141.628 | 144.832 | 8.07396 | 1.013 | 17.98 |
| matmul_1024x1024x1024_cc_prepared | 238.487 | 167.613 | 8.28875 | 0.7032 | 19.83 |
| matmul_1024x1024x1024_fc_direct | 239.128 | 163.759 | 8.34746 | 0.6632 | 19.3 |
| matmul_1024x1024x1024_fc_prepared | 232.751 | 178.977 | 8.24104 | 0.7594 | 21.16 |
| matmul_512x512x512_zero_a_c_b_ordinary | 11.8556 | 0.501208 | 1.29496 | 0.04301 | 0.3789 |
| matmul_512x512x512_zero_a_c_b_direct | 11.8318 | 0.547834 | 1.56629 | 0.04555 | 0.3708 |
| matmul_512x512x512_zero_a_c_b_prepared | 11.7425 | 0.510625 | 1.38021 | 0.04317 | 0.3854 |
| matmul_512x512x512_zero_a_f_b_direct | 13.3664 | 0.437959 | 1.54346 | 0.03268 | 0.2967 |
| matmul_512x512x512_zero_a_f_b_prepared | 13.3228 | 0.457334 | 1.55708 | 0.03435 | 0.2809 |
| matmul_32768x64x32_zero_a_c_b_ordinary | 6.43358 | 0.958708 | 2.04237 | 0.1504 | 0.4763 |
| matmul_32768x64x32_zero_a_f_b_ordinary | 7.43996 | 0.968625 | 2.04775 | 0.1303 | 0.4724 |
| matmul_64x32768x64_zero_a_c_b_ordinary | 46.3585 | 4.57854 | 3.03379 | 0.09762 | 1.484 |
| matmul_64x32768x64_zero_a_f_b_ordinary | 13.794 | 2.09842 | 6.49038 | 0.1521 | 0.3208 |
| matmul_512x512x512_zero_a_padded_c_b_ordinary | 11.4785 | 0.432583 | 1.42975 | 0.03732 | 0.3064 |
| matmul_64x32768x64_fc_direct | 60.4786 | 60.9011 | 2.33729 | 1.003 | 25.97 |
| matmul_64x32768x64_fc_prepared | 61.369 | 60.2146 | 2.33146 | 0.9811 | 25.91 |

## Exact heap and separate RSS differences

| Metric | New−old min B | Median B | Max B | Positive cases / 51 |
|---|---:|---:|---:|---:|
| heap.cold_peak_bytes | 0 | 0 | 0 | 0 |
| heap.warm_peak_bytes | 0 | 0 | 0 | 0 |
| heap.cold_allocated_bytes | 0 | 0 | 0 | 0 |
| heap.warm_11_calls_allocated_bytes | 0 | 0 | 0 | 0 |
| rss.rss_baseline | -3751936 | 16384 | 2031616 | 26 |
| rss.rss_peak | -51937280 | 98304 | 26689536 | 27 |
| rss.rss_delta | -50511872 | -32768 | 25083904 | 19 |
| rss.rss_idle_delta | -65536 | 0 | 65536 | 3 |

| Backend | Cold heap peak range B | Warm heap peak range B | RSS total peak range B | RSS request delta range B |
|---|---:|---:|---:|---:|
| old | 14540–16791808 | 15014–33569498 | 40943616–143884288 | 638976–67977216 |
| new | 14540–16791808 | 15014–33569498 | 41041920–143884288 | 622592–59375616 |
| njit | 876–25199552 | 996–33596464 | 126025728–236781568 | 32768–33636352 |

Copy counter mismatches: 0. Direct API counters were not collected and are not inferred as zero.

Repeated-row reuse writes all independent output rows and copies (M−1)×N×8 bytes; this is not zero-copy. Small fixed heap increases remain increases. RSS peak is total resident process memory; RSS delta is the request window peak minus its own baseline. These fields and heap peaks must not be summed or called allocation equivalents. Positive RSS differences remain unresolved measurements; this report does not dismiss them as noise.

All 114 timing and 51 memory rows, exact byte differences, CPU measurements and copy counters are retained in the companion JSON. The original reports remain unchanged. No combined performance/memory acceptance is asserted.
