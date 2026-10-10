# Final variance and standardization public comparison

All 34 fixed cases retained;12 balanced rounds, CPU1, all independent variance/standardization outputs. Bootstrap uses paired rounds as clusters and seed20260930. Counts are not a complete engine acceptance.

| Case | Old ms | New ms | NJIT ms | New/old paired | New/NJIT paired |
|---|---:|---:|---:|---:|---:|
| variance_axis_16x8_c_ordinary | 0.005938 | 0.005916 | 0.000792 | 0.9576 | 7.2644 |
| variance_axis_31x64_c_ordinary | 0.006750 | 0.006875 | 0.002708 | 1.0261 | 2.5386 |
| variance_axis_32x64_c_ordinary | 0.006729 | 0.005979 | 0.002833 | 0.8805 | 2.1121 |
| variance_axis_252x8_c_ordinary | 0.006667 | 0.005916 | 0.004333 | 0.8979 | 1.3658 |
| variance_axis_2048x8_c_ordinary | 0.019896 | 0.015125 | 0.033834 | 0.7260 | 0.4347 |
| variance_axis_32768x8_c_ordinary | 0.189917 | 0.117813 | 0.523354 | 0.6040 | 0.2212 |
| variance_axis_2048x64_c_ordinary | 0.121979 | 0.084895 | 0.259124 | 0.7152 | 0.3197 |
| variance_axis_32768x64_c_ordinary | 6.056417 | 5.621729 | 8.692709 | 0.9310 | 0.6413 |
| variance_axis_8192x128_c_ordinary | 1.143021 | 0.772895 | 2.204791 | 0.6550 | 0.3416 |
| variance_axis_32768x128_c_ordinary | 7.053646 | 5.351417 | 20.407854 | 0.7536 | 0.2611 |
| variance_axis_8192x256_c_ordinary | 4.250833 | 2.868333 | 9.199521 | 0.6785 | 0.3096 |
| variance_axis_2048x9_c_ordinary | 0.026520 | 0.023854 | 0.038750 | 0.9374 | 0.6160 |
| variance_axis_2048x65_c_ordinary | 0.121459 | 0.110187 | 0.262105 | 0.9042 | 0.4146 |
| variance_axis_32768x64_f_ordinary | 4.118604 | 4.168583 | 4.074667 | 1.0126 | 1.0254 |
| variance_axis_32768x64_reverse_ordinary | 9.436397 | 9.470396 | 8.802167 | 1.0005 | 1.0741 |
| variance_axis_32768x64_c_direct | 6.033396 | 5.594792 | 9.006500 | 0.9416 | 0.6246 |
| variance_axis_32768x64_c_prepared | 6.161667 | 5.729166 | 9.269063 | 0.9160 | 0.6281 |
| standardize_16x8_c_ordinary | 0.013166 | 0.013251 | 0.001729 | 0.9950 | 7.7605 |
| standardize_31x64_c_ordinary | 0.015834 | 0.016146 | 0.005146 | 1.0287 | 3.1097 |
| standardize_32x64_c_ordinary | 0.015938 | 0.015562 | 0.005791 | 0.9752 | 2.6519 |
| standardize_252x8_c_ordinary | 0.017125 | 0.016166 | 0.006375 | 0.9309 | 2.4699 |
| standardize_2048x8_c_ordinary | 0.068667 | 0.064542 | 0.057105 | 0.9531 | 1.0955 |
| standardize_32768x8_c_ordinary | 0.933625 | 0.838896 | 1.090291 | 0.9069 | 0.7558 |
| standardize_2048x64_c_ordinary | 0.334021 | 0.359916 | 0.473354 | 1.0747 | 0.7063 |
| standardize_32768x64_c_ordinary | 10.716729 | 9.685083 | 19.412479 | 0.9074 | 0.4911 |
| standardize_8192x128_c_ordinary | 3.269021 | 3.177479 | 6.777938 | 0.9628 | 0.4712 |
| standardize_32768x128_c_ordinary | 16.508000 | 14.900688 | 45.342438 | 0.8988 | 0.3276 |
| standardize_8192x256_c_ordinary | 6.808021 | 5.663458 | 19.987979 | 0.8302 | 0.2841 |
| standardize_2048x9_c_ordinary | 0.087105 | 0.117479 | 0.064959 | 1.0733 | 1.3919 |
| standardize_2048x65_c_ordinary | 0.304500 | 0.284229 | 0.507812 | 0.9647 | 0.6590 |
| standardize_32768x64_f_ordinary | 10.006667 | 13.322750 | 4.532501 | 1.3397 | 2.9571 |
| standardize_32768x64_reverse_ordinary | 15.911375 | 15.573708 | 19.824687 | 0.9844 | 0.7904 |
| standardize_32768x64_c_direct | 13.125958 | 12.369000 | 19.575250 | 0.9482 | 0.6355 |
| standardize_32768x64_c_prepared | 14.120916 | 13.237688 | 23.118688 | 0.9580 | 0.5815 |

Heap/RSS separate fresh children; six heap stages and seven counters compared. RSS delta is request increment, not total process RSS.

| Case | Old/New/NJIT warm heap B | New-old RSS peak B | New-old RSS delta B |
|---|---:|---:|---:|
| variance_axis_32768x8_c_ordinary | 14380 / 14380 / 524 | 2277376 | 16384 |
| variance_axis_32768x64_c_ordinary | 15276 / 15276 / 1420 | 212992 | -49152 |
| variance_axis_32768x128_c_ordinary | 16300 / 16300 / 2444 | 540672 | -49152 |
| standardize_32768x8_c_ordinary | 4221778 / 4221778 / 4195436 | 2670592 | 4210688 |
| standardize_32768x64_c_ordinary | 33584650 / 33584650 / 33557356 | -65536 | -32768 |
| standardize_32768x128_c_ordinary | 67142218 / 67142218 / 67113836 | 1179648 | -49152 |
