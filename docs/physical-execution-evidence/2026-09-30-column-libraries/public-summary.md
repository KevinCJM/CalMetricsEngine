# Final F-column reductions and standardization public comparison

All 72 fixed cases retained;12 balanced rounds, CPU1, all independent sum/mean/variance/standardization outputs. Frozen NJIT standardization returns F-order for F input; native protocol returns C-order, with no extra packing charged to NJIT. Bootstrap uses paired rounds as clusters and seed20260930. Counts are not a complete engine acceptance.

| Case | Old ms | New ms | NJIT ms | New/old paired | New/NJIT paired |
|---|---:|---:|---:|---:|---:|
| sum_axis_16x8_f_ordinary | 0.005708 | 0.005916 | 0.000792 | 1.0002 | 6.8114 |
| sum_axis_31x64_f_ordinary | 0.005875 | 0.005750 | 0.001084 | 0.9408 | 5.1222 |
| sum_axis_32x64_f_ordinary | 0.005729 | 0.005563 | 0.001125 | 0.9444 | 4.8590 |
| sum_axis_252x8_f_ordinary | 0.006229 | 0.005666 | 0.001813 | 0.9181 | 3.1052 |
| sum_axis_2048x8_f_ordinary | 0.029937 | 0.019480 | 0.017958 | 0.6369 | 1.0216 |
| sum_axis_32768x8_f_ordinary | 0.278083 | 0.073646 | 0.264042 | 0.2646 | 0.2695 |
| sum_axis_2048x64_f_ordinary | 0.136875 | 0.045791 | 0.126959 | 0.3346 | 0.3670 |
| sum_axis_32768x64_f_ordinary | 2.536646 | 0.697500 | 2.333416 | 0.3114 | 0.2982 |
| sum_axis_8192x128_f_ordinary | 1.096542 | 0.332792 | 1.045875 | 0.3095 | 0.3193 |
| sum_axis_32768x128_f_ordinary | 4.125042 | 0.987750 | 4.103542 | 0.2397 | 0.2384 |
| sum_axis_8192x256_f_ordinary | 2.597958 | 0.661917 | 2.288792 | 0.2670 | 0.3126 |
| sum_axis_2048x9_f_ordinary | 0.048125 | 0.054605 | 0.028750 | 0.8865 | 1.3878 |
| sum_axis_2048x65_f_ordinary | 0.218354 | 0.297937 | 0.179333 | 1.0577 | 1.2930 |
| sum_axis_2048x8_c_ordinary | 0.036146 | 0.026354 | 0.022958 | 1.0001 | 1.2003 |
| sum_axis_32768x64_c_ordinary | 3.573062 | 3.489646 | 4.927521 | 0.9940 | 0.7091 |
| sum_axis_32768x64_reverse_ordinary | 2.132292 | 2.156208 | 2.119063 | 1.0108 | 1.0156 |
| sum_axis_32768x64_f_direct | 2.084208 | 0.532875 | 2.071833 | 0.2442 | 0.2463 |
| sum_axis_32768x64_f_prepared | 2.087687 | 0.566979 | 2.093250 | 0.2737 | 0.2645 |
| mean_axis_16x8_f_ordinary | 0.005250 | 0.005104 | 0.000833 | 0.9875 | 6.2884 |
| mean_axis_31x64_f_ordinary | 0.005542 | 0.005812 | 0.001229 | 1.0546 | 4.5582 |
| mean_axis_32x64_f_ordinary | 0.005500 | 0.005458 | 0.001125 | 0.9678 | 4.8800 |
| mean_axis_252x8_f_ordinary | 0.006187 | 0.005542 | 0.001917 | 0.8746 | 2.8788 |
| mean_axis_2048x8_f_ordinary | 0.027875 | 0.015354 | 0.017375 | 0.5154 | 0.8787 |
| mean_axis_32768x8_f_ordinary | 0.269479 | 0.063354 | 0.257896 | 0.2383 | 0.2474 |
| mean_axis_2048x64_f_ordinary | 0.133167 | 0.035416 | 0.123708 | 0.2668 | 0.2836 |
| mean_axis_32768x64_f_ordinary | 2.152000 | 0.593417 | 2.113563 | 0.2709 | 0.2659 |
| mean_axis_8192x128_f_ordinary | 1.043980 | 0.250416 | 1.018083 | 0.2436 | 0.2460 |
| mean_axis_32768x128_f_ordinary | 4.146729 | 0.946125 | 4.161354 | 0.2267 | 0.2298 |
| mean_axis_8192x256_f_ordinary | 2.158771 | 0.541937 | 2.106520 | 0.2538 | 0.2542 |
| mean_axis_2048x9_f_ordinary | 0.032709 | 0.020855 | 0.020708 | 0.6054 | 1.0672 |
| mean_axis_2048x65_f_ordinary | 0.141980 | 0.045000 | 0.129375 | 0.3231 | 0.3493 |
| mean_axis_2048x8_c_ordinary | 0.022583 | 0.019145 | 0.019605 | 0.9324 | 0.9133 |
| mean_axis_32768x64_c_ordinary | 3.640813 | 3.510646 | 5.047563 | 0.9539 | 0.6987 |
| mean_axis_32768x64_reverse_ordinary | 2.145958 | 2.137542 | 2.122666 | 0.9845 | 1.0073 |
| mean_axis_32768x64_f_direct | 2.116458 | 0.537396 | 2.094417 | 0.2528 | 0.2550 |
| mean_axis_32768x64_f_prepared | 2.114021 | 0.596729 | 2.128104 | 0.2855 | 0.2771 |
| variance_axis_16x8_f_ordinary | 0.005208 | 0.005437 | 0.000791 | 1.0125 | 6.8674 |
| variance_axis_31x64_f_ordinary | 0.007354 | 0.007333 | 0.002250 | 0.9835 | 3.3183 |
| variance_axis_32x64_f_ordinary | 0.007167 | 0.006416 | 0.002292 | 0.8940 | 2.7954 |
| variance_axis_252x8_f_ordinary | 0.008688 | 0.006750 | 0.003875 | 0.7680 | 1.7221 |
| variance_axis_2048x8_f_ordinary | 0.041438 | 0.019813 | 0.033500 | 0.4714 | 0.5846 |
| variance_axis_32768x8_f_ordinary | 0.534208 | 0.194250 | 0.520250 | 0.3685 | 0.3731 |
| variance_axis_2048x64_f_ordinary | 0.271083 | 0.104729 | 0.258209 | 0.3863 | 0.3847 |
| variance_axis_32768x64_f_ordinary | 4.123479 | 1.466438 | 4.082478 | 0.3588 | 0.3557 |
| variance_axis_8192x128_f_ordinary | 2.063771 | 0.696646 | 2.052896 | 0.3355 | 0.3432 |
| variance_axis_32768x128_f_ordinary | 8.167938 | 2.760187 | 8.078938 | 0.3421 | 0.3447 |
| variance_axis_8192x256_f_ordinary | 4.110938 | 1.392708 | 4.043105 | 0.3394 | 0.3445 |
| variance_axis_2048x9_f_ordinary | 0.052000 | 0.031187 | 0.038812 | 0.5803 | 0.7941 |
| variance_axis_2048x65_f_ordinary | 0.276187 | 0.105499 | 0.259583 | 0.3902 | 0.4039 |
| variance_axis_2048x8_c_ordinary | 0.021771 | 0.020376 | 0.036021 | 0.9941 | 0.5648 |
| variance_axis_32768x64_c_ordinary | 5.804583 | 5.776542 | 9.386729 | 0.9945 | 0.6159 |
| variance_axis_32768x64_reverse_ordinary | 4.183583 | 4.138041 | 4.242771 | 0.9839 | 0.9762 |
| variance_axis_32768x64_f_direct | 4.119209 | 1.500854 | 4.264334 | 0.3660 | 0.3684 |
| variance_axis_32768x64_f_prepared | 4.223542 | 1.545292 | 4.177834 | 0.3690 | 0.3743 |
| standardize_16x8_f_ordinary | 0.012063 | 0.011750 | 0.001584 | 0.9874 | 7.3117 |
| standardize_31x64_f_ordinary | 0.016271 | 0.016771 | 0.004479 | 1.0270 | 3.7487 |
| standardize_32x64_f_ordinary | 0.016084 | 0.015437 | 0.004458 | 0.9288 | 3.4013 |
| standardize_252x8_f_ordinary | 0.019605 | 0.014084 | 0.005958 | 0.6908 | 2.3719 |
| standardize_2048x8_f_ordinary | 0.107604 | 0.045791 | 0.041897 | 0.4293 | 1.1210 |
| standardize_32768x8_f_ordinary | 1.526833 | 0.671480 | 0.813979 | 0.4414 | 0.8331 |
| standardize_2048x64_f_ordinary | 0.743583 | 0.269979 | 0.307875 | 0.3718 | 0.8374 |
| standardize_32768x64_f_ordinary | 13.354126 | 4.794895 | 4.535188 | 0.3567 | 1.0497 |
| standardize_8192x128_f_ordinary | 6.050209 | 2.897229 | 3.138333 | 0.4801 | 0.9246 |
| standardize_32768x128_f_ordinary | 23.855313 | 12.901312 | 12.039667 | 0.5473 | 1.0627 |
| standardize_8192x256_f_ordinary | 13.254646 | 4.376542 | 4.442459 | 0.3340 | 1.0072 |
| standardize_2048x9_f_ordinary | 0.149500 | 0.056542 | 0.046375 | 0.5099 | 1.2763 |
| standardize_2048x65_f_ordinary | 0.581354 | 0.277770 | 0.293874 | 0.4106 | 0.8244 |
| standardize_2048x8_c_ordinary | 0.060750 | 0.083334 | 0.056104 | 1.2718 | 1.3020 |
| standardize_32768x64_c_ordinary | 9.273521 | 9.227937 | 18.867875 | 0.9946 | 0.4880 |
| standardize_32768x64_reverse_ordinary | 13.855292 | 12.969625 | 9.017896 | 0.9337 | 1.4465 |
| standardize_32768x64_f_direct | 15.367667 | 9.640458 | 4.451334 | 0.6201 | 2.1553 |
| standardize_32768x64_f_prepared | 11.132875 | 6.337167 | 4.477250 | 0.5745 | 1.4122 |

Heap/RSS separate fresh children; six heap stages and seven counters compared. RSS delta is request increment, not total process RSS.

| Case | Old/New/NJIT warm heap B | New-old RSS peak B | New-old RSS delta B |
|---|---:|---:|---:|
| sum_axis_32768x64_f_ordinary | 15276 / 15276 / 1420 | 655360 | 16384 |
| sum_axis_32768x128_f_ordinary | 16300 / 16300 / 2444 | 835584 | -16384 |
| mean_axis_32768x64_f_ordinary | 15276 / 15276 / 1420 | -2064384 | 49152 |
| mean_axis_32768x128_f_ordinary | 16300 / 16300 / 2444 | 540672 | -32768 |
| variance_axis_32768x64_f_ordinary | 15276 / 15276 / 1420 | -163840 | 81920 |
| variance_axis_32768x128_f_ordinary | 16300 / 16300 / 2444 | 311296 | 81920 |
| standardize_32768x64_f_ordinary | 33584650 / 33584650 / 33557356 | 737280 | -16384 |
| standardize_32768x128_f_ordinary | 67142218 / 67142218 / 67113836 | -573440 | 32768 |
