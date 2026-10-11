# Final F-column reductions and standardization public comparison

All 72 fixed cases retained;12 balanced rounds, CPU1, all independent sum/mean/variance/standardization outputs. Frozen NJIT standardization returns F-order for F input; native protocol returns C-order, with no extra packing charged to NJIT. Bootstrap uses paired rounds as clusters and seed20260930. Counts are not a complete engine acceptance.

| Case | Old ms | New ms | NJIT ms | New/old paired | New/NJIT paired |
|---|---:|---:|---:|---:|---:|
| sum_axis_16x8_f_ordinary | 0.005875 | 0.006104 | 0.000792 | 1.0527 | 6.9382 |
| sum_axis_31x64_f_ordinary | 0.006188 | 0.005979 | 0.001125 | 0.9194 | 5.2454 |
| sum_axis_32x64_f_ordinary | 0.005562 | 0.005834 | 0.001125 | 1.0479 | 4.9807 |
| sum_axis_252x8_f_ordinary | 0.005583 | 0.005500 | 0.001854 | 1.0001 | 2.8981 |
| sum_axis_2048x8_f_ordinary | 0.017188 | 0.017583 | 0.020042 | 0.9380 | 0.8692 |
| sum_axis_32768x8_f_ordinary | 0.074625 | 0.080292 | 0.269042 | 0.9970 | 0.2931 |
| sum_axis_2048x64_f_ordinary | 0.041604 | 0.038125 | 0.125416 | 0.9325 | 0.3008 |
| sum_axis_32768x64_f_ordinary | 0.540063 | 0.538000 | 2.149958 | 1.0332 | 0.2569 |
| sum_axis_8192x128_f_ordinary | 0.270896 | 0.259896 | 1.025000 | 0.9569 | 0.2519 |
| sum_axis_32768x128_f_ordinary | 1.058604 | 1.031645 | 4.395271 | 0.9827 | 0.2356 |
| sum_axis_8192x256_f_ordinary | 0.548375 | 0.551354 | 2.054396 | 1.0207 | 0.2655 |
| sum_axis_2048x9_f_ordinary | 0.018937 | 0.017708 | 0.020271 | 0.9907 | 0.8565 |
| sum_axis_2048x65_f_ordinary | 0.043396 | 0.045646 | 0.125271 | 1.0102 | 0.3496 |
| sum_axis_2048x8_c_ordinary | 0.017479 | 0.018271 | 0.018687 | 1.0167 | 0.9970 |
| sum_axis_32768x64_c_ordinary | 3.402750 | 3.401771 | 4.738041 | 0.9895 | 0.6822 |
| sum_axis_32768x64_reverse_ordinary | 2.153062 | 2.156479 | 2.142230 | 0.9902 | 1.0028 |
| sum_axis_32768x64_f_direct | 0.475458 | 0.505917 | 2.117271 | 1.0848 | 0.2389 |
| sum_axis_32768x64_f_prepared | 0.510229 | 0.488812 | 2.050479 | 0.9884 | 0.2376 |
| mean_axis_16x8_f_ordinary | 0.004896 | 0.005167 | 0.000750 | 1.0260 | 6.6845 |
| mean_axis_31x64_f_ordinary | 0.005354 | 0.005605 | 0.001167 | 1.0419 | 4.8065 |
| mean_axis_32x64_f_ordinary | 0.005125 | 0.005250 | 0.001125 | 1.0253 | 4.6462 |
| mean_axis_252x8_f_ordinary | 0.005229 | 0.005375 | 0.001833 | 1.0200 | 2.9481 |
| mean_axis_2048x8_f_ordinary | 0.015854 | 0.015791 | 0.017604 | 0.9929 | 0.9007 |
| mean_axis_32768x8_f_ordinary | 0.069417 | 0.068375 | 0.251250 | 1.0521 | 0.2716 |
| mean_axis_2048x64_f_ordinary | 0.035916 | 0.037041 | 0.124083 | 0.9905 | 0.2979 |
| mean_axis_32768x64_f_ordinary | 0.491208 | 0.502063 | 2.049750 | 0.9696 | 0.2424 |
| mean_axis_8192x128_f_ordinary | 0.247562 | 0.235313 | 0.996354 | 0.9743 | 0.2282 |
| mean_axis_32768x128_f_ordinary | 1.131458 | 1.093770 | 4.227500 | 0.9493 | 0.2574 |
| mean_axis_8192x256_f_ordinary | 0.539791 | 0.565146 | 2.063499 | 1.0128 | 0.2632 |
| mean_axis_2048x9_f_ordinary | 0.013729 | 0.014771 | 0.019042 | 1.0381 | 0.7675 |
| mean_axis_2048x65_f_ordinary | 0.052562 | 0.051625 | 0.125625 | 1.0792 | 0.4079 |
| mean_axis_2048x8_c_ordinary | 0.021021 | 0.016792 | 0.019333 | 0.8590 | 0.8449 |
| mean_axis_32768x64_c_ordinary | 3.515063 | 3.566937 | 4.677874 | 1.0174 | 0.8165 |
| mean_axis_32768x64_reverse_ordinary | 2.095041 | 2.108291 | 2.015687 | 1.0059 | 1.0401 |
| mean_axis_32768x64_f_direct | 0.530687 | 0.504958 | 2.153250 | 1.0316 | 0.2345 |
| mean_axis_32768x64_f_prepared | 0.510208 | 0.511749 | 2.032833 | 1.0100 | 0.2484 |
| variance_axis_16x8_f_ordinary | 0.005333 | 0.005374 | 0.000812 | 0.9838 | 7.0222 |
| variance_axis_31x64_f_ordinary | 0.007188 | 0.007188 | 0.002270 | 0.9767 | 3.1026 |
| variance_axis_32x64_f_ordinary | 0.006729 | 0.006458 | 0.002292 | 1.0095 | 2.8704 |
| variance_axis_252x8_f_ordinary | 0.006125 | 0.006333 | 0.003833 | 1.0319 | 1.6230 |
| variance_axis_2048x8_f_ordinary | 0.022958 | 0.022812 | 0.033542 | 0.9464 | 0.6588 |
| variance_axis_32768x8_f_ordinary | 0.191979 | 0.196271 | 0.523000 | 1.0421 | 0.3785 |
| variance_axis_2048x64_f_ordinary | 0.092105 | 0.092896 | 0.248729 | 1.0037 | 0.3718 |
| variance_axis_32768x64_f_ordinary | 1.472979 | 1.466041 | 4.115250 | 0.9953 | 0.3523 |
| variance_axis_8192x128_f_ordinary | 0.705375 | 0.696584 | 2.062375 | 0.9905 | 0.3410 |
| variance_axis_32768x128_f_ordinary | 2.810417 | 2.810395 | 8.055646 | 0.9961 | 0.3505 |
| variance_axis_8192x256_f_ordinary | 1.411125 | 1.375729 | 4.079250 | 0.9894 | 0.3367 |
| variance_axis_2048x9_f_ordinary | 0.024729 | 0.024000 | 0.037562 | 0.9892 | 0.6524 |
| variance_axis_2048x65_f_ordinary | 0.102979 | 0.098521 | 0.259250 | 0.9642 | 0.3802 |
| variance_axis_2048x8_c_ordinary | 0.018999 | 0.016875 | 0.035042 | 0.9548 | 0.4837 |
| variance_axis_32768x64_c_ordinary | 5.744001 | 5.705292 | 9.310645 | 0.9968 | 0.6098 |
| variance_axis_32768x64_reverse_ordinary | 4.153625 | 4.110792 | 4.107104 | 0.9950 | 1.0062 |
| variance_axis_32768x64_f_direct | 1.448792 | 1.483021 | 4.164416 | 1.0024 | 0.3539 |
| variance_axis_32768x64_f_prepared | 1.454624 | 1.471958 | 4.103312 | 1.0060 | 0.3589 |
| standardize_16x8_f_ordinary | 0.011250 | 0.011562 | 0.001583 | 1.0168 | 7.0243 |
| standardize_31x64_f_ordinary | 0.015917 | 0.016459 | 0.004375 | 1.0116 | 3.8041 |
| standardize_32x64_f_ordinary | 0.014875 | 0.014375 | 0.004375 | 0.9723 | 3.3513 |
| standardize_252x8_f_ordinary | 0.014313 | 0.013812 | 0.005875 | 0.9580 | 2.3385 |
| standardize_2048x8_f_ordinary | 0.062584 | 0.045896 | 0.040584 | 0.8422 | 1.1176 |
| standardize_32768x8_f_ordinary | 0.624812 | 0.579167 | 0.774979 | 0.9179 | 0.7287 |
| standardize_2048x64_f_ordinary | 0.290375 | 0.202833 | 0.351959 | 0.7427 | 0.6059 |
| standardize_32768x64_f_ordinary | 3.968667 | 3.819021 | 4.493229 | 0.9657 | 0.8443 |
| standardize_8192x128_f_ordinary | 3.042813 | 2.677271 | 3.151188 | 0.8693 | 0.8416 |
| standardize_32768x128_f_ordinary | 13.879855 | 11.812771 | 12.068646 | 0.8597 | 0.9630 |
| standardize_8192x256_f_ordinary | 5.498709 | 4.197667 | 4.534980 | 0.7608 | 0.9240 |
| standardize_2048x9_f_ordinary | 0.045145 | 0.056104 | 0.040750 | 1.2222 | 1.3302 |
| standardize_2048x65_f_ordinary | 0.233875 | 0.302104 | 0.293250 | 1.1092 | 1.0247 |
| standardize_2048x8_c_ordinary | 0.060833 | 0.059250 | 0.052146 | 1.0189 | 1.0214 |
| standardize_32768x64_c_ordinary | 9.390729 | 9.253749 | 19.693770 | 0.9710 | 0.4694 |
| standardize_32768x64_reverse_ordinary | 12.985979 | 14.046458 | 9.278583 | 1.0856 | 1.5121 |
| standardize_32768x64_f_direct | 8.379000 | 9.606604 | 4.467896 | 1.1403 | 2.1539 |
| standardize_32768x64_f_prepared | 6.207521 | 5.818771 | 4.466563 | 0.9325 | 1.2876 |

Heap/RSS separate fresh children; six heap stages and seven counters compared. RSS delta is request increment, not total process RSS.

| Case | Old/New/NJIT warm heap B | New-old RSS peak B | New-old RSS delta B |
|---|---:|---:|---:|
| sum_axis_32768x64_f_ordinary | 15276 / 15276 / 1420 | 0 | 16384 |
| sum_axis_32768x128_f_ordinary | 16300 / 16300 / 2444 | -966656 | 16384 |
| mean_axis_32768x64_f_ordinary | 15276 / 15276 / 1420 | -262144 | 32768 |
| mean_axis_32768x128_f_ordinary | 16300 / 16300 / 2444 | 1425408 | 0 |
| variance_axis_32768x64_f_ordinary | 15276 / 15276 / 1420 | -212992 | -16384 |
| variance_axis_32768x128_f_ordinary | 16300 / 16300 / 2444 | 49152 | -16384 |
| standardize_32768x64_f_ordinary | 33584650 / 33584650 / 33557356 | -278528 | -49152 |
| standardize_32768x128_f_ordinary | 67142218 / 67142218 / 67113836 | -606208 | -16384 |
