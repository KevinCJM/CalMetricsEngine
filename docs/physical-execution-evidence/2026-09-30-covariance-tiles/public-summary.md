# Public covariance/correlation evidence summary

108 time cases, old `6ff48aac8a819ad1854a33d51cd76221d902bcc7ede4e293860e9e7f86a30dd0`, new `63090c5a7971acd8c947ef5b0e20a46bf207f88772e8c322ab94a3537ecd96bb`. Memory:36cases/216children, not108case coverage.

| Group | Cases | New/old range | Faster/slower old | New/NJIT range | Faster/slower NJIT |
|---|---:|---:|---:|---:|---:|
| all | 108 | 0.3701–1.0386 | 93/15 | 0.5539–6.6629 | 18/90 |
| layout=c | 27 | 0.3823–1.0297 | 24/3 | 0.5887–4.1954 | 9/18 |
| workload=small | 48 | 0.8966–1.0386 | 33/15 | 1.4919–4.1954 | 0/48 |
| request=covariance | 36 | 0.3759–1.0131 | 31/5 | 0.5539–6.6629 | 6/30 |
| roots=1 | 72 | 0.3701–1.0240 | 63/9 | 0.5539–6.6629 | 12/60 |
| c/small/roots=1 | 8 | 0.8966–1.0097 | 7/1 | 1.5706–3.9502 | 0/8 |
| request=correlation | 36 | 0.3701–1.0240 | 32/4 | 0.5814–3.2411 | 6/30 |
| request=joint | 36 | 0.3851–1.0386 | 30/6 | 0.5887–5.1309 | 6/30 |
| roots=2 | 36 | 0.3851–1.0386 | 30/6 | 0.5887–5.1309 | 6/30 |
| c/small/roots=2 | 4 | 0.9307–1.0297 | 2/2 | 3.1153–4.1954 | 0/4 |
| layout=f | 27 | 0.7038–1.0386 | 23/4 | 1.3026–6.6629 | 0/27 |
| f/small/roots=1 | 8 | 0.9211–1.0131 | 5/3 | 1.5913–3.8832 | 0/8 |
| f/small/roots=2 | 4 | 0.9668–1.0386 | 3/1 | 3.3416–4.0395 | 0/4 |
| layout=padded_c | 27 | 0.3701–1.0384 | 22/5 | 0.5539–4.1535 | 9/18 |
| padded_c/small/roots=1 | 8 | 0.9149–1.0240 | 5/3 | 1.4919–3.6900 | 0/8 |
| padded_c/small/roots=2 | 4 | 0.9476–1.0384 | 2/2 | 3.1453–4.1535 | 0/4 |
| layout=padded_f | 27 | 0.7018–1.0104 | 24/3 | 1.2076–5.4109 | 0/27 |
| padded_f/small/roots=1 | 8 | 0.9098–1.0104 | 6/2 | 1.6895–3.6799 | 0/8 |
| padded_f/small/roots=2 | 4 | 0.9676–1.0078 | 3/1 | 3.3501–3.9610 | 0/4 |
| workload=long | 36 | 0.4399–0.8711 | 36/0 | 0.5539–3.5536 | 18/18 |
| c/long/roots=1 | 6 | 0.4416–0.5238 | 6/0 | 0.6170–0.8462 | 6/0 |
| c/long/roots=2 | 3 | 0.4399–0.5301 | 3/0 | 0.5887–0.8341 | 3/0 |
| f/long/roots=1 | 6 | 0.7050–0.8711 | 6/0 | 1.3026–3.5536 | 0/6 |
| f/long/roots=2 | 3 | 0.7038–0.8255 | 3/0 | 1.3861–3.4953 | 0/3 |
| padded_c/long/roots=1 | 6 | 0.4540–0.5256 | 6/0 | 0.5539–0.8970 | 6/0 |
| padded_c/long/roots=2 | 3 | 0.4523–0.5240 | 3/0 | 0.5967–0.9016 | 3/0 |
| padded_f/long/roots=1 | 6 | 0.7057–0.7657 | 6/0 | 1.2076–3.2280 | 0/6 |
| padded_f/long/roots=2 | 3 | 0.7018–0.7237 | 3/0 | 1.3647–3.0191 | 0/3 |
| workload=wide | 24 | 0.3701–0.8908 | 24/0 | 1.0148–6.6629 | 0/24 |
| c/wide/roots=1 | 4 | 0.3823–0.4011 | 4/0 | 1.0148–2.6161 | 0/4 |
| c/wide/roots=2 | 2 | 0.3958–0.4101 | 2/0 | 1.3419–2.6551 | 0/2 |
| f/wide/roots=1 | 4 | 0.8009–0.8873 | 4/0 | 2.2130–6.6629 | 0/4 |
| f/wide/roots=2 | 2 | 0.8033–0.8908 | 2/0 | 2.8733–4.8551 | 0/2 |
| padded_c/wide/roots=1 | 4 | 0.3701–0.3965 | 4/0 | 1.1174–2.6727 | 0/4 |
| padded_c/wide/roots=2 | 2 | 0.3851–0.4050 | 2/0 | 1.2510–2.4770 | 0/2 |
| padded_f/wide/roots=1 | 4 | 0.7806–0.8738 | 4/0 | 1.6872–5.4109 | 0/4 |
| padded_f/wide/roots=2 | 2 | 0.7962–0.8760 | 2/0 | 2.5211–5.1309 | 0/2 |

## All absolute case medians and paired ratios

| Case | Old ms | New ms | NJIT ms | New−old ms | Paired new/old | Paired new/NJIT |
|---|---:|---:|---:|---:|---:|---:|
| covariance_31x8_c | 0.007020 | 0.006916 | 0.001770 | -0.000104 | 0.9940 | 3.9502 |
| correlation_31x8_c | 0.006583 | 0.006625 | 0.003250 | +0.000041 | 1.0097 | 2.0134 |
| joint_31x8_c | 0.008958 | 0.009229 | 0.002270 | +0.000271 | 1.0297 | 4.0358 |
| covariance_31x8_f | 0.006396 | 0.006479 | 0.001666 | +0.000084 | 1.0127 | 3.8832 |
| correlation_31x8_f | 0.006855 | 0.006605 | 0.003375 | -0.000250 | 0.9296 | 1.9296 |
| joint_31x8_f | 0.009041 | 0.008833 | 0.002167 | -0.000208 | 0.9880 | 3.9619 |
| covariance_31x8_padded_c | 0.006666 | 0.006791 | 0.001834 | +0.000125 | 0.9811 | 3.6900 |
| correlation_31x8_padded_c | 0.007208 | 0.007541 | 0.003792 | +0.000333 | 1.0086 | 2.0087 |
| joint_31x8_padded_c | 0.009041 | 0.009416 | 0.002375 | +0.000375 | 1.0261 | 3.8425 |
| covariance_31x8_padded_f | 0.006188 | 0.006250 | 0.001750 | +0.000063 | 1.0104 | 3.5951 |
| correlation_31x8_padded_f | 0.007125 | 0.006896 | 0.003792 | -0.000229 | 0.9819 | 1.8244 |
| joint_31x8_padded_f | 0.009542 | 0.009646 | 0.002708 | +0.000104 | 0.9907 | 3.5036 |
| covariance_32x7_c | 0.006750 | 0.006500 | 0.001708 | -0.000250 | 0.9816 | 3.7610 |
| correlation_32x7_c | 0.006208 | 0.006229 | 0.002833 | +0.000021 | 0.9897 | 2.2017 |
| joint_32x7_c | 0.008562 | 0.008667 | 0.002042 | +0.000105 | 1.0049 | 4.1954 |
| covariance_32x7_f | 0.005896 | 0.005979 | 0.001667 | +0.000083 | 1.0131 | 3.5769 |
| correlation_32x7_f | 0.005812 | 0.005750 | 0.002750 | -0.000063 | 1.0074 | 2.0837 |
| joint_32x7_f | 0.008105 | 0.008396 | 0.002125 | +0.000292 | 1.0386 | 4.0395 |
| covariance_32x7_padded_c | 0.006104 | 0.006146 | 0.001667 | +0.000042 | 1.0036 | 3.6409 |
| correlation_32x7_padded_c | 0.006083 | 0.006313 | 0.002875 | +0.000230 | 1.0240 | 2.1859 |
| joint_32x7_padded_c | 0.008771 | 0.009021 | 0.002208 | +0.000250 | 1.0384 | 4.1535 |
| covariance_32x7_padded_f | 0.006458 | 0.006646 | 0.001855 | +0.000188 | 1.0034 | 3.6799 |
| correlation_32x7_padded_f | 0.006375 | 0.006125 | 0.002854 | -0.000250 | 0.9827 | 2.1534 |
| joint_32x7_padded_f | 0.008709 | 0.008688 | 0.002167 | -0.000021 | 1.0078 | 3.9610 |
| covariance_32x8_c | 0.006333 | 0.005750 | 0.002042 | -0.000583 | 0.9016 | 2.8048 |
| correlation_32x8_c | 0.006437 | 0.005854 | 0.003708 | -0.000583 | 0.8966 | 1.6012 |
| joint_32x8_c | 0.008875 | 0.008396 | 0.002583 | -0.000479 | 0.9307 | 3.2434 |
| covariance_32x8_f | 0.006188 | 0.005917 | 0.001875 | -0.000270 | 0.9546 | 3.2208 |
| correlation_32x8_f | 0.006208 | 0.005917 | 0.003208 | -0.000291 | 0.9452 | 1.8244 |
| joint_32x8_f | 0.008479 | 0.008458 | 0.002312 | -0.000021 | 0.9668 | 3.5927 |
| covariance_32x8_padded_c | 0.005792 | 0.005354 | 0.001792 | -0.000438 | 0.9241 | 2.9888 |
| correlation_32x8_padded_c | 0.006292 | 0.005750 | 0.003875 | -0.000542 | 0.9149 | 1.4919 |
| joint_32x8_padded_c | 0.009125 | 0.008562 | 0.002771 | -0.000563 | 0.9476 | 3.1453 |
| covariance_32x8_padded_f | 0.006208 | 0.005854 | 0.001792 | -0.000354 | 0.9613 | 3.3289 |
| correlation_32x8_padded_f | 0.006167 | 0.005875 | 0.003230 | -0.000292 | 0.9513 | 1.7713 |
| joint_32x8_padded_f | 0.008479 | 0.008250 | 0.002292 | -0.000229 | 0.9793 | 3.5979 |
| covariance_33x9_c | 0.006605 | 0.006125 | 0.002292 | -0.000479 | 0.9461 | 2.6385 |
| correlation_33x9_c | 0.006375 | 0.005875 | 0.003667 | -0.000500 | 0.9183 | 1.5706 |
| joint_33x9_c | 0.008396 | 0.007958 | 0.002521 | -0.000438 | 0.9419 | 3.1153 |
| covariance_33x9_f | 0.006625 | 0.006125 | 0.001937 | -0.000500 | 0.9211 | 3.0610 |
| correlation_33x9_f | 0.007166 | 0.006729 | 0.004271 | -0.000437 | 0.9312 | 1.5913 |
| joint_33x9_f | 0.008646 | 0.008583 | 0.002542 | -0.000063 | 0.9900 | 3.3416 |
| covariance_33x9_padded_c | 0.006063 | 0.005583 | 0.001917 | -0.000479 | 0.9275 | 2.8564 |
| correlation_33x9_padded_c | 0.006083 | 0.005583 | 0.003542 | -0.000499 | 0.9202 | 1.5615 |
| joint_33x9_padded_c | 0.008292 | 0.008000 | 0.002458 | -0.000292 | 0.9492 | 3.2151 |
| covariance_33x9_padded_f | 0.006959 | 0.006417 | 0.002333 | -0.000542 | 0.9098 | 2.7194 |
| correlation_33x9_padded_f | 0.006500 | 0.006188 | 0.003625 | -0.000312 | 0.9507 | 1.6895 |
| joint_33x9_padded_f | 0.008521 | 0.008209 | 0.002458 | -0.000312 | 0.9676 | 3.3501 |
| covariance_32768x8_c | 1.195146 | 0.623646 | 0.978583 | -0.571500 | 0.5207 | 0.6361 |
| correlation_32768x8_c | 1.194749 | 0.626125 | 0.978979 | -0.568624 | 0.5238 | 0.6406 |
| joint_32768x8_c | 1.260188 | 0.663958 | 1.122229 | -0.596229 | 0.5301 | 0.5887 |
| covariance_32768x8_f | 1.733834 | 1.223416 | 0.953750 | -0.510417 | 0.7057 | 1.3026 |
| correlation_32768x8_f | 1.671542 | 1.183209 | 0.858355 | -0.488333 | 0.7050 | 1.3915 |
| joint_32768x8_f | 1.679563 | 1.180459 | 0.852313 | -0.499104 | 0.7038 | 1.3861 |
| covariance_32768x8_padded_c | 1.210562 | 0.646375 | 1.160438 | -0.564187 | 0.5256 | 0.5604 |
| correlation_32768x8_padded_c | 1.204625 | 0.634438 | 1.065605 | -0.570187 | 0.5239 | 0.5894 |
| joint_32768x8_padded_c | 1.241626 | 0.654917 | 1.095563 | -0.586709 | 0.5240 | 0.5979 |
| covariance_32768x8_padded_f | 1.753083 | 1.190103 | 0.953583 | -0.562979 | 0.7064 | 1.2728 |
| correlation_32768x8_padded_f | 1.754729 | 1.238500 | 1.023084 | -0.516229 | 0.7057 | 1.2076 |
| joint_32768x8_padded_f | 1.680750 | 1.180813 | 0.861542 | -0.499937 | 0.7018 | 1.3647 |
| covariance_32768x16_c | 2.716292 | 1.251125 | 2.032334 | -1.465167 | 0.4619 | 0.6170 |
| correlation_32768x16_c | 2.716104 | 1.258895 | 1.971542 | -1.457209 | 0.4627 | 0.6359 |
| joint_32768x16_c | 2.727625 | 1.262459 | 2.038729 | -1.465166 | 0.4622 | 0.6159 |
| covariance_32768x16_f | 3.841979 | 2.727396 | 1.771146 | -1.114583 | 0.7103 | 1.5358 |
| correlation_32768x16_f | 3.867062 | 2.737958 | 1.950834 | -1.129104 | 0.7108 | 1.4468 |
| joint_32768x16_f | 3.838563 | 2.726313 | 1.848000 | -1.112249 | 0.7100 | 1.4935 |
| covariance_32768x16_padded_c | 2.757792 | 1.319687 | 2.455500 | -1.438105 | 0.4783 | 0.5539 |
| correlation_32768x16_padded_c | 2.736584 | 1.288521 | 2.213208 | -1.448063 | 0.4691 | 0.5814 |
| joint_32768x16_padded_c | 2.742459 | 1.290104 | 2.145812 | -1.452355 | 0.4698 | 0.5967 |
| covariance_32768x16_padded_f | 3.980980 | 2.802792 | 1.991021 | -1.178187 | 0.7077 | 1.4391 |
| correlation_32768x16_padded_f | 3.853541 | 2.731375 | 1.894416 | -1.122167 | 0.7085 | 1.4368 |
| joint_32768x16_padded_f | 3.840167 | 2.715917 | 1.793042 | -1.124251 | 0.7075 | 1.5089 |
| covariance_32768x64_c | 20.080770 | 8.837187 | 10.298541 | -11.243583 | 0.4416 | 0.8462 |
| correlation_32768x64_c | 20.355667 | 9.003854 | 10.858895 | -11.351813 | 0.4440 | 0.8240 |
| joint_32768x64_c | 20.117521 | 8.890937 | 10.513563 | -11.226583 | 0.4399 | 0.8341 |
| covariance_32768x64_f | 27.856792 | 24.253208 | 6.867667 | -3.603583 | 0.8711 | 3.5536 |
| correlation_32768x64_f | 27.922729 | 21.845792 | 6.797333 | -6.076938 | 0.7836 | 3.2411 |
| joint_32768x64_f | 27.899771 | 23.039313 | 6.594167 | -4.860459 | 0.8255 | 3.4953 |
| covariance_32768x64_padded_c | 20.389626 | 9.254208 | 10.534771 | -11.135418 | 0.4559 | 0.8907 |
| correlation_32768x64_padded_c | 20.103333 | 9.267105 | 10.198292 | -10.836228 | 0.4540 | 0.8970 |
| joint_32768x64_padded_c | 20.737500 | 9.314958 | 10.244896 | -11.422542 | 0.4523 | 0.9016 |
| covariance_32768x64_padded_f | 27.873375 | 21.257229 | 6.605417 | -6.616146 | 0.7657 | 3.2280 |
| correlation_32768x64_padded_f | 28.154208 | 21.291980 | 6.938041 | -6.862229 | 0.7599 | 3.0161 |
| joint_32768x64_padded_f | 27.918542 | 20.240000 | 6.655979 | -7.678542 | 0.7237 | 3.0191 |
| covariance_512x128_c | 1.324833 | 0.526292 | 0.278875 | -0.798541 | 0.3961 | 1.8891 |
| correlation_512x128_c | 1.356958 | 0.545729 | 0.528709 | -0.811229 | 0.4011 | 1.0148 |
| joint_512x128_c | 1.403812 | 0.572458 | 0.417583 | -0.831354 | 0.4101 | 1.3419 |
| covariance_512x128_f | 1.285208 | 1.016750 | 0.395375 | -0.268458 | 0.8009 | 2.7139 |
| correlation_512x128_f | 1.267771 | 1.022500 | 0.466916 | -0.245271 | 0.8025 | 2.2130 |
| joint_512x128_f | 1.277917 | 1.031292 | 0.358958 | -0.246625 | 0.8033 | 2.8733 |
| covariance_512x128_padded_c | 1.375708 | 0.544812 | 0.359729 | -0.830896 | 0.3965 | 1.4852 |
| correlation_512x128_padded_c | 1.331187 | 0.516042 | 0.463770 | -0.815146 | 0.3885 | 1.1174 |
| joint_512x128_padded_c | 1.386208 | 0.565875 | 0.457417 | -0.820333 | 0.4050 | 1.2510 |
| covariance_512x128_padded_f | 1.361938 | 1.066709 | 0.455437 | -0.295229 | 0.7848 | 2.3644 |
| correlation_512x128_padded_f | 1.368605 | 1.073146 | 0.648188 | -0.295459 | 0.7806 | 1.6872 |
| joint_512x128_padded_f | 1.389250 | 1.104479 | 0.444500 | -0.284771 | 0.7962 | 2.5211 |
| covariance_512x256_c | 5.840770 | 2.278104 | 0.869062 | -3.562666 | 0.3883 | 2.6161 |
| correlation_512x256_c | 5.585938 | 2.128833 | 1.679979 | -3.457104 | 0.3823 | 1.2677 |
| joint_512x256_c | 5.705625 | 2.255167 | 0.844458 | -3.450459 | 0.3958 | 2.6551 |
| covariance_512x256_f | 4.502021 | 4.059500 | 0.607916 | -0.442521 | 0.8868 | 6.6629 |
| correlation_512x256_f | 4.415979 | 3.928167 | 1.713063 | -0.487812 | 0.8873 | 2.2910 |
| joint_512x256_f | 4.612792 | 4.115312 | 0.859333 | -0.497479 | 0.8908 | 4.8551 |
| covariance_512x256_padded_c | 5.712729 | 2.148125 | 0.798104 | -3.564604 | 0.3759 | 2.6727 |
| correlation_512x256_padded_c | 5.579458 | 2.063604 | 1.773833 | -3.515854 | 0.3701 | 1.1711 |
| joint_512x256_padded_c | 5.707624 | 2.188292 | 0.882062 | -3.519332 | 0.3851 | 2.4770 |
| covariance_512x256_padded_f | 4.473854 | 3.913792 | 0.718792 | -0.560062 | 0.8738 | 5.4109 |
| correlation_512x256_padded_f | 4.467791 | 3.874979 | 1.643729 | -0.592812 | 0.8685 | 2.3571 |
| joint_512x256_padded_f | 4.604563 | 4.028354 | 0.796854 | -0.576208 | 0.8760 | 5.1309 |

## All regressions

Against old: 15/108 cases slower by paired median.
- correlation_31x8_c: 1.0097; median latency 0.006625ms.
- joint_31x8_c: 1.0297; median latency 0.009229ms.
- covariance_31x8_f: 1.0127; median latency 0.006479ms.
- correlation_31x8_padded_c: 1.0086; median latency 0.007541ms.
- joint_31x8_padded_c: 1.0261; median latency 0.009416ms.
- covariance_31x8_padded_f: 1.0104; median latency 0.006250ms.
- joint_32x7_c: 1.0049; median latency 0.008667ms.
- covariance_32x7_f: 1.0131; median latency 0.005979ms.
- correlation_32x7_f: 1.0074; median latency 0.005750ms.
- joint_32x7_f: 1.0386; median latency 0.008396ms.
- covariance_32x7_padded_c: 1.0036; median latency 0.006146ms.
- correlation_32x7_padded_c: 1.0240; median latency 0.006313ms.
- joint_32x7_padded_c: 1.0384; median latency 0.009021ms.
- covariance_32x7_padded_f: 1.0034; median latency 0.006646ms.
- joint_32x7_padded_f: 1.0078; median latency 0.008688ms.
Against njit: 90/108 cases slower by paired median.
- covariance_31x8_c: 3.9502; median latency 0.006916ms.
- correlation_31x8_c: 2.0134; median latency 0.006625ms.
- joint_31x8_c: 4.0358; median latency 0.009229ms.
- covariance_31x8_f: 3.8832; median latency 0.006479ms.
- correlation_31x8_f: 1.9296; median latency 0.006605ms.
- joint_31x8_f: 3.9619; median latency 0.008833ms.
- covariance_31x8_padded_c: 3.6900; median latency 0.006791ms.
- correlation_31x8_padded_c: 2.0087; median latency 0.007541ms.
- joint_31x8_padded_c: 3.8425; median latency 0.009416ms.
- covariance_31x8_padded_f: 3.5951; median latency 0.006250ms.
- correlation_31x8_padded_f: 1.8244; median latency 0.006896ms.
- joint_31x8_padded_f: 3.5036; median latency 0.009646ms.
- covariance_32x7_c: 3.7610; median latency 0.006500ms.
- correlation_32x7_c: 2.2017; median latency 0.006229ms.
- joint_32x7_c: 4.1954; median latency 0.008667ms.
- covariance_32x7_f: 3.5769; median latency 0.005979ms.
- correlation_32x7_f: 2.0837; median latency 0.005750ms.
- joint_32x7_f: 4.0395; median latency 0.008396ms.
- covariance_32x7_padded_c: 3.6409; median latency 0.006146ms.
- correlation_32x7_padded_c: 2.1859; median latency 0.006313ms.
- joint_32x7_padded_c: 4.1535; median latency 0.009021ms.
- covariance_32x7_padded_f: 3.6799; median latency 0.006646ms.
- correlation_32x7_padded_f: 2.1534; median latency 0.006125ms.
- joint_32x7_padded_f: 3.9610; median latency 0.008688ms.
- covariance_32x8_c: 2.8048; median latency 0.005750ms.
- correlation_32x8_c: 1.6012; median latency 0.005854ms.
- joint_32x8_c: 3.2434; median latency 0.008396ms.
- covariance_32x8_f: 3.2208; median latency 0.005917ms.
- correlation_32x8_f: 1.8244; median latency 0.005917ms.
- joint_32x8_f: 3.5927; median latency 0.008458ms.
- covariance_32x8_padded_c: 2.9888; median latency 0.005354ms.
- correlation_32x8_padded_c: 1.4919; median latency 0.005750ms.
- joint_32x8_padded_c: 3.1453; median latency 0.008562ms.
- covariance_32x8_padded_f: 3.3289; median latency 0.005854ms.
- correlation_32x8_padded_f: 1.7713; median latency 0.005875ms.
- joint_32x8_padded_f: 3.5979; median latency 0.008250ms.
- covariance_33x9_c: 2.6385; median latency 0.006125ms.
- correlation_33x9_c: 1.5706; median latency 0.005875ms.
- joint_33x9_c: 3.1153; median latency 0.007958ms.
- covariance_33x9_f: 3.0610; median latency 0.006125ms.
- correlation_33x9_f: 1.5913; median latency 0.006729ms.
- joint_33x9_f: 3.3416; median latency 0.008583ms.
- covariance_33x9_padded_c: 2.8564; median latency 0.005583ms.
- correlation_33x9_padded_c: 1.5615; median latency 0.005583ms.
- joint_33x9_padded_c: 3.2151; median latency 0.008000ms.
- covariance_33x9_padded_f: 2.7194; median latency 0.006417ms.
- correlation_33x9_padded_f: 1.6895; median latency 0.006188ms.
- joint_33x9_padded_f: 3.3501; median latency 0.008209ms.
- covariance_32768x8_f: 1.3026; median latency 1.223416ms.
- correlation_32768x8_f: 1.3915; median latency 1.183209ms.
- joint_32768x8_f: 1.3861; median latency 1.180459ms.
- covariance_32768x8_padded_f: 1.2728; median latency 1.190103ms.
- correlation_32768x8_padded_f: 1.2076; median latency 1.238500ms.
- joint_32768x8_padded_f: 1.3647; median latency 1.180813ms.
- covariance_32768x16_f: 1.5358; median latency 2.727396ms.
- correlation_32768x16_f: 1.4468; median latency 2.737958ms.
- joint_32768x16_f: 1.4935; median latency 2.726313ms.
- covariance_32768x16_padded_f: 1.4391; median latency 2.802792ms.
- correlation_32768x16_padded_f: 1.4368; median latency 2.731375ms.
- joint_32768x16_padded_f: 1.5089; median latency 2.715917ms.
- covariance_32768x64_f: 3.5536; median latency 24.253208ms.
- correlation_32768x64_f: 3.2411; median latency 21.845792ms.
- joint_32768x64_f: 3.4953; median latency 23.039313ms.
- covariance_32768x64_padded_f: 3.2280; median latency 21.257229ms.
- correlation_32768x64_padded_f: 3.0161; median latency 21.291980ms.
- joint_32768x64_padded_f: 3.0191; median latency 20.240000ms.
- covariance_512x128_c: 1.8891; median latency 0.526292ms.
- correlation_512x128_c: 1.0148; median latency 0.545729ms.
- joint_512x128_c: 1.3419; median latency 0.572458ms.
- covariance_512x128_f: 2.7139; median latency 1.016750ms.
- correlation_512x128_f: 2.2130; median latency 1.022500ms.
- joint_512x128_f: 2.8733; median latency 1.031292ms.
- covariance_512x128_padded_c: 1.4852; median latency 0.544812ms.
- correlation_512x128_padded_c: 1.1174; median latency 0.516042ms.
- joint_512x128_padded_c: 1.2510; median latency 0.565875ms.
- covariance_512x128_padded_f: 2.3644; median latency 1.066709ms.
- correlation_512x128_padded_f: 1.6872; median latency 1.073146ms.
- joint_512x128_padded_f: 2.5211; median latency 1.104479ms.
- covariance_512x256_c: 2.6161; median latency 2.278104ms.
- correlation_512x256_c: 1.2677; median latency 2.128833ms.
- joint_512x256_c: 2.6551; median latency 2.255167ms.
- covariance_512x256_f: 6.6629; median latency 4.059500ms.
- correlation_512x256_f: 2.2910; median latency 3.928167ms.
- joint_512x256_f: 4.8551; median latency 4.115312ms.
- covariance_512x256_padded_c: 2.6727; median latency 2.148125ms.
- correlation_512x256_padded_c: 1.1711; median latency 2.063604ms.
- joint_512x256_padded_c: 2.4770; median latency 2.188292ms.
- covariance_512x256_padded_f: 5.4109; median latency 3.913792ms.
- correlation_512x256_padded_f: 2.3571; median latency 3.874979ms.
- joint_512x256_padded_f: 5.1309; median latency 4.028354ms.

## Memory differences (new−reference, bytes unless field is a count)

Against old:
- setup_heap/allocated_bytes: min +0, median +0.0, max +0; positive 0/36.
- setup_heap/live_bytes: min +0, median +0.0, max +0; positive 0/36.
- setup_heap/peak_bytes: min +0, median +0.0, max +0; positive 0/36.
- cold_heap/allocated_bytes: min +0, median +0.0, max +0; positive 0/36.
- cold_heap/live_bytes: min +0, median +0.0, max +0; positive 0/36.
- cold_heap/peak_bytes: min +0, median +0.0, max +0; positive 0/36.
- cold_retained_heap/allocated_bytes: min +0, median +0.0, max +0; positive 0/36.
- cold_retained_heap/live_bytes: min +0, median +0.0, max +0; positive 0/36.
- cold_retained_heap/peak_bytes: min +0, median +0.0, max +0; positive 0/36.
- warm_baseline_heap/allocated_bytes: min +0, median +0.0, max +0; positive 0/36.
- warm_baseline_heap/live_bytes: min +0, median +0.0, max +0; positive 0/36.
- warm_baseline_heap/peak_bytes: min +0, median +0.0, max +0; positive 0/36.
- warm_heap/allocated_bytes: min +0, median +0.0, max +0; positive 0/36.
- warm_heap/live_bytes: min +0, median +0.0, max +0; positive 0/36.
- warm_heap/peak_bytes: min +0, median +0.0, max +0; positive 0/36.
- warm_retained_heap/allocated_bytes: min +0, median +0.0, max +0; positive 0/36.
- warm_retained_heap/live_bytes: min +0, median +0.0, max +0; positive 0/36.
- warm_retained_heap/peak_bytes: min +0, median +0.0, max +0; positive 0/36.
- rss_baseline: min -2064384, median +0.0, max +1818624; positive 18/36.
- rss_peak: min -5111808, median +73728.0, max +4751360; positive 19/36.
- rss_delta: min -3047424, median +0.0, max +4194304; positive 17/36.
Against njit:
- setup_heap/allocated_bytes: min +17894, median +17894.0, max +23438; positive 36/36.
- setup_heap/live_bytes: min +8662, median +8662.0, max +10820; positive 36/36.
- setup_heap/peak_bytes: min +12087, median +12087.0, max +14245; positive 36/36.
- cold_heap/allocated_bytes: min -34716484, median -3123820.0, max +26248; positive 12/36.
- cold_heap/live_bytes: min +11758, median +11758.0, max +14158; positive 36/36.
- cold_heap/peak_bytes: min -34591296, median -3134016.0, max +10866; positive 12/36.
- cold_retained_heap/allocated_bytes: min -34716484, median -3123820.0, max +26248; positive 12/36.
- cold_retained_heap/live_bytes: min +11412, median +11924.0, max +13890; positive 36/36.
- cold_retained_heap/peak_bytes: min -34591296, median -3134016.0, max +10866; positive 12/36.
- warm_baseline_heap/allocated_bytes: min -208417474, median -18861490.0, max +8238; positive 4/36.
- warm_baseline_heap/live_bytes: min +11412, median +11924.0, max +13938; positive 36/36.
- warm_baseline_heap/peak_bytes: min -34591296, median -3134016.0, max +10914; positive 12/36.
- warm_heap/allocated_bytes: min -590559636, median -53484348.0, max -31296; positive 0/36.
- warm_heap/live_bytes: min +11774, median +12286.0, max +14286; positive 36/36.
- warm_heap/peak_bytes: min -34590934, median -3133654.0, max +11302; positive 12/36.
- warm_retained_heap/allocated_bytes: min -590559636, median -53484348.0, max -31296; positive 0/36.
- warm_retained_heap/live_bytes: min +11428, median +12452.0, max +14002; positive 36/36.
- warm_retained_heap/peak_bytes: min -34590934, median -3133654.0, max +11302; positive 12/36.
- rss_baseline: min -115064832, median -103694336.0, max -97665024; positive 0/36.
- rss_peak: min -131497984, median -113139712.0, max -100073472; positive 0/36.
- rss_delta: min -20152320, median -5521408.0, max +671744; positive 12/36.

36/108 declared cases only. Continuous tracked heap phases are not summed; RSS total peak and request increment remain separate. Positive differences retained; one fresh child per backend/kind/case is not repeated RSS attribution or proof of noise.

Ratios are median of12 paired round ratios, each small round clusters5calls. Overall medians and their ratios remain separate; group ratio extrema describe cases, not confidence bounds.

not inferred; correctness/regression/phase2/Excel, substantial NJIT advantage and memory requirements are separate
Cold means the first numerical request after inputs/imports/signature compilation and hashing. Input hashing may allocate and free an input copy beforehand; this is not a pristine allocator/page-state experiment.
