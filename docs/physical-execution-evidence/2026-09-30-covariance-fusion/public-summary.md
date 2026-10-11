# Public covariance/correlation evidence summary

108 time cases, old `63090c5a7971acd8c947ef5b0e20a46bf207f88772e8c322ab94a3537ecd96bb`, new `f0e62a925728d638bcadfce326bdc054b08fcdf80251cee5fc0a23f55fe0b72e`. Memory:36cases/216children, not108case coverage.

| Group | Cases | New/old range | Faster/slower old | New/NJIT range | Faster/slower NJIT |
|---|---:|---:|---:|---:|---:|
| all | 108 | 0.4171–1.0734 | 76/30 | 0.5628–4.2124 | 30/78 |
| layout=c | 27 | 0.9524–1.0333 | 14/12 | 0.5628–4.0858 | 9/18 |
| workload=small | 48 | 0.8684–1.0734 | 29/17 | 1.4584–4.2124 | 0/48 |
| request=covariance | 36 | 0.4213–1.0333 | 28/7 | 0.5818–4.0047 | 10/26 |
| roots=1 | 72 | 0.4171–1.0385 | 54/16 | 0.5756–4.0047 | 20/52 |
| c/small/roots=1 | 8 | 0.9687–1.0333 | 5/2 | 1.5058–3.9224 | 0/8 |
| request=correlation | 36 | 0.4171–1.0385 | 26/9 | 0.5756–2.2152 | 10/26 |
| request=joint | 36 | 0.4209–1.0734 | 22/14 | 0.5628–4.2124 | 10/26 |
| roots=2 | 36 | 0.4209–1.0734 | 22/14 | 0.5628–4.2124 | 10/26 |
| c/small/roots=2 | 4 | 0.9524–1.0283 | 1/3 | 3.3298–4.0858 | 0/4 |
| layout=f | 27 | 0.4171–1.0151 | 25/2 | 0.5711–4.0279 | 6/21 |
| f/small/roots=1 | 8 | 0.8751–1.0151 | 6/2 | 1.4584–3.7774 | 0/8 |
| f/small/roots=2 | 4 | 0.9175–0.9883 | 4/0 | 3.1238–4.0279 | 0/4 |
| layout=padded_c | 27 | 0.9729–1.0458 | 13/13 | 0.5648–4.2124 | 9/18 |
| padded_c/small/roots=1 | 8 | 0.9729–1.0385 | 4/3 | 1.5432–3.6368 | 0/8 |
| padded_c/small/roots=2 | 4 | 1.0150–1.0458 | 0/4 | 3.2353–4.2124 | 0/4 |
| layout=padded_f | 27 | 0.4220–1.0734 | 24/3 | 0.5675–4.0047 | 6/21 |
| padded_f/small/roots=1 | 8 | 0.8684–1.0260 | 7/1 | 1.4883–3.6147 | 0/8 |
| padded_f/small/roots=2 | 4 | 0.9132–1.0734 | 2/2 | 3.1201–3.9804 | 0/4 |
| workload=long | 36 | 0.4171–1.0026 | 31/5 | 0.5628–1.6869 | 30/6 |
| c/long/roots=1 | 6 | 0.9892–1.0010 | 4/2 | 0.5988–0.8304 | 6/0 |
| c/long/roots=2 | 3 | 0.9963–1.0012 | 2/1 | 0.5628–0.8229 | 3/0 |
| f/long/roots=1 | 6 | 0.4171–0.4957 | 6/0 | 0.5756–1.6732 | 4/2 |
| f/long/roots=2 | 3 | 0.4209–0.4956 | 3/0 | 0.5711–1.6869 | 2/1 |
| padded_c/long/roots=1 | 6 | 0.9935–0.9980 | 6/0 | 0.5784–0.8901 | 6/0 |
| padded_c/long/roots=2 | 3 | 0.9945–1.0026 | 1/2 | 0.5648–0.8954 | 3/0 |
| padded_f/long/roots=1 | 6 | 0.4220–0.5602 | 6/0 | 0.5818–1.6632 | 4/2 |
| padded_f/long/roots=2 | 3 | 0.4246–0.5652 | 3/0 | 0.5675–1.6632 | 2/1 |
| workload=wide | 24 | 0.5932–1.0058 | 16/8 | 1.0168–4.0047 | 0/24 |
| c/wide/roots=1 | 4 | 0.9978–1.0041 | 1/3 | 1.0168–3.5408 | 0/4 |
| c/wide/roots=2 | 2 | 0.9925–1.0018 | 1/1 | 1.4051–2.8873 | 0/2 |
| f/wide/roots=1 | 4 | 0.5932–0.6050 | 4/0 | 1.4100–3.5577 | 0/4 |
| f/wide/roots=2 | 2 | 0.6092–0.6141 | 2/0 | 1.9377–3.3183 | 0/2 |
| padded_c/wide/roots=1 | 4 | 0.9984–1.0058 | 1/3 | 1.0395–2.8844 | 0/4 |
| padded_c/wide/roots=2 | 2 | 0.9958–1.0040 | 1/1 | 1.7184–2.7493 | 0/2 |
| padded_f/wide/roots=1 | 4 | 0.6006–0.6156 | 4/0 | 1.3024–4.0047 | 0/4 |
| padded_f/wide/roots=2 | 2 | 0.6137–0.6213 | 2/0 | 1.8434–2.9443 | 0/2 |

## All absolute case medians and paired ratios

| Case | Old ms | New ms | NJIT ms | New−old ms | Paired new/old | Paired new/NJIT |
|---|---:|---:|---:|---:|---:|---:|
| covariance_31x8_c | 0.007376 | 0.007521 | 0.001917 | +0.000145 | 0.9706 | 3.9224 |
| correlation_31x8_c | 0.006771 | 0.006709 | 0.003250 | -0.000063 | 0.9717 | 2.0800 |
| joint_31x8_c | 0.010292 | 0.009729 | 0.002375 | -0.000563 | 0.9524 | 4.0096 |
| covariance_31x8_f | 0.006812 | 0.006542 | 0.001750 | -0.000270 | 0.9558 | 3.7774 |
| correlation_31x8_f | 0.006500 | 0.006437 | 0.003250 | -0.000063 | 1.0151 | 2.0298 |
| joint_31x8_f | 0.009000 | 0.008833 | 0.002167 | -0.000167 | 0.9883 | 4.0279 |
| covariance_31x8_padded_c | 0.006666 | 0.006562 | 0.001792 | -0.000104 | 0.9964 | 3.6191 |
| correlation_31x8_padded_c | 0.006521 | 0.006437 | 0.003125 | -0.000084 | 0.9935 | 2.0132 |
| joint_31x8_padded_c | 0.008750 | 0.008937 | 0.002292 | +0.000187 | 1.0190 | 3.9818 |
| covariance_31x8_padded_f | 0.006562 | 0.006437 | 0.001791 | -0.000125 | 0.9850 | 3.5773 |
| correlation_31x8_padded_f | 0.006458 | 0.006250 | 0.003209 | -0.000208 | 0.9634 | 1.9385 |
| joint_31x8_padded_f | 0.008625 | 0.009208 | 0.002292 | +0.000583 | 1.0734 | 3.9598 |
| covariance_32x7_c | 0.006042 | 0.006292 | 0.001666 | +0.000250 | 1.0333 | 3.8436 |
| correlation_32x7_c | 0.006084 | 0.006000 | 0.002709 | -0.000084 | 0.9895 | 2.2152 |
| joint_32x7_c | 0.008230 | 0.008209 | 0.002000 | -0.000021 | 1.0026 | 4.0858 |
| covariance_32x7_f | 0.005791 | 0.005875 | 0.001667 | +0.000084 | 1.0037 | 3.5002 |
| correlation_32x7_f | 0.005938 | 0.005937 | 0.002750 | -0.000001 | 0.9894 | 2.1066 |
| joint_32x7_f | 0.008292 | 0.008230 | 0.002083 | -0.000062 | 0.9877 | 3.8728 |
| covariance_32x7_padded_c | 0.005937 | 0.005958 | 0.001625 | +0.000021 | 1.0035 | 3.6368 |
| correlation_32x7_padded_c | 0.006020 | 0.005958 | 0.002750 | -0.000063 | 0.9965 | 2.1667 |
| joint_32x7_padded_c | 0.008250 | 0.008646 | 0.002063 | +0.000396 | 1.0458 | 4.2124 |
| covariance_32x7_padded_f | 0.005708 | 0.005959 | 0.001666 | +0.000250 | 1.0260 | 3.6147 |
| correlation_32x7_padded_f | 0.006063 | 0.006000 | 0.002730 | -0.000063 | 0.9965 | 2.1456 |
| joint_32x7_padded_f | 0.008271 | 0.008167 | 0.002063 | -0.000103 | 1.0053 | 3.9804 |
| covariance_32x8_c | 0.005271 | 0.005396 | 0.001750 | +0.000125 | 1.0288 | 3.1089 |
| correlation_32x8_c | 0.005292 | 0.005375 | 0.003145 | +0.000083 | 1.0000 | 1.6937 |
| joint_32x8_c | 0.007646 | 0.007771 | 0.002333 | +0.000125 | 1.0247 | 3.4814 |
| covariance_32x8_f | 0.005771 | 0.005000 | 0.001750 | -0.000771 | 0.8872 | 2.8638 |
| correlation_32x8_f | 0.005708 | 0.004958 | 0.003125 | -0.000750 | 0.8828 | 1.5851 |
| joint_32x8_f | 0.008125 | 0.007562 | 0.002250 | -0.000563 | 0.9338 | 3.3666 |
| covariance_32x8_padded_c | 0.005500 | 0.005479 | 0.001792 | -0.000021 | 1.0000 | 3.1197 |
| correlation_32x8_padded_c | 0.005646 | 0.005625 | 0.003270 | -0.000021 | 1.0040 | 1.7217 |
| joint_32x8_padded_c | 0.007979 | 0.008125 | 0.002208 | +0.000146 | 1.0150 | 3.5851 |
| covariance_32x8_padded_f | 0.005666 | 0.004916 | 0.001708 | -0.000750 | 0.8684 | 2.9022 |
| correlation_32x8_padded_f | 0.005666 | 0.005083 | 0.003145 | -0.000583 | 0.8913 | 1.5907 |
| joint_32x8_padded_f | 0.007833 | 0.007146 | 0.002167 | -0.000687 | 0.9132 | 3.2819 |
| covariance_33x9_c | 0.005729 | 0.005458 | 0.001937 | -0.000271 | 0.9687 | 2.8443 |
| correlation_33x9_c | 0.005521 | 0.005417 | 0.003542 | -0.000104 | 0.9768 | 1.5058 |
| joint_33x9_c | 0.008230 | 0.008313 | 0.002500 | +0.000083 | 1.0283 | 3.3298 |
| covariance_33x9_f | 0.005917 | 0.005313 | 0.001875 | -0.000605 | 0.9043 | 2.8092 |
| correlation_33x9_f | 0.005917 | 0.005208 | 0.003563 | -0.000709 | 0.8751 | 1.4584 |
| joint_33x9_f | 0.008459 | 0.007855 | 0.002458 | -0.000604 | 0.9175 | 3.1238 |
| covariance_33x9_padded_c | 0.005375 | 0.005313 | 0.001855 | -0.000062 | 0.9729 | 2.8855 |
| correlation_33x9_padded_c | 0.005500 | 0.005583 | 0.003625 | +0.000084 | 1.0385 | 1.5432 |
| joint_33x9_padded_c | 0.007959 | 0.008146 | 0.002458 | +0.000188 | 1.0185 | 3.2353 |
| covariance_33x9_padded_f | 0.005792 | 0.005333 | 0.001875 | -0.000459 | 0.9178 | 2.8199 |
| correlation_33x9_padded_f | 0.006250 | 0.005458 | 0.003708 | -0.000792 | 0.8794 | 1.4883 |
| joint_33x9_padded_f | 0.008187 | 0.007709 | 0.002459 | -0.000478 | 0.9528 | 3.1201 |
| covariance_32768x8_c | 0.614625 | 0.611916 | 0.934833 | -0.002708 | 0.9998 | 0.6582 |
| correlation_32768x8_c | 0.613125 | 0.612771 | 0.965958 | -0.000354 | 1.0010 | 0.6334 |
| joint_32768x8_c | 0.612749 | 0.612125 | 0.991187 | -0.000624 | 0.9963 | 0.6168 |
| covariance_32768x8_f | 1.160251 | 0.488687 | 0.844750 | -0.671563 | 0.4213 | 0.5850 |
| correlation_32768x8_f | 1.172438 | 0.489750 | 0.840167 | -0.682688 | 0.4171 | 0.5756 |
| joint_32768x8_f | 1.162813 | 0.489063 | 0.857458 | -0.673750 | 0.4209 | 0.5711 |
| covariance_32768x8_padded_c | 0.622833 | 0.616334 | 1.058292 | -0.006499 | 0.9935 | 0.5872 |
| correlation_32768x8_padded_c | 0.618896 | 0.618729 | 1.067979 | -0.000167 | 0.9976 | 0.5784 |
| joint_32768x8_padded_c | 0.625605 | 0.628146 | 1.074521 | +0.002541 | 1.0016 | 0.5836 |
| covariance_32768x8_padded_f | 1.154249 | 0.489979 | 0.826708 | -0.664270 | 0.4220 | 0.5818 |
| correlation_32768x8_padded_f | 1.156729 | 0.488563 | 0.849833 | -0.668166 | 0.4228 | 0.5966 |
| joint_32768x8_padded_f | 1.159229 | 0.492167 | 0.861667 | -0.667062 | 0.4246 | 0.5675 |
| covariance_32768x16_c | 1.230000 | 1.226604 | 1.958667 | -0.003396 | 0.9992 | 0.6262 |
| correlation_32768x16_c | 1.314709 | 1.291438 | 2.196604 | -0.023271 | 0.9892 | 0.5988 |
| joint_32768x16_c | 1.267708 | 1.273187 | 2.277938 | +0.005479 | 1.0012 | 0.5628 |
| covariance_32768x16_f | 2.797771 | 1.276291 | 1.911417 | -1.521480 | 0.4574 | 0.6387 |
| correlation_32768x16_f | 2.690688 | 1.217146 | 1.715813 | -1.473542 | 0.4530 | 0.7101 |
| joint_32768x16_f | 2.676938 | 1.218187 | 1.737645 | -1.458750 | 0.4555 | 0.7028 |
| covariance_32768x16_padded_c | 1.260771 | 1.257562 | 2.078604 | -0.003209 | 0.9973 | 0.6036 |
| correlation_32768x16_padded_c | 1.260938 | 1.259791 | 2.172625 | -0.001146 | 0.9980 | 0.5821 |
| joint_32768x16_padded_c | 1.261791 | 1.265687 | 2.235771 | +0.003896 | 1.0026 | 0.5648 |
| covariance_32768x16_padded_f | 2.664688 | 1.213501 | 1.754708 | -1.451187 | 0.4556 | 0.6915 |
| correlation_32768x16_padded_f | 2.670396 | 1.212541 | 1.844333 | -1.457855 | 0.4550 | 0.6574 |
| joint_32768x16_padded_f | 2.668979 | 1.215479 | 1.875896 | -1.453500 | 0.4561 | 0.6511 |
| covariance_32768x64_c | 8.838687 | 8.776521 | 10.587229 | -0.062166 | 0.9991 | 0.8304 |
| correlation_32768x64_c | 8.803980 | 8.741562 | 10.761104 | -0.062418 | 1.0003 | 0.8113 |
| joint_32768x64_c | 8.840730 | 8.756771 | 10.588000 | -0.083958 | 0.9966 | 0.8229 |
| covariance_32768x64_f | 22.933042 | 11.248605 | 6.750896 | -11.684437 | 0.4957 | 1.6732 |
| correlation_32768x64_f | 23.439083 | 11.243979 | 6.759291 | -12.195104 | 0.4840 | 1.6712 |
| joint_32768x64_f | 22.768625 | 11.288000 | 6.693792 | -11.480625 | 0.4956 | 1.6869 |
| covariance_32768x64_padded_c | 9.118208 | 9.046542 | 10.178667 | -0.071666 | 0.9954 | 0.8901 |
| correlation_32768x64_padded_c | 9.125771 | 9.035021 | 10.216646 | -0.090750 | 0.9946 | 0.8845 |
| joint_32768x64_padded_c | 9.093896 | 9.074563 | 10.171771 | -0.019334 | 0.9945 | 0.8954 |
| covariance_32768x64_padded_f | 20.147063 | 11.231750 | 6.802562 | -8.915313 | 0.5591 | 1.6574 |
| correlation_32768x64_padded_f | 20.179750 | 11.259771 | 6.803604 | -8.919979 | 0.5602 | 1.6632 |
| joint_32768x64_padded_f | 19.988667 | 11.245354 | 6.751437 | -8.743313 | 0.5652 | 1.6632 |
| covariance_512x128_c | 0.519917 | 0.519687 | 0.362792 | -0.000229 | 0.9978 | 1.4423 |
| correlation_512x128_c | 0.528000 | 0.531021 | 0.524000 | +0.003020 | 1.0027 | 1.0168 |
| joint_512x128_c | 0.542813 | 0.542917 | 0.391958 | +0.000104 | 0.9925 | 1.4051 |
| covariance_512x128_f | 1.009146 | 0.611583 | 0.290083 | -0.397562 | 0.6050 | 2.1005 |
| correlation_512x128_f | 1.014020 | 0.614938 | 0.439188 | -0.399083 | 0.6041 | 1.4100 |
| joint_512x128_f | 1.036792 | 0.633604 | 0.327625 | -0.403188 | 0.6141 | 1.9377 |
| covariance_512x128_padded_c | 0.512208 | 0.515875 | 0.331000 | +0.003667 | 1.0058 | 1.5777 |
| correlation_512x128_padded_c | 0.514062 | 0.518042 | 0.508563 | +0.003979 | 1.0000 | 1.0395 |
| joint_512x128_padded_c | 0.531084 | 0.530688 | 0.307521 | -0.000395 | 0.9958 | 1.7184 |
| covariance_512x128_padded_f | 1.001375 | 0.617605 | 0.359333 | -0.383770 | 0.6156 | 1.8203 |
| correlation_512x128_padded_f | 0.999687 | 0.614209 | 0.472521 | -0.385478 | 0.6149 | 1.3024 |
| joint_512x128_padded_f | 1.019208 | 0.630687 | 0.340958 | -0.388521 | 0.6213 | 1.8434 |
| covariance_512x256_c | 2.096563 | 2.104250 | 0.615688 | +0.007687 | 1.0041 | 3.5408 |
| correlation_512x256_c | 2.115959 | 2.115645 | 1.637666 | -0.000313 | 1.0017 | 1.2890 |
| joint_512x256_c | 2.234792 | 2.234354 | 0.776125 | -0.000437 | 1.0018 | 2.8873 |
| covariance_512x256_f | 3.900187 | 2.327812 | 0.705688 | -1.572375 | 0.5932 | 3.5577 |
| correlation_512x256_f | 3.889708 | 2.315355 | 1.574375 | -1.574354 | 0.5943 | 1.4779 |
| joint_512x256_f | 4.008083 | 2.444312 | 0.737771 | -1.563771 | 0.6092 | 3.3183 |
| covariance_512x256_padded_c | 2.039646 | 2.035521 | 0.739501 | -0.004125 | 0.9984 | 2.8844 |
| correlation_512x256_padded_c | 2.047021 | 2.053501 | 1.586625 | +0.006480 | 1.0031 | 1.2989 |
| joint_512x256_padded_c | 2.170709 | 2.177250 | 0.794875 | +0.006541 | 1.0040 | 2.7493 |
| covariance_512x256_padded_f | 3.847167 | 2.311125 | 0.583833 | -1.536042 | 0.6006 | 4.0047 |
| correlation_512x256_padded_f | 3.863000 | 2.323875 | 1.567479 | -1.539125 | 0.6018 | 1.4804 |
| joint_512x256_padded_f | 3.992188 | 2.453917 | 0.839146 | -1.538271 | 0.6137 | 2.9443 |

## All regressions

Against old: 30/108 cases slower by paired median.
- correlation_31x8_f: 1.0151; median latency 0.006437ms.
- joint_31x8_padded_c: 1.0190; median latency 0.008937ms.
- joint_31x8_padded_f: 1.0734; median latency 0.009208ms.
- covariance_32x7_c: 1.0333; median latency 0.006292ms.
- joint_32x7_c: 1.0026; median latency 0.008209ms.
- covariance_32x7_f: 1.0037; median latency 0.005875ms.
- covariance_32x7_padded_c: 1.0035; median latency 0.005958ms.
- joint_32x7_padded_c: 1.0458; median latency 0.008646ms.
- covariance_32x7_padded_f: 1.0260; median latency 0.005959ms.
- joint_32x7_padded_f: 1.0053; median latency 0.008167ms.
- covariance_32x8_c: 1.0288; median latency 0.005396ms.
- joint_32x8_c: 1.0247; median latency 0.007771ms.
- correlation_32x8_padded_c: 1.0040; median latency 0.005625ms.
- joint_32x8_padded_c: 1.0150; median latency 0.008125ms.
- joint_33x9_c: 1.0283; median latency 0.008313ms.
- correlation_33x9_padded_c: 1.0385; median latency 0.005583ms.
- joint_33x9_padded_c: 1.0185; median latency 0.008146ms.
- correlation_32768x8_c: 1.0010; median latency 0.612771ms.
- joint_32768x8_padded_c: 1.0016; median latency 0.628146ms.
- joint_32768x16_c: 1.0012; median latency 1.273187ms.
- joint_32768x16_padded_c: 1.0026; median latency 1.265687ms.
- correlation_32768x64_c: 1.0003; median latency 8.741562ms.
- correlation_512x128_c: 1.0027; median latency 0.531021ms.
- covariance_512x128_padded_c: 1.0058; median latency 0.515875ms.
- correlation_512x128_padded_c: 1.0000; median latency 0.518042ms.
- covariance_512x256_c: 1.0041; median latency 2.104250ms.
- correlation_512x256_c: 1.0017; median latency 2.115645ms.
- joint_512x256_c: 1.0018; median latency 2.234354ms.
- correlation_512x256_padded_c: 1.0031; median latency 2.053501ms.
- joint_512x256_padded_c: 1.0040; median latency 2.177250ms.
Against njit: 78/108 cases slower by paired median.
- covariance_31x8_c: 3.9224; median latency 0.007521ms.
- correlation_31x8_c: 2.0800; median latency 0.006709ms.
- joint_31x8_c: 4.0096; median latency 0.009729ms.
- covariance_31x8_f: 3.7774; median latency 0.006542ms.
- correlation_31x8_f: 2.0298; median latency 0.006437ms.
- joint_31x8_f: 4.0279; median latency 0.008833ms.
- covariance_31x8_padded_c: 3.6191; median latency 0.006562ms.
- correlation_31x8_padded_c: 2.0132; median latency 0.006437ms.
- joint_31x8_padded_c: 3.9818; median latency 0.008937ms.
- covariance_31x8_padded_f: 3.5773; median latency 0.006437ms.
- correlation_31x8_padded_f: 1.9385; median latency 0.006250ms.
- joint_31x8_padded_f: 3.9598; median latency 0.009208ms.
- covariance_32x7_c: 3.8436; median latency 0.006292ms.
- correlation_32x7_c: 2.2152; median latency 0.006000ms.
- joint_32x7_c: 4.0858; median latency 0.008209ms.
- covariance_32x7_f: 3.5002; median latency 0.005875ms.
- correlation_32x7_f: 2.1066; median latency 0.005937ms.
- joint_32x7_f: 3.8728; median latency 0.008230ms.
- covariance_32x7_padded_c: 3.6368; median latency 0.005958ms.
- correlation_32x7_padded_c: 2.1667; median latency 0.005958ms.
- joint_32x7_padded_c: 4.2124; median latency 0.008646ms.
- covariance_32x7_padded_f: 3.6147; median latency 0.005959ms.
- correlation_32x7_padded_f: 2.1456; median latency 0.006000ms.
- joint_32x7_padded_f: 3.9804; median latency 0.008167ms.
- covariance_32x8_c: 3.1089; median latency 0.005396ms.
- correlation_32x8_c: 1.6937; median latency 0.005375ms.
- joint_32x8_c: 3.4814; median latency 0.007771ms.
- covariance_32x8_f: 2.8638; median latency 0.005000ms.
- correlation_32x8_f: 1.5851; median latency 0.004958ms.
- joint_32x8_f: 3.3666; median latency 0.007562ms.
- covariance_32x8_padded_c: 3.1197; median latency 0.005479ms.
- correlation_32x8_padded_c: 1.7217; median latency 0.005625ms.
- joint_32x8_padded_c: 3.5851; median latency 0.008125ms.
- covariance_32x8_padded_f: 2.9022; median latency 0.004916ms.
- correlation_32x8_padded_f: 1.5907; median latency 0.005083ms.
- joint_32x8_padded_f: 3.2819; median latency 0.007146ms.
- covariance_33x9_c: 2.8443; median latency 0.005458ms.
- correlation_33x9_c: 1.5058; median latency 0.005417ms.
- joint_33x9_c: 3.3298; median latency 0.008313ms.
- covariance_33x9_f: 2.8092; median latency 0.005313ms.
- correlation_33x9_f: 1.4584; median latency 0.005208ms.
- joint_33x9_f: 3.1238; median latency 0.007855ms.
- covariance_33x9_padded_c: 2.8855; median latency 0.005313ms.
- correlation_33x9_padded_c: 1.5432; median latency 0.005583ms.
- joint_33x9_padded_c: 3.2353; median latency 0.008146ms.
- covariance_33x9_padded_f: 2.8199; median latency 0.005333ms.
- correlation_33x9_padded_f: 1.4883; median latency 0.005458ms.
- joint_33x9_padded_f: 3.1201; median latency 0.007709ms.
- covariance_32768x64_f: 1.6732; median latency 11.248605ms.
- correlation_32768x64_f: 1.6712; median latency 11.243979ms.
- joint_32768x64_f: 1.6869; median latency 11.288000ms.
- covariance_32768x64_padded_f: 1.6574; median latency 11.231750ms.
- correlation_32768x64_padded_f: 1.6632; median latency 11.259771ms.
- joint_32768x64_padded_f: 1.6632; median latency 11.245354ms.
- covariance_512x128_c: 1.4423; median latency 0.519687ms.
- correlation_512x128_c: 1.0168; median latency 0.531021ms.
- joint_512x128_c: 1.4051; median latency 0.542917ms.
- covariance_512x128_f: 2.1005; median latency 0.611583ms.
- correlation_512x128_f: 1.4100; median latency 0.614938ms.
- joint_512x128_f: 1.9377; median latency 0.633604ms.
- covariance_512x128_padded_c: 1.5777; median latency 0.515875ms.
- correlation_512x128_padded_c: 1.0395; median latency 0.518042ms.
- joint_512x128_padded_c: 1.7184; median latency 0.530688ms.
- covariance_512x128_padded_f: 1.8203; median latency 0.617605ms.
- correlation_512x128_padded_f: 1.3024; median latency 0.614209ms.
- joint_512x128_padded_f: 1.8434; median latency 0.630687ms.
- covariance_512x256_c: 3.5408; median latency 2.104250ms.
- correlation_512x256_c: 1.2890; median latency 2.115645ms.
- joint_512x256_c: 2.8873; median latency 2.234354ms.
- covariance_512x256_f: 3.5577; median latency 2.327812ms.
- correlation_512x256_f: 1.4779; median latency 2.315355ms.
- joint_512x256_f: 3.3183; median latency 2.444312ms.
- covariance_512x256_padded_c: 2.8844; median latency 2.035521ms.
- correlation_512x256_padded_c: 1.2989; median latency 2.053501ms.
- joint_512x256_padded_c: 2.7493; median latency 2.177250ms.
- covariance_512x256_padded_f: 4.0047; median latency 2.311125ms.
- correlation_512x256_padded_f: 1.4804; median latency 2.323875ms.
- joint_512x256_padded_f: 2.9443; median latency 2.453917ms.

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
- rss_baseline: min -1277952, median -24576.0, max +1474560; positive 15/36.
- rss_peak: min -1720320, median +32768.0, max +3604480; positive 19/36.
- rss_delta: min -1605632, median +16384.0, max +3342336; positive 19/36.
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
- rss_baseline: min -114262016, median -102252544.0, max -95944704; positive 0/36.
- rss_peak: min -129777664, median -111083520.0, max -99188736; positive 0/36.
- rss_delta: min -22839296, median -4128768.0, max +704512; positive 12/36.

36/108 declared cases only. Continuous tracked heap phases are not summed; RSS total peak and request increment remain separate. Positive differences retained; one fresh child per backend/kind/case is not repeated RSS attribution or proof of noise.

Ratios are median of12 paired round ratios, each small round clusters5calls. Overall medians and their ratios remain separate; group ratio extrema describe cases, not confidence bounds.

not inferred; correctness/regression/phase2/Excel, substantial NJIT advantage and memory requirements are separate
Cold means the first numerical request after inputs/imports/signature compilation and hashing. Input hashing may allocate and free an input copy beforehand; this is not a pristine allocator/page-state experiment.
