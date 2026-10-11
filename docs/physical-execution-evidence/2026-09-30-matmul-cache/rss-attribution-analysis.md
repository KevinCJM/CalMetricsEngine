# Cache MatMul RSS阶段归因

132个子进程、6案例各11独立AB/BA配对、每进程13阶段，共1716快照。旧93dc、新18977；只读已完成JSON，不运行内核或构建。原51项/306子进程正式内存报告保留，不以归因复测替代。

## 原始增加与配对复测

均为new−old，单位B；配对差先在同一对内相减。中位数不是无增加保证，所有正/负/零对保留。

| case | 原request差 | 原peak差 | 配对request中位[范围]；正/零/负 | 配对peak中位[范围]；正/零/负 |
|---|---:|---:|---|---|
| matmul_1024x1024x1024_cc_ordinary | 8585216 | 8454144 | 32768 [-8437760,25264128]；7/0/4 | 475136 [-8765440,25657344]；8/0/3 |
| matmul_1024x1024x1024_fc_ordinary | 16744448 | 16695296 | 32768 [-16646144,16826368]；6/0/5 | -1015808 [-17055744,17399808]；3/1/7 |
| matmul_64x32768x64_cc_ordinary | 49152 | 1966080 | -16384 [-196608,147456]；4/0/7 | -589824 [-1523712,1097728]；3/0/8 |
| matmul_64x32768x64_fc_ordinary | -16384 | 131072 | 16384 [-131072,212992]；6/0/5 | -49152 [-1146880,950272]；5/0/6 |
| matmul_1024x1024x1024_cf_ordinary | 8339456 | 8749056 | -8241152 [-8503296,81920]；3/1/7 | -7651328 [-10010624,1228800]；3/0/8 |
| matmul_512x512x512_cc_ordinary | -4145152 | -2719744 | -81920 [-4243456,4276224]；3/1/7 | -98304 [-5865472,5046272]；4/0/7 |

## 关键释放阶段净差

`[(new阶段−new backend_import)−(old阶段−old backend_import)]`中位B；所有13阶段的双方原值、双方基线变化和逐对净差均在JSON，以下只摘取两结果及释放阶段。不同列不相加。

| case/phase | malloc live | reserved | reserved−live | internal | external | reusable | compressed | footprint | resident |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| matmul_1024x1024x1024_cc_ordinary/warm_two_results | -1584 | -1048576 | -1048576 | 180224 | 32768 | 0 | 0 | 246016 | 32768 |
| matmul_1024x1024x1024_cc_ordinary/results_released | 165712 | -1048576 | -8296032 | -49152 | 32768 | 0 | 0 | -32704 | 32768 |
| matmul_1024x1024x1024_cc_ordinary/engine_closed | 168272 | -1048576 | -8298592 | -49152 | 32768 | 0 | 0 | -32704 | 32768 |
| matmul_1024x1024x1024_cc_ordinary/gc_after_close | 168272 | -1048576 | -8298592 | -49152 | 32768 | 0 | 0 | -32704 | 32768 |
| matmul_1024x1024x1024_fc_ordinary/warm_two_results | -8320960 | 1048576 | 6453872 | 0 | 32768 | 0 | 0 | 16448 | 32768 |
| matmul_1024x1024x1024_fc_ordinary/results_released | -8321024 | 1048576 | 7308880 | 0 | 32768 | -327680 | 0 | 16384 | 32768 |
| matmul_1024x1024x1024_fc_ordinary/engine_closed | -8318464 | 1048576 | 7308864 | 0 | 32768 | -327680 | 0 | 16384 | 32768 |
| matmul_1024x1024x1024_fc_ordinary/gc_after_close | -8318464 | 1048576 | 7308864 | 0 | 32768 | -327680 | 0 | 16384 | 32768 |
| matmul_64x32768x64_cc_ordinary/warm_two_results | 16 | -1048576 | -1014816 | -32768 | 16384 | 0 | 0 | -32768 | -16384 |
| matmul_64x32768x64_cc_ordinary/results_released | -26128 | -1048576 | -1014944 | -32768 | 16384 | 0 | 0 | -32768 | -16384 |
| matmul_64x32768x64_cc_ordinary/engine_closed | -26128 | -1048576 | -1014816 | -32768 | 16384 | 0 | 0 | -32768 | -16384 |
| matmul_64x32768x64_cc_ordinary/gc_after_close | -26128 | -1048576 | -1014816 | -32768 | 16384 | 0 | 0 | -32768 | -16384 |
| matmul_64x32768x64_fc_ordinary/warm_two_results | -31168 | -8388608 | -8357488 | 0 | 16384 | 0 | 0 | 0 | 16384 |
| matmul_64x32768x64_fc_ordinary/results_released | -34768 | -8388608 | -8324800 | 0 | 16384 | 0 | 0 | 0 | 16384 |
| matmul_64x32768x64_fc_ordinary/engine_closed | -34784 | -8388608 | -8324816 | 0 | 16384 | 0 | 0 | 0 | 16384 |
| matmul_64x32768x64_fc_ordinary/gc_after_close | -34784 | -8388608 | -8324816 | 0 | 16384 | 0 | 0 | 0 | 16384 |
| matmul_1024x1024x1024_cf_ordinary/warm_two_results | -1472 | -2097152 | -8387136 | -81920 | 16384 | 0 | 0 | -114816 | -8241152 |
| matmul_1024x1024x1024_cf_ordinary/results_released | 2352 | -2097152 | -7241712 | -16384 | 16384 | -8290304 | 0 | -16384 | -8241152 |
| matmul_1024x1024x1024_cf_ordinary/engine_closed | 2240 | -2097152 | -7241712 | -16384 | 16384 | -8290304 | 0 | -16384 | -8241152 |
| matmul_1024x1024x1024_cf_ordinary/gc_after_close | 2240 | -2097152 | -7241712 | -16384 | 16384 | -8290304 | 0 | -16384 | -8241152 |
| matmul_512x512x512_cc_ordinary/warm_two_results | -2031568 | 0 | -37888 | 65536 | 16384 | -2064384 | 0 | 81984 | -81920 |
| matmul_512x512x512_cc_ordinary/results_released | -2064224 | 0 | 62880 | -65536 | 16384 | 65536 | 0 | -81984 | -81920 |
| matmul_512x512x512_cc_ordinary/engine_closed | -2064240 | 0 | 60320 | -65536 | 16384 | 65536 | 0 | -81984 | -81920 |
| matmul_512x512x512_cc_ordinary/gc_after_close | -2064240 | 0 | 60320 | -65536 | 16384 | 65536 | 0 | -81984 | -81920 |

## 分配器保留：仅记录观察，不证明因果

以下是每个版本GC后相对其backend_import的zone容量/live变化，及新旧净差范围。容量保留、live仍增长或reusable变化可以同时发生；数值小于或大于0都不自动证明内核scratch、泄漏或唯一根因。

| case | old reserved/live中位 | new reserved/live中位 | reserved配对净差中位[范围] | live配对净差中位[范围] |
|---|---:|---:|---|---|
| matmul_1024x1024x1024_cc_ordinary | 2097152/16716800 | 1048576/25005632 | -1048576 [-10485760,11534336] | 168272 [-16648032,24870816] |
| matmul_1024x1024x1024_fc_ordinary | 0/25040928 | 1048576/24907200 | 1048576 [-9437184,9437184] | -8318464 [-16746048,16577504] |
| matmul_64x32768x64_cc_ordinary | 8388608/66048 | 8388608/36560 | -1048576 [-18874368,9437184] | -26128 [-99808,95168] |
| matmul_64x32768x64_fc_ordinary | 9437184/70336 | 0/34912 | -8388608 [-9437184,11534336] | -34784 [-92640,64144] |
| matmul_1024x1024x1024_cf_ordinary | 2097152/25008192 | 0/24975200 | -2097152 [-19922944,10485760] | 2240 [-16711808,16677440] |
| matmul_512x512x512_cc_ordinary | 0/4066688 | 0/2035328 | 0 [-10485760,18874368] | -2064240 [-4162864,4131616] |

## 有效性与限制

- 原正式输入、完整SHA/源码/SO、所有权、END后完整输出协议核验；Mach字段全部有效。线程/Mach分布：`{"mach_status": {"0": 1716}, "mach_count": {"93": 1716}, "valid_mask": {"524272": 1716}, "page_size": {"16384": 1716}, "endpoint_threads": {"1": 1716}}`。
- 原生resident与外部端点共1716次比较：差值中位0.0B、范围[0,0]，非零0次；原生值在该阶段采样范围之外0次。两者不是同一时刻，所有不匹配保留。
- 外部样本201380个，最大全局间隔8.486708ms，包含1584个handoff；最大handoff0.246250ms、单次观察调用7.856083ms。名义0.5ms不等于最大分辨率，短峰可能漏采。
- Phase pauses and native snapshot instrumentation alter allocation/residency timing; this does not replace the formal request measurement.
- Inputs and input hashing precede cold; tobytes may allocate/free input copies, so cold is first numeric execution, not pristine allocator.
- malloc zones include all Python arenas/caches; reserved-minus-live is not proof of reusable or resident pages.
- internal/external/reusable/compressed/footprint overlap or differ in scope; never add field medians or difference lifetime peaks as request peaks.
- Persistent post-release zone capacity/in-use changes are observed allocator state, not proof of a specific kernel allocation or unique cause.
- Original positive single-child RSS measurements, including8–16MB in1024 cases, remain valid evidence; different paired medians cannot erase them.
- Post-END numerical parity/ownership is verified by the recorded completed protocol; this analyzer does not rerun numerical comparisons.

本报告只给归因观察；不宣称完整内存不增加，不取消原8–16MB增加，也不宣称NJIT性能目标已闭环。
