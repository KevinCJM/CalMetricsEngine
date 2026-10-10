# V99：组合权重四路保序计算与实际 owned MC 复验

日期：2026-10-10；本机macOS arm64、CPython3.12、CPU1。这是该日期与以下精确身份的计算链证据，不是当前全局性能、平台迁移或业务资格通过。

## 实现与身份

2026-10-10，仅 `cpp/compact_path.cpp` 与 `tests/test_compact_weight_lanes.py` 的 V99 变更已进入实际源码；本机实际 AOT 构建身份为 `f789954a…`。实际普通相关 Python **1,049**、全量 **17,179** 与 native **9/9** 无失败或跳过（相关不与全量相加）；普通 owned MC 的 **42/42** fresh 任务及其原始证据经独立复核。实际这一轮是普通非 sanitizer（Normal）构建，不是 actual SAN；隔离候选的相关 SAN/OFF 证据保留各自身份与范围。

已依赖的 Eigen 固定四路 Array 仅并行四个独立组合；每组仍按 asset 升序累加，weights 左乘的产品先具体求值，再分别更新 period 和 contribution。只在 Eigen packet 可用、最近舍入、完整唯一的连续资产贡献映射及受检几何下准入；负零、非有限或非零次正规权重、不满足完整唯一贡献映射、非最近舍入和组合数不是四的倍数的批次继续已有 canonical 计算。输入及旧结果不作 scratch，每次重读当前权重；pack 和最多128 KiB的保守 reserve 进入 Planner/审计，不新增数值缓存、线程池、公开算子或 expm1 库替换。原190注册、AST/DAG、IEEE/错误顺序、结果所有权与 Excel 数学保持。

| 对象 | SHA256 / 身份 |
| --- | --- |
| [实际compact源码](../../../cpp/compact_path.cpp) | `c7115c1acdd934d19c3bad68705a1bab88a158199c095438f034e33d78a584d4` |
| [新增权重lane回归](../../../tests/test_compact_weight_lanes.py) | `a90300283c4a07c82e9cc698d97b6b9938f409eb6c3e253ec3fd461486bf85a7` |
| actual AOT build ID | `f789954a788004453f362c5aba57f9606c3a37d3148d63875c27ae8dae853434` |
| actual package identity | `c89bb454821ddc73784af69c30bbb0a33748eaccaa15a69787b5071111c0cf53` |
| actual wheel SHA | `a37fda1adbb27b32476b678d463aaeba6dfc78f024fca1e7da10d91776b56daa` |
| actual普通全量独审 | `f5285f7347a1f4cc9353627e3a60be566a2a2b1a87dd302b58feba7ab4354437` |
| actual root最终收据 | `2e70654be5df2e507d4436ce8b19d4d95af9dc7d4ea5133b59be0ac2a9add145` |
| actual owned42独审 | `b459a99a136fb07602e962eba21808c1ccdd7ad9abbd497c52e313e7f123d693` |

root最终收据确认affected/full/owned42均通过；实际普通相关1,049、full17,179/native9，failure/error/skip和重复nodeid均0。overall_performance_accepted=false、goal_complete=false、upstream_business_migration=false继续保留。actual owned42独审核对6负载、42个串行fresh任务、126份raw/log/launch；非SAN。候选相关SAN/OFF及native证据保留候选身份，不能换称实际SAN或累加成实际全量。

## 完整测量范围

双方同 CPU1，计入相同固定生产 NJIT RNG 的独占输入准备、normal/Student 变换、投影、普通 `AdaptiveScheduler.execute`、NAV/fan、全部统计/样本/贡献及两个保留的独立结果。静态夹具/构图和编译在计时外；covariance-root 生成、coverage、limits、JSON 封装和平台按业务阶段选首错不在该测量中。算法复制与 root copy 如实计入，不能称全链 zero-copy。

两个结果owner同时保留，input hash和全部MC命名根（fan、14项统计、样本、贡献）的shape/dtype/readonly/status以及离散字段按冻结driver核验；数值使用原断言，未放宽容差。独审只读重核成功断言、输出元数据和身份，没有重新运行数值。算法复制与root copy在请求中实际计入；例如Normal C1 validation audit记录algorithm_copy_bytes=16,241,024、root_copy_bytes=368，不用boundary/output copy为0称完整零复制。

全部冻结调用均为普通 `scheduler.execute`，没有显式 prepared execution。Normal C1 RSS 的 post-epoch validation audit 为 `prepared_cached=true`，表示普通 execute 的输入绑定缓存命中；其余41份该验证 audit 为 false。验证 audit 不覆盖每次计时/缓存调用，不能称所有普通调用均无缓存。

## 六组保守时间

下表逐项取独审 `rows[i].time`，i=0..5。较快旧C++是before/after median较小者；NI3是三个时间阶段median较小者，不是最慢控制轮次。每个时间job三轮，无置信区间。

C1/C8：T200/P10000/F4/A32，分别1/8组合；C32：T5/P50000/F32/A32/C32。每轮完整owned请求的输出契约相同，不以投影内核代替完整链。

| 负载 | 较快旧 C++ ms | V99 C++ ms | NI3 ms | NI3/V99 | 相对旧 C++ 耗时变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal C1 | 468.527 | 467.608 | 518.814 | 1.110× | -0.20% |
| Normal C8 | 1459.567 | 785.024 | 3187.812 | 4.061× | -46.22% |
| Normal C32 | 1445.250 | 1135.274 | 4149.537 | 3.655× | -21.45% |
| Student C1 | 605.980 | 607.791 | 648.056 | 1.066× | +0.30% |
| Student C8 | 1600.970 | 923.810 | 3329.174 | 3.604× | -42.30% |
| Student C32 | 1466.657 | 1157.686 | 4195.090 | 3.624× | -21.07% |

## 六组内存

下面逐项取 `rows[i].heap` 与 `rows[i].rss` 的fresh内存任务；此处NJIT不是三个时间阶段的选择或平均。owned链setup、cold/warm/replacement连续记录，snapshot峰不能累加。旧CPP内存不在这份actual42汇总中，不补造。

| 负载 | cold heap：V99 / NJIT B | warm heap：V99 / NJIT B | warm V99−NJIT B |
| --- | ---: | ---: | ---: |
| Normal C1 | 65,074,729 / 96,492,616 | 65,108,601 / 96,520,816 | -31,412,215 |
| Normal C8 | 68,205,794 / 112,681,456 | 68,462,514 / 112,906,736 | -44,444,222 |
| Normal C32 | 77,076,162 / 72,476,104 | 77,207,442 / 72,528,008 | +4,679,434 |
| Student C1 | 97,086,240 / 112,492,608 | 97,120,120 / 112,520,808 | -15,400,688 |
| Student C8 | 100,281,065 / 128,681,448 | 100,537,793 / 128,906,728 | -28,368,935 |
| Student C32 | 81,349,561 / 74,476,096 | 81,535,433 / 74,528,000 | +7,007,433 |

| 负载 | prewarm baseline：V99 / NJIT B | total RSS peak：V99 / NJIT B | peak−baseline：V99 / NJIT B |
| --- | ---: | ---: | ---: |
| Normal C1 | 137,838,592 / 143,245,312 | 202,194,944 / 239,566,848 | 64,356,352 / 96,321,536 |
| Normal C8 | 143,015,936 / 138,870,784 | 213,221,376 / 268,533,760 | 70,205,440 / 129,662,976 |
| Normal C32 | 139,165,696 / 142,409,728 | 234,782,720 / 274,743,296 | 95,617,024 / 132,333,568 |
| Student C1 | 142,573,568 / 142,884,864 | 239,419,392 / 255,361,024 | 96,845,824 / 112,476,160 |
| Student C8 | 143,081,472 / 139,493,376 | 244,809,728 / 285,261,824 | 101,728,256 / 145,768,448 |
| Student C32 | 142,114,816 / 139,198,464 | 253,853,696 / 274,776,064 | 111,738,880 / 135,577,600 |

heap计malloc-family请求，未追踪释放保留，不含Python arena子分配、mmap、栈、预存Numba TLS或未拦截分配器。RSS baseline是prewarm采样基线；total peak与peak−baseline单列，0.5ms采样可能漏短峰。heap与RSS不是可相加的两个成本，也不代表完整allocator/leak审计。

## 保留异常与未完成项

Normal C8的NJIT三阶段median为3231.069667/8929.548000/3187.811875 ms；Normal C32为4149.537250/5487.536208/4180.005500 ms。参数、签名和线程caps相同的元数据已保留，波动原因尚未确立；本表仅用较快median，异常倍率不是采用依据。若发现实际条件差异需再复验，不把异常记录删除。

Normal C8/C32 的 NJIT 控制轮次漂移仍保留且原因未确立；时间取三个阶段中最快的 NJIT median（NI3），不用异常慢轮次扩大优势。C32 的 warm tracked heap 仍比 NJIT 多 **4,679,434 / 7,007,433 B**（Normal/Student）；Student C1 只比保守 NJIT 快 **1.066×**，比较快旧 C++ 慢约0.30%，不说所有负载都显著改善。LongK 的历史退化与完整 QP/前沿/探索等主要链仍开放，不据此关闭整体性能或平台业务资格。

既有[V93历史LongK负实验](../../physical-execution-optimization.md)按原身份保留，不是本轮V99复测。covariance-root生成、coverage、limits/JSON、平台阶段首错与完整上游请求未在这42项中验收。新增Microsoft Excel实机重算没有执行，187–190历史重算欠项继续保留。没有提交、推送、全局安装或平台部署结论。

## 归档入口与可追溯来源

[V99完整证据归档](mc-weight-lanes-v99.zip)的成员及SHA以归档内清单为准。归档保留候选和实际复验各自的构建、日期与成功/失败范围；不能用某一历史身份代替实际构建资格。

本 ZIP 是验收证据包，不是可独立重建的完整发行包；部分收据还绑定外部完整源码、历史安装包、SDK 与依赖。

原始收据定位：

- root最终收据：`/private/tmp/calmetrics-v99-actual-owned42-root-review-20261010-r1/reviewed.json`，SHA如上；affected/full/owned42通过，整体目标仍未完成。
- owned42：`/private/tmp/calmetrics-v99-actual-owned42-independent-review-20261010-r1/reviewed.json`，SHA如上。
- actual full：`/private/tmp/calmetrics-v99-actual-full-independent-review-20261010-r1/reviewed.json`，SHA如上。
- full pipeline：`/private/tmp/calmetrics-v99-adoption-plan-20261010/r7/actual-qualification-full-r1/pipeline.json`，SHA `ca45a9b2d3ea0c40460cdff3aedbed48efc5a5981fa7e0ed133a7df3de0822d9`。
- Python XML：同目录`python.xml`，SHA `fc05f2993100fa765c7cf4b75e021391b16da23a24aa436b305d573c4d4727c5`；native XML：`native.xml`，SHA `a25c7aed946120d80a4ccb6d592e8b06deb5982da7bcea86e6f3f936eb228373`。
- package identity：`/private/tmp/calmetrics-v99-adoption-plan-20261010/r7/actual-r2/package/identity.json`，SHA如上；build/install/wheel身份以该收据为准。

旧目录位置仅用于定位当时执行来源；长期阅读使用ZIP和冻结SHA，不把/tmp路径当部署依赖。[投影契约](../../mc-projection-primitives.md)、[执行指南](../../execution-guide.md)、[文档导航](../../README.md)保留唯一使用入口。
