# 私有拼接系数：分阶段接入与KKT完整验收（2026-10-04）

### Eigen矩形复制已接入：正式完整链与峰值复验

2026-10-05，当前实际源码、正式 wheel、扩展与 worker 身份为 `670720df09e608c3ad832e2370a4eaeb540c5e87121e317d0b976d24989d4d10`。本轮接入三处 C++ 物理复制修改和32项新增回归；已有数学、奇异阈值、错误、AST/DAG、输出布局及所有权不变，没有 Git 操作。

既有输入支持的拼接矩形在 macOS15+ arm64/Eigen 开启、正向列连续且两维至少32时，复用 Eigen 3.4.0 的32×32有界 Map 直接写入原目标。不新增矩阵、堆分配、缓存、线程或输入布局转换；其他路径保留受支持的原生复制。不是要求用户将F输入/结果转换成C。

实际正式包相关 **274项**、原生 **9/9**通过；隔离同源普通、真实ASan/UBSan和全库OFF分别相关 **274项**、原生 **9/9**通过，无failure/error/skip。覆盖偏移/填充F输入、边界、特殊IEEE位、四运输、两区间/两代独立结果及原错误反例。没有将20cf历史12,979全量或528专项当作新版本的全量通过，也没有LeakSanitizer或新的Microsoft Excel重算声明。

正式CPU1六轮交替完整KKT构造/求解，计入只读独立输入准备、普通Scheduler、计算和独立结果物化返回；NJIT已预热，编译排除：

| 8192阶输入布局 | C++秒 | NJIT秒 | C++/NJIT |
| --- | ---: | ---: | ---: |
| C | 4.944917 | 5.444061 | 0.9083 |
| F | 4.971662 | 5.455426 | 0.9113 |

此前同轮旧20cf/新6707/NJIT的F完整链为5.411993/4.793669/5.400359秒，旧/新输出逐位相等；本次正式复测见上表，不用更有利轮次替代。最大NJIT尾差约3.23e-13，原方程与约束残差通过。一列结果物理上同时C/F连续，不能据此声称多列结果布局普遍优势。

新旧完整F申请峰约持平（16字节差只作诊断），分配次数相同。正式fresh C++/NJIT F申请峰为 **1639.639/2147.899MB**，少约 **508.3MB**；各三个fresh进程的RSS增量中位为 **1674.772/2687.156MB**，少约 **1012.4MB**。内存优势主要来自已接入的拼接/私有求解工作区减少，不能归功于本轮无额外分配的复制。heap与RSS分别报告，不相加；Python arena/mmap及0.5ms采样短峰盲区保留。

本轮阶段优化已进入实际路径；**整体目标仍未完成**。完整QP、平台候选评估/随机探索、一般大求解和其他主要模型/模拟链仍需完整测量与优化，不能用该KKT任务代替。下一项优先处理候选×历史的大型中间矩阵及重复求解，避免继续围绕小任务微调。普通C++不触发Excel生成，公式/预算/公开签名不变；用户指南及算子参考无需因此修改。未提交或推送。

以下各阶段的“当前”指形成时版本，原始正负证据保留。


### 批量校验证书已接入：实际源码和正式完整链复验

2026-10-05，当前实际源码、正式 wheel、扩展和 worker 为 `20cfaf3ed42593a7596230f80fd8621261042d64620e366b37a251939e82711a`。本轮按预像接入一份 C++ 数值文件和五份测试，保留其他脏改动；没有 Git 操作。成熟 LU/BLAS 的原分解与求解不变，校验证书改为有界列组和依赖安全的批量正误差传播，复用已结束阶段的缓冲。原奇异阈值、错误回放、独立结果及逻辑 AST/DAG/Excel 公式不变。

同源原数值版本完整 Python **12,979 项**、普通原生 **9/9**，真实 ASan/UBSan 相关 **528 项**及原生 **9/9**、全库关闭相关 **528 项**及原生 **9/9**通过。随后仅修正一个测试模块对 Eigen-OFF/Accelerate-ON 的逐位浮点和复制量假设，该模块在普通/SAN/OFF各 **39 项**复核通过；Eigen-OFF相关 **528 项**及原生 **9/9**通过。没有宣称修正测试文字后再次执行12,979全套，也没有全量SAN/OFF或LeakSanitizer验收。实际源码正式包另验相关 **528 项**和原生 **9/9**，含四运输、可变prepared输入、两代只读独立结果及原错误反例。

正式串行完成 **20 项测量任务**：四组完整计时、四组连续申请峰、十二个fresh RSS进程。CPU1、相同数据/算法/输出/两代保留，计入owned readonly准备、普通Scheduler图执行及结果物化返回；NJIT实际签名已预热，编译排除。每组六轮交替，原方程stationarity/约束残差通过。这是KKT构造/求解的一项完整任务，**不能代替完整QP、随机探索或平台请求验收**。

| 系数阶数 | 输入/输出布局 | C++秒 | NJIT秒 | C++/NJIT |
| --- | --- | ---: | ---: | ---: |
| 4112 | C/C | 0.681215 | 0.640713 | 1.0632 |
| 4112 | F/F | 0.786330 | 0.720504 | 1.0914 |
| 8192 | C/C | 4.705447 | 5.265408 | 0.8937 |
| 8192 | F/F | 5.581052 | 5.352096 | 1.0428 |

8192阶C布局约快11%，F仍慢约4%；4112阶本轮慢约6%/9%。此前同20cf源隔离结果的NI比值较好，当前正式复测未稳定重现全部时间优势；原始正负报告均保留，不能改用有利轮次关闭目标。旧7a在同条件隔离对照的8192阶回放约29–30秒，优化的重大旧版收益已进入实际代码；该历史旧版结果不冒充本轮重测。

8192阶完整内存如下（十进制MB，C++ / NJIT）：

| 输入/输出布局 | 申请峰MB | RSS增量中位MB | 绝对RSS峰中位MB |
| --- | ---: | ---: | ---: |
| C/C | 1639.574 / 2147.899 | 1675.756 / 2687.844 | 1821.737 / 2828.272 |
| F/F | 1639.639 / 2147.899 | 1679.409 / 2687.795 | 1824.735 / 2826.895 |

申请峰约少508MB，RSS增量约少1GB，两种口径分别成立，不能相加。heap覆盖指定malloc家族完整fixture/准备/冷暖执行/结果保留；RSS为每后端每布局三个独立进程、0.5ms采样。Python arena/mmap与短峰盲区保留，未把未覆盖部分记零。

源码与正式包收益按当前范围结项；**全引擎目标仍未完成**。F输入的系数再生成/残差、大型一般求解，以及平台大量候选、反复QP求解、其他重要完整链继续开放。平台当前随机探索最多20万候选，前沿网格最多30资产/200目标点；不能用大阶KKT覆盖这些不同负载。普通C++不执行Excel导出，本轮没有改变公式或预算，也未宣称Microsoft Excel重新验收。未提交或推送。

[实际构建与验收证据](certificate-diagnosis/measured-integration/production-batched/acceptance.json)、[设计及自审核](certificate-diagnosis/measured-integration/review.md)。

当前实际源码/正式包为20cf批量校验证书，验收见下节。之后各阶段原始记录按其形成时版本保留；其中“当前”不表示最新身份。

3d7d已按精确源码预像接入当前目录，并重新构建、复验实际wheel与worker；15ea保留为前阶段对照。仅以[三行物理资格补丁](graph.patch)允许既有concatenate的非公开、单消费者、独占C系数arena进入原canonical求解，不扩大成熟LU的DD准入域。输入、原1e-14阈值、错误顺序、结果所有权和Excel数学不变；Planner保留完整回放上界。

14项初始夹具在执行前因缺少共同symbolic shape声明失败，修正声明后14通过；相关781和原生9/9通过，再补负步长/后续消费者/isolate后专项20通过。原普通全量12676通过、59项因漏设显式Eigen验收开关而跳过，控制器随后在零skip断言终态1；原始XML/日志/状态保留。补跑完全相同的59个node ID全部通过，共12735个不同普通用例通过。真实SAN/OFF新路径各20项、原生各9/9通过；这是新增路径的SAN/OFF范围，不冒充全部12735项的SAN/OFF。原始失败及版本均保留，[自审](self-review.md)明确各阶段范围。

19项完整测量串行结束：三组六轮交替时间、12组新包/NJIT连续heap与fresh RSS、四组旧15ea内存控制。CPU1、相同dense SPD方向块和16个独立约束；2048/4096变量分别产生2064/4112阶系数，计入owned readonly输入准备、普通Scheduler拼接/求解及独立结果返回；旧/新两代按相同规则保留，NJIT实际签名已预热，编译不计入。stationarity及活跃约束残差独立通过，候选与原canonical结果逐位一致。范围为KKT构造和求解的一项完整任务，**不是完整QP或平台请求迁移**。

| 变量数/布局 | 旧C++秒 | 新C++秒 | NJIT秒 | 新申请峰MB / NJIT | 新RSS增量MB / NJIT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048 C/C | 0.336 | 0.342 | 0.090 | 135.888 / 136.468 | 136.331 / 171.000 |
| 4096 C/C | 2.789 | 2.733 | 0.743 | 540.179 / 541.309 | 542.376 / 680.641 |
| 4096 F/F | 2.931 | 2.754 | 0.737 | 540.179 / 541.309 | 542.720 / 679.789 |

4096阶旧申请峰为675.448MB，候选少135.268MB；RSS比旧包少约134–136MB，比NJIT少约137–138MB。但requested heap仅比NJIT少约1.13MB，不能宣称该维度显著领先；**耗时仍约为NJIT的3.7倍**。2048相对旧C++多约5.95ms只作诊断，不掩盖4096的实质时间差距。F/F为声明输出布局，单列结果本身同时满足C/F连续，不能将差异归因于独立的输出布局收益。

原始[任务状态](measure-kkt-status.json)、[旧包内存控制](measure-old-kkt-status.json)、[逐项验收](acceptance.json)、[构建身份](candidate-layout/identity.json)和[源码复现](source-reproduction.json)保留精确数据、轮次及哈希。heap覆盖指定malloc家族，RSS以0.5ms采样补证；Python arena/mmap及短峰盲区明确，二者不相加。2064阶[运行复制审计](audit-kkt-after.json)另列，只能证明算法复制/Workspace下降，不能替代完整内存测量。

下一优先级为现有一般/非DD求解的成熟原生库准入与失败回放证明。当前DD/identity-pivot证书主动排除该KKT域，4112阶还超出现有成熟LU的4096上限；不能删除检查来取得表面提速。该候选不新增金融算法、公共算子、线程池或跨请求数值缓存；未提交/推送，整体目标未完成。

## 实际目录接入与正式包复验

仅替换graph.cpp资格判断、增加20项测试，其余87份数值/配置源码与Python接口不变。实际目录重建包为同一3d7d源码身份，相关787项、原生9/9通过；single/thread/process-inline/process-shared四条路径均验证两行、两代独立结果、只读输入、零求解Workspace及每行只复制RHS。新增优化不依赖隐式线程或缓存。

普通Scheduler、4096变量/16约束、F/F、CPU1、六轮轮换的正式旧/新/NJIT为 **2.658/2.714/0.681秒**。本轮较旧包多56ms（约2.1%），不宣称稳定时间收益；相比NJIT仍约慢4倍。正式热申请峰新/NJIT为 **540.179/541.309 MB**，较前阶段同条件旧包约少135.268 MB；fresh RSS增量 **542.147/677.544 MB**，比NJIT少135.397 MB。内存大额节约复现，但申请峰对NI基本持平，不能称整体目标完成。仅一个fresh RSS进程/后端，不能冒充多次fresh进程置信区间。

[接入记录](integration/adoption.json)、[实际源码与二进制身份](integration/production/identity.json)、[逐任务终态](integration/verify-root-status.json)、[完整时间轮次](integration/root-complete/timing.json)和[四路径所有权核验](integration/transport-probe.json)分别列出证据。完整normal/SAN/OFF来自同源隔离包；正式重建包另外跑相关787、原生9及运输、完整测量，不冒充再次跑全部或SAN。

## 一般LU准入诊断

使用相同KKT输入及原生因子/证书源码作[独立诊断](certificate-diagnosis/diagnosis.json)，只在诊断中统计行交换而暂时省略identity限制，不改生产准入。1040/2064/4112阶分解INFO均为0，但原逆范数上界与误差的乘积分别为24.7、4123.9、511440.3，全部无法满足小于0.25的保护。取消行交换限制也不会通过。

DGECON返回的逆范数估计约3.01–4.12，只用于分析；[LAPACK官方说明](https://www.netlib.org/lapack/explore-html/d4/daf/group__gecon_ga4f9b830e19e12c7f082ddb497a57af18.html)将它定义为估计，不能单独当严格上界。即使诊断代入该估计，后续[全局Schur界](certificate-diagnosis/diagnosis-schur.json)仍远超主元间距。单换范数估计不足以放行，以上原始负结果保留。后续[局部误差准入原型](certificate-diagnosis/local-prototype/design.md)已完成按行/列块的系数、RHS和回代有限性筛选：普通及真实ASan/UBSan各145组（73准入/72回退），四个历史阈值反例均回退；1040/2064/4112阶KKT通过当前原生求解对照，最大差3.42e-14。4112阶一次组件测量分解/保护约0.227/0.230秒，仅为原型组件诊断。实际DAG接入、原系数有界再生成与回放、初始库误差模型复核和完整链性能/内存均未完成；当前生产仍fafa，未扩大成熟库准入域。

## 保序内核筛选：均未采用

以同一批2064/4112阶KKT系数作单核求解入口筛选，复用当前Eigen运算体，比较集中列块、额外17N doubles的F跨度padding，以及原C内核的128/256/512列面板。所有候选保持升序减法、主元规则、即时RHS更新和原回代；不以放宽错误契约提速。

| 4096变量/C输入的求解入口 | 中位秒 | 同轮原C内核秒 | 同轮NJIT秒 |
| --- | ---: | ---: | ---: |
| 8行×4列集中F列块 | 5.286 | 2.422 | 0.573 |
| 同列块，F跨度增加17 | 6.043 | 2.417 | 0.578 |
| 128列C面板 | 2.406 | 2.403 | 0.611 |
| 256列C面板 | 2.782 | 2.403 | 0.611 |
| 512列C面板 | 3.581 | 2.403 | 0.611 |

列块两个版本的99项值/错误/只读输入/独立结果控制分别通过；行面板67项边界控制覆盖全部三个候选至1025阶，大规模记录另显式核验512对32的逐位一致性。128列的几毫秒差只作诊断，没有采用理由；其他候选在大矩阵上更慢。padding未改善，不据此宣称已经用硬件计数器证实缓存或TLB原因。

这些是独立求解入口筛选，计入系数/RHS复制、消元、回代和独立结果；未计入KKT构造、owned输入准备、DAG/Planner和运输。不能代替前述正式完整任务，也不拿局部结果更新整体验收。NJIT实际readonly签名已预热，各后端使用相同数据和单核，轮换计时；原始轮次和版本见[筛选记录](ordered-kernel-screen/acceptance.json)。全部拒绝，因此不额外运行峰值内存或接入验收；生产数值源码仍为3d7d。下一步集中在更广的成熟LU准入/错误回放证明及其他实质计算链欠项，停止微调这些无收益候选。

## 嵌套拼接融合：已接入，修正版完整验收与正式复测结束

将私有嵌套concatenate的输入/借用视图直接复制到最终C矩形；中间逻辑节点、原位置prepare校验、AST/DAG与Excel步骤仍保留。只延迟非公开、单消费者的float64拼接，axis必须为常量，末端叶子须由输入支持。公开计算结果不能因暂时标记persistent就延迟读取，F结果收尾可能恢复arena。程序含scope时不选；合格平面子程序只使用本次稳定的bound输入。observer、deferred-error和isolate保留原物化。正向F矩形32×32分块，其他signed/zero strides按原坐标读取；F结果沿用原独立返回，不要求输入改为C。没有新算子、数学内核、数值大缓冲、池或跨请求缓存；C++中间区减少不代表Excel预算下降。

未分块候选使完整F/F KKT由2.656秒退化到3.032秒，已拒绝。分段诊断中F准备/拼接260→637ms，求解约2.4秒；未用硬件计数器证明缓存原因。32×32分块后的c5ff历史候选耗时2.457秒、申请峰404.977MB、RSS增量406.520MB，但raw native探针发现int/mask输入的GRAPH_STORAGE_KIND延后至独立DOMAIN_ERROR之后，故C5没有接入。其12,891全量及SAN/OFF通过仍保留为历史证据，不转用于修正版。

修正版fafa在原deferred节点补回容量和numeric storage校验。原生回归验证首错及canonical失败缓冲区字节；首次测试错误要求失败buffer保持sentinel，原行为会初始化失败槽，已修正为原字节对照，未改变运行时或放宽契约。fafa同源普通全量 **12,891项**、原生 **9/9**通过；真实ASan/UBSan和全库OFF新增路径各 **156项**、原生各 **9/9**，全部零失败/错误/skip，未宣称全量SAN/OFF或LeakSanitizer验收。

按88份数值/配置精确预像核对后，仅接入graph.cpp、graph.hpp和两份测试。实际目录重建wheel、扩展和worker身份均为 `fafa1da26cdab07fbf59f519ac76dc1524f1a689fc7622c5e14d818297450f98`；其Python接口及全部测试源码与同源全量一致。正式包另验相关 **943项**、原生 **9/9**，single/thread/process-inline/process-shared四路径验证两行、两代结果、关闭Scheduler后存活、只读与输入不变。正式包没有重复执行全部12,891或SAN/OFF；两层证据范围分别说明。

26个串行正式测量结束。4096变量、16约束、4112阶系数；CPU1，相同数据、owned readonly输入准备、普通Scheduler拼接/求解、逐输出物化和结果保留方式，NJIT完成实际签名预热，编译不计入。每布局六轮三方轮换时间、各后端热申请峰及三个独立fresh RSS进程；所有实际分配、复制和调度计入。范围是KKT构造/求解一项完整任务，不是完整QP或平台请求。

| 布局 | 旧3d C++秒 | 当前fafa秒 | NJIT秒 | 当前/NI申请峰MB | 当前/NI RSS增量中位MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| C/C | 2.512 | 2.426 | 0.791 | 404.944 / 541.309 | 406.995 / 678.904 |
| F/F | 2.668 | 2.470 | 0.729 | 404.977 / 541.309 | 407.241 / 678.887 |

比旧包少约135.2MB申请峰，比NI少约136.3MB申请峰和271.6–271.9MB RSS增量，三个fresh进程均复现。新旧结果逐位一致，NI最大尾差约1.45e-13；独立stationarity和约束残差通过。heap是指定malloc家族连续峰值，RSS为0.5ms采样的进程增量，仍有Python arena/mmap及短峰盲区，二者不相加。单列结果同时C/F连续，不能据此归因独立输出布局收益。**求解耗时仍约为NI的3.1–3.4倍，整体目标未完成。**

[逐项验收](tree-fusion/acceptance-snapshot.json)、[当前接入记录](tree-fusion/integration/adoption.json)、[正式构建身份](tree-fusion/integration/production/identity.json)、[正式验收终态](tree-fusion/integration/verify-root-status.json)、[测量终态](tree-fusion/integration/measure/status.json)、[C完整轮次](tree-fusion/integration/complete-c/timing.json)、[F完整轮次](tree-fusion/integration/complete-f/timing.json)、[四运输路径](tree-fusion/integration/transport-probe.json)与[首错反例](tree-fusion/error-probe-before.log)分别标明身份和范围。[复现补丁](tree-fusion/candidate.patch)以旧3d为基线；源数据和输出二进制不入库，证据哈希见[清单](tree-fusion/manifest.json)。公开数学签名、registry/Typed IR、结果/IPC协议及Excel公式不变，使用手册与算子参考无需改写；执行指南已同步物理资格。

下一重点仍为一般/非DD求解的成熟库安全准入及其他重要完整计算链的当前构建复验。不能删除奇异阈值、主元或有限性保护来取得提速。修改仅在本地工作树，未提交或推送。

### 前阶段隔离测量（历史记录）

该阶段实际数值源码为 fafa。隔离候选 `7a6f446c` 已贯通现有 solve_multi、私有拼接矩形再生成、错误回放与 Planner 工作区预算；没有新 opcode、Python 数值回调、跨请求分解缓存或 Excel 公式变化。初始误差改为完整测量 P*A−L*U，并加上明确的常规 FP64 BLAS 产品误差模型；不再假设厂商 LU 的分解误差上界，但不能据此宣称所有厂商 BLAS 的形式化误差保证已经成立。主元、原 1e-14 奇异阈值及有限性仍受保护，不确定时恢复原系数并执行 canonical。终端浮点结果另检查原方程残差；这不是病态问题的普适前向误差保证。

候选相关490项、同源原生9/9通过。新增套件扩为66项，增加C/F、single/thread/process-inline/process-shared、两区间、两代结果及关闭后的寿命检查；8个故障传输用例另与fafa核对，保留本地ValueError及进程RuntimeError/原消息。全量Python 12,957项已确认通过，无失败或跳过；该记录形成时相关506的真实ASan/UBSan和全库OFF及各原生9/9尚未终态，候选尚未接入；这些状态已由下方正式证据更新，历史原始测量不改写。

4112阶KKT完整任务、CPU1、相同只读输入准备/普通图/独立结果返回/保留方式、实际NJIT签名预热、编译排除，每布局六轮轮换：C当前/候选/NJIT为2.447/0.834/0.744秒，F为2.496/0.944/0.759秒。原方程stationarity及约束残差通过。候选申请峰约417.8MB，NI541.3MB；各布局三个fresh进程的RSS增量中位C436.9/F435.0MB，对NI678.9/679.1MB，绝对RSS峰中位约584MB，对NI818–819MB。内存控制复用紧接此前、相同输入身份/输出/保留方式/CPU1/测量方法的独立进程；其范围和盲区分别记录，heap与RSS不相加。候选较fafa申请峰增加约12.9MB，较NI减少约123.5MB；较当前C++耗时减少约62–66%，但仍比NI慢约12–24%。不代表完整QP、平台请求或整体目标已达成。

更宽的重构库调用版本 `872fd1a5` 相关506项通过，但完整C/F耗时0.857/0.977秒、NI0.625/0.710秒，申请峰432.6MB，比原分块多约14.75MB；没有显示时间收益，已拒绝采用。保留其全部原始负结果。原型中的真正原地Eigen LU也因大阶分解较慢未采用。该宽版本没有接入，后续实际接入和复验见下方。

[设计](certificate-diagnosis/measured-integration/design.md)、[自审核与数学限定](certificate-diagnosis/measured-integration/review.md)、[候选补丁](certificate-diagnosis/measured-integration/candidate.patch)、[C完整计时](certificate-diagnosis/measured-integration/complete-c/timing.json)、[F完整计时](certificate-diagnosis/measured-integration/complete-f/timing.json)、[内存摘要](certificate-diagnosis/measured-integration/v2-memory/summary.json)、[运行中验收快照](certificate-diagnosis/measured-integration/validate-tiled-status-running.json)与[证据哈希](certificate-diagnosis/measured-integration/manifest.json)均明确隔离状态；快照不能替代后续进程终态。

### 测量 LU 重构已接入：正式完整链复验结束

实际源码/正式包、扩展与 worker 现为 `7a6f446c76ea4b5edd6f49389192b62b918d830a06c03bcd3da23d12695397d1`。仅按预像替换三份数值文件和两份测试，核对全部88份源码、测试树、Python树及HEAD，保留其他共享脏改动，没有Git操作。现有私有拼接的同步矩形再生成、完整P*A−L*U测量、原主元/有限性保护和原系数恢复进入实际路径；常规FP64 BLAS乘积模型及病态前向误差边界仍明确。

同源普通全量 **12,957项**、原生 **9/9**，真实ASan/UBSan和全库OFF相关各 **506项**、原生各 **9/9**，全部通过，无失败/错误/skip。实际目录正式重建包另验相关 **1,009项**和原生 **9/9**，含四运输路径、两区间、两代保留和关闭后的结果寿命。SAN/OFF是相关覆盖，不声称全量SAN或LeakSanitizer通过。

18个正式串行测量任务终态0；CPU1、相同输入/算法/输出/核数/两代保留，普通完整链含owned readonly准备、图执行和独立结果返回，已预热NJIT且排除编译。4112阶KKT每布局六轮轮换，C旧fafa/当前/NI **2.861/0.969/0.695秒**，F **2.646/1.032/0.772秒**。当前比旧C++快61%–66%，仍比NI慢34%–39%；stationarity/约束残差≤1e-10。完整QP、平台请求与其他重要链仍开放。

requested heap峰当前/NI：C **417.818/541.309 MB**、F **417.851/541.309 MB**，少约123.5 MB；三fresh进程RSS增量中位：C **435.995/678.789 MB**、F **444.760/678.347 MB**，少约234–243 MB。绝对RSS峰中位：C **582.861/817.447 MB**、F **590.512/817.316 MB**。两种口径分别报告，allocator/0.5ms采样盲区保留，不相加。前阶段同方法fafa申请峰约405MB，本轮未重测该旧heap控制，不能混称。

原始计时报告的production_adopted=false为模板残留；实际源码/安装/二进制来源独立核验为已接入。原始样本保留不改，由[来源更正](certificate-diagnosis/measured-integration/benchmark-provenance-correction.json)说明。组件配对筛选中，单借用DGEMM前缀+DTRMM仅节约约20ms（6%局部重构），没有完整任务收益证据，未采用；继续优先降低主要重构校验成本。

[正式来源与接入](certificate-diagnosis/measured-integration/production-adoption.json)、[完整质量终态](certificate-diagnosis/measured-integration/validate-tiled-status-latest.json)、[实际目录复验](certificate-diagnosis/measured-integration/verify-root-status.json)、[C完整计时](certificate-diagnosis/measured-integration/production-tiled-complete-c/timing.json)、[F完整计时](certificate-diagnosis/measured-integration/production-tiled-complete-f/timing.json)、[正式内存](certificate-diagnosis/measured-integration/production-tiled-memory/summary.json)、[自审核](certificate-diagnosis/measured-integration/review.md)及[证据哈希](certificate-diagnosis/measured-integration/manifest.json)限定到对应源码与实测范围。整体目标未完成，未提交或推送。
