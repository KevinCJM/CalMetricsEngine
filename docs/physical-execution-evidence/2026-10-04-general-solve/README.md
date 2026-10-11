# 通用求解与公共C/F根：分阶段优化证据（2026-10-04）

候选ac54已构建并在实际安装包路径完成710相关回归、9/9原生回归及40个串行完整任务。**ac54阶段的独立构建未直接采用；后续15ea、3d7d及当前fafa嵌套拼接融合已接入并正式复验（见下文），未提交/推送，整体目标未完成。**同源候选全量12,658项通过，零失败/错误/skip；真实ASan/UBSan与关闭库OFF的相关回归各710项、原生各9/9通过。OFF初次有一项复制审计预期失败，保留原终态；只修测试断言后，normal对应7项及OFF全710项复跑通过。normal全量与SAN完成于该测试断言修正前，88份数值源码及包未变。

## 已验证的收益与缺口

冻结同一4096阶行对角占优非对称矩阵、8个RHS；ordinary Scheduler、owned readonly输入准备、独立结果物化及两代保留；NJIT预热后，相同单核和C/F输出，六轮轮换串行。编译排除，输入复制/调度/结果包装均在调用内。fixture和图setup另报告，连续申请峰与fresh RSS包括它们。

- 直接输入四种布局：候选0.409–0.509秒，NI0.599–0.830秒；旧C++2.361–2.470秒。
- 公共系数四种布局：候选0.443–0.643秒，NI0.645–0.882秒；旧C++2.416–3.720秒。保留初始候选公共F的1.270秒/NI0.610秒反例；同F顺序和双向有界transpose tile复用唯一canonical逐项数学体。
- 私有F系数/F输出：候选0.422秒，NI0.671秒；申请峰cold692,262,535/809,276,384 bytes、warm692,526,367/809,538,640 bytes。RSS增量695,943,168/811,892,736 bytes。该链约快37%，申请峰少约117MB。
- **直接/公共retained系数路径申请峰仍比NI高约17MB，RSS也更高。**这是未闭环项，不用小差值规则豁免，也不宣称整体时间/内存通过。小63数据的微秒差只记录。

两种现成LAPACK入口的分解段筛选：DGETRF2约少10.6MB申请峰，但warm分解+求解约0.410秒，DGETRF约0.322秒；未用于候选或生产，仅供下一阶段选库诊断，不是完整任务验收。

## 范围与复现

[设计](final/design.md)、[自审](final/self-review.md)、[构建身份](final/candidate-layout/identity.json)、[逐任务状态](final/measure-layout-status.json)及[验收状态](acceptance.json)。raw timing/memory保留完整轮次、审计、输入hash、核数、错误和边界；heap/RSS不相加，RSS为0.5ms采样补证，heap盲区明确。大型npz临时输出不入文档归档，位于原隔离目录；种子/输入hash、复现脚本、数学核验和完整文本源码快照保留。

S阶段最初测试失败、布局脚本失败、公共F性能失败及全部正负结果保留在initial/；ab2/a226阶段身份和相关终态在intermediate/。该阶段冻结时根924保持前一阶段12,487全量通过的实现；前阶段15ea/3d7d及当前fafa接入证据见下文。矩阵输入可不写入、公共/共享系数可不破坏、错误阈值/顺序和旧快照独立性是必须条件。静态双根、独立operator接口、非DD/更大矩阵及更多平台常用组合任务仍有重要待办。

[最终SAN/OFF逐任务记录](closure/validate-layout-status.json)保留初次OFF失败；[OFF修复复跑终态](closure/resume-off-status.json)、[仅测试断言补丁](closure/off-audit-test-only.patch)和[测试版本范围](closure/test-correction.json)另列。初次OFF把F根契约误当成物理F求解，忽略C暂存可复用系数区；数值/输入/所有权检查已通过，不是通过放宽数学验收修复。

## 后续隔离候选096e：分块LU与失效因子区复用

[设计](panel/design.md)与[自审](panel/self-review.md)记录唯一C++变化：按Netlib成熟库分块顺序限制factor工作区，并在DGETRS与证书完成后将不再消费的独占N²因子区用于残差。实际Normal包88数值源一致，726项相关通过；4个完整时间和16个heap/RSS任务串行结束。直接C/F、公共F、privateF耗时新/NI约0.573/0.692、0.476/0.579、0.485/0.681、0.490/0.651秒。privateF申请峰约676.055/809.539MB，少133.484MB；相对ac54申请峰少约16.5MB但时间多约30–60ms，取舍如实保留。

retained申请峰差缩到约0.47–0.73MB，RSS仍比NI多约11–21MB；不宣称全部内存通过。仅上述4种完整布局已复测，其他布局沿前阶段证据但不能冒充新候选实测。同源全量normal **12,674通过**，真实ASan/UBSan与OFF相关各 **726通过**、原生各 **9/9通过**，零失败/错误/skip；[串行验证终态](panel/validate-panel-status.json)及对应原始日志/XML已归档。096e阶段冻结时root924未采用；后续15ea已包含这些改动，所有重要主负载仍开放。[逐任务测量终态](panel/verify-measure-status.json)与[准确范围](panel/acceptance-running.json)，复现基于原完整快照、OFF审计补丁和本阶段唯一数值文件/新增16测试。


## 后续隔离候选15ea：共享私有矩阵的双求解根

[设计](shared-pair/design.md)、[自审](shared-pair/self-review.md)及[实际构建身份](shared-pair/candidate-layout/identity.json)。仅修改四份既有C++文件，在同一受护库核心之上将两个terminal solve_multi根的RHS合并，借用两根独占的私有系数区，只做一次分解；成功后写入两份独立C/F结果，失败按原物理顺序从persistent叶子重建系数再正常执行。逻辑算子/AST/DAG、canonical错误阈值、数学/Excel契约、IPC不变。实际Workspace为2NK+5N，Planner仍保守覆盖canonical失败回放的N²，不虚称降低最坏准入预算。

767相关回归（新增41）、原生9/9通过；4组完整时间、16组heap/RSS串行终态0。所有输入准备/两根执行/结果物化计入，CPU1、readonly签名预热、六轮轮换；另加入同样合并RHS只分解一次的NJIT对照，避免把其重复分解当作全部优势。4096 C/C、F/F、F/C三链生产root924约4.757/4.884/4.937秒，15ea约0.594/0.494/0.476秒，联合NJIT约0.761/0.659/0.650秒；候选比联合NJIT快约22–27%。warm申请峰新约676.056–676.318MB、联合NI809.801MB，少约133–134MB；RSS增量少约119–130MB。2048 C新0.103秒对联合NI0.093秒，约9.7ms劣化保留为诊断；申请峰170.274/203.585MB。所有原始正负轮次及数学/所有权检查保留在[准确范围](shared-pair/acceptance-running.json)和[逐任务终态](shared-pair/verify-measure-status.json)。

15ea同源normal全量 **12,715通过**，真实ASan/UBSan与OFF相关各 **767通过**、原生各 **9/9通过**，均无失败/错误/skip；原始日志/XML及逐任务终态已归档。已按精确预像接入当前源码，仅替换四份C++、更新三份测试、增加四份测试，其他84份数值/配置文件与Python接口不变。重新构建的实际root包身份仍为15ea，767相关和原生9/9通过；4096 F/F完整链六轮复验为 **0.516/0.645秒**（新/联合NI），约快20%；warm申请峰 **676.318/809.801MB**，少133.483MB，fresh RSS增量 **702.841/816.497MB**，少113.656MB，实际每行仅一次LU。见[正式包串行终态](shared-pair/integration/verify-root-status.json)和[完整时间原始轮次](shared-pair/integration/root-complete/timing.json)。同源全量/SAN覆盖与重新构建后的正式复验范围分别列出，不冒充正式包再次跑全量或SAN。非DD/KKT、retained RSS、更多主要计算链及整体目标继续开放。复现基于前阶段源档/测试补丁、panel单文件及本阶段四份替换文件/新增41测试；不归档大型npz或二进制重复副本。


## 后续已接入3d7d：私有拼接系数与KKT时间缺口

[设计、自审与19项完整测量](concat-private/README.md)表明，在现有canonical求解中复用死拼接系数，4096阶较15ea减少约135MB申请峰、比NJIT少约137–138MB RSS增量；requested heap与NI基本持平。但C/F完整任务仍为2.733/2.754秒对NI0.743/0.737秒，约慢3.7倍。该重要负载没有通过时间验收，整体目标不关闭。12735个不同normal用例通过（原12676+补跑59），真实SAN/OFF新增路径各20及原生各9/9；实际目录已接入并重建3d7d，相关787、原生9及四条运输路径通过。正式完整F/F旧/新/NI为2.658/2.714/0.681秒，新/NI申请峰540.179/541.309MB、RSS增量542.147/677.544MB；内存改善复现，耗时仍慢约4倍。一般LU误差证书诊断另列，不冒充完整QP迁移或全域成熟库覆盖。
