# 文档导航

本文档集对应当前源码的 190 个 canonical 算子。引擎版本号相同不代表构建内容相同；部署时同时核对构建身份、算子目录和计划凭据。

## 使用者从这里开始

| 任务 | 阅读入口 |
| --- | --- |
| 了解标量、时序、向量、矩阵；准备输入；构建和运行 DAG | [用户使用手册](user-guide.md) |
| 查找全部算子、签名、输出形状和数学逻辑 | [数学算子参考](operator-reference.md) |
| 了解何时使用 SIMD、线程、进程、协程和共享内存 | [执行与性能指南](execution-guide.md) |
| 直接调用算子、复用 out/Workspace、理解借用视图 | [算子接口与内存契约](canonical-operators.md) |
| 接入逐指标错误、快照和执行凭据 | [平台执行契约](platform-execution-contracts.md) |
| 深入了解状态编码、确认、峰谷及区间边界 | [状态与事件契约](state-event-contracts.md) |
| 从 AST/DAG 导出 Excel 步骤、公式和 C++ 对照 | [Excel 导出使用说明](excel-export.md) |

首次接入建议依次阅读使用手册第 1–5 节、执行指南第 1–3 节，再查所需算子。算法数值契约以当前原生实现及其回归测试为依据；发现文档与实现冲突时先核查，不凭文档静默改变数学逻辑。

## 维护与实现

本次源码提交保留已采用优化的设计、摘要及所链接的文本证据。未分发的本机历史原始归档已在引用处标明，文件哈希见 [归档清单](physical-execution-evidence/local-archives.json)；原始大数组、wheel 和编译缓存不进入源码分发。历史证据中的本机绝对路径不是可下载地址，也不代表当前构建已重新执行。

- [V99 组合权重四路保序计算：实际 owned MC 复验](physical-execution-evidence/2026-10-10-mc-weight-lanes-v99/README.md)：只接入compact源码与新回归；实际f789954a普通全量17179/native9及42/42独审通过。Normal/Student C8为785/924 ms，保守NI3为3188/3329 ms，4.06/3.60倍，相对旧C++省46.22%/42.30%。Normal C8/C32的NI漂移、C32多4,679,434/7,007,433B堆峰、Student C1仅1.066倍及普通绑定缓存命中均保留；不称actual SAN、零复制或整体目标完成。
- [V93 Eigen列密度扩展](physical-execution-evidence/2026-10-09-latent-current-v69/future-result-screen-v74/qualified-adoption/row-state-reuse-v78/README.md#density-eigen-v93)：现有F密度8→16成分，同源全量17130、相关SAN/OFF各2628/native各9及49份完整记录已核对。实际重建身份79b494c5一致，另相关2628/native9及18项actual完整记录独立审核通过；GMM/HMM比预热NJIT快8.11/3.46倍，总RSS分别低45.13%/47.05%。非零RSS增量及长K负实验保留，整体目标开放。
- [V91 接入与复验](physical-execution-evidence/2026-10-09-latent-current-v69/future-result-screen-v74/qualified-adoption/row-state-reuse-v78/README.md#gaussian-filter-stream-v91)：源码接入自隔离候选 723c06b3；普通全量 17066、相关 SAN/OFF 各 2385/native 各 9 已通过。实际目录重建、相关 Python 2385/native 9 已通过，实际四组完整C/F复验36个fresh已通过，13输出、两代所有权及F宽链相对旧C++的权衡保留。整体目标未完成。

- [V85 历史高斯/filter尾行融合](physical-execution-evidence/2026-10-09-latent-current-v69/future-result-screen-v74/qualified-adoption/row-state-reuse-v78/README.md#gaussian-filter-stream-v85)：V85 阶段的实际0ebc源码及AOT包已接入；R4普通/SAN/OFF各1353/native9、实际另1353/native9及36个新完整配对测量通过。750k完整C++/NI约2.369/5.167秒，较旧C++少9MB堆；250k堆仍高2.03MB，长K及完整求解/前沿/探索目标开放。

- [V69当前完整隐状态链复验](physical-execution-evidence/2026-10-09-latent-current-v69/README.md)：实际a87f2c25包完成40个fresh测量，三组时间及全部13输出/所有权核验通过，四组省时53%–72%。默认三状态HMM/GMM申请峰仍高于NJIT14.82/20.06MiB；另8项连续视图实验中大GMM多约40MiB，未采用。[V70隔离HMM私有融合](physical-execution-evidence/2026-10-09-latent-current-v69/fusion-followup-v70/README.md)原生9/相关Python701及30个完整time/heap进程通过，HMM250k/750k少约6/18MB，但仍高于NJIT。共享标准化负实验中大GMM多约45MB；候选未采用。[V71逐节点诊断](physical-execution-evidence/2026-10-09-latent-current-v69/arena-diagnosis-v71/README.md)已定位相邻融合受阻，简单按大小best-fit反而增大GMM容量；[V72](physical-execution-evidence/2026-10-09-latent-current-v69/gmm-replay-screen-v72/README.md)/[V73](physical-execution-evidence/2026-10-09-latent-current-v69/gmm-replay-eigen-screen-v73/README.md)虽少约9MB，完整耗时仍有实质退化，均未采用。后继验证未写入的未来结果区复用，整体目标开放。
- [V68输入布局与LU证书优化](physical-execution-evidence/2026-10-08-general-solve-v68/README.md)：R5隔离完整84项测量通过，8192 C/F比同轮强NJIT省时约27%、申请峰少约491MiB。正常全量16808、ASan关联5522、OFF关联5521通过/1环境skip及三套原生9/9通过；2份C++及3份测试已接入，实际目录重建身份一致、另5522关联及原生9/9通过，52项实际完整复验已通过，8192 C/F省时26.87%/26.55%、申请峰少约491MiB。R1–R4的失败和较弱收益按版本保留，整体目标未完成。
- [V67通用LU完整筛选（未接入）](physical-execution-evidence/2026-10-08-general-solve-v67/README.md)：56个串行fresh时间/申请峰/RSS任务、native9与相关1107通过。更强直接构造NJIT下，当前8192申请峰少约508MB但时间仍接近；vendor DGETRF没有明显稳定提速且增加14–30MB，因此保留当前199d成熟分块路径。直接F系数填充正在隔离验收，整体目标未完成。
- [当前V66有界私有矩阵优化](physical-execution-evidence/2026-10-07-matmul-terminal-proof-v63/multiroot-followup-v65/sgemm-followup-v66/README.md)：完整数组恢复R2已接入，身份199d1135…；同源全量16796与SAN/OFF专项、actual 4825/native9及20fresh完整请求通过。受限双矩阵根C/F比同轮强NI快约24%/31%，申请峰与V65持平，无显著NI内存优势；一般矩阵、求解、GMM及整体目标仍开放。
- [当前V33独立行、混合根及矩阵布局接入](physical-execution-evidence/2026-10-05-candidate-summaries/quantiles/row-risk-design/mixed-roots-v25/input-bounds-v28/packed-f-successor/integration-v33/README.md)：同源全量15062、SAN/OFF及三套native通过，实际另验1195与普通入口证明；70组/420计时、280fresh峰值闭合。派生三风险约0.20秒、原实际约1.75秒、NI约0.27秒，申请峰3.4GB→30MB但仍高于NI。F矩阵结果少约138MB，高量级/非整块仍慢约9–12倍，整体目标开放。
- [私有矩阵统计融合与正式复验](physical-execution-evidence/2026-10-05-candidate-summaries/integration/README.md)：c605阶段已接入，同源完整13,074项、真实SAN/OFF各254及原生各9/9；正式包另验254及9/9。10万/20万候选完整链约7.7/14.6ms，流式NJIT约2.17/4.32秒；10万相对旧C++申请峰少约2.013GB，但仍比NI多3.602MB。同算法NI也进入低毫秒范围，小差只作诊断；完整随机探索、其他主要负载与整体目标继续开放，前阶段筛选保留。
- [通用求解与共享私有系数](physical-execution-evidence/2026-10-04-general-solve/README.md)：15ea已接入，同源全量12,715、真实SAN/OFF相关各767、原生各9/9通过；正式重建包另验767及原生9/9。4096 F/F双求解完整链0.516/0.645秒（C++/联合NJIT），申请峰少133 MB、RSS增量少114 MB；历史ac54/096e负结果保留。非DD/KKT、retained RSS和其他主要负载仍开放，整体目标未完成。
- [私有拼接系数与KKT完整对照](physical-execution-evidence/2026-10-04-general-solve/concat-private/README.md#eigen矩形复制已接入正式完整链与峰值复验)：6707阶段已接入Eigen有界矩形复制，正式源码/包/worker一致。实际相关274/native9，同源SAN/OFF各274/9通过；8192 C/F正式完整链4.945/4.972秒，NJIT5.444/5.455秒。F申请峰少约508MB，RSS增量少约1012MB；其他主要负载及整体目标未闭环，旧全量与负结果按原版本保留。
- [全部计算算子复验](operator-acceptance.md)：保留185个算子的历史全量重算及性能未通过项；187项阶段的共享投影与新增公式边界见[投影原语](mc-projection-primitives.md)，当前188–190及完整固定带消费者链见[按轴原语](mc-axis-primitives.md)。不能沿用185项的Microsoft Excel PASS。
- [编程与数学计算纪律](computation-design-rules.md)：相关设计、开发与审核的必读规则；覆盖成熟原生库优先、依赖、复用、矩阵结构、状态、数值等价及性能证据，不代表待开发能力已经实现。
- [Excel 公式生成的数学与编程纪律](excel-generation-rules.md)：相关开发与审核必读；规范数学映射、真实可见步骤、状态/迭代、预算、流式写出、所有权、真实重算与变更联动。
- [架构](architecture.md)：模块职责、编译执行链路和所有权。
- [文档体系设计与验收](documentation-design.md)：本轮组织方案、维护规则和检查边界。
- [数学组合设计](mathematical-composition-design.md)：分块、过滤、分组、求根及组合算法。
- [递推与状态设计](stateful-series-design.md)：当前递推、typed 输出和错误传播细节。
- [物理执行优化与验收](physical-execution-optimization.md)：成熟库候选、Eigen 旋转与协方差、独立协方差根的守卫 DSYRK、独立 exp/log 的 vForce 及 F 连续窗口、异常块回放、标准库顺序选择与连续打包、掩码工作区按本次消费量准备及高水位复用边界、Cholesky四行共享及独立 F 根直写、独立特征值死输出消除、特征分解双投影直写、保序矩阵向量、C/F 布局 EM、独立高斯分量保序库计算与完整异构结果验收、有界大矩阵面板、保序LU尾部行分组与Cholesky行面板、协方差→Cholesky临时缓冲复用、独立非负 MatMul 库路径、缓存块融合、直接根输出、迭代缓冲、ParentSlots节点owner压缩、NodeInfo不可变类型共享、原始带旋转/缩放及多组合资产投影共享与审计；保留实质 RSS/NJIT 欠项，微小正差不单独作为否决。
- [连续内存布局规划](physical-layout-design.md)：按计算链选择 C/F、输出布局描述、进程/共享内存贯通与兼容验收；C/F 输入运输、图输出策略、独立协方差/Cholesky显式 F 根直写和独立 exp/log 跨布局有界原生库分块及私有特征分解双投影独立F结果已实施；自动规划目前限于已准入路径，直接算子布局 API 和一般整图成本优化仍待完成。
- [概率模型与情景识别数学基础](latent-model-operators.md)：25 个统计、EM、推断、初始化及回填原语，含 Excel、对照和性能边界。
- [保序收缩与共享投影](mc-projection-primitives.md)：新增 affine_contract/expm1、可编辑多组合 DAG、分块物理执行、保序Eigen独立资产producer、V99独立组合lane及独立 Excel 公式；普通owned完整消费者复验已通过，C32堆峰差额、平台完整请求及整体内存目标仍开放。
- [按轴递推与多分位数](mc-axis-primitives.md)：新增三项通用原语、完整固定带消费者图及有界物理闭包；原始raw→normal/Student变换→投影→NAV/fan/统计/样本/贡献在私有证明下共享，保留全部可编辑步骤。85构建阶段12,267不同Python用例及三类原生9/9通过。CPU1的readonly raw replay中C8/短宽C32分别快于NJIT约2.3/4.8倍，六对完整输入heap也降低；原平台owned RNG不支付本参考的64MB replay copy，不能泛化该内存结论。完整随机/业务请求、阶段首错与Excel 187–190真实Microsoft Excel重算仍待验收。[原始带阶段证据](physical-execution-evidence/2026-10-03-mc-raw-prefix/README.md)；[类型共享前阶段](physical-execution-evidence/2026-10-03-mc-node-types/README.md)与[ParentSlots阶段](physical-execution-evidence/2026-10-03-mc-parent-slots/README.md)保留历史边界。
- [当前V74未来结果区复用](physical-execution-evidence/2026-10-09-latent-current-v69/future-result-screen-v74/qualified-adoption/README.md)：已实际接入d0f5e50f；全量16824、SAN6116、OFF6115通过/1环境skip及native各9，实际另717/native9/7完整测量通过。500k/2M GMM较旧C++少9/36MB申请峰，重复全链较同轮NJIT省时55%/58%；GMM申请峰仍高12.03/48.03MB，HMM和主要平台链仍开放。[后继子区复用设计与容量反事实](physical-execution-evidence/2026-10-09-latent-current-v69/future-result-screen-v74/qualified-adoption/next-design.md)。初筛负证据按原范围保留。
- [HMM私有尾行物化](physical-execution-evidence/2026-10-04-hmm-filter-tail/README.md)：977源码已接入，正式全量12,407项、真实SAN/OFF扩展各88项、三类原生各9/9通过。正式9个串行任务复验500k完整13根约5.941/11.740秒（C++/NJIT），较0fe申请峰少约24 MB；RSS增量中位比本轮NI少约76 MB，requested heap仍多3.161 MB。接口、数学及Excel不变，矩阵及其他完整主负载仍开放；该阶段结项时128 MiB私有系数区候选尚未实施，整体目标未完成。
- [前阶段HMM拟合工作区与回填优化](physical-execution-evidence/2026-10-04-hmm-scatter-pair/workspace-loan/README.md)：0fe阶段12,379项全量、真实SAN及OFF扩展各60项、三类原生各9/9通过。500k完整13根耗时5.713/11.247秒（C++/NJIT），较7aba热申请峰减少约104 MB；RSS比NI少约40 MB，requested heap仍多约27 MB。保留数学、错误和结果所有权，无新算子或Excel公式变化；原回填单项只省12 MB的历史失败保留。
- [短周期大状态批次](mc-axis-primitives.md#12-短周期大状态批次2026-10-04)：已有原始带闭包的大状态任务改为每批4组；Planner计入真实重算，接口/数学/Excel不变。正式全量12,319、实际ASan扩展52及三类原生9/9通过，31个串行性能任务结束。短宽完整链快约64%、较046少33.607 MB热申请峰，Student RSS比NJIT少约19%；requested heap差距及其他主负载仍开放；[正式证据与剩余项](physical-execution-evidence/2026-10-04-mc-state-batching/README.md)。
- [普通输入缓存与 owned-input 复验](physical-execution-evidence/2026-10-03-binding-owner-cache/README.md)：046构建移除旧输入及dtype metadata间接保留；当前12,304项Python、绑定ASan/UBSan363及三类原生9/9通过。同一NJIT RNG准备下C8快53–55%、申请峰少28–44MB；短宽C32快约76%，但内存仍落后，单组合只快7–9%，整体目标未完成。
- [矩阵与数组基础设计与验收](matrix-array-foundation.md)：14 个新增原语、Excel 复现和本轮性能/内存状态。
- [M0/M1 开发设计与验收](platform-foundation-design.md)：本工作树的三维协议、具名载荷和原生迭代；验收状态以该记录为准。
- [平台自研计算完整承接规划](platform-compute-roadmap.md)：后续算法、类型协议、平台接入及等价验收工作；附[函数盘点初表](platform-compute-inventory.csv)，均不代表迁移已完成。
- [Excel 公式复现与核验契约](excel-reproduction-design.md)：引擎公式/写出后端、平台迁移、200 万单元格预算与验收记录；含 Microsoft Excel 16.89.1 全量夹具验收和独立符号上界规划；平台迁移仍待完成。
- [发布](publishing.md)：构建和分发流程；发布前还需遵守当前检出的提交规范。

- [私有系数完整图的成熟LU](physical-execution-evidence/2026-10-04-guarded-lu/README.md)：924源码已接入，正式相关539项及原生9/9通过，同源SAN/OFF扩展各403项及原生各9/9通过。正式24个串行测量结束，4096阶C/F完整链0.544/0.448秒，NJIT0.711/0.626秒；申请峰692.264/809.539MB。正式完整回归12,487项通过、零skip；直接大求解及其他主负载仍有缺口，整体目标未完成。
- [私有系数求解工作区](physical-execution-evidence/2026-10-04-private-solve/README.md)：b43前阶段正式全量12,443项、实际SAN/OFF扩展各359项及三类原生9/9通过；正式14个完整性能任务结束。4096阶峰值较旧包约少128MiB、比NJIT少131MiB，但C/F链耗时2.394/2.578秒，NJIT为0.686/0.606秒。保留该阶段证据，不代替924本轮完整回归。
- [成熟LU与布局筛选](physical-execution-evidence/2026-10-04-lapack-lu-screening/README.md)：隔离原型4096阶C/F为0.487/0.414秒，NJIT0.733/0.610秒；申请峰689.269/809.539MB。但有4个真实奇异阈值反例，guard、回放及正式调用未实现；生产仍是b43，不能以原型关闭目标。

## 阶段设计与验收记录

以下文件保留当时的范围、决策和实际测量，不能作为最新 API 总目录。历史“118/125 个算子”、旧 dtype/协议版本和当时不支持的能力不代表当前限制。

- 第一阶段：[设计](canonical-operators-design.md)、[验收](canonical-operators-acceptance.md)。
- 原生计算图：[设计](phase2-execution-graph-design.md)、[验收](phase2-execution-graph-acceptance.md)。
- C++ 执行链路：[设计](cpp-first-design.md)、[验收](cpp-first-acceptance.md)。
- Typed DSL 与窗口：[阶段设计](typed-dsl-rolling-cpp-design-2026-09-20.md)。
- 物理 DAG：[设计](planner-physical-dag-optimization-design-2026-09-20.md)、[验收](planner-physical-dag-optimization-acceptance-2026-09-20.md)。
- 指标能力：[变更前审计](metrics-factory-capability-audit.md)、[数学组合验收](mathematical-composition-acceptance.md)、[状态扩展验收](stateful-series-acceptance.md)。
- 性能：[多负载记录](multiworkload-benchmark.md)、[C++/NJIT 对照](cpp-vs-njit-benchmark-matrix-2026-09-20.md)、[0.3.0 验证记录](verification-0.3.0.md)。

历史验收通过仅覆盖其记录的构建、机器、数据和检查范围，不自动证明新修改、其他平台或上层金融业务通过验收。
