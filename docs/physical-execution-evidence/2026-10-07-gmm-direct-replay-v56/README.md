# V56：有界 GMM 后验重放与实际接入

2026-10-07，本轮已经接入当前 CalMetricsEngine；分类为**有实质进展，整体目标未完成**。基于 V55 的公开分块方法筛选，复用现有 Eigen，未新增依赖、公开算子或算法语义。源码未提交或推送，分支/HEAD、源文件与二进制身份见 [验收事实](raw/acceptance.json)。

## 修改与数学边界

实际改动为 `cpp/operators/latent.cpp`、`cpp/planner.cpp`、`cpp/graph.cpp`、`cpp/include/calmetrics_engine/operators.hpp`，新增 `tests/test_gmm_replay.py`（22项）。[设计](raw/gated-design.md)记录准入与证明，[自审核](raw/self-review.md)记录数学/所有权/错误边界。设计标题保留当时“尚未接入”的历史状态；后续接入以 [实际接入记录](raw/actual-adoption.json) 和 [实际构建身份](builds/actual-v56/identity.json) 为准。原始 qualification JSON 内 `actual_adopted=false` 是测量器固定字段，不代表后来实际接入失败，不能据此覆盖源码/身份核验。

Apple arm64、macOS15+、Eigen开启，K≥16、8≤D≤32、NK≥16,777,216，且存储S<NK时使用128行后验，S=`K*(128+2*D+3)+128*D` doubles。保留本轮旧权重/均值/方差重算E步；方差围绕新均值进行中心化两遍计算，保持样本累加顺序、下限、严格停止条件、末轮E似然/迭代诊断和失败发布前缀。内部密度直接写完整K行跨度，消除候选旧中间tile转移。F列直接借读，其他合法stride仅打包当前128行；不强迫调用方转换布局，不宣称全算法零复制。小样本、K<16、未准入D、其他平台及库关闭继续原数学路径，scalar保持同序原生计算且不宣称相同提速。

Graph在新形状上跳过原完整NK借用，避免旧借用几何及无符号差值错误。Planner使用同一latent_scratch覆盖最大区间和阈值以下最后旧形状，兼顾S≥NK的交叉边界；实际执行不因此分配闲置NK。成本单位保持既有启发式路由规则，非实际FLOP计数。逻辑AST/DAG、公开类型/协议、CPU预算、结果独立所有权及Excel公式/预算保持不变。本轮不运行Microsoft Excel，不宣称新的实机Excel验收。

## 同核完整请求与内存

所有计时包含同一独立只读输入准备、普通Scheduler/暖态NJIT完整计算和两代独立结果。定义/JIT在计时外完成，库与NJIT线程数均为1；测量期间不与编译/测试混跑。CPU/吞吐诊断在实际确认记录中，不能把单位吞吐当全平台验收。

| 完整请求 | 原C++ | 最终资格候选 | 暖NI | 当前目录重建确认 |
|---|---:|---:|---:|---:|
| 1048576×8/K24，4迭代，6项输出 | 3.2615s | 2.5413s | 10.8902s | 2.5045s（本次NI 10.8054s） |
| 1398104×8/K16，训练1048578，13项输出 | 2.3850s | 2.1092s | 7.3263s | 2.1150s |

第一组是六轮交替first的18个fresh计时进程，每进程预热后1次样本。[全部配对记录](raw/paired-state.json)与原始JSON/小型输出保留。第二组每backend一个fresh计时进程、预热后3次样本，是扩展冻结夹具到K16的压力链复现，不能替代已部署平台业务资格。[链路记录](raw/chain-v2-state.json)覆盖标准化、初始化、拟合、推断、排名、scatter及state。两个样本组分别报告，不能拼成统一中位。当前目录确认每组3次，身份与候选一致且全部输出逐字节一致，见 [实际确认](raw/actual-confirmation-state.json)、[输出核验](raw/actual-output-parity.json)。离散与诊断精确，NI浮点按既有4.9e-5预算核验，NaN位置保留。

| 暖态请求申请堆峰值（bytes） | 原C++ | 候选/实际 | NI |
|---|---:|---:|---:|
| 6项拟合 | 268464526 | 67173774 | 469774681 |
| 13项完整链 | 1022072178 | 932617522 | 851456928 |

每backend独立fresh启动malloc进程测3个完整请求，与计时和RSS分别运行。拟合少约192MiB；完整链少约85MiB，但**仍较NI高81160594bytes（约77.4MiB）**，未闭环。实际fit/chain峰值与资格候选相同。完整链两代结果本身保持独立，不能通过覆盖旧结果降内存。startup probe包含全请求malloc/NumPy/NRT，不是Planner估计；暖阶段统计中的极少量untracked frees及无table overflow原样保留，不宣称allocator级穷尽证明。

RSS另保留绝对峰值及夹具准备后的增量，拟合基线包含夹具残留驻留页；新拟合458752bytes暖增量不等于整任务占用。链RSS绝对峰值原/新/NI为1217757184/1128136704/1415462912bytes；运行时总驻留不能抵消NI申请堆优势。GMM拟合算法复制536892608bytes、工作区36416bytes；完整链复制584422000bytes、工作区27136bytes。重放打包可能增加数据移动，实测更快并不证明零复制或全引擎复制审计完成。

## 筛选、资格与失败记录

未门控D原型在K12慢17%～33%，不统一启用；最终门控E排除这些形状。完整 [初筛](raw/screen-state.json)、[集中筛选](raw/focused-screen-state.json) 和 [门控追加](raw/gated-extra-screen-state.json) 保留不同构建身份，禁止把D/E样本合并。门控追加K16/D10、K16/F和K64/C相对旧路径约快59%、26%、30%；未准入K12的约1ms差异不作阻止采用的门禁。形状阈值是受测保守选择，不证明所有邻近/极端形状有相同收益。

| 验证 | 正常最终候选 | ASan/UBSan | 库关闭 | 当前目录重建 |
|---|---:|---:|---:|---:|
| Python全量 | 15898/15898 | 未跑全量 | 未跑全量 | 同源身份核验，未重复全量 |
| Python相关 | 654/654 | 654/654 | 654/654 | 654/654 |
| native | 9/9 | 9/9 | 9/9 | 9/9 |

上表全部已执行项无skip；三资格构建source一致、编译选项分别记录，实际normal源码/ID匹配已验最终候选。XML/日志/启动命令全部归档；不能把相关、全量、actual重复计数相加。另19项直接大形状检查包括停止nextafter邻居、scalar/auto、布局、独立Workspace与完整显式输出失败前缀；4种失败输出与旧构建逐字节一致。新22项覆盖dispatch邻界、非单调Planner、虚拟巨大K几何交叉、readonly/所有权、多运输lane/CPU及故障隔离。

初次相关547通过/3失败来自新测试误将Prepared强制thread/process；随后18专项通过。缺少PYTHONPATH的一次collection error完整保留。初次最终相关653通过/1失败来自新几何测试误用`op.requirements`，改为真实API`op.get(...).requirements`后完整重跑通过。第一次chain摘要因NaN比较失败，修正仅将NaN转为可稳定核验摘要后完整重跑；真实输出字节/数值核验未降低。旧既有测试未弱化，失败记录不会因后续绿色删除。

## 下一步与文档维护

[下一阶段源码事实](raw/next-stage.md)及[节点寿命](raw/chain-node-lifetimes.json)定位私有训练`take_axis→subtract→divide`约64MiB物化和16个组件比较约21MiB掩码。现有pair shortcut要求第一段父为持久输入，私有训练take不满足；扩展私有寿命/最后消费证明与状态映射融合必须保留错误顺序、view别名、离散相同、logical节点、Planner与独立结果，本轮尚未实现。公开标准化/probability根已直接写结果，根复制仅2192bytes，不能虚构大根复制。[静态slot上界](raw/static-slot-census.json)与实际arena分开。其他重要矩阵仍慢NI，主要平台计算链及完整分配/复制审计仍开放。

现有执行指南、潜变量说明、优化计划更新工作区/准入/阶段证据；公共operator、typed/user与Excel契约无变化，因此无需机械改写。R没有Hermes JSON或`check_documentation.py`，未挪用F的脚本，也未修改F。最终CodeGraph同步/源码复读、保护文件hash、Markdown链接和diff空白检查另见`raw/final-validation.json`（在归档后生成）。

V55的研究依据保留：[cuVS 官方GMM指南](https://docs.nvidia.com/cuvs/user-guide/api-guides/clustering-guide/gaussian-mixture-model)、[Armadillo公开方法](https://arxiv.org/abs/1707.09094)、[官方gmm_diag API](https://arma.sourceforge.net/docs.html#gmm_diag)。本轮没有新的网络包安装或GPU替换；这些是此前方法筛选来源，不能把其性能描述当成本机采用证据。

## 证据与复现

`source-current/`冻结93份source/config，`source-before/`冻结4份被改动前文本；组合恢复可构建前一实际版本。`source-prototype/`单独保留D的4份文本，身份明确，不能用其数据代表E。`frozen-helpers/`保存同次NI参考、链夹具、memory probe与新测试；所有源码/脚本均加`.txt`，不进入生产构建。`raw/`保留设计、启动器/benchmark文本、全样本/错误日志/输出摘要/校验，`builds/`保存flags、源hash、wheel和native/worker hash，不包含可执行二进制、wheel、vendor或object。大型链npz约302MB/份仅留临时目录，归档完整逐输出hash/数值核验及重建脚本，不把GB级结果复制入仓库；小型fit输出和4种错误数组保留。

复现时在隔离目录恢复`.txt`源码/脚本，按build identity的flags构建，先执行资格脚本再按保存的run脚本CPU1顺序测量。路径和具体版本记录在原始命令；应调整隔离目录但保持输入seed、shape、dtype/layout、普通入口、只读准备、两代输出、预热、CPU预算和测量scope，不能直接执行旧绝对路径声称新验收。`manifest.json`保存所有归档文件的SHA与大小，自身除外。
