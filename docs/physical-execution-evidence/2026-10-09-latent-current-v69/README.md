# V69：当前完整 GMM/HMM 链与分配缺口

2026-10-09 本地完成。实际源码、wheel、worker身份为 `a87f2c25ae40a633f3f22ae0bfc66ecdefb721294e349d2ae21d74687eb960a2`。本阶段复验已进入实际包的执行路径，未修改引擎实现；下面时间优势包含历史优化，不能全归因于本轮。

完整40个fresh测量全部通过：24个时间进程（三组轮换，各三次请求）、8个申请峰及8个独立进程增量RSS，2个smoke另列。CPU1/库线程1、相同六份owned readonly准备、普通入口、13个独立完整输出、两代结果保留；编译、静态fixture/图定义及事后核验排除。所有输出/NaN、离散状态/索引/迭代、输入/旧结果hash、两代与输入非别名及概率/方差不变量通过。所有配对输入hash相同。

| 完整链 | C++/NJIT秒，三fresh进程中位 | 省时 | 申请峰较NJIT | 增量RSS较NJIT |
| --- | --- | --- | --- | --- |
| HMM 500k×4、12状态、最多4轮 | 4.1911 / 8.9433 | 53.14% | 少6.62MiB | 少85.94MiB |
| HMM 250k×4、3状态、最多60轮 | 0.8253 / 1.7765 | 53.54% | 多14.82MiB | 少57.16MiB |
| GMM 1,398,104×8、16状态、最多4轮 | 2.0773 / 7.3287 | 71.66% | 少15.93MiB | 少334.25MiB |
| GMM 500k×4、3状态、最多60轮 | 0.8259 / 1.8949 | 56.41% | 多20.06MiB | 多4.20MiB |

60是最大轮数，不是实际完整60轮：默认参数HMM为10轮/30预训练，GMM38轮。平台默认3状态/最大60轮，但这里75%训练、随机初始化seed17是可配置大型走步负载；默认实时训练60行和默认quantile初始化未覆盖。参考是引擎内冻结的独立NJIT测试逻辑，不依赖平台源码；未宣称平台HTTP/data/PIT服务、全部C/F/批次/多核或全算法均已验收。

## 受控视图实验：不采用

另8个完整C++进程仅将固定fixture的连续训练/OOS区用现有slice表达，其他输入、数学、随机、停止和全部13输出不变，均通过核验。HMM申请峰减少约16MB/6MB，但大GMM增加41,942,864bytes，三状态GMM增加999,744bytes。大GMM的增长来自numeric arena，fit workspace仍27,136bytes：它原本已用有界posterior replay，没有失去full posterior loan。HMM12复制量由77,011,328增至205,011,360bytes，是累计复制量，不能作为峰值内存相加。

各视图时间只有一组，不与正式三组混算或宣称稳定提速。实验未进入生产优化器；一般take允许非连续/负索引，不能由这个fixture推广为无条件借用。

## 源码原因和下一步

当前arena按编译时逻辑slot的实际shape最大值求和。减少gather可能延长backing生命周期并改变slot分配，因此删去一份数组不保证峰值更低。HMM还会借用现有arena/本次未来结果保存历史；省arena与增加history复制须一并估算。

当前history loan要求NK≥2,097,152，而finalize对含任意HMM的图都禁止新私有算术来源融合。小状态大样本可能没有该loan，却保留全ND中间值。[V70后续筛选](fusion-followup-v70/README.md)已完成隔离候选原生9/相关Python701及30个完整time/heap对照：HMM250k/750k较当前少约6/18MB，但仍高于NJIT；候选未采用，大history保持原路径。单独降低loan门槛不够：活跃ND输入可能比拟合NK工作区更大，save/restore后的reserve仍须小于原scratch。

按真实大小与借用生命周期规划arena、history成本选择仍待完整开发/资格。V70另4个共享标准化控制图通过核验，但大GMM申请峰多约45MB，不采用。[V71逐节点诊断](arena-diagnosis-v71/README.md)已完成30项离线比较，11项原路径容量与真实审计逐字节一致：连续视图控制图在take与scatter之间插入slice，失去现有相邻融合，额外arena约41.94MB。视图自身不分配；简单按大小best-fit反而让原GMM16/3多约19.57/2.00MB，因此不采用。分析参考[OpenXLA缓冲分配说明](https://openxla.org/xla/hlo_to_thunks)，没有引入XLA/ORT或JIT依赖；[V70历史设计与自审核](fusion-followup-v70/self-review.md)。

[V72有界重算](gmm-replay-screen-v72/README.md)及[V73复用Eigen行包](gmm-replay-eigen-screen-v73/README.md)各通过原生9/相关Python701、30个完整time/heap进程及另2smoke。GMM500k均少8,992,640bytes，但完整耗时分别增加约70%/58%；HMM750k峰值不变而耗时增加约27%/22%。都不采用，实际仍为a87f2c25。后继验证借本次未写入的独立未来结果区保存GMM后验，保留原单次E步；完成资格与完整复验后才决定是否接入。

[V74未来结果区初筛](future-result-screen-v74/README.md)已完成三种构建原生各9/9、正常相关701及新增专项16、OFF相关6115通过/1项既有平台豁免。首组30个完整time/heap进程通过13输出及所有权；GMM500k少9,000,000bytes，耗时0.82516→0.84590秒，保留相对NJIT约55%的速度优势。其余四链申请峰不变，GMM/HMM仍有内存缺口。普通全量/SAN相关回归在运行，通过后再串行复测50万/200万行；未采用，不能称正式验收或整体目标完成。

GMM3/HMM3申请峰差额仍未关闭，完整active-set QP/前沿/随机探索及其他主要负载继续开放。整体目标未完成，无平台迁移、Microsoft Excel本轮重算、提交或推送。

## V74当前：正式资格与实际接入闭合

上述初筛未采用/仍在运行的文字保留形成时事实。graph.cpp及新增16项回归已进入实际目录，93份source/config、192份测试和wheel/native/worker身份为d0f5e50f；341份其他文件与平台工作区未变。正常同源全量16824、SAN关联6116、OFF6115通过/1项既有环境skip与三种native各9完成；实际包另native9/关联717及7完整测量/2smoke通过，不称实际再次完整或SAN全量。

另21个串行fresh重复对照确认500k/2M GMM申请峰较旧C++少9/36MB，时间基本持平，较同轮预热NI省55.44%/57.71%；实际重建另复验0.814437/3.096012秒、122534582/490034582bytes。实际时钟与候选/旧/NI配对epoch分别保留。GMM申请峰仍比NI多12.03/48.03MB，HMM750k多46.54MB；RSS优势不等于申请峰欠项关闭。[完整资格、全部重复与实际原始记录](future-result-screen-v74/qualified-adoption/README.md)。

当前实际wheel重新编译五图的节点/根与V71诊断逐项一致；容量反事实证明单独删除raw take或训练结果都不减少slot最大值。三个联合省略只可能让GMM500k少10MB、大GMM少22.37MB、HMM750k少13.5MB；大HMM仍为0。这是离线容量分析，未测试新借用执行，不宣称新收益已实现。[后继C++proof/lifetime/错误与联合history预算设计](future-result-screen-v74/qualified-adoption/next-design.md)。执行指南已同步V74物理准入；公开数学/算子/AST/IPC/Excel契约未变，无需机械修改对应手册。完整QP/前沿/随机探索和整体目标继续开放。

[设计与范围](design.md)、[实际完整比较](comparison.json)、[原始测量](screen.json)、[第二/三组与默认内存](followup-screen.json)、[未采用视图](view-screen.json)、[闭合复核](closeout-validation.json)。
