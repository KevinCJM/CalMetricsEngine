# V72：三状态大样本 GMM 有界后验重算

2026-10-09，隔离候选，只改latent.cpp的既有replay几何准入。实际a87f2c25源码/包、93源/配置与191测试保持冻结。

V71原生逐节点trace与11个实际记录逐byte匹配：audit numeric_arena包含double和int64两池。原图的shape无关槽染色与私有融合有依赖；简单按声明尺寸best-fit使GMM16/GMM3 arena增长19,573,456/2,000,000bytes，不采用。先处理GMM3 full posterior工作区，不在这轮接入新通用arena。

既有GMM replay已支持正停止容差，严格按样本顺序累计counts/means，再用旧E参数重放各tile并围绕新mean累计方差；不改初始化、随机、停止、最大迭代、排序、异常、输出或EM公式。复用同一有界实现、现有Eigen固定宽度加权特征和density dispatch，不增加新数学内核或算子。

原大矩阵准入K>=16、D8..32、NK>=16Mi elements保留。候选只增K=3、D=4、NK>=1Mi elements准入；仍须storage<full NK，macOS15/Apple arm64/原Eigen编译条件。OFF、其他形状和小任务继续原支持路径。按实际prepare/workspace与executor共同使用的gmm_replay_scratch判断，不做Python策略；不改GMM图和六份输入准备。

GMM3训练375k、K3的原NK工作区约9MB可由有界tile代替，但会重算E步。HMM750k的GMM预训练也达到新门槛，须检查完整HMM耗时/正停止/原history和所有权；未知收益不能只按workspace预测。现有large-HMM12、GMM16、HMM250k路径应保持。

适用D1/D2/D5/D6/D8及X1/X6/X9/X10：纯物理存储/执行准入、同数学与原逻辑节点；输入与旧结果不作可写scratch，多输出全部返回；无分解缓存、全局线程设置、公共接口或Excel recipe/budget变化。普通请求不执行Excel，无本轮Microsoft Excel重算宣称。

先身份一致AOT构建，原生9和相关数值/latent/工作区/结果/错误回归；再同CPU1、同原图、六份owned readonly准备、13根/两代保留的current/candidate/warm NJIT独立fresh time/heap/RSS。全输出/NaN/离散/实际停止迭代、数学不变量、输入/旧结果hash和非别名核验。若有实际收益，再补门槛/多布局/异常/transport专项、重复组、全量/SAN/OFF、实际重建复验才可采用。当前未采用，完整QP/前沿/探索与总体目标仍开放。
