# V72：三状态有界重算负实验

2026-10-09，实际a87f2c25保持，候选b31f215a未采用。只扩展已有replay的K3/D4、NK>=1Mi准入，不改数学、图、参数、输出和输入准备。

原生9/9、相关Python701/701，无skip；30个fresh完整进程和另2smoke通过全部13输出、NaN/离散/实际迭代、数学不变量、输入/旧結果hash和非别名。CPU1/相同六owned readonly准备、普通完整请求、两代结果保留，NJIT已预热，编译/fixture/定义/事后核验排除。只有一组time/heap，无候选RSS/重复组/全量/SAN/OFF。

GMM500k三状态申请峰少8,992,640bytes（131,534,582→122,541,942），工作区9,000,096→7,456bytes，但仍比NJIT多12,038,550bytes；完整耗时0.8235→1.3982秒（约+70%），NJIT1.8966秒。HMM750k耗时2.4635→3.1397秒（约+27%）而完整申请峰239,295,726bytes不变。其余三组申请峰/workspace不变。约0.57/0.68秒是实际完整请求代价，不是几毫秒门禁；本候选不合入。

下一步复用已有Eigen样本lane density kernel加速bounded E步；代价和复制须重新测量。公开类型/算子/错误/所有权与Excel recipe/budget无变化，没有本轮Excel重算/平台迁移/提交推送。主要arena缺口、完整QP/前沿/探索与整体目标仍开放。

[设计](design.md)、[自审核](self-review.md)、[完整对照](comparison.json)、[原始筛选](screen.json)、[闭合复核](closeout-validation.json)。
