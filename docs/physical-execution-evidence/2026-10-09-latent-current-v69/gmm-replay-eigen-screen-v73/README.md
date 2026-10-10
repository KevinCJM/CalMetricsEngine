# V73：有界重算复用 Eigen 行包负实验

2026-10-09，实际a87f2c25不变，候选243b8fff未采用。复用已有Eigen列密度核和有界tile，不改变数学、图、参数、输入准备或所有根。

原生9/9、相关Python701/701，无skip；30个fresh完整进程及另2smoke通过全部13输出、NaN/离散/实际迭代、数学不变量、输入/旧结果hash和非别名。CPU1、六owned readonly准备、普通完整请求、两代结果保留、暖NJIT。只有一组time/heap，无候选RSS/重复/全量/SAN/OFF。

GMM500k申请峰少8,992,640bytes，但仍比NJIT多12,038,550bytes；完整耗时0.82295→1.29865秒，约+58%。HMM750k申请峰不变，耗时2.46597→3.01794秒，约+22%。GMM累计算法复制4,000,280→916,011,224bytes。上述是完整请求代价，候选不采用；不能以复用成熟库或局部吞吐代替实际收益。

后继尝试本次未来独立结果区作GMM后验，保留原单次E步。公开类型/算子/错误/所有权与Excel recipe/budget无变化，没有本轮Excel重算/平台迁移/提交推送。主要arena缺口、完整QP/前沿/探索与整体目标仍开放。

[设计](design.md)、[自审核](self-review.md)、[完整对照](comparison.json)、[原始筛选](screen.json)、[闭合复核](closeout-validation.json)。
