# V74 初筛自审核：未来独立结果区作 GMM 工作区

2026-10-09，实际a87f2c25保持；隔离候选d0f5e50f只改graph.cpp的物理工作区选择，latent.cpp及数学内核、逻辑图、输入、输出、参数和停止含义均保留。正常/SAN/OFF包完成源码/native/wheel/worker身份核对，三个构建原生均9/9；正常相关Python701/701。新增专项修正后16/16。OFF相关Python6115通过、1项既有平台/库开关豁免（6116总项）；新增16项均未跳过。完整回归及SAN相关Python尚在运行，不作为通过证据；全部结束后才执行21个串行完整复测。未接入实际源码，目标未完成。

## 本阶段完整证据

30个fresh完整time/heap进程和另2smoke通过13输出、NaN/离散/实际迭代、数学不变量、输入/旧结果hash和非别名。CPU1、同六owned readonly准备、普通入口、两代结果保留、暖NJIT。编译/fixture/图定义及事后核验排除，实际分配/复制/调度均计入请求。只有首组time/heap，尚无本候选重复组或RSS结论。

GMM500k申请峰131,534,582→122,534,582bytes，少9,000,000bytes；完整时间0.82516→0.84590秒，约+21ms/2.5%，仍比NJIT1.88915秒省55%。没有重复E，保留原数学及成熟库路径。不能把该21ms作为单独否决条件，也不能据单组宣称稳定提速。仍比NJIT110,503,392bytes多12,031,190bytes，内存目标尚未闭合。其余四条链申请峰保持原值：HMM250k/750k仍高于NJIT15,540,614/46,540,614bytes；大HMM/GMM保持已有优势。

## 所有权与资格范围

只借本次严格typed请求未写入的float64 rank2 scatter根，检查所有<=fit的物理writer、当前row/root容量、byte bounds、alignment和fit输出非重叠。Program::validate先要求root_outputs与typed roots同数；现有prepare和FIT_SCRATCH_CONTRACT验证形状、workspace及指针。借用不涉及arena输入，不保存/恢复ND，不读前次结果；原GMM单E body每次完整重算。原prepared/observer、isolate/scopes、deferred/input-failure禁用条件保持。strict失败不发布本次结果；未来scatter照原协议完整写入其区域。

新专项覆盖C/F/reverse、NK门槛两侧、两个区间的线程及inline/shared进程、早已写入或容量不足根的拒绝、两次不同拟合、isolate、零方差正规化、NaN/Inf/负方差错误恢复和旧结果保留。首次15项中4项失败来自测试假设：prepared不支持强制多通道，零方差不应要求报错；改为独立单通道prepared参考并增加合法零方差测试，不修改生产契约，初始源码/log/XML/结果均保留。

下一步完成192个文件的完整回归及真实SAN/OFF、重复完整time/heap/RSS和实际重建复验，之后才决定是否采用。进一步内存方向需要闭合gather/标准化的物理复用证明及按生命周期的arena成本，不能以离线slot估算替代真实峰值。完整QP/前沿/随机探索和其他主要负载继续开放。

仅历史设计/证据/索引需要更新；公共算子、类型、参数、错误、结果布局/所有权及Excel recipe/budget无变化，用户/算子/执行/Excel手册不机械改写。没有本轮Microsoft Excel重算、平台迁移、提交或推送。
