# V73：有界 GMM 重算复用 Eigen 行包

2026-10-09，隔离候选，实际仍为a87f2c25。V72首组完整时间显示三状态GMM由0.8235增到1.3982秒，HMM750k由2.4635增到3.1397秒；各仍快于NJIT，但多做一次E步的代价不能隐藏。V72完整申请峰仍按冻结runner继续测，不在它的正式测量期间编译新候选。

本后继保留V72的K3/D4、NK>=1Mi elements有界准入，复用**已经存在**的`eigen_density_columns`：最多128×4 doubles临时列块，以8个独立样本为Eigen数组lane。当前kernel的feature累计次序、原高斯公式和每行/组件finite检查不改，不手写新的矩阵数学。

只为gmm_tile_density增加内部light-replay许可：当K3/D4时，将需要的只读block显式打包到既有bounded input_tile，然后允许现有density_rows的Eigen列路径。其他density_rows调用默认false，原HMM/density/小任务与其他尺寸准入不变。已有大GMM8..32宽度路线不变；显式scalar/OFF/不支持平台继续原支持路径。组件组K3<=8，内部64 double返回tile容量充分；尾行仍走原标量公式。没有扩大数据扫描范围或改变输入、根、DAG、错误次序、EM参数、停止含义和随机。

必要代价：每次旧E的tile可复制128D到有界列块，累计copy可能增加，须实际报告；不能用“不新增大数组”声称零复制或最终内存已经更低。已有gmm replay scratch已预留input_tile，因此没有新增大工作区。保存old means、再次旧E、围绕新means的方差、正停止和多投影输出仍完整。

只改latent.cpp，93源/配置、191测试、真实编译参数与native/wheel/worker身份核对；原实际代码/包不先改变。D1/D2/D5/D6/D8、X1/X6/X9/X10适用：纯物理布局/成熟原生库选择，同数学/逻辑图、所有权；无新的算子/公共接口/Excel recipe或预算，普通C++不执行Excel，不宣称本轮Excel重算。

先原生9/相关回归701，再current/candidate/warm NJIT同CPU1、同六owned readonly准备、13根与两代保留、普通完整请求的独立fresh time/heap；完整输出/NaN、离散/迭代、数学不变量、input/old hash及非别名检查。不采用局部微基准作为收益。若有实际收益，再门槛/多布局/负stride/异常/传输、全量/SAN/OFF、重复time/heap/RSS及实际重建复验才可进入真实源码。目标仍未完成，arena联合成本与完整QP/前沿/随机探索等主要负载继续开放。
