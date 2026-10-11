# V36：公开方法调研与隔离筛选闭合，未采用新运行实现

2026-10-06。实际工程继续使用 V35r2；本轮只在隔离目录开发、验证及测量四个候选，归档后仍未修改数值运行代码、提交或推送。下列数据是相同只读输入准备、CPU1、两代独立保留输出的普通请求/私有ABI对照；图定义一次性编译和JIT预热不计时，IPC不在本批计时中。私有C ABI不包含真实Scheduler/Graph，其数据只能筛选，不能替代实际接入验收。

四个候选均通过原152个相同输入控制，88准入/64拒绝；原304输入hash及80位Decimal独立核验保留。连续扫描/更紧误差深度/组合的首次6轮中位数分别1.1327/1.3787/1.2538秒，私有原版1.2358秒、实际V35r2为1.2155秒、NJIT为0.6811秒。更紧深度将512次DGEMM降为256次，仍变慢；原设计文件中的“尚未编译/测量”是前期冻结设计状态，当前资格与计时由本目录终态记录补充。

后续尾部复用只借本次独立、未发布、尚未完成的结果区域，保留保守gamma(2b+2+q/depth)及0.000049预算。顺序和平衡树均使用同一Accelerate DGEMM和Eigen合并，不新增乘法微内核；C/F完成前缀、当前目标与借用槽不相交。原152控制最大Decimal误差2.520e-8；普通/ASan+UBSan各96专项通过，其中各40项实际触发尾部借用，覆盖全矩阵精确整数参考、奇数尾块、input/canary、两代所有权、取消及不足工作区。它们不代表真实Graph失败回放/worker接入通过。

4096×4096×4096 FF→F、CPU1六轮配对结果：私有原版1.070866秒，尾部复用1.182437秒，实际V35r2 1.080305秒，NJIT 0.669029秒。调用数512→160并未提速。因此不采用尾部复用、更紧深度或组合；连续扫描的有限收益仍未取得完整接入与fresh峰值资格。本轮没有测量V36峰值，不声称内存节约、持平或不增加。

源码及实际动态链接核对发现：[Numba官方文档](https://numba.readthedocs.io/en/stable/user/performance-tips.html#linear-algebra)所述SciPy绑定在本环境也成立。安装的Numba _lapack.c把dgemm导向SciPy Cython capsule，dladdr解析到cython_blas，nm显示dgemm$NEWLAPACK，otool显示Accelerate依赖。以前raw报告的actual_blas_pools记录的是加载的NumPy OpenBLAS，不是NJIT乘法的实际库；旧原始数据保留，不能将该字段当成NJIT使用OpenBLAS的证明。

按[Apple线程接口说明](https://developer.apple.com/documentation/accelerate/blassetthreading(_:))，新增测量在实际调用线程显式BLASSetThreading(1)、read-back并finally恢复，正常/异常恢复控制通过。重新六轮配对普通请求，实际V35r2为1.040245秒，NJIT为0.644696秒；双方实际上都使用Accelerate，新证据仍显示C++差距开放。这一结果不与前一批1.08/0.67混搭计算收益。

已核对[Intel Packed GEMM](https://www.intel.com/content/www/us/en/developer/articles/technical/introducing-the-new-packed-apis-for-gemm.html)、[内存高效调度论文](https://arxiv.org/abs/0707.2347)、[OpenBLAS官方构建说明](https://www.openmathlib.org/OpenBLAS/docs/build_system/)及原BLIS/GEMMFIP/误差分析来源。打包复用可能增加接近输入规模的内存；MKL的x86成绩、Strassen调度、OpenBLAS的其他ISA/精度内核不能作为本机float64采用证明。尾部划分是本任务自己的设计，论文不为其性能背书。没有全局安装或引入新运行依赖。

[完整审核](review.json)、设计、私有源、原测试摘要/单项记录、12组72次原始时间观测及过程日志均保留。所有已知父/子任务已终结；新候选未接入，未重跑不变的实际工程全量、wheel或Excel重算。下一步应定位同一Accelerate下额外分块、校验、结果物化与完整链路的真实成本；强NJIT时间/内存及其他主要负载整体目标仍未完成。
