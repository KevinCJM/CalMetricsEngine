# V35r2：已接入实际代码，完整观测闭合，整体目标仍开放

2026-10-06。当前工作区十处实现/测试已接入分块系统DGEMM及Eigen归并。93份数值源、实际目录构建的wheel和独立worker身份一致：`b17f2bbea7aa9cf2c018f289c01757b69f5134046006a70ed9bac35cad16158a`。未提交、推送、全局安装或部署平台。

同源普通全量15146项通过；普通/ASan+UBSan相关各3352、关闭库相关3042，三套原生各9/9，均零失败、错误或跳过。原152 Decimal80控制的输入hash保持一致，最大误差2.520e-8；新增84项覆盖C/F、私有转置、普通/线程/进程inline/shared、所有权、拒绝与取消。实际目录的新wheel另验572项及原生9/9。完整回归沿用字节一致的301份计算、接口、测试和工具资格，没有重复一次全量来扩大计数。

实际新wheel首轮在受限沙箱中462项通过、110项受共享内存创建禁止影响，均包含`shm_open create: Operation not permitted`；原XML、日志、状态与hash保留于`actual-verification/restricted-attempt`和[环境重试记录](v35r2-actual-retry-environment.json)。同一runner、源码、wheel及断言在允许共享内存的环境重跑通过；没有为环境限制删测试、改内核或降低断言。

45组270次完整计时、180次fresh申请堆/RSS观测均已终结。双方计入输入准备、普通Scheduler绑定/计划复核/图执行/结果返回和两代只读独立结果，预热NJIT、核数一致。一次性AST/DAG构建、wheel/JIT编译及运行时预热不在计时中；矩阵用single CPU1，风险链用array_row thread CPU4，worker传输为资格测试而非这批计时lane。部分原始scope字符串中的AST/worker表述过宽，原文保留并由[源码范围核对](v35r2-measurement-scope-review.json)纠正，不能据此称本轮包含图编译或IPC耗时。26万行风险链C/F约1.58秒，对照强四核NJIT约2.16秒；派生/混合链仍约0.20–0.21秒，对照约0.27秒。这三链仍不含RNG、约束修复、多轮搜索、前沿或完整平台服务。

4096高幅值混合符号FF完整请求从6.1677降至1.0525秒，约5.86倍；强NJIT为0.6644秒，因此尚有58%的相对时间差距。该用例申请堆峰815.884MB，原815.818MB，强NJIT811.600MB：基本持平，不能称大额内存节约。常规4096 CC约0.918→0.975秒、FF约0.709→0.688秒；这些诊断变化保留，不据此单独否决已有秒级收益。三条风险链的申请堆与当前基线相同，仍比更省内存的NJIT块高约4–6MB；RSS独立记录，不能用运行时基线差或页噪声掩盖申请堆差距。

最后一组计时在中断时仅有部分样本；确认原父/子进程退出后，原日志、状态和样本保留于`v35r2-supplied-time`及[中断记录](v35r2-performance-interruption-record.json)。恢复仅使用新目录，六轮完整重测及独立峰值见`v35r2-supplied-time-recovery1`、`v35r2-supplied-memory-recovery1`；残缺样本未计入270次。

[设计](v35r2-design.md)及[完整自审核](v35r2-complete-adoption-review.json)保留同一0.000049预算、有限幅值/strides准入、确定性范数与分块/归并误差界、近零符号回放和晚失败整根覆盖。预检拒绝不会改变工作区或结果；库线程状态在异常路径恢复。临时范数先释放再增长主工作区；厂商packing估算不是已证明的malloc硬上界。单次DGEMM、已有批量检查及整根canonical回放仍不可内部取消，不能宣称全路径最坏响应仅一块。

首次五项物理断言失败、原生暴露的拒绝前扩容缺陷及两次准备脚本错误保留。修复未降低0.000049预算，也未改离散、错误或所有权契约。早期状态保留在[历史检查点](historical-full-checkpoint.md)；当前结论以[检查点](checkpoint.json)、终态XML、原始观测和身份文件为准。

已核对[BLIS官方接口](https://raw.githubusercontent.com/flame/blis/master/docs/BLISObjectAPI.md)、[GEMMFIP论文](https://arxiv.org/abs/2302.08417)和[Higham/Mary误差分析](https://eprints.maths.manchester.ac.uk/2731/1/paper.pdf)。前期BLIS本机筛选未胜出；论文中的packing融合是后续候选，不能把不同硬件/float32成绩当作本项目float64证明。后续[公开方法与四个隔离候选筛选](follow-up-v36/README.md)已闭合：更紧误差深度及未发布结果尾部复用都未带来时间收益，未接入运行代码。核对实际Numba→SciPy绑定后，确认NJIT乘法也使用Accelerate；加载的NumPy/OpenBLAS线程池不能证明NJIT的实际库。显式单线程控制后，4096 FF请求V35r2/NJIT分别1.0402/0.6447秒，差距仍开放。历史原始数据和前期[公开方法核对](v35r2-next-method-review.json)保留。

本次未新增逻辑算子、修改AST/DAG、类型/错误/结果/IPC协议或Excel公式展开预算。普通C++执行不运行Excel，也没有重做桌面Excel验收。大矩阵相对强NJIT的时间及内存差距、其他主要完整负载仍开放，整体目标未完成。

后续[V37库调研、诊断与真实入口筛选](follow-up-v37/README.md)已闭合：长K FF路径有时间收益，但无条件改块及仅按输出布局选择分别导致CC→C、CC→F退化，因此两版均未采用。补齐实际隔离AOT相关3360/原生9资格及fresh峰值；未获大额内存节约，下一步需同时考虑输入布局和减少重复归并。实际数值代码仍为V35r2。

后续[V38成熟库分组累加筛选](follow-up-v38/README.md)已闭合：448项私有数学/布局控制及普通/SAN原生各96项通过；两版在同批4096方阵和长K请求均慢于原私有方案，长K分别慢6.9%/8.6%，因此未采用。没有数值回退，减少归并并未带来总耗时收益；未继续fresh峰值或实际引擎全量资格。实际数值代码不变，后续聚焦完整组合链的大额重复工作和中间数组。
