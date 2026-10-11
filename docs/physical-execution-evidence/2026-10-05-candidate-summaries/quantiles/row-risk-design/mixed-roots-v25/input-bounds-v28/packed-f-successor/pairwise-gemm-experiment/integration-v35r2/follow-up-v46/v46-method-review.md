# V46：成熟库方法与源码审核

本文保留接入前候选阶段的判断；当前已完成实际接入与复测，终态见同目录README及v46-review.json。

2026-10-06。本轮按重复未解瓶颈继续调研，并在当前V45基础上建立隔离完整AOT候选；没有安装新依赖或改实际运行源码。

- [Eigen官方归约/访客说明](https://libeigen.gitlab.io/eigen/docs-nightly/group__TutorialReductionsVisitorsBroadcasting.html)提供minCoeff及boolean all等现成能力。本轮沿实际连续物理行使用这两类检查，先全值finite，再判断最小绝对值；非有限不会被minCoeff掩盖，含模糊值的行仍逐单元进入原canonical fold。网页为nightly概念资料，实际复用本项目固定3.4.0归档，不升级依赖。
- [Apple DGEMM接口](https://developer.apple.com/documentation/accelerate/cblas_dgemm(_:_:_:_:_:_:_:_:_:_:_:_:_:_:))声明C/F、转置、leading dimension等参数。保留已有Accelerate DGEMM、每调用线程的预算设置/恢复与独立最终布局，不通过更高核数、结果转C或移出输入复制成本换成绩。
- [Higham的Strassen数值分析](https://nhigham.com/wp-content/uploads/2023/08/high90s.pdf)给出快速乘法与经典乘法的不同误差界，不能用代数等价证明它满足现有绝对预算。结合本项目高量级负载、额外加减缓冲和先前实测，在本轮不采用Strassen；这是本项目取舍，不是声称所有Strassen实现都更慢。
- [BLIS硬件支持](https://github.com/flame/blis/blob/master/docs/HardwareSupport.md)用于核对平台/ISA边界。先前固定firestorm单线程的完整筛选已慢于现有Accelerate且无大额峰值收益，保留拒绝证据，不将其他机器成绩当本机证明。

新候选只按真实FF输入且F输出的大几何选择窄块，不再按输出F单独选择；其他几何起始tile不变。已有根上限8192与Eigen内部有限/幅值/零边界扫描对齐，字段、原限制和全值检查不放宽。唯一finish_tile在两种归并策略中共用，保持误差界、每个模糊单元的K升序、全部根回放、输入只读及旧结果独立。连续行Map为只读表达式，不制造全长缓存、第三矩阵或额外线程池；正常计算写结果不作为复制。

资格边界：候选正常原生9/9、相关1513项通过，无skip；独立私有元数据/有限/近零/canary控制262项在普通与ASan/UBSan均通过。私有sanitizer不是新完整引擎SAN或全量资格。此前V45的15203全量不是V46全量；还需根据完整性能筛选决定后续全量/变体与实际接入，不提前宣称生产收益。

首次性能驱动器因后端重命名后的旧断言失败，没有收到正式样本；其任务已终止回收，原脚本/日志/状态保留于v46-initial-attempt。修复的是仍要求旧后端名的物理路径断言，当前/候选均仍必须证明多次Accelerate分块调用；数值实现和数学容差未改。

## 实际接入完成后的资料复核

2026-10-06，实际构建18次完整计时与8次fresh峰值均已终结。长K完整请求3.0205→2.3530秒，强NJIT1.3152秒；连续malloc峰值基本不变。全量15219属于与当前93份native/build输入完全同源的候选；实际目录重建另验1513相关及9原生。完整引擎SAN1513、OFF1173和各native9均已通过，先前待验事项不能再据历史快照标为正在运行。

- [Eigen 3.4官方BLAS/LAPACK接入说明](https://libeigen.gitlab.io/eigen/docs-3.4/TopicUsingBlasLapack.html)明确密集乘法、LU、Cholesky、QR、特征分解等可转到现成BLAS/LAPACK，并说明部分替换的数值稳健性差异。网页当前生成版本为3.4.1，本项目仍固定3.4.0；本轮不升级或全局打开替换宏，也不由功能支持推定性能收益。
- [BLIS firestorm真实配置源码](https://raw.githubusercontent.com/flame/blis/master/config/firestorm/bli_cntx_init_firestorm.c)有double GEMM、packing、布局偏好和MC/KC/NC设置。硬件概览表可能滞后，应核对配置源码。它解释可调物理策略，不能推翻本机之前已慢于Accelerate且未减内存的完整筛选，本轮不新增BLIS依赖。
- [Goto/van de Geijn矩阵乘法论文](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)讨论分层存储、packing与TLB对内核周边算法的影响。由此形成的本项目推断是：下一步应该测归并、重打包和组合链总成本，不能只数DGEMM调用；论文老机器成绩不代表本机提升或本项目误差预算已经满足。

重复多轮仍无显著收益时，先查阅成熟库官方文档、目标平台实际实现和公开算法方法，再建立隔离候选；仍需同核数、相同输入/输出/保留方式下的完整请求与fresh内存验收。可采用显著提速且内存基本持平的阶段优化，但大矩阵仍慢于强NJIT、其他主要平台完整链及全面分配审计继续保持开放。没有以尾差许可为由修改原0.000049数学契约，没有新算子/API/依赖/线程池或Excel公式变更。
