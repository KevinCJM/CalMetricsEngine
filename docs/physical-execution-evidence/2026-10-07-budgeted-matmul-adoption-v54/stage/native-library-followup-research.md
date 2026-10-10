# V54：成熟库调研与采用边界

2026-10-07，针对当前 Apple M1 Max（本轮只读sysctl确认）矩形 float64 MatMul 的完整请求欠项。调研不是本机性能证明。

- [Arm 官方 macOS 发布说明](https://developer.arm.com/community/arm-community-blogs/b/tools-software-ides-blog/posts/first-macos-release-of-arm-performance-libraries)：提供 C/C++ BLAS/LAPACK，LP64/ILP64 及串行/OpenMP 版本，静态与动态库。历史说明明确为 M1/M2 优化；若有新几何/实现依据再复测，仍限定串行 LP64，不能靠新增私有线程超过 Scheduler 的核数预算。
- [Arm 当前安装说明](https://learn.arm.com/install-guides/armpl/)：当前示例为 26.07，安装脚本涉及 EULA 和默认系统目录。本轮未运行安装脚本、未变更系统。既有 V49 已安装到临时目录；本轮读取其 wrapper、资格与终态原始记录后确认：serial LP64静态wrapper通过48组控制，但4组/72次含独立输入准备与F结果的请求明显更慢，4096 FF ArmPL/当时C++/NI为3.0613/0.7152/0.6544秒。动态签名失败分支没有执行或重签，详情保留在[既有事实记录](../../2026-10-06-library-screen-v49/README.md)。ArmPL不是未经测试的新候选，不能仅凭介绍重复同一失败实验；新几何复测需有实质依据，且重审ABI/Fortran依赖、AOT、隐藏分配与线程行为。
- [Eigen 官方 October 2026 M4 测量](https://libeigen.gitlab.io/benchmarks/2026-09-apple-m4-blas-lapack-sme/)：同条件单线程比较 Eigen SME、Accelerate、ArmPL、OpenBLAS，但硬件是 M4，非当前已确认的 M1 Max。其 n=1024 double GEMM 数据没有证明 ArmPL 优于 Accelerate；不得直接照搬库选择或 SME 性能结论。该测量只计库调用，不是本项目含输入准备、审核和结果返回的完整请求。
- [GEMMFIP 原论文](https://arxiv.org/abs/2302.08417)：把 packing 与首轮计算融合，作为减少大矩阵重复打包/读取的设计参考。这里的“可参考”是推断；未在本项目实现或验收，不改变当前数值/预算契约。

已有 BLIS firestorm 串行 AOT 和 ArmPL 历史同图/请求预筛测量失败，不重复作为新候选。当前 V53 仍复用成熟 Accelerate；与 NI 使用相同底层库不必然产生整体领先。必须区分库调用时间、输入准备、数值守卫、输出保留和工作区峰值，下一实验优先验证大额复制/打包与组合图复用；独立 MatMul 的秒级相对旧版收益不能代替整体 NJIT 门禁。
