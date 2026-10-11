# 有界LU分块与后置残差工作区复用（隔离设计）

保持root924原样，基于已全量/SAN/OFF验证的ac54候选。只优化既有solve_multi数学与物理图，维持类型、阈值、错误、结果所有权及Excel公式；本阶段不新增canonical算子、Python数学、数值缓存、线程池或IPC变化。

## 底层库选择

Accelerate DGETRF完整入口在4096筛选中因内部工作区多约13.7MB。按Netlib DGETRF的固定512列分块调度DGETRF2、DLASWP、DTRSM、DGEMM，全由成熟库执行因子/主元/三角/矩阵数学。仅C++有界调度，不手写消元或GEMM。单一Scheduler预算内限制BLAS线程并恢复原thread-local设置。SDK/平台关闭仍使用唯一原canonical路径。官方顺序：https://www.netlib.org/lapack/lapack_routine/dgetrf.f 。初筛不等于完整任务验收，不用局部结果决定采用。

## 依赖与独占缓冲

系数factor仅在DGETRF、canonical_lu_certificate与DGETRS期间存活。DGETRS完成后，无任何消费者需要LU因子，故该独占N²区可装下64N absA面板以及3NK残差数组（N>=1024、K<=64时严格容纳）。RHS解始终保留在另一个NK区；5N证书缓冲紧跟RHS，只在factor尚存活时使用。生成残差前检查所有解有限/范围；发布结果前完成原方程与DD forward-bound校验。不重用结果/输入/公开系数或共享区。private producer仍只从持久叶父重建残差；失败由调用者重建canonical系数并同体回放，retained原系数保持不可变。

Planner scratch查询改为NK+5N，copy query仍包含N²、ABI与512N保守vendor估计，private显式F回放需完整N²覆盖不变。实际vendor估计不能当硬上界，堆/RSS完整实测。真实copy/residual计数保持，kernel ISA名称说明实际panel库路径。

## 验证

partial32/64/512边界、1024/1025/1537、RHS1/8/64；readonly/public/shared/prepared两代结果、近阈值状态反例、失败恢复、四lane、normal/SAN/OFF。完整任务仍同输入、CPU1、相同C/F输出、owned readonly准备和两代独立结果；warm NJIT，排除编译，六轮轮换；申请峰计入fixture和构图，RSS fresh进程补证。主负载及retained内存问题不因局部筛选而宣称关闭。Excel公式与符号预算不变，无普通计算Excel成本。
