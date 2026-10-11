# V29：复用成熟 Eigen 打包内核，直接写 F 结果

## 事实、边界与目标

V28 完整请求84组/504轮和176峰值完成、全量/SAN/OFF质量通过。2048混合符号高量级输入不满足现有绝对误差准入，V26/V28 F回退7.5–8.8秒，而实际0216约0.8秒；不能接入。F分支在matrix.cpp先调用完整K的4×4 unblocked内核，跳过已有Eigen pack/gebp。V29只修复有序回退的物理写出，不放宽BLAS准入，不新增数学算子，不更改Excel或公开AST/DAG/Planner协议。V28和全部已测来源冻结；只在独立source-v29开发。

## 复用与物理映射

继续复用已有固定Eigen3.4.0 gemm_pack_lhs/gemm_pack_rhs/gebp_kernel及私有true/true特化。当前该内核计算Bᵀ×Aᵀ，其madd明确恢复A再B乘法，+0起始、逐K分离乘加，alpha1覆盖独立结果；这些数学代码、特化组合、6×4full-K路径不变。

现有C写出：kernel(i,j)落在output[i+j*N]。新F写出：同一kernel(i,j)落在output[i*M+j]，kernel i仍是原B列、j仍是原A行。AssignmentMapper/LinearMapper只按F/C使用两种编译期地址映射；F向量包用Eigen公共pscatter写到同一最终F缓冲，标量/编译要求的gather/scatter分支保持零初读与正确轴距。没有逻辑转置、操作数交换、整矩阵置换或第二输出。

prefix仍M向下取4、bulk_cols仍N向下取6，pack192/128尺寸与顺序不变。ordered_region两种尾区仅改变最终输出位置F=j*M+i，按原有A*B逐K左折叠。所有C调用默认false，保留原实现编译路径。F分支在automatic/mac15/arm64/Eigen条件下先尝试同一eligible packing，再用既有F writer。它覆盖既有large M>=1024,K>=256,N>=1024与small-row M32..128,K1..128,N>=128矩形，不按2048精确形状特判。

## 内存、预算和所有权

复用原geometry/Planner已声明的(320*K+72)*8工作区，保持192K+128K两个panel、对齐和预取guards。V28 F路径已经预算并分配这部分但未使用；V29不增加scratch或native库线程池。input只借用、final output由上游独立所有权提供，调用不保存指针。algorithm_copy_bytes仍准确记录pack输入搬运，F尾写与C尾写直接写最终结果；root_copy_bytes和numeric arena均为0。isa标记F私有物理路径，不新增公开算子。process传输仍按声明F保存，旧结果不得被复用覆盖。

## 正确性与审核

对mapper证明轴/leading，检查gebp full/tail的所有写口。新增用例必须包含非方阵、M%4/N%6边界、C/F四种输入、正padding、负步长拒绝、IEEE/cancellation、两代只读结果、single/thread/process_inline/process_shared和预算。大矩形高量级例必须明确accelerate0并独立整数公式核验，不能只比较同一个packed函数。

正常影响范围、原生9项、正常全量、SAN影响范围及原生、Eigen/Accelerate/Highway全OFF影响范围及原生；所有源码/wheel/worker身份同源且无skip。未通过时保留冻结源码和原始失败，不通过改数学/删用例消除失败。

## 性能验收

先验证新候选能消除2048 CC/FF F回退的秒级退化，再做原完整12案例矩阵对照（实际0216、V28、新候选、真实预热NI direct/copy）、60组/360轮及80fresh峰值。补足large rectangular F回退，确认不是方阵专用修补。继续复核derived/mixed/supplied最大CPU4 C/F（各8组/48轮及32峰值）。全部计入输入准备、普通Scheduler、整个计算链与两代结果。

本轮不把ABI/布局144次内核诊断当作完整验收，不采用尚未完整资格验证的preferred-layout方案。消除8秒退化仍不代表高量级回退已胜过NI约0.08秒，普通4096与强NI的时间/峰值也仍需闭环。未通过全部证据前不修改实际R数值源码、提交或推送。
