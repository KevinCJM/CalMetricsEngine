# 有界面板 GEMM 正式接入设计

本轮前置证据：2026-09-30 gemm-workspace，192×128固定方案在4096F/F节省41.77秒，增加约10MiB。上轮为有实测证据的进展；没有等待中的数值进程。整体目标仍是小任务无明显劣化、中大型完整时间与内存领先NJIT；本阶段接入不能缩小或宣称完成该目标。

## 数学与准入

复用固定Eigen3.4 pack/完整6×4 gebp，保留升序K、分别乘加、A*B操作数方向及从+0开始。固定面板192×128；M,N>=1024、K>=256，两个输入均C/F或正leading-dimension补边布局。布局不兼容、重复行、显式scalar/ISA、Eigen-OFF、未验证平台和macOS15以前保留现有实际职责路径。新库物理实现不是新算子；只用已有数学尾部填最多3行/5列。NaN/Inf、signed zero、失败与输入/输出契约不变。输入不整块转换，当前MatMul结果仍为其已有独立C输出，未推广为全链C策略。

## 分配与图预算

形状上界共用matmul_scratch_doubles：320*K+72个double，覆盖192K/128K面板、两段256B预取保护及64B对齐。Prepared报告形状预算；执行仅在auto且真实布局准入时申请。使用现有独占Workspace::doubles；不引入内核私有malloc、缓存或线程。

原Workspace::ensure的double扩容可能几何增长并同时保留旧、新数组。scratch旧值没有跨算子身份，增长时先释放再按实际数量分配，等量/缩小请求保持复用。graph::matmul_workspace_doubles由Planner和Scratch共享：顺序节点只取最大面板，嵌套scope和并发worker按既有递归/复制路径分别计预算。未知形状以实际数组capacity作K上界。原其他算子的保守scratch估计保留，与面板项相加形成安全上界，暂不以减小几MiB估算为目标。

同一图可以保持最大数组元素数不变、改变K；仅比较arena_window会保留超出新计划的面板。因此Scratch记录本次面板上界，变化时释放旧double scratch；结果、状态、捕获输入均保留各自独立所有权。

实际打包字节数按K*(M完整块 + ceil(M完整块/128)*N完整块)*8计入algorithm_copy_bytes；输入绑定复制与算法打包分开。所有乘加/对齐边界写前检查。内核使用受预算缓冲，分配失败经既有C++边界传播。

## 构建与验证

特殊Eigen traits限定单一TU，固定版本/packet/mr/nr静态断言，关闭该TU的unity/LTO；正常Eigen实数GEMM不会选择true/true专用特化。hidden visibility不替代ODR核对。保留最终对象反汇编无FMA检查与普通Eigen路径测试。

回归覆盖大形状四布局/补边/尾部、原顺序与IEEE、显式ISA及小形状不分配、重复调用、工作区增长、相同最大面积但K缩小、预算不足、单线程/多线程/进程与共享输入、结果独立和下游比较。完整Python/native/ASan/Eigen-OFF、185夹具与适用Excel历史快照复比单独记录，不声称新Excel重算。然后以真正安装wheel做完整Scheduler旧/新/NJIT配对和独立进程内存；小额差异不独立否决，原测量保留。
