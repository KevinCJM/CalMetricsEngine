# 有序F回退修复候选：完整质量通过，性能验收中

2026-10-06。根据[完整性能失败证据](../complete-performance/README.md)，详细设计后在独立source-v29复用现有成熟Eigen pack/gebp内核，加入F最终结果地址映射，避免F提前绕过打包内核。数学、逐K分离乘加、BLAS准入、误差预算、输入所有权和工作区预算保持。

[精确变更](source-v29.patch)仅3个私有C++运行源及3个测试文件，新增49项，包含非方阵/M%4/N%6、C/F四种输入、IEEE/零/取消、padding/负步长/广播、两代只读结果、single/thread/process_inline/shared与预算。正常影响范围2,478项和原生9/9全部通过，无失败/错误/跳过；normal build20e3a7b0，259个可复现源与wheel/扩展/worker身份已复核。[阶段证据](checkpoint.json)。

完整14,764现已全部通过，无失败/错误/跳过；同源normal wheel/扩展/worker身份已复核。[完整正常证据](normal-full-proof.json)。SAN相关2,478/native9也已全部通过，无失败/错误/跳过；[同源wheel/扩展/worker身份已核验](sanitizer-proof.json)。三库OFF相关2,238/native9也已全部通过，无失败/错误/跳过；[完整终态资格](v29-terminal-quality-proof.json)核验3种独立构建、90个一致数值源、259个冻结文件及全部wheel/扩展/worker摘要。正式性能94组/564轮与256fresh峰值已准备，加入矩形高量级F回退及其峰值；全部质量与精确源码/构建身份核验后，已经启动同一正式串行队列。当前没有新候选性能结论，8秒退化是否消除尚待观测。

[冻结设计](v29-packed-f-design.md)中“公共pscatter”的表述需更正：采用的是Eigen::internal::pscatter，属于现有固定Eigen3.4.0的私有ABI适配，沿用现有特化组合；没有新增公共Eigen API保证。冻结文件为了构建证据未在编译期间改写，此处明确真实边界。

实际工作区数值源码仍为0216，没有接入/提交/推送。公开使用说明、算子、Excel生成与数学文档无需变化，因为本候选只改变私有最终结果地址映射，未改变任何公开计算语义；整体性能目标仍开放。

在正式性能尚未开始的前期发现矩形K257在scale1024下可能通过既有误差准入。已核对并停止本轮唯一的等待进程（0个测量job），保留旧驱动/设计/取消记录；[修正的冻结设计](v29-corrected-performance-design.json)使用K513，同时每次rejected_scale引擎调用明确要求accelerate0，保证测到回退。数值源码、测试和质量队列均未修改。94组/564轮与256峰值的数量不变。

已核对真实平台完整链的剩余范围，并冻结正式campaign结束后的实际调用栈定位；见[覆盖边界与定位方案](scope-and-profile.md)，不据此宣称完整随机探索/前沿或性能通过。

为防止仅按大方阵推广，新加入打包准入外的长内积/非方阵补测；见[补充范围与顺序](fallback-supplement.md)，尚无成绩。

完整矩阵70组/420轮已结束，含全部正常/高量级/非方阵/小诊断。秒级F旁路退化已恢复，但高量级仍比真实NJIT慢约9–10倍，交叉布局4096也尚未领先；见[完整计时与保留负例](complete-matrix-time/README.md)。峰值和风险回归仍未结束，不宣称性能闭环。

主正式性能94组/564轮与256个fresh峰值均已执行完，完整时间和内存分别核对最强NI；高量级和整体NI峰值目标仍未解决。见[全部计时、峰值与保留问题](complete-performance/README.md)。打包准入外补测/调用栈、实际源码接入与完整平台资格仍开放。


准入外[50组/300轮和80fresh峰值](outside-packed-complete/README.md)及[三条真实栈采样](complete-stack-profile/README.md)已结束；发现V29 F根最高约6.1倍对当前退化。已写[隔离缓存修复与更正测试](f-cache-successor/README.md)，相关2539/native9通过，完整/SAN/OFF与新完整性能尚未结束。当前运行源码0216未改变，整体目标未完成。


后续事实更新：V31[全量/SAN/OFF质量已关闭](f-cache-successor/closed-quality/README.md)，[50组/300轮打包外计时](f-cache-successor/closed-outside-packed-time/README.md)已结束，确认退化修复但仍有NI明显差距。另有[掩码校验与子图漏查隔离后继](mask-validation-successor/README.md)，237新增覆盖与构建/测试均未验收。其余性能/峰值和实际接入开放。


最新闭环：[V31完整144组/864轮与336fresh峰值结束](f-cache-successor/closed-performance/README.md)，质量通过而整体性能目标仍开放；[后继26处漏查及失败/修复依据](mask-validation-successor/qualification-failures-and-repair/README.md)记录候选仍待完整验收。当前实际0216没有变更。


最新质量事实：正常全量15062、正常/SAN相关各3228、OFF相关2988及三套native各9项全部通过；完整性能仍按精确流程继续，实际代码接入等待其冻结旧基线测量结束。另有[高量级矩阵分块BLAS原型](pairwise-gemm-experiment/README.md)，152项小输出长K检查通过，尚无普通Scheduler完整性能或实际接入结论。


当前事实：[V33实际源代码接入、当前目录1195及完整70/420、280峰值闭环](integration-v33/README.md)。旧实际0216与候选阶段的未接入描述按形成时版本理解；高量级NI缺口保留，整体目标未完成。


后继诊断事实：[范数库30/180及120fresh完整闭合](pairwise-gemm-experiment/refined-norm-bounds/closed-prescreen/README.md)，仍慢于强NI；[宽块保持152输入的资格与初版设计否决](pairwise-gemm-experiment/wide-block-qualification/README.md)，未进入引擎；后续25/150与100fresh已闭合，仍慢于NI。
