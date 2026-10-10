# 本轮自审核

范围只含有界面板的生产接入、必要预算/工作区改动及其测试与文档。保留开始时的1677文件哈希；不提交、不推送、不改研究平台。

## 数学与库适配

复用Eigen3.4 pack_lhs/pack_rhs/完整6×4 gebp，仅适配赋值与保序独立乘加。输出通过B转置×A转置映射到既有C结果；madd恢复A*B的操作数方向。完整bulk的M前缀是4的倍数，N前缀是6的倍数，固定128/192面板保持整块条件，尾区互斥覆盖，没有在库小尾部分组求和。显式ISA及不适配strides仍有当前职责路径。全部计时数据还须与旧安装包逐位比较，不能仅凭精确整数fixture接受浮点替换。

版本/Index/NEON packet静态断言，独立TU禁用unity/LTO。Eigen真实数表达式blas_traits的NeedToConjugate默认false，实数conjugate仍false；未修改普通false/false实例。静态库中专用true/true定义仅来自matmul_packed.cpp.o；对象及链接worker相关函数无FMA，有分别fmul/fadd。安装worker的整个text与已审查链接worker相同。安装扩展剥离本地符号，不把缺失名称声称为额外反汇编证明；扩展使用同一静态对象并经过完整Python回归。

## 分配、溢出、所有权

双面板320*K+72个double，包含最大7double对齐、两段32double预取保护。形状/helper/复制计数检查均在写之前；内核无私有数值分配或线程。Workspace增长先释放已死缓冲，再按新请求resize；同/小容量复用。Graph不仅观察最大数组面积，还观察本次面板上界，处理M*N不变、K变小的旧scratch释放。Planner按顺序最大面板加计，保留已有其他operator保守上界，scope和并发实例通过原预算复制链处理。直接requirements提供保守形状上界，显式ISA/布局不准入时实际不分配面板。

输入原C/F/补边strides继续借用，不整块转置或复制；算法打包流量如实计入algorithm_copy_bytes。既有MatMul C结果与独立所有权不变，未以此声称全图C布局更快。Prepared快照、single/thread/inline-process/shared-process、嵌套iterate及错误离散消费者均回归。普通block/filter作用域原本只接受1D捕获，测试改用已有合法iterate矩阵捕获，没有扩展生产scope契约。

## 验证与边界

完整Python9017通过；原生/ASan-UBSan/Eigen-OFF各9/9；185 canonical夹具；已有Microsoft Excel真实重算快照与当前公式/参考复比通过，并非本轮新Excel重算。源码、wheel扩展、worker身份一致。macOS ASan未启用LeakSanitizer；非本机平台发布仍需各自CI。

历史phase2保留3/4通过、63点ordinary比值1.018的原始失败；按当前用户规则，微秒级绝对差异不单独否决，不能改写历史门槛为通过。大矩阵完整时间、内存实验独立进行，全部正负样本保留；只有实测后才能填写收益，整体NJIT目标仍开放。
