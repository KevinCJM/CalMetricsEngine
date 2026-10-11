# V22：独立 MatMul 的一般符号 GEMM 与 F 根直写

2026-10-05；隔离候选，尚未构建或采用。基于冻结 V21 源码，实际工作树仍为 88ea。

## 目标及依赖

解决现有一般符号大矩阵未使用 GEMM，以及普通 F 矩阵根保留完整 C arena 后复制的问题。只修改既有 MatMul 的物理实现，不新增算子或数学定义。单一 typed float64 矩阵根、可选唯一 transpose、无 scope/isolate/下游消费者的现有证明继续约束 GEMM。离散、多根共享、observer/Excel 数值域诊断及预绑定内部路径继续 canonical。

## 数学准入

保留 macOS15+、M/N512..4096、K256..4096、借用正向 C/F（含 padding）、非零绝对值 2^-128..2^128、无负零/非有限值及既有线程控制。A 每行/B 每列同号域沿用当前路径。一般符号域要求 FE_TONEAREST；在标准 binary64 乘加误差模型下，canonical 左折叠和库点积的差以 2*gamma(2K)*K*max|A|*max|B| 保守估计，正运算逐步向上取界。只有界 <=0.000049 才准入，符合本次用户小数点4位后尾差许可；不截断结果，不改变 dtype。幅度界排除溢出及下溢。库输出绝对值 <=界的格用唯一 canonical dot_value 回放，保护零/符号抵消及稀疏零；其他格满足同一数学乘积与明确误差界。此误差模型是准入依据，需要独立长精度/残差与真实库差分验证，不能冒充厂商的逐指令证明。

## 布局与分配

DGEMM 直接接受借用的 C/F inputs，通过行/列主序和 transpose descriptors 写独立 C 或 F output，不做输入转换。F 独立根的 metadata 准入必须同时存在 portable canonical F writer；不能只靠库成功假设移除 arena。复用当前 Eigen 保序四行/四列模板，仅改变 output map/尾格位置；库关闭、小形状或特殊 stride 时复用 dot_value 写 F。这保留 ascending-K 和原 A*B 操作数顺序，不以交换 A/B 来翻转存储。完整 F arena 和 root copy 的理论减少量为 M*N*8（4096方阵128MiB），实测前不称为峰值收益。输出不可变、独立所有权以及 IPC layout 描述继续现有实现。库内部 reserve 沿用原估算，非数学严格上界，必须测实际 heap/RSS。

## 调度、恢复与失败

BLAS thread-local single 模式、恢复校验及 Scheduler 外层 CPU预算不变。无全局环境 mutation/新线程池/跨请求数据缓存。每次读取本次数值准入，prepared 地址不作为内容身份。失败准入不写 output/workspace/audit；库关闭和不支持平台有可移植执行。所有权、结构错误、未消费 padding、范围和 input 不变。

## 验收计划

1. 新域随机 C/F/auto 和 transpose，single/thread/process inline/shared，padding/只读/保留旧结果；长精度及数学双线性不变量。
2. canonical F canary，负 stride/broadcast/empty，非有限/负零/四种 rounding，取消/错误路径；一般符号超误差界拒绝且不写。
3. 抵消到零/近零回放，混合符号 prepared 内容变化，已有严格消费者和 scope/isolate 不使用 GEMM。
4. 完整全量、相关、native、ASan/UBSan、三库关闭；冻结 source/wheel/worker 身份。
5. 完整普通请求对照当前 C++ 和 warmed NJIT，1024/2048/4096，C/F输入与输出，CPU相同且实测 BLAS 1线程；完整输入准备、调度、保留旧结果的峰值 heap 与三次 fresh RSS。
6. 所有重型构建/测量在当前 V20b/V21正式链终止后串行运行，避免互相干扰。失败结果保留，未验证路径不进入实际代码。

## Excel 与文档

既有 MatMul 数学、逻辑 DAG 与公式不变；物理布局不影响 Excel 单元格映射，普通 C++ 仍不调用导出。适用规则 D1/D2/D5/D6/D8 与 X1/X6/X9/X10；实际重算证据单独管理，本候选不宣称启动 Microsoft Excel。候选及本轮实测在原候选汇总设计中追加 scoped 状态，采用后再同步 execution-guide 的真实路径。
