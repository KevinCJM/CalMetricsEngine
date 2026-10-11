# 带符号、无消去矩阵乘法：2026-10-02 局部采用

复用既有系统 Accelerate DGEMM，把准入从全非负输入扩大到 **A每行、B每列的非零元素分别同号** 的独立矩阵根。完整请求明显提速，但仍未达到全算子时间、内存全面优于NJIT的目标；任意混合符号矩阵未获此加速。当前轮未提交或推送。

新构建 `f90ddcc0b42dddba625903abf18e8860303b8e747548dd14e8b2e2d1ed246a57`；旧构建 `c9f75aed3f4d38fd44caf893b9faba83619c1a094f006870ce6062f814a238ec`。只有3项native构建输入改变：共享实现、内部声明和唯一graph调用者；其余原生源保持本轮基线。见[身份](identity.json)、[原生输入差异](native-input-changes.json)、[精确增量](implementation.patch)；本轮增量不同于整个共享脏工作区的Git差异。

## 数学、布局与所有权

[详细设计](DESIGN.md)。沿用macOS 15+ arm64和原M/N=512–4096、K=256–4096范围、正向C/F leading dimension、独立float64根/私有转置根、无观察器/子作用域/离散消费者的限制。扫描只消费真实元素，不读padding；每次执行重新核对，prepared不缓存数值资格。4KiB是一次扫描标签数组的逻辑容量，实际调用栈由编译器决定。没有绝对值矩阵、输入重排副本、新堆缓冲、全局线程设置或私有线程池。

非零绝对值仍须在[2^-128,2^128]；输入-0/非有限、A行或B列混符号仍回原实现。新增负值域只准入FE_TONEAREST，不改调用者舍入模式。每个点积无正负消去，非零乘积至少2^-256、绝对和最多2^268，因此不存在非零下溢/溢出或消去为零。负域含+0时只把库输出的精确零恢复为canonical +0，不截断非零值。正常最近舍入、常规稳定GEMM误差模型下沿用原无消去边界；不是对任意厂商算法/平台的形式化证明。实际差分仍使用既有rtol=2e-12、atol=2e-14，零符号等分类精确比较。

借用只读输入，写独立已规划结果；F输入不强制改C，私有转置auto/F根直接保留F连续结果。显式F的普通MatMul仍走已有结果布局转换，本轮不虚称其零复制。复用原预算8MiB+512*K*8及audit/IPC，不把vendor reserve当厂商硬上界。直接算子、AST/类型/Excel数学与公式、展开预算均未改。

## 完整请求时间

[9轮原始数据](timing/timing.json)、[测量脚本](benchmark.py)和[冻结公共工具](tools/check_large_matrix_performance.py)。普通Scheduler，实际旧/新wheel，1 CPU，布局和只读独立结果一致；NI使用真实`@njit`矩阵乘法，编译及预热在计时外。三种后端顺序各3轮；只有NI受外部库/TLS线程设置，native依赖自身调度预算。公开结果物化及必要输出转换在计时内。输入与完整输出hash逐轮核对，容差/分类不放宽。

|负载|旧版ms|新版ms|NJIT ms|旧→新绝对变化ms|
|---|---:|---:|---:|---:|
|small [32, 32, 32]|0.009375|0.009084|0.002375|-0.000291|
|positive [2048, 2048, 2048]|90.099208|96.803916|75.992208|+6.704708|
|signed_cc [2048, 2048, 2048]|822.327667|95.252500|76.020208|-727.075167|
|signed_ff_transpose [2048, 2048, 2048]|773.199792|107.086084|83.961209|-666.113708|
|large_ff_transpose [4096, 4096, 4096]|6341.832167|869.257875|816.491625|-5472.574292|
|wide_ff_transpose [4096, 512, 4096]|790.849500|116.916625|98.820917|-673.932875|
|sparse_ff_transpose [2048, 2048, 2048]|792.385666|106.767250|85.382792|-685.618416|
|mixed_sign_rejected [2048, 2048, 2048]|744.135125|763.349875|70.270000|+19.214750|
|explicit_f [2048, 2048, 2048]|812.361458|141.064875|105.864750|-671.296583|

4096阶F转置根减少5.473秒，约7.30倍；2048阶C根减少727.075ms，约8.63倍。新准入负载约5.76–8.63倍提速，但仍慢于NJIT；不能据此关闭全引擎目标。

**保留负结果**：原正值路径增加6.705ms；扫描完整A和几乎完整B后拒绝的负载增加19.215ms（约2.58%），仍远慢于NI。新符号检查带来真实扫描开销；按用户最新规则，短请求的几KB/几ms变化不独立否决重大收益，但持续批量/并发的累计成本仍需按实际负载评估，本轮不外推未测吞吐。小32阶普通请求新版0.009084ms、NI0.002375ms，差6.709µs，不作为门禁。

全部9类完整结果通过原容差和分类检查；4096阶new与NI本轮输出SHA相同，但各自相对canonical有5503464个元素的位模式变化，最大绝对误差6.821e-13；不宣称新旧逐位一致。[结束记录](timing/timing-worker-cleanup.json)确认NI线程模式恢复。

## 内存

[12个fresh进程](memory/memory.json)、[后置验证](memory-validation.json)：2形状×3后端×heap/RSS。heap含setup、首次调用、5次warmup、3次warm测量，RSS含setup和首次请求；没有数值oracle预热库。每个测量结束后，核对输入身份、结果形状和与已通过数值验证的timing结果完全相同的输出hash。36个heap阶段均无tracker overflow/untracked free。

|负载|后端|cold堆峰MiB|warm堆峰MiB|采样RSS增量MiB|
|---|---|---:|---:|---:|
|signed_cc|old|37.010454|69.011005|5.734375|
|signed_cc|new|38.009367|70.009943|18.609375|
|signed_cc|njit|38.000828|70.000935|50.078125|
|large_ff_transpose|old|138.010979|266.011530|10.734375|
|large_ff_transpose|new|134.009922|262.010498|18.671875|
|large_ff_transpose|njit|134.000828|262.000965|146.078125|

2048阶堆峰增加约0.999MiB、RSS增量增加12.875MiB；4096阶堆峰减少约4.001MiB，但RSS增量增加7.938MiB。这些是实测代价，不写成内存持平或零增加；相对秒级时间收益作局部采用。新版堆峰比NI高约9–10KiB，不单独否决。完整内存没有全面严格优势。RSS采样受分配器驻留/页复用影响，不能把本轮很低的RSS增量当输出全部不占内存；不与堆峰相加、不把进程总RSS当增量。heap不覆盖全部Python arena、mmap及调用栈。

## 回归与自审核

- [最终全量Python](python-full-final.log)：三个库期望开关强制ON，**10546通过**；包含新域布局/进程/共享内存、padding、prepared重读、稀疏零、原形状/预算与消费者限制。早期专项316通过见[日志](focused.log)，最终修改由全量复验覆盖。
- [原生](native-test.log)、[ASan/UBSan](asan-test.log)、[三库OFF原生](off-native-test.log)各**9/9**。原生验证非默认舍入拒绝及不写输出/workspace/audit；macOS不支持LeakSanitizer，未声称泄漏扫描通过。
- [OFF最终专项](off-python-final.log)：**107通过、209按库关闭预期跳过**；保留可移植路径、精确结果和ownership检查。[OFF身份](identity-off.json)与安装wheel一致。
- [185直接算子夹具](direct-fixtures.json)scalar/auto通过；[历史Excel快照](excel-snapshot-report.json)**469案例、3056值通过**。公式和输入未变，没有新Microsoft Excel重算。
- [最终源码/ON wheel/extension/worker身份](final-identity.json)及[两种wheel检查](wheel-check.log)、[OFF检查](wheel-off-check.log)通过；Ruff和差异空白检查通过。本轮无依赖/打包规则变动，没有重建sdist。

自审核中保留并修复两处测试问题：关闭库时，转置测试无条件要求发生打包，[旧OFF包也复现失败](off-baseline-test-failure.log)；现在使用既有后端断言并核对graph/direct复制字节，原精确结果、F布局、零arena/rootcopy断言不变。强制ON后发现原packed整数fixture被新准入接管，[初次失败](focused-before-fixture-fix.log)完整保留；共享fixture对同一K的A列/B行同时翻号，使每项乘积及零乘积符号不变、继续测试原packed路径，容量差和复制断言均未放宽。最终三库ON全量及OFF专项通过。

[独立只读审核与自审核记录](review.json)核对准入轴、fenv/零、拒绝无写、C/F/进程/所有权、预算、基准公平性和最终fixture修正，无未解决阻断。执行指南同步实现，公开算子/AST/DAG/Planner/IPC/Excel契约与公式没有变化，因此未改其他数学契约文档；未新增注册、依赖或接口。通用混合符号GEMM、协方差、Cholesky、LU、有界Jacobi等性能欠项保持开放。

[执行口径和复现注意事项](commands.md)。所有冻结文件以[SHA256清单](manifest.json)核对；没有把大结果NPZ、构建目录或wheel二进制复制到仓库。
