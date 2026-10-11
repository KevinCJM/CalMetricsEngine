# Jacobi 连续前缀与行距筛选

2026-10-01；当前生产基线为 `6f02e6bb69d2d59a62781c357eda16f9604ae9eee2377af383b3b56b29c2bf6f`。目标针对稠密1024阶仅值约3秒、完整约5秒与NJIT/LAPACK约0.1–0.16秒的差距；不是为毫秒/KB进行微调。本阶段先隔离验证，不修改生产数学内核，不提交/推送。

## 依赖证明与详细设计

保留canonical缩放、逐sweep的i/j顺序、tau/t/c/s、每个元素乘减加顺序、tolerance/max_sweeps与NON_CONVERGENCE、稳定排序、符号归一化、完整/仅值所有权。成熟Eigen JacobiRotation继续负责连续区段，不能直接换成不同停止语义的LAPACK。

原packed上三角在k<i前缀访问时同时跨列扫描(i/j列)。候选使用一个N×N的私有矩阵，处理进度为当前行i：

- 未处理的行k≥i，其有效上三角仍按C行存储。
- 已处理的行k<i，其有效值(k,j)存于转置位置(j,k)，此时沿k读两个列前缀，实际上是两段连续行，可由现有Eigen Map直接旋转。
- 当前i的中间区间i<k<j仍读上三角，不改变数学；j后的连续尾部和特征向量仍走既有Eigen旋转。
- 每个i的全部j处理结束后，只把该行上三角复制到转置下三角；每个sweep总O(N²)复制，不在每次旋转都维护两份矩阵。此后原上三角那一行不再消费。
- sweep结束后，有效非对角值全部在下三角，停止检查按原逻辑i/j顺序读其转置位置。仅在确需下一sweep时，把下三角转回上三角，更新处理进度；对角始终在原位置。

这是内部表示的生存期优化，不改变逻辑依赖或公开形状。每次sweep只有一份权威值；旧位置不同时被不同消费者读取。最后特征值取对角，排序继续借用已死矩阵作行缓冲，原输入只读、公开输出独立。失败缓冲/错误与观察器等价必须通过差分，不能仅比较成功残差。

候选从现有canonical模板机械派生，用现有静态库提供Prepared、MatrixRotation及基础语义，不新增Python数学运行路径。Workspace由packed约N(N+1)/2增为N² doubles：1024阶多约4MiB、2048多约16MiB，属于实质增长，必须与NJIT完整请求、预算和大额时间收益一起判断，不能称为KB波动。先筛选n≤1024，若没有显著收益不接入，不追求小差。

## 验证与采用条件

C/F/补边/反向；空/边界尺寸；默认/显式tolerance与sweeps；重复特征值、对角排序、有符号零、缩放极值、非有限/非对称/未收敛。完整值/向量及仅值与同一真实原生基线逐位对照，记录失败前缀、错误码、停止差异、输入hash和canary；原数值口径不放宽。随后单CPU、独立结果保留、均衡轮换，测512/768/1024稠密完整/仅值及成熟NJIT/LAPACK参考。

只有隔离正确性和显著时间收益成立后，才设计Planner上界、auto资格、四lane与安装wheel的正式接入及全回归、堆/RSS、Excel快照复核。Excel保持同一逻辑有界Jacobi步骤，不把物理地址转存展开成另一算法。本阶段没有正式采用结论，整体全算子目标保持开放。

## 已完成的时间与正确性筛选

原稠密候选及[补边行距控制](padded-design.md)分别通过107组、214次完整/仅值逐位对照；包含失败前缀、无写入拒绝、只读输入与输出canary。每组直接调用已安装原生完整算子交叉核对（成功且n≤256时），大规模则核对真实普通Scheduler。两种候选都未放宽数值、排序、符号或停止条件。

| 阶数 / 结果 | 原 C++（稠密试验） | 稠密候选 | 原 C++（补边试验） | 补边候选 | 热 NJIT（补边试验） |
| --- | --- | --- | --- | --- | --- |
| 512 / 仅值 | 372.620 ms | 802.265 ms | 375.633 ms | 267.317 ms | 16.542 ms |
| 512 / 值与向量 | 556.439 ms | 998.590 ms | 557.975 ms | 462.425 ms | 26.160 ms |
| 768 / 仅值 | 1,285.534 ms | 1,785.020 ms | 1,280.381 ms | 886.638 ms | 42.404 ms |
| 768 / 值与向量 | 2,024.436 ms | 2,493.901 ms | 2,012.424 ms | 1,590.030 ms | 64.617 ms |
| 1024 / 仅值 | 3,122.121 ms | 10,811.752 ms | 3,117.654 ms | 2,266.933 ms | 109.911 ms |
| 1024 / 值与向量 | 5,120.845 ms | 12,592.619 ms | 5,139.848 ms | 4,334.598 ms | 169.402 ms |

两次试验分开执行，均为单CPU、六轮均衡轮换与独立结果保留；列中同一次试验原C++才是配对基线。完整结果用两个合法公共投影根，候选也复制为独立值/向量，不把内部record伪装成合法公共根。仅值与NJIT做既有1e-9比较；完整向量只与原C++逐位比较，**没有完成向NJIT每个向量元素的对照**，不把LAPACK与有界Jacobi误写成同一算法。候选仍是ABI筛选，缺少正式入口全部成本。

稠密方案不接入，1024阶实质慢约7.69/7.47秒；补边只是增加8KiB却使仅值从10.81秒降到2.27秒，说明行距选择对这一负载有重大影响。缓存组冲突是合理解释，未采集硬件计数器，不能称为已证明因果。补边相对原C++少约0.851/0.805秒，仍是NJIT约20.6/25.6倍，不能关闭全引擎性能缺口。

1024阶补边工作区约8.016MiB，包含8KiB索引；相对原packed约多4MiB，区别于补边本身的8KiB。内存及生产预算须据实复核。几KB/几毫秒增量不会单独否决，但不能把额外4MiB隐去。

## 独立进程内存筛选及采用结论

探针控制通过，12个fresh进程（六后端×heap/RSS），追踪溢出均为0。1024阶同固定输入、单CPU、独立结果，编译预热在epoch之外，heap连续记录setup、首结果释放与warm峰值，不相加重复计量；全部阶段见`memory/reports.json`。

| 结果 / 后端 | 冷 heap 峰值 | 热 heap 峰值 | 热执行 RSS 增量 |
| --- | --- | --- | --- |
| 仅值 / 当前普通引擎 | 4.021 MiB | 4.029 MiB | 0 MiB |
| 仅值 / 补边 ABI | 8.025 MiB | 8.033 MiB | 0.047 MiB |
| 仅值 / NJIT | 8.274 MiB | 8.282 MiB | 0.531 MiB |
| 完整 / 当前普通引擎 | 20.037 MiB | 28.045 MiB | 8.125 MiB |
| 完整 / 补边 ABI | 16.025 MiB | 24.033 MiB | 16.203 MiB |
| 完整 / NJIT | 25.304 MiB | 33.312 MiB | 15.313 MiB |

必须区分生命周期：ABI每次调用释放workspace后再复制完整投影，当前普通引擎保留workspace及arena，hot阶段有不同常驻缓存。故ABI完整heap看起来少约4MiB，**不能直接认定同一生产Planner接入后也少4MiB**；若原样保留新增workspace，正式峰值可能更高。仅值ABI多原引擎约4MiB，NJIT约少0.249MiB。需要在正式整图中证明死工作区/record生存期及精确预算；不以微小差值遮盖生命周期差异。

完整ABI的RSS请求增量高于NJIT约0.891MiB、原引擎约8.078MiB；仅值原引擎有0.047MiB idle对照。本轮只有每配置一次fresh观察，页级增量、分配器复用和常驻workspace分别解释；不能将RSS零增量写成零内存。malloc探针不完整覆盖mmap、NumPy/Python缓存和其他进程，且记录申请字节，不是完整物理内存。

取舍：原稠密候选不接入；补边有0.8秒级C++收益且逐位一致，保留为待正式设计的候选。接入前须处理额外4MiB工作区、未累计搬运copy、arena/workspace生存期、CPU预算及完整入口。它仍比NJIT慢约20–26倍，因此不是全引擎性能通过；不因补边的8KiB或页级微差否决，也不提前声称生产采用。未提交或推送。

## 自审核、身份与复现

- 生产C++、AST/DAG、Planner、公开错误/结果协议与Excel均未修改；本轮未重跑全量回归、sanitizer或新Excel重算。上一阶段9,966项不能作为新候选验收。
- 初次计时误用内部record公共根，按既有规则非零拒绝；原错误与身份保留。修正为两个合法投影后重新运行全部512/768/1024，没有跳过结构检查。
- 稠密计时完成后显式绑定同步循环闭包并加入`zip(strict=True)`，修复Ruff诊断；`screen-timing-loaded.py.gz`保存实际运行时的源码，解压SHA与`timing-identity.json`核对通过，不能用后来脚本hash替代。补边试验加载最终修正脚本，单独身份记录。
- `candidate.cpp`是隔离ABI，不是公开API：数组范围、所有权和单线程在测试持有者内控制，不等于任意调用方校验。异常审计、CPU admission、Planner内存上界、四lane、非目标平台及观察器链路尚未接入。
- 原型copy审计只记初始矩阵，不完整覆盖每sweep的上下三角搬运；若正式接入必须如实累计这部分copy。记录的workspace容量和实际库调用不是完整复制/分配证明。

```bash
/usr/bin/clang++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off -mmacosx-version-min=15.0 -dynamiclib -Icpp/include docs/physical-execution-evidence/2026-10-01-jacobi-frontier/candidate.cpp /private/tmp/calmetrics-covchol-next/wheel-build/libcalmetrics_engine_operators.a -framework Accelerate -o /private/tmp/calmetrics-jacobi-frontier.dylib
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 docs/physical-execution-evidence/2026-10-01-jacobi-frontier/screen.py correctness
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 docs/physical-execution-evidence/2026-10-01-jacobi-frontier/screen.py timing
```

补边源按`padded.patch`从同一`candidate.cpp`生成到`/private/tmp/calmetrics-jacobi-frontier-padded.cpp`，相同编译命令换源和输出dylib名称；然后运行`padded.py`。`memory.py`分别以独立进程调用heap和RSS，探针注入只用于heap，计时不加载探针。`derivation.json`、各identity、原始JSON/日志与manifest约束证据身份。
