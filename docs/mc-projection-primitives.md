# 保序收缩、稳定指数差与共享投影

2026-10-03 阶段补齐平台计算链需要的两个通用数学原语。它们不包含金融输入规则或业务命名的模拟黑盒。原 ID 1–185 的契约不变，新 ID 为 186–187；从[算子参考](operator-reference.md)查询当前目录。

## 1. 数学契约与粒度

### affine_contract(values, coefficients, initial)

最后一轴具有显式初态的升序收缩。float64 输入：`X[B,F]` 或 `X[T,P,F]`、`K[F,A]`、标量初态或 `initial[A]`；输出为 `X` 前置轴加 `[A]`。

```text
accumulator = initial[a] 或标量 initial
for f = 0 .. F-1:
    term = coefficients[f,a] * values[...,f]
    accumulator = accumulator + term
```

乘法与加法分别舍入，禁止 FMA、归约重排或省略零系数。`matmul(X,K)+initial` 的初态和舍入顺序不同，不是该原语的替代。系数位于乘法左侧，保留参考内核写法。NaN/Inf/溢出按 IEEE 传播；空 F 返回初态，任一输出轴为空时返回对应空数组。类型、维度、收缩轴、初态轴及 Typed IR 量纲相容性在写结果前校验。

它解决独立的“带初态保序收缩”，属于 primitive；指数变换、权重计算和归约都是可单独替换的真实后继节点。现有 BLAS GEMM/GEMV 无法保持上述初态折叠合同。直接算子和可移植路径复用唯一保序 fold；已准入的私有投影闭包用 Eigen Array 同时计算八个独立资产，仍实例化同一 fold，不能用默认 GEMM 静默替换。

### expm1(values)

float64 标量或 rank1–3，逐位置调用成熟原生 `std::expm1`，输出 shape/axes 不变，Typed IR 输入为 dimensionless。近零稳定，保留负零；NaN 与正 Inf 传播，负 Inf 返回 −1，有限正溢出返回正 Inf。原 `exp` 的非有限输出 `DOMAIN_ERROR` 合同保持不变。

它是独立的 primitive，不通过 Python 回调，也不以 `exp(x)-1` 代替。首轮没有新增显式 SIMD 标签、私有线程池或向量指数库准入；库已存在不等于实际使用。

## 2. AST/DAG、布局及所有权

两项通过唯一 canonical 注册表、原有 AST 调用语法和结构 CSE 进入共享图。CSE 身份包含全部输入，尤其 `initial`；prepared 每次读取当前内容，不缓存上一次数值结果。rank3 仍遵守 time 首轴契约。

只读 C/F、合法正负 stride 输入由边界保活并借用。直接 API 的 `out` 继续要求 C 连续、不别名输入；两个算子的 typed 图根可以显式 F 直写独立结果，不强制先转成 C。普通结果和 snapshot 可独立保留；prepared 的复用结果继续遵守既有生命周期规则。新 opcode 经过现有 native worker 协议，worker 与扩展须为同一构建。

固定已变换带的显式组合：

```text
L = affine_contract(factors, transpose(coefficients), initial)
R = expm1(L)
period_c = sum_axis(multiply(broadcast_like(weight_c,R),R),last_axis)
contribution_c_a = sum(multiply(first(slice_axis(weight_c,0,a,a+1,1)),
                                slice_axis(R,last_axis,a,a+1,1))) / P
```

多个组合共享同一 L/R。权重位于乘法左侧；period 沿 asset 升序、初态 +0，贡献沿 step→path 顺序单个累加器，只在末尾除 P。不得先算均值再乘权重、合并 tile 小计或跳过零权重与 Inf 的乘法。按消费者需要显式交换 period 轴，不由 Python 数值适配改变语义。

## 3. 有界物理执行设计

一般图路径用于全部已声明输入布局和可编辑中间输出。大型默认采用方案须识别上述真实完整消费者闭包，在 strict typed、无作用域/isolate/观察器且全部依赖受检时，按有界 tile 计算资产映射并立即消费各组合 period/贡献。公开 L/R 或额外不支持的消费者保持完整通用图，不删结果。

计划和执行使用相同几何准入；拒绝物理优化不能在已按紧预算准入后偷偷分配完整张量。单 tile 最多约 1 MiB，贡献保持每根一个跨 tile 的顺序状态；shape、参数、临时 descriptors、贡献状态与结果都须计预算。独立结果直接写已规划的 C/F 地址。调度仍由原生 Scheduler 管理，首轮单线程共享/融合优先，后续并行只有完整负载实测收益才采用。

已实施的扫描融合按同一个 weights 节点身份为贡献选择唯一 period carrier，共享已经舍入的权重左乘产品。完整且连续的 asset→贡献映射使用直接状态访问，运行时再次核对实际资产数；部分选择、重复资产或不连续映射继续使用稀疏链接，无匹配者独立扫描。不同输入即使内容或地址相同也不合并；每个标量仍保持跨 tile 的 step→path 单一左折叠。所有持久链接的实际 capacity 同时进入 Planner 和审计，没有新增产品矩阵。

Eigen producer 仅在库开启、最近舍入、非空 row、F>0、A≥8、F×A≤16,384 时执行；只跨独立资产并行，因子仍升序，乘积先具体求值再相加，指数仍调用标量 `std::expm1`。asset stride=1 的系数直接借用，否则每次执行打包一次，prepared 内容变化仍重新读取。几何阶段不能从虚拟 transpose 的 stride 推定实际连续，因此保守预留 F×A doubles，最多128 KiB；系数放在最大工作区尾部，不与不同区间的 tile/贡献状态重叠。实际打包字节进入 `algorithm_copy_bytes`，`vector_elements` 记录真正执行的 Eigen Array lanes，不能当作硬件指令计数。空轴、超上限、窄资产、非最近舍入及禁库保持标量职责；没有新增算子、线程池、输出转换或传输协议。

失败输入的未知几何以独立标记传播，不把容量占位的空向量当真实 rank。affine 及后续数组原语继续校验已知的独立结构约束；未知 iterate seed 不伪造形状错误。strict 和 isolate 的非有限值政策分别保留：isolate 对 typed 非有限元素记录状态 4，独立健康根仍可返回。

`T*P=2,000,000,A=32` 的完整 L/R 各为 512 MB；它们是普通展开图会新增的物化，原平台 NJIT 没有分配这两块。不能把取消它们称为比原 NJIT 节省 1 GB。全部 period 若公开返回仍需 `8*C*T*P` 字节，两份保留结果也要计入；后续 NAV/fan 流水才能按平台实际汇总输出进一步降低大额内存。

## 4. Excel 的并行数学实现

仅下载/核验请求触发原生 Excel 规划；普通 C++ 不运行 Excel 检查、展开或写出。

- affine 每输出写一个初态引用和每个因子的独立乘积、累加步骤，上界 `O+2*O*F`；保留真实引用，不用一个 MMULT 隐藏成本。导出专属域检查核对内部乘积与部分和，拒绝 Excel 不可表示的 Inf、次正规或超范围中间值，包括最终 NaN 掩盖内部溢出的情况。
- expm1 写显式有保护的 12 阶 Horner 辅助链，`2^-54<abs(x)<=1/8` 使用稳定级数；更小的 x 返回原值，远离零使用 EXP(x)−1，x<=−40 返回 −1 避免无关下溢。每元素最多 13 公式。级数在该近零域的解析余项小于 `4e-22`，该界不含浮点舍入，也不能替代桌面 Excel 重算。
- NaN 保留独立值类别；不支持的特殊值/范围明确拒绝，Excel 不保真负零位模式。预检覆盖完整工作簿、最大形状及公式成本，使用受检整数，默认 200 万单元格为项目预算。
- 原生执行、公式生成、目标 Excel 重算分别验收。新 recipe 在真实重算通过前保持 `NOT_RECALCULATED`；原 185 算子的历史 Excel PASS 不证明新两项已经重算。

## 5. 验证与完整目标边界

### V99 组合权重四路保序计算：实际 owned MC 复验（2026-10-10）

2026-10-10，仅 `cpp/compact_path.cpp` 与 `tests/test_compact_weight_lanes.py` 的 V99 变更已进入实际源码；本机实际 AOT 构建身份为 `f789954a…`。实际普通相关 Python **1,049**、全量 **17,179** 与 native **9/9** 无失败或跳过（相关不与全量相加）；普通 owned MC 的 **42/42** fresh 任务及其原始证据经独立复核。实际这一轮是普通非 sanitizer（Normal）构建，不是 actual SAN；隔离候选的相关 SAN/OFF 证据保留各自身份与范围。

已依赖的 Eigen 固定四路 Array 仅并行四个独立组合；每组仍按 asset 升序累加，weights 左乘的产品先具体求值，再分别更新 period 和 contribution。只在 Eigen packet 可用、最近舍入、完整唯一的连续资产贡献映射及受检几何下准入；负零、非有限或非零次正规权重、不满足完整唯一贡献映射、非最近舍入和组合数不是四的倍数的批次继续已有 canonical 计算。输入及旧结果不作 scratch，每次重读当前权重；pack 和最多128 KiB的保守 reserve 进入 Planner/审计，不新增数值缓存、线程池、公开算子或 expm1 库替换。原190注册、AST/DAG、IEEE/错误顺序、结果所有权与 Excel 数学保持。

双方同 CPU1，计入相同固定生产 NJIT RNG 的独占输入准备、normal/Student 变换、投影、普通 `AdaptiveScheduler.execute`、NAV/fan、全部统计/样本/贡献及两个保留的独立结果。静态夹具/构图和编译在计时外；covariance-root 生成、coverage、limits、JSON 封装和平台按业务阶段选首错不在该测量中。算法复制与 root copy 如实计入，不能称全链 zero-copy。

Normal/Student 的 T200/P10000/F4/A32/C8 完整链分别为 **785.024/923.810 ms**；保守 NI3 为 **3187.812/3329.174 ms**，即 **4.06/3.60×**，相对较快旧 C++ 分别省 **46.22%/42.30%**。全部六组时间、cold/warm heap、RSS baseline/total/delta 和具体身份见[V99 证据](physical-execution-evidence/2026-10-10-mc-weight-lanes-v99/README.md)。

Normal C8/C32 的 NJIT 控制轮次漂移仍保留且原因未确立；时间取三个阶段中最快的 NJIT median（NI3），不用异常慢轮次扩大优势。C32 的 warm tracked heap 仍比 NJIT 多 **4,679,434 / 7,007,433 B**（Normal/Student）；Student C1 只比保守 NJIT 快 **1.066×**，比较快旧 C++ 慢约0.30%，不说所有负载都显著改善。LongK 的历史退化与完整 QP/前沿/探索等主要链仍开放，不据此关闭整体性能或平台业务资格。

全部冻结调用均为普通 `scheduler.execute`，没有显式 prepared execution。Normal C1 RSS 的 post-epoch validation audit 为 `prepared_cached=true`，表示普通 execute 的输入绑定缓存命中；其余41份该验证 audit 为 false。验证 audit 不覆盖每次计时/缓存调用，不能称所有普通调用均无缓存。

### 历史 Eigen producer 阶段复验（2026-10-03）

生产构建 `5dcec3922588f4f912a2e6afa55775ac83865771e6e6e2363f46f941e3c1aa97`，源码、wheel、扩展和worker身份已复核。原suite全量11,419项通过；采用后联合专项273项（原语193、Excel37、新Eigen43）及新增传输/预算16项通过。原生、ASan/UBSan和三库OFF各9/9，包含新A17原生fixture。全量运行在永久新测试复制前，不能写成新增后11,478项已经完整重跑。新路径包括非NaN逐位/NaN类别、多区间/跨tile、四舍入、系数变更、C/F与带符号stride；A17专项证明thread/inline process/shared process确实命中Eigen及544B打包，精确总预算−1拒绝、恰好准入，保留两代独立结果并在close/源释放后可读。其他平台及worker定向舍入尚未运行验收。

正式生产wheel、CPU=1、强共享热态NJIT、六轮换序、相同全部period/贡献与两代独立结果：

| 负载 | C++ ms | NJIT ms | C++/NJIT |
| --- | ---: | ---: | ---: |
| N=200万、T=200、P=10000、F=4、A=32，单组合 | 205.029 | 286.081 | 0.717 |
| 同形状，8组合 | 366.426 | 656.839 | 0.558 |
| 同形状，F根 | 362.159 | 659.931 | 0.549 |
| 同形状，prepared独立snapshot | 369.955 | 659.138 | 0.561 |
| N=25万、T=5、P=50000、F=32、A=32、8组合 | 65.047 | 145.928 | 0.446 |
| N=252、T=1、P=252、F=4、A=32，单组合 | 0.067729 | 0.038250 | 1.771 |

大型单组合的旧13%落后已有本轮独立复证：当前快约28%；多组合快约44%，宽因子快约55%。新小用例增加约29微秒，只作诊断，不否决大负载收益。镜像筛选与生产的build ID相同，但wheel/native/worker二进制SHA不同，两批原始结果分别保存，不混用。

八个fresh内存子进程：普通/ prepared C++热申请峰258.763/258.763MB，service/shared NJIT均256.014MB；普通C++/共享NJIT的RSS执行增量258.032/256.246MB。两代完整输出约256MB是同一合同的必要结果，本轮**仍无显著内存优势**，不把1KiB打包增量当作独立门禁，也不据此宣称峰值没有增加。heap/RSS测量范围、未追踪释放与allocator盲区保留。下一步应减少真实compact消费者未返回的period/NAV，而非反复微调这点打包空间。

实现、数学证明、正式包计时/内存、静态ISA及全部终态证据见[本轮证据](physical-execution-evidence/2026-10-03-mc-eigen-producer/README.md)。仅物理优化，原逻辑DAG和Excel公式不变；前轮新增Microsoft Excel重算欠项仍保留。未提交或推送，完整MC/平台迁移及整体目标未完成。

### 前一保序扫描阶段（历史构建63a1d）

2026-10-03，本机 macOS arm64、构建 `63a1dacd485576f5d148be0fb577bf2d277c91a00ab35c970e6bc88d2330a39a`：Python 全量 **11,419** 项；新增原语/物理闭包专项 **193** 项、Excel 预算/域专项 **37** 项；原生、ASan/UBSan及成熟库关闭原生各 **9/9**；187 个算子直接运行覆盖与 scalar/auto 回放全部通过。覆盖 direct/AST、多根、ordinary/prepared、C/F/strided、空维、初态消去、近零/负零/overflow、独立结构校验、输入内容变化、独立结果及进程传输，不表示全部可能组合穷举。

单 CPU、热态 NJIT、六轮换序，相同完整 period/贡献输出与独立结果寿命：

| 负载 | C++ | NJIT | 结论 |
| --- | --- | --- | --- |
| 200万 path-period、4因子、32资产、8组合，原服务型计算 | 481.828 ms | 2073.790 ms | 约快 4.3 倍 |
| 同形状，共享投影后的强 NJIT | 481.037 ms | 658.758 ms | 快约 27%，没有把 NJIT 共享能力遗漏 |
| 同形状，F 根 / prepared | 481.611 / 484.050 ms | 658.041 / 658.426 ms | 仍快约 27%；未强制转 C |
| 同形状，单组合 | 325.049 ms | 286.753 ms | **仍慢约 13%**，未闭环 |
| 20万 path-period、16资产、4组合 | 21.141 ms | 22.920 ms | 约 1.8 ms 差异，仅诊断 |
| 252点、单资产、单组合 | 0.007229 ms | 0.004438 ms | 约 2.8 μs 增加，仅诊断 |

八个 fresh 内存子进程，保留两代完整输出（约256 MB）：普通 C++ 热申请峰 **258.762 MB**，NJIT **256.014 MB**；RSS 执行增量约 **257.999 / 256.360 MB**。当前**没有显著内存优势**。C++ 几何/图和有界 tile 成本如实保留，不通过削减输出改善数字。heap 含 setup，RSS 的 setup 在 baseline 前；malloc 未追踪的释放和未覆盖分配器、页采样盲区均保留，不宣称完整分配审计通过。审计显示数值 arena、输入复制和根复制均为0，实际直接输出128,002,048字节，但这些字段不等于全请求内存。

Excel：原185算子的469组/3,056值与历史 Microsoft Excel 重算快照等价，只有允许的身份字段变化；这不是新桌面重算。新增10组/48值在 LibreOffice 25.8.4.2 实际重算通过。Microsoft Excel 新增公式仍因 Mac 锁屏未完成，保持 **NOT_RECALCULATED**。完整设计、失败及最终证据、自审核见[证据目录](physical-execution-evidence/2026-10-03-mc-projection-integration/README.md)。仅本地修改，未提交或推送。

固定已变换随机带的投影子链不验证旧 seed/RNG、normal/chi-square 消费顺序或 covariance 的 LAPACK 根。任意初态 horizon 递推、NAV/fan 轴统计、首故障 prefix 诊断和平台实际服务接入仍待完成。整体目标保持为全部平台计算与中大型完整负载的明显时间、内存优势。

新增完整固定带参考链剖析：同样 N=200万、T=200、P=10000、F=4、A=32、8组合，按平台顺序逐组合执行投影→NAV→fan，仅返回 fan、14项统计、前12条样本路径及贡献，三次热执行中位 **3.108秒**；投影约 **2.091秒**、NAV约 **0.085秒**、fan约 **0.931秒**。这是一份自包含、冻结上游数学体的 NJIT 定位证据，排除 coverage、随机生成、协方差根、factor transform、limits 和 JSON，不是 C++ 完整链通过或生产服务耗时。两次独立 compact 结果保留，输入hash不变。两个 fresh 内存进程的 tracked warm heap峰 **48.903 MB**、RSS执行增量 **65.798 MB**；共同输入 **64.003 MB**在epoch外，heap未追踪释放和RSS盲区继续保留。这个输出契约与上表全部period根的256MB两代实验不同，不能直接相除得出优势。

2026-10-03 的下一阶段设计曾将 `cumulative_product_axis`、`cumulative_max_axis`、`quantiles_axis` 标为尚未实现；这是当时187目录的历史状态。三项现已成为188–190，实际可编辑 NAV/fan/统计/样本/贡献及其私有有界闭包见[按轴原语](mc-axis-primitives.md)。[当时设计](physical-execution-evidence/2026-10-03-mc-eigen-producer/compact-chain-implementation-next.md)继续保留原日期与证据，不当作当前未实现说明。V99 的普通 owned 链通过不覆盖 covariance-root 生成、coverage、limits/JSON、平台首故障顺序或新增 Microsoft Excel 实机重算；这些与完整平台迁移仍需各自验收。已有 mean 的 Welford、quantile 的整数rank插值及有界 Jacobi 不能无声替代平台原数学契约。年度moments不属于当前 scenario_stress MC 的实际调用链。
