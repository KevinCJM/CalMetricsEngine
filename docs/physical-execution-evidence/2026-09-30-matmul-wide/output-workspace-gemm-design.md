# Eigen GEMM 借用本次输出尾区：静态可行性

日期：2026-09-30。仅检查本机固定 Eigen 3.4.0 源码与已冻结构建配置；未编译、未运行数值或性能、未修改仓库及冻结原型。本文是后续候选设计，不是采用或验收结果。

## 结论

**内存布局可以成立，直接套用默认 GEMM 的数值契约不成立。** Eigen 内部顺序入口允许借入两块 packing 空间；它们可以位于本次独立输出尚未计算的尾行。公开 `noalias()` 没有这个工作区接口，不能据此声称零 packing。默认 NEON GEMM 显式 FMA、K 分块及部分尾核分组归约，不因 `-ffp-contract=off` 自动变成原顺序。

父任务提出的变体有进一步研究价值：只使用完整 6×4 double tile、一次覆盖整个 K、替换 packet 的乘加，并把所有尾部留给现有内核。**仅覆盖 `madd` 仍不够**：末尾 `acc` 仍融合、stock kernel 会读取旧输出，且 `gebp_kernel` 没有可传入自定义 Traits 的公共参数。下文说明可研究的内部适配边界；没有据此增加全域 bitwise 要求或放宽现有容差。

## 固定源码与范围

Eigen 根目录：`/private/tmp/calmetrics-matmul-build/_deps/eigen-3.4.0/Eigen/src/Core/`。下文的文件、行号均相对此目录。

基线为当前正式 `1aec1a6b0d4a07f0caee43b58cd4b339bdcc165f5e615f2190b15da0ba66e86b`。实际发布配置为 arm64、C++17、O3、`-fno-fast-math -ffp-contract=off`；本设计不改 Eigen 宏、不新增线程池、不把数值空间移到 TLS/栈，也不改 AST/DAG、结果协议或 Excel 数学展开。

原 MatMul 的普通浮点验收仍是既有逐项容差；IEEE 分类、已存在的图内比较/除零等下游另验。保序候选须证明其自己的保序主张，但不能把本候选的 bitwise 目标反向变成全部 MatMul 的新永久契约。

## 1. 可借入空间的真实入口

- `products/GeneralMatrixMatrix.h:61–82`：内部 `general_matrix_matrix_product::run` 接受 `level3_blocking&`，创建成熟的 `gemm_pack_lhs`、`gemm_pack_rhs`、`gebp_kernel`。
- `:162–167`：顺序入口请求 `sizeA=kc*mc`、`sizeB=kc*nc` 个 scalar，并把 `blocking.blockA/B()` 传给临时空间宏。
- `util/Memory.h:751–757,768–771`：BUFFER 非空时直接借用；空指针才会走 alloca/aligned_malloc。借入指针不会交给临时空间清理器释放。
- `products/GeneralMatrixMatrix.h:250–276`：`level3_blocking` 的两个指针与 mc/nc/kc 是 protected，允许一个小的无所有权派生类初始化它们。
- **不能派生/填充 owning `gemm_blocking_space` 来冒充借用**：`:398–401` 析构无条件 aligned_delete 两块指针。固定尺寸 owning 版本也只会把数组搬到栈，仍是新增空间。
- 公开 Product 路径 `:494–509` 内部创建 BlockingType，再进入 parallelize_gemm；没有公开 external-workspace 参数。本设计属于固定版本 `Eigen::internal` 适配，不是稳定公共扩展 API。

必须提供两个非空地址、检查正维度和乘法溢出，并使用顺序 `info=nullptr` 路径；不调用公开 parallelize_gemm。关闭 Eigen 并行仍需在实际对象和完整调用链验证，不以源码意图替代证据。

## 2. 输出尾区、转置 mapper 与容量证明

设输入 A 为 M×K、B 为 K×N，本次独立输出为 C 连续 M×N。取 h 个前缀行，r=M−h 个尾行：

- 最终前缀目标：`[out, out+h*N)`。
- 临时 packing 租约：`[out+h*N, out+M*N)`；所有元素跨度、字节乘法与对齐加法均先做溢出检查。
- 两块真实打包容量：`8*kc*mc`、`8*kc*nc` 字节；另计每块起始地址的实际对齐损耗，保守最多 `2*(alignment−1)`。不能默认所有 NumPy/显式 out 起点都满足 Eigen 的 packet 对齐。
- 不采用 PanelMode/非零 pack offset；若以后改它们，容量公式须重新证明，不能沿用上述普通 pack 模式的预算。

**RowMajor 输出会转置。** `GeneralMatrixMatrix.h:19–44` 交换输入并调用 `run(cols,rows,...)`。因此前缀 h×N 的内部维度是 rows=N、cols=h，`mc<=N`、`nc<=h`。内部 ColMajor mapper 的 outer stride 仍为原 N，目标地址为 `out + internal_row + internal_col*N`，最大不超过 `h*N−1`。不能把内部 mc/nc 误当成原 M/N 后安排 scratch。

顺序执行：完成所有几何/别名/准入检查；把两段 pack 放在尾区；仅对前缀执行内部 GEMM；所有前缀 tile 完成后结束租约；最后用现有无 scratch 内核补全部尾行并覆盖暂存数据。前缀也可另有右边界列留给原内核，但它必须与 pack 尾区及已完成 tile 分开。完成前不得向使用者或并行消费者发布部分结果。

stock GEMM 是 `C += A*B`，普通使用必须先初始化前缀，不能读未初始化输出。绝不能一边用尾区 packing 一边让 full-M GEMM 写这些尾行，也不能在尚有 tile 使用 blockA/B 时提前补尾。

**空间例子只是设计算术，非实测：** kc=128、mc=96、nc=128，需要 229,376 B 加对齐余量。N=1024 时按完整行保留 29 行足够覆盖该保守对齐预算，N=512 时为 57 行。实际选择要按 h 和真实内部布局重新验证；不能把旧默认 GEMM 的历史 packing 数字当成新候选固定需求。

## 3. 输入 stride、别名与审计

`util/BlasUtil.h:164–194,387–395` 的默认 const mapper 只有 outer stride，inner stride 为 1。它可直接承接 C/F 兼容输入；**不能直接涵盖现有任意二维 signed/zero stride A**。外层准入不满足就走当前实现。自定义输入 mapper 再喂 pack 是另一个可研究的小适配，但必须检查成熟 pack 所调用的全部 scalar/packet/linear mapper 接口，不能只实现 `operator()` 后宣称支持一般 stride。

A/B 允许共享只读存储；两者不能与本次可写 output 重叠。两段 pack 也不能互相重叠、与任何前缀输出 tile 重叠。遵守现有显式 out 检查、结果独立生命周期、Workspace 锁及 worker ownership；不借上一轮结果或输入做 pack。

没有新增容量不代表无成本：packing 复制字节、尾区多次写入、最后补算、初始化、内核及全部活跃调用链栈都计入请求。需要记录实际 pack 调用次数，不能只累计一份容量。若沿用 Eigen 预取，应额外检查其可能越过 active panel 的地址表达式；成熟源码对完整 tile 的普通 load/store 有界，不等于任意自定义子缓冲的预取地址也自动落在 allocation 内。需要的保护间距只能从同一尾区扣除，不能暗增 allocation。

异常前完成独立结构检查；若调用失败，遵守既有部分写出与错误契约，不能发布含 packing 数据的成功结果。静态方案不证明 RSS、首次库页、栈或 wall time 非增；这些仍需正式旧/新同口径验收。

## 4. 默认内核的数值阻碍

| 位置 | 实际行为 | 对当前思路的影响 |
| --- | --- | --- |
| `arch/NEON/GeneralBlockPanelKernel.h:109–176` | double NEON traits 显式 `vfmaq_n_f64` / `vfmaq_laneq_f64` | 编译器 contraction=off 不会拆这些 intrinsic。 |
| `arch/NEON/PacketMath.h:3762–3768` | double packet `pmadd` 显式 fused intrinsic | 改用普通 pmadd 仍然融合。 |
| `products/GeneralMatrixMatrix.h:176–198` | 每个 kc panel 从零求和，再加到 C | `kc<K` 改变单个输出的归约括号。 |
| `products/GeneralBlockPanelKernel.h:1230–1287` | 一 packet×4 尾核把偶数 k、奇数 k 分开，最后 C+D | 即使 kc=K、madd 非融合，该尾核仍不保原顺序。 |
| `:533–541,1552–1593` | 末尾 `acc` 用 pmadd，且先读旧输出 | 仅改 madd 没有处理最后一步。 |

一个可嵌入任意大 K 的解析反例：先累计 `−2^54`，再累计 `(2^27+1)*(2^27−1)`。分离 binary64 乘法将第二项舍入为 `2^54`，原顺序得到 0；融合乘加得到 −1。它超过原绝对容差，也能改变比较/除零。这里只给数学构造，没有在本轮执行。对非有限、极值、子正常、零系数的行为同样不能只靠随机 allclose。

本项目允许通过既有契约验证的浮点归约变化；上述证据说明默认 full GEMM 不能**无条件**承接全部已支持输入，并不说明一切重排或 FMA 都被制度禁止。有关终端浮点准入、误差保护和现有失败样本，参见同目录 `library-admission-review.md`，不在此重复建立另一套契约。

## 5. 追加变体：保留 gebp 主体，非融合 traits、完整 K、只用完整 tile

### 5.1 主 tile 的 k 顺序可以隔离出来

当前 arm64 double packet 为 2 个元素；`GeneralBlockPanelKernel.h:429–453` 的 `mr=6,nr=4`。完整 3packet×4 分支 `:1455–1461` 为每个独立输出 lane 初始化一个累加器；`:1493–1532` 对每个 lane 依次执行 k=0..7，`:1539–1547` 按顺序补 K 的尾项，没有偶/奇第二累加器。内部 `actual_panel_rows` 也是 6 的倍数（`:1440–1446`）。

因此下面的**组合**具有保序研究路径：kc=整个 K；每次 gebp 的 rows 是 6 的倍数、cols 是 4 的倍数；madd 明确执行 `padd(c, pmul(a,b))`；编译继续禁 contraction/reassociation；其他所有形状尾块由现有原序核处理。不能只改 `mr` 数字并继续走其余尾 kernel。

若使用 RowMajor 转置映射，这些限制反映到原输出为：bulk 列数是 6 的倍数，bulk 前缀行数是 4 的倍数。mc/nc 的各次实际块也必须满足对应倍数，最后一次不能意外落入 1packet/half/quarter 分支。必要的右侧尾列与底部尾行都单独补算。

### 5.2 Traits 不是公共可插拔参数

`gebp_kernel` 的模板 `:1056–1072` 没有 Traits 参数；内部根据 scalar、ConjLhs/ConjRhs、`Architecture::Target` 重新选 traits。现存 `double,double,false,false,NEON` 已明确特化，不能在另一个头/TU 再定义一个不同版本，否则重定义/ODR 出错。单纯继承它再把对象传进去不可行。

可能的最小内部研究入口是一个**尚未占用、未先实例化的真实模板组合**，例如实数 double 的 conjugation 组合，再显式特化该组合并直接实例化对应 kernel。实数共轭本身不改变数值；pack 的实数 conjugation 也是 no-op（`:2112`）。但该做法仍需证明 include/首次实例化顺序、全程序 ODR、Half/Quarter/SwappedTraits 的一致性和所选 pack 格式，不能把布尔位随意当通用标签，更不能影响其他 Eigen 调用。另一选择是明确的最小版本绑定适配/上游扩展，而不是悄悄复制整个 GEMM 数学主体。当前仅识别路径，没有证明任一特化已经可构建合法。

### 5.3 只覆盖 madd 不够，“只写一次”还需要 assignment mapper

stock 6×4 主体在 `:1555–1593` **先 load 输出，再 `traits.acc`，然后 store**。原 acc 为 `pmadd(c,alpha,r)`。在 alpha=1、输出预零的模式下，至少多一次初始化写入，并多一个最后算术操作；不得自动声称和原 +0 起点的所有零/NaN 细节完全相同。

若目标是前缀每元素仅做一次最终写出：需要 assignment-only `acc`（在明确 alpha=1 且唯一 K panel 的私有入口里直接 `r=c`），以及一个特殊输出 mapper，让 kernel 要求的 loadPacket 返回定义好的零而**不读取未初始化实际输出**。storePacket 仍写真实独立输出；prefetch 与 sub/linear mapper 的地址范围必须合法。只依赖优化器删掉未初始化读取不构成合法源码方案。由于 `general_matrix_matrix_product::run` 在 `:71` 硬编码 ResMapper，使用自定义 mapper 时应直接用成熟 pack + gebp 做小型外层驱动，不能继续声称原 run 入口原封不动支持它。

这种 mapper 仅用于完整 tile；不支持继续 `C+=`、alpha 任意值、多 K panel 或复用同一目标的第二次调用。若它们发生就是契约错误，不能静默丢掉旧 C。所谓一次写出也仅指 bulk 的最终结果 store；借作 packing 的尾区显然会多次写，完整复制/写流量仍需计量。

### 5.4 kc=K 的真实代价

空间变为 `8*K*(mc+nc)` 加对齐/必要保护间距。完整 6×4 最小组合也需 `80*K` B；M=N=64、K=32768 时，仅最小 pack 就需 2,621,440 B，而整个输出只有 32,768 B，绝无可借空间。这类输入必须在写出前回退，不能额外分配或借 TLS。

1024³、mc=96、nc=128 的示例需要 1,835,008 B，保守行预算为 225 行；这个尾部比例和原序补算可能抵消吞吐收益。缩小 mc/nc 可减少空间，但降低数据复用。更改独立 i/j tile 遍历合法，拆 K 来降空间却会破坏本变体要证明的单次归约，不能两者同时口头承诺。

### 5.5 最小下一步判据

仅在明确上述 traits/mapper 是合法内部适配之后，才值得建立一个隔离原型：完整 tile 主路径 + 现有 tails、不改变库主体；覆盖实数乘法操作数交换的 NaN/零分类、强消去、溢出、0×Inf、K=1/7/8/9/长 K、6/4 block 边界、对齐及尾区 canary、只读别名、输出未初始化/先填 NaN、下游 bool/error。

随后必须查真实对象是否仍有 FMA、是否进入被排除的分组尾核、完整调用链 stack/寄存器 spill/alloc、pack 数量及数值；最后才看包含 pack、尾部和独立结果所有权的正式性能/内存。这不是本轮采用决定。本次没有编译或运行这条路径，不能把它写成已满足时间、内存或全域数学契约。
