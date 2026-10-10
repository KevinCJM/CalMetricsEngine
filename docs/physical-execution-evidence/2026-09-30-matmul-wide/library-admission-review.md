# MatMul 成熟库受控准入复核

日期：2026-09-30。只读设计研究；本审核者未构建、未执行数值、未计时，未修改仓库。父任务随后提供的新入口数值筛选已只读核对并列在下文；没有新增生产采用或性能验收结论。

## 结论与建议次序

1. **MatMul 没有全域逐位一致契约。**现有普通浮点验收是逐项容差，IEEE/错误及实际离散下游另验。纯浮点终端输出采用改变归约顺序的成熟库，在既有容差及边界内可以是合法物理分派；不能因为外部调用者“可能比较任意阈值”而新增全域 bitwise 要求。
2. **新版入口差异已核实，换入口没有修复边界。**上一轮原型引用 `_cblas_dgemm`，当前 NJIT 经 SciPy 引用 `_dgemm$NEWLAPACK`。父任务已补新版 LP64 筛选，132 个受支持样本仍有同样 7 个失败。因此继续方向必须是受控数值准入，不能无条件替换。新版辅助堆测试也已完成：1024 CC/CF 冷暖峰值均为 6,291,464 B，这是本轮独立证据，不是套用旧入口数字。
3. **严格 C 连续的 `vDSP_mmulD` 也需要数值与内存准入。**父任务补测 79 个受支持样本，仍有 3 个取消/溢出失败。官方 API 能直接写本次独立输出，但本轮 1024 CC 冷暖堆峰值同样为 6,291,464 B；不能假设无工作区。线程及完整栈仍待单独核验，不能假设服从 BLAS TLS。F 布局不应为此隐式复制成 C 布局。
4. **若完整库仍有额外 packing，可研究固定版本 Eigen 内核 + 本次输出未计算尾区的临时租借。**这有实在的无新增数值堆空间实现路径，并非把成本藏入栈。需要一个小的内部 API 适配器，不能称为公共稳定 Eigen API。时间、复制字节和最后尾行补算必须完整计入。
5. 当前证据**不能承诺同核数单次 GEMM 明显胜过 NJIT**：NJIT 本来就在调用 Accelerate。大幅领先更可能来自避免一般 stride 的输入复制、相同广播行的重复计算、以及合法的图级中间结果消除；不能用这一判断降低单算子的时间/内存目标。

## 依据与当前身份

- 本轮起点：`start-manifest.json`；当前基线构建 `1aec1a6b0d4a07f0caee43b58cd4b339bdcc165f5e615f2190b15da0ba66e86b`。
- 已先用 CodeGraph 定位 `matmul/library_matmul_f/Workspace/root output`，再补读未覆盖的当前源文件。
- AGENTS、数学纪律、Excel 纪律与起点清单 SHA 一致；适用 `docs/computation-design-rules.md` 第 2、2.1、D1、D2、D5、D8，Excel X1–X3/X6–X10。纯物理变化可保留原 Excel 数学步骤，不能把历史 Excel 快照写成新重算。
- 旧筛选：`/private/tmp/calmetrics-matmul-libraries/{correctness-screen.json,run_screen.py,screen_fixtures.py,blas_probe.cpp,blas-heap.log}`。
- 旧契约审核 `contract-adoption-review.md` 的“尚无 MatMul 比较图回归”已过时：当前 `tests/test_matmul_library.py` 已覆盖 FMA、分裂归约、CSE、比较、除零错误及四个执行通道。本报告按现状说明。

## 真实数值契约，而非自行加严

`docs/operator-reference.md` 定义 MatMul 为矩阵收缩、内维一致、没有高维广播，并未声明固定浮点位模式。`tests/test_operator_contracts.py:356–380` 对 NumPy 用 `rtol=2e-12, atol=2e-14`；新 `tests/test_matmul_library.py` 延续这一容差。上一轮保序原型的 bitwise 统计证明其保序主张，不能倒推成所有未来算法的永久契约。

这不等于只要 allclose 即可：

- 当前非有限/极值夹具分别比较 NaN、正负 Inf、零及特定负零/极小值行为；`0*Inf` 不能因稀疏式跳零而丢失。
- `test_matmul_worker_local_f_rhs_keeps_comparison_and_error_boundary` 用真实 `matmul → greater_than`、`matmul → sum → divide → greater_than` 图验证状态与 bool，且包含共享 MatMul 的 CSE。普通浮点容差不能掩盖这些差异。
- `[-1, 1+2^-27]` 与 `[1, 1-2^-27]` 的乘积归约可因 FMA 得到非零尾数；`[1, 2^-54, -1]` 可因分组归约由 0 变为正尾数。尾数能落在浮点绝对容差内，同时改变除零/比较。这解释了真实图测试的必要性，不是对全部普通浮点结果提出逐位要求。
- 现有 `matrix_backend.cpp:188–201` 的列布局 covariance 已采用 Eigen `.sum()` 重排归约并有极值保护。它说明项目允许经验证的浮点顺序变化；不证明 GEMM 自动满足同样的边界。

**最小图内准入设计：**首版只允许顶层、纯浮点、没有任何图内消费者的 MatMul 根；显式 scalar/SIMD 保持原语义。`root_destinations[node] >= 0` 本身不够：同一根可以继续被比较等节点借用。须检查普通父边、作用域捕获/参数边及外层消费，子作用域的局部根也不能当成全局终端。共享浮点根与离散根必须统一走受保护路径，不能靠 CSE 顺序碰运气。新资格需要真实传至 execute/worker，或由现有序列化图在 finalize 重建，不能仅在 Python 外壳检查。已有 metadata 没有通用“无消费者浮点根”标志，不可声称已完成。

**Direct API 边界：**普通 `op.matmul(..., simd="auto")` 返回浮点矩阵，按既有容差允许库间尾差；无需窥探未知外部代码，也不应承诺其任意精确阈值结果不变。采用时应在执行说明明确 auto 的候选范围与尾差、显式 scalar 的可复现用途。图内离散契约由引擎保护；用户在外部如何解释近阈值浮点数是另一边界。不能为了统一而给比较算子添加 epsilon，也不能新增默认“relaxed”数学模式来绕过失败。

## 上一轮 7 个 BLAS 失败具体说明了什么

168 个夹具中旧 BLAS 支持 132 个；7 个失败如下。

| 夹具 | 原容差 | 非有限分类 | `> 0` |
| --- | --- | --- | --- |
| `cancel_5_3_17_cc` | 失败 | 相同 | 不同 |
| `overflow_5_3_17_cc/cf/ff` | 失败 | 不同 | 不同 |
| `overflow_129_3_33_cc/cf/ff` | 失败 | 不同 | 不同 |

这 7 个不是“仅 bitwise 不同被错杀”；都有原容差失败，6 个还有分类失败。该筛选里 7 个零比较差异与这 7 个失败重叠，不能虚构另外 7 个“容差内但离散不同”结果。后者由当前独立边界图夹具阐明。

失败的 K 都是 3，不构成 `K>=32` 的安全证据：同样乘积序列可补零嵌入 32/64/129/512 的实际块、尾块及巨大矩阵。具体动态分派结果必须实测，补零构造本身只是应该保留的对抗测试。

单次扫描 finite 和最大幅值可以排除一部分中间溢出，不能检测一般消去。对每个输出再扫描全部绝对乘积或直接完整复算原算法，会吃掉库加速；不能把这称为低开销保护。

## 新发现：NJIT 与旧筛选的 Accelerate 入口不同

全部为当前本机静态取证，没有运行数值代码：

1. `tools/check_matmul_performance.py:122–134` 的 NJIT 是实际 readonly/layout 签名的 `a @ b`，compile 后首次数值仍在测量 epoch。
2. 本机 Numba `_lapack.c:11–25,363–400` 从 `scipy.linalg.cython_blas` 的 `dgemm` capsule 获取函数地址，`numba_xxgemm` 以 Fortran BLAS ABI 调用它。`numba/np/linalg.py` 的矩阵乘法降到该入口。
3. `otool -L .../scipy/linalg/{_fblas,cython_blas}.cpython-312-darwin.so` 均显示依赖系统 Accelerate；`nm -u cython_blas...so` 是 `_dgemm$NEWLAPACK`。
4. `nm -u /private/tmp/calmetrics-matmul-libraries/blas_probe.dylib` 是 `_cblas_dgemm`，源码没有 `ACCELERATE_NEW_LAPACK`。本机 SDK 的旧 `cblas.h` 已标记旧接口 deprecated；`cblas_new.h` 使用 `__LAPACK_int` 和 `__LAPACK_ALIAS`。

因此，timing identity 中 NumPy 自带的 OpenBLAS 信息不能代表 NJIT DGEMM。也不能根据速度或链接 Accelerate 推断已使用 AMX；本次没有运行时调用栈或具体后端指令证据。

**已补的最小原型：**父任务隔离 TU 使用 `ACCELERATE_NEW_LAPACK`、LP64，未开 ILP64，已确认导入 `_cblas_dgemm$NEWLAPACK`。新 CBLAS 和与 SciPy 对应的新 Fortran 入口仍是不同调用入口，不能假定内部实现必然相同。若进一步做对应 Fortran 对照，应分别冻结身份；无需为换入口删除旧证据。

官方新接口说明见 [Apple cblas_dgemm](https://developer.apple.com/documentation/accelerate/cblas_dgemm(_:_:_:_:_:_:_:_:_:_:_:_:_:_:))。

**新增证据（父任务执行，本审核者仅解析 JSON）：**

- `blas-new-correctness.json`：168 总数、132 支持，原容差 125/132、分类 126/132、零比较 125/132；失败名称与上表 7 项完全一致。原型 dylib SHA `e93685ba7df8fec3033896065f99490f370810e44877a14e02552605fcb767c2`。
- `vdsp-correctness.json`：168 总数、79 支持，原容差 76/79、分类 77/79、零比较 76/79；失败为 `cancel_5_3_17_cc`、`overflow_5_3_17_cc`、`overflow_129_3_33_cc`。原型 dylib SHA `6983755de5302fb97f5d201b3eb29ac1c5402d1f6df9fb46eb58cb751b214d72`。
- 两者使用本轮 baseline wrapper SHA `8e5ea716e397b7b64b2ceeaf2aefe6d7e3c250dd232ad0577fb3c518ef370e0d`。bitwise 列是诊断：新 BLAS 92/132、vDSP 68/79，不把剩余所有尾差计为失败。
- 父任务辅助冷暖 heap 已完成，`library-heap/heap.json` 的 10/10 个独立 child 结果完整；详见下方本轮辅助堆表。此审核仅解析既有结果，没有重新执行测量。

## 可控线程、输入布局与工作区

### Accelerate

Apple [BLASSetThreading](https://developer.apple.com/documentation/accelerate/blassetthreading(_:)) 明确是线程局部设置；本机 SDK `thread_api.h:21–45` 同样说明。受控模式是在实际执行计算的 Scheduler worker 内保存当前值、设单线程、验证、调用、恢复并验证；所有返回路径都要恢复。设置失败走既有内核；恢复失败不能当作成功请求。不能在主线程设一次就假定工作线程也生效。无需更改进程全局环境或线程数；不允许库再私开超预算并行。

CBLAS C/F leading-dimension 接口允许把符合布局的只读输入直接传入，`beta=0` 直接写本次独立 C 输出；但这只能证明没有额外整块输出副本。标准 [DGEMM 接口](https://www.netlib.org/lapack/explore-html/dd/d09/group__gemm_ga1e899f8453bcbfde78e91a86a2dab984.html) 没有外部 workspace 参数，不能把引擎 arena 指针传进去替代库 packing。

旧入口 `blas-heap.log` 在输入/输出预分配的同一 kernel epoch 下：512 CC 增 65,544 B，512 CF 增 1,048,584 B，1024 CC/CF 增 6,291,464 B；旧 native/Eigen 候选均 0。冷暖相同。它不是完整请求内存验收，但足以否定该旧原型“无额外数值堆空间”。预热到计时外或改成 TLS 常驻缓存不能使这些字节消失。

**本轮新入口辅助堆已完成：**`library-heap/heap.json`，10 个独立 child，每个记录 17 次 kernel 调用；首次实际数值执行位于 cold epoch。结果为被追踪的 requested heap 字节，不是 RSS/footprint 或完整 Python 请求。

| 形状/布局 | 当前 native 冷/暖峰值 | 新 LP64 CBLAS 冷/暖峰值 | vDSP 冷/暖峰值 |
| --- | ---: | ---: | ---: |
| 512 CC | 0 / 0 B | 65,544 / 65,544 B | 65,544 / 65,544 B |
| 512 CF | 0 / 0 B | 1,048,584 / 1,048,584 B | 不支持此布局，未测 |
| 1024 CC | 0 / 0 B | 6,291,464 / 6,291,464 B | 6,291,464 / 6,291,464 B |
| 1024 CF | 0 / 0 B | 6,291,464 / 6,291,464 B | 不支持此布局，未测 |

所有记录 `untracked_frees=0`、`table_overflows=0`；两库各有效 child 在测量阶段结束时仍记录 8 B live，native 为 0。不将这 8 B 未归因常驻量直接称为泄漏/TLS，也不遗漏它。导入、输入及 caller-owned 输出分配均排除在该辅助窗口之外；库自身 packing 位于窗口内。此证据已否定两个当前直接调用候选“额外堆不增加”的说法，因此不能只凭其快而按用户当前条件采用。它没有建立完整线程、系统栈、RSS 或请求内存结论。

### vDSP_mmulD

[Apple vDSP_mmulD](https://developer.apple.com/documentation/accelerate/vdsp_mmuld) 与本机 `vDSP.h:2495–2528` 的映射是 `A[(m*K+k)*IA]`、`B[(k*N+n)*IB]`、`C[(m*N+n)*IC]`。单个 stride 是整个扁平矩阵的元素间距，**不是 BLAS leading dimension，也不能表示一般 F、padded-row 或二维独立 stride**。首版 `IA=IB=IC=1`、严格 C 连续、空维原路径是清晰范围；输出直接用现有独立结果。

API 无外部 workspace，也未在此文档承诺零内部分配、固定归约顺序或全部形状单线程。`BLASSetThreading` 的公开范围是 BLAS/LAPACK，不足以证明 vDSP 服从它。当前临时 `vdsp_probe.cpp` 虽调用该设置，实际线程/预算及系统栈仍须独立证明；已完成的辅助堆与数值筛选分别暴露额外空间和边界失败。无需为研究新增输出转置副本。

### Eigen 公共与内部接口

[Eigen noalias](https://eigen.tuxfamily.org/dox/group__TopicAliasing.html) 能去掉别名结果临时，不承诺去掉 packing。固定 3.4.0 `GeneralMatrixMatrix.h:169–203,286–301,365–392` 有 A/B 打包；固定尺寸也可能把数值缓冲移到栈，仍应计入内存。

当前版本 `general_matrix_matrix_product::run` 接受 `level3_blocking&`，它具有受保护的 blockA/blockB 与 mc/nc/kc。给出非空 caller-owned buffers 后，顺序实现的 `ei_declare_aligned_stack_constructed_variable` 使用这些地址。这允许小型版本绑定适配器复用库 pack/kernel，**不需抄写 GEMM 数学内核**。必须薄派生不拥有内存的 `level3_blocking`，不能把输出地址塞进 `gemm_blocking_space` 后让其 `aligned_delete` 析构；普通输入 mapper 只有 outer stride、inner stride=1，因此不能直接承接当前任意 A stride。它位于 `Eigen::internal`，不是稳定公共 API；必须固定版本、静态/消毒测试及升级检查，并禁用内部 OpenMP 路径。默认内部 kernel 仍可能 FMA、分块及尾部重排；不能宣称禁 compiler contraction 即保序。

## 既有内存可以复用到什么程度

| 现有内存 | 可否合法作为 packing | 原因/限制 |
| --- | --- | --- |
| A/B 输入、共享输入、前次返回结果 | 不可 | 只读、可能别名、独立结果生命周期不能改变。 |
| 当前图中已死的独占 numeric slot | 有条件 | 要由实际 liveness 证明无当前输入/借用视图/捕获/其他根仍消费；容量和对齐真实足够，租约仅覆盖同步 kernel 调用。当前没有通用租借实现。 |
| 同一次执行已分配、当前不活跃的 `Workspace.doubles` | 有条件 | `Workspace` 独占锁与容量可用，但 root-only MatMul 的容量为零。不能靠上次较大图残留容量；`Scratch::ensure` 会按图/窗口收缩清理。 |
| 当前尚未公开输出的未计算尾行 | **可研究** | 本次写独占且不与 A/B 重叠；完成前最终覆盖为正确结果。必须证明消费及异常边界，不能提前发布部分结果。 |
| 额外新建“备用” arena、TLS 缓存、栈数组 | 不是免费复用 | 都增加实际内存；预分配只是更换时间或存储位置。 |

源码依据：`graph.cpp:103–145` 保证 workspace/arena 不可随旧图无限保留；`:773–799` 根直接输出不分配重复 arena；`:2190–2228` 根属于本次结果、下游借用；`operator_bindings.cpp:211–228` 显式 out 要可写 C 连续且不与输入重叠；`operators.hpp:274–285` workspace 容量属于真实预算。故单个 MatMul 不能声称“现成有一块同尺寸 scratch”。

**输出尾区租借的具体候选：**

1. 所有几何、平台、数值准入与整数溢出检查在写出前完成；取得本次 C 输出，选定固定末尾 `r` 行作为临时区。
2. 对齐后的尾区分给 Eigen blockA/blockB。尺寸按真实 `kc*(mc+nc)*sizeof(double)` 加对齐余量计算，不能超出 `r*N*sizeof(double)`。首版只用于足够高/宽且 `r/M` 很小的形状。
3. 用 Eigen 内部顺序 GEMM 写 `[0,M-r)` 行，tail scratch 与任何目标 tile 不重叠。输出主区按库累加要求初始化，保持 K 分块正确；行主序内部会转置/交换 operands，其实际 rows=N、cols=M-r，目标范围仍应落在输出前 `(M-r)*N` 元素，按真实 mapper/stride 验证。两个 packing 区各自的对齐损耗均须计入。
4. 库返回、packing 租约结束后，用已有无 scratch MatMul 路径计算最后 r 行，覆盖临时内容；此后才发布/观察结果。全矩阵每个请求仍独立分配，旧结果完全不变。
5. 内部 packing 仍是真实复制，须累计 `algorithm_copy_bytes` 并计时；新适配器实际帧、库活跃栈、冷库映射和独立两结果 footprint 也必须复测。不能仅据“动态堆 0”宣称总内存不增。

该办法适用于可提供 caller buffers 的 Eigen 接口，**不适用于没有 workspace 参数的标准 Accelerate DGEMM**。小输出/长 K 可能容不下有效 packing 或尾行成本过高，直接回退。若显式 out 的失败写入范围、observer 或共享执行存在更强中间可见约束，必须先验证再准入；不强改所有权来迁就候选。

## 数值准入怎样避免每次完整复算

建议先采用窄且可解释的组合，而不是开发完整区间解释器：

- 结构资格：仅上述终端浮点根/direct auto；形状/布局可零输入复制；空、显式 ISA 和不支持布局走当前实现。
- 一遍 O(MK+KN) 只读扫描：检查 finite、极值以及可安全计算的量级上界。计算上界本身不得先溢出；用指数/分段比较。任何可能中间溢出、严重下溢、特殊值输入先回退。这处理 6 个 overflow 失败，但不能独自处理 cancel。
- 对易准入的数据域：例如全部乘积同号且非零项正常、和有安全量级余量，再验证该 K 下已核对误差模型的相对上界满足原容差；“没有消去”本身仍不等于任意 K 准入。另一子域是有界整数/二进制定点使所有乘积及部分和精确可表示，可以用一次扫描给出足够条件（如安全计算的 `K*max_abs(A)*max_abs(B)<=2^53` 对整数绝对和的充分限制，并核验库的实际累加模型）。精确域还需验证初始 +0、零结果及库分类；别把 dtype=float64 当成整数域证明。这些域只是候选子集，不能把普通混合符号金融矩阵排除后宣称完成全目标。
- 对一般混合符号：可在库直接输出后，使用保守的每格误差上界选择性原序重算，复用同一输出，无第二矩阵。若有可靠 `|candidate-canonical|<=D`，则 `D*(1+rtol)<=atol+rtol*abs(candidate)` 足以满足原容差。`D` 需要包含已核对算法的归约/FMA/下溢模型；不是从 132 个样本拟合经验阈值。零/分类等受保护边界另判，不能被这个浮点公式吞掉。
- `S_ij=sum(abs(A_ik*B_kj))` 的完整计算本身是 O(MNK)。可先用全局或行/列范数乘积作较松上界（O(inputs)，全局量可无数组），但在严重消去或大 K 下可能让大量格子回放。公开库无完整误差模型时，不能把假设写成形式证明。应该同时记录回放率与其时间；若逼近全量，立即判该数据域不值得采用，而不是隐藏额外扫描。
- 更务实的第一轮数值验收仍是：提前冻结原容差、完整失败/极值/子正常/消去/块尾样本、独立高精度抽查及图内离散回归。不存在要求所有成熟库都做全输入形式证明的新纪律；但已经观察到的失败必须有真实覆盖的防护范围，不能用随机 allclose 掩盖。

如果候选对全部受支持有限正常输入还没有足够安全域，就保留对应原生路径。当前报告不建议修改比较语义、把强取消样本删掉、或仅放宽 direct 的容差。

## 最小下一步实验及可靠采用条件

1. 新 Accelerate LP64 与 vDSP 的原夹具已筛选且存在失败。下一步针对准入保护补真实大尺寸零填充失败、block/tail、输出先填 NaN、同输入内容变化、只读别名、空维。分别记录浮点容差、IEEE 与 bool/error，bitwise 仅作诊断或保序候选特有证据。
2. 先测内部 heap 与活动栈/线程，再做完整普通/prepared/direct/IPC 时序；保留原布局和失败，不只筛选跑赢的形状。新入口若仍增加真实活跃内存，本用户当前“不增内存”要求下不能只凭大幅提速采用。
3. 只有确有优势再做图内资格接线；真实 shared/process 路径必须验证，不以 host F 输入代表 worker F 输入。数值敏感图仍走现有路径，并验证实际 dispatch。
4. Eigen 输出尾区复用先做独立 prototype；相比独立 caller-output kernel 之外，还需含扫描、packing、尾行补算、独立结果保留的完整请求。精确统计每请求复制而不新增无意义常驻 audit 元数据。
5. 同核 NJIT 对照应写清 SciPy NEWLAPACK/Accelerate，与 NumPy OpenBLAS 区分。无运行证据不写 AMX。冷、暖、两结果、close 的 heap/外部 RSS/footprint分别验，不能把上一轮 31/31 堆持平替代总内存持平。

若上述库不能同时闭环，优先继续真实收益域：C_B 无 packing 的成熟 Array SIMD、零行 stride 广播的依赖复用、非连续输入避免 NJIT 强制复制；随后才是合法图融合。不可仅为了追求单一 GEMM 倍率而改数学契约或移出测量成本。
