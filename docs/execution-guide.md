# 执行与性能指南

## V99：实际 owned MC 复验通过，整体目标仍开放（2026-10-10）

2026-10-10，仅 `cpp/compact_path.cpp` 与 `tests/test_compact_weight_lanes.py` 的 V99 变更已进入实际源码；本机实际 AOT 构建身份为 `f789954a…`。实际普通相关 Python **1,049**、全量 **17,179** 与 native **9/9** 无失败或跳过（相关不与全量相加）；普通 owned MC 的 **42/42** fresh 任务及其原始证据经独立复核。实际这一轮是普通非 sanitizer（Normal）构建，不是 actual SAN；隔离候选的相关 SAN/OFF 证据保留各自身份与范围。

Normal/Student C8 的完整 owned 链为785/924 ms，保守预热 NJIT 为3188/3329 ms，约快4.06/3.60倍；相对较快旧 C++ 省46.22%/42.30%。Normal C8/C32 的 NJIT 控制轮次漂移仍保留且原因未确立；时间取三个阶段中最快的 NJIT median（NI3），不用异常慢轮次扩大优势。C32 的 warm tracked heap 仍比 NJIT 多 **4,679,434 / 7,007,433 B**（Normal/Student）；Student C1 只比保守 NJIT 快 **1.066×**，比较快旧 C++ 慢约0.30%，不说所有负载都显著改善。LongK 的历史退化与完整 QP/前沿/探索等主要链仍开放，不据此关闭整体性能或平台业务资格。 [六组原始汇总与证据](physical-execution-evidence/2026-10-10-mc-weight-lanes-v99/README.md)。

以下其他阶段的身份、通过项及未完成项按其形成时范围阅读；不能累计为当前一次全局验收。

<!-- V21-ACTUAL-VERIFIED:BEGIN -->
## V91 源码接入；实际完整性能复验待完成

4 份原生实现文件及 5 份测试文件按冻结候选接入。候选身份 `723c06b3…` 的完整普通 Python 为 17066 项；相关 ASan/UBSan 和关闭加速库各 2385 项，三种模式原生各 9 项，无失败或跳过。sanitizer 未启用 leak detector，不能称全套 Python 已做 sanitizer。上述是隔离候选的同源资格；实际目录已经重新构建，93 份原生 source/config 的身份仍为 `723c06b3…`，25 份 wheel 载荷逐项核对；actual wheel SHA 为 `caf802b4…`。实际目录另通过相关 Python 2385 项和 native 9/9，测试 ID 与冻结候选一致。实际重建包的完整 C/F 负载复测尚待完成。[V91 接入与复验](physical-execution-evidence/2026-10-09-latent-current-v69/future-result-screen-v74/qualified-adoption/row-state-reuse-v78/README.md#gaussian-filter-stream-v91)。长 K 矩阵、完整 QP/前沿/随机探索等主要计算链及整体目标继续开放。

### V85 历史验收（保留其版本范围）

V85 阶段的 7 份 C++/头文件及 4 份测试已接入，93 份 source/config、200 份测试与实际 AOT wheel/native/worker 身份 `0ebc5921…` 一致。R4 普通、ASan/UBSan、关闭加速库各通过相关 Python 1353 和 native 9，无 skip；实际目录重建后另验 Python 1353/native 9。4 项新回归含在 1353 内，不重复计数。原 R3 全量与三轮 54 项性能证据保留其身份；R4 与实际包补验 36 项完整配对测量，不将旧全量改称当前身份全量。该阶段的 75 万行 HMM 为 2.369/5.167 秒（C++/预热 NJIT），堆峰 183.797/192.755 MB；较旧 C++ 少约 9 MB 堆峰，时间基本持平。25 万行堆峰仍多约 2.03 MB。RSS 独立计量，不与堆相加。[完整资格、实际复验和剩余项](physical-execution-evidence/2026-10-09-latent-current-v69/future-result-screen-v74/qualified-adoption/row-state-reuse-v78/README.md#gaussian-filter-stream-v85)。长 K、完整 QP/前沿/随机探索等主要负载仍待验收。

### V78 历史验收（保留其版本范围）

V78 的 93 份 source/config、199 份测试与实际 wheel/native/worker 身份 `75d26e3c…` 一致。同源冻结全量 16,899、SAN 相关 6,364、关闭库 6,363 通过/一个既有环境 skip，各 native 9/9；实际目录重建后另验相关 965/native 9 及 28 项完整配对测量。三组冻结完整链耗时为暖 NJIT 的 27%～47%，比 V74 实际代码少 8～56 MB 堆峰。实际 GMM 200 万行为 3.308/7.618 秒（C++/NJIT）；HMM 75 万行为 2.433/5.399 秒。HMM 25 万行仍多约 7.53 MB 堆峰；41 KB 或毫秒级变化只作诊断，不能称整体内存目标完成。[当前资格、三轮对照和实际复验](physical-execution-evidence/2026-10-09-latent-current-v69/future-result-screen-v74/qualified-adoption/row-state-reuse-v78/README.md)。V66 矩阵及 V68 求解按原版本证据保留，完整 QP/前沿/随机探索等主要负载仍待验收。

### V35r2历史验收（保留其版本范围）

93份数值源与实际工作区AOT wheel/worker身份一致。同源全量15146项，普通/SAN相关各3352、关闭库3042和三套native各9通过；当前目录新wheel另验572项及native9。输入、旧结果、C/F布局和worker所有权已验证；未提交、推送、全局安装或部署平台。

270次完整计时和180次fresh峰值已闭合。4096高量级FF矩阵从6.17→1.05秒，强同核NJIT约0.66秒；申请堆815.88MB与原815.82MB基本持平，NJIT约811.60MB，没有大额内存收益。26万行风险链四核约1.58秒，强NJIT约2.16秒；派生/混合链约0.20–0.21秒，NI约0.27秒，申请堆仍高于更省内存的NI。V33此前3.4GB→30MB及常规F矩阵少138MB的成果作为[历史证据](physical-execution-evidence/2026-10-05-candidate-summaries/quantiles/row-risk-design/mixed-roots-v25/input-bounds-v28/packed-f-successor/integration-v33/README.md)保留。[当前证据](physical-execution-evidence/2026-10-05-candidate-summaries/quantiles/row-risk-design/mixed-roots-v25/input-bounds-v28/packed-f-successor/pairwise-gemm-experiment/integration-v35r2/README.md)包含中断恢复与完整原始观测；不等于随机探索、有效前沿、模型或平台迁移通过。

## 1. 五个概念并不互斥

| 能力 | 解决的问题 | 如何启用 |
| --- | --- | --- |
| SIMD | 一个 CPU 核的一条指令处理多个数值 | 内核、CPU/OS、布局与长度满足条件时派发 |
| 多线程 | 同一进程分担独立计算块 | Planner 选择 thread，统一 C++ ThreadPool 执行 |
| 多进程 | 原生计算隔离、重负载分块、Hard Stop | Planner 选择 process，启动/复用 calmetrics_worker |
| 协程 | 调用方事件循环不被同步请求阻塞、并发等待请求 | 显式调用 execute_async/execute_many_async |
| 共享内存 | 减少跨进程重复搬运，管理共享输入/输出 | process 计划满足运输条件，或显式建立 SharedInputBundle |

可以同时使用“协程提交 → 多进程执行 → 每个进程单计算线程 → 内核 SIMD → 共享内存”。目前不启用每个进程再套一组计算线程的混合并行。

## 2. SIMD：什么情况下真正使用

当前 canonical 目录标记 18 个显式 SIMD 算子：add、subtract、multiply、divide、minimum、maximum、negate、absolute、sqrt、reciprocal、equal、not_equal、less_than、less_equal、greater_than、greater_equal、finite_mask、matmul。

- ARM64 使用 NEON；x86_64 基线 SSE2，AVX2 还要求独立目标已构建且 CPU/OS 支持。当前无 AVX-512 实现。
- 元素路径需要符合内核要求的连续布局和足够长度；尾部用安全标量计算。int64 比较等特定重载不因算子总标签而保证 SIMD。
- matmul 按形状与 strides 选择 SIMD 路径，保持内积累加顺序。下述大矩阵路径使用受 Planner 预算约束的局部面板打包；输入仍按原布局借用，打包字节单独审计。
- 非单位步长、小数组、不支持的重载、顺序敏感的归约/递推可能走 C++ 标量路径；仍然是原生计算。
- 直接调用可用 simd="auto" 或 "scalar"；显式要求不可用 ISA 会报错。NaN 处理不因 SIMD 而改成自动删行/填零。
- Planner 的 simd_min_elements=128 影响计划中的候选节点标记，**不是所有内核统一的硬切换阈值**。实际向量化情况看审计，不仅看 simd_nodes 或 simd_eligible。

```python
import numpy as np
from calmetrics_engine import operators as op

x = np.arange(256, dtype=np.float64)
y, audit = op.add(x, 1.0, simd="auto", audit=True)
np.testing.assert_array_equal(y, x + 1.0)
print(op.available_simd())
print({key: audit[key] for key in ("isa", "vector_elements", "input_copy_bytes")})
```

不能从“C++”或“SIMD 可用”推导一定超过系统 BLAS。矩阵、递推、排序、窗口需在目标机器按真实端到端负载测量。

macOS 15+ arm64 且 Accelerate 开启时，至少1024元素的 `exp/log` 可用 vForce。直接算子仅 automatic；图中仅单一 typed float64 数组根、无消费者/作用域/隔离/观察器。同序 C/F 连续输入、至少1024元素的 F 连续列段或 C 连续行段，可无新数值工作区直接写独立输出。准入要求标准舍入方式，exp 有限且在[-700,700]，log 有限且不小于最小正规数。直接算子仍全量预检后才写，拒绝后原内核保持定义域、IEEE 与写前缀；独立图根按最多65536元素的实际连续块预检，坏块执行原标量数学，其余块继续库计算，不创建整块事务结果或有效性位图。失败根结果不发布。小任务、内部节点、显式 scalar/SIMD、其他平台及库关闭仍原路径。`accelerate_vforce_calls` 记录实际库调用数，贯通 direct audit、native_chunks 与当前响应布局25；`vector_elements` 仅记录实际库处理元素。

指定结果布局与输入相反时，二维/三维至少65536元素、第一轴至少32、其余轴展平至少8，可用两块总计最多64KiB的 Workspace。F→C按128×32、C→F按64×64分块，vForce直接读取连续输入段，vDSP转置再写独立结果；不创建完整矩阵中间数组。F允许正向列补边和部分窗口，C要求每行内层连续、允许行pitch补边；其他几何保留原路径。转置与输出搬运的有效字节合计2×输入消费字节，记录为 `algorithm_copy_bytes`，不能称为全路径零拷贝。Planner按形状保守覆盖两种方向，实际分配按选中路径发生；预算内的已分配 Workspace 可保留复用。详见[跨布局设计与证据](physical-execution-evidence/2026-10-01-vforce-cross-layout/README.md)。

自动布局对上述独立 exp/log 根、直接大数组 F 输入保留 F；逻辑轴不变，部分窗口只读取本窗口的列段，不因窗口外坏值失去准入。series 根应显式 `result_format="typed"` 才使用上述图路径。prepared 仅现有 single lane，每次重新检查当前输入，不缓存数值或资格；仍占现有 Scheduler CPU token，不新增私有池或全局线程设置。

`symmetric_eigh` 在已测的 macOS 15+ arm64 上，n≥256 且为 128 倍数时采用压缩上三角工作区，连续旋转区段复用 Eigen 3.4 的 `Map/JacobiRotation`。仍执行原有有界循环 Jacobi，默认容差、sweeps、错误和两项投影不变。其他尺寸/平台保留密集路径；`CALMETRICS_ENGINE_USE_EIGEN=OFF` 使用同契约原生旋转。尺寸限制来自访存实测，不能视为其他机器的普遍性能保证。

当类型化 DAG 只消费相邻分解的特征值时，健康且无观察器的原生执行省去特征向量旋转和完整 n×(n+1) 中间结果；保留有界 Jacobi 的每个矩阵计算与停止，结果逐位相同，N 元素根直接写独立结果。输入先完整复制到工作区，因此也支持 arena 中的派生矩阵。共享特征向量、isolate 或 Excel 观察器仍执行完整逻辑节点；直接 `op.eigenvalues(record)` 仍是借用视图。完整分解 n≥128 时使用标准库索引排序及原位行置换，增加 N 个 `size_t`，计入预算；不因几个 KB 的增加否决大负载收益。Planner 保守估算覆盖未融合和继承隔离的子图，不能把估算等同实际申请堆峰值。

`triangular_solve` 在 macOS 15+ arm64、Eigen 开启、非 scalar、至少32行和8个 RHS 时，可用固定16/8/4/2/1元素的 Eigen Array/Map 块减少 RHS 的重复读写。A 的列步长非1时暂限定至32个 RHS，宽列输入保持原路径；A 无须打包或转换成 C。独立 C 解缓冲承接既有 RHS 复制，每个元素仍按原 k 顺序消去，随后按原列序除法并检查有限性；只检查消费的三角区，单位对角不读取。非有限系数走该行原顺序更新，保留失败前缀。没有额外数值工作区或私有线程，直接审计 `isa="eigen-array-triangular"`。这不是 Eigen 默认三角求解器：后者在已测强消去案例中不满足现有数值口径。显式 F 图结果仍遵守下述已规划转换规则；本内核的 C 解缓冲不代表全图必须采用 C 布局。

Eigen 以头文件编译到原生引擎，借用独占缓冲，不建立线程池、调用 BLAS、修改调用方线程设置或触发 Python 数值计算。直接算子 audit 和整图 `native_chunks` 的 `eigen_rotation_calls` 记录完成算子的实际库旋转次数，包含子作用域及进程传输；未使用为 0。`vector_elements` 统计当前路径报告的向量处理规模（MatMul为输出单元数），不等于全部Eigen内部指令或浮点运算数量；`isa` 还可用实现路径名称标记下述 EM 内核。必要缩放复制、压缩 scratch 仍计量，Planner 保留保守上界。时间和活跃堆峰值已有改善证据，但 RSS 及 NJIT 联合门禁仍未闭环，详见[物理执行优化](physical-execution-optimization.md)。

直接算子和普通 canonical 路径的单参数矩阵 `covariance/correlation` 复用独立结果的最后一行暂存均值和标准差，数值 scratch 为 0；输入、旧结果均不写入。macOS 15+ arm64、至少 32 行时，行方向连续且至少 8 列使用 Eigen Array 的 4×4 特征块和 128 行时间块，各结果格沿原顺序继续累加；均值仍按 Welford 顺序更新，尾部单独完成。时间步长为−1、特征步长不小于行数且至少8列的负向F视图也复用同一有序块，通过 Eigen InnerStride 借用列数据，不打包矩阵、不新增数值工作区；每格保持原scalar观察顺序，内部消费者和库失败重放也适用，详见[有序步长验收](physical-execution-evidence/2026-10-03-covariance-ordered-strides/README.md)。正向列方向连续且至少 8 列、有限且在原溢出界限内、列 stride 不小于行数时，使用固定 8 列 Eigen Array 做 Welford 均值扫描，并将 2×2 列对的居中乘积合并计算，共享读取、保持原 Eigen 两包归约分组；不足块的列保留完整内积。其他列连续布局保持原逐列均值与 Eigen Map 内积。极值溢出风险的列归约、小样本、其他 strides/平台、显式 scalar 或禁用 Eigen 时使用原生标量路径。`eigen_covariance_calls` 记录成功完成的库协方差内核次数，相关矩阵也计一次，贯通子作用域和 worker。该字段不是 CPU 指令数；上三角完成后才归一化、镜像，样本除数、零方差错误和独立结果契约不变。Planner 的保守图容量、框架分配及 RSS 仍须单独计量。

独立 typed float64 矩阵根 `covariance(X)` 或 `correlation(X)` 在相同平台、Eigen/Accelerate 均启用、`32≤T≤1,000,000`、`256≤D≤4096` 时，可采用守卫 Accelerate DSYRK。支持单根及下文受限的同源双输出；限定无内部消费者、非 isolate、无子作用域和观察器的图。逐次检查输入布局、有限值与受控中心化尺度、标准舍入环境，不按地址缓存数值准入。C方向要求列步长为1、行步长绝对值至少D，支持倒序及带间隔的时间行；F方向允许时间步长为+1或−1、特征步长至少T，准入leading步长限制在±INT_MAX。负向F沿原scalar顺序计算，不能改用正向F的归约树。倒序行按逻辑顺序做均值、对角和重放，只有本来需要的中心化面板交给BLAS，不先复制整块输入；进程传输原有的非连续输入打包仍单独计量，不能称整链零复制。共用 canonical 均值和局部重放接口，以确定性舍入模型检查原 1e-12 容差、对角位模式及正负/零分类；不确定格局部重放，过密或尺度不合格时完整原生重放。模型假设及适用构建见[设计与验收](physical-execution-evidence/2026-10-01-covariance-syrk-integration/README.md)，不能外推成所有厂商库版本的形式化保证。倒序C方向扩展见[倒序行验收](physical-execution-evidence/2026-10-03-covariance-reverse/README.md)，后续负向F扩展见[负向F验收](physical-execution-evidence/2026-10-03-covariance-reverse-f/README.md)。现有IPC将反向连续列打包为正向F，worker按该实际布局执行；本地负向F与worker的精确对角线分别对照各自canonical路径，完整结果保持原容差。

相关矩阵使用独立的归一化误差检查：保留 canonical 对角，按原标准差乘积归一化后再检查原 1e-12 容差，不能用协方差的小绝对误差代替相关系数误差。协方差仍保留源幅度≤1e4；C/负向F、512≤T≤4096时，实际中心面板幅度超过8且不超过32会重试一次128行面板，使用既有逐块递推的保守分组界，普通幅度保留原全局界；更大幅度、正向F及其他行数保持canonical回退。相关矩阵允许有限源幅度≤2^400，保留非零中心值≥2^-400、标准舍入及归一化误差/符号检查。这些范围结合行数上限约束平方和、范数乘积与分母的浮点指数，不是公共输入限制；超界或误差不合格仍走canonical。[范围推导与验收](physical-execution-evidence/2026-10-03-correlation-range/README.md)。复用已不再使用的块范数向量保存标准差，不新增向量；局部重放后沿唯一原规则归一化与镜像，不强设对角为1或截断到[-1,1]。零方差/非有限分母拒绝候选并执行原错误路径，全零相关矩阵不能使用下文的协方差零成功特例。原审计字段统计两种算子共享的协方差计算阶段，不新增IPC字段。

该图路径借用输入，复用本次独占 C 结果下三角承载暂时界和标记，无额外 D×D 矩阵；同一独立根显式 F 可直接写最终独立结果，完成位级对称镜像后公布 F strides；不产生 C arena 或根转换。中心化面板由 Workspace 管理，实际大小 `(B+3)*D*8`，默认 B=512，T/D 均≤2048 时 B=T，其中受限同源双根的C输入将B限制到1024，减少同时保留两份结果时的面板占用；正向F输入保留完整面板以复用centered fallback。短面板重试会先释放原面板并重算，不同时保留两份；首次大面板仍真实分配过，全部尝试都计入中心化字节，不能把最终小容量当作请求峰值。短面板路径将每行的上三角按32列分组，用组内 prior 绝对最大值与范数最大值代入原递推；每组只在下三角起点存界，最终先读取再覆盖重放标记，不新增动态工作区。更保守的界可能增加回放，原容差与密集回退规则不变。其余逐块路径使用一个32×32 double局部Eigen转置块，载荷8 KiB，完整栈和库峰值仍需实测。按区间内所有可能面板大小预留，另加 32MiB 库工作区估算。该库估算不是厂商硬上界，也不等于实际分配；原 Planner 保守容量仍存在。恢复调用线程原 BLAS 策略后才完成或重放，仍只占一个 Scheduler CPU token。拒绝后释放无用面板，prepared 重读当前值，快照保持独立。协方差若首次中心幅度拒绝时已完整生成本次 F 中心化面板且 D 为偶数，先复用现有 Eigen 固定归约树消费面板，再释放；省去重复均值和减法，保留原 N−1 除法、尾部及逐位镜像。tiny 首先拒绝、部分面板、奇 D、C 布局及 DSYRK 后的拒绝继续原重放；这项复用不扩大 DSYRK 数值准入域，原面板分配峰值仍计入审计。

同一已准入的独立 `covariance(X)` 根中，实际输入全为 +0 时，标准舍入下 canonical 的均值、居中乘积、累加与样本除法均精确为 +0，可由标准库直接写独占结果并释放旧面板。`covariance_zero_calls` 标记这一精确特例，库尝试/调用数为 0；全零数据只要含 -0 就沿 canonical 原布局归约，不进行无用的 DSYRK 尝试。输入改变时重新检查，不缓存零资格；未准入图继续原逻辑。

`native_chunks` 的 `accelerate_covariance_attempts/calls` 分别记录开始尝试和成功；`covariance_syrk_calls` 为实际 DSYRK 块数，`covariance_replayed_cells` 为实际 canonical 重放格数（含 F 对角及完整回退），`covariance_fallbacks` 为尝试后完整重放次数，`covariance_panel_materialization_bytes` 为所有已写中心化面板字节。完整重放保留之前工作计数，避免审计重置掩盖成本；这些字段和精确零计数通过当前 IPC24 传输。面板属于计算物化，不能用 `input_copy_bytes=0` 宣称整个请求零分配或零写入。除下文受限双输出外的共享/比较/状态消费、直接算子、Excel 观察器和未准入布局继续上段 canonical 逻辑；逻辑 AST/DAG、数学与 Excel 公式不变。

单一独立根 `cholesky(covariance(X))` 在 macOS 15+ arm64 且启用 `CALMETRICS_ENGINE_USE_ACCELERATE` 时，对至少 2048 列、行数不少于列数、精确连续的 C/F 输入，可将两个已融合步骤交给 Accelerate `dsyrk`/`dpotrf`。准入还检查有限值、数值幅度、接近零的协方差及分解主元；不满足条件或库分解失败时重放原路径，保留原错误契约。库线程在本次调用内限定为单线程并恢复，仍占用 Scheduler 的一个 CPU token。512 行中心化面板与均值的保守上界为 `513*D*8` 字节，计入 Planner；成功调用计入 `accelerate_covchol_calls`。直接算子、共享协方差的多根图和其他形状仍使用原实现。显式 F 根结果保留 F 连续布局；符合下述私有根条件时，在独立结果内完成原 C 数值计算，成功后原地换位，不再分配第二块方阵。

同一顶层图先计算 `covariance(X)`、后计算 `correlation(X)`，且两者都是直接矩阵输出、来自同一输入节点时，可复用本次已完成且有限的协方差结果。引擎将它复制到相关矩阵自己的输出区，再按原规则归一化；两个输出仍独立，零方差导致相关矩阵失败不会修改健康的协方差。每次共享实际复制 `D*D*8` 字节，计入 `algorithm_copy_bytes`，即使随后归一化失败也保留该计数；启用上述 Eigen 路径时，两根合计只调用一次协方差内核。不同输入节点、反向执行顺序、非直接根、子作用域、观察器或非有限结果保持各自计算。复用不跨区间或请求，prepared 每次读取当前数据；没有新数值缓存、工作区或进程协议，Planner 仍保留原保守成本。

恰好按该顺序输出这两个同源 typed float64 矩阵、两者均无内部消费者/作用域/isolate 时，可共享一次守卫 DSYRK。协方差候选必须同时通过协方差容差和相关系数归一化容差；仅协方差误差足够小不能准入。仍使用协方差源/中心幅度限制，未扩大成单独相关矩阵的较大数值范围。两份独立结果支持C/F连续存储，完成的协方差严格对称，因此F字节可直接复制后按原C内核归一化；F双根均无需额外arena或最终转换。观察器保持canonical计算。额外输出、反序、不同源、重复根或任一内部消费者不准入此优化；[设计和实测](physical-execution-evidence/2026-10-03-covcorr-pair/README.md)记录适用范围与成本。

macOS 15+ arm64、启用 Eigen 时，`matvec` 对至少128行、8列的两类矩阵采用Eigen Array跨行计算：列连续、正向不重叠列距时，每4096行分块；行连续、正向不重叠行距时，每8行使用固定大小的借用Map，剩余尾行复用同一借用Map的单行实例。每行从+0开始，严格保留按列递增的乘法/加法顺序，禁止FMA收缩，不跳过零系数；NaN/Inf和次正规数也沿原顺序传播。输入向量可以有正、负或零stride；独立结果直接作累加器，无数值scratch、输入打包、误差复算或库线程设置。其他布局/平台、显式scalar或Eigen-OFF使用原生有序归约。

`portfolio_returns(returns,weights)` 现按相同准入条件复用上述有序Eigen矩阵向量路线，直接累加到本次独立时序结果。现有数学、类型、形状与所有权规则继续适用；`eigen_matvec_calls` 在直接审计及整图 `native_chunks` 中记录实际调用。当前同源资格和普通CPU1完整风险图见[V64验收](physical-execution-evidence/2026-10-07-matmul-terminal-proof-v63/portfolio-followup-v64/README.md)。

`quadratic_form(x,A)` 也复用上述内核，但严格保持先计算 `transpose(A)*x`、再与 `x` 点积的既有顺序，不要求A对称。投影写入原有N个double工作区；不增加矩阵打包或数值缓存。准入条件按转置后的视图判断，显式scalar仍绕过库；原形状检查、错误和独立结果契约不变。

`eigen_matvec_calls` 记录完成算子的实际Eigen矩阵向量内核次数，贯通图、作用域与worker；失败算子的部分审计仍不保证返回。先前未发布的Accelerate GEMV路径已由Eigen实现替换；系统 Accelerate 用于上述受限协方差→Cholesky 和下述独立无消去 MatMul 图路径。连续rank-1 `dot` 和其消费者仍用标准C++ `std::inner_product`，保持原顺序及通用strides。直接调用的完整Cholesky及通用BLAS GEMM候选尚未替换；下述保序矩阵路径分别复用Eigen Array及固定版本的Eigen pack/gebp。

`solve_multi(A,B)` 和单右端 `solve(A,b)` 在相同平台、Eigen开启、n≥128且非scalar时，复用同一32列面板消元，并用固定16列Eigen Map融合两次有序更新。n≥1024时，已完成主元面板后的独立尾部行再用4行×8列的固定Eigen数组共享主元读取，按原主元顺序更新后一次写回；面板内递推、剩余行和列尾部保序。它复用原有n²工作区保存当前面板因子；分组数组的逻辑容量为256 B，不新增数值堆缓冲，实际调用栈另行测量。主元、阈值、RHS更新、回代和失败输出契约不变。直接审计的 `isa="eigen-array-blocked-lu"` 表示采用该物理路径；整图不单独归因此子步骤，不新增库调用计数。显式scalar、较小矩阵和未准入平台仍用原路径。单右端仍保留原 `n²+n` 私有系数/RHS 工作区及原回代；共享内核用编译期主元判据保留其特殊规则：只拒绝 `abs(pivot)<=1e-14`，不把 NaN 主元改报奇异，也不新增有限性检查。单右端消元失败前不写公开输出；多右端仍保留其原有限性检查和失败前缀。这是物理内核复用，不统一两个数学契约，也不表示时间和完整内存验收已全面通过。

单一typed float64矩阵根 `matmul(a,b)` 或私有 `transpose(matmul(a,b))`，在macOS 15+ arm64、Accelerate开启、M/N为1–INT_MAX、K为1–8192，且 M×K×N 至少为 2^26 时，可使用受限系统DGEMM。这里只扩大可选物理库路径，不限制公开MatMul形状；LP64 leading dimension须可表示。输入须为正向C/F leading-dimension布局；每次重读实际数据，不读取padding，不缓存prepared输入的数值资格。允许+0或绝对值位于`[2^-128,2^128]`的有限非零值；负零、非有限、域外幅值及不合适strides保留canonical。含负值要求当前线程为FE_TONEAREST，不改变舍入模式。多根、下游消费（除唯一私有转置）、isolate、作用域、内部prebound和观察器不准入；直接算子接口仍是原实现。已准入根直接写本次独立C/F结果。

无正负消去时保持原一次DGEMM路径；混合符号在完整输入界及0.000049误差预算内也可一次计算，否则启用Eigen+Accelerate分块候选。分块仅在Eigen开启、确定性范数/舍入界证明准入时执行：每次DGEMM以beta=0独立计算，首块直写结果，后续Eigen显式顺序或平衡归并；不把库内部隐式部分和当作可证明的外部归并。近零或符号歧义单元仍按原k序回放；晚失败由唯一canonical实现覆盖整个私有根，完成前不发布部分结果。非法域、未知舍入和无法证明的情况继续canonical，不采用概率误差界准入。

另有私有矩阵标量链：单一 typed float64 矩阵根至少65536元素，MatMul两端是直接输入，后接1～8步单消费的add/subtract/multiply，另一操作数为rank-0常量、参数或输入，且图内无窗口/作用域时，中间矩阵可共用本次独立C/F根存储。观察器、isolate或额外消费者保留原路径；多根仅下述共享闭包可优化，divide仍保持原失败前缀。负literal的negate节点不属于该闭包，负参数可按实际资格判断。所有数值资格每次重读，不跨请求缓存，旧结果和输入不成为工作区。

共享MatMul后返回2～8个完整矩阵根时，若各分支分别为1～8步单消费标量add/subtract/multiply，根声明的order一致、实际结果同序C/F连续、至少65536元素，且逻辑最后分支在其他分支完成后才开始，可把共同乘积写入逻辑最后完成根的独立存储。其余根分别写本次独立结果，最后原地完成最后分支；根选择按逻辑拓扑，不依赖用户输出列表顺序。每条尾链独立证明0.000049预算，输入只在同次调用扫描复用，不跨请求缓存数值。混合声明auto/C/F、混合实际布局、非连续或非法slot、额外消费者/共享尾链、交错分支、scope/isolate、观察器及内部prebound保持canonical。混合声明在metadata准入前排除，Planner与实际arena一致；公共C/F布局能力保留。见[多输出实际验收](physical-execution-evidence/2026-10-07-matmul-terminal-proof-v63/multiroot-followup-v65/actual/README.md)。

仅当该共享闭包一次执行chunk恰有一行、根缓冲紧密且无padding时，省去重复预清零；成功逐根完成后才发布，异常清零未发布根，已完成根保留。多行、padding、降级和普通输出继续原清零/前缀契约，不将行批处理与未分块失败前缀混为同一行为。省掉共同乘积arena并不消除必需多输出的内存；强NJIT同样复用必需输出。上述V65 double路线的历史时间差不因下述受限SGEMM通过而全部关闭，当前本双根负载仍无显著NI内存优势。

相同平台的Accelerate路线还须以完整输入界、标准舍入和全链误差放大证明准入：最多8个标量操作，乘数绝对值在`[2^-24,1]`，加减量不超过16，非零加减量不小于`2^-128`；原绝对误差预算0.000049不放宽。不把标量移入DGEMM alpha，原两步乘加仍分别舍入。输入证明只在同次调用、同一Prepared实例使用；末端近零/符号歧义回放原K序及标量步骤。取消、超时或分配失败清零未发布的私有根，普通/公开输出仍保留原契约。资格不符或库关闭时按现有数学计算，不能据此推断其他平台的性能。

在上述终端标量闭包及原binary64证明通过后，可进一步选用系统SGEMM。只在FE_TONEAREST、K≤8192、原库工作量/整数/字节预算成立、原输入紧密C/F且为+0或绝对值[2^-36,1]的有限非零值、每个完整根的binary32转换/乘积及原binary64尾链误差界≤0.000049、向外舍入的终端区间离零超过该根误差时准入。任一根不符即保留原double/canonical路线；公开MatMul、其他消费者及原输入域不被限制。不同输入/结果C/F保留各自连续布局，不要求统一为C。

M×K+K×N≤M×N时，仅在尚未发布的独立double根字节内放置float乘积和两个打包输入；库同步返回后先恢复完整double数组及新指针，经character表示倒序扩展，不保留跨请求数值状态。输出仍为float64、跨根/跨代独立；未完成根错误恢复、CPU1和线程状态协议沿用既有规则。打包与扩展目的字节计入algorithm_copy_bytes；本例为576MiB，并计入完整耗时。没有新Planner容量或最终复制，不称全算法零拷贝。小例默认Excel预算会明确拒绝该大工作量组合，本轮没有实机Excel重算；公式与核验规则未放宽。

矩阵标量两步复用已有Eigen表达式。早选和实际执行共用真实同序连续资格；F输入/C结果、pitch或反向步长不被早返回截断，继续已有column/tile或canonical路线。保留指定结果布局，没有新增整矩阵转换副本、线程或协议字段。单根、padding、多行及降级路径的typed初始清零和调用者输入准备复制仍计入完整请求；下述已准入共享单行闭包省去重复清零，不能称为全链零拷贝。V63/V65原double路线的历史资格与差距见[V63/V65阶段](physical-execution-evidence/2026-10-07-matmul-terminal-proof-v63/README.md)，当前有界SGEMM的适用数据及完整对照见[V66验收](physical-execution-evidence/2026-10-07-matmul-terminal-proof-v63/multiroot-followup-v65/sgemm-followup-v66/README.md)。本轮受限双根时间领先不关闭原始MatMul、长K及其他平台负载差距，整体性能目标仍未完成。

分块初始tile默认1024×1024；仅当两个输入均为F兼容布局、结果明确为F、M/N≥2048且K≥4096时，列宽改为256，之后仍按原scratch容量缩块。结果检查沿最终C/F连续物理行复用Eigen：先检查每个值有限，再检查最小绝对值；仅含模糊值的行逐单元进入原K升序回放。两种归并共用同一检查，不生成完整结果副本或另一个数值缓冲。长输出轴通过不超过8192的只读输入子视图逐块做范数预检；原私有输入界helper的8192保护保持不变，避免把未验证的巨大视图交给它。

库线程状态按调用线程设置、核对并恢复单线程，不增加私有池。符号标签的逻辑栈容量最多8KiB；更长轴另用字节向量，形状预算最多增加向上对齐至8字节的`max(M,N)`字节，实际栈由编译器决定。分块范数预检临时区最多`min(320*K+72,2*max(M,N)+8)*8`字节，拒绝时不改变调用者工作区、输出或审计；成功时先释放它再增长数值scratch。分块scratch上界`(320*K+72)*8`，Planner为潜在库根同时计入厂商估算`8MiB+512*K*8`；估算不提前分配，也不是厂商承诺的malloc硬上界。普通一次DGEMM仍可没有数值scratch；成功零根用标准库填充+0。`accelerate_matmul_calls`、`matmul_zero_calls`和库workspace估算贯通图/worker；平衡归并的最终tile搬运计入`algorithm_copy_bytes`，顺序直写无需最终tile复制，库内部packing另用fresh堆/RSS验收。

在扫描、范数、分块库调用前后、结果检查与逐单元回放中检查取消/截止时间；单次DGEMM、已有完整输入检查及canonical整根回放仍不可内部中断，不能保证所有路径最坏取消响应以单块为界。数学算子与Excel公式契约未改，前期根路径结果和强NJIT差距见[前期验收](physical-execution-evidence/2026-10-05-candidate-summaries/quantiles/row-risk-design/mixed-roots-v25/input-bounds-v28/packed-f-successor/pairwise-gemm-experiment/integration-v35r2/follow-up-v46/README.md)。长矩形及预算重选已接入实际代码，新的完整时间、fresh峰值和未解决NI差距见[接入验收](physical-execution-evidence/2026-10-07-budgeted-matmul-adoption-v54/README.md)。

未采用上述受限根路径的 `matmul(A[M,K],B[K,N])`，在相同平台、Eigen开启、自动ISA、M/N≥1024、K≥256，且两个输入均为C/F或正leading-dimension补边布局时，使用固定192×128面板的Eigen pack/gebp适配。每个输出沿完整K升序分别乘加；尾部由原有标量顺序计算。固定Eigen 3.4内部模板及NEON适配只在独立编译单元使用，升级库时须复核。直接审计为 `isa="eigen-packed-matmul"`；图/worker没有新增逐MatMul ISA标签，不能从聚合审计推定该字段存在。

该路径的数值工作区上界为 `(320*K+72)*8` 字节，包含两块面板、预取保护与对齐空间；由独占Workspace分配，Planner计入每个并发worker/分支和子作用域的预算。顺序节点共用工作区；K变化时即使最大数组面积不变，也会释放旧面板容量。没有全量输入布局转换或私有线程，`input_copy_bytes` 可为0，但 `algorithm_copy_bytes` 如实记录重复打包流量，因此不能称为算法零复制。直接Python MatMul保持既有独立C结果/out要求；无消费者、scope/isolate或内部prebound的独立typed float64矩阵根可明确F并直接写最终存储，复用同一有序pack/gebp地址映射或Eigen缓存块/尾部，无完整C转换副本。其他F根仍按原预算转换，不能外推为全部图直写。显式ISA、负/间隔步长、小形状、Eigen关闭或未准入平台继续使用相应已有路径。

未命中上述条件时，`matmul(A[M,K],B[K,N])` 在相同平台、Eigen开启、自动ISA且M/K/N均≥32时，C兼容右矩阵（元素stride为 `(pitch,1)`，pitch≥N）在K≥683时使用分块Eigen Array路径。固定32×64×64宏块内，以4×8独立输出块计算；各K面板从独立结果读取上一个面板的累加状态，继续按k递增乘加，首面板从+0开始。不使用分组部分和、FMA、输入打包或动态数值工作区；尾行/尾列按完整k顺序处理。A可保留已验证的负步长、间隔、F布局和零列步长。直接审计为 `isa="eigen-block-matmul"`。K门槛来自固定64 KiB候选工作集估计与实测，不是检测到的硬件缓存容量，也不是跨机器性能保证。

K较小时，方阵（M=K=N）的完整C连续右矩阵在N为8的倍数时仍使用原4×8路径；列连续右矩阵 `(1,pitch)`、pitch≥K保留原4×4路径，审计均为 `isa="eigen-array-matmul"`。显式ISA、小矩阵和其他未准入情况保留已有分派。分块计算会多次读写结果中的累加状态，不能称为结果只写一次；无数值workspace也不能代替完整堆、栈和RSS验收。

M/K/N均≥32且B为C/F布局（此项不要求方阵或N为8的倍数）或行连续补边布局 `(pitch,1)`、pitch≥N，且A行步长为0时，各行读取相同输入：本次调用只计算第一结果行，再复制到其余独立结果行，不缓存跨请求数值。这项依赖复用优先于一般矩阵分块。此时直接审计为 `isa="matmul-repeated-rows"`，`algorithm_copy_bytes` 如实记录 `(M−1)*N*8` 字节；输入复制与数值工作区仍为0，不能将结果行复制称为零拷贝。图和worker审计不单独传递ISA标识；数学契约、Excel公式展开和预算不变。实际栈、冷/热堆与RSS另行验收，零workspace不等于完整内存不增。

GMM/HMM的M-step在相同平台、Eigen开启、非scalar、至少32行时复用固定Eigen Map。特征连续且维数为4/5/8/9/16/32/64/128/256时，同时更新独立特征；32/64/128/256维使用独立函数，避免扩大公共入口栈帧。对F兼容输入（行stride=1、列距≥行数），维数≥8且为8的倍数时，改用8特征strided Map直接读取原布局，审计为 `isa="eigen-array-column-em"`。两路均保留样本累加顺序及 `(weight*delta)*delta` 的结合顺序，直接复用输出参数区，不打包输入、不新增数值堆工作区。其他维数、步长和平台保留原生路径，动态维和输入打包候选未接入。

`gaussian_log_density` 及GMM/HMM的同一密度步骤，在上述F兼容输入、至少32行/8维、1至16个成分时，沿8个独立观察行使用Eigen Array。每个值仍按特征原序平方、除法和累加，保留原exp/log、floor、停止规则；不足8行沿用原公式。结果先存固定128-double局部块，再按原行/成分次序写出并检查有限性，保留显式out的失败前缀。直接审计 `isa="eigen-array-column-density"`；EM后续参数更新可覆盖该路径标签。这1KiB暂存（V93比原实现增加512B）及编译器栈帧属于调用栈，不是 `workspace_bytes` 的数值堆工作区，不能宣称整体零内存增加。`eigen_density_calls` 记录已完成的Eigen密度调用，贯通direct audit、整图、子作用域和当前请求30/响应29；首次引入该字段的26/25记录属于历史版本；EM每轮计一次，不等于指令数，失败算子的部分审计仍沿现有规则不保证返回，也不计为旋转/协方差次数；无库私有线程或跨请求数值缓存。完整请求性能及剩余欠项见[物理执行优化](physical-execution-optimization.md)。

同一密度步骤对特征stride为1、N≥4096、D≥32、K为4正倍数的样本，使用固定4/8个独立分量的Eigen Array；K为8倍数选8，否则选4。每个分量仍按原特征次序累加，减、乘与真实除法保留分别舍入，逐分量检查失败前缀；模型按真实正/负/零strides借用，不打包、不新增数值堆工作区或线程。直接路径标记 `isa="eigen-array-state-density"`，EM M-step可覆盖标签，实际调用以 `eigen_density_calls` 为准。资格每次按当前几何判断，scalar、短样本、其他维数/步长、平台及Eigen-OFF继续既有路径；AST/DAG和Excel公式不变。[正式设计与验收](physical-execution-evidence/2026-10-01-density-integration/README.md)区分完整引擎测量与ABI筛选。

`variance_axis(X,0,ddof)` 对 C 连续二维矩阵，在相同平台、Eigen 开启、非 scalar、至少 32 行和 8 列时，使用固定 8 列 Eigen Map 处理独立列。每列仍按行序累加均值，再进行第二遍平方偏差计算；尾列、错误检查分组及失败前已写结果保持原规则。借用原有三个 8-double 局部缓冲，无数值堆工作区、输入打包或库私有线程；其他布局、轴和平台保留原生路径。直接审计 `isa="eigen-array"` 表示采用该路径；不新增整图库调用计数，Excel 公式契约不变。完整请求与内存结果见[物理执行优化](physical-execution-optimization.md)。

`sum_axis(X,0)`、`mean_axis(X,0)` 和 `variance_axis(X,0,ddof)` 对列内正向或反向连续（第一轴stride为±1）、正向且不重叠列距的二维矩阵，也可在相同平台、Eigen 开启、非 scalar、至少32行/8列时复用8列 Eigen Map。每列严格沿原行序累加；方差仍两遍计算，并按原列序进行有效性检查和输出，sum/mean 保留原来的非有限值传播。尾列使用原标量公式，无数值堆工作区或打包。直接审计为 `isa="eigen-array"`。

既有混合DAG双算子融合中，若第一个 add/subtract/multiply 的一个输入是上述正向列连续矩阵，另外两个输入是标量或按列广播值，可用 Eigen 固定4×4矩阵块、每16行分块直接写独立C序结果；第二步支持 add/subtract/multiply/divide，运算与操作数顺序不变。Eigen的存储序赋值承接列序输入到行序输出，尾行与尾列按原标量公式处理，不打包整个矩阵。仅原融合规则允许的输入/root持久后备参与，不扩大融合范围。中间结果作除数或任一广播除数为零时，先于任何写入回到原路径，保留错误顺序及失败前缀。分离准入、矩阵块、尾部与标量执行，控制实际调用链栈；保留 `fused_pointwise_calls` 审计，不把它单独当作Eigen调用计数。此优化不改变数学与Excel逻辑，也不等于全引擎时间/内存验收完成。

新图可用 `output_order="auto"|"C"|"F"` 选择结果存储策略，默认 auto。C/F 都是连续内存。已准入 F 列算术链使用固定8行 Eigen Array 沿列连续读取、直接写独立 F 根结果；反向连续列输入用固定块reverse表达式恢复逻辑行方向，无完整矩阵复制；尾行保持有符号步长。明确要求 C 时，正向列仍使用上述4×4存储序分块，反向列保留通用写入。完整逐元素图沿共同输出顺序遍历，F 输入无需整块转置后再执行 SIMD。通用数学形状和轴不变。

错误隔离请求的 auto 保守采用 C，因为逐位置状态仍按 C 写入；显式 F 不受此策略限制。auto 当前覆盖稠密身份根、C 输入转置根、单一私有 `transpose(matmul(a,b))` 根、私有特征分解双投影、F 逐元素图及受限列算术融合；其他路径保守采用 C。该私有转置根把 MatMul 的 C 顺序字节直接写入最终 F 结果，省去整块中间矩阵和根复制；显式 C、隔离及共享乘积图不采用此直写。其他不支持直写的显式 F 内核仍使用已规划 C arena 后转换，并计入根复制审计。中间节点及迭代状态仍遵循现有内核布局；直接算子入口暂未新增 order 参数。实际布局见 `outputs[r].order` 与值数组 strides，状态数组保持逻辑 C。详见[布局设计与验收](physical-layout-design.md)。

独立单参数 `covariance(X)` 或 `correlation(X)` 的 typed float64 矩阵根也支持显式 F 直写：一个根，或上文受限的有序同源双根，均须无消费者、非 isolate、无 apply/rolling scope。既有 C 内核写入本次独立结果，在完成上三角的逐位镜像后按实际 F strides 返回；同一对称方阵的物理字节保持全部逻辑值，不分配 C 根 arena 或执行最终转换。auto 继续返回 C，直接算子接口保持原契约；observer 使用 canonical 数学路径，同样可观察完成后的 F 值。该存储优化独立于 DSYRK 准入，覆盖 canonical、局部/全量重放和协方差零数据；相关矩阵零方差仍按原规则报错。未准入图保持原有 F 物化和预算。详见[协方差历史证据](physical-execution-evidence/2026-10-01-covariance-f-root/README.md)及[相关矩阵验收](physical-execution-evidence/2026-10-03-correlation-library/README.md)。

同一个私有 `symmetric_eigh` 的 `eigenvalues/eigenvectors` 两个紧邻typed float64根，可在macOS 15+ arm64、Eigen开启、N≥512且为256倍数时直接写独立特征值向量及F特征向量矩阵。图必须恰有这两个根，无其他消费者、作用域或isolate，且输出策略为auto/F；根顺序和名称可交换。复用唯一有界Jacobi及Eigen公开旋转，补边连续前缀工作区为N(N+1) doubles及N个排序索引，按共享几何进入Planner与Scratch。累计上下三角搬运和排序计入`algorithm_copy_bytes`；两根计入`direct_output_bytes`，不留记录arena或根复制。auto在这类图选择F；未准入形状仍沿记录物化同一F结果，明确C、观察器、隔离、库关闭等沿原计算链。进程按原logical program重新finalize，不新增IPC字段；prepared每次重读输入，快照保持独立。数学与Excel公式不变，性能结论限于[实测设计与证据](physical-execution-evidence/2026-10-01-jacobi-direct/README.md)，不能据此宣称有界Jacobi已经领先NJIT/LAPACK。

`softmax_axis` 在相同平台、Eigen开启、非scalar、总元素数至少256且轴长度至少8时，可对输入轴stride=1、输出轴连续的行复用Eigen向量指数。每行先完成有限值检查；只有 `max-min <= 700-log(轴长度)` 才使用库，避免Eigen指数截断和次正规边界，其他行保持原实现。指数直接写独立结果，按原顺序求和后做除法；没有额外数值工作区、输入打包或私有线程。直接审计 `isa="eigen-array"` 表示至少一行采用库；该指数内部含显式FMA，保证适用容差而非逐位等同。算子、图、错误隔离和Excel数学口径不变。

完整逐元素 typed 图还可以使用分段寄存器融合：保持逻辑 DAG，将相邻的两个标量广播加/减/乘及可选条件组合为已有 AOT SIMD 内核。长链可分成多个区域，共享中间值仍按消费者需求保留；其余安全逐元素节点继续在缓存块内执行。泛用中间块复用已有生命周期槽，整图短链减少重复描述及逐块结果计数。该路径保留逐次几何/参数校验、独立结果、负步长和错误契约；不代表任意子图、窗口或矩阵计算都已融合。设计、前后实测与尚未通过的 NJIT 门禁见[物理执行优化记录](physical-execution-optimization.md)。

混合 typed 图也支持两类受限融合：单一消费者的同轴 `take_axis → sum_axis/mean_axis`，以及两步加/减/乘（第二步还可除法）。仅当被延迟的输入来自本次调用的输入或独立根缓冲、不会被 arena 复用覆盖时启用；共享中间根、隔离错误和诊断观察路径保持逐节点执行。它们仍是相同的公开算子组合，Excel 保留逻辑步骤。按原 stride 访问和逐列累加，不改变选择顺序、浮点归约顺序、默认精度或迭代停止条件。

普通执行工作区按每个生命周期槽实际可能承载的 shape 上界分配，并区分 float64、bool、int64；typed 运算根直接写独立结果。相同线程切换图或缩小几何时释放不再需要的容量。Planner 使用同一容量计算，并保守计入未融合回退、子作用域、数值内核 scratch 与错误状态；计划值不是 RSS/分配峰值实测。

## 3. Planner 的选择顺序

这里的 rows 是**请求的区间条数**，不是观察点数 T；work 是 lowering/CSE/融合后物理 DAG 的结构性成本估计，不是 FLOPs、毫秒或节点数。

1. 校验输入、区间、dtype/轴、图身份，估算输入/输出/状态及 scratch 容量。
2. 若 hard_stop=True，选择 process。否则同时满足 `rows >= max(2,min_rows_per_worker)`，以及工作量达到 process_work_units **或**输入字节数达到 process_input_threshold_bytes，才走普通 process。
3. 未选择 process 时，单区间 typed 图若满足下述私有独立候选行闭包，线程数取 `min(cpu_budget,N/128,work/thread_work_units)`（整数除法）；至少两个线程时选择 `array_row`。各线程保留完整历史和所有共享根，按无重叠候选行范围执行。
4. 否则，若为标量根图、`cpu_budget>1`、`0<rows<cpu_budget`，存在至少两个足够重的独立编译分支且总成本达到 dag_branch_work_units，则考虑 thread 的 dag_branch 计划。共享上游与融合组先合并，不按每个节点开线程。
5. 单区间的完整逐元素 typed 图，在输入为连续 float64、数组形状相同、每个线程至少一个完整 2048 元素块，且每线程工作量达到 thread_work_units 时，可选择 thread 的 tensor 分块；所有输出共享同一分块内的上游计算。
6. 否则，当 `work>=thread_work_units`、`rows>=2` 且 `cpu_budget>1`，选择普通 thread。
7. 其余为 single，在调用者的原生线程计算。
8. 内存预算先降低 worker 数量。降至 1 仍超预算时，只有可选扩展MatMul根几何的原路径scratch确实更小，才在同一请求Plan中改用原几何并重新规划worker和运输；原因码为 `memory_budget_uses_original_matmul_geometry`。最多重选一次，原几何仍超预算则失败；原库支持的几何仍保留库预留，不删减必要scratch、不改变算法或精度。该策略贯通单线程、线程和进程，不能缓存进共享逻辑图。当前 CPU 准入还可能使请求排队；计划 worker 数不等于任意时刻真正并发数。

普通进程数初值为 `min(cpu_budget,max_processes,max(1,rows/min_rows_per_worker))`（整数除法）；普通线程数初值为 `min(cpu_budget,rows)`。Hard Stop 即使仅一个区间、一个 CPU 也需要进程隔离。单区间普通时序根不自动获得跨时间递推并行。

普通区间分块优先考虑 product_ids 提供的足够产品组，否则按区间；使用物理成本加权，而不是简单按行均分。单区间的 `array_row` 和 `tensor` 分别使用候选行范围与逐元素块。DAG 分支是区间并行不足时的受限补充。目前仅 scalar 根适用 dag_branch 策略；独立轴分块不能沿递推时间轴拆分。

### 当前默认值

| PlannerConfig 字段 | 默认值 | 含义 |
| --- | ---: | --- |
| thread_work_units | 250000 | 普通线程成本门槛；也参与重分支判定 |
| dag_branch_work_units | 5000000 | 独立 DAG 分支并行总成本门槛 |
| process_work_units | 4000000000 | 普通进程成本门槛 |
| process_input_threshold_bytes | 268435456（256 MiB） | 普通进程输入大小门槛 |
| shared_memory_threshold_bytes | 8388608（8 MiB） | 进程共享输入运输门槛 |
| min_rows_per_worker | 16 | 进程准入/数量与分块相关参数；不是每个观察窗口的长度 |
| simd_min_elements | 128 | Planner SIMD 候选标记长度 |
| max_processes | 8 | 进程数上限，还受 CPU/内存预算限制 |
| max_async_jobs | 8 | 每次 execute_many_async 的并发桥接上限 |

这些配置可通过 PlannerConfig 构造指定。改变阈值需实测，不建议照抄测试中的极小门槛作为生产默认。cpu_budget 是 CPU 并发上限，不是承诺一定启动该数量的 worker。

### 查看计划，再执行

```python
import numpy as np
from calmetrics_engine import GraphCompiler, AdaptiveScheduler, PlannerConfig

graph = GraphCompiler({"x": "series"}).compile("mean(x)")
x = np.arange(32, dtype=np.float64)
start, end = np.array([0, 16], np.int64), np.array([16, 32], np.int64)
# 教学示例：降低线程门槛以观察 thread；不代表该小样本并行更快。
config = PlannerConfig(thread_work_units=1, process_work_units=1e100)
with AdaptiveScheduler(cpu_budget=2, config=config) as scheduler:
    plan = scheduler.plan(graph, {"x": x}, start, end)
    print(plan.metadata()["reason_codes"])
    result = scheduler.execute(graph, {"x": x}, start, end, plan=plan)
    assert result.plan.lane == "thread"
np.testing.assert_allclose(result.values[:, 0], [7.5, 23.5])
```

同一进程内的图、直接算子和 cal_* 共用原生 CPU 准入。创建多个 Scheduler 不等于获得多个互不约束的 CPU 池；cal_* 的 n_threads 也只表示请求上限。

## 4. 进程、共享内存和复制边界

**自动共享运输条件**：已经选择 process，且满足 hard_stop、输入已是 SharedInputBundle/ModelPayload、输入字节数达到共享阈值，或任一分块的完整结果响应超过 256 MiB IPC 单帧上限。结果检查包含容量、逐元素状态、实际形状、根状态和协议头；不能只看输入大小。降低进程数后重新检查分块与内存预算。共享路径的状态/形状响应仍须在单帧上限内，否则规划阶段报错，调用方应缩小区间批次。单独超过 8 MiB 不会强制多进程，也不保证普通单线程请求新建共享内存。

小型 process 请求可使用显式计量的 inline IPC。大输入或 Hard Stop 路径由父进程持有共享映射、worker 只读附加；worker 将不同结果块写入不重叠区域。共享输出返回的 NumPy 视图持有原生 owner，无最终整块结果复制。typed 多输出按“区间→根”使用 8 字节对齐的容量槽，每个 worker 写互不重叠的区间块；每根实际形状和错误状态另行回传。Planner 计入容量、状态和描述信息，不能只用所有根的统一 dtype/长度估算。

| 环节 | 复制含义 |
| --- | --- |
| 普通 ndarray → 原生视图 | 符合入口契约时不复制 |
| 普通连续 ndarray → SharedInputBundle | 一次明确的共享存储初始化复制；多次调用可复用 |
| 小输入 inline IPC | 存在序列化/运输复制，不能宣称全路径零拷贝 |
| 分位数排序、线性求解 | 必要算法 scratch 初始化复制 |
| 共享输出 → NumPy 视图 | 原生 owner 保活，无最终数值数组复制 |
| statuses 结果/进程运输 | 独立计量，不能被数值零拷贝口号忽略 |

SharedInputBundle.from_inputs 接受 C 或 F 连续输入，并保留物理字节顺序；两者都是连续内存。SharedArrayOwner.from_array 同样保留布局，empty(shape, dtype, order="C") 或 order="F" 可指定布局。descriptor.order/strides 描述实际存储，attach 按同一布局建立视图。两种布局都成立时（标量、向量、空数组和部分单元素轴）from_array 归一为 C；空数组使用零 strides。负 stride、补边、广播等非稠密输入仍不能直接建立 bundle，可通过普通 typed 输入绑定使用既有打包路径。bundle 的输入映射初始化后物理只读；普通 SharedArrayOwner 默认可写，attach 默认只读，不能将只读 bundle 提升为可写。不可在执行未结束时释放共享输入；返回的独立结果在关闭 scheduler/bundle 后仍有效。

```python
import numpy as np
from calmetrics_engine import GraphCompiler, AdaptiveScheduler, PlannerConfig, SharedInputBundle

x = np.arange(32, dtype=np.float64)
graph = GraphCompiler({"x": "series"}).compile("mean(x)")
config = PlannerConfig(process_work_units=1, min_rows_per_worker=1,
                       max_processes=2)
# 仅演示可控进程/共享路径；不作为性能配置建议。
with SharedInputBundle.from_inputs({"x": x}) as bundle:
    with AdaptiveScheduler(cpu_budget=2, config=config) as scheduler:
        result = scheduler.execute(graph, bundle,
            np.array([0, 16], np.int64), np.array([16, 32], np.int64))
        assert result.plan.lane == "process"
        assert result.audit["use_shared_memory"]
        assert result.audit["output_copy_bytes"] == 0
np.testing.assert_allclose(result.values[:, 0], [7.5, 23.5])
```

worker 是随同 wheel 分发的独立 C++ 可执行文件，不启动 Python。当前请求26/响应25；mode word的bit0表示共享运输、bit1表示原MatMul几何。不匹配协议在解码前拒绝，未知位拒绝；同一worker缓存逻辑图仍逐请求读取策略。当前请求在既有 dtype 字中记录 C/F，连续 F 输入不再按 C 顺序重新排列；非稠密float64二维视图在第一轴反向连续、列距为正且不重叠时按逻辑列打包成 F，其余非稠密输入仍打包成 C。既有打包次数和容量不增加。共享与 inline 运输均恢复实际布局，数值 dtype/shape 不变。父进程把已解析的输出顺序编码入 worker 计划；响应在原 rank 字高位声明 C/F，不增加逐位置帧字节。父进程校验返回顺序、rank 和容量后按实际 strides 创建独立结果视图。协议版本必须匹配；不能从另一版本随意拷贝 worker。执行环境需要允许创建进程和共享映射，资源/worker 错误直接失败，不回退到 Python。

## 5. 协程：由调用方选择

execute_async 用 asyncio.to_thread 桥接阻塞的完整原生请求，数值循环不在事件循环里。execute_many_async 用 Semaphore 限制本次批次的桥接并发，并按输入任务顺序返回结果；原生 CPU 预算仍统一控制所有批次和同步请求。

```python
import asyncio
import numpy as np
from calmetrics_engine import GraphCompiler, AdaptiveScheduler

async def main():
    graph = GraphCompiler({"x": "series"}).compile("mean(x)")
    with AdaptiveScheduler(cpu_budget=2) as scheduler:
        jobs = [dict(graph=graph, inputs={"x": np.array([v, v+2.])},
                     starts=np.array([0], np.int64), ends=np.array([2], np.int64))
                for v in (1., 3.)]
        results = await scheduler.execute_many_async(jobs)
        np.testing.assert_allclose([r.values[0, 0] for r in results], [2., 4.])
        assert all(r.audit["async_orchestration"] for r in results)

asyncio.run(main())  # notebook / 已有事件循环中改为 await main()
```

异步适用于 Web 服务或需要并发等待的应用，不是数学提速开关。取消 awaiter 不会强杀已经进入原生计算的工作；输入所有权、scheduler 和资源必须保持到实际工作结束。

## 6. 超时、Hard Stop 与结果保存

普通线程超时是协作式边界检查，正在运行的长内核不能被安全强杀。需要硬截止使用 `execute(..., hard_stop=True, timeout=秒数)`；原生运行时采用隔离的一次性进程，失败/超时终止并等待 worker 后再释放资源。仅传 timeout 不等于自动要求硬杀线程。

```python
import numpy as np
from calmetrics_engine import GraphCompiler, AdaptiveScheduler

graph = GraphCompiler({"x": "series"}).compile("mean(x)")
with AdaptiveScheduler(cpu_budget=1) as scheduler:
    result = scheduler.execute(graph, {"x": np.array([1., 2., 3.])},
        np.array([0], np.int64), np.array([3], np.int64), hard_stop=True, timeout=30)
    assert result.plan.lane == "process"
    assert result.audit["hard_stop"] and result.audit["use_shared_memory"]
assert result.values[0, 0] == 2.0
```

typed 的输出形状可随参数/数值变化，但必须落在计划容量内；prepared 复用时不缓存旧实际形状。typed run 返回带状态的 Result，取值见[使用手册](user-guide.md)。

prepared 当前只接受 single-lane 计划，是稳定输入/几何的重复执行入口，不是并行 execute 的通用替代或流式状态容器。需留存结果用 run_snapshot；借用输出会被覆盖，参见[使用手册](user-guide.md#9-重复执行与结果所有权)。

普通 execute 在同一引擎的输入绑定缓存中复用计划；每次仍复核输入布局、区间/产品内容及 CPU 限制。内存预算、Hard Stop 或异步策略变化时重新规划。普通入口的区间几何变化会重建计划；用户显式提供的旧计划和 prepared 的几何变化仍报错。计划复用不缓存结果，每次结果及执行审计均独立。

隐式绑定缓存只保留布局描述；本次执行以强引用保活实际校验的数组至原生任务结束和结果返回。动态 mapping 不为保活而再次取值，缓存替换不释放正在执行的输入。显式 prepared 保留原强绑定，详见[使用手册](user-guide.md#9-重复执行与结果所有权)和[输入缓存设计](mc-axis-primitives.md#11-普通输入缓存与执行所有权)。

缓存计划在同一次请求内只做一次完整几何校验；校验证明仅供本次原生执行消费，不跨请求复用。引擎 CPU 上限在构造时确定，执行仍获取本地和进程全局两层 CPU token。prepared 的借用输出延迟到首次需要时分配，独立快照直接写新结果。

## 7. 如何判断本次执行是否符合预期

- plan.metadata()：lane、reason_codes、parallel_dimension、chunks/branch_tasks、预计 worker/内存和 simd_nodes。
- result.audit：实际 cpu_tokens、native_threads/native_processes、排队/计算耗时、复制字节数、共享输出所有权及 native_chunks。
- native_chunks 新增 fused_pointwise_calls、pointwise_tiles、vector_elements、root_copy_bytes、state_copy_bytes、direct_output_bytes 和 pointwise_workspace_capacity_bytes。向量计数仅包含显式 SIMD 实际处理的“算子／扫描 × 元素”总数，同一元素可计多次，不包含未经核实的编译器自动向量化；没有逐节点 ISA 明细，不能据此宣称每个候选节点均向量化。容量不是本次新分配字节数，direct_output_bytes 是计算写入量，不属于复制。
- engine_build_id、plan_fingerprint、registry/IR 版本和 input_dtypes/output_dtype 用于追溯。凭据不是外部认证协议。
- estimated_total_memory_bytes 是输入/输出/运输/arena/scratch 估计，不是 OS RSS 上限；峰值 RSS 另行测量。

性能排查顺序：先验证公式与区间 → 看复制和布局 → 看共享依赖是否复用 → 看计划分块/排队 → 最后调整阈值与 CPU 预算。重复固定任务考虑 prepared；需要保留结果时比较 run_snapshot，不能拿复用裸缓冲的耗时冒充独立结果成本。

当前性能验收口径见 [AGENTS.md 第 16 节](../AGENTS.md#16-performance-evidence) 和 [D8 完整负载规则](computation-design-rules.md#10-d8用对等的完整负载证明收益)；[原有对照工具](../tools/check_phase2_performance.py)保留历史严格判定。几 KB 的内存增加、几毫秒的耗时退化不能单独作为阻断条件，须结合绝对差值、相对变化、真实批量/并发累积及资源预算判断。本页示例验证功能路径，不提供性能提升或跨平台运行证明。


## 8. M0/M1 开发分支的执行边界

M0/M1 初版将原生图和 worker 协议升级为 7。当前 Program 使用 v8，全部 C 输出可保留 v7 编码；worker 已为 v22，必须与当前 wheel 同构建，详见[平台执行契约](platform-execution-contracts.md#native-interval-bindings)。旧图定义通过重新编译获得新计划。rank-3 的 time 首轴参与区间裁切，静态轴保持整块；空张量保持已知空形状。不同 dtype/shape 的具名输出继续共享一个带容量上界的结果缓冲。

ModelPayload 初始化有一次独立持久化复制，使用 8 字节对齐字段；稠密 C/F 字段保留各自顺序，非稠密字段按逻辑 C 顺序打包。含真正 F 字段的载荷使用 v2 编码和 model-2 身份；全 C 载荷继续使用 v1 编码和原身份规则。旧 v1 解码保持兼容。Planner 计入包含对齐填充的载荷字节数；共享进程路径按一个模型区域复用只读映射，不能按字段重复计算同一映射字节数。请求/响应携带模型内容身份，父进程校验后接收结果。模型身份不是安全认证或金融适用性证明。

有界迭代的最坏迭代次数、子图工作量、双状态缓冲及诊断纳入计划。每轮检查计算预算与进程内取消/截止；强制硬停止仍使用隔离进程。通用解释执行的子图开销及中间复制必须在基准中计入，不能仅凭 C++ 实现宣称快于 NJIT。

新增协议负载由 `docs/platform-foundation-workloads.json` 固定，运行 `tools/check_foundation_performance.py --output <报告路径>`；旧四组门禁继续运行 `tools/check_phase2_performance.py`。脚本保留配对原始耗时、单侧 95% 上界和隔离进程 RSS 样本；历史严格规则会将零增量或少于两页的内存优势标记未证实并失败。保留这些诊断，当前采用结论按[完整负载验收规则](computation-design-rules.md#10-d8用对等的完整负载证明收益)判断，微小字节或耗时差异不单独阻断。RSS 采样不能替代完整原生分配追踪，也不能代表所有平台算法已验收。最新状态见[开发设计](platform-foundation-design.md)。

## 9. 本轮物理执行优化与审计

完整逐元素 typed 图可融合 add/subtract/multiply/minimum/maximum/negate/absolute、数值比较和 finite_mask；数组需具有同一形状，标量可广播。不支持的 dtype、含数值域异常的运算或错误隔离图继续使用通用执行器。逻辑 DAG 与公共算子不合并成新的业务黑盒。

一个数组输入串联最多两个 add/subtract/multiply、末尾可选比较或 finite_mask 时，使用预编译寄存器链；常数仍在执行时绑定。中间值留在寄存器，仅根写结果；超出该范围的兼容图继续缓存块执行。保持操作及舍入顺序，不使用 JIT 或乘加收缩。该短链支持连续、负步长、零步长及按步长取数的 SIMD，并合并可连续遍历的相邻轴；没有为 SIMD 打包输入。NEON 将布尔结果批量压缩成连续的 0/1 字节。

分段融合的参数每次请求刷新，多块任务在块循环外绑定，单块任务直接绑定而不保留缓存；被消除的中间节点不占物理缓冲槽编号。单 chunk 审计内联保存，多个 chunk 使用动态存储；形状/状态按实际数量分配，张量并行任务的共同形状描述只保留一份。相关保留容量与独立结果仍有内存成本，不能将减少 malloc 次数等同于 RSS 或申请峰值下降。

具备独占槽位的 typed 根算子直接写结果，后继节点读取该槽。借用输入、重复根和其他需要物化的输出仍明确复制。一般迭代以结果槽和候选缓冲交替写入，保留最后有限状态；候选块的有限性检查与残差归约使用 SIMD。

初态有限，且更新仅由最多两次绝对值不超过 1 的有限系数乘法或加减零组成时，可证明状态始终有限。此时复制初态到独占结果后原地更新，在覆盖前读取旧值计算残差；不会改写输入或上次结果。首次执行完整校验，后续仅在同一迭代作用域复用固定描述，保留每轮预算与逐块取消检查；下一次请求重新检查系数。其余子图保留候选缓冲。Planner 仍保守计入双状态容量，不能把估计值当成实际分配量。普通快照依旧独立，prepared.run 的借用结果规则不变。

执行复制计数与独立 malloc 追踪、RSS 采样共同使用。`tools/heap_probe.cpp` 是仅供 macOS 开发验证的注入式探针，不进入生产 wheel；覆盖指定 malloc 家族调用的申请字节与存活峰值，不把 Python arena 内部分配、mmap 或子进程冒充已覆盖。详见[设计及验收](physical-execution-optimization.md)。

同预算对照使用 `tools/check_execution_scaling.py`，同时测已预热的 NJIT 串行与并行，取该 CPU 预算内更快的路径；不强制短任务用更慢的 NJIT 并行。原有固定单核对照仍须单独测量并保留严格诊断；当前采用结论按 D8 判断，不能仅因微小差异触发历史失败就否决候选。

矩阵与数组新增原语的内核不创建私有线程池；`cholesky/solve_multi/triangular_solve/symmetric_eigh` 内部单线程，批次由原生 Scheduler 统一分配预算。多个 RHS 共用一次消元，特征值/向量投影共用分解。slice/swap/broadcast 为视图，但普通根结果仍需独立存储；零拷贝视图不代表整条图零分配。上述库旋转和消元可使用Eigen内部SIMD，其指令不计入引擎的 `vector_elements`。初始验收见[矩阵与数组验收](matrix-array-foundation.md)，后续库优化及未关闭项见[物理执行优化](physical-execution-optimization.md)。

严格typed/raise图中，输入或稳定借用视图支持的私有 `matmul(A[N,K],B[K,T])`，若全部直接消费者为axis1的sum/mean/variance，且完整下游只含正有限常量缩放、至多一次sqrt及float64向量根，可共享行统计。共享几何要求32≤N,T≤1,000,000、K≤128；无scope、observer、isolate、继承/延迟故障或prebound子程序。macOS15+ Accelerate在有限性、尺度、舍入方式及误差放大证明通过时，用512观测面板和1024候选块计算公共均值/中心化Gram及二次型，保留实际C/F输入；不确定时在同一有界预算内按现有canonical顺序回放。私有N×T不物化，参数及数值错误仍在原逻辑消费者检查；公开明细或未证明的比较/排序/状态消费者保持原矩阵。每次重新读取输入，不缓存数值；统计缓存、辅助区及1MiB厂商资源估计计入Planner，工作量仍覆盖最坏回放。已有库调用/复制/arena/workspace审计贯通四运输与独立结果；逻辑AST/DAG及Excel公式/预算不变，收益与仍未优于NJIT的峰值边界见[正式验收](physical-execution-evidence/2026-10-05-candidate-summaries/integration/README.md)。

同一私有矩阵组包含 `quantiles_axis(Y,q,1)` 时，也可使用有界轨迹批次；概率须来自已检查的借用视图或静态向量，每个向量最多128项，全部分位数下游只含 negate、正有限常量缩放及float64矩阵根。统计和全部概率共享一次逐行排序。批宽为 `min(64,N,max(1,163840/T))`，长历史至少保留一条完整轨迹；内部摘要缓存为 `N*(2+sum(Q))`，达到原轨迹大小时不准入。符合数值域、右矩阵正C/F leading dimension、舍入和误差条件时使用受单线程预算约束的Accelerate GEMM，否则在同一有界空间回放原数学顺序。非法概率、短样本及独立错误仍在原逻辑节点检查；输入只读、结果保持指定连续C/F布局及独立所有权。Planner辅助上界同时覆盖较短区间可能增大的批宽；厂商预留仍计入预算。没有公开新算子、协议或Excel公式变化；实测范围、申请峰与RSS盲区见[多分位数设计与验收](physical-execution-evidence/2026-10-05-candidate-summaries/quantiles/design.md)。

同一 typed/raise 图恰好包含两个终端 `solve_multi(A,B1)`、`solve_multi(A,B2)` 根，且 A/B1/B2 是静态矩阵输入、两根实际返回 C、求解节点之间只有静态输入绑定时，可共享现有 Eigen 消元。macOS15 arm64、Eigen开启、N≥128准入；F输入和受支持strides按原逻辑读取，显式F输出、scope、isolate、observer和prebound子图保持原节点执行。每个RHS的运算顺序不变，先完整回代第一根再回代第二根，成功后才发布第二值；失败恢复第二结果区并按原顺序重放。结果各自独立，复制审计减少一次 `8N²` 的A初始化，但旧节点本已顺序复用同一N²工作区，不能据此宣称峰值减半。prepared每次重读当前A/B，不缓存数值分解；进程解码后重建共享资格。Planner保留原保守预算/工量，typed不走dag_branch；描述性branch数不代表实际消元次数。

typed/raise且无子作用域的图，若A直接来自非公开、仅被solve_multi消费一次的add/subtract/multiply或concatenate物化节点，至少1,048,576 doubles且运行证明为本次独占C连续arena，可在C求解阶段复用A进行同一消元，省去第二份N²工作区及A初始化复制。仍先复制/验证B，再完整验证A；输出与RHS的重叠区、输入/视图/共享或公开系数、观察器、故障隔离及延迟错误保持原路径。拼接系数进入成熟LU的资格另按下述测量重构路径证明；显式F结果仍按声明独立返回。每次重新生成系数，不缓存分解，不写输入或旧结果；适用于已验证的prepared、线程和进程路径。`algorithm_copy_bytes`与`operator_workspace_capacity_bytes`反映实际复制及高水位容量，前序其他算子的容量可能保留；Planner继续保守覆盖原scratch。数学和Excel逻辑不变，验收边界见[私有系数工作区](physical-execution-evidence/2026-10-04-private-solve/README.md)。

严格 typed 图还可消除私有嵌套 `concatenate` 的中间数组：内部拼接必须为 float64、非公开且只有一个消费者，axis 为常量，所有末端叶子须由本次输入或其稳定借用视图支持；程序自身不含子作用域。每个逻辑节点仍在原位置检查类型、轴和形状，最终按原坐标将叶子直接复制到同一个 C 目标，正向 F 矩形以32×32分块读取。macOS15+ arm64、Eigen开启且两维至少32时，使用无堆分配的Eigen列/行主序Map将数据逐位搬到原目标；不改变输入布局、矩形资格或结果所有权，其他平台/库关闭保留原生分块复制。显式 F 结果沿用原独立 F 返回流程，不要求调用方转换输入布局。公开/共享/计算叶子、观察器、延迟错误及故障隔离保持原物化；合格的平面子程序仅在该次调用的稳定输入上使用。Planner与执行复用同一deferred证明，`numeric_arena_bytes`反映实际中间区减少；逻辑DAG、Excel展开及预算不变。[接入与完整链证据](physical-execution-evidence/2026-10-04-general-solve/concat-private/README.md#嵌套拼接融合已接入修正版完整验收与正式复测结束)说明范围与仍未关闭的求解耗时差距。

typed/raise、无子作用域的图可在 macOS15+ arm64、Accelerate 开启时使用受限 LAPACK LU：1024–4096 阶、1–64 RHS、独立终端 float64 矩阵根，支持 C/F 结果。私有 add/subtract/multiply 系数的直接父须是持久叶节点；直接输入或公开系数先复制到独占工作区，保留原值。运行时验证所有权、有限值、有界尺度、严格行对角占优及当前 LU 因子的原路径证书，避免只凭库 INFO 跳过原 1e-14 奇异阈值。相同 dense F 父用 canonical 逐元素操作直接写 F 私有系数；C 私有系数原地分块换 F。失败从原父重建或重新读取原系数，再执行唯一 canonical 求解，不写输入或旧结果，不缓存数值分解。

LU 以固定 512 列面板调度成熟 DGETRF2/DLASWP/DTRSM/DGEMM；证书和 DGETRS 完成后，失效的独占因子区用于 64×N 残差面板及 3NK 乘积缓冲，解与证书工作区为 NK+5N，不分配第二份完整系数。显式工作区及库资源估计由原保守 Planner 预算覆盖。`accelerate_lu_attempts/calls/fallbacks` 分别记录实际库尝试、成功和尝试后的回放，`solve_residual_materialization_bytes` 记录累计原系数面板物化；贯通直接 audit、native_chunks 与 IPC24。面板写入不等于输入复制，C 原地换位仍如实计入算法数据移动。库 reserve 是估计，不能代替实测 heap/RSS。库线程在调用线程内限定单线程并恢复；其他平台、未准入形状/布局或非对角占优矩阵、下游消费者、scope、isolate、observer及延迟错误保持原路径。浮点尾差按当前许可验收，数学、错误和独立结果契约不变；[设计与各阶段证据](physical-execution-evidence/2026-10-04-general-solve/README.md)区分旧隔离结果与当前源码复验。

严格 typed/raise、无子作用域且无观察器/隔离/延迟故障的图，输入支持的私有拼接系数可使用受护 LU：macOS15+ arm64、Accelerate开启、1024–8192阶、1–64 RHS、独立终端float64矩阵根。这是可选优化范围，不限制公开求解形状。运行证明系数为本次独占arena、单消费者，RHS/输出不重叠且稳定输入叶子可以同步再生成原系数矩形。普通入口仅在全部叶子正向C/F或pitch布局可分类、且F叶子消费字节占优时直接构造F系数；C占优、并列、未知/逆序/广播步幅或非最近舍入保留原C构造。调用方无需转换输入。声明C/F结果均保持独立所有权；internal prebound排除，公开prepared snapshot/audit沿普通executor每次重读数据，不缓存数值分解。

当前常规FP64分块GEPP在限定数值域及最近舍入下，以保守componentwise后向界初始化局部误差，不再重构LU或执行QR。其后完整核验原路径主元间距、有限性和原方程残差；不确定时从持久叶子恢复C系数并canonical回放，原1e-14阈值及B-before-A错误顺序保留。该模型不能自动推广到Strassen、降低精度或新后端，也不是病态问题的普适前向误差保证；前述DD路径对非对角占优矩阵的排除不能套用到本路径。

有界列组、连续64N正包络和依赖安全的批量正误差传播复用同一工作区；失效因子区承载最终残差面板，不增加第二份完整系数。健康路径完整读取原系数一次，`solve_residual_materialization_bytes=N²*8`；F私有构造省去原整矩阵转置，C路径仍如实计入转置，均不等于整条请求零复制。实际scratch、Planner上界及库资源估计分别审计，库预算不是实测heap/RSS。完整LU、所有原条目及主元/残差检查保留；[当前设计、同源资格及完整链复验](physical-execution-evidence/2026-10-08-general-solve-v68/README.md)区分隔离比较与实际目录结果，并保留其他主要负载欠项。

恰好两个终端 solve_multi 根还可共享私有 add/subtract/multiply 系数的受护 LU：系数只被这两根消费、无公开系数或其他消费者、父是持久叶节点、两个 RHS 是静态输入，两次求解之间只有静态输入绑定。两根 RHS 总列数须为 1–64 且各自非空；先合并 RHS，一次分解和求解，证书及原方程残差通过后直接写两份独立 C/F 结果。合并原 RHS 与求解缓冲共 2NK+5N；失败重建系数后按原顺序执行两根。prepared 每次读取当前数据，进程解码重建资格，不缓存跨请求数值分解；前述静态输入系数的 Eigen 双根路径继续遵守其 C 输出限制。Planner 仍覆盖完整 canonical 回放预算，单行 `accelerate_lu_calls=1` 才是本次共享成功的运行证据。KKT 等非对角占优矩阵不在这条成熟 LU 准入域内。


Cholesky在macOS15+arm64、Eigen开启、自动ISA、n≥1024时复用Eigen Array/Map，将16行面板的8步更新合并写回。因子完成后，每4个独立行用固定8列数组共享源读取；只更新共同三角前缀，各行不同长度的尾部继续原路径。每个元素的升序乘减、除法和平方根顺序不变；固定数组没有新增数值堆工作区，调用栈另行计量。输入按原C/F或任意受支持步长读取，现有独立C结果不改；行面板与因素表上界为 `(16*(n+17)+128)*8` 字节。Planner与准备/执行使用同一上界，顺序节点复用工作区，变阶时释放预算外旧容量，子作用域/worker分别预算。该路径审计ISA为 `eigen-array-panel-cholesky`，算法复制与绑定零拷贝分别计数。数值失败按原行序重放以保持错误优先级；边界和资源异常不捕获。其他平台、Eigen-OFF、小矩阵与显式ISA保持原路径。

独立明确 F 的单一 typed float64 Cholesky 根可直接写独占结果：原算法完成时三角因素已有逐位镜像，仅改变成功末尾清零的物理三角，公开逻辑下三角和原数学顺序不变。无需另一份 C arena 或最终整矩阵转换；此存储优化不依赖面板库准入。多根、消费者、isolate 和作用域保留原路径；私有 covariance→Cholesky 融合另按下文复用 F 根缓冲。observer/prebound/继承错误恢复原 C arena。Planner 对 prebound 子图计入该降级的完整矩阵，直接算子 F out 仍遵守原拒绝契约。

单个 `median/quantile` 及对应 masked 算子在至少1024个消费值、不含NaN且没有混合正负零时，复用原工作区选择排名。Apple arm64、Highway启用、自动ISA且运行时支持float64时，调用静态编入的 Highway 1.3.0 `VQSelect`，直接审计 `isa="highway_vqselect"`；显式ISA（含scalar）、库关闭或其他平台继续标准库 `nth_element`。库仅使用当前线程，没有私有线程池或跨请求数值缓存，首次线程状态/栈开销计入完整内存验收；不把消费值数量冒充精确SIMD指令处理量。`CALMETRICS_ENGINE_USE_HIGHWAY=OFF`可关闭此实现，开关纳入构建身份；sdist包含固定校验和的上游源码，wheel包含许可证，无运行期下载或编译。图与进程沿既有算子入口执行，不新增IPC字段，direct ISA标签不在图审计中单独传播。需要相邻排名时扫描较短的一端：选上排名后用 `max_element` 补下排名，高分位数则选下排名后用 `min_element` 补上排名。原中位数和插值表达式不变，输入仍只读。其他消费集继续原排序，多个图根仍共享完整排序；因此不能把单指标提速外推为所有多指标图的提速。掩码算子在已有完整结构校验中重计本次选中数量，实际工作区仅请求该数量；准备/Planner仍按源数组上界预算，足够的历史容量继续复用，显式reserve或密集转稀疏不承诺释放旧容量。计数不跨执行缓存。C连续且非标量的输入（有掩码时掩码也须C连续）直接按逻辑顺序读取；其他步长继续通用访问。掩码校验保持原顺序，打包时先判断是否消费，再读取该值，并复用实际消费计数填写算法复制审计，省去审计专用的重复扫描；不改变NaN、正负零分类、排序及插值规则。

`cholesky(covariance(X))` 的 typed/raise 图，在协方差是紧邻的单消费者临时节点、不是公开根且源有持久后备时，可复用消费方独占缓冲。两个既有原生算子及逻辑 DAG 都保留；C 根直接写独立结果。唯一 typed float64 矩阵根明确要求 F，且无消费者、apply/rolling scope 时，复用本次独立 F 结果缓冲暂存 C 计算；既有 Accelerate、canonical 和失败重放均保持原 C 数学路径。成功后以有界索引块原地换位并发布 F strides，不新增数值缓冲。换位写入的 `D*(D-1)*8` 字节计入 `algorithm_copy_bytes`；根复制为零不代表整个算法没有搬运。auto 布局和直接算子 F out 契约不变。Cholesky 行面板发生数值失败时，先重算纯协方差恢复输入，再按原 canonical 顺序重放；前置校验、资源错误和普通直接算子的 out 别名限制不变。isolate、值观察器、共享协方差及不满足依赖证明的图不使用此融合。prepared 借用值仍仅有效至下一次运行，独立快照不受后续失败影响。

该融合通过实际 `numeric_arena_bytes`、工作区和算法复制审计核验；不计为 `fused_pointwise_calls`。小矩阵原地分解跳过自复制，面板的两次下三角搬运仍照实计数。符合 F 独立根资格时，普通执行不需要第二块根矩阵；观察器、prebound 和继承错误路径会取消直接目的地并恢复 C arena。Planner 对 prebound 子图采用同一判据保留该矩阵预算，面板和库工作区上界不删减；不能只凭 F 标签省略容量。完整请求实测及剩余欠项见[物理执行优化](physical-execution-optimization.md)。

概率模型内核同样不新建线程池。GMM 通常复用 N×K 后验，符合下段准入时改用有界重放；HMM 复用 K×K 转移工作区；大形状通常只分配两块 N×K 历史，消费 emission 后覆盖成 beta 并重放小块密度，图若已有可免费借用的第三块当前缓冲则保留完整历史，小形状保留三历史路线，平滑借用独立根存放前向值并使用 K 级后向缓冲。多输出参数投影共享一次拟合，prepared 重读当前数值，不能按地址复用旧模型。Planner 将迭代轮数和 N/K/D 纳入成本，保守覆盖工作区；malloc/RSS 实测与估计分开。新算子的显式 SIMD 目录仍为空，不能宣称矩阵运算自动获得 SIMD 优势。见[时间与内存实测](latent-model-operators.md#验收结果)。

2026-10-07，Apple arm64、macOS 15+、Eigen开启，GMM 的 K≥16、8≤D≤32、N×K≥16,777,216，且有界存储小于N×K时，复用现有Eigen密度与有序加权归约，只保留128行后验、旧均值、持续累加量和必要列打包。工作区为 `K*(128+2*D+3)+128*D` 个double；小样本、K<16、D范围外、库关闭及其他平台保持原完整后验。形状准入也用于scalar工作区，显式scalar仍按同序标量数学执行，不能据此宣称scalar有同样的提速。E步重放严格使用本轮旧权重/均值/方差，方差仍围绕新均值做两遍计算；不使用原始二阶矩相减，不改变下限、严格停止、诊断或公开逻辑图/Excel公式。列输入可直接借读；其他合法stride只打包当前128行，真实pack、旧均值保存及和的发布分别计入 `algorithm_copy_bytes`，绑定零拷贝不等于全算法零复制。

GMM的Planner估计覆盖最大形状及阈值以下最后一个完整后验形状：最大区间选择重放，较短区间仍可能需要N×K，不能只按小tile准入。实际执行不因此分配未用的完整后验。重放形状跳过旧N×K借用几何；其余GMM/HMM继续既有免费arena借用。多输出共享本次拟合，普通结果和快照保持独立，CPU/进程协议不变；成本单位仍为既有路由估计，不是实际FLOP计数。新路径的整链内存仍未全面领先NI，当前范围与剩余问题见[本轮证据](physical-execution-evidence/2026-10-07-gmm-direct-replay-v56/README.md)。

HMM 的转移统计在非 scalar、N≥128、K≥3时交错处理最多128个独立时间行。每行仍按原source/target次序计算同一 `std::exp/std::log1p`，每个转移单元仍按时间升序累加；前后向递推、停止判断和数学契约不变。这是私有调度优化，不是新的SIMD或库路径。形状对应工作区增加 `min(N-1,128)` doubles，在取得任何指针前一次预留；显式scalar也保守预留该上界，但实际按单行执行。无新全量矩阵、输入复制、缓存或线程。普通/prepared完整链路实测、堆峰与RSS的不同结论见[本轮证据](physical-execution-evidence/2026-10-03-hmm-xi-integration/README.md)，不外推为全平台验收。

大样本拟合的普通 typed 严格入口，full-posterior GMM 在 NK≥1,048,576 且原 scratch 包含完整 NK 时，优先复用本次未写入且通过容量、边界、对齐、writer 和不相交证明的独立 float64 `scatter_rows` 矩阵根。它不保存 arena 输入、不增加 E 步，后续 scatter 仍按原协议写入。bounded replay 保持原路线。arena 历史借用的最低 NK 为 HMM 131,072、GMM 2,097,152；这只是物理准入，原活跃值保存、模型矩阵数量、工作区及所有权证明仍须通过。未来根借用的前缀在正常返回、异常或取消时恢复原零值，再由其逻辑节点写入；独立故障不能暴露拟合暂存。两类路线均不借输入或旧结果，仍排除子作用域、isolate、观察器及公开 in-process Prepared。Planner 保留 canonical 工作区上界，保存/恢复和真实复制照实审计。数学、算子及公开结果协议不变；内部请求协议为 26，响应布局为 25，parent/worker 必须来自同一构建。

严格 typed 单行图中，已验证的正向连续行索引可使私有 `take_axis(axis=0)` 借读 C 布局输入或已完成的本次独立矩阵根。相同依赖、相同算术树的私有训练标准化可借读更早完成的完整标准化根，原节点准备与错误时点保留。当前检查树深度最多 12；未知索引、布局、观察器、scope/isolate 等不满足证明时恢复原物化。索引、owner、几何每请求验证，不能跨请求缓存数值。新增每节点 proof 和固定预留若会使原本可准入的请求超预算，Planner 关闭该优化，采用原行几何；不能降低原预算保证。

独立 int64 状态根可借给其私有 `argmax_axis(axis=1)` 前驱，但源只能被已证明的逐位置状态决策闭包消费。长度和物理共用槽节约均至少 131,072 项；仅有长序列不足以准入。每个 4,096 项 tile 先保存原标签，再执行同一比较和 state_select，32 KiB 标签暂存及复制进入审计。失败或取消保留已完成前缀并清零未发布后缀，输入、旧结果、其他根不借用。这里复用的是本次独立结果，不改变公开结果所有权。

严格typed图中，若私有 `hmm_filter` 的唯一消费者是常量 `slice_axis`，沿axis0、step1恰好选择 `[N-1,N)`，且无子作用域、N≥3、N×K≥1,048,576，引擎可只物化最后一行。仍在原filter节点扫描全部emission、检查initial/transition概率，并执行同一个有序行递推；两个K状态进入独占Workspace与Planner预算，最后K值复制到原arena slot，真实切片保持原借用寿命。完整历史的公开输出、额外消费者、小样本、未知几何、isolate、观察器及故障诊断保持canonical路径；最后一行自身可公开或被多个下游消费。输入和旧独立结果不会作为scratch，没有跨请求数值缓存、新线程或新算子，逻辑Excel公式不变。[本轮完整链及内存证据](physical-execution-evidence/2026-10-04-hmm-filter-tail/README.md)。

## 按轴扫描与紧凑消费者执行

`cumulative_product_axis`、`cumulative_max_axis` 和 `quantiles_axis` 保留原逻辑轴和合法 signed strides；prefix 直接写独立 C/F typed 根，多分位数每条 line 仅保留 N 项排序工作区并共享全部 Q。算子契约及完整图边界见[按轴原语设计](mc-axis-primitives.md)。

已实施的私有紧凑执行从完整可编辑 DAG 证明投影、扫描、排序和小结果的全部消费者闭包。Planner 与执行共享所有区间长度的几何准入；不符合闭包时按完整 canonical 内存预算执行。准入后只保留分批当前状态和有界投影工作区，fan/样本直接写本次结果；后段统计使用独立 child Scratch 调用原算子。普通 C++ 不生成 Excel，Excel 仍展开原逻辑图。此前从已变换固定带开始的实测及其单组合、短周期 owner 内存欠项，保留在[按轴原语与验收](mc-axis-primitives.md)；不能当作新增原始带前缀的验收结果。

当前还可从原始 normal 带及 Student chi 带证明私有前缀：`affine_contract` 从 +0 保序旋转，随后乘 1 或显式 Student scale，再由 means 左加，最后进行资产 `affine_contract` 与 `expm1`。这些仍是可编辑的逻辑节点；不把 means 放入旋转初态，不合并 root×beta，也不新增业务黑盒。前缀与消费者的完整依赖闭包必须私有；公开中间根、额外消费者、isolate、观察器或未支持作用域继续 canonical 图。每个 interval 的 Student scale 依原拓扑完成 canonical 检查，保留 CME 自身的定义域和错误顺序；这不等于复现平台负 chi 的 IEEE 行为或按业务阶段选首错。

设区间步数 T、路径数 P、变换后因子数 F、资产数 A、组合数 C，原状态批量 `B0=min(C,max(1,T/3))`（整数除法）。只有 `(C-B0)*(3*P+A)<=T*P*F` 时，才允许扩大到全部 C；B0 已等于 C 时无需扩展。不满足成本条件仍执行 canonical。通过原准入后，若 `T<C` 且完整组合状态 `C*(3*P+A)*8` 至少为16 MiB，则选择 `B=min(C,4)`，用确定性的原始带重算 producer 换取大额状态节省；其他几何仍只旋转和投影一次。每组的时间、路径、因子、资产累加顺序不变，Student scale 仍只按原拓扑顺序完整校验和计算一次。Planner 和执行使用同一有溢出检查的形状准入，按 `ceil(C/B)` 计入真实 raw/asset producer 工作量，并计入状态、旋转行、scale、系数 pack、child Scratch 及结果。16 MiB 是当前物理策略的形状分派条件，不是公开输入限制或性能验收门槛。精确 fan 仍需要当前全部路径分布，不能用小 tile 声称同时免存储和免重算。设计与验收范围见[大状态批次](mc-axis-primitives.md#12-短周期大状态批次2026-10-04)。

原始带、root、means、beta 和 chi 保持只读；支持的 C/F 和 signed strides 按原逻辑读取，不强制把整块 tape 转为 C。正常 execute 与 run_snapshot 的结果独立，prepared 借用寿命不变；工作区和必要有界系数 pack 属于本次执行，不按输入地址缓存数值。空数组容量推导现先比较 rank，再在同 rank 比较 size，避免空 tensor 被 scalar 的一个元素覆盖形状；数学广播、输入校验与错误协议没有扩张。

`compact_path_raw_rows` 记录实际旋转行数（健康准入 interval 的 `Σceil(C/B)*T*P`），`compact_path_raw_scale_elements` 在 normal 为 0、Student 为 ΣT×P，`compact_path_raw_pack_bytes` 记录实际 rotation/beta 打包字节；均通过 IPC24 返回。pack 是 `algorithm_copy_bytes` 的子计数，不能重复相加；无需打包时为 0。这些是执行证据，不能代替完整时间/内存验收。当前成功链性能参考包括双方相同的冻结 NJIT RNG 准备；该外部准备不是引擎生产随机算子或回退。NJIT 为只读重放复制带的成本不能冒充平台已独占、可原地变换的 RNG 带成本。随机源在引擎内执行、协方差 root 构造、coverage、阶段首错、limits/JSON 及原始带组合的完整 Excel 预算和 Microsoft Excel 重算仍需分别验收。

V99 在已准入完整紧凑闭包中复用 Eigen 的四个独立组合 lane。组合权重按实际逻辑 stride 每批打包，保序 period/贡献共享已舍入的权重左乘产品；不改资产累加顺序。当前几何预留包含 weight pack 与 period lane，真实 pack 字节记入 `algorithm_copy_bytes`，`vector_elements` 是执行元素计数而非机器指令数；scalar 尾批次及 IEEE/舍入/映射守卫继续既有实现。原公共算子、逻辑图与 Excel 公式不变。

双方同 CPU1，计入相同固定生产 NJIT RNG 的独占输入准备、normal/Student 变换、投影、普通 `AdaptiveScheduler.execute`、NAV/fan、全部统计/样本/贡献及两个保留的独立结果。静态夹具/构图和编译在计时外；covariance-root 生成、coverage、limits、JSON 封装和平台按业务阶段选首错不在该测量中。算法复制与 root copy 如实计入，不能称全链 zero-copy。

全部冻结调用均为普通 `scheduler.execute`，没有显式 prepared execution。Normal C1 RSS 的 post-epoch validation audit 为 `prepared_cached=true`，表示普通 execute 的输入绑定缓存命中；其余41份该验证 audit 为 false。验证 audit 不覆盖每次计时/缓存调用，不能称所有普通调用均无缓存。

`quantiles_axis` 与紧凑消费者共用数值 line 排序入口。Apple arm64、Highway开启、自动ISA、每条line至少1024值且运行时支持float64时，可调用 Highway 1.3.0 `VQSort`。每次检查当前排序scratch；含NaN、任意负零或任意非零次正规数时保留 canonical `NaNLast` 的标准库排序，正零和Inf本身不排除库路径。负零保护同时避免 ARM 开启浮点冲零时产生额外的次正规输入异常标志。不修改借用输入或调用方浮点环境，不增加私有线程或跨请求数值缓存；分位数的排名和插值数学不变，不承诺稳定的同值或NaN载荷次序。此完整排序能力与上文单个分位数的 `VQSelect` 排名选择不同。

`highway_vqsort_calls` 记录实际完成的库排序调用数；该字段在 IPC21 引入，当前贯通直接算子、`native_chunks`、子作用域和 IPC24，不代表 SIMD 指令数。Planner在存在区间、库可用且图或嵌套作用域含 `quantiles_axis` 时，为每个活动worker保守增加16KiB预计scratch，reason为 `native_numeric_line_sort_resource_reserve`；即使本次line不满足数值或长度准入，也不删除该估算。此库栈/TLS预留在既有N-double排序scratch之外，不实际分配占位缓冲，也不是所有OS/编译器栈帧的严格上界。`native_library_workspace_reserve_bytes` 记录每chunk最大库资源估算，不能用它代替完整堆或RSS测量。

完整恒等compact分支可共享主Program的 `const ExecutionMetadata` 及不可变物理proof。准入逐项检查全部Program字段、根和节点映射、槽位、结果schema以及数学常量bits；有作用域或非恒等分支仍独立finalize。结构validate仍执行，worker预算不减。此机制只减少重复计划元数据；请求Scratch、可变状态和结果不因它共享，普通独立结果与prepared借用/快照契约不变。

`native_chunks` 的 `compact_path_groups/batches/steps` 记录实际紧凑执行的组、批次和扫描步数，通过 IPC24 传输；计数不是性能验收。完整链性能、分配峰值与实现状态由上述设计记录，未验收条目不据此标为通过。

### 私有独立轴的有界分块与归约融合

严格 typed 图中，若只有一个私有 `matmul` 生产节点、左输入为静态候选矩阵，且所有根都是沿候选轴独立的 float64 向量，引擎可在已证明的支持操作闭包内分块执行。当前准入为 128～1,000,000 个候选、内维度不超过 128、轨迹宽度不超过 1,000,000；每块最多 64 行，并按整个变化依赖闭包的最大宽度控制到约 163,840 个元素（单行更宽时保留完整单行）。时间归约和递推保持完整历史，借用输入不修改，各根直接写本次独立结果的相应区间。

内部子图还可融合私有 `where→sum_axis(axis=1)`、连续的净值/峰值/回撤最大归约闭包，以及连续的减法/自乘/选择/求和闭包。使用 canonical 的同源数学步骤和原观察顺序；原逻辑节点及其参数、结构和 payload 校验位置保留，故障在原节点回放。私有缓存只保存逐候选归约值，不物化未请求的完整路径。Planner 对不同窗口取保守 arena 上界，库资源和子图工作区仍计入预算；native_chunks 报告实际 arena、scratch、复制与直接输出审计，估算不等于实测内存峰值。

公开中间矩阵、未识别消费者/形状、scope、isolate、观察器、预绑定执行或输入故障保留通用执行。分块后的异常选择仍按原节点优先级及行顺序；取消和资源异常立即生效。逻辑算子、数学口径、结果协议和 Excel 公式不变，普通计算不执行 Excel。自动 ISA 的单概率 `quantiles_axis` 在每条 line 至少 128 项时，可复用已有 Highway VQSelect 获取相邻排名；库关闭或不支持 float64 时使用标准库 `nth_element`。单 line 仍只使用原有 N-double 排名 scratch，Planner 的既有 16KiB 库资源预留不变。含 NaN、负零、次正规数、显式 scalar、多概率或不足 128 项时保留原排序；特殊数值的保护和原插值数学不变。Apple arm64、macOS 15+ 的小 K 长输出矩形复用已有有序 Eigen 内核，保持每个输出的升序乘加。内维度 1～128、输出宽度至少 128 时，F 右操作数的 1～3 行也可使用该内核；C 右操作数的短行保留实测更快的原路径。准入识别原始连续或正步长带填充布局，不为了该优化复制或转换整个输入。

普通Scheduler在单区间、至少两个CPU token及已证明的上述独立候选闭包下，可选择`parallel_dimension="array_row"`，复用持久ThreadPool。每worker至少128行，数值工作区独占，各自直接写全局独立根的无重叠行范围；内存准入按并发工作区计量，预算收缩至单线程时恢复顺序路径。数值错误按全图原节点优先级汇总，取消/资源错误排空worker后返回。自动prepared执行保留既有single策略；多区间继续原调度，不把物理行块当成新的区间或产品。空K视图可保留极端stride，但不得据此形成无效行偏移。

Apple arm64、macOS15+、Eigen开启且自动ISA时，私有条件求和/平方求和还可用四条独立行的Packet2d并行归约；各行仍按历史升序累加、不横向重排、不额外分配。masked未消费值不读取，非支持平台/ISA继续原路径。这是同一数学逻辑的执行优化，不能将审计lane数量当成全请求性能证据。

质量及完整负载范围见[三风险图设计与证据](physical-execution-evidence/2026-10-05-candidate-summaries/quantiles/design.md)。本轮减少了 GB 级中间数组；部分完整负载仍慢于强 NJIT，不能据此宣称整个平台计算已完成性能验收。


V33进一步支持同一静态候选矩阵输入派生的逐行变换/归一化闭包，逻辑节点、先后错误和独立结果保留；不可跨候选行的归约不准入。向量与矩阵多根一起按候选行分块，明确F矩阵使用有界C块再按完整全局leading dimension散写；复制及每worker块空间如实预算。终端MatMul普通混合符号仅在完整有限/幅度/舍入域与原0.000049差分界成立时调用受护DGEMM，近零/符号不确定格回放；高量级界不成立仍用有序路径，不把随机中等尺度结果外推为全部GEMM。

C连续、非标量、至少128字节的uint8掩码可复用Highway检查完整0/1域及所需选中数；坏字节、未对齐及尾部全检，没有额外掩码或跨请求缓存。F/反向/广播/小规模/OFF保持原访问。消费scope严格子图先发生数值故障时，对同一真实成员复用既有隔离结构审计，额外工作扣原预算；未消费成员不检查。公开类型、数学公式、结果和Excel预算未改。

已准入的私有 MatMul 分位数/统计融合还支持共用概率：所有有效分位数消费者（包括重复列和不同输入节点）只请求同一数值概率时，复用既有 Highway VQSelect/标准库 nth_element 获取相邻排名，沿用原凸组合插值。均值和二次偏差在重排前按原顺序完成；多种不同概率、特殊数值和短样本继续原 helper 的适用路径。prepared 每次重读概率，保留原错误、工作区/库资源预算、取消及独立 C/F 结果，不增加线程池或缓存。该策略适用于已证明的私有投影消费者闭包，直接算子的多分位数契约保持不变，Excel 继续展开原数学图。

2026-10-06，本机实际普通入口复测 131,072 个给定候选、2,520 点、30 个因子的投影及三统计请求，单核 C/F 为 1.885/1.879 秒，预热 NJIT 为 4.380/4.392 秒；相对原 C++ 约减少 71% 耗时，峰值基本持平。完整测试、实际路径、所有权及测量范围见 [V47 验收](physical-execution-evidence/2026-10-05-candidate-summaries/quantiles/row-risk-design/mixed-roots-v25/input-bounds-v28/packed-f-successor/pairwise-gemm-experiment/integration-v35r2/follow-up-v47/README.md)。这些结果不代表其他大矩阵、多个不同分位点或平台完整流程已通过性能验收。

私有、单消费者的两段算术融合可保存一个显式短向量广播输入：第一段必须是add/subtract/multiply且不作为公开根，二维广播的向量宽度必须等于结果列数。其他父值通常须持久；现在也允许一个已物化、独立拥有的私有C数组，要求它与两段结果同形、缓存slot独立，且原slot在消费者前不被覆盖。若消费者复用该slot，另一操作数不能通过视图借用同一底层数组。原位置prepare后，只保存有界短向量；消费者复用既有Eigen/标量算术，按原操作数顺序写结果，避免第一段完整矩阵。借用投影、未知/不适用几何、子作用域、isolate、观察器和上游故障保留原路径，分母零不进入库写入。包含gaussian_hmm_em的图暂保留此前已验收的私有数组路线，避免缩小arena后丢失免费第三块history并增加beta复制。C连续float64、N≥32、8≤D≤256的非显式scalar轴0和/均值/方差仍可复用Eigen有序行归约，保持每列升序、两遍方差、ddof与错误前缀；F和其他几何保留原生路径。

私有二维`take_axis(source,columns,1)`还可与独立`scatter_rows`根或既有scatter/argmax配对共同执行，不物化中间列重排矩阵。必须证明数值源和int64列索引的实际底层存活，间隔只允许输入/常量/参数叶节点；索引边界、回填去重和dtype错误仍在原节点验证。Planner和执行共用同一几何检查，并在F布局最终选择后复核直接根写入是否仍有效；无法证明独立目标、未知形状、其他轴/秩和HMM图保留原路径。公开C/F布局与两代结果所有权不变，逻辑AST/DAG、算子和Excel公式不变。[V57设计与验收](physical-execution-evidence/2026-10-07-private-mapping-v57/README.md)区分完整链实际收益和剩余NJIT内存缺口。

私有比较驱动的多级`state_select`可按4096点分块复用掩码，直接写独立int64根。仅适用同一一维源、稳定标量分支、同一输入valid掩码且中间节点单消费的闭包；前态可位于任一分支。每个逻辑节点仍在原位置验证；比较保留完整输入准备，选择只豁免已证明的内部条件/前态描述符，真实父值先保留NULL_INPUT检查，避免同时故障改变首错。必须证明底层源在最终选择前没有实际slot重写；未知形状、暴露中间根、数组分支、计算得到的valid、隔离/最少观测策略及含作用域的图保留原执行。Planner与执行共用同一准入，并计入4KiB掩码tile；Prepared仍只支持单通道。该物理优化不改变原AST/DAG、算子、数学/错误契约和Excel步骤。[V58设计与验收](physical-execution-evidence/2026-10-07-state-decision-v58/README.md)。

2026-10-06，[V48实际验收](physical-execution-evidence/2026-10-06-standardization-v48/README.md)对百万行×32列标准化完整请求确认申请峰值比原实现减少约256MiB，与强预热NJIT基本持平；实际同场64次配对C/F约0.130/0.108秒，NI约0.097/0.135秒。F已提速，C仍有缺口；首轮wall波动和全部负结果保留。该优化尊重请求的C/F结果布局，不新增线程、公开算子、序列化字段或Excel计算。


## 高斯密度与 HMM 尾行的物理融合

私有 `gaussian_log_density` 仅被一个 `hmm_filter` 消费，filter 又仅被沿 axis0/step1 的常量最后一行切片消费时，普通 typed 严格执行在 N≥3、NK≥131072、1≤D≤32、1≤K≤16且底层缓冲寿命可证明的几何中，可不物化两块 NK 历史。先用128行 tile 扫完原密度以保留首错顺序，再按原行序重算密度并执行同一 filter 递推，仅保存 K 尾值。独占工作区保守计入131K doubles；逻辑节点、切片和输出形状不变。

扩展到 D/K 超过旧 1–4 范围时，若同图存在可借用历史的大 HMM 拟合，准入还须保留其原来可免费取得的三块 NK 历史。证明复用同一个 arena 规划器：以所有可能的视图/融合消除计算安全下界；符合条件的唯一、尚未发布的 scatter 根可作为第三块历史。实际借用继续检查容量、字节范围、对齐、重叠、作用域和恢复，不因尾行融合改为每轮重新计算拟合密度。无法证明时保留原执行；没有相应拟合或不能借用历史的图仍遵循各自原工作区规则。

`gaussian_filter_streams` 在 `native_chunks` 记录实际完成次数，贯通子图和当前请求30/响应29。旧 worker 在读取新帧或附加共享内存前被拒绝。Planner 对第二次密度计算计费，包括健康的嵌套 `iterate` 子图；隔离父图不会漏计健康子图。额外消费者、未证明的 backing 寿命、公开 in-process Prepared、观察器及不适用几何保持原路径；失败按原节点重放，保持错误、独立根和原预算降级。输入及旧结果不作 scratch，没有跨请求数值缓存或新线程。算子、数学和 Excel 逻辑 DAG/公式不变。[资格与完整链证据](physical-execution-evidence/2026-10-09-latent-current-v69/future-result-screen-v74/qualified-adoption/row-state-reuse-v78/README.md#gaussian-filter-stream-v85)。

## 2026-10-07：固定轮数HMM的检查点工作区

零容差且N≥4096、16≤K≤128、1≤D≤32、NK≥8,388,608的HMM，在图未选择更便宜的免费完整history借用时，可以只保留一块NK：完整密度校验后原地转为beta，保存每128行的alpha检查点，再重放128行alpha与136行density；xi仍按时间升序累加，最后一个消费者结束后才将beta覆盖为gamma。工作区为 `NK+4K²+4K+512+ceil(N/128)*K+265K` doubles。原始log概率下限、gamma、中心化M步和严格停止不变。正容差与K8–15保留原路线；Planner不知道运行时容差，继续保留完整history包络及较短区间上界，不能按紧凑工作区降低未知参数的预算准入。

Apple arm64/macOS15+、Accelerate开启、非scalar、nearest且有界数据准入时，前向行按 `exp(alpha-max)` 与转移缓存做DGEMV，后向行做相应矩阵向量乘法，复用既有vForce。缓存是 `exp(log(max(A,1e-12)))`，不重新归一化；logA须在[log(1e-12),0]、行操作数绝对值不超过1e8，不适用行在写入前回到有序log数学。仅固定轮数使用该浮点重排，既定绝对误差预算0.000049及离散/错误断言不变。scalar/OFF/非nearest可使用相同检查点存储，但不执行新的BLAS路线；没有对应平台性能承诺。

`accelerate_matmul_calls`现也在直接算子audit中公开，含HMM实际DGEMV调用；整图native_chunks和worker复用已有计数。检查点/边界复制仍计入algorithm_copy_bytes，不把少分配一块NK称为全算法零复制。prepared混合大小或容差请求可能保留先前较大的容量；当前结果和run_snapshot仍独立，旧结果不成为工作区。参数、输出布局、线程预算和IPC版本不变。完整请求与正常/SAN/OFF证据见[V61验收记录](physical-execution-evidence/2026-10-07-hmm-alpha-checkpoints-v61/README.md)；只对已测几何报告性能，不宣称全部情景链或平台验收完成。

## 连续行选择的私有 C/F 视图

普通 typed 严格单区间执行中，至少 262144 元素的私有 `take_axis(...,axis=0)` 可借用正向完整连续 C/F 输入，或本次已经完成且独立拥有的 C/F 矩阵根。每次读取并验证 int64 行索引必须连续升序、位于源范围内；保留源 strides，起点为 `first*stride[0]`。外部源还须证明完整字节 span 不与本次输出重叠。已请求的选择根仍按原独立输出契约返回。

允许共享已完成根只用于只读输入；不将输入、已发布根或前次结果变成可写 scratch。不为它强制转换 F 为 C，不按输入地址缓存数值。padding、负步长、非连续索引、不能证明底层存活、子作用域、isolate、观察器和公开 in-process prepared 等不适用路径继续原物化执行；F 子区间的列距若不等于本次完整行数，也不准入。每个原逻辑节点的 prepare 与错误位置保留。

[V91 接入与复验](physical-execution-evidence/2026-10-09-latent-current-v69/future-result-screen-v74/qualified-adoption/row-state-reuse-v78/README.md#gaussian-filter-stream-v91)。这项物理实现不改变算子、AST/DAG、数学或 Excel 公式；普通 C++ 请求仍不运行 Excel。
