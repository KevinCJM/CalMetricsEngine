# V44：按实际分配成本选择 HMM 物理 history

前轮为 PROGRESS：V43 通过相关 783、原生 9/9、实际 V42 对照 16 组，并测得完整训练 malloc 峰值减少 35,987,712 B；完整识别因已有两块 arena 加一块未来输出，峰值基本不变。生产仍为 V42，本轮在隔离源中开发。

## 数学与原生库

HMM EM 的输入、类型、概率检查、密度表达式、一次 forward/backward、原在线 xi 归一化/升序累加、gamma、M、方差下限、似然、严格停止和模型结果不变。复用唯一现有 density/Eigen、有序 transition tile 及 Apple vForce；不新增算子、Python 数值回调、请求期编译、线程池或全局线程设置。库资料和旧检查点方案拒绝理由沿用 V43，新的工作是生命周期选择。

## 两个计划，不增加第三份分配

- 大 HMM 的直接原语规划为两块 NK 历史，加 K²/日志/128 行密度及 vForce 的固定小尾部；所有几何溢出检查在取指针前完成。其它小形状保留原路径。
- `loan_latent_workspace` 首先尝试三块**已存在**的独占 span：arena 里可用完整矩阵和一个当前调用的未来 float64 scatter 结果。只能使用图预算已经计入的空间；仍执行原根容量/对齐/重叠、活值保存、借用视图 backing 和错误恢复检查。旧一代结果永不借用。
- 若现有 span 不够三块，就回到两块选择；不为省密度重算而额外保留第三 NK。两块缺少的 span 可以来自原 private 工作区，保留/缩减容量与 overflow 检查不变。
- `planned_matrices` 决定小尾部大小和无借用原语上界；`matrices` 决定本次提供的独占 span。二者不能混用，否则减去 3NK 会导致 tail 下溢。
- internal fit 的 beta==em 表示两块/重放；beta 独立时使用完整 history。registry 对合法别名关系进行检查。是否借到第三块不改变 logical DAG、参数依赖或结果寿命。
- 图整体预算覆盖 arena、根输出及 private reservation。算子 `scratch_doubles` 是无 loan 的原语计划；不能把借用的第三块当成新的 malloc，也不能把它从图/根驻留内存中隐去。实际峰值仍以完整连续 epoch 和 RSS 测量为准。

## 源码唯一与错误纪律

三块/两块使用同一个模板 backward 表达式和同一 transition tile。两块在密度覆盖前扫描 alpha/em/logA，并在 backward 追踪 beta 范围；三块在 sweeps 后扫描三块历史。两个完整范围证明都必须成功，才使用 vForce；否则原有序路线继续处理相同输入。任一失败恢复所有借用活值和输出前缀，独立根/旧结果不变。

三块使用原完整密度，不调用 replay；两块每 128 行调用同一 density_rows，仍读本轮 E 步模型和 logs，M 尚未运行。Eigen 的选择 N 使用原完整 N，实际 block 长度仅决定循环/尾行。

## 复制审计

两块 backward 把滚动 beta 写入历史，按每次实际完成的 `copy_n` 加入既有 `algorithm_copy_bytes`：每个成功 E 步 N*K*8 B，异常只计已完成的复制。它是历史物化流量，不是额外 NK 分配或输入复制。三块直接写 beta，没有该复制。初始化参数、图保存/恢复、根复制及 input_copy_bytes 保持原定义。仍未覆盖的 Python arena、库内部 allocator/mmap、栈/worker 范围继续明确，不能称完整审计已闭环。

## 验收顺序

1. 正常原生和相关回归，包括预算拒绝、块尾、C/F/负 stride、FE、极值、严格停止邻点、直接/普通/prepared、两代所有权、错误恢复及进程/共享内存。
2. 新增阈值附近无 loan 的大模型，以及能免费借第三矩阵的完整链；用结果、预算和真实复制计数证明路线，没有第三块时不能无声多分配。
3. 实际 V42 对照及预热强 NJIT，CPU1、相同输入准备/图/输出/两代保留。完整训练和完整识别各六轮；连续 malloc epoch 与三组新进程 RSS 分别测，不混合阶段峰值。
4. 若整体证据支持，补全量回归、ASan/UBSan、库关闭；按任务范围接入实际代码，建立 source/wheel/worker 身份，再复测实际构建。最终采用前不改生产数值源。
5. 当前 Excel 算法/依赖/公式未变，仅物理布局选择，适用数值证据沿 X10 记录；不宣称新增 Microsoft Excel 桌面重算。平台有效前沿/随机探索及大矩阵 FF 的主要缺口继续保留。
