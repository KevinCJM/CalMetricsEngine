# 分块 Accelerate Cholesky：隔离筛选，未接入

2026-09-30，macOS arm64、单核。当前 C++ 构建是已安装的 `91267c23ac014c947e67ae31054dddc8aaa102fadd12f2ce4331c87752c63758` wheel；对照是预热、同核数的 NJIT `np.linalg.cholesky`。候选为独立 C ABI 原型，先检查有限和严格对称，再复制输入到独立输出，单线程调用系统 Accelerate。`block=0` 直接调用 `dpotrf`；分块候选复用 `dpotrf/dtrsm/dsyrk`，最后清零上三角。源码见 [原型](calmetrics_cholesky_block_probe.cpp)；编译使用 Clang `-O3 -fno-fast-math -ffp-contract=off -framework Accelerate`。未修改生产内核、Planner、图或 Excel。

4096 阶严格对角占优矩阵的隔离直接入口，旧 C++ / NJIT / 64 阶块 / 512 阶块的中位数分别约为 **1775 / 875 / 470 / 268 ms**（每条 2 轮）；2048 阶 3 轮分别约 **212 / 82 / 50 / 39 ms**。输出独立分配，输入和 NJIT 签名先准备；候选计时包含输入检查、输出分配和库调用。不同入口的 Python 包装并不完全相同，2–3 轮不足以做正式置信区间或完整 Scheduler 验收。[1024/2048 原始时间](calmetrics_cholesky_block_screen.json)、[4096 原始时间](calmetrics_cholesky_block_4096_screen.json)及[小块补测](calmetrics_cholesky_block_small_screen.json)保留全部轮次。

4096 阶 fresh 进程的冷请求 malloc 活跃峰值（MiB）为旧 C++ **128.503**、NJIT **256.067**、32/64/512 阶块 **129.032 / 129.969 / 138.000**。64 阶块相对旧 C++ 明确增加约 **1.47 MiB**，相对 NJIT 减少约 **126 MiB**；512 阶块更快但相对旧 C++ 多约 **9.5 MiB**。两份独立结果的暖峰值、申请总量及探针计数见[冷暖原始记录](calmetrics_cholesky_block_memory_4096.json)、[其他块宽](calmetrics_cholesky_block_memory_more.json)和[小块内存](calmetrics_cholesky_block_memory_small.json)。[测量子进程源码](calmetrics_cholesky_block_memory.py)将输入构造放在 epoch 外、输出分配放在 epoch 内。malloc 探针不覆盖全部映射、Python arena 或 RSS；这些值也不是图/worker 的峰值。

**不能全域接入。**现有逐项 `rtol=2e-12, atol=2e-14` 对照下，2048 阶 Wishart 输入的 64 阶块有 16 个元素超容差；直接 `dpotrf` 与其他块宽也失败，详见[所有块宽失败值](wishart-variants.json)。另一组 Wishart 输入在[对抗样本](calmetrics_cholesky_block_adversarial.json)中有 5 个超容差元素。70 组 256/1024 阶严格对角占优样本通过该容差，记录于[占优样本](calmetrics_cholesky_block_dominant_screen.json)，但有限样本不能证明所有占优矩阵等价。非有限、非正定和近奇异仍需保留原错误、重放及下游离散结果。

下一步若接入，只能在可核验的数值域和独立浮点结果边界准入，并在库失败或守卫不通过时重放原内核；还须处理 C/F/任意 stride、prepared、子作用域、错误隔离、进程传输、审计、预算和正式同核数时间/内存验收。当前没有生产采用决定，也不能把此原型的直接入口性能称为全引擎收益。
