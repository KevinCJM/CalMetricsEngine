# Accelerate 协方差→Cholesky：隔离筛选，尚未接入生产

2026-09-30，本地 macOS arm64、单计算核。基线是上一轮已验收的 `a63f72ba69ac969b2a60f90254ad15e8075a05ae15639cc9fe7561b229efa699` 安装包。此目录只增加独立 C++ ABI 原型和原始测量；生产算子、AST/DAG、Planner、worker、Excel 公式均未改变。原型通过 `ctypes` 调用编译后的 C++，数值计算不经 Python/NumPy 回调；原型成绩**不是完整 Scheduler 验收**。

## 瓶颈及收益上界

在相同只读输入、单核、独立结果的分段测量中，4096 阶 C/F 布局的现有协方差分别约 **5381/5373 ms**，同签名已预热 NJIT `np.cov` 约 **708/683 ms**；现有 Cholesky 约 **1819/1795 ms**，NJIT 约 **924/942 ms**。当前完整原生图约 **7223/7153 ms**，显著差距首先来自协方差。[原始分段计时](isolate-stages.json)记录各三轮样本。

原型以有界 512 行中心化面板调用系统 Accelerate `cblas_dsyrk`，再对同一独立结果调用 `dpotrf`；BLAS 线程策略在当前线程设为 `SINGLE_THREADED` 并恢复。4096 阶 C/F 独立协方差约 **522/487 ms**，组合约 **717/670 ms**。六组普通随机矩阵的组合结果均在 `rtol=atol=1e-12` 内，4096 阶最大绝对差 C/F 约 **3.00e-13/4.37e-13**；数百万元素位模式不同。[协方差计时](compare-blas.json)与[组合计时](compare-covchol.json)均包含逐轮值。正式图中的绑定、准备、预算、错误恢复和结果运输还未接入，因此不能将约0.7秒对比正式图或宣称达到对 NJIT 的完整优势。

独立新进程的原型 malloc 跟踪：4096 阶 C/F 冷峰均约 **150.03 MiB**，保留前次独立结果的暖峰约 **278.03 MiB**；冷态进程 CPU/墙钟约 **1.00**，跟踪表溢出与未跟踪释放为0。[四组堆原始值](memory-4096-c.json)及 F/2048 数据在同目录。原型分配 512×D 面板，尚未纳入 Planner；这些 ABI 堆数字与上一轮完整 Scheduler 的 **128.512 MiB** 冷峰不能直接作为同入口新旧比较，系统库内部工作区和 RSS 仍需正式计量。CPU/墙钟比只支持本样本未观察到显著并行，不能单独证明所有库路径服从预算。

## 阻断直接采用的数值事实

- 66组 C/F、128/512/1024阶对抗输入中，重复或近重复列出现旧实现失败而 `dpotrf` 成功、或相反的情况；NaN/Inf 在独立原型的拒绝阶段也与旧链不同。必须保留原失败类别与传播位置。
- 高幅值 `1e8` 缩放的协方差和 Cholesky 有超过现有 `1e-12` 逐元素门槛的结果。不能把普通正态矩阵通过外推到完整数值域。[全部对抗结果](adversarial-blas.json)保留成功与失败样本。
- 另用正交化的 529×512 输入，BLAS 与当前协方差仅有约 `1e-15` 尾差，却在 C/F 布局分别令 **38/103** 个近零项符号不同。下游 `>0` 等离散结果可改变；不能用浮点 `allclose` 掩盖该差异。[独立脚本](orthogonal_sign_probe.py)与[原始结果](orthogonal-sign.json)保留输入构造和计数。

结论：**原型不接入生产**。下一轮先设计并验证可证明的准入/回退，包括非有限和极值输入、近奇异主元、靠近实际比较阈值的协方差及失败重放；无法维持旧离散行为的范围需保留旧内核或明确版本化契约。随后才可连接 Workspace/Planner、direct 与图的同源路径、独立结果和 worker，执行全量正确性及同入口新旧/NJIT 时间与内存验收。不得通过仅优化这条链声称所有算子目标完成。

## 复现与证据身份

使用本机 macOS SDK 与 AppleClang，将[原型源码](covariance_blas_probe.cpp)编译为本机动态库：

```bash
clang++ -std=c++17 -O3 -DNDEBUG -dynamiclib -fPIC covariance_blas_probe.cpp -framework Accelerate -o libcovariance_blas_probe.dylib
```

各 Python 脚本保留本机绝对路径与所用安装包目录，异机复现需替换路径。`candidate_memory.py` 要求进程启动时注入已有 malloc 探针。所有数值测量串行运行，不与构建或其他性能测试并发。[SHA清单](manifest.json)覆盖16份原始源码、脚本和JSON；没有把临时动态库或旧/新安装树视作可提交源码。
