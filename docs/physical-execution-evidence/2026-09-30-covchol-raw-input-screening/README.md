# 直接借用输入的 Accelerate 协方差→分解：内存筛选

2026-09-30，本机 macOS 15 arm64、单线程。为检查已接入的 512 行中心化面板能否进一步缩小，在**独立 ABI 原型**中直接对连续原始输入调用 Accelerate `dsyrk`，再减去 `T*mean_j*mean_k`、归一化及 `dpotrf`。C/F 各比较 `Trans` 与等价的 `NoTrans` 布局调用；不修改生产算子、Planner 或 Excel。中心化前要求 `max|mean|≤0.05*max|x|`，避免大偏置下的消去误差；其余非有限、近零协方差和分解守卫与前一候选同类。

[两种布局调用的 46 组结果](raw-probe-screen.json)与[NoTrans 结果](raw-notrans-screen.json)各有 8 组准入，准入项对旧算子均在 `rtol=atol=1e-12` 内；其余回退。4096 阶 C/F 的 Trans 原型约 `733/623 ms`，NoTrans 约 `733/624 ms`，均不含 Scheduler、Planner 和结果运输，**不能与正式完整请求直接相比**。2048 阶 Trans 原型约 `97/97 ms`，同样只有 ABI 内核计时。

新进程 malloc 探针中，4096 阶[Trans 原型](raw-probe-memory.json)与[NoTrans 原型](raw-notrans-memory.json)冷峰均约 `140.03 MiB`，比已接入面板路径的完整请求 `150.04 MiB` 少约 10 MiB，但和旧完整 C++ 的 `128.51 MiB` 比仍高约 11.5 MiB。两种入口层级不同，不能将这个差额当作正式替换后的精确收益或合格证据。单独的 `dpotrf` 调用还观察到约 `5.08 MiB` 临时 malloc 峰值，说明去掉面板并不能自动使整条库路径恢复到旧内存峰值。

因此本筛选**不替换生产实现**。如需采用，仍须扩大高维对抗输入、验证原路径错误与离散行为，接入受控工作区和预算，并重新跑同入口新旧/NJIT 时间、堆和 RSS。源文件与脚本保留本机绝对路径，异机复现需要替换安装与库路径；原始输出均在本目录。
