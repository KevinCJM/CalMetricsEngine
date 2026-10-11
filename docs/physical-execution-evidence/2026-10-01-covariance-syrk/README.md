# 宽矩阵协方差 DSYRK 筛选

2026-10-01，本机 macOS arm64。沿用当前 D8：微小时间/内存增加只记录，不独立否决；优先减少重型计算耗时与大额内存占用。本轮只筛选候选，不改变生产算子、错误语义、DAG、Planner、worker 或 Excel。

## 设计与采用前置条件

- 对照当前正式安装包的完整普通 Scheduler 请求与已预热、同核数 NJIT `np.cov(x.T)`。候选为独立 C++ ABI；包含独立结果分配、输入检查、均值、面板和对称填充，但尚无 Scheduler/预算/运输开销，成绩不作为正式入口验收。
- 重用既有 Accelerate DSYRK 方案，以至多 512×D 中心化面板取代 T×D 中间数组，分别保持 C/F 输入读法。结果独立、C 连续；输入只读，原始 strides 不变。
- 均值使用当前 `matrix_backend.cpp` 的 C 行向量 Welford 和 F 八列 gather 表达式；筛选原型暂时复述这两个表达式。正式接入前必须抽取共享实现，不能维持第二套生产均值数学逻辑。
- DSYRK 会改变乘加顺序。按既有 `rtol=atol=1e-12` 比较完整结果，同时单独记录近零符号、IEEE 类别、对称性、输入及输出哨兵、跨次所有权。普通随机样本通过不能证明整个数值域等价；高幅值、相消、常数列、极值和非有限输入必须单独核对。
- 当前线程强制单核 BLAS，保存并恢复此前策略。时间测量串行、轮换顺序、保留上一份结果；申请堆及 RSS 在独立新进程另测，不与计时并发。面板、库私有工作区与两份保留结果均计入。
- 本轮不新增算子或更改 Excel 数学契约。正式采用还需解决离散下游、观察器、错误隔离、预算和回退；没有新 Microsoft Excel 重算验收。

## 本轮证据

正式基线源码、wheel 和安装扩展的构建身份为 `33c2f2b95e374d396dc283c78d2ca5a43ad51a7a97995c3534ffdbae61ff7d39`，已重新核对源码哈希及实际普通协方差请求的审计身份。候选 ABI 单独编译，未替换安装包；[清单](manifest.json)记录源码、二进制、工具链与运行边界。

六轮均衡、单计算核、保留前次独立结果，完整普通入口与原型的墙钟中位数如下；不是三者均已接入相同 Scheduler 的结果。

| 输入形状与布局 | 当前 C++ 普通入口 | 候选独立 ABI | warmed NJIT |
| --- | ---: | ---: | ---: |
| 32768×64 C | 8.924 ms | 2.941 ms | 10.686 ms |
| 32768×64 F | 11.420 ms | 4.421 ms | 6.785 ms |
| 2048×2048 C | 584.298 ms | 93.311 ms | 94.618 ms |
| 2048×2048 F | 607.161 ms | 110.526 ms | 82.862 ms |
| 4096×4096 C | 5272.944 ms | 427.589 ms | 712.070 ms |
| 4096×4096 F | 5198.243 ms | 403.390 ms | 646.808 ms |

4096 阶候选相对当前 C++ 约快 12.3–12.9 倍，相对 NJIT 约快 1.60–1.67 倍；不能外推到全部形状。2048 阶 F 仍比 NJIT 慢约 27.664 ms，保留该实测差距。63×8 的 C/F 原型分别为 7.354/5.250 μs，NJIT 均约 3.375 μs；这些微小绝对增加不单独阻断采用。全部十项原始样本、完整矩阵容差比较及 CPU/墙钟观察保存在 [timing.json](timing.json)；普通随机负载的候选/NJIT 对照都在既有 `1e-12` 内。

十二个新进程将申请堆和 RSS 分开测量，计入 setup、cold、释放首份结果、五次预热及暖态两份保留结果。4096 阶 C/F 的申请堆结果相同：

| 请求申请堆峰值 | 当前 C++ 普通入口 | 候选独立 ABI | warmed NJIT |
| --- | ---: | ---: | ---: |
| cold | 128.008 MiB | 150.035 MiB | 390.000 MiB |
| warm，保留两份结果 | 256.008 MiB | 278.035 MiB | 518.000 MiB |

候选比 NJIT 各少约 239.966 MiB，比当前 C++ 各增加约 22.027 MiB。这是有界面板和库工作区的实质取舍，不按几 KB 波动处理，也不写成零增加。所有跟踪表溢出和未跟踪释放为0；输入及运行环境预先存在，申请堆未覆盖 mmap、全部 Python arena 或另一个 worker。原型还没有 Planner 预算，实测峰值不等于厂商工作区硬上界。

RSS 暖态请求增量 C：当前 C++/候选/NJIT 为 134,381,568 / 140,673,024 / 134,283,264 B；F：134,381,568 / 134,414,336 / 146,817,024 B，F 候选 idle 对照为32,768 B。候选 C 的 RSS 增量仍比 NJIT 大约6.09 MiB，不能用申请堆节约或较低进程总 RSS 抹去这个观察。[memory.json](memory.json)保留完整阶段及采样范围，不声称所有内存指标领先。

## 数值问题与自审核

[correctness.json](correctness.json)记录176个普通/对抗样本：128个接受、48个在写出前拒绝。检查了 C/F、补边/不对齐、只读输入、输出两端哨兵、对称性、IEEE 类别、跨次独立所有权和 BLAS 策略恢复；拒绝前没有改写输出哨兵。筛选不是176项数值验收通过：

- 6个高幅值但仍处于 `abs(x)≤1e4` 准入范围的 scaled 样本超出既有逐元素容差。`n=1041,d=128` 的 C/F 单项各有4/10个越界元素；补边布局也复现，不能仅依靠输入幅值上限采用。
- 16个正交样本发生近零正负或零/非零分类变化。最大幅度尾差约 `1e-17`，但可能改变离散下游，不能只凭 allclose 宣称等价。
- 非有限/极值输入在原型中拒绝，不能视为复现了 canonical 的局部健康结果、逐位置故障或图错误隔离。

数值检查命令因此返回 **2**，不是成功验收；最初脚本只记录问题而返回0，自审核后改为非零退出并复测。问题记录保留，没有改容差或删样本。时间/内存采用结论与数学等价验收分开，微小性能退化不是本候选未接入生产的原因。

下一步采用前须共享 canonical 均值、给出符合原数值域的准入/逐项重放或完整回退，保留近零及离散行为，再连接独立根、Workspace、Planner、worker 与观察器限制。原型目前**未接入生产**。本轮不重跑正式引擎的全量回归或 Microsoft Excel：正式 C++/公式源码未变，前轮9481项等验收属于前轮证据，不能当作候选已通过。未提交、推送或修改研究平台。

## 复现

在相同本机工具链与 Eigen 3.4 头文件下编译候选（`$TASK_EIGEN_INCLUDE`、`$TASK_COVARIANCE_LIB`、`$TASK_ENGINE_INSTALL`、`$TASK_HEAP_PROBE` 替换为本机路径）：

```bash
clang++ -std=c++17 -O3 -DNDEBUG -fno-fast-math -ffp-contract=off -mmacosx-version-min=15.0 -dynamiclib -fPIC -I"$TASK_EIGEN_INCLUDE" candidate.cpp -framework Accelerate -o "$TASK_COVARIANCE_LIB"
PYTHONPATH="$TASK_ENGINE_INSTALL" python screen.py --mode check --library "$TASK_COVARIANCE_LIB" --output .
PYTHONPATH="$TASK_ENGINE_INSTALL" python screen.py --mode timing --install "$TASK_ENGINE_INSTALL" --library "$TASK_COVARIANCE_LIB" --output .
PYTHONPATH="$TASK_ENGINE_INSTALL" python screen.py --mode memory-sweep --install "$TASK_ENGINE_INSTALL" --library "$TASK_COVARIANCE_LIB" --probe "$TASK_HEAP_PROBE" --output .
```

三个命令分开执行；第一条数值检查应因已知问题非零退出。malloc 需在子进程启动时注入探针，RSS 需允许只读查询自身进程。普通入口和候选分别见 [screen.py](screen.py)、[candidate.cpp](candidate.cpp)，原始正负结果均保留。
