# exp/log 跨 C/F 布局：分块原生库设计与验收

2026-10-01；基线为 vForce 首轮接入 build `81f677b3bab648df8e8fb7b686f0cb471997ec8c59999216600f00f176c0aeac`。目标是消除大数组强制跨布局的差距，保持原数学、错误、所有权和输出布局契约。本阶段限定范围开发、自审核与验收完成，尚未提交或推送；整体性能目标仍开放。

## 详细设计与候选选择

| 候选 | 2048² F→C exp/log 原语耗时 | 数值工作区 | 决策 |
| --- | --- | --- | --- |
| 原图路径 | 约112/78ms | 无 tile | 保留为旧 wheel 对照 |
| 三缓冲：先打包，再数学，再转置与搬运 | 约19/20ms | 三个 tile | 有效，但输入打包可省去 |
| 全根转置，再 vForce 原地计算 | 约23/26ms | 无 tile | log 慢于同布局分块 NJIT；不采用 |
| 连续输入段直接 vForce 到 tile，再转置搬运 | 约18/19ms | 两个 tile，最多64KiB | 采用；真实 wheel 验收另列 |

这些是隔离候选数据，不是实际 wheel 验收。Apple 明确允许 vForce 输入输出相同地址；完整矩阵原地候选因为实测更慢而未采用，不能为了几 KB 工作区而牺牲大量时间。原语签名与坐标依据本机 SDK vDSP.h、[Apple矩阵转置](https://developer.apple.com/documentation/accelerate/vdsp_mtrans)和[vForce说明](https://developer.apple.com/documentation/accelerate/veclib/vforce)。

- 数值仍交给成熟 vForce；vDSP_mtransD 与 vDSP_mmovD 负责转置/有pitch搬运，C++仅做坐标、连续复制与准入。
- 独立 exp/log、automatic、macOS15 arm64 的原有资格不变。跨布局至少65536元素、第一轴至少32、其余轴展平至少8；二维、三维都按真实逻辑坐标写最终指定连续 C/F 结果。
- F→C 使用128×32块：每列连续输入段直接计算到第一tile，vDSP转置到第二tile，再移动到最终C结果。C→F 使用64×64块：每行连续输入段直接计算到第一tile，转置到第二tile，按F列地址连续复制到最终结果。没有输入打包或全矩阵中间数组。
- F正向列允许第一轴补边、部分窗口及正向平面跨度；C正向行允许行pitch补边，三维内层必须连续。补边和窗口外值不消费、不扫描。反向、重叠和其他几何保持canonical。
- 同序连续输入保留整段库调用；原F列段方案保留，新增每行至少1024元素的C连续行段，可零tile写C结果。
- 全量消费值域预检必须早于任何输出或tile写入。exp有限且[-700,700]，log有限且≥最小正规数，FE_TONEAREST；拒绝后canonical保留首次错误与已写前缀。prepared每次重新读取当前值，不缓存值域资格。
- Workspace独占两个tile；形状上界统一供Planner、Scratch容量限制与执行查询，最大8192个double/64KiB。上界保守覆盖两个方向，实际同序路径不新分配tile；关闭库时上界为零。旧容量可在预算内保留，不能把准备前分配记作零。
- `accelerate_vforce_calls`仍计实际数学调用；`algorithm_copy_bytes`计转置+最终搬运，共2×消费元素数×8字节。输入绑定零拷贝不等于全链零拷贝。每个根直接写本次独立owner，保留旧结果不受下一次执行影响。
- 保持Scheduler CPU token、逻辑DAG、185算子、输出协议和IPC18。不新增私有并发、全局线程策略、Python数值回调或新数学公式。观察器、消费者、作用域、多根、isolate仍走原有数学路径。

## 公平对照与证据边界

上一阶段冻结24组证据中，强制F输入/C输出时，NJIT `np.exp/log(F)`实测返回F。该NJIT时间保留为布局不等价的诊断，不能作为相同C输出的验收。新对照保留两个warmed单核NJIT基线：NumPy数学后转为指定布局；自研分块单输出NJIT，避免完整中间数组。两者都按指定C/F返回独立连续结果，前一结果在下一结果生成期间继续保留。

真实新旧wheel测ordinary/prepared完整入口，12次轮转配对，原容差rtol=2e-12/atol=2e-14及IEEE/符号/零分类。独立新进程分别测malloc和RSS，记录预绑定/结果寿命/工作区/复制/Planner；无表溢出才使用heap结论。增加跨布局二维/三维、部分窗口、补边、阈值、失败预检、标准舍入、process inline/shared和寿命回归；完整Python/native/ASan及双库OFF回归。

按D1–D8及Excel X1/X3/X4/X6/X9/X10，公式、数学、DAG依赖与Excel预算不变。历史Microsoft Excel快照复核不称为本轮新Excel重算；不放宽数值容差，不将本轮局部收益当作全引擎目标完成。几KB/几ms正差不是单独否决依据，正确性/所有权/预算仍是硬条件。

## 实际 wheel 时间与内存验收

新 build `6f02e6bb69d2d59a62781c357eda16f9604ae9eee2377af383b3b56b29c2bf6f`；原始20组完整调用、100组输出对照及CPU/wall在 `timing.json`，12次轮转的六个单CPU backend都保留前一结果；CPU/wall中位数最大约1.00。`performance-summary.json` 保留每组绝对差与对较快NJIT的比值。单位ms：

| 输入/算法/输出 | 旧ordinary | 新ordinary | 新prepared | NJIT NumPy+布局 | NJIT单输出分块 | 新ordinary/较快NJIT |
| --- | --- | --- | --- | --- | --- | --- |
| 2048² exp F→C | 109.874 | 15.619 | 15.718 | 44.334 | 39.288 | 0.398 |
| 2048² exp C→F | 48.262 | 15.261 | 15.395 | 56.733 | 34.754 | 0.439 |
| 4096² exp F→C | 505.586 | 83.107 | 82.925 | 192.302 | 157.871 | 0.526 |
| 4096² exp C→F | 221.890 | 82.315 | 81.529 | 227.620 | 139.818 | 0.589 |
| 2048² log F→C | 77.095 | 16.493 | 16.669 | 33.156 | 22.040 | 0.748 |
| 2048² log C→F | 30.352 | 16.340 | 16.311 | 44.599 | 25.749 | 0.635 |
| 4096² log F→C | 365.743 | 85.917 | 86.127 | 147.471 | 88.676 | 0.969 |
| 4096² log C→F | 190.972 | 88.273 | 88.816 | 181.360 | 101.218 | 0.872 |

- 4096² exp 对较快分块NJIT耗时少约41%–47%，相较旧C++缩短约2.7–6.1倍。4096² F→C log 85.917 vs 88.676ms，仅少约3.1%，记为接近持平；不能宣称所有大负载都明显领先。
- 三维1024×33×17跨布局exp约2.14/2.22ms，较快NJIT约6.07/7.65ms；log约2.42/2.40ms，较快NJIT约3.72/4.21ms。相同C/F布局控制基本不变。
- 63点普通入口约0.0028ms，较快NJIT约0.0017ms，差约1微秒；1024点log差约2.3微秒，均记录但不作单独采用阻断。这里没有另设统一比例或几毫秒硬界线。
- `memory.json` 包含48个fresh进程，分别测malloc/RSS，两种布局、两种函数、ordinary/prepared及两NJIT基线；heap追踪表溢出均0，control通过。
- 4096²正常新ordinary申请堆cold/warm约128.070/256.071MiB，旧约128.008/256.008MiB，增加64KiB tile及8字节元数据；新prepared约128.070/256.071MiB。同布局NumPy-NJIT为256.000/384.000MiB，各少约128MiB；单输出分块NJIT为128.000/256.000MiB，新包多约71.6KiB接口+tile，属于允许的小幅增长。不能声称所有基线的所有内存维度都下降。
- 新包RSS请求增量约128.000–128.031MiB，NJIT约128.016MiB，补充观测视为页级持平，未用总RSS差证明增量内存优势。保留0–64KiB idle control，不扣除或抹平。
- 实际工作区及Planner库reserve均最大65536字节；输入绑定copy和rootcopy为0，跨布局算法转置/搬运为268435456字节/4096²结果；math写tile不算输入pack。结果终态独立，只读，不与输入或后续结果共享。

## 自审核、回归及身份

| 检查 | 本轮结果 |
| --- | --- |
| 新跨布局Python回归 | 74项；连同原exp/log专项280通过 |
| 当前完整Python | 9966通过，99.55s |
| 原生 / ASan+UBSan / 双库OFF | 各9/9通过 |
| 双库OFF Python专项 | 339通过，164仅库开启专项skip |
| 全185直接算子夹具 | 185通过，原容差 |
| 真实MS Excel历史快照复比 | 469案例/3056值通过；938差异仅Readme B4/B9身份，未新重算 |
| wheel内容及源码身份 | ON/OFF分别77源文件、extension/worker双二进制一致，check_dist通过 |
| malloc/RSS | 48子进程，控制通过，追踪表溢出0 |

自审核确认：仅三个生产文件相较前一原生build改变（transcendental_accelerate.cpp、matrix_backend.hpp、graph.cpp）；private物理路径不增逻辑算子或新数学公式；统一Planner上界，懒分配和独占Workspace；全消费预检在所有输出写入之前，原NaN定义域错误/IEEE/首次错误前缀保持；F/三维坐标映射由native补边canary、Python差分与四lane运输覆盖；prepared逐次重读、快照独立。观察器/消费者/作用域/多根/isolate的资格排除保持原实现，无Python数值回调、私有池、全局线程策略或IPC schema变更。

首次native在sandbox内的两个shm_open失败以 `native.log` 保留，允许POSIX资源后9/9通过。首次OFF Python未设置既有 `CME_EXPECT_ACCELERATE=0`，184项错误地期待库调用；保留原日志，按既有OFF测试配置复测339通过/164跳过，不修改生产代码来迎合错误配置。初次新测试六处错误预期（消费窗口低于65536、NaN按原契约抛错）在focused日志保留，修正测试后专项与全回归通过。

ASan detect_leaks=0；系统Accelerate二进制不受本项目ASan插桩，不能据测试宣称外部库内存安全已全面证明。其他硬件/操作系统未在本轮真实运行；关闭库构建证明当前fallback编译和回归，不冒充跨平台全面性能验收。

早期三缓冲/root原型的identity对应当时运行快照；后续脚本修正闭包/排版并追加span候选，早期source hash不能当作当前文件身份。采用方案依赖span筛选及正式wheel的当前可核对identity，不能混同历史原型与生产实现。

## 未闭环及收尾

本轮采用依据是大跨布局请求显著缩短、避免NumPy-NJIT完整中间矩阵、微小工作区增加符合预算；不会因为KB/ms正差单独判失败，也不把可接受写成零增长。4096² F→C log仅接近分块NJIT；末尾值域拒绝的额外预检、其他混合符号矩阵/协方差放大与高相关回退、有界Jacobi及一般布局仍按各自前阶段正负证据开放。本轮未重新测这些未修改路径，不用新正常域成绩关闭其欠项。

已同步架构、执行/用户指南、README、文档索引、布局设计与物理优化记录；数学契约和Excel规则未改变，核对后不改公式/预算。214个本地文档目标存在性、Ruff与git diff --check通过；没有全面校验锚点或外部网页，不据此宣称链接语义全部正确。前一vForce integration的93个冻结文件及screening的14个冻结文件哈希均未改变。没有提交、推送或改远端。限定范围自审核通过不代表全引擎目标完成。

## 可复现命令

ON wheel：`PYTHONPATH=/private/tmp/calmetrics-matmul-deps MACOSX_DEPLOYMENT_TARGET=15.0 CMAKE_BUILD_PARALLEL_LEVEL=3 CMAKE_ARGS='-DCMAKE_MAKE_PROGRAM=/private/tmp/calmetrics-perf-followup-deps/bin/ninja -DCALMETRICS_ENGINE_BUILD_TESTS=ON' /Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pip wheel . --no-index --no-deps --no-build-isolation --wheel-dir /private/tmp/calmetrics-vforce-cross-wheels --config-settings=build-dir=/private/tmp/calmetrics-covchol-next/wheel-build`，安装到独立cross-install。完整测试命令见full日志对应工具记录，`PYTHONPATH=/private/tmp/calmetrics-vforce-cross-install python -m pytest -q`；本机进程/shm测试需允许POSIX资源。

`benchmark.py --mode timing` 与 `--mode memory-sweep` 用 `--old-install /private/tmp/calmetrics-vforce-install --new-install /private/tmp/calmetrics-vforce-cross-install --output <本目录>`，heap probe `/private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib`。脚本在import前设单核并在对照中暖JIT；不在产品源码执行Python数学。重新测试须使用新的结果目录，避免覆盖本冻结证据。OFF构建使用 `CALMETRICS_ENGINE_USE_EIGEN=OFF` / `CALMETRICS_ENGINE_USE_ACCELERATE=OFF`，Python测试须带 `CME_EXPECT_ACCELERATE=0`。

