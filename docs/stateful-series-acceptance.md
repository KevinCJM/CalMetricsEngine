# 时序递推与状态能力验收

验收日期：2026-09-21。本记录针对 `codex/stateful-series` 的本地改动，基线为
`359e8e288d385cdf932772a2680dad6f19e5b51a`。范围仅为 CalMetricsEngine；
没有切换 FundInvestmentResearchPlatform 的执行后端或启动流程。

## 实现与契约

- 原有 1–125 号算子保留；126–146 提供自适应递推、二阶滤波、cos、标量 Kalman、
  状态选择/滞回/连续确认、回撤状态、峰谷、PS、完整分段和共享字段投影。
- KAMA、Super Smoother 与分段收益使用真实 C++ 计算图组合。没有新增专用指标黑盒、
  Python 数值回调、运行期机器码构建或平台运行依赖。
- 布尔信号、int64 状态可穿过单线程、线程、进程、共享内存及 Hard Stop 路径。
  同图根输出必须同 dtype；失败占位值与业务状态通过 statuses 区分。
- 共享状态只求解一次，投影借用其内存；C++ liveness 保留底层所有者。
  prepared 输出复用与独立 snapshot 的所有权分别验证。
- `state`、`event`、`phase`、`index` 不隐式混用。峰谷修订、PS 与完整波段均为事后结果，
  本轮没有把引擎局部时序声明当作平台实时/回测发布资格。

详细定义见[设计](stateful-series-design.md)和[状态事件契约](state-event-contracts.md)。

## 可重复检查

实际环境：macOS 15.6.1、arm64、CPython 3.12.11；原生算子执行 scalar / NEON 检查。
通过本地 wheel 构建并安装到全新目录，检查 Python 包及 `_native` 都来自该目录后运行测试。
不以源码导入代替 wheel 验证。

```sh
CMAKE_GENERATOR='Unix Makefiles' python -m pip wheel \
  --no-build-isolation --no-deps . --wheel-dir /tmp/cme-wheels
python -m pip install --no-deps --target /tmp/cme-installed /tmp/cme-wheels/*.whl
PYTHONPATH=/tmp/cme-installed python -m pytest tests -q

cmake -S . -B /tmp/cme-native -DCMAKE_BUILD_TYPE=Release \
  -DCALMETRICS_ENGINE_BUILD_PYTHON=OFF -DCALMETRICS_ENGINE_BUILD_TESTS=ON
cmake --build /tmp/cme-native --parallel 4
ctest --test-dir /tmp/cme-native --output-on-failure

cmake -S . -B /tmp/cme-asan -DCMAKE_BUILD_TYPE=Debug \
  -DCALMETRICS_ENGINE_BUILD_PYTHON=OFF -DCALMETRICS_ENGINE_BUILD_TESTS=ON \
  -DCALMETRICS_ENGINE_SANITIZE=ON
cmake --build /tmp/cme-asan --parallel 4
ctest --test-dir /tmp/cme-asan --output-on-failure

PYTHONPATH=/tmp/cme-installed python tools/check_phase2_performance.py \
  --better-root /path/to/BetterSaaTaa --output-dir /tmp/cme-performance
```

wheel 构建需要项目声明的 CMake/pybind11/scikit-build-core 构建依赖。
性能参考代码只用于验收；独立安装及正式计算不导入 BetterSaaTaa 或 Numba。

## 覆盖内容

| 检查 | 关键证据 |
| --- | --- |
| `test_recursive_state.py` | KAMA / Super Smoother 独立参考、Kalman 闭式后验、初始化、缺失保留和显式重置、只读负 stride、精确大整数 |
| `test_state_events.py` | 阈值等号、保持期、确认时点、缺失/无候选区别、回撤重置、固定平台 PS 对照、事件分段几何 |
| `test_stateful_graph.py` | 真正的 AST/DAG 组合、记录名义类型、量纲、CSE 与借用寿命、分段端点、逐段隔离、嵌套作用域预算 |
| `test_series_output_types.py` | bool/int64 精确输出、inline/shared worker、序列化、Hard Stop、statuses、prepared/snapshot 所有权、Planner 资源估计 |
| `test_segment_error_status.py` | 176 项故障回归：前轮 129 项及本轮 47 项；覆盖失败标量、未知 capture/成员、嵌套子图、真实消费边界、四执行通道及 prepared 恢复 |
| `test_cpp_first_runtime.py` / `runtime_native_tests.cpp` | 缓存命中后的输入/参数/区间变更拒绝、独立结果所有权、公开规划统计及复核溢出检查 |
| `operator_native_tests.cpp` | 全部 146 个算子的 scalar / 自动 ISA 结果校验，共 60,411 项检查 |
| `stateful_native_tests.cpp` | 原生共享记录、事件边界、序列化、完整波段执行及 sanitizer 回归 |

## 功能与性能验收结果

最终 wheel：`calmetrics_engine-0.3.0-cp312-cp312-macosx_15_0_arm64.whl`。

- wheel SHA-256：`4354ec42a9e420f557baa94620f0d052b99639bb3d6c61db6fd00d6756eefd80`。
- 原生 `engine_build_id`：`157c5d54eb75ab6c6240dd0fdc6bacc066787ddd26d7fff20544b9e7462fae06`。
- 编译器：AppleClang 17.0.0.17000603；`runtime_jit=0`，真实图审计 `python_fallback=0`。
- 全量 Python：**4528 passed，0 failed，0 skipped**（4.14 秒），本轮新增 47 项故障回归及 5 项入口/计划回归。
- Release 原生 CTest：**7/7 通过**。
- ASan/UBSan 原生 CTest：**7/7 通过**。
- `git diff --check`、本轮 README/设计/契约/验收文档的本地链接检查通过。
- 本轮原性能门禁 **4/4 通过**：63 点普通 Scheduler 配对比值为 **0.947**，满足 <=1.00；
  四组 prepared/batch 比值均满足 <=0.90。前轮 1.018 的失败记录保留在下方。

早期审查修复了两个回归点：普通状态码冒充波段方向，以及零捕获嵌套分段子图的容量不足。
后者先以独立原生复现触发 ASan `heap-buffer-overflow`，修复后同一复现退出0，正式原生回归也通过。
Planner 同步计入零捕获子图按实际区间生成的数组和工作量；资源不足仍报错，不被 isolate 转为缺失。

### P1：分段异常经过信号计算丢失状态

复审发现旧 4347 项测试遗漏了 `segment_apply(1/std(x,0),…) > 0` 的下游故障状态：
原实现只写 NaN，比较输出 False 后 statuses 错误地变成0。旧测试通过不能覆盖这项契约，
原先完整验收结论因此不足。

修复使用独立的逐位置故障状态，沿受影响依赖传播，并纳入 Planner 内存预算和实际工作区审计。
异常位置不再参加后续逐元素数值计算，但仍执行完整结构校验。未完成段等普通 NaN 不冒充执行异常。
对于没有精确位置映射的非局部节点，保守让整个依赖节点失效；不宣称已支持递推状态的精确故障恢复。

复现价格 `[7,2,2,2,3,4,5,6,7]`、事件 `[0,-1,0,1,0,-1,0,1,0]`：
修复后比较结果仍为 `[False,False,False,True,True,True,True,False,False]`，
statuses 为 **`[0,4,4,0,0,0,0,0,0]`**。坏段保留错误，健康段、独立根及未完成边界保持各自契约。
新回归在旧 wheel 上为 **24 失败 / 8 通过**；最终修复 wheel 上 **32/32 通过**。

### P2：故障提前传播跳过结构校验

P1 后的复审发现：非局部节点收到故障后提前 `continue`，会掩盖独立输入中的非法 mask、
状态码、索引及长度错误。已用同一输入的健康/故障源对照复现；4379 项测试通过不足以覆盖此契约。

修复保留失败值的类型与可推导的精确几何，未知长度显式标记不可知。
故障路径与正常路径共用 canonical 结构校验，既不执行失败数值，也不读取不可用 payload。
作用域按真实成员把 capture 故障传入同一子图执行器，继续检查健康分支；未消费的坏 mask 不误报。
递归子图的隔离工作区及暂存状态进入 Planner，实际保留容量进入审计。

测试覆盖直接和多跳递推、规约、状态参数、gather 边界、scope selector/body、四执行通道、
prepared 反复坏/好切换、未知窗口，以及 `returns` 滚动计数对失败数据的安全处理。
旧 wheel 已复现 **25 个新增失败用例**；该轮专项 **99/99 通过**，原 P1 回归全部保留。
进程原有错误运输契约仍是 `RuntimeError("native worker: INVALID_MASK")`，
线程是 `ValueError("INVALID_MASK")`，均由同通道健康源对照锁定。
原本可隔离的 `INVALID_PARAMETER` 未升级为批次结构错误。

### P2 补充：bisect 子作用域结构校验

再次复审发现 `scope_maps_status` 显式排除了 `bisect`，导致捕获值失败时整个子图被跳过。
上一轮 **4446 项通过**没有覆盖该路径，不能作为此项已关闭的证据。
在上一轮 wheel 上，新增本机回归复现 **12 失败 / 12 通过**；失败源掩盖了非法 mask、
长度不匹配及非法状态码，健康源仍正确抛错。

修复让 `bisect` 复用现有子图故障传入机制：每个 capture 按自身完整长度汇总故障，
在上下界两个实际端点检查可判定的子图结构；失败 payload 不参与数值运算，随后保留故障结果，
不执行求根迭代。正常求根、独立指标、线程/进程异常运输和 prepared 复用契约保持原样。
没有新算子或数值算法；子图状态所需内存已在上一轮 Planner 预算中计入。

新增 30 项 Python 用例覆盖局部/整体捕获故障、嵌套 filter、两个端点、不同捕获长度、
四个执行通道、prepared 坏/好切换及独立快照。原生回归还覆盖计划序列化和复用。
该轮全量 **4476/4476**、Release **7/7**、ASan/UBSan **7/7** 通过；
实际安装路径和原生审计构建身份与该轮源码哈希一致。

首次本机检查受到沙箱 `shm_open` 限制，允许本地 IPC 后完成全量验证；
同时修正新测试把正常缺失值误认为健康数值的夹具：显式排除未完成段 NaN，
独立原生根使用有效值计数。没有更改生产缺失值契约。

### P2 补充：失败标量与未知长度仍须校验子图

后续复审指出，失败的 body 标量或未知 capture 长度仍会触发提前返回。
在上一轮 wheel 上，首批新增对照复现 **15 失败 / 15 通过**，覆盖 bisect、block、filter、group、segment 和 rolling。
这说明上一轮 4476 项通过仍不足以关闭全部子作用域路径。

本轮把成员关系与失败 body 参数分开；成员已知时继续按真实范围传递故障和验证健康输入。
成员未知时复用同一子图执行器，保留未知几何与不可用 payload，只检查不依赖失败值的约束。
bisect 仍保留各 capture 自身的长度，未知长度不阻止其他健康 capture 的 INVALID_MASK 等错误抛出。
未知成员不会被伪装成全选或空选，未消费的坏数据不误报；独立结果与数值隔离规则保持原样。

新增 47 项故障测试覆盖失败标量、未知长度/控制参数、嵌套、prepared 坏/好恢复、原生运输以及未消费数据边界。
另有 5 项 Python 测试验证缓存命中后输入和参数的变更检查、结果独立所有权与计划几何复核。
原生回归覆盖计划序列化、公开 inspect 统计保留，以及仅复核身份时的观测总量溢出检查。
最终 wheel 全量 **4528/4528**、Release **7/7**、ASan/UBSan **7/7** 通过，构建身份与冻结源码一致。

### 性能记录

P1 修复的首次性能复核中，63 点普通 Scheduler 比值为 **1.018**，超过 <=1.00 门槛；
其他三组及全部 prepared 比值通过。这次失败保留为实测证据，不能沿用旧版本的通过结论。
随后移除未启用位置追踪时的无用缓冲重置，并将同一执行实现按是否需要追踪进行 AOT 模板实例化，
由图的依赖元数据选择，避免普通图逐节点承担新增分支。没有运行期编译，也没有另存历史计算实现。
该轮重新构建后四组通过；前轮 P2（4446 项测试对应构建）也曾四组通过。
前轮 `bisect` 修复后的冻结 wheel 重跑四组门禁，63 点普通 Scheduler 比值再次为 **1.018**，
超过 <=1.00，因此该轮性能验收 **未通过**。其余三组及全部 prepared 比值通过。
没有降低阈值、替换工作负载或反复重跑直至通过；前轮通过结果不替代本轮失败记录。

本轮按顺序完成 prepared 所有权用法说明、普通入口键/dtype 构造精简，以及计划复核统计和临时数组移除。
公开普通入口仍独立分配结果，输入变更与原有几何/计划身份检查保留。
冻结新 wheel 后运行一次原门禁，四组全部通过；没有更改门槛、基准脚本或基准工作负载。

以下为上述最终 wheel 的性能实测：

| 工作负载 | NJIT 中位数 | C++ prepared 中位数 | 配对 C++/NJIT |
| --- | ---: | ---: | ---: |
| 500 产品 × 2520 历史点 × 12 区间 × 16 指标 | 50.656 ms | 34.514 ms | 0.681 |
| 1000 产品 × 2520 历史点 × 12 区间 × 16 指标 | 102.005 ms | 78.487 ms | 0.769 |
| 1 产品 × 63 历史点 × 1 区间 × 5 指标 | 4.583 µs | 3.083 µs | 0.673 |
| 1 产品 × 252 历史点 × 1 区间 × 5 指标 | 9.625 µs | 7.084 µs | 0.736 |

prepared 门槛保持 C++/NJIT <=0.90。两组微工作负载的普通 `Scheduler.execute`
中位数分别为 4.458 µs、8.416 µs，配对比值分别为 **0.947**、**0.874**；原门槛仍为 <=1.00，每种路径独立配对测量，
不能把另一组 NJIT 中位数当成其分母。未降低阈值或改变工作负载以通过验收。

本机最终原始记录位于 `/private/tmp/calmetrics-scope-final-{pytest,ctest,sanitized}.log`
及 `/private/tmp/calmetrics-scope-final-performance/`；旧 wheel 的本轮复现为
`/private/tmp/calmetrics-partial-scope-old.log`。前轮 bisect 失败门禁保留在
`/private/tmp/calmetrics-bisect-performance/`，更早证据保留在
`/private/tmp/calmetrics-structural-final-*`；P1 首次性能失败保存在
`/private/tmp/calmetrics-segment-status-performance/`。最终 wheel 及独立安装目录分别为
`/private/tmp/calmetrics-scope-final-wheels/`、`/private/tmp/calmetrics-scope-final-install/`。
这些是本次实测产物，
复现应使用上面的命令重新生成，不能依赖临时目录长期存在。

## PR 阶段的依赖与 CI 兼容性

首次 PR CI 在 NumPy 2.5.3 下复现 3 项失败：测试故意修改已绑定 ndarray 的 dtype/shape，
被新增的弃用警告提前中断。只在这两项测试中局部匹配预期警告；保留原数组、原生拒绝断言及全局 warnings-as-errors。
修正后 NumPy 2.5.3 全量 **4528/4528 通过**。没有用新 view 代替原地变更来绕开所有权检查。

Windows wheel 的构建和原生测试通过，但 cibuildwheel 4.2.1 用 `shell=True` 执行
`pip install pytest>=8,<10`，版本约束进入 cmd 重定向解析，依赖安装失败。
改用既有 `test` extra 安装相同 pytest 约束，避免独立 shell 参数，也避免重复维护测试依赖。

Linux sanitizer 的原生 7/7 通过，Python 在首个原生异常边界测试处退出且原诊断被 fd 捕获隐藏。
工作流改为同时预加载 ASan 和 libstdc++，让 C++ 异常拦截符号在初始化时可用；pytest 增加 `-s` 保留原生诊断。
全部原有测试、平台、sanitizer 和 CI 必需检查保持启用；实际远端结果以 PR 当前 HEAD 为准。

调整 pyproject 测试依赖配置后重新构建并独立安装 wheel：

- wheel SHA-256：`816b93092a20684a5a052fb7217c10b0232803985d1dc375ffb3b49eec30c7cb`。
- engine_build_id：`1a81843ccb1f0b244f782edaf28a15976a9ce13ca2047e1b8d8bb50cc8900ea5`，已按当前源码及构建配置独立复核。
- NumPy 2.5.3 全量：**4528 passed**（4.04 秒）；Ruff 0.16.8 的 `src tests tools` 检查通过。
- 产物、安装及日志分别在 `/private/tmp/calmetrics-pr1-wheels/`、`/private/tmp/calmetrics-pr1-install/`、`/private/tmp/calmetrics-pr1-pytest.log`。

C++ 源码没有因这组兼容性修订改变；上文性能和原生 sanitizer 数字来自明确记录的功能验收构建，
不是声称对每次仅测试/CI 配置变更重复测量了性能。

## PR 阶段的 manylinux CPU 准入死锁修复（2026-09-21）

Linux wheel 长时间运行不是编译慢。历史主线与旧 PR 的 manylinux2014 CPython 3.10
测试在约 3% 后停止；用当前 PR 源码和 CI 固定的 ARM64 镜像
`sha256:f4cd164263e4ec2b7da7ee40b319bb5e30f0d7a2abd7ad4730e716a512dfb529`
独立复现，停在 `test_concurrent_requests_share_native_cpu_admission`。

纯 C++ 最小复现同样停在 CPU token 的释放：默认 `steady_clock::time_point::max()`
被传给 `condition_variable::wait_until`，旧 libstdc++ 转换到系统时钟时溢出，
在持锁情况下重复等待，使释放配额的一方无法进入。无限期限现在使用带谓词的
`wait`；有限期限仍使用原 `wait_until`，保留关闭检查、CPU 上限和资源所有权。
回归覆盖有限/无限竞争、有限超时后的配额复用、扩容唤醒和关闭唤醒。

同一 manylinux 环境中，原最小程序超时，修复后正常唤醒退出；原生 **7/7**，
已修复并安装的 CPython 3.10 wheel 全量 **4528 passed（6.87 秒）**。
本机新 Release wheel 全量 **4528 passed（4.40 秒）**，原生与 ASan/UBSan 各 **7/7**。

- 本机 wheel SHA-256：`5ebe5fd9afa7acfad0b771386ade14fe29ef294107f489f379940ea005d3dc8e`。
- engine_build_id：`7e914bdeabdc45998d63fdbbdf7f2de81db63ccf16d4721bdba2ed5ce96a16c3`，已独立核对源码身份。
- 复现与修复日志：`/private/tmp/calmetrics-linux-repro.log`、`/private/tmp/calmetrics-linux-fixed.log`。
- 本机日志与 wheel：`/private/tmp/calmetrics-cpuwait-*.log`、`/private/tmp/calmetrics-cpuwait-wheels/`。

修复后的 wheel 按上一轮相同 10 CPU 预算复跑原四组门禁，全部通过：

| 工作负载 | prepared/batch C++/NJIT | 普通 Scheduler C++/NJIT |
| --- | ---: | ---: |
| 500 产品 × 2520 点 × 12 区间 × 16 指标 | 0.632 | — |
| 1000 产品 × 2520 点 × 12 区间 × 16 指标 | 0.644 | — |
| 1 产品 × 63 点 × 1 区间 × 5 指标 | 0.598 | 0.929 |
| 1 产品 × 252 点 × 1 区间 × 5 指标 | 0.710 | 0.869 |

阈值仍为 prepared/batch <=0.90、普通入口 <=1.00；记录在
`/private/tmp/calmetrics-cpuwait-performance-cpu10/`。另一次 8 CPU 预算的四组门禁也通过，
保存在 `/private/tmp/calmetrics-cpuwait-performance/`；不把两个 CPU 预算的耗时混为直接对照。

wheel CI 改为打印测试名称，60 秒无响应时输出 Python 堆栈，单个 wheel 矩阵任务
最多执行 45 分钟；原生 CPU 准入回归设置 60 秒 CTest 超时。超时仍属于失败，
没有跳过任何测试或平台。旧卡住任务已停止，远端通过条件仍以修复提交的全量 CI 和 Bot 结论为准。

## PR 阶段的 NumPy / manylinux2014 测试环境修复（2026-09-21）

CPU 准入修复提交 `6fde7adaa2c3541df327cf85a0006f29069a3a0a` 的远端两种 Linux
架构已分别通过 CPython 3.10、3.11 全量 4528 项测试，证明原卡点已消除。
随后 CPython 3.12 安装 NumPy 2.5.3 失败：上游二进制包最低为 glibc 2.27，
pip 在 glibc 2.17 的 manylinux2014 中改为源码构建，而源码要求 GCC >=10.3，
镜像只有 GCC 10.2。该次 CI 仍按失败处理，没有合并。

保留 manylinux2014 构建与全部 wheel 测试，仅在其测试虚拟环境预装
`numpy>=1.26,<2.5`。运行依赖仍为 `numpy>=1.26,<3`，没有降低引擎的 Linux
兼容范围，也没有跳过 Python 3.14、musllinux 或其他平台。
glibc 2.17 用户使用新版 NumPy 仍须满足上游编译要求；引擎 wheel 的兼容标签
不代表所有第三方依赖都有同范围的预编译包。

两个必需 Linux 检查还在 Ubuntu 24.04 上，为 CPython 3.10–3.14 各建独立环境，
安装刚构建的同一个 repaired manylinux wheel 与该 Python 支持的最新 NumPy，
再次执行全量测试。每个 ABI 必须恰好对应一个 wheel；缺失/重复、依赖冲突、
安装错误或测试失败都使检查失败，不上传未经验证的产物。
这样同时验证旧 glibc 与最新 NumPy；旧环境的编译器约束不能掩盖新版依赖回归。

本地使用 cibuildwheel 4.2.1 和同一固定 manylinux2014 ARM64 镜像，实际构建、
repair、安装并测试 CPython 3.12 / 3.13 / 3.14；NumPy 均为 2.4.6，
分别 **4528 passed（5.73 / 6.56 / 6.10 秒）**。
完整日志 `/private/tmp/calmetrics-numpy-compat-cibw.log`，三个 wheel 保存在
`/private/tmp/calmetrics-numpy-compat-wheels/`。
工作流的 YAML、TOML、Bash 语法和缺少 wheel 时的失败退出也已验证。
同一 CPython 3.14 wheel 在官方现代 Linux Python 3.14.7 镜像、NumPy 2.5.3 下，
使用新增 CI 脚本的对应分支和完整项目 pytest 配置（含 warnings-as-errors），
再次 **4528 passed（6.99 秒）**；日志为
`/private/tmp/calmetrics-numpy-modern-config-cp314.log`。
完整双架构、多 Python、musllinux 与新 Linux 路径仍由最新提交的远端 CI 验收。

本次仅调整 CI / 测试环境，C++、Python 数值接口与测试断言均未改变。
上述 CPU 准入修复的性能结果继续作为相同 C++ 源码的证据；没有把仅 CI 配置
变化后的构建身份或性能写成重新测量的结果。

## PR 分段投影边界校验修复（2026-09-21）

Bot 提出的 P2 属实：直接传入形状正确但端点越界的 `T×2` int64 矩阵，
`segment_starts` / `segment_ends` 只检查形状就返回视图。旧 wheel 上
`[[0,3],[0,3],[-1,-1]]` 被接受并返回 `[0,0,-1]` / `[3,3,-1]`，
新增回归在旧实现上因未抛出错误而失败。

两种投影现在在 payload 可用时复用同一个 `segment_geometry` 校验器，
检查完整的两列、端点范围、半开区间成员一致性和未知哨兵，再借用列视图。
`out=` 和 requirements 查询共用同一准备路径；失败不会写入调用方输出。
失败上游 payload 不可用时仍只验证可判定的 dtype/shape，不读取空指针。
Planner 不再将这两个投影视为无需扫描的字段读取，计入 O(T) 校验成本。
合法输入的数值、半开区间、零拷贝、stride、所有权与空输入契约不变。

新增 54 项 Python 回归覆盖两种投影、六类非法边界、借用/显式输出/requirements，
以及连续、正/负 stride、只读、空输入、未知段和相邻段。原生回归另验证
普通准备和结构校验均拒绝越界，以及 payload/geometry 不可用时的安全行为。

当前源码重新构建并安装 wheel：Python **4582 passed（4.55 秒）**；
原生 Release 与 ASan/UBSan **各 7/7**。本机沙箱最初禁止 `shm_open`，
在正常权限下复跑全量通过，没有改动共享内存测试。Ruff 0.16.8 通过。
wheel SHA-256：`7a729994794f2c6bad8f0b13c9ce70c256c5c22d27ca2a1b7a8531004aa2e085`；
源码独立复核的 engine_build_id：
`049b6a73b4a9a35b5dd3924e878594f4b83a97fbe0cd7a128e2493ca1b3f33c2`。

相同 10 CPU / 7 次配对的原四组性能门禁全部通过：batch 500/1000 产品
比值 **0.662 / 0.650**；63/252 点 prepared **0.588 / 0.703**，
普通入口 **0.965 / 0.897**。原始记录在
`/private/tmp/calmetrics-segment-boundary-performance/`；阈值和工作负载没有改变。
这些标准工作负载不包含分段投影，不能把结果解释为新增校验无成本。

此外，之前 Bot 关于 setup-python 多版本输入的 P1 有官方源码反证；
`9a0742b` 的 Linux ARM job 106313372063 已真实执行五版本 setup 并通过完整检查。
其日志 47958–47962 行记录五个解释器成功安装，不能据此跳过新提交的 CI 或 Bot 复审。

## PR 普通滚动与分组异常溯源修复（2026-09-21）

Bot 对 `1750723` 提出的两条 P2 均已复现：没有分段作用域时，
`rolling_apply(1/std(x,0),2)>0` 与 `group_apply(1/std(x,0),key)>0`
会将常量窗口/组的除零转为 False，同时返回 status=0。
原因是原位置状态可达性只由分段作用域产生，两个已有 catch 分支没有记录普通局部失败。

编译元数据现在将普通 rolling/group 也作为位置故障源；窗口只标记该右端输出，
分组标记原始成员。现有下游传播、worker 状态传输、独立根隔离与 Planner 最坏内存
准入共用同一套元数据。组 catch 必须保留 failure，滚动 catch 在 isolate 下必须标记。
block/filter/bisect 的普通异常原本可向外抛出，没有用 NaN 吞错；但其中的嵌套 rolling
会提前吞错，已独立复现四种外层作用域。因此子程序显式继承异常传播上下文，让异常
到达能够记录成员范围的隔离作用域，根请求每次重置该上下文。

追踪覆盖范围不再兼任严格 rolling 的历史策略开关；单独的分段存在标记保持该兼容规则。
正常返回的 NaN、预热、严格 rolling 的原数值异常转 NaN 行为，以及严格 group 抛错规则
均保留。没有通过把所有子程序切到 isolate 模式来改变普通缺失语义。

新增 37 项 Python 回归覆盖 bool/int64、finite_mask、四执行通道、非连续组、独立根、
跨区间、序列化、prepared 状态恢复/快照、四类嵌套作用域和故障前内存准入。
新增原生序列化回归验证普通窗口/组的精确失败位置。旧 wheel 的直接及四类嵌套用例均失败；
严格和普通缺失对照在旧 wheel 通过。新 wheel 全量 **4619 passed（5.92 秒）**；
原生与 ASan/UBSan **各 7/7**，Ruff 0.16.8 全量通过；补强同一 Scheduler 从 isolate
切回 strict 的测试后，三个兼容专项再次通过。

wheel SHA-256：`4caf1a2a53aa71d5311927e75f4cd5cc2466911479897d340e7a3d043bbf2731`；
独立复核源码一致的 engine_build_id：
`c5dfec2ad6bde5290db2cb10718154db2824d2a7279177ddd5b6065d21808dce`。
产物与日志前缀 `/private/tmp/calmetrics-scope-provenance-`。
相同 10 CPU / 7 次配对的原性能门禁全部通过：batch **0.734 / 0.639**，
63/252 点 prepared **0.658 / 0.733**、普通入口 **0.931 / 0.868**。
原始记录 `/private/tmp/calmetrics-scope-provenance-performance/`，阈值没有改变。
这些基准不代表所有窗口/分组故障路径的性能。

前一提交 `1750723` 的远端 10 项 CI 已全部通过；本次 C++ 修复仍须重新构建并由
最新提交的完整 CI 和真实 Bot 结论验收，旧提交的成功不作为合并依据。

## 证据边界

- 本机验证不代替 Linux/Windows、其他 Python 版本或 x86 SIMD CI。
- 本轮性能门禁衡量原有 16 指标及 5 指标工作负载，不能推出每个新增算子均快于 NJIT。
- 测试证明支持路径上的输入视图/投影共享，以及 shared 输出无最终拷贝；
  独立输出、必要工作区和 inline IPC 仍有明确分配或传输，不宣称整个计算零分配。
- Kalman 限于一维随机游走估计；多维 Kalman、公共同图混合 dtype 输出、公开矩阵根均未实现。
- 引擎能力补齐不等于 MetricsFactory 所有定义已迁移，也不等于平台时序适配、前后端门禁已验收。
