# Ordered Eigen 投影 producer：2026-10-03 证据归档

本目录保存八资产 Eigen Array producer 的筛选、采用、审核、回归和完整请求时间/内存证据。原始材料来自根代理已完成的执行；归档只读取、复制、核对 SHA 和整理文字，没有重新运行构建、数值、测试、性能或内存任务。

**生产 wheel 的大任务完整投影比强共享 NJIT 快约 28%–55%，原生与专项验证通过；完整输出内存没有显著优势。完整 MC compact 链、随机流兼容和平台接入仍未完成。**

## 1. 身份与历史范围

源码/工具链 build ID：`5dcec3922588f4f912a2e6afa55775ac83865771e6e6e2363f46f941e3c1aa97`。生产源码 `cpp/affine_projection.cpp` SHA 为 `ff0119105ae53af372cb74f4426ef517f588a3404bf5eb481def15f451011003`；共同数学体 `cpp/operators/affine_contract.hpp` SHA 为 `1afc6798ff05a26e1bf3bca7768083f0e29b57b8925ca15cdd79982d293da38e`。

| 身份 | 镜像筛选 | 已采用的生产 wheel |
| --- | --- | --- |
| 完整清单 | [identity-screen.json](identity-screen.json) | [identity-production.json](identity-production.json) |
| wheel SHA | `cc70593818693e8fad68c45319a4f4ee18cc99a81e30d566a4be94dd2353ddba` | `ab3cc83cb43433177b8d4feb811bc5e3508f141609a9bedfebf55426187335f4` |
| native SHA | `af8b93b3aebc6fff606be7bf2f4e03e1c0fd3f82e180ecb8a68409a82dddcf7f` | `0a947e81b71655221d4e377108f530b4d6e7ecadadbaaa740a2748e6d07cffac` |
| worker SHA | `3ee34281c5639b632aa5e10aa5556757b3fe65420a3bca2d57e4694f4c048303` | `d6a09da8a5963a5b40792597d059e2383d6ad375b186b10396b9e824275e47a4` |

两套源码/工具链 build ID 相同，二进制字节不同。筛选计时绑定 screen manifest，采用后的生产复测绑定 production manifest，不能只按 build ID 合并。根代理的[终态身份](identity-production-final.json)与生产清单字节一致；[核对日志](identity-eigen-final.log)和[六报告/八 PID 一致性检查](production-eigen-consistency.json)保存原文。一致性检查只核对既有终态 JSON，不是新的数值测试。[archive-manifest.json](archive-manifest.json)记录归档材料的原始绝对位置、目标名、大小和 SHA。

前一[固定随机带投影归档](../2026-10-03-mc-projection-integration/README.md)的 dense `63a1dacd485576f5d148be0fb577bf2d277c91a00ab35c970e6bc88d2330a39a` 为历史证据，生产 producer 当时 SHA 为 `77bb…`。其大型单组合落后约 13% 是当时的事实；本次新分支提供独立复证，没有改写原始数字、失败或日志。早期 `identity.json` 也不能代替正式 dense 或本轮身份。

## 2. 实现、数学与执行范围

复用 Eigen 3.4.0 Array，仅跨八个相互独立的资产 lanes 更新；factor 保持升序，initial 先入，系数左乘，具体 product 求值后再相加。scalar 与 Eigen 路径使用同一个 canonical `ordered_fold`；没有 GEMM、FMA、factor subtotal、零系数跳过或私有并行。每个资产最终仍调用相同 `std::expm1`。

启用条件为 Eigen 可用、执行线程最近舍入、非空 row、F>0、A≥8、F×A≤16,384。必要的连续系数 packing 最多 128 KiB：Planner 保守预留，实际 asset stride≠1 才在每次执行 pack 一次；多区间复用只发生在同一请求，下一请求重新读取当前输入内容。超限、非最近舍入、窄资产、F0 和禁库保留 scalar 职责。严格 closure 不能证明的图，包括公开 R/intermediate observer、额外消费者、isolate 或 scope，继续 canonical 路径。

逻辑 DAG、period 的 weight-first/asset 升序 fold、贡献的 step→path 单状态跨 tile fold、最后除 path 数、C/F 根输出与独立所有权保持。[静态复审原文](eigen-production-review.md.txt)给出 pack 尾区不重叠、空轴、错误/取消、预算和所有权证明；原文的“16 项尚未执行”是写作时状态，后续终态见第 5 节。

正式计时范围为健康、已变换的固定 factor tape，所有提供资产均选中，两侧都返回全部 period 矩阵与每个资产贡献。CPU=1、NJIT 已预热、六轮交替先后配对；输入准备、图构建、同签名 NJIT 编译分别记录，不计入热态执行中位数。

[reference.py](reference.py)是自包含数学参考，不 import 平台 runtime。`service_chain`保留逐组合重复 affine/expm1 的结构；本轮主要对照使用额外的强 `shared_chain`，同一 tape 的 producer 只计算一次，工作向量为资产长度，**也没有完整 N×A 的 L/R**。不能把取消 L/R 物化说成相对该 NJIT 节省了 1 GB 或 4 GB。

## 3. 三组镜像筛选

实际 driver：[screen-bench.py](screen-bench.py)，SHA `c32450e31f9ea910fafd5460033d671fbcbaf6e03abcf4324a19a33a8ed091b5`。它在加载身份 helper 后显式使用 `CME_SCREEN_SOURCE_ROOT` 指定镜像，并记录 Eigen 分支命中。原始结果绑定 screen native SHA。

| N / T / P / F / A / 组合 | C++ ms | 强共享 NJIT ms | C++ / NJIT | 原始结果 |
| --- | ---: | ---: | ---: | --- |
| 2M / 200 / 10,000 / 4 / 32 / 1 | 206.513 | 286.964 | 0.720 | [JSON](screen-large-one.json)、[日志](screen-large-one.log) |
| 2M / 200 / 10,000 / 4 / 32 / 8 | 370.858 | 658.389 | 0.563 | [JSON](screen-large-eight.json)、[日志](screen-large-eight.log) |
| 250k / 5 / 50,000 / 32 / 32 / 8 | 65.149 | 146.973 | 0.443 | [JSON](screen-wide-eight.json)、[日志](screen-wide-eight.log) |

三组所有输出的最大绝对差为 0，两代独立结果检查通过，测量前后源码不变。对应实际 Eigen lanes 为 64M / 64M / 8M，coefficient pack 为 1,024 / 1,024 / 8,192 B。这些健康夹具不能代替异常边界、跨平台或完整 MC 验收。

## 4. 采用后的生产完整请求复测

实际 driver：[bench.py](bench.py)，SHA `6d43e5e537e12299dd141f7112202daa2e2d9b876e1cda44fcd754d02d94cb63`；六组串行调用见[perf-eigen-cases.py](perf-eigen-cases.py)及[总日志](perf-production-eigen.log)。全部绑定 production manifest/native SHA，不能换入筛选计时。该 driver 已直接检查原 NJIT 贡献 ndarray 的两代所有权。

| 完整输出请求 | C++ ms | 强共享 NJIT ms | C++ / NJIT | 原始结果 |
| --- | ---: | ---: | ---: | --- |
| N2M/F4/A32/C1，ordinary，C 根 | 205.029 | 286.081 | 0.717 | [JSON](large-single-shared-eigen.json)、[日志](large-single-shared-eigen.log) |
| N2M/F4/A32/C8，ordinary，C 根 | 366.426 | 656.839 | 0.558 | [JSON](large-shared-eigen.json)、[日志](large-shared-eigen.log) |
| N250k/F32/A32/C8，ordinary，C 根 | 65.047 | 145.928 | 0.446 | [JSON](wide-shared-eigen.json)、[日志](wide-shared-eigen.log) |
| N2M/F4/A32/C8，ordinary，F 根 | 362.159 | 659.931 | 0.549 | [JSON](large-f-shared-eigen.json)、[日志](large-f-shared-eigen.log) |
| N2M/F4/A32/C8，prepared snapshot，C 根 | 369.955 | 659.138 | 0.561 | [JSON](large-prepared-shared-eigen.json)、[日志](large-prepared-shared-eigen.log) |
| N252/F4/A32/C1，ordinary，C 根 | 0.067729 | 0.038250 | 1.771 | [JSON](small-shared-eigen.json)、[日志](small-shared-eigen.log) |

大任务完整请求节省约 81–298 ms，较强共享 NJIT 快约 28%–55%。小任务多约 29 微秒；保留严格比值作为诊断，按用户当前的小任务政策不据此否决。不能把这些已测形状扩展为全算子或所有机器上的速度保证。

每组完整输出最大绝对差为 0，独立两代结果和源码不变检查通过，实际 Eigen 分支命中，`numeric_arena_bytes=0`、Python 数值回退/算子调用/请求期编译为 0。N2M/C1 返回 33 根、每代 16,000,256 B；N2M/C8 返回 264 根、每代 128,002,048 B。arena 0 只说明该闭包不生成全量 L/R 中间矩阵，不能当成请求内存为 0。

计时进程每次调用保留该侧旧结果，同时保留另一侧的一份结果；产生新结果时瞬间可有三份数学结果。这是公平的逐后端两代替换，不是总 RSS 的两份结果实验。内存另用 fresh 子进程测量。

## 5. 测试与审核终态

| 验证 | 终态原始证据 |
| --- | --- |
| 生产 wheel 原 suite 全量 | [11,419 通过，144.39 秒](full-python-eigen.log)；运行在永久新测试复制前 |
| 镜像位置的新增 Eigen 专项文件 | [43 通过，0.93 秒](screen-eigen-focused.log)；文件位于镜像，但这次对生产 `install-eigen` 执行 |
| 采用后生产联合专项 | [273 通过，1.99 秒](focused-production-eigen.log)：原语 193、Excel 37、新 Eigen 43 |
| 新 Eigen 传输/预算专项 | [16 通过，1.21 秒](eigen-transport.log) |
| ON 原生 | [9/9 通过，4.77 秒](native-eigen.log)，含 A17 fixture；[构建日志](native-build-eigen.log) |
| ASan/UBSan 原生 | [9/9 通过，102.16 秒](asan-eigen.log)，含 A17 fixture；[构建日志](asan-build-eigen.log) |
| 三库 OFF 原生 | [9/9 通过，10.97 秒](off-eigen.log)，含 A17 scalar duty；[构建日志](off-build-eigen.log) |
| lint | [原始日志](ruff-eigen.log) |

这些是不同范围的运行，不写成新增后的 11,462 或 11,478 全量重新通过。采用后的 273 联合复验覆盖同一 43 项，并以 `CME_SCREEN_INSTALL` 的 autouse 路径断言绑定所选生产包；模块须显式 `CALMETRICS_TEST_EXPECT_EIGEN=1`，普通环境 skip 不算新路径通过。

新 43 项覆盖 A8/9/17、C/F/strided/negative input、C/F 根、多区间不同 tile 容量/pack 位置、prepared 同对象内容变化、F0/A0/空轴、cap 恰等/超限、四舍入、NaN/Inf/零产品、initial-first、状态和独立所有权。IEEE 比较所有非 NaN 位与 NaN 类别，不承诺 NaN payload 保真；公开 R 根迫使 canonical 退准入，另以公开 scalar 原数学体复核 R。

新增 16 项使用 A17 实际命中新分支，覆盖 thread、inline process、shared process、C/F 根、padded/negative input，每 chunk 的 544 B pack/lanes、同 CPU≤2、独立两代结果、close/源释放后的寿命，以及总预算−1拒绝/恰好准入。其静态[准备片段](eigen-extra-transport-tests.py.txt)与实际[永久测试源码](test_affine_projection_transport.py.txt)分别保留；不能把准备时状态当成最终未执行，也不能将实际测试归结为静态审核。

源码测试副本使用 `.py.txt`/`.hpp.txt`，供版本和覆盖审查，避免成为另一个 pytest/CodeGraph 入口：

- [Eigen 43 项](test_affine_projection_library.py.txt)、[Eigen 传输/预算 16 项](test_affine_projection_transport.py.txt)。
- [原语与共享图 helper/回归](test_mc_projection_primitives.py.txt)、[Excel 原语/共享图测试](test_mc_projection_excel.py.txt)。
- [原生 A17 及边界 fixture](affine_projection_native_cases.hpp.txt)。

这不是所有可编辑 DAG 的穷举。Linux/Windows directed-rounds、worker directed-rounds 与该分支 mid-flight cancellation 的运行时见证仍未验收；原生目前覆盖预取消和实际 pack 后算术失败/恢复。

## 6. 本轮 fresh heap/RSS：没有显著内存优势

四种 backend 各在两个 fresh 子进程测量 heap/RSS，共八个子进程；shape 均 N2M/T200/P10,000/F4/A32/C8、CPU=1。所有报告身份、driver/helper/probe SHA 和测量前后源码检查保留，原始材料如下：

| backend | cold tracked heap 峰 B | warm tracked heap 峰 B | RSS 执行增量 B | 原始报告 |
| --- | ---: | ---: | ---: | --- |
| C++ ordinary | 130,736,963 | 258,762,931 | 258,031,616 | [JSON](memory-ordinary-eigen.json)、[日志](memory-ordinary-eigen.log) |
| C++ prepared snapshot | 130,737,051 | 258,762,995 | 257,769,472 | [JSON](memory-prepared-eigen.json)、[日志](memory-prepared-eigen.log) |
| 服务型 NJIT | 128,012,408 | 256,014,216 | 256,212,992 | [JSON](memory-service-njit-eigen.json)、[日志](memory-service-njit-eigen.log) |
| 强共享 NJIT | 128,012,408 | 256,014,216 | 256,245,760 | [JSON](memory-shared-njit-eigen.json)、[日志](memory-shared-njit-eigen.log) |

ordinary 比强共享 NJIT 的 warm tracked peak 多 2,748,715 B，RSS 增量多 1,785,856 B；本轮没有显著内存优势。每代数学输出 128,002,048 B，两代输出本身约 256 MB，C++ 还包含 graph/audit/typed ownership 等开销。1,024 B 系数 packing 如实记账，不作为值得继续反复优化的主要目标。

两侧均保留独立旧结果并产生新结果。共同输入和同签名 NJIT 编译在测量 epoch 外。heap 用 `continuous-live-1` 保留连续 setup/cold/warm 存活量，避免把阶段峰相加；graph/setup、首次执行、warmups 和替换按 helper 原口径记录。RSS 在 owner/graph 构造后取 baseline，采样首次完整执行及两代替换；采样间隔 0.5 ms，idle 为 0，未随意扣减。申请字节与 RSS 不混用。

申请追踪仍有限制：C++ cold/warm `untracked_frees=10`，NJIT cold 为 3、warm 为 17，`table_overflows=0`；值是连续 epoch 的累计计数。它可能包含 epoch 之前合法分配的释放或未拦截路径，不能仅据此判定泄漏，也不能把这些结果称为全面分配审计。Python arena 子分配、mmap、其他分配器与 RSS 采样盲区保留。

测量源码为 [memory.py](memory.py)、[memory-eigen-cases.py](memory-eigen-cases.py)；[总日志](memory-production-eigen.log)、[benchmark helper 原文](benchmark-memory-helper-source.py)、[heap probe 源码文本](heap-probe-source.cpp.txt)和每个 heap JSON 的实际 probe SHA 一并保存，不随归档复制二进制。

## 7. 实际 ISA、构建与复现条件

[isa-source-evidence.md](isa-source-evidence.md)区分数学源码、静态 object 指令、runtime lanes 与完整请求测量。[production-affine-isa.json](production-affine-isa.json)绑定生产 wheel-build 的 object、[反汇编](production-affine-disassembly.txt)和[编译命令](compiler-command-eigen.txt)：记录 40 条独立 NEON double `fmul/fadd`、无 FMA 指令，编译使用 `-fno-fast-math -ffp-contract=off`。这是静态 object 证据，不是 CPU 采样。

镜像 object 的[原始反汇编](screen-affine-projection-disassembly.txt)另存。两处 object SHA 相同，反汇编文本因位置前缀不同而 SHA 不同；whole wheel/native/worker 身份仍按第 1 节分开。`vector_elements`只计实际 Eigen Array lanes，不是指令数或跨平台 ISA 保证。

| 材料 | 作用与复现限制 |
| --- | --- |
| [screen-preparation-note.md](screen-preparation-note.md)、[prepare-mirror.sh](prepare-mirror.sh)、[screen-build.py](screen-build.py) | 原筛选准备记录；旧 guard 锁 `77bb…` 源码且拒绝覆盖 mirror，当前 `ff011910…` 生产源码不能直接运行该历史准备流程 |
| [screen-build.log](screen-build.log)、[screen-install.log](screen-install.log)、[screen-build-driver.log](screen-build-driver.log) | 镜像构建/安装原始日志 |
| [build-eigen.py](build-eigen.py)、[build-eigen.log](build-eigen.log)、[install-eigen.log](install-eigen.log)、[build-eigen-driver.log](build-eigen-driver.log) | 已采用 production wheel 的构建/安装原始命令及日志 |
| [identity-helper-source.py](identity-helper-source.py) | 实际 source/native/manifest 完整匹配 helper 的源码，保留原宿主路径合同 |
| [eigen-producer-design.md](eigen-producer-design.md) | 写作时的设计与候选状态，后来终态由本 README/原始报告说明 |

raw drivers 保留执行时绝对路径，helper 原位为 `/private/tmp/calmetrics-platform-latent-closure/memory.py`，memory helper 从当时 CME `tools` 读取。归档中的 helper 是源码证据，不会自动替换运行路径。迁移宿主需在隔离环境准备匹配路径，或明确修改路径后产生新的 driver SHA 与结果；不能让一个不匹配的现行 wheel 复用旧 manifest。测试文本副本原来依赖同版本相对 test module，也不能直接把改名副本当独立 pytest 执行。

本目录不包含第二份可执行 C++ 数学实现、wheel/native/worker/object/probe 二进制、整个 mirror/build/install、pickle 或缓存。计时与内存 driver/NJIT 仅是受控证据，不得成为生产回退。

## 8. 仍开放的完整链与 Excel 范围

下一阶段来源是已有[compact 数学链合同草案](../2026-10-03-mc-projection-integration/calmetrics-mc-compact-contracts-next.md)及本轮[具体组合与实施设计](compact-chain-implementation-next.md)：显式初态的轴累乘、严格轴累积最大值、多概率共享轴排序，以及由它们组成的 NAV/fan/compact 结果与有界物理闭包。新设计补齐真实概率向量输入、S=min(P,12)样本范围、N0初列插值、ES空尾部及按组合批次预算；这些文档均为设计，**本轮没有实现或验收这些能力**。

旧 seed/RNG 消费顺序、primitive 随机带、covariance 根、factor transform、NAV/peak/drawdown、fan/tail、多概率分位、首故障 prefix、不可用资产选择/回填和平台 HTTP/业务接入均不在本次完整投影范围。已有 complete fixed-tape NJIT profile 与约 49 MB heap 的 compact 测量是另一输出合同，只用于定位下阶段机会，不能与全部 period 输出拼接成 C++ 内存节省或完整链通过证据。

Excel 仍从相同 canonical DAG 生成，本次只改 producer 物理执行，不改公式。273 联合专项中的 37 项验证当前 Excel 原语/图测试，不表示新桌面重算。前轮新增 186/187/共享图的 LibreOffice 实际重算，以及旧 185 算子的历史 Microsoft Excel 快照比较，保留在原归档；新增公式的 Microsoft Excel 重算欠项没有因 Eigen 提速而关闭。
