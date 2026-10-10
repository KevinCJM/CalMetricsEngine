# 私有协方差→Cholesky F 根：2026-10-02 局部采用

本轮减少的是完整方阵存储，未新增数学算法或库。单一独立的 `cholesky(covariance(x))` 显式 F 根复用最终结果缓冲，保留原 Accelerate DSYRK/DPOTRF、canonical 计算顺序和失败重放；成功后原地换位。4096 阶实际冷/暖申请峰均减少 **128 MiB**。这是一项局部采用，**全算子中大型时间与内存领先 NJIT 的目标仍未完成**；未提交或推送。

旧构建 `f90ddcc0b42dddba625903abf18e8860303b8e747548dd14e8b2e2d1ed246a57`，新构建 `5d658a12626a65ea524e82d680c17c04b792bc3e1067f194e18cae2a2704ed8f`。77 项 native 构建输入中仅 graph.cpp 与 matrix_foundation.cpp 改变：[最终身份](final-identity.json)、[输入差异](native-input-changes.json)、[本轮精确增量](implementation.patch)。该增量不同于整个共享脏工作区的 Git 差异。

## 设计与边界

[详细设计](DESIGN.md)。只接纳原执行元数据已经证明的私有、紧邻、单消费者、持久输入 covariance→Cholesky pair，且是唯一 typed float64、明确 F、无消费者/apply/rolling scope/isolate 的矩阵根。finalize 使用正在构造的 metadata，不读取旧指针推断资格。

两条数值路径都暂用局部 C Output 描述，DPOTRF 的 U 三角、输入读取、算式、误差容差和错误优先级不变。两条成功路径共用 32×32 索引块的标准库原地交换，没有数值临时分配。失败时不换位。公开结果仍 F 连续、只读、独立所有权；直接算子 F out 和别名限制不变。

不再分配 D² 根 arena，也不执行最终根复制。原地换位仍写入 `D*(D-1)*8` 字节，计入 `algorithm_copy_bytes`，不能称整链零复制。九组完整请求及重复诊断均核对了该精确增量、direct_output_bytes、arena/root_copy 和库调用数；new/old 全矩阵逐位一致。observer、prebound、继承错误仍恢复原 C arena；Planner 的 prebound 子图预算使用同一判据，面板/库上界不删减。共享协方差、临时源、多根等继续原路径。

## 完整请求时间

实际安装旧/新 wheel，普通 Scheduler，CPU budget=1，结果字段物化和必要 F 转换都在计时内，并保留上一次独立结果。NJIT 按实际 readonly/layout 签名编译、预热后计时，同样返回独立只读目标布局。各组 9 轮，轮换先执行的后端。单位 ms；[全部轮次/身份/审计](timing-combined.json)，不是纯内核微基准。

| 输入与结果 | 旧 C++ | 新 C++ | NJIT |
| --- | ---: | ---: | ---: |
| 33×16，小 F 根 | 0.010125 | 0.009459 | 0.005208 |
| 529×512 C，canonical F 根 | 15.530 | 14.966 | 4.097 |
| 1041×1024 F，canonical F 根 | 98.919 | 95.865 | 20.466 |
| 2065×2048 C，库 F 根 | 162.895 | 166.201 | 165.232 |
| 2065×2048 F，库 F 根 | 208.759 | 201.614 | 149.855 |
| 4113×4096 C，库 F 根 | 966.751 | 844.573 | 1059.897 |
| 4113×4096 F，库 F 根 | 861.317 | 757.505 | 1022.729 |
| 2065×2048 C，C 根控制 | 154.731 | 140.428 | 143.277 |
| 2065×2048 F，放大 1e8、回退，仅诊断 | 846.620 | 831.372 | 145.874 |
| 同一回退组，修复证据落盘后复测 | 893.313 | 950.012 | 152.199 |

2048 C 输入/F 根增加 3.305 ms，不单独否决大额内存收益。4096 两种输入均实测快于旧版和本次 NJIT。C 根控制没有改变数学路径，其时间变化不能归为 F 复用收益。512/1024 canonical、2048 F 库根以及高幅值回退仍有实质 NJIT 差距。

高幅值回退两批计时方向不同，第二批后半轮两边都明显波动；原始 CPU/墙钟样本保留，不选较好一批替代负结果，也不外推稳定提速。第二批新旧差 56.699 ms，不简单套用“几毫秒”豁免。阶段采用依据是独立验证的大额内存节约及正常大矩阵完整请求收益；回退时间稳定性仍开放。

数值方面，8 组正常负载通过原 `rtol=atol=1e-12` 的 NJIT 对照。放大 1e8 组旧 C++ 原本即有 10,042/4,194,304 个元素超过该严格容差，最大绝对差约 1.318e-5；新旧 C++ 仍逐位一致。这组只作性能诊断，`strict_njit_equivalent=false`、`correctness=false`，没有放宽产品容差。[原始严格失败](timing/njit.log)、[诊断结果](timing-scaled-fixed/timing.json)均保留。

## 申请堆与 RSS

两组 F 输入/F 结果，三个后端，各自独立 fresh heap/RSS 子进程，共 12 个；[原始记录](memory/memory.json)。输入生成临时块最大 64 KiB，哈希逐连续块/行读取，避免测量前释放全矩阵临时影响 RSS。setup、cold、释放首结果、warmup、保留前次结果的 warm 申请峰连续追踪，峰值不相加；首次库工作区位于测量期。36 个 heap 阶段无 table overflow 或 untracked free，数据/结果 SHA 与计时已校验输出一致。[核对结果](validation.json)。

| 方阵阶数 | 旧 cold / warm MiB | 新 cold / warm MiB | NJIT cold / warm MiB |
| --- | ---: | ---: | ---: |
| 2048 | 80.024 / 112.025 | 48.024 / 80.025 | 102.532 / 134.532 |
| 4096 | 278.040 / 406.041 | 150.040 / 278.041 | 391.063 / 519.063 |

每组冷暖申请峰分别减少 32 / 128 MiB。采样 RSS 请求增量：2048 为 **117.891→88.156 MiB**，NJIT 152.859；4096 为 **459.797→328.312 MiB**，NJIT 537.297。idle control 均为 0。RSS 每组每后端仅一个 fresh 监控样本，是页级补充，不是完整分配计数。malloc 探针不覆盖所有 Python arena 内部分配、mmap 或整个进程树，不据此声称所有内存状态领先。

## 回归、自审核与文档

- 三库 ON Python：[10564 通过](python-full.log)，专项 [136 通过](focused.log)。
- 原生、ASan/UBSan、三库 OFF 原生：[各 9/9](native.log)、[sanitizer](asan.log)、[OFF](native-off.log)；OFF Python 专项 [118 通过](focused-off.log)。macOS 使用 detect_leaks=0，没有 LeakSanitizer 结论。
- [185 直接算子夹具](direct-fixtures.json)，scalar/auto 各回放一次；[历史 Excel 快照](excel-snapshot/report.json) 469 案例、3056 值通过，未做新的 Microsoft Excel 重算，公式与输入不变。
- ON/OFF wheel 检查通过，[OFF 身份](identity-off.json) 与源码/安装扩展/worker 一致；无依赖或打包规则变化，未重建 sdist。
- 原生测试覆盖 C/F 位模式、canary、源不变、失败未转置及恢复、观察器降级后普通执行、native F prebound 子图的额外 D² 预算和预算不足拒绝。Python 覆盖真实 single/thread/process-inline/process-shared、prepared 快照、故障恢复、布局/面板预算及排除图。继承错误分支复核复用同一降级判据；未新增专门 native F 子图继承错误后恢复用例，不将静态证据写成该专项运行证据。

独立审核未发现生产阻断项；发现的证据脚本问题已修复：原诊断只在整批结束后把 false 写回，现于每组记录构造时就写真实严格状态，批次中断不会遗留该组 true。只改本轮临时 harness 的结果字段，原数值、计时和内存测量不变；保留修改前脚本、原工具及两次诊断结果。修复后复测确认首次组记录即为 false。[审核记录](review.json)。

Ruff 发现所触及故障测试原有五处同一行分支；仅展开格式，[AST 等价检查](test-format-check.json)通过，没有修改测试语义。初次 wheel 检查误传文件而非目录，修正调用后 ON/OFF 通过，错误日志保留。未通过格式调整重跑整套数值，也不需要以重复数值运行证明空白改动。

执行指南及布局设计同步当前准入、预算和搬运；数学算子、AST/DAG、公开结果/IPC、Excel 公式均未变，其他数学契约和参考表无需修改。用户关于微小增减不独立作门禁的要求已在 AGENTS/D8，未重复维护另一套门槛。[复现口径](commands.md)与[文件清单](manifest.json)保留证据；大 NPZ、wheel 和构建目录不复制进仓库。全引擎、其他平台和进程树性能目标保持开放。
