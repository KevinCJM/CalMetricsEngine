# 固定随机带投影子链：2026-10-03 证据归档

本目录保存 `affine_contract`（186）、`expm1`（187）及共享 period/贡献消费闭包的设计、原始结果和复现驱动。当前契约与任务结论见[投影原语说明](../../mc-projection-primitives.md)。归档期间只读取并复制既有证据，没有重新运行编译、数值、测试、时间或内存任务。

**当前 dense 构建为 `63a1dacd485576f5d148be0fb577bf2d277c91a00ab35c970e6bc88d2330a39a`。8 组合的完整投影明显提速；大型单组合仍慢约 13%，完整输出内存没有显著改善。新增公式完成 LibreOffice 重算，Microsoft Excel 仍未重算。不能据此宣称完整 MC 或整体性能目标已经闭环。**

## 1. 版本、原始文件与冻结规则

- [identity-dense.json](identity-dense.json) 是正式 dense 结果的源码、工具链、wheel、已安装扩展和 worker 身份清单。
- [identity-revalidated.json](identity-revalidated.json) 是根代理在归档前重新核对的终态身份；与 dense 清单字节一致。该清单不是归档代理重新编译或执行得到的结果。
- [archive-manifest.json](archive-manifest.json) 列出每份归档文件的原始位置、大小及 SHA-256。JSON 和日志保持原文；其中 `/private/tmp` 路径是当时执行位置，不代表这些安装目录随归档发布。
- 不归档 wheel、原生二进制、Excel 工作簿二进制、15 MB pickle、Numba 缓存或历史生产实现副本。数值运行仍使用唯一当前生产代码；本目录中的 NJIT 只作为受控参考，不能成为生产回退。

正式 dense 计时使用 [bench-dense-measured.py](bench-dense-measured.py)，SHA-256 为 `41284a86ae3b3d5035e4a91e617c5f3d500f9ea7f423380cf72e7f6aad595c38`。所有 dense 计时 JSON 的 `identity.driver_sha256` 对应此文件。

后来修正的 [bench.py](bench.py)（SHA-256 `6d43e5e537e12299dd141f7112202daa2e2d9b876e1cda44fcd754d02d94cb63`）直接检查 NJIT 返回的贡献向量是否跨调用共享内存。原检查将向量拆成新建零维标量，不能证明原贡献数组的独立所有权。修正后只追加了 [large-shared-dense-ownership.json](large-shared-dense-ownership.json)：`rounds=0`、时间中位数为空，证明同形状两代完整结果的值和所有权；**没有重新计时，也没有替换原始计时证据**。

## 2. 数学和测量范围

正式范围是健康、已变换的固定 factor tape，所有提供的资产均选中；保留全部 period 矩阵和每个资产贡献。输入相同、CPU 预算为 1、NJIT 已预热、普通入口或独立 `run_snapshot`，六轮交替先后配对。输入准备、图构建与明确的 NJIT 编译分别记录，不混入热态计算中位数。

[reference.py](reference.py) 不 import 平台。它冻结上游投影数学的初态优先、系数左乘、factor 升序、`expm1`、权重左乘、asset 升序 period 累加、step→path 单状态贡献累加及最后除 path 数；上游只保留符号、日期和源码 SHA 引用。该参考是自包含的数学对照，不是平台实际生产二进制：只读 specialization 和全可用夹具与平台固定签名有所区别。

- `service_chain` 保留平台逐组合重复 affine/expm1 的数值结构，排除业务 Python/HTTP 耗时。
- `shared_chain` 是额外的强 NJIT 对照：同一 tape 的 affine/expm1 只算一次，使用资产长度临时向量，无完整 N×A 的 L/R。不能遗漏这项对照来扩大 C++ 收益。
- 不覆盖随机 primitive tape、旧 seed/RNG 消费顺序、covariance 根、首故障 prefix 诊断、NAV/peak/drawdown、fan/tail、不可用资产选择/回填或平台服务接入。

计时进程在每侧调用期间保留该侧旧结果，并产生独立新结果，同时还保留另一侧的一份结果。因此瞬间可有三份数学结果；这是公平的逐后端两代替换，不是总 RSS 的两份结果实验。内存另用 fresh 子进程测量，不拿计时进程的 RSS 作结论。

## 3. 当前 dense 时间证据

除特别注明外，N=2,000,000、T=200、P=10,000、F=4、A=32，所有数学输出完整返回。

| 文件与模式 | 组合数 / 对照 | C++ 中位数 ms | NJIT 中位数 ms | 解释 |
| --- | --- | ---: | ---: | --- |
| [large-service-dense.json](large-service-dense.json)，普通 C 根 | 8 / 服务型 NJIT | 481.828 | 2073.790 | 约快 4.3 倍 |
| [large-shared-dense.json](large-shared-dense.json)，普通 C 根 | 8 / 共享 NJIT | 481.037 | 658.758 | 时间减少约 27% |
| [large-shared-dense-f.json](large-shared-dense-f.json)，普通 F 根 | 8 / 共享 NJIT | 481.611 | 658.041 | 保留 F 根布局，时间减少约 27% |
| [large-prepared-shared-dense.json](large-prepared-shared-dense.json)，独立 snapshot | 8 / 共享 NJIT | 484.050 | 658.426 | 时间减少约 26% |
| [large-single-shared-dense.json](large-single-shared-dense.json)，普通 C 根 | 1 / 共享 NJIT | 325.049 | 286.753 | **仍慢约 13%，约增加 38 ms；欠项保留** |
| [medium-shared-dense.json](medium-shared-dense.json)，N=200,000、A=16 | 4 / 共享 NJIT | 21.141 | 22.920 | 约 1.8 ms 差异，仅诊断 |
| [small-single-shared-dense.json](small-single-shared-dense.json)，N=T=252、P=A=C=1 | 1 / 共享 NJIT | 0.007229 | 0.004438 | 约增加 2.8 μs，仅诊断 |

时间驱动在计时外核对完整值/schema/status、两代所有权、实际有界 fusion/tile、零 numeric arena、全部根直接写出及前后源码/wheel 身份。原计时版本的 NJIT 贡献向量所有权证据缺口由独立补证关闭，不能把该补证称为新的六轮时间测量。报告的 `audit` 来自首次未计时正确性调用；末次调用重新核对有界路径，但未替换该审计字典。

`source_unchanged` 前后验证覆盖清单中的 CME C++/构建配置和实际扩展；驱动及参考本身只记录 SHA，不能扩大解释成所有运行文件的前后校验。根代理冻结脚本并串行运行；本次归档再按文件 SHA 保存这些版本。

## 4. 历史慢结果，不代表当前 dense

同样的 N=2M、F=4、A=32、8 组合、CPU=1：

| 阶段 | 构建 | 文件 | C++ / NJIT ms |
| --- | --- | --- | ---: |
| 首轮有界闭包 | `2bbb5714…` | [large-service.json](large-service.json)，服务型 NJIT | 936.005 / 2069.886 |
| 首轮有界闭包 | `2bbb5714…` | [large-shared.json](large-shared.json)，共享 NJIT | 935.144 / 651.822，慢约 43% |
| 稀疏扫描合并 | `80c86599…` | [large-shared-scan.json](large-shared-scan.json)，共享 NJIT | 973.776 / 654.485，慢约 49% |
| dense 连续链接终态 | `63a1dacd…` | [large-shared-dense.json](large-shared-dense.json)，共享 NJIT | 481.037 / 658.758 |

历史构建清单为 [identity-accepted.json](identity-accepted.json)、[identity-scan.json](identity-scan.json)；更早的 [identity.json](identity.json) 为 `96432631…`，只作为阶段身份保留。历史失败不能删除，也不能与当前计时混用。历史驱动 SHA 已记录在对应 JSON；没有以当前脚本冒充当时源码。

## 5. 完整输出的内存证据

四种后端各做 heap/RSS 两个 fresh 子进程，共八个 PID，CPU=1，完整输出每代 `128,002,048` 字节，保留两代约 256 MB。单位为十进制 MB。

| 文件 | 热态 tracked heap 峰 MB | RSS 执行增量 MB |
| --- | ---: | ---: |
| [memory-ordinary-dense.json](memory-ordinary-dense.json) | 258.762 | 257.999 |
| [memory-prepared-dense.json](memory-prepared-dense.json) | 258.762 | 256.672 |
| [memory-service-njit-dense.json](memory-service-njit-dense.json) | 256.014 | 256.360 |
| [memory-shared-njit-dense.json](memory-shared-njit-dense.json) | 256.014 | 256.279 |

**当前没有显著内存优势。** 没有减少输出或寿命以改善数字。消除普通展开图本来会产生的两块各 512 MB 的 L/R，不等于比平台 NJIT 节省 1 GB；两个 NJIT 参考本来就不分配这两块。

[memory.py](memory.py) 复用现有 `tools/benchmark_memory.py`。heap 连续 epoch 包含 graph/setup、首个完整结果、五次预热、两次独立保留结果替换与释放，阶段峰不能相加。RSS 的 graph/setup 在 baseline 之前，完整执行与替换在观察区间内；按 0.5 ms 采样，idle 页变化单列、不扣除。共同输入和同签名 NJIT 编译在两种 epoch 之外。

malloc interposition 存在未追踪释放，并可能遗漏 Python arena 子分配、mmap 或其他分配器；RSS 存在页粒度和采样盲区。零 table overflow 不证明全分配审计通过。审计的零 input/root copy、零 numeric arena及直接输出字节只证明对应执行路径，不能代替全请求内存。CPU=1 的内存结果不声称 worker/shared 页聚合已测量。

## 6. 功能回归、历史失败和修正

本轮后续只做测试代码整理：将回调改为具名函数、整理导入顺序，没有改变 C++ 数值实现。根代理生成的 [focused-after-lint.log](focused-after-lint.log) 记录两个专项合计 **230 passed in 1.39s**；[ruff-after-lint.log](ruff-after-lint.log) 记录 **All checks passed**。C++ 构建仍为 `63a1dacd…`。

下表的全量 11,419 项与三组原生 9/9 是本次测试整理前已终态的 dense 基线，**没有在整理后再次全量运行**；不能把 230 项复验或 Ruff 通过写成重新完成全部回归。

| 既有 dense 基线证据 | 结果 |
| --- | --- |
| [full-python-dense.log](full-python-dense.log) | 11,419 项通过 |
| [focused-dense.log](focused-dense.log) | 原语/物理闭包专项 193 项通过 |
| [excel-focused-dense-axes.log](excel-focused-dense-axes.log) | Excel 预算/支持域专项 37 项通过 |
| [native-dense.log](native-dense.log) | 原生 9/9 |
| [asan-dense.log](asan-dense.log)、[asan-build-dense.log](asan-build-dense.log) | ASan/UBSan 9/9 |
| [off-native-dense.log](off-native-dense.log)、[off-build-dense.log](off-build-dense.log) | 成熟库关闭原生 9/9 |
| [direct-coverage-187.json](direct-coverage-187.json) | 当前 187 项直接算子运行覆盖 |
| [direct-replay-187.json](direct-replay-187.json)、[direct-replay-187.log](direct-replay-187.log) | 当前 187 项 scalar/auto 回放通过 |
| [dist-check-dense.log](dist-check-dense.log) | wheel 内容检查通过 |

这些不是所有可编辑 DAG 的穷举，也不代替 Linux/Windows、平台 HTTP 或业务验收。

保留的过程失败日志及终态依据：

- [focused-final.log](focused-final.log)、[focused-scan.log](focused-scan.log)：strict 与 isolate 对非有限输出的既有政策不同，不能直接互当等价参考；修正专项使用 strict canonical 退准入对照，并独立检查 isolate 状态 4 与健康根。修正后见 [focused-scan-policy.log](focused-scan-policy.log)，当前终态见 193 项日志。
- [excel-focused-dense.log](excel-focused-dense.log)：夹具使用不支持的 `sample` 轴，27 通过/10 失败。按现有公开轴改成矩阵 `scenario/factor`、向量 `asset`，没有扩展生产轴合同；当前 37 项通过。
- [native-scan.log](native-scan.log)：两项因沙箱 `shm_open: Operation not permitted` 失败；同阶段正常环境复验 [native-scan-shm.log](native-scan-shm.log) 为 9/9。当前 dense 的三组原生结果另列。

## 7. Excel：三类证据分别解释

1. 原 185 算子的 469 组/3,056 值：[历史快照比较报告](excel-historical-dense/report.json) 为 `PASS_AGAINST_RECALCULATED_SNAPSHOT`，`newly_recalculated=false`。当前公式/输入与历史 Microsoft Excel 16.89.1 全依赖重算快照相同，仅允许 Readme B4/B9 的导出/构建身份变化；不是本轮新的桌面重算。对应 [manifest](excel-historical-dense/manifest.json)、[重试日志](excel-historical-dense-retry.log) 和 [比较驱动](verify-excel-historical-dense.py) 均保留。
2. 新 186/187 及共享图 10 组/48 值：[LibreOffice 报告](excel-projection-lo-dense/report.json) 与 [manifest](excel-projection-lo-dense/manifest.json) 记录 LibreOffice 25.8.4.2 实际重算 PASS；重算后工作簿 SHA 为 `db755b088e9dbc1860feaab671169b4945a69bc4e6218128821826a2d36d3577`。工作簿二进制不随本目录归档；报告明确重算引擎，不标为 Microsoft Excel PASS。
3. 新公式的 Microsoft Excel 桌面重算因 Mac 锁屏尚未完成。保持 **NOT_RECALCULATED**；不能以 C++ 对照缓存、历史快照或 LibreOffice PASS 代替。生成尝试见 [excel-generate.log](excel-generate.log)，桌面尝试脚本为 [recalculate.applescript](recalculate.applescript)。

## 8. 完整 fixed-tape NJIT 数值链剖析

[complete-reference-profile.json](complete-reference-profile.json) 和 [原始日志](complete-reference-profile.log) 是另外一项受控参考剖析：固定已变换 tape → 服务型 projection → NAV/drawdown → fan/14 项数值指标 → 最多 12 条样本路径与资产贡献。最终保留的是 compact 输出，不是全部 period；N=2M、T=200、P=10,000、F=4、A=32、CPU=1，每形状三轮。

| 组合数 | 总链 ms | projection ms | NAV ms | fan ms | compact 输出字节 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 390.098 | 260.978 | 10.863 | 116.763 | 27,704 |
| 8 | 3107.504 | 2091.329 | 84.693 | 930.693 | 221,632 |

各阶段是分别取得的中位数，不要求它们相加恰好等于总链中位数。总链包含逐组合 Python 数值编排；准备、编译、输入 hash 在计时外。该阶段报告只有 NJIT，没有 C++ 对照，**不是 C++ 完整链通过证据**，也不能替换前面的全部 period 输出时间/内存范围。coverage、旧随机流、covariance 根、factor transform、limits、JSON 包装及首故障案例均未覆盖。

[profile-complete-reference.py](profile-complete-reference.py) 是受控测试驱动：只读核对上游源码 `bb530566…` 后冻结函数体；不 import 平台 runtime，不属于引擎生产依赖。[complete_reference.py](complete_reference.py) 是自包含的冻结 NAV/quantile/fan 数学参考。报告记录完整输入 SHA、上游源码 SHA、冻结函数字节 SHA（`45e2ff60a55291697b55796725f79e96fff8ede8a1a3f89e758faa141d307c7f`）及驱动 SHA，另检查两代 compact 结果独立与输入未改。只能在隔离研究环境主动运行该驱动，普通 CME 执行不能读取上游源码。

同一 compact NJIT 数值链另有两个 fresh 子进程的内存测量：[complete-reference-heap.json](complete-reference-heap.json) 的冷态 tracked heap 峰为 **48,678,168 B**、热态峰为 **48,903,416 B**；[complete-reference-rss.json](complete-reference-rss.json) 的 RSS 执行增量为 **65,798,144 B**、idle 为 0。共同输入 **64,003,328 B** 及同签名编译在 epoch 外；替换时保留两代 compact 独立结果，每代数学载荷 221,632 B，含包装的 tracked 存活量约 225 KB。这个输出合同与第 5 节两代完整 period 的约 256 MB 不同，不能混为等价比较或称为 C++ 节省。

[memory-complete-reference.py](memory-complete-reference.py) 继续复用现有内存 helper。heap 的 `untracked_frees=7`、`table_overflows=0`，Python arena/mmap/其他分配器和 RSS 采样盲区均保留，不声称全分配审计通过。两个原始 stdout 为 [heap 日志](complete-reference-heap.log) 和 [RSS 日志](complete-reference-rss.log)；RSS 日志中的 `None` 是打印代码读取了不存在的 `rss_delta_bytes` 字段，JSON 的真实字段 `rss_delta` 正确。已执行脚本和日志保持原文，没有回写成另一版本。

## 9. 驱动、设计与复现条件

| 文件 | 角色 |
| --- | --- |
| [bench-dense-measured.py](bench-dense-measured.py) | 已计时的冻结源码，原 NJIT 贡献向量 noalias 检查缺口如上披露 |
| [bench.py](bench.py) | 修正后的独立所有权补证驱动；尚未用于新的计时证据 |
| [reference.py](reference.py) | 服务型/共享型自包含 NJIT 数学参考 |
| [profile-complete-reference.py](profile-complete-reference.py)、[complete_reference.py](complete_reference.py) | 独立完整 fixed-tape NJIT 阶段剖析；未做 C++ 对照 |
| [memory-complete-reference.py](memory-complete-reference.py) | compact 完整 NJIT 链的 fresh-child heap/RSS；输出范围与全部 period 对照不同 |
| [memory.py](memory.py)、[memory-dense-cases.py](memory-dense-cases.py) | fresh-child heap/RSS 与串行组合驱动 |
| [benchmark-dense-cases.py](benchmark-dense-cases.py) | 当时串行 case 驱动；当时 `bench.py` 内容对应现在的 measured 文件 |
| [build-dense.py](build-dense.py)、[validate-native-dense.py](validate-native-dense.py) | 当时构建安装及原生检查命令；宿主绝对路径保留 |
| [identity-helper-source.py](identity-helper-source.py)、[benchmark-memory-helper-source.py](benchmark-memory-helper-source.py) | 当时复用 helper 的源码快照，供 hash/范围审查；不是新内存框架 |
| [DESIGN.md](DESIGN.md) | 最初测试/测量设计，未运行和待完成措辞是编写时状态 |
| [benchmark-static-review.md](benchmark-static-review.md) | 较早静态审核；fusion/身份核验缺口后来补齐，具体终态见本页 |
| [shared-product-design.md](shared-product-design.md) | 稀疏扫描合并设计与边界审核；未声称当时性能已通过 |
| [vforce-expm1-static-design.md](vforce-expm1-static-design.md) | vvexpm1 未采用的数值/离散边界说明 |
| [eigen-producer-next-design.md](eigen-producer-next-design.md) | 下一 Eigen 独立 asset lanes 候选，只读设计，未实现/未数值验收 |
| [calmetrics-mc-compact-contracts-next.md](calmetrics-mc-compact-contracts-next.md) | 下一完整 compact 链的合同设计；未实现、未验收 |

复现要求：安装对应 AOT 包，独立准备 NumPy/Numba/threadpoolctl，核对 source/wheel manifest，运行时 CPU=1 且没有其他数值任务。以下只展示命令，归档没有执行它：

```sh
python /path/to/evidence/bench.py \
  --install /path/to/installed-aot-package \
  --identity-manifest /path/to/matching-identity.json \
  --cpu 1 --n 2000000 --steps 200 --factors 4 --assets 32 \
  --combinations 8 --reference shared --correctness-only \
  --output /private/tmp/projection-new-ownership.json
```

驱动保留宿主路径：身份 helper 位于 `/private/tmp/calmetrics-platform-latent-closure/memory.py`，内存 helper 从当前 CME `tools` 读取。两个 helper 原文已归档；迁移宿主需在隔离环境准备匹配路径，或明确修改路径并生成新的驱动 SHA/证据。不能拿一个不匹配的现行 wheel 配旧清单。heap probe 二进制需由已有 `tools/heap_probe.cpp` 在目标宿主单独构建；未随本目录保存。

后续计时使用修正后的驱动时必须生成新文件，保留现有 measured 驱动与结果。不得回写旧中位数、改变其源码身份或把独立补证说成重测。Microsoft Excel 解锁后的真实重算也需保留新引擎/版本/构建/工作簿身份，才能关闭该欠项。

## 10. 尚未实现的下一完整链设计

[calmetrics-mc-compact-contracts-next.md](calmetrics-mc-compact-contracts-next.md) 保存下一里程碑的只读设计：显式初态的轴累乘、严格轴累积最大值、多概率共享轴排序，以及由它们组成的 NAV/fan/compact 结果和有界私有物理闭包。文中的新名称和建议 ID 是设计草案，不是当前公开目录。

设计明确保留不同阶段的首故障顺序、独立结果、全部预算、Excel 展开和实际服务接入边界。这三项原语、compact 物理闭包、对应 Excel、旧随机流及真实平台接入均未由本轮证据宣称实现或验收。当前未终态的 Eigen 筛选源码、构建或性能结果没有归档；不能拿候选准备状态替换 dense 基线。
