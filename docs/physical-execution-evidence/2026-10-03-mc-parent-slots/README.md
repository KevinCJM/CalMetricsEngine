# ParentSlots 图存储验收（2026-10-03）

本阶段仅改变图节点父依赖的内部存储。数学算法、canonical 算子目录、逻辑图、错误行为、执行模式、Excel 公式与程序传输版本保持原契约。当前正式构建为 `b8dc2f16aef13017d8652575f282cc6522992fdabb08e051c68634b76b26cb4d`；原始源码与二进制身份见 [identity.json](completed/identity.json)。

## 设计和审核范围

`Node.parents` 仍有 32 个逻辑槽，前 8 槽内联，后 24 槽独占并按需分配。未分配尾部的只读访问返回零；可写访问建立零初始化尾部。拷贝独立、依赖顺序和所有 32 槽的比较保持，包含未使用槽。紧凑路径预算按实际尾部所有权计入 96 B，不凭 arity 假定尾部为空。

普通算子最大 arity 为 8；高捕获作用域保留 32 槽限制，33 槽仍拒绝。Python GraphNode、metadata、pickle 及 worker 值协议不变。公开 C++ 结构的类型、布局及旧 std::array 接口已改变；本仓库没有发布稳定 C++ SDK/ABI，不能据此声称外部 C++ 源接口兼容。详细设计见 [MC 轴基础第 7 节](../../mc-axis-primitives.md#7-parentslots-设计与当前验收边界2026-10-03)。

独立静态审核覆盖父槽存储、编译器、Graph、绑定、资源预算和原生回归，未发现确定执行缺陷。通过的测试是以下实际运行记录，静态审核不替代运行证据。

## 正确性验收

| 检查 | 当前记录 |
| --- | --- |
| 完整 Python 套件 | [12,071 通过，无跳过](completed/full-python.xml) |
| 新增高捕获/跨执行模式专项 | [49 通过](completed/focused-python-corrected.xml) |
| 原生 | [9/9](completed/native.xml) |
| ASan/UBSan | [9/9](completed/asan.xml)，macOS 未启用 LeakSanitizer |
| Eigen/Accelerate/Highway 三库 OFF | [9/9](completed/off.xml) |

完整 Python 运行设置 `CALMETRICS_TEST_EXPECT_EIGEN=1`，原先可跳过的 Eigen 检查真实执行。专项覆盖 2/8/9/10/31/32 依赖、33 槽拒绝、严格/隔离策略、单线程/多线程/inline 进程/共享内存进程、pickle、只读输入和两代独立结果生命周期。原生覆盖 32 槽值、深拷贝/移动、未使用槽相等规则以及实际尾部预算。

初次专项有 25 项测试夹具失败：严格策略本来返回 `statuses=None`，且捕获上限错误文字与测试正则不同。只修正夹具后 49 项通过；[初次 XML](completed/focused-python.xml) 与日志保留，未覆盖历史失败。

## 完整计算链时间

固定已经变换的随机带，计算投影 → NAV → fan → 14 个统计量 → 12 条抽样路径 → 逐资产贡献；CPU 预算 1，同核数、六次交替配对，保留两代独立输出。NJIT 使用冻结的生产函数及其固定签名、cache、nogil 与分位数 inline 语义。普通入口含绑定、校验、规划、调度和紧凑结果出口；图构造另外记录并计入内存测量。

| 负载 T / paths / factors / assets / combinations | C++ 普通入口中位数 ms | NJIT 中位数 ms | 配对比值中位数 |
| --- | ---: | ---: | ---: |
| 200 / 10,000 / 4 / 32 / 1 | 318.967646 | 384.762063 | 0.827153 |
| 200 / 10,000 / 4 / 32 / 8 | 1282.510438 | 3069.9719375 | 0.417398 |
| 5 / 50,000 / 32 / 32 / 32 | 2154.869250 | 4015.3380005 | 0.536656 |

原始逐轮数据见 [普通入口](completed/complete-ordinary.json)。8 组合的 [F 布局普通入口](completed/complete-f-ordinary.json) 为 1280.156354 / 3071.7981875 ms；[prepared 独立快照](completed/complete-prepared-snapshot.json) 为 1286.245604 / 3071.289979 ms。微型负载为 0.1016875 / 0.0116455 ms，差 0.090042 ms，只作诊断。

与前阶段相比，几 ms 的波动不构成本次显著提速证据。当前已有约 58% / 46% 的两组大负载时间收益保持；本次收益主要是实际 MB 级图存储节约。

## 内存和真实剩余差距

每种负载和实现分别在 fresh PID 测量 requested heap 与 RSS。heap 连续 epoch 覆盖构造、pytest/helper 导入、首次执行、五次预热以及两代结果；共同输入和微型 NJIT 编译位于 epoch 外，两侧规则一致。预建图或工作区没有移到 epoch 外。

| 负载 | 前阶段 C++ warm heap B | 当前 C++ warm heap B | 当前 NJIT warm heap B | 当前 C++ / NJIT RSS 执行增量 B |
| --- | ---: | ---: | ---: | ---: |
| 1 组合 | 4,569,476 | 4,429,636 | 32,519,400 | 278,528 / 32,784,384 |
| 8 组合 | 8,657,737 | 7,948,801 | 48,903,400 | 3,653,632 / 66,273,280 |
| 短周期 32 组合 | 16,635,367 | 13,821,031 | 8,503,168 | 9,830,400 / 54,591,488 |

六份原始记录：[1 组合 C++](completed/memory-large_c1-ordinary.json)、[NJIT](completed/memory-large_c1-njit.json)，[8 组合 C++](completed/memory-large_c8-ordinary.json)、[NJIT](completed/memory-large_c8-njit.json)，[短周期 C++](completed/memory-short_many-ordinary.json)、[NJIT](completed/memory-short_many-njit.json)。前阶段原始记录仍在 [排序/元数据归档](../2026-10-03-mc-sort-metadata/README.md)。

短周期请求峰值下降 2,814,336 B，但仍高于 NJIT 5,317,863 B，属于实际未闭环的内存差距。setup retained 从 11,974,496 降到 9,160,160 B；[分阶段构造诊断](completed/owner-staged-short_many.json)的 compiler retained 为 5,659,511 B，前阶段为 8,473,839 B。pytest 导入仍占 3,266,115 B，保留在正式 epoch 中。[原构造器对照](completed/owner-original_control-short_many.json)保留，分阶段诊断不能替代完整请求测量。

[原生拆解诊断前值](diagnostic/retained-before.json) / [后值](diagnostic/retained-after.json)使用同一冻结定义：7,405 节点、1,536 根。编译存活 heap 从 8,473,791 降到 5,659,463 B；累计分配从 93,747,306 降到 88,773,282 B，峰值从 10,182,992 降到 7,008,216 B。四类 owner 中的 Node 数组对应共 2,814,328 B 节约。探针只编译并拆解不可再执行的所有者，未执行数值计算。前值探针未内嵌构建身份，其归因是辅助诊断；前阶段正式 owner 测量仍是验收基线。

heap 探针没有 table overflow，但存在 pre-epoch/nonintercepted frees，未证明覆盖全部分配器；CPython arena、mmap 和非拦截分配器仍有盲区。RSS 为 0.5 ms 采样，idle 增量均为零；其 baseline 在构造后，不证明完整 owner 的 RSS 优势，更不能用 RSS 优势关闭 heap 差距。CPU1 比较没有 worker/PSS/共享页聚合验收。

## 证据复核与复现范围

- [manifest.json](manifest.json)列出所有归档文件的字节数和 SHA256；运行 `python verify_manifest.py`只验证归档完整性，不执行数学验收。
- 源码快照（本机保留的历史归档：`source-parent-slots-final.tar.gz`，未纳入此次提交）及[快照清单](source-parent-slots-final.tar.gz.json)保存 239 个源码、测试、工具和构建输入。它不是可单独构建的完整 sdist；第三方源码、二进制和非文本夹具未全部打包，外部库版本及身份见构建记录。
- `completed/` 保存完整构建、测试、性能、构造诊断、终态状态与入口脚本；`drivers/` 保存实际 driver、冻结 NJIT 提取代码、consumer 和探针来源；`diagnostic/` 保存本次原生拆解的定义和源码。driver 的原始绝对路径是当时运行位置，迁移目录时需要配置同样输入和工具链，不能把历史源码当成生产 fallback。
- [acceptance-status.json](completed/acceptance-status.json)中的 12 项全部 exit 0，三类原生运行另见 [native-regression.json](completed/native-regression.json)。归档 verifier 或 exit 0 不自行扩大数学/性能覆盖范围。

本阶段没有关闭完整随机流、覆盖率、协方差根、因子变换、limits、JSON 封装和按阶段首错前缀的业务请求验收；也没有证明全部平台算法、全部 CPU/布局或 Microsoft Excel 187–190 重算通过。全引擎目标仍在推进，未提交或推送。
