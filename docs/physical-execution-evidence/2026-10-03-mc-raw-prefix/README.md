# 原始固定带共享前缀：正式证据与范围

2026-10-03，本机 macOS 15 arm64、CPython 3.12、AppleClang 17，正式构建
`85d2003fc8c93506d85d561346577e4894097f7aff00cd36d715edd26eff4581`。
完整输入、实际 installed extension/worker 与 wheel 的身份见 [identity](completed/identity.json)。
wheel SHA256 为 `32a2ab82d4dc5a162b0f13e7f0c786a6752c00879812ba1aa7b4731834a803d7`。

本轮证明的是 **CPU1、ordinary、C 输出的成功 readonly raw replay**：
原始带旋转与 normal/Student 变换 → 资产投影 → NAV → fan → 14 项统计 →
最多 12 条样本路径 → 每资产贡献。设计与完整平台请求边界维护在
[MC 轴基础第9节](../../mc-axis-primitives.md#9-原始固定带前缀与完整请求扩展)。
不以这一条成功链代表所有平台算法、随机请求或异常行为迁移完成。

## 改动与数学契约

可编辑 AST/DAG 仍保留 `affine_contract(raw,transpose(root),0)`、缩放、均值左加、
资产 `affine_contract` 和 `expm1`。只有私有依赖闭包才共享每个时间/路径行；
不改写 root×beta，不重排 factor fold，不启用 fast-math，不增加公共算子或私有线程池。
已有 Eigen 独立列运算与成熟排序库继续负责相应底层工作。

令 `B0=min(C,max(1,T/3))`。需要扩到全部组合时，仅在
`(C-B0)*(3P+A) <= TPF` 准入；比较及实际 workspace 都做整数溢出检查。
输入支持既有 C/F、非连续及负 stride，结果仍按显式布局直写独立所有者。
rotation 行、Student 的 TP 缩放缓冲、两种系数 pack、贡献、排序与 child scratch
进入同一预算；实际 raw_rows、raw_scale_elements、raw_pack_bytes 贯通 audit/IPC22。
pack 字节是 algorithm_copy_bytes 子计数，不能重复相加。

复核另外发现 canonical 空数组容量描述会被 size=1 scalar 覆盖 rank。
当前先比较 rank，同 rank 才比较 size；全局容量仍逐节点取最大尺寸。
这修复空 factor 收缩的 `RANK_MISMATCH`，没有扩大广播，也没有把
`sum(empty)` 的既有 `INSUFFICIENT_SAMPLE` 改成零。

## 正确性与回归覆盖

| 范围 | 实际证据 |
| --- | --- |
| 当前全部 Python node IDs | [12,267 个不同用例全部通过、无 skip](completed/regression-coverage-style.json) |
| raw 前缀专项 | [65/65](completed/raw-focus-final.xml) |
| 通用空轴、错误、四通道与所有权专项 | [120/120](completed/empty-style-final.xml) |
| 原生 | [9/9](completed/native.xml) |
| ASan/UBSan | [9/9](completed/asan.xml)，本机未启用 LeakSanitizer |
| Eigen/Accelerate/Highway 全 OFF | [9/9](completed/off.xml) |

Python 通过总数是互不重复的精确 node-ID 集合与当前 collect 的等集验证，
**不是一次单体全绿运行**：未改旧用例 12,082、raw 65、空轴 120。
首次沙箱运行 [11,459 通过 / 743 失败](diagnostic/candidate-wide-final/remaining-python.xml)；
允许共享内存后只重跑这 743 项，为 [631 通过 / 112 失败](diagnostic/candidate-wide-final/shared-rerun.xml)。
剩下 112 项均为新空轴夹具误将 `sum(empty)` 当成零；仅修正夹具，随后全部 120 项通过。
源码、installed 二进制和其他测试未变化。最终空轴夹具另移除一行多余空白，AST相同，并再次120/120；最终源码SHA与collect等集记录对应，格式检查通过。原始错误记录未覆盖或删除。

当前专项覆盖 normal/Student、不同创新/因子尺寸、C/F/strides、多 interval、
prepared 数据和参数变化、公开中间根及 unsupported 成本的 canonical 退回、
selected roots、exact budget 和 single/thread/process inline/shared。
χ/df 异常对照的是 CME 自己的 canonical 数学/错误合同；FIRP 负 χ 的 sqrt NaN
及平台按组合/阶段选首错尚未宣称等价。

每份正式计时报告核对全部 48/384/1536 个 named roots 的 shape、dtype、status 和值；
两种离散概率精确比较，浮点 `rtol=2e-12, atol=2e-13`，参考固定生产签名并禁用新编译。
原输入指纹保持不变。正式 driver 核对旧 compact 出口稳定与原始 typed 根不别名；
公共专项另保存所有原始 named values（包括 0D scalar），复查旧值/只读/寿命，构成组合证据。
NaN 仅核对分类；prepared 多代及释放后的旧 metadata 未全量保存，不称所有位元和
metadata 生命周期逐项全面验收。

## 同核数完整成功链时间

普通入口包括绑定、输入检查、规划、调度、数值执行与结果出口；输入生成、构图与
初始 JIT/warm 在计时外。六轮交替首跑，每次保留前一代完整结果。
C1/C8 为 T200/P10000/F4/A32；短宽 C32 为 T5/P50000/F32/A32。

| 负载 / 分布 | C++ ms | NJIT ms | 减少耗时 |
| --- | ---: | ---: | ---: |
| C1 normal | 349.923 | 406.335 | 13.9% |
| C1 Student | 355.248 | 406.887 | 12.7% |
| C8 normal | 1310.518 | 3088.058 | 57.6% |
| C8 Student | 1325.429 | 3088.493 | 57.1% |
| 短宽 C32 normal | 860.570 | 4091.820 | 79.0% |
| 短宽 C32 Student | 858.044 | 4084.319 | 79.0% |

全部六对及两项 tiny 记录见 [时间终态](completed/wide-timing-status.json)和
[机器可读汇总](acceptance-summary.json)。raw 与生产 NJIT 提取/计时脚本在 `drivers/`，
逐轮数据保留在 `completed/timing-wide-final-*.json`。
tiny T3/P8/F4/A3/C1 约增加 0.089ms，只作诊断，不成为性能门禁。
C1 只有约 13–14% 时间收益，不将它说成数量级提升。

**NJIT 的 replay copy 是本次输入契约的成本。** transform 原地改写 tape，
不可变 replay 每次在计时及 heap epoch 内复制完整 64,000,000B raw。
原平台 RNG 已产生可写独占 tape，不付这项额外复制；本次结果不能直接推广为其
完整请求的时间或内存验收。原服务 owned RNG 对照仍是独立未完成项。

## 完整输入与两代结果的实际内存

每个后端/metric 使用 fresh PID；heap epoch 覆盖完整输入创建、构图/plan、首次执行、
五次完整预热和两代结果替换。微型固定签名预热与大结果交叉验证在 epoch 外。
本表为真实 warm requested-heap peak，单位 B；不相加各阶段 peak，
也不将累计 `allocated_bytes` 当峰值。

| 负载 / 分布 | C++ peak B | NJIT peak B | 减少峰值 |
| --- | ---: | ---: | ---: |
| C1 normal | 68,372,142 | 160,520,504 | 57.4% |
| C1 Student | 100,383,731 | 176,520,504 | 43.1% |
| C8 normal | 71,710,243 | 176,906,472 | 59.5% |
| C8 Student | 103,785,336 | 192,906,472 | 46.2% |
| 短宽 C32 normal | 114,078,041 | 136,527,744 | 16.4% |
| 短宽 C32 Student | 118,351,518 | 138,527,744 | 14.6% |

原 [11 项终态](completed/memory-status.json)加 [C1 Student 两项补测](completed/c1-student-memory-status.json)：
含六对 heap、一对 C8 normal RSS 和一次旧版本 heap，共 15 个 fresh PID metric 记录。
没有 allocator table overflow；未跟踪 free 及 CPython arena、mmap、非拦截分配器盲点保留，
不称完整分配审计闭环。原前阶段 inputs 在 epoch 外，不能直接与本表拼接对比。

RSS 只有 [C8 normal C++](completed/memory-wide-final-ordinary-large_c8-normal.json) /
[NJIT](completed/memory-wide-final-njit-large_c8-normal.json)一对：baseline/peak
129,531,904/206,061,568B 与 133,709,824/327,467,008B；增量 76,529,664/193,757,184B。
baseline 在完整输入及 owner 前；heap/RSS 来自不同 PID，不能相加。
单次采样观测不证明稳定 RSS 优势或全部瞬时峰值，更不代表 worker/PSS 聚合。

## 失败与历史身份

- [旧97原始带 C8 normal 时间](diagnostic/timing-baseline-large_c8-normal.json)
  为 9412.604ms；[同入口旧 heap](diagnostic/memory-baseline-ordinary-large_c8-normal.json)
  为 4,854,327,635B。它说明旧 CME canonical 张量成本，不说明 NJIT 曾物化同等张量。
- [105 中间候选短宽 normal](diagnostic/timing-candidate-short_many-normal.json)
  为 4465.716ms，对 NJIT4091.943ms，Student 同样约慢9%；最终全batch成本条件另行复证。
- [105 原生8/9失败](diagnostic/candidate-audit-final/native.xml)保留空rank问题。
  当前 rank 修复后原生、ASan/OFF分别复证9/9。
- [最终首次专项8失败/55通过](diagnostic/candidate-wide-final/raw-focus.xml)源于正向准入夹具
  P7 不满足已声明成本式；改用合法P17，并保留P7拒绝回归，最后65/65。
  这不是放宽数学、内存上界或取消回退。

历史87份 native build 源码分别保存在 97快照（本机保留的历史归档：`source-baseline-97.tar.gz`，未纳入此次提交）与
105快照（本机保留的历史归档：`source-intermediate-105.tar.gz`，未纳入此次提交）。105仅从当前/旧字节还原三个差异文件后，
逐文件核对原105 manifest SHA；它不是当前85源码，也不声称历史测试体等于当前测试体。
97完整原源码/测试快照仍见[类型共享历史证据](../2026-10-03-mc-node-types/README.md)。

## 完整性、复现与未关闭项

当前源码快照（本机保留的历史归档：`source-raw-prefix-final.tar.gz`，未纳入此次提交）及[清单](source-raw-prefix-final.tar.gz.json)
保存244份源码、测试、工具和构建输入，内含snapshot manifest与build identity；
87份 C++/CMake/包配置和两份 installed 二进制身份核对。
第三方源码归档仅记录版本/哈希，外部平台树、二进制及非文本夹具未打包，
不能称独立完整 sdist。NJIT 五个函数的原AST、decorator、固定dtype/C签名、cache/nogil
及 quantile inline=always 在 `drivers/raw/` 和 `drivers/compact/` 保存；参考不进入生产运行时。

原始 driver 的绝对路径是实际测量位置，迁移复测应恢复对应工具链、依赖、固定模板、
source-root/install/manifest 与 heap dylib；旧/中间 wheel 必须用各自冻结源码身份核验。
普通结果所有权与 Excel 展开仍遵守当前契约，普通 C++ 不运行 Excel。

`manifest.json`记录归档字节与SHA256，运行 `python verify_manifest.py`仅验证归档完整性，
不重算数学或证明全平台目标。源码/worker/wheel身份、测试集合、时间、heap与RSS是不同证据层。

完整 RNG/旧seed、协方差根生成、coverage、FIRP 异常/阶段首错、limits、JSON、
全平台剩余算法与真实服务接入仍未闭环；Excel187–190 Microsoft Excel真实重算和
完整raw组合预算/重算也未完成。继续以秒级时间和大额内存瓶颈为重点。
本轮仅本地修改，未提交或推送；不结项M6/M9或整体目标。

本轮数学、预算、所有权、证据范围及文档一致性核对见[自审核记录](self-review.md)。
