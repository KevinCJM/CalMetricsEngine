# 完整数值 line 排序与恒等分支元数据共享

2026-10-03，本机 macOS arm64、AppleClang 17、CPython 3.12；生产构建
`faf875f1f6025f515c07dcc97fc1cc5f580576b09305a4df5a737c438b47b5a2`。
全部执行来自正式 AOT wheel 及其同源 worker；87 项 C++/构建配置身份逐项核对。
这是成功固定带消费者链的阶段验收，完整平台计算、随机模拟及短周期内存目标仍开放。

## 1. 设计、依赖与采用理由

- 沿用 190 个 canonical 算子及原逻辑 DAG。`quantiles_axis` 和私有紧凑消费者调用同一个
  `numeric_line_sort`，复用已固定版本的 Highway 1.3.0 `VQSort`。
  Apple arm64、库开启、自动 ISA、至少 1024 值及 float64 支持才准入。
  整数位检查排除 NaN、任意负零、任意非零次正规数；正零及 Inf 本身不排除。
  其他输入/ISA/平台执行原 `NaNLast` 标准排序。排名、插值和错误契约不变。
- 排序只写每条 line 原有的 N-double 工作区，借用输入不变，没有新的 N 大小堆缓冲、
  库线程池或跨请求数值缓存。`highway_vqsort_calls` 统计实际完成的调用；graph、子图、
  compact 后处理及进程传输均累加。IPC21 双向精确匹配，必须使用同包 worker。
- Planner 对可能排序的图（含嵌套作用域）每活动 worker 加一次 16KiB 保守库栈/TLS估算。
  它与 N-double 工作区分开，不是实际 malloc，也不是所有系统/编译器栈帧的严格上界。
  [库栈静态分析](screening/formal-stack-usage/manifest.json)属于同版本、同 Release 工具链
  的库 TU 证据，不覆盖外部 memcpy、TLV/clock 和所有操作系统调用帧。
- 完整恒等 compact 分支在逐字段核对 Program、全部节点/parents、常量 bits、root 顺序、
  dtype/rank/order、映射和槽位后，共享主图的 `const ExecutionMetadata` 及 proof。
  独立结构校验继续运行；部分图、重映射图和含子作用域图保持独立 finalize。
  请求 Scratch、迭代状态、输入和独立结果均不共享；worker 预算不随 owner 去重而减半。
  没有改变公开图定义、序列化或 Excel 数学展开。

共享的是同一图的不可变结构，排序数据在每次调用中重新检查和计算。以上物理优化不修改
金融含义、数学算法、初始化、停止条件或随机源；按 AGENTS、D1/D2/D5/D6/D8 与 X10验收。

## 2. 完整负载时间

原平台四函数的 AST、显式 C-layout float64/uint8 签名、cache/nogil 和 quantile
`inline=always`保持原样，且关闭新签名编译。单份实际输入，NJIT 使用 writable base，
C++ 使用同 buffer 的 readonly view；没有第二份 64MB tape。
同 CPU=1，六轮交替先跑，保留前后两代独立结果。普通入口含规划/绑定/调度、全部
fan、14 项统计、12 样本和资产贡献的返回；构图、预热、哈希及核验计时外单列。

| 负载 | C++ / NJIT 中位数（秒） | 比值 | 节省时间 | 实际 VQSort 调用 |
|---|---:|---:|---:|---:|
| T200/P10000/F4/A32/C1 | 0.318641 / 0.385143 | 0.8273 | 0.066502秒 | 203 |
| T200/P10000/F4/A32/C8 | 1.280300 / 3.073570 | 0.4166 | 1.793270秒 | 1624 |
| T5/P50000/F32/A32/C32 | 2.156035 / 4.015152 | 0.5370 | 1.859117秒 | 256 |

每个大负载六轮均领先 NJIT；原始样本/配对比值见
[普通入口](completed/complete-ordinary.json)。C8 的 F 独立输出为
**1.286441/3.071675秒**，prepared snapshot为 **1.280830/3.070547秒**，
均保留所有输出及两代所有权，见对应 `completed/complete-*.json`。
tiny为 **0.102917/0.011750毫秒**，多0.091167毫秒仅作诊断，不作为独立门禁。
不把此前独立筛选或不同轮次的旧 C++ 耗时与本次结果当作同场配对因果测量。

## 3. 内存与剩余缺口

六组内存各有 fresh heap/RSS子进程，共12个不同PID；两种指标保留各自范围。
连续 requested-heap epoch含构图/spec/plan/worker/工作区与两代结果，五次完整 warmup
也在同 epoch；共同输入、固定签名编译及 tiny预热在 epoch外且保持驻留。

| 负载 | C++ setup live（B） | C++ cold/warm峰（B） | NJIT cold/warm峰（B） |
|---|---:|---:|---:|
| C1 | 3722197 | 4541568 / 4569476 | 32491360 / 32519400 |
| C8 | 5484199 | 8424237 / 8657737 | 48677992 / 48903400 |
| 短周期 C32 | 11974496 | 16540379 / 16635367 | 8451136 / 8503168 |

C1/C8 热峰显著低于 NJIT；短周期仍多 **8132199 B**，不能关闭完整内存目标。
相同冻结 owner测量相较 compact3，短周期 setup和warm峰各少 **1761387/1761227 B**；
这是已实测的约1.76MB去重收益，没有把第二份 descriptor或整个10MB构图成本都算成消除。
独立 staged/control fresh观测进一步验证 compiler retained从前阶段
[10235386 B原始staged测量](history/owner-staged-short_many-ordinary.json)降到
**8473839 B**，最终control全部setup **11974496 B**。
旧staged及[control原始测量](history/owner-control-short_many-ordinary.json)属于 compact3历史构建
`0b0457dc35130e7ac9089c09a8426ce097757954388b4129b532dbb0bec3d6d4`；
对应终态日志保存在 `history/`，不能当作本页最终构建的执行结果。
staged/control相差30400 B，保留其诊断夹具开销，不能把分阶段观测当精确分配调用栈。
pytest导入的 **3266115 B**仍留在原有 owner口径内，未移出或扣除。

| 负载 | C++ / NJIT RSS执行增量（B） |
|---|---:|
| C1 | 901120 / 32489472 |
| C8 | 3342336 / 65929216 |
| 短周期 C32 | 13139968 / 52854784 |

RSS baseline在 owner构造后，上表不代表完整 owner内存。短周期较低RSS执行增量不能
抵消其较高 heap峰；也不把几页变化当独立否决。malloc观察不覆盖CPython arena子分配、
mmap或所有第三方分配；存在合法的未追踪释放，`complete_allocator_evidence=false`。
0.5ms RSS采样可能遗漏短峰；未宣称全分配审计或通用无泄漏证明。

## 4. 自审核与回归

- 只读审核未发现恒等共享、预算、审计/IPC及所有权的确定缺陷；原静态设计与冻结审核
  保存在 `drivers/`，其中“未执行”及 IPC20等表述是当时状态，以本页当前实现为准。
- 首次原生回归暴露负零/FZ新增异常标志（130对2），结果bits及FPCR/rounding虽相同，
  仍更正准入为排除任意负零。另一失败来自 x=2048/q=3的 block捕获对齐违规；
  改为2048个0.5概率，仅更正测试。失败日志/XML和先期wheel身份保存在 `initial/`。
- 首次专项366通过/16共享内存沙箱限制，允许本地共享内存后382通过；这是先期构建证据。
  最终构建完整Python套件 **11963通过/59明确Eigen opt-in跳过**；按XML精确补跑这59个
  node ID后 **59/59通过**，同源合并覆盖 **12022个不同用例全部通过**。
  原始两份XML/日志均保留，不能写成一次无跳过的全量运行。
- 最终原生、ASan/UBSan、关闭Eigen/Accelerate/Highway各 **9/9通过**。
  新排序原生矩阵在ARM含5000个完整line/ISA/rounding/FPCR比较：bits、规范插值、
  浮点标志/环境、canary、只读源和原路径均覆盖。macOS不支持LeakSanitizer，
  `detect_leaks=0`限制保留。OFF与ASan各自配置在 `completed/`。
- 36个新Python用例覆盖direct C/F/正负stride、NaN/负零/混合零/次正规guard、
  single/thread/inline/shared/Hard Stop、审计和独立结果；完整回归覆盖原有Excel数学/预算。
  仅物理排序与owner变更，不改公式；新增188–190的Microsoft Excel真实重算仍欠缺。

## 5. 复现和身份

`source-sort-final.tar.gz`固定238个源文件，包含87项构建身份源；依赖源码归档按SHA引用，
非文本夹具、实际wheel/二进制和上游完整目录不内嵌。不是独立可安装的发行包。
该源码快照保持历史字节不变；后补的owner历史证据及本页修订由 `manifest.json`单独校验，
不属于其238个源码文件。
实际wheel SHA256为 `16cec14d53943721ca4df230d285df3561ec7c07f602ed87760cd4cf8e588cf8`，
本机位置、extension/worker SHA及配置见 [identity](completed/identity.json)。
`screening/`只保留先期排序筛选及库栈分析，不能替代最终生产结果。

冻结driver使用相邻文件目录解析依赖。复测前将 `drivers/`内文件放到同一工作目录，
选择全新输出路径；当前参考源必须匹配 `production_reference.py` 的固定SHA。
从本机已安装wheel复测普通链：

```sh
env PYTHONPATH=/private/tmp/calmetrics-matmul-deps:/private/tmp/calmetrics-perf-followup-deps \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  CME_MC_EXPECT_BUILD_ID=faf875f1f6025f515c07dcc97fc1cc5f580576b09305a4df5a737c438b47b5a2 \
  CME_MC_SOURCE_ROOT=/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine \
  /Users/chenjunming/Desktop/myenv_312/bin/python3.12 \
  /private/tmp/calmetrics-mc-compact-integration/bench-complete-production.py \
  --install /private/tmp/calmetrics-mc-compact-integration/sort-integration-final/install \
  --identity-manifest /private/tmp/calmetrics-mc-compact-integration/sort-integration-final/identity.json \
  --case all --mode ordinary --rounds 6 --output /private/tmp/new-mc-sort-ordinary.json
```

内存用相同的 `memory-complete-production.py`，每case/backend创建新heap/RSS PID，原有probe
SHA保持不变。快照中的 `run-acceptance.py`记录本轮实际命令，不应重跑到已有输出目录。
用 `python verify_manifest.py`核对本页及原始证据；校验不运行数值、不替代验收。

## 6. 未完成范围与下一步

优先定位剩余约8.47MB compiler owner的真实分配，再决定是否缩减重复NodeInfo/Program
结构和请求descriptor；静态sizeof不等于heap，不先承诺节约量。保留原短周期失败负载。
完整随机流/旧seed、coverage、limits、JSON、首故障prefix、平台真实服务切换及全平台
中大型时间/内存优势均未完成。未提交或推送，整体目标保持开放。
