# 顺序选择、vForce 异常块回放及 Cholesky F 根直写

本阶段完成三项物理优化及一项子图预算修复，采用理由是完整中大型请求的显著旧C++收益。**全算子相对NJIT的时间/内存目标仍开放**；几KB、几ms增加不是独立否决，正确性、所有权和预算仍是硬约束。修改未提交或推送，未改 FundInvestmentResearchPlatform。

## 设计与实现

[设计](DESIGN.md)保留依赖、颗粒度、错误、所有权、预算和Excel纪律。公开数学算子、AST/DAG、Excel公式、结果/worker协议均未新增版本：

- 单个 `median/quantile` 及 masked 对应算子，在至少1024个消费值且无NaN/混合正负零时，使用标准库 `nth_element` 选上排名、必要时 `max_element` 选相邻下排名。原算式和源大小工作区上界不变；其他输入及共享多根排序继续原路径。没有第二套数学实现或额外数值堆缓冲。
- 独立终端 `exp/log` 保留原vForce范围，每个连续消费块最多65536元素，坏块重放原scalar_math，其他块仍用库。直接调用保持全消费值首写前判别；消费者、observer、scope、isolate等不准入路径不改变错误前缀。审计仅计实际库元素/调用；同序不新分配数值临时，跨序沿既有有界tile并计搬运。
- 单个独立、显式F、typed float64 Cholesky根，成功时改清零的物理三角，利用已有逐位镜像生成原逻辑下三角F结果；原算术及验证顺序不变。省去C arena与最终N²转换，仍返回独占结果。auto/direct F out契约不变；covariance临时融合、作用域及其他消费者继续原路径。
- 自审核发现并修复P2：native F子图作为prebound执行会恢复C arena，原Planner却按独立根省略它。Planner/执行共用降级判据，子图显式prebound；精确预算、低预算拒绝、双迭代状态和实际arena有非对角SPD原生回归。observer另验证C描述、F结果和后续普通执行恢复零arena。

相对冻结旧构建只有[七个生产源文件](final-source-changes.json)变化；测试新增177项。自审核与独立审核覆盖三项优化、降级metadata、内部子图C布局及公平性能政策，没有剩余已确认代码问题；不能把审核替代运行或全算子性能验收。

## 最终身份与功能验证

机器为macOS arm64、AppleClang17、CPython3.12；Release优化库ON部署目标macOS15。ASan/UBSan和双库OFF构建另用macOS11目标。实际安装wheel运行，未通过平台源树数值回退。

| 验证 | 结果 | 证据 |
| --- | --- | --- |
| 最终源/wheel/extension/worker | 77源文件、2二进制身份一致 | [最终复核](identity-final-confirmed.json) |
| 完整Python | 10333通过 | [日志](logs/python-final.log) |
| 原生 | 9/9 | [日志](logs/native-final.log) |
| ASan/UBSan | 9/9；未启用LeakSanitizer | [日志](logs/asan.log) |
| 双库OFF原生/Python | 9/9；540通过、164仅库开启路径跳过 | [原生](logs/off-native.log)、[Python](logs/off-python-final.log)、[身份](identity-off.json) |
| 加强非对角预算测试 | ON/ASan/OFF各1/1 | [ON](logs/native-budget-final.log)、[ASan](logs/asan-budget-final.log)、[OFF](logs/off-budget-final.log) |
| 直接算子固定夹具 | 185算子，scalar/auto均对照 | [结果](direct-fixtures-final.json) |
| 历史Microsoft Excel快照复比 | 469案例/3056值；公式/输入不变 | [报告](excel-snapshot/report.json) |

最终构建 `2ebc5d22f827d240a99a0507b8153213b9fd9216290c22e0bec01f96285567a0`，旧构建 `e619f8cd39317fc9d36356050393b715e0932874593f689b89dbc1eabff65cdd`，[旧身份](identity-baseline.json)。`identity.json`及先前full日志属于预算修复前候选，最终结论只引用final命名记录。Excel是对Microsoft Excel16.89.1已有重算快照的重新比较，**本轮没有新的Microsoft Excel重算**。

## 完整请求时间

共49类，9轮均衡轮转，单CPU/同原生库线程预算，NJIT预编译后测量；普通请求含编译已完成后的绑定/规划/Scheduler和独立结果，prepared比较独立快照，上一结果保留至下一结果返回。输入只读且SHA一致，输出类型、布局、形状和所有权对等。跨布局NJIT有vector与bounded-tiled两种实现，比较更快者；全部原始样本、构建审计及结果SHA见[计时](timing/timing.json)、[摘要](summary.json)。小63元素每轮5次，其他每轮1次。

| 普通完整请求 | 旧C++ ms | 新C++ ms | warmed NJIT ms |
| --- | ---: | ---: | ---: |
| 4M随机quantile | 335.653 | 47.657 | 52.521 |
| 4M随机median | 332.063 | 42.404 | 35.445 |
| 4M masked quantile | 214.523 | 61.019 | 42.379 |
| 16M尾部异常exp | 211.194 | 43.035 | 114.325 |
| 4096² F→F尾部异常exp | 551.587 | 42.010 | 112.966 |
| 16M全异常exp | 100.529 | 74.318 | 49.062 |
| 4096² F→C尾部异常log（tiled对照） | 380.844 | 93.385 | 91.867 |
| 4096阶 C输入/F结果Cholesky | 1985.702 | 1819.559 | 409.036 |
| 4096阶 F输入/F结果Cholesky | 2100.672 | 1937.060 | 465.107 |

表中跨布局log差约1.5ms，不单独否决；已排序/逆序/全等quantile相对旧版多约0.48–1.09ms也如实记录。4M median比NJIT多约7ms，不据此单独宣告采用失败，也不称为明显领先。中大型Cholesky仍慢4.2–4.5倍，masked选择和全异常exp/log仍有实质差距，需继续优化。历史严格字段是诊断，不能自动当作当前采用结论。

新旧selection/Cholesky全输出逐位一致；exp/log沿原2e-12/2e-14库容差，IEEE类别、符号、零及特殊输入canonical bits另核验。Cholesky另查完整三角和正对角、81个确定系数重建；不把抽样重建称为完整N³残差。[异常NaN旧新逐位对照](order-exception-new.json)有24案例/120值。

## 内存及RSS修正

11类负载×三个后端，原66个fresh子进程分别测heap/RSS；heap从setup开始连续追踪，以准备后的存活量计算首次cold与保留前次结果的warm申请峰，包含缓存库工作区。不将分配次数减少或更低总RSS当作请求增量证明。探针无table overflow，限制及SHA见[验证](validation.json)；它不覆盖mmap、Python arena内部分配或进程树。

原驱动在F数据生成、masked修改和哈希时释放整块临时数组，会让准备期驻留页掩盖请求RSS。原heap数据仍有效，**原RSS只保留诊断，被修复后33个fresh进程替代**；生成/mask/hash改用最多64KiB临时，五类受影响输入SHA不变且全部复测与冻结计时输入一致。[输入一致](rss-input-repair.json)、[修复后RSS](rss-repaired/memory.json)、原记录 `memory/`（本机保留，未随此次提交附带）、[修复前驱动](benchmark-before-rss-repair.py)。每类每后端只有一次fresh监控样本。

| 请求 | 旧cold/warm申请峰 MiB | 新cold/warm申请峰 MiB | NJIT cold/warm MiB |
| --- | ---: | ---: | ---: |
| 4096 Cholesky，C或F输入 | 256.511 / 384.511 | 128.511 / 256.511 | 256.001 / 384.001 |
| 4M quantile | 32.009 / 32.009 | 32.009 / 32.009 | 68.001 / 68.001 |
| 4M masked quantile | 32.010 / 32.010 | 32.010 / 32.010 | 54.985 / 54.985 |

Cholesky相对旧版cold/warm均少约128MiB；修正ordinary RSS请求增量C/F输入分别少128/128.516MiB，prepared少约127.703MiB。quantile比对应NJIT基线约少36MiB、masked约少23MiB，这些相对NJIT的工作区优势并非本轮相对旧C++的新增节约。

同序exp/log申请峰相对旧版多16B，跨序有界tile多65,552B；与NJIT相比约多8–74KiB。RSS相对NJIT仍有约0.5–0.64MiB正差，单fresh样本/页驻留不能单独否决明显完整请求收益，也不能写为零或宣称所有内存状态领先。所有原始正负数据均保留，预算不因当前采用政策放宽。

## 证据重放及范围

`benchmark.py`、`tools/`、`run_memory.py`、`run_rss_repaired.py`冻结本次测试实现；脚本是开发证据，不进入生产wheel。`summarize.py`交叉检查49计时/66原内存/33修正RSS、输入、构建和overflow，区分严格诊断与采用政策。已安装wheel路径和版本在identity中；旧候选的二进制留在临时目录，未复制进源码，未承诺清理临时目录后仍能重放未提交的历史wheel。完整输出NPZ约384MiB留在临时目录，摘要保存全数组比较及SHA，未塞入仓库。

单独重测计时可运行冻结驱动，指定匹配的旧/新安装目录、全新输出目录及 `CME_OPT_NEW_BUILD`；不得复用已有输出掩盖失败。内存脚本还固定探针和本机安装路径，需要先核对identity。`verify_identity.py`和[文件清单](manifest.json)分别检查源/wheel身份与证据SHA，`verify_manifest.py`核对文件清单。

首轮错误记录保留：六参数`musttail`编译错误已改为同签名模板；新测试的非整数NaN payload oracle在旧wheel也失败，改为原规则的整排名并补24案例逐位对照；首轮双库OFF缺少预期环境导致计数断言失败，正确OFF配置结果单列；新原生fixture初次把静态矩阵声明为time轴，已修正并加强非对角值。没有为通过测试修改数学公式、降低预算或隐去失败。

用户指南及算子参考无需改签名/数学规则；执行指南、布局设计、优化验收和文档导航已同步实际准入/回放/预算。185夹具是功能覆盖，本轮49类并非全部算子的性能验收；Linux/Windows、进程树RSS、一般DAG布局搜索及平台业务承接不在本阶段验收结论内。
