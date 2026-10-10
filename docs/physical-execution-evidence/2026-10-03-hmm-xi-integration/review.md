# HMM xi 分块优化静态自审核

审核日期：2026-10-03。仅静态阅读；本审核未运行数值计算、测试、编译或性能任务，也未修改项目源码。

## 结论

未发现本次 C++ 差异中可确认的正确性、越界、结果所有权或可移植性问题。该结论限定于 `accumulate_hmm_transitions`、HMM `latent_scratch` 和 E-step 调用替换，不代表全项目回归或平台所有情景模型已经验收。

正式时间驱动 `pair.py` 与内存驱动 `memory.py` 的十三输出和结果保留范围一致、清楚。独立运行 `bench.py::main` 存在一处非阻断的测量说明偏差：它会持续保留最初的 `result`、`expected` 和 `p`，后续重复调用可同时保留初始、上一轮和本轮结果。该入口未被用作本轮正式时间证据；正式 `pair.py` 已在开始测量前删除 `expected` 和 `arrays`，`memory.py` 不保留额外数组。不能把独立 `main` 的数据与正式旧/新结果替换生命周期混为一谈。

## 基线与定位

- 先使用 CodeGraph 定位 `cpp/operators/latent.cpp` 的 helper、scratch 和 HMM E-step，再核对文件 diff、Planner、算子执行入口与驱动。
- 当前 `latent.cpp` SHA256：`6d73858b0d6331e3b1683d7c55786ec26ff197037c3b4cb1d13eaa26583783c1`。
- `/private/tmp/calmetrics-latent-xi-screen/baseline-latent.cpp` SHA256：`91fb3cb6cb00e720f5a099517e5be8b32b98d2344928b990505e5569161e11cc`。
- 当前构建清单：`identity.json`，build ID `2abcc292fd55b1d41af8aad343e083b848a6e7ee182208ce953e7046cf55389f`；其中 `latent.cpp` 哈希与本次读取一致。
- 已阅读 CME AGENTS、计算设计纪律和相关 Excel 纪律；本差异没有改变算子数学契约或 Excel 公式。

## 数学、累加顺序与边界

原实现先按时间 `i` 处理单行归一化，再按 `s → t` 累加 xi。新实现对最多 128 个独立时间行交错执行同一 `std::log1p(std::exp(...))` 链：每个时间行的归一化仍按原 `s → t` 顺序；每个 `xi[s,t]` 仍按递增时间顺序累加。表达式的加法树、`v > z` 条件、负无穷初始化、floor、前后向算法、M-step 和停止条件没有变化。只改变独立行/独立单元格之间的执行交错。

helper 执行期间 `alpha`、`beta`、`em`、`log_a` 只读；xi 在 helper 完成后才被下游消费。不存在本轮中间状态被回调读取的新增路径。静态结论不额外承诺任意浮点异常 trap 或 libm 全局错误状态的事件顺序；项目既有数据错误校验和抛错位置未改。

边界核对：`n=1` 不进入转换循环，xi 保持零；`n=127`、`k<3` 或显式 scalar 使用 Block=1；`n=128` 有 127 个转换并预留 127 doubles；`n=129` 有 128 个转换；后续尾块 `count<=128`。空训练输入在索引前失败。shape/scratch 乘法的既有溢出检查限制了 helper 内 `k*k`、`(i+1)*k` 和时间递增的可达范围。

## 工作区、错误与所有权

HMM 基础工作区为 `3*N*K + 2*K*K + 4*K` doubles。新增归一化工作区仅在 `N>=128 && K>=3` 预留 `min(N-1,128)` doubles，最多 1024 字节；该微小增加不能单独成为性能否决项。

指针顺序为 `em → alpha → beta → log_a → xi → temp → effective → initial → logs → normalizers`，新增区域从 `logs+K` 开始，无覆盖既有区域。Block=1 复用至少 K 个元素的 `temp`，只使用其中一个。`ops::execute` 在取得这些指针前依据 prepared scratch 完成 `work.ensure`；fit 内没有新增 resize 或动态分配。

已知输入形状的 prepare 和 Planner 都通过 `latent_scratch` 计算预算，公开绑定执行路径传入输入几何。显式 scalar 在大形状下也保守预留新增区域，这与 audit 预算一致。没有在当前可达绑定路径发现 Planner 少报本次新增 scratch。无输入几何的通用 Planner fallback 未作为本轮执行证据，不能据此宣称所有非绑定内部调用都经过验证。

原有输入有限性、方差、概率、形状校验及模型复制顺序不变；失败输出行为没有新增处理。结果依旧位于独立结果缓冲区，helper 的临时区不逃逸。

## ISA 与可移植性

显式 `Isa::scalar` 强制 Block=1；其余 ISA 在满足形状门槛时使用 Block=128。新 helper 使用 C++17 标准库和既有 scalar libm，没有新增 CPU 指令、平台头文件、线程、BLAS 或平台库依赖。它是交错独立计算链的调度优化，不应表述为新增 SIMD 算法。既有禁止 fast-math/关闭 FMA contraction 的构建约束未被本差异修改。

## 驱动的数学链与十三输出

两侧均保留以下独立输出：状态、score、confidence、概率矩阵、初始中心、component order、实际迭代次数、预训练次数、完整标准化矩阵、权重/初始分布、拟合均值、拟合方差、HMM 转移矩阵（GMM 为同形约定的空矩阵）。不丢弃额外审计输出来减轻某一侧负担。

输入切分和图配置包括：有限输入、固定 realtime 训练/OOS 切分、LCG 随机初始化；训练期 standardize fit 与全量 apply；GMM 或 GMM 预训练 `min(iterations,30)` 后 HMM；训练过滤后的 OOS 连续过滤；组件排序、原 posterior winner 的逆映射、confidence 与概率回填。原 posterior tie 先确定 winner，再通过 one-hot 重排，避免组件排序改变 tie 的状态选择。有效性掩码独立于概率回填；训练状态为 -1，而不是把零概率行当作状态 0。

完整 N 长度的 positions 与 raw/valid 对齐，DAG 内切片得到 train/classify/index identity。所有图输入只传实际使用的 key；interval 为 `[0,N)`。数学参考来自 CME 的自包含 `tests.latent_reference`，不依赖 FIRP 生产导入。

这是一条源代码对应的数学链，而不是 FIRP HTTP 服务或业务资格验收。仍未覆盖动态 finite-row 过滤、available/knowledge-time 切分、长排序的 quantile 初始化 ties、retrospective、Markov，以及状态数量大于 component 数量的分支。

## 正式时间和内存范围

`pair.py` 使用旧、新、NJIT 三个驻留进程；每次仅给一个进程执行指令，按轮次轮换测量顺序，核数均为 1。JIT、初始全量正确性对照与十三输出快照在计时前完成。首次 parity 引用被删除，测量只保留一个上一轮结果并生成独立新结果。旧/新十三输出逐位对照覆盖 shape、dtype 和所有数组字节。

`memory.py` 复用已有 `benchmark_memory.measure_heap/measure_rss`。每个 metric 使用新的串行子进程；双方都在 epoch 前预热相同 dtype/layout/readonly 签名的 NJIT。公共输入构造不计入双方 epoch。heap setup phase 包含图/调用构造、首次执行、helper 预热和保留旧/新结果替换；execution phase 排除事先构造的图/owner，两个 scope 不可将峰值相加。RSS 子进程不继承 heap 子进程的 malloc 热缓存，包含首次全量执行，报告未扣除的 idle control。

capture 只保留 shape/dtype/字节数/迭代次数/审计标量，不保留数组；首轮诊断元数据开销会计入测量。heap interposition 对 Python arena 子分配、mmap 和绕过被拦截 malloc API 的分配有盲区；RSS 含导入运行时/JIT 基线并按 0.5ms 采样，可能漏过瞬时峰值。CPU=1 路径没有 worker 子树，不宣称已测多进程共享页或树级内存。

内存驱动默认 identity 路径属于旧 solve-pair 版本；测本次新版本必须显式传本目录 `identity.json`。不传时，驱动会报告 `build_id_verified=false`，不会把旧 build ID 冒充当前源码身份。正式验收须检查该字段和实际 native binary SHA。

## 已读取的执行证据与未验收范围

本次只读取已有记录，未重跑。`focused-unrestricted.log` 记录 390 passed（包含新增 50 项及相关模型测试）。新增用例静态覆盖块边界、C/F/负 stride/padded 输入、预算与 canary、scalar/auto、停止阈值邻点、四种浮点舍入模式、极端有限输入、失败代码/输出/工作区恢复和结果独立生命周期。

`hmm-large-ordinary-fixed/timing.json` 记录 N=20000、D=4、K=12、max_iter=200、实际 HMM 93 轮、GMM 预训练 30 轮，六轮中位数旧版 6749.667ms、新版 3771.521ms、NJIT 7335.106ms；新版/旧版为 0.5588，新版/NJIT 为 0.5142，旧/新十三输出逐位一致。它证明该普通入口代表负载显著提速，不能外推为所有形状、所有算法全面领先。

完整回归、native/ASan/UBSan、成熟库关闭构建、其他平台、prepared 性能、正式增量 heap/RSS 对照和更广平台分支，不在本静态审核已读取证据范围内；这些事项须由主任务各自的执行记录闭环。现有 Excel 数学契约未变，但本静态审核没有重做 Excel 实机复算。

驱动 SHA256：bench `2486a460cf9382eae55d156fcb066173810bafb40055172e9f2897c9fb32f780`；memory `5d39e66df717413189969f477c8e646e31027c032f60f087dbb966e09c76c984`；pair `ab618b5ad91c6166d080a6a985cbeab94a83a55a95e8f01ffaa2d8fc47a19d6f`。
