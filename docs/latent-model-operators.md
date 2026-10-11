# 概率模型与情景识别数学基础：设计与验收

2026-09-27。状态：本轮 25 个算子已实现，功能回归通过，性能验收未闭环。目标是补齐平台情景识别使用的通用数学算子，保留已有 160 个 opcode 和未提交的矩阵/执行优化工作；不改平台服务、不提交或推送。

2026-09-29 [全量复验](operator-acceptance.md)：Microsoft Excel 全部 469 组及编辑重算已通过，关闭下文历史锁屏阻碍；性能仍未全面通过。下文 9 月 27 日数字保留为历史证据，内存分阶段叠加口径已由新记录修正，不能直接与新数值比较为优化收益。

## 来源与范围

平台本地基线 `9b99048d188017816358f581a86b27df1c388161`；实际链路为 `historical_regimes/v2_service.py` 的训练样本选择 → 标准化 → 初始化 → GMM/HMM/Markov → 后验 → 标签映射/回填。数值依据 `numba_kernels.py`，SHA256 `6be805b1284912f39c79196489f81e03715c195526ecf4c868579077cbc186e3`。CodeGraph 对该目录的覆盖不足，已回到注册表与调用源码核对，不能将索引命中视为链路验收。

本轮覆盖自研 GMM/HMM/硬分类 Markov 的拟合与推断数学基础，以及特征统计、初始化和后处理原语。阈值/滞回/确认/峰谷模型优先复用已有比较、递推、状态和事件算子；外部优化模型训练器、平台可靠性验证服务、业务训练窗口、时点认证、命名/发布与界面不在本次变更中。完整平台迁移仍须逐链路验收。

## 粒度决策

遵守 AGENTS §15、计算纪律 D1–D8、Excel 纪律 X1–X10。统计、似然、归一化、训练、过滤、平滑、分类、计数和回填各有独立消费者，必须分开。EM 的 E/M 更新共享参数与停止状态，属于耦合迭代内核；其初始化、特征计算和推断可以替换，故全部显式作为输入或独立节点。Markov 训练是 GMM 后验 → argmax → 转移计数/平滑/归一化的组合，不新增业务命名黑盒。

| 新增能力 | 唯一职责与粒度 |
| --- | --- |
| `variance_axis(x,axis,ddof=0)` | 两遍、顺序和的轴方差；primitive，不改变旧 Welford variance |
| `max_axis` / `argmax_axis` | 数值轴最大值 / 首个最大值的 int64 位置；独立 primitive |
| `logsumexp_axis` / `softmax_axis` | 有限 log 值的稳定归约 / 归一化；primitive |
| `gaussian_log_density(x,means,variances)` | 对角协方差高斯 log 密度；primitive，方差下限 1e-12 |
| `gaussian_mixture_em(x,weights,means,variances,max_iterations=100,tolerance=1e-6)` | 从显式参数开始的一次 GMM EM 拟合；coupled kernel |
| `gaussian_hmm_em(x,initial,transition,means,variances,max_iterations=100,tolerance=1e-5)` | 从显式参数开始的一次高斯 HMM Baum–Welch；coupled kernel |
| `mixture_weights/mixture_means/mixture_variances` | 共享 GMM 拟合的独立只读投影 |
| `hmm_initial/hmm_transition/hmm_means/hmm_variances` | 共享 HMM 拟合的独立只读投影 |
| `em_iterations/em_log_likelihood/em_converged` | 迭代次数、最后 E 步似然、收敛状态的独立投影；int64/float64/bool |
| `hmm_filter(emissions,initial,transition,predict_first=0)` | 前向概率递推；首行是否先转移显式声明 |
| `hmm_smooth(emissions,initial,transition)` | 完整序列的前向/后向平滑，明确依赖未来 |
| `transition_counts(codes,states)` | 相邻有效 int64 状态的 K×K 计数，不包含平滑和归一化 |
| `one_hot(codes,states)` | int64 编码展开；-1 为未知、输出全零，不与状态 0 混同 |
| `partition_mean(x,order,partitions)` | 按显式排列等数量分段取列均值；排序与选择独立 |
| `sample_unique_indices(template,count,seed=0)` | 模板第一轴上的局部 Park–Miller 不重复行索引；显式 seed，无全局 RNG |
| `scatter_rows(values,indices,length,fill=NaN)` | 按唯一 int64 行下标回填 rank 1–3 float64 数组；缺失填充值显式 |

## 数学与边界

训练输入 N×D，参数 K×D，权重/初始分布 K，转移 K×K，均为 float64；N、D、K 必须非零。参数、训练/推断数据有限；概率非负且总量正，方差非负，零方差按明确下限处理。结构校验（dtype/rank/轴/几何/索引）独立于上游数值失败，不能因为 isolate 跳过。数组输入保留带符号 strides 和只读所有权；未知长度不虚构为零。

GMM 按来源顺序：log 密度 → log 权重下限 1e-12 → logsumexp → 后验 → 有效数下限 1e-8 → 权重/均值 → 围绕更新均值的方差下限 1e-5。HMM 保留 log 域 alpha/beta、按时间正序的 xi 累加、初始/转移下限 1e-8、上述均值和方差更新；不更换优化器或静默修补数据。停止在完成当前 M 步后检查相邻 E 步似然绝对差 < tolerance。最多 10,000 轮；耗尽返回最后参数及 `converged=false`，不冒充收敛。`em_log_likelihood` 是最后 E 步值，不是更新后参数重新计算的似然。

标准化是已有 mean/broadcast/subtract/sqrt/where/divide 加新增两遍方差的显式图；全局初始方差下限 1e-4、初始均匀权重也在拟合外构造。HMM 的 GMM 预训练（min(iterations,30)）及对角 0.9 的转移初始化属于组合。Markov 的计数加 1e-3 和逐行归一化属于组合。训练后概率最大值、状态确认、顺序映射、未知行回填均不嵌入 EM。

`partition_mean` 要求 order 是 0..N-1 的完整排列，边界 floor(N*k/K)；初始化采用何种排序是显式输入契约。既有稳定 argsort 与平台 Numba 默认 quicksort 在同值时可能排序不同，本轮不能据此宣称所有量化初始化同值样本均已完成历史模型迁移；显式相同 order/初始中心的拟合必须等价。随机初始化保留平台 LCG（48271、2147483647），重复索引线性前探；seed 为精确整数 [-2^53+1,2^53-1]，取数学非负模后加一，局部状态每次重置。

轴统计输入为 float64 rank 1–3；argmax 同值取最先逻辑位置。统计与概率原语拒绝被消费的非有限值；空输出可返回，非空结果上的空归约轴报 INSUFFICIENT_SAMPLE。方差 ddof 为非负整数且小于轴长。one_hot/transition_counts 的 -1 保留未知；未知相邻边不计数、不跨缺口连接。scatter 下标越界/重复拒绝，不能静默覆盖。axis、states、partitions、count、length 在 AST 为字面整数，直接 API 每次校验。

## 协议、复用与执行

追加 opcode 161–185。训练返回内部 nominal 数值载荷，公开图结果通过专用投影：GMM 为 K×(2D+4)，HMM 为 K×(2D+K+4)；存储分别为权重/初始、可选转移、均值、方差、三个诊断。迭代数与收敛位通过 typed 投影恢复 int64/bool。参数投影默认只读借用，编译器保活同一训练节点；图根及 run_snapshot 独立拥有。单次多输出 CSE 共用拟合，prepared 每次重新读取并拟合本次输入，不按地址缓存旧模型。

几何推导、独立结构验证、Planner scratch、类型化根、worker/共享内存沿既有协议扩展。GMM 复用完整后验或符合下文准入的有界重放工作区；HMM 基础路线复用 N×K emission/alpha/beta，固定轮数可用下文 V61 检查点，xi 始终按原顺序累加；不每轮新建矩阵。过滤尽量直接写根，平滑复用根中的前向值及 K 级后向状态。复用已有成熟原生库，不建立新线程池，单次训练服从 Scheduler 的 CPU 预算；独立拟合任务按现有图/区间调度。

2026-10-03的私有HMM转移统计优化只交错独立行，保留每行及每个xi单元的原累加顺序和标量libm。非scalar且N≥128、K≥3使用128行块；形状工作区上界为 `3*N*K+2*K*K+4*K+min(N-1,128)` doubles，否则为原基础量，scalar也按同一形状上界预留。没有修改公开算法、公式或状态协议。[完整有限输入链路验收](physical-execution-evidence/2026-10-03-hmm-xi-integration/README.md)包含新旧模型逐位、停止邻点、四种舍入模式、所有权及内部缓冲哨兵；平台其他分支及整体内存优势仍未闭环。

2026-10-06现行私有 HMM xi 路径复用 Apple arm64 的 Accelerate vForce，在非 scalar、nearest、N≥4096/K≥3且完整数学范围证明通过时批量执行独立时间行 exp/log1p；保留原 s/t 归一化、xi 时间升序、gamma/M/下限和严格停止。大形状的直接原语规划为 `2*N*K+2*K*K+4*K+512+128*K` doubles：在消费后用 emission 存 beta，以同一 density/Eigen 每128行重放本轮 E 步密度。图若可从已计入预算的 arena/当前未来输出免费借第三块矩阵，则保留完整 history，不为此新分配第三块；小尾部始终按原语计划计算。小形状、关闭库、scalar、非nearest或幅值拒绝继续使用同一有序数学计算。Planner另取两块路线与切换点前最后一个三块形状的scratch上包络，保证最大区间估计覆盖较短区间和子作用域，不改变实际原语预留；真实滚动 beta 复制计入 `algorithm_copy_bytes`；借用第三块仍计入图/根驻留预算，算子 scratch 是无 loan 原语上界。无迭代新全矩阵，旧结果独立不被借用。[按实际成本接入与复验](physical-execution-evidence/2026-10-04-hmm-filter-tail/follow-up-v45/README.md)包含正常、库关闭、sanitizer、构建身份及实际完整请求证据；V42 的384小块与三历史说明保留在历史记录。Excel公式、公开数学、AST/DAG和算子目录未变。

2026-10-07，V60在固定轮数（容差0）HMM训练的前向/后向状态扫描复用Accelerate vForce：Apple arm64/macOS15+、非scalar、nearest、N≥4096、8≤K≤128及表达式有界时，将独立状态的指数放入原有xi K×K临时区批量计算，仍按原状态顺序求和并调用scalar log。无新工作区、复制、线程或公开算子；正容差保留原扫描，防止改变严格停止边界；2NK/arena借用、gamma/M/下限及Excel EXP/LN数学不变。该阶段34项回归与完整/SAN/OFF通过，完整固定训练约18%提速，新增路径不减少申请峰值；符合V61准入时再使用检查点和DGEMV。[V60完整研究、失败与验收](physical-execution-evidence/2026-10-07-hmm-state-lanes-v60/README.md)。

基础浮点工作区分别为普通 GMM `NK+3K`、符合下段准入的有界 GMM `K*(128+2D+3)+128D`、小形状 HMM `3NK+2K²+4K`（适用时另加最多128行 normalizer）、大形状 HMM 按上段两块 history 及固定 tile 上界、高斯密度 `K`、平滑 `K²+4K` 个 double；分段初始化/回填另外使用 N 个索引做独立边界/去重校验。Planner 按真实父节点形状计算两类缓冲的最大保留量，不能按“最大公开节点元素数的固定倍数”估算隐藏的 NK，尤其 K>D 时。无输入几何时使用有溢出检查的保守上界；子作用域采用相同规则。轴归约每组只计算一次基址，后续按带符号 stride 扫描；连续回填按行复制，不省略去重或边界检查。

2026-10-07有界GMM已接入当前未提交源码：Apple arm64/macOS15+且Eigen开启、K≥16、8≤D≤32、NK≥16,777,216，并且有界工作区小于NK时，按128行计算并重放本轮旧E步。原Eigen密度可直接写完整K行跨度的组件列，消除密度临时tile到后验tile的搬运；均值/中心化方差按原样本顺序持续归约，整组件原始和在除法/有限检查前发布，保留显式out失败前缀。其余形状/平台保留原路径。Planner取阈值前完整后验与有界工作区的安全包络，不省略较短区间或子作用域的旧容量；无需改变AST/DAG、IPC24、公式或所有权。[设计、资格与完整请求实测](physical-execution-evidence/2026-10-07-gmm-direct-replay-v56/README.md)区分独立拟合、13输出压力链与整体平台目标；不能把局部少192MiB当作每条识别链都节省192MiB。

现行V57物理图还可保留已证明存活的私有训练输入，在两段算术中省去完整中间矩阵；二维私有列重排可直接供独立回填根或既有回填/argmax使用，复用原索引检查、填充值、有限值和同分量优先规则。底层数值/整数索引必须存活，根布局最终选择后再次确认独立写入；未知或不适用形状继续原路径。HMM图不启用这两项新消除，保留已验收的arena/history借用成本，不能把少分配一个私有数组当作整链成本必然更低。GMM训练、推断和Excel数学未改。[V57完整链验收](physical-execution-evidence/2026-10-07-private-mapping-v57/README.md)记录约61MiB实际峰值改善及仍高于NJIT约16MiB的申请峰值。

## Excel 与验证计划

每个新增 opcode 同步 C++ Excel recipe 和符号预算。训练按最大迭代数展开，收敛后参数/诊断冻结，初始参数和每轮计算可追溯；不读取 C++ 参考结果。采样使用整数 MOD 和可见去重步骤，不调用 RAND；模板值不参与抽样。超 200 万全工作簿单元格或无法证明有限上界时先拒绝，不能省略训练降级成冻结模型。

Excel 输入的 float64 大整数种子按二进制公式精确写入。取模前逐段分解，偏置商后用余数修正，防止 Excel/Calc 对接近整数的 INT/MOD 和近相等大数相减产生舍入；正负极限种子都必须精确匹配抽样索引，不能按浮点尾差放行。直接导出时 Python int 仍遵守既有 15 位整数限制，精确的 53 位种子须显式传 float64，不放宽通用 int64 导出契约。

`softmax_axis/logsumexp_axis/hmm_filter/hmm_smooth` 的 Excel 输入绝对值上界为 `max_serialized_number/(8*max(element_count,1))`，不能证明中间减法/加法不溢出时在 READY 前返回 `UNREPRESENTABLE_VALUE`；普通 C++ 路径不受该 Excel 数值域限制。迭代方差和 logsumexp 的中间错误必须显式传播，不能被 MAX 的下限截断或忽略文本掩盖。默认公式字符预算仍为 128 MiB，可能早于 200 万单元格门槛拒绝。

验证包括：冻结的来源算法测试参考（不运行时导入平台）及独立概率和/后验枚举不变量；病态、极端 log 值、单状态/单样本、重复中心、空轴、非有限、非连续、失败隔离、共享训练投影和所有权；单/线程/进程/共享内存完整图；Excel 覆盖、上界与实际占用、参数编辑及真实重算；最终 wheel 身份、完整 Python/原生/ASan/UBSan 回归。

性能先固定小/中/大 GMM/HMM 训练及过滤/平滑、统计和组合负载，同 CPU=1、热态 NJIT、相同输入与保留输出、相同初始参数/轮数/停止状态。普通/prepared 分开，交替配对，bootstrap 95% 上界 <1；malloc 与 RSS 独立进程测量，准备后常驻 scratch 不隐藏，零/页级 RSS 增量记为未证实。原四组、M1 和矩阵基础门禁保留，未通过项不删除或放宽。

## 验收结果

证据保存在[本轮证据目录](latent-model-evidence/2026-09-27/README.md)。环境为 macOS arm64、AppleClang 17、CPython 3.12，原有脏工作树和历史报告均保留。

| 验证 | 本轮结果 |
| --- | --- |
| 新增能力 | 25 个 opcode，canonical 总数 185；AST/DAG/Planner、类型投影、错误隔离、进程及共享内存贯通 |
| 完整 Python 回归 | 6298 项通过；包含 82 项情景识别专项 |
| 原生 / ASan+UBSan | 各 9/9，通过；未声明 LeakSanitizer 或跨平台验收 |
| 文档示例 | 27 个 Python 代码块通过；目录、签名与链接另核对 |
| LibreOffice 25.8.4.2 | 全目录 469 案例、3056 个值，公式身份与 2 项输入编辑验证全部通过 |
| Microsoft Excel | 当前 38 个新增案例工作簿已生成；Mac 锁屏阻止真实重算，保持 NOT_RECALCULATED，不以 LibreOffice 替代该结论 |
| 新增时间 / 堆分配 / 联合门禁 | 144 组分别通过 32 / 45 / 6 组，性能整体未通过 |
| 原四组性能门禁 | 4/4；63 点普通 Scheduler 比值为 1.000，仍处原阈值边界 |
| M1 / 矩阵基础复测 | 联合分别通过 1/7、0/44；矩阵时间通过 8/44，未关闭历史性能缺口 |

144 组由 16 类负载 × 3 组 N/D/K × 直接/普通图/prepared 快照入口组成。负载包括密度、GMM/HMM 训练、过滤/平滑、5 项轴统计、转移计数、one-hot、初始化、回填与显式标准化组合。训练两侧均显式提供相同参数、固定 12 轮、tolerance=0；首次编译不计热态时间。时间要求配对比值单侧 95% 上界 <1；冷/热堆峰值都必须更小；RSS 零增量或噪声不能判为通过。

以直接入口为例，N=252/2048 的 GMM 时间比上界为 0.675/0.567，热态堆峰值比为 0.451/0.494；HMM 对应 0.825/0.843 和 0.681/0.742。中大样本训练有优势，不代表小样本、图入口或 RSS 全部达标。N=16 的直接 GMM 时间比仍为 1.043；标准化普通图三档时间比为 2.468/3.922/3.466，不能移除这类失败负载。

联合通过仅限本次测到的直接密度 N=2048、直接 GMM N=2048、平滑 N=2048 的三个入口，以及直接 softmax N=2048。该列表受 RSS 页粒度影响，不是跨机器或其他形状的性能承诺。原四组、M1、矩阵基础门禁的最终复测记录见同目录 `gate-summary.json`；其负载和阈值未放宽。

自审核修复了三个问题：数值失败不能掩盖独立几何/索引错误；Excel 的近 2^53 种子和迭代中间溢出不能被舍入/MAX 掩盖；Planner 不能用最大公开节点尺寸低估隐藏 NK 缓冲。最后一项先在旧构建复现两项测试失败，再验证新构建的估计与内存拒绝。优化保留数值顺序，移除密度参数副本、复用迭代工作区、按 stride 扫描轴、连续行批量复制；没有新增私有线程池、Python 数值后备或显式 SIMD 宣称。

剩余性能问题集中在：小任务入口/检查开销、标准化等组合多次扫描，以及通用图按最大元素容量保留槽位和独立根物化的内存成本。例如 N=252 GMM 普通图热态堆峰值为 NJIT 的 2.799 倍。后续应继续做真实形状的槽位容量规划、组合物理融合和轻量入口优化；独立重负载才评估线程/进程，不能用进程启动掩盖小任务开销。来源排序在同值上的差异、其他聚类/可靠性链路、平台接入与 Microsoft Excel 验收仍未关闭；不将本轮标为 M5 或整个平台迁移完成。

### 复现

先构建并安装当前 AOT wheel，再运行；本机隔离安装目录通过 PYTHONPATH 选择，不能误用旧安装包。

```bash
python -m pytest -q
ctest --test-dir <native-build> --output-on-failure
ASAN_OPTIONS=detect_leaks=0 ctest --test-dir <asan-build> --output-on-failure
python tools/check_excel_parity.py --engine libreoffice --all-cases --check-edits --output <excel-lo>
python tools/check_excel_parity.py --engine generate --filter latent: --output <excel-ms>
# 在 Microsoft Excel 打开、全量重算并保存后才能执行 read；不能合成缓存。
python tools/check_excel_parity.py --engine read --engine-version '<实际 Excel 版本>' --output <excel-ms>
python tools/check_latent_performance.py --heap-probe <heap_probe.dylib> --output <latent.json>
python tools/check_phase2_performance.py --cpu-budget 4 --repeats 41 --output-dir <phase2>
python tools/check_foundation_performance.py --output <m1.json>
python tools/check_matrix_array_performance.py --heap-probe <heap_probe.dylib> --output <matrix.json>
```

四组基准须串行执行，期间不并行构建、重算工作簿或跑回归。新基准的 exit 0 只表示测量完成，是否通过必须读取逐项 `time_pass/heap_pass/rss_pass/joint_pass`，不能以退出码代替性能验收。


## 2026-10-07：分类决策的私有分块执行

GMM/HMM分类模板仍由原比较、`state_select`和显式标量映射组成。符合执行指南闭包证明的私有决策链可以按4096点复用掩码，省去完整比较掩码和中间状态数组；比较、状态代码和valid校验仍按原节点顺序执行，int64比较不经float64转换。源的真实底层存活和独立结果目标必须得到证明；输入不变、旧结果不被覆盖。中间根、隔离策略及其他不满足条件的图保留原路径。没有新增金融模板或黑盒算子，原Excel函数扩展保持不变。

CPU1、1398104行/8维/K16/4迭代、13输出和两代结果的GMM完整压力链，最终V58D同源全量16226项、相关SAN/OFF各1488项与原生各9/9通过，实际重建另验相关1488及原生9。完整申请峰值834,750,250bytes，比V57减少33,554,488bytes（约32MiB）；同条件暖NJIT851,456,928bytes。这是该GMM几何的证据，不能据此宣称全部潜变量、HMM或平台计算已完成验收。原HMM数值历史缓冲路线保持，HMM整链收益尚需另测。[设计与原始证据](physical-execution-evidence/2026-10-07-state-decision-v58/README.md)。


## 2026-10-07 V61：固定轮数HMM检查点与原生矩阵向量复用

零容差且N≥4096、16≤K≤128、1≤D≤32、NK≥8,388,608的HMM，在图未选择更便宜的免费完整history借用时，可以只保留一块NK：完整密度校验后原地转为beta，保存每128行的alpha检查点，再重放128行alpha与136行density；xi仍按时间升序累加，最后一个消费者结束后才将beta覆盖为gamma。工作区为 `NK+4K²+4K+512+ceil(N/128)*K+265K` doubles。原始log概率下限、gamma、中心化M步和严格停止不变。正容差与K8–15保留原路线；Planner不知道运行时容差，继续保留完整history包络及较短区间上界，不能按紧凑工作区降低未知参数的预算准入。

Apple arm64/macOS15+、Accelerate开启、非scalar、nearest且有界数据准入时，前向行按 `exp(alpha-max)` 与转移缓存做DGEMV，后向行做相应矩阵向量乘法，复用既有vForce。缓存是 `exp(log(max(A,1e-12)))`，不重新归一化；logA须在[log(1e-12),0]、行操作数绝对值不超过1e8，不适用行在写入前回到有序log数学。仅固定轮数使用该浮点重排，既定绝对误差预算0.000049及离散/错误断言不变。scalar/OFF/非nearest可使用相同检查点存储，但不执行新的BLAS路线；没有对应平台性能承诺。

本轮已接入6份C++及24项Python回归，normal全量16284、SAN相关815、原生各9/9通过；OFF原批814通过，唯一plan测试误用参数已修复并单项复验通过，失败记录保留。实际目录重建与93份source/config、wheel/worker身份一致，另验58项及原生9/9。完整CPU1、262144×8/K32/F/两轮/8输出申请峰值84,583,792bytes，较V60少66,533,120bytes（约63.45MiB），暖NJIT为285,269,146bytes。完整训练时间与RSS证据见[V61记录](physical-execution-evidence/2026-10-07-hmm-alpha-checkpoints-v61/README.md)。

未减少真实复制审计：检查点与beta边界搬运仍计入algorithm_copy_bytes；根和两代结果独立。图更便宜的免费history借用继续可用，不按地址缓存模型。直接audit现公开既有accelerate_matmul_calls，与图/worker同计实际DGEMV调用，无新IPC字段。公开逻辑AST/DAG、算子、类型和Excel数学未变；本轮没有真实Microsoft Excel重算声明。重要矩阵、其他平台链、正容差性能和全面复制审计继续开放。

## V91：宽状态尾行融合与训练行视图

私有高斯密度→过滤→最后一行的物理路线扩展到 1≤D≤32、1≤K≤16、N≥3、NK≥131072。密度在原节点用 128 行 tile 完整校验，再按原顺序重放同一 filter 递推；工作区仍为最多 131K doubles。重放使用原完整 N 的库分派条件，保持原特征归约顺序和失败位置。大 HMM 拟合存在免费三历史借用时，新几何须经 arena 安全下界证明保留这项复用，不能用额外每轮重算换取表面 arena 下降。

至少 262144 元素的连续升序训练行，可在受证明的普通单区间图中只读借用完整连续 C/F 输入或本次已完成的独立矩阵根，保留源步长及正确的行起点；公共根、输入和旧结果不成为可写工作区。所有不适用的几何或执行模式保持原路线。公式、EM/停止/诊断、公开结果所有权及 Excel 数学不变；当前 worker 请求30/响应29。源码接入和实际路径证据见 [V91 接入与复验](physical-execution-evidence/2026-10-09-latent-current-v69/future-result-screen-v74/qualified-adoption/row-state-reuse-v78/README.md#gaussian-filter-stream-v91)；V91实际目录四组完整复验已通过，后继Eigen列密度扩展的当前验收见下节。

## V93：现有Eigen列密度覆盖扩展

同一数学密度步骤的F兼容路线从最多8个成分扩到16个；前置条件、原特征顺序、有限检查和显式out失败前缀保持不变。固定128-double栈块不引入数值堆workspace、输入打包或库线程，K17等不适用形状保持原路径。ISA/几何与复制范围由[执行指南](execution-guide.md)维护，EM初始化、下限、停止、诊断、算子粒度和Excel公式不变。

源码已进入当前目录，实际重建身份79b494c5与同源完整17130项资格一致；实际另验相关2628/native9，243专项含于关联集合，不重复计数。SAN/OFF的相关2628/native各9限定已验收同源变体，leak detector关闭。49份冻结完整链原始记录已经复核，GMM/HMM两轮复验与actual证据分开保留。实际重建包另18项完整链记录及独立审核通过：CPU1、相同六输入准备、13输出和两代结果，GMM/HMM分别比预热NJIT快8.11/3.46倍；HMM增量RSS比旧C++增加7.80MiB，照实保留。本轮每组3次计时，其他主要负载及长期稳定验收仍开放。[范围、非零RSS增量与未完成项](physical-execution-evidence/2026-10-09-latent-current-v69/future-result-screen-v74/qualified-adoption/row-state-reuse-v78/README.md#density-eigen-v93)。
