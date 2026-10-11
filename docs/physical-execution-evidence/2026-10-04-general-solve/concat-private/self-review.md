# Private concatenate coefficient — review and current acceptance scope

## 当前6707矩形复制与正式验收

2026-10-05，当前实际源码、正式 wheel、扩展与 worker 身份为 `670720df09e608c3ad832e2370a4eaeb540c5e87121e317d0b976d24989d4d10`。本轮接入三处 C++ 物理复制修改和32项新增回归；已有数学、奇异阈值、错误、AST/DAG、输出布局及所有权不变，没有 Git 操作。

既有输入支持的拼接矩形在 macOS15+ arm64/Eigen 开启、正向列连续且两维至少32时，复用 Eigen 3.4.0 的32×32有界 Map 直接写入原目标。不新增矩阵、堆分配、缓存、线程或输入布局转换；其他路径保留受支持的原生复制。不是要求用户将F输入/结果转换成C。

实际正式包相关 **274项**、原生 **9/9**通过；隔离同源普通、真实ASan/UBSan和全库OFF分别相关 **274项**、原生 **9/9**通过，无failure/error/skip。覆盖偏移/填充F输入、边界、特殊IEEE位、四运输、两区间/两代独立结果及原错误反例。没有将20cf历史12,979全量或528专项当作新版本的全量通过，也没有LeakSanitizer或新的Microsoft Excel重算声明。

正式CPU1六轮交替完整KKT构造/求解，计入只读独立输入准备、普通Scheduler、计算和独立结果物化返回；NJIT已预热，编译排除：

| 8192阶输入布局 | C++秒 | NJIT秒 | C++/NJIT |
| --- | ---: | ---: | ---: |
| C | 4.944917 | 5.444061 | 0.9083 |
| F | 4.971662 | 5.455426 | 0.9113 |

此前同轮旧20cf/新6707/NJIT的F完整链为5.411993/4.793669/5.400359秒，旧/新输出逐位相等；本次正式复测见上表，不用更有利轮次替代。最大NJIT尾差约3.23e-13，原方程与约束残差通过。一列结果物理上同时C/F连续，不能据此声称多列结果布局普遍优势。

新旧完整F申请峰约持平（16字节差只作诊断），分配次数相同。正式fresh C++/NJIT F申请峰为 **1639.639/2147.899MB**，少约 **508.3MB**；各三个fresh进程的RSS增量中位为 **1674.772/2687.156MB**，少约 **1012.4MB**。内存优势主要来自已接入的拼接/私有求解工作区减少，不能归功于本轮无额外分配的复制。heap与RSS分别报告，不相加；Python arena/mmap及0.5ms采样短峰盲区保留。

本轮阶段优化已进入实际路径；**整体目标仍未完成**。完整QP、平台候选评估/随机探索、一般大求解和其他主要模型/模拟链仍需完整测量与优化，不能用该KKT任务代替。下一项优先处理候选×历史的大型中间矩阵及重复求解，避免继续围绕小任务微调。普通C++不触发Excel生成，公式/预算/公开签名不变；用户指南及算子参考无需因此修改。未提交或推送。

[限定作用、自审核与原始测量](certificate-diagnosis/measured-integration/layout-copy/self-review.md)、[正式逐项证据](certificate-diagnosis/measured-integration/layout-copy/acceptance.json)。

## 前阶段20cf实际源码与验收边界

2026-10-05，当前实际源码、正式 wheel、扩展和 worker 为 `20cfaf3ed42593a7596230f80fd8621261042d64620e366b37a251939e82711a`。本轮按预像接入一份 C++ 数值文件和五份测试，保留其他脏改动；没有 Git 操作。成熟 LU/BLAS 的原分解与求解不变，校验证书改为有界列组和依赖安全的批量正误差传播，复用已结束阶段的缓冲。原奇异阈值、错误回放、独立结果及逻辑 AST/DAG/Excel 公式不变。

同源原数值版本完整 Python **12,979 项**、普通原生 **9/9**，真实 ASan/UBSan 相关 **528 项**及原生 **9/9**、全库关闭相关 **528 项**及原生 **9/9**通过。随后仅修正一个测试模块对 Eigen-OFF/Accelerate-ON 的逐位浮点和复制量假设，该模块在普通/SAN/OFF各 **39 项**复核通过；Eigen-OFF相关 **528 项**及原生 **9/9**通过。没有宣称修正测试文字后再次执行12,979全套，也没有全量SAN/OFF或LeakSanitizer验收。实际源码正式包另验相关 **528 项**和原生 **9/9**，含四运输、可变prepared输入、两代只读独立结果及原错误反例。

正式串行完成 **20 项测量任务**：四组完整计时、四组连续申请峰、十二个fresh RSS进程。CPU1、相同数据/算法/输出/两代保留，计入owned readonly准备、普通Scheduler图执行及结果物化返回；NJIT实际签名已预热，编译排除。每组六轮交替，原方程stationarity/约束残差通过。这是KKT构造/求解的一项完整任务，**不能代替完整QP、随机探索或平台请求验收**。

| 系数阶数 | 输入/输出布局 | C++秒 | NJIT秒 | C++/NJIT |
| --- | --- | ---: | ---: | ---: |
| 4112 | C/C | 0.681215 | 0.640713 | 1.0632 |
| 4112 | F/F | 0.786330 | 0.720504 | 1.0914 |
| 8192 | C/C | 4.705447 | 5.265408 | 0.8937 |
| 8192 | F/F | 5.581052 | 5.352096 | 1.0428 |

8192阶C布局约快11%，F仍慢约4%；4112阶本轮慢约6%/9%。此前同20cf源隔离结果的NI比值较好，当前正式复测未稳定重现全部时间优势；原始正负报告均保留，不能改用有利轮次关闭目标。旧7a在同条件隔离对照的8192阶回放约29–30秒，优化的重大旧版收益已进入实际代码；该历史旧版结果不冒充本轮重测。

8192阶完整内存如下（十进制MB，C++ / NJIT）：

| 输入/输出布局 | 申请峰MB | RSS增量中位MB | 绝对RSS峰中位MB |
| --- | ---: | ---: | ---: |
| C/C | 1639.574 / 2147.899 | 1675.756 / 2687.844 | 1821.737 / 2828.272 |
| F/F | 1639.639 / 2147.899 | 1679.409 / 2687.795 | 1824.735 / 2826.895 |

申请峰约少508MB，RSS增量约少1GB，两种口径分别成立，不能相加。heap覆盖指定malloc家族完整fixture/准备/冷暖执行/结果保留；RSS为每后端每布局三个独立进程、0.5ms采样。Python arena/mmap与短峰盲区保留，未把未覆盖部分记零。

源码与正式包收益按当前范围结项；**全引擎目标仍未完成**。F输入的系数再生成/残差、大型一般求解，以及平台大量候选、反复QP求解、其他重要完整链继续开放。平台当前随机探索最多20万候选，前沿网格最多30资产/200目标点；不能用大阶KKT覆盖这些不同负载。普通C++不执行Excel导出，本轮没有改变公式或预算，也未宣称Microsoft Excel重新验收。未提交或推送。

## 前阶段7a源码与验收边界

该阶段实际源码/正式包为7a6f，完整测量P*A−L*U替代厂商LU分解误差假设；仍明确常规FP64 BLAS乘积模型和病态前向误差限制。private矩形同步再生成、原系数恢复、canonical错误回放和工作区预算已贯通。internal prebound排除；公开prepared snapshot/audit走普通executor读取当前内容。接口、原1e-14阈值、错误和结果所有权不变。

同源full12957/native9、真实SAN/OFF相关各506/native各9全部通过；实际根正式重建另验1009/native9，包括四运输、两区间、两代及关闭后的寿命。18个正式串行测量终态0。CPU1完整4112 KKT，C旧/当前/NI2.861/.969/.695秒，F2.646/1.032/.772秒；较旧快61%–66%，仍比NI慢34%–39%。申请峰当前417.8/NI541.3MB；fresh RSS增量中位C436.0/678.8MB、F444.8/678.3MB。分别保留口径及盲区；完整QP/平台请求和整体目标未关闭。

原始报告的隔离模板标志保留，在独立来源更正中说明实际已接入；不改写样本。宽重构及只有20ms组件收益的三角库组合均未采用。详见[当前设计、自审和正式证据](certificate-diagnosis/measured-integration/review.md)。以下历史证据按各自版本保留。

Only graph.cpp differs from validated15ea: concatenate joins the sole-use private coefficient producer set, while arithmetic remains a required guard for mature LU. The numerical solve, canonical error/finite scans, pivot threshold, AST/registry, IPC, Excel recipes and Scheduler budget are unchanged. Exact arena ownership and output/RHS span proofs remain required; public or later coefficient consumers and isolate/deferred/observer routes retain the ordinary execution path. A successful private solve consumes the dead coefficient, never a user input or old independent output. Planner conservatively retains canonical fallback capacity.

Normal related781 and native9 passed when the new suite contained14 cases. A further20-case run includes the six new negative-stride/later-consumer/isolate cases; all passed. Initial14 fixture failures were SHAPE_MISMATCH before native execution because declarations omitted common symbolic shapes; only the fixture declaration was corrected, and initial source/log/XML are retained. The original full run passed12676 with59 explicit Eigen skips because the normal invocation omitted the acceptance switch; its controller subsequently failed the zero-skip assertion. Original XML/log/state are retained. Supplementary exact59 skipped node IDs all passed with the correct Eigen/Accelerate flags;12735 distinct normal tests passed across the two runs. Real SAN/OFF new-path20 and each native9 passed on the same88 numerical/config source hashes; this is new-path SAN/OFF, not full sanitizer coverage.

Actual2064 block KKT copy audit drops algorithm movement from34,097,280 to16,512 bytes and operator Workspace from34,080,768 to0; arena remains68,161,536 bytes. Bitwise canonical results, original equation and input identities passed. This audit is not peak-memory or timing evidence.

All19 formal jobs finished serially after predecessor validation was observed terminal: three complete-task timing runs, twelve new/NJIT heap/RSS measurements and four predecessor memory controls. CPU1, same owned readonly input preparation, block concatenation graph, independent result generation/retention, requested order and warmed signatures. Dense SPD direction block and16 independent active rows mirror the observed KKT construction; stationarity and active-constraint residuals are checked outside timed calls. It is one complete KKT construction/solve task, not a complete active-set QP, platform request or optimizer migration.

4096 C/F complete solves remain2.733/2.754s against NJIT0.743/0.737s; the candidate is roughly3.7x slower. New requested peaks540.179MB versus predecessor675.448MB, a135.268MB reduction; NJIT541.309MB is only1.13MB higher. Fresh RSS deltas542.376/542.720MB versus NJIT680.641/679.789MB are137–138MB lower. Heap and RSS are independent evidence with different coverage; no broad heap advantage over NJIT or combined peak sum is claimed. Positive memory adoption evidence does not close the material solve-time gap. The2048 +5.95ms change against the previous C++ is diagnostic.

Next priority is mature-library support for the existing general/non-DD solve contract. Current DD and identity-pivot certificate rejects the KKT domain; removing its checks would lose the exact singular/error contract. Verify a broader admission/replay proof before changing the library domain, then measure complete construction, solve and output return again. No new financial algorithm/operator, hidden pool, cross-request numerical cache or Git mutation was introduced.

Actual root adoption replaced only graph.cpp and added the20-case suite after source preimage, all88 hashes, Python/test tree and GitHEAD checks. Rebuilt root wheel/worker still identify3d7d; root related787/native9 and all four transport lanes passed with two rows and two independent retained generations. Five serial complete measurements passed: F/F old/new/NJIT2.658/2.714/0.681s; new requested peak540.179MB and RSSdelta542.147MB, versus NJIT541.309/677.544MB. New/old time changed+56ms (~2.1%) in this run; do not claim a stable time gain. The substantial135MB previous requested-heap reduction reproduced, and the material ~4x NJIT solve-time gap remains open. Source/bin identity and independent output contracts are verified; no Git mutation.

Subsequent isolated exact-order screens rejected all variants: column8x4 was5.286s versus2.422s current; padded-F was6.043s versus2.417s; C panels128/256/512 were2.406/2.782/3.581s versus2.403s. These solver-only comparisons exclude full KKT/graph/input preparation and cannot substitute for actual-root acceptance. Column controls99 each and row-boundary67 passed in their recorded scopes; large row checks explicitly cover512 only. No numerical runtime changes, new heap/IPC acceptance or hardware-counter diagnosis claimed. Detailed raw results are archived in ordered-kernel-screen.


### 嵌套拼接物理候选（未接入）

输入支持的borrowed视图保持地址寿命；computed/public roots不作稳定叶子，避免F收尾回到arena后被复用。私有内部concat的唯一消费者仍是树内concat，几何占位只被prepare使用，最终拷贝只读终端叶子。C写入以经过各原节点验证的矩形origin/dim界为限；空维不读数据，C内层连续线用memcpy，F矩形32×32分块，其他signed/zero strides按原坐标读取。无数学变换、额外大矩阵、池、跨请求缓存、公共opcode或Excel公式变化；Planner和执行共享同一deferred资格。

只读输入、两代独立结果、rank1/2/3、不同axis、空维、C/F、逆步长、广播zero stride、公开/共享/计算叶子、isolate及四运输方式由156项回归和同源C5全量/SAN/OFF覆盖。C5完整KKT省135MB申请、比NI RSS少约271MB，耗时比旧减少195ms，但仍明显慢于NI。未分块版本的实质退化原始证据保留，未采用。

全量通过后仍发现raw native错误顺序缺口，故C5未接入。fafa在原deferred concat处检查容量及numeric storage，恢复GRAPH_STORAGE_KIND先于后续DOMAIN_ERROR。新增原生回归核验canonical失败buffer字节；首次错误地要求失败buffer保持sentinel，原引擎会初始化失败槽，测试改为原行为等价，运行时源码未因测试修正而改变。

修正版全部数值/配置源码88项冻结：普通全量12891、原生9/9；真实ASan/UBSan和全库OFF各156新路径/原生9/9通过。接入前核对旧3d全部88源码、HEAD和Python/测试树，只复制2份C++及2份测试，保留其他共享脏改动。该阶段目录正式重建包与修正版源/wheel/native/worker同身份fafa，另验943相关、9原生及四条运输路径；full/SAN/OFF与正式重建相关验证没有混称。

正式26个串行测量全部终态0，C/F各六轮、CPU1、相同只读输入准备和独立保留结果、实际NJIT签名预热、编译排除。C/F当前耗时2.426/2.470秒，对NI0.791/0.729秒；申请峰约405MB，对NI541MB；各后端/布局3个fresh进程，当前RSS中位约407MB，对NI679MB。均核验方程与约束残差，原canonical逐位一致、NI尾差1.45e-13。申请峰及RSS收益分别计算，未相加；新结果没有覆盖旧结果，输入未修改。完整KKT时间仍为NI约3.1–3.4倍，不能以大额内存收益关闭整体目标；也不代表完整QP或平台业务请求迁移。未新增数学算子或修改错误/Excel契约，未提交或推送。

局部LU准入后续原型：改用行/列块的Schur半径，严格核验每步主元间距、原阈值及系数/RHS/回代有限性，不使用DGECON估计充当严格界。普通与同源真实ASan/UBSan各145（73/72准入/回退），小矩阵逐阶段原路径与半径/主元对照无违例；四历史反例全部回退，其中一例原canonical成功，不能误称四例都奇异。三阶大KKT的原生oracle状态与数值通过，4112阶原型组件分解+保护约0.457秒，不包括图/输入/结果返回，不能冒充完整性能。直接DGETRF替代及更宽列块仅为筛选，未凭一次几十ms变化采用。首次0/0残差报告错误已修，仅影响零RHS的测试归一化，原失败保留。该原型形成时实际数值源码88份、wheel和worker为fafa；该原型本身未进行生产准入变更。初始厂商BLAS误差包络、private concat的有界原系数再生成及回放、Planner预算和真实完整链尚待验证；详见同目录certificate-diagnosis/local-prototype。
