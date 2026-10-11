# Measured concat LU: actual adoption, evidence and remaining performance gap

## 当前实际6707矩形复制采用与正式复验

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

[作用、别名、数学与Excel复核](layout-copy/self-review.md)、[实际终态证据](layout-copy/acceptance.json)。最后静态检查发现新增测试的import分组缺一空行，仅修正排版；Python AST逐项一致、数值源88项和实际包身份未变，Ruff及补丁格式通过。未因该排版改动重复274套件；原执行XML与测试预像/补丁保留，见[排版复核](layout-copy/test-format.json)。

以下各阶段记录适用于当时的版本。

## 前阶段20cf采用与正式复验

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

之后的设计、失败和验证记录按形成阶段保留，7a已是前一实现。

2026-10-04. Actual source and formally rebuilt package are7a6f; predecessor fafa and rejected wide candidate872f remain historical evidence. This review does not authorize Git operations or claim whole-goal completion.

The existing solve_multi operator, AST/DAG outputs, singular threshold and error policy remain authoritative. This is private physical lowering: checked rank2 rectangles regenerate original coefficients from persistent input-backed concat leaves. Exclusivity, one coefficient consumer, independent floating solver root, destination/RHS span checks, unobserved execution and no isolated/deferred faults are required. Public coefficients and solver consumers retain canonical arithmetic. Internal prebound execution is excluded; public prepared snapshot/audit APIs still read current inputs through their ordinary native executor.

The measured initial error uses every entry of P*A-L*U plus an explicit conventional FP64 BLAS product envelope. Local induction proves separated canonical pivots above the unchanged threshold and bounded coefficient, RHS and back-substitution arithmetic before success. INFO alone is insufficient. All uncertainty regenerates original C coefficient bytes and replays the same canonical solver, including B-before-A diagnostics. No failure is relabeled successful. Final original-equation residuals and finite outputs precede publication. This removes reliance on a vendor LU factorization backward-error bound, but is not a universal vendor-independent proof of the BLAS product bound; normwise backward residuals are not universal forward-error bounds for ill-conditioned systems.

All callback/context lifetimes are synchronous native C++; no numerical Python callback, opcode, hidden pool, cross-request factor cache or new result buffer protocol is introduced. C/F/signed/zero-stride leaves use the existing byte-copy lowering. The coefficient arena is destructive only after exclusive ownership proof; its dead factors become final residual panels. Old independent results, input owners and public coefficients are untouched. Query and execution cover explicit panels, pivot/permutation vectors and conservative vendor reserve; no second full coefficient array is allocated. The logical Excel graph/formulas remain unchanged and ordinary execution does not trigger Excel.

Original tiled integration:490 related tests passed. The66-case new suite now includes8 healthy and8 singular C/F tests across single/thread/process-inline/process-shared, two intervals, two generations and survival after Scheduler/bundle close. A fresh fafa baseline verified all8 singular transport cases: local/thread retain ValueError, native processes retain RuntimeError with the native-worker prefix. Initial transport expectations were corrected to this observed contract, not by changing runtime errors. Same-source9 native tests passed; full12957 passed with no failure/error/skip; related506 ASan/UBSan and related506 all-library-OFF plus each9 native passed on identical source; actual-root rebuilt package additionally passed1009 related and9 native. No full sanitizer or LeakSanitizer claim.

Historical isolated complete4112 KKT task, CPU1, identical owned readonly preparation, ordinary graph, independently returned retained results and warmed NJIT, six serial rotating rounds per layout: fafa/tiled/NJIT C2.4474/.8336/.7443s; F2.4958/.9441/.7589s. Original-equation stationarity and constraint residuals passed; floating tails follow the user's policy. This is complete KKT construction/solve, not an entire active-set QP or platform request.

Historical isolated tiled requested heap peaks C417,818,469/F417,851,391 bytes versus identical immediately preceding NI541,308,880 controls. Fresh RSS delta medians C436,928,512/F435,044,352 versus NI678,903,808/679,084,032; absolute RSS peak medians tiled584,450,048/583,827,456 versus NI817,676,288/818,806,784. Same input digests, output shapes, retention and method were checked. Heap covers requested malloc-family live storage, misses Python arena/mappings; RSS is0.5ms sampled whole-process storage after warmed monitor/runtime baseline. These are separate metrics, never added. Tiled increases heap about12.9MB against fafa while saving about123.5MB against NI. It is substantial baseline time progress, but still12–24% slower than NI in those rounds.

Wide reconstruction combines output-column products into512-by-N panels (256 rows below2048). Related506 tests and source/wheel/worker identity passed. Full repeated C fafa/wide/NJIT2.5208/.8568/.6248s; F2.5098/.9765/.7100s. Requested peaks432.564/432.597MB, about14.75MB more than tiled; RSS delta medians454.705/456.737MB. It showed no timing benefit over tiled and was rejected; no numerical production adoption. Preserve raw negative results instead of presenting larger workspaces as free.

Actual adoption replaced only three numerical files and two tests after checking preimages, HEAD and complete source/Python/test trees. Formal wheel/native/worker and all88 current source hashes agree at7a6f. Actual-root ordinary complete4112 KKT runs finished18 serial jobs with six rotating timing rounds per layout and16 fresh-process memory jobs. C fafa/current/NJIT2.860817/.968728/.695154s; F2.645901/1.031971/.771789s. Current is61–66% faster than predecessor but34–39% slower than NJIT. Exact same data, CPU1, outputs, retained generations and owned readonly preparation; compilation excluded. Original stationarity/constraint residuals≤1e-10 passed. This is complete KKT construction/solve, not a full active-set QP or platform request.

Actual requested malloc-family peak C417818477/F417851399 versus NI541308880bytes, saving123490403/123457481bytes. Fresh sampled RSS delta medians C435994624/F444760064 versus NI678789120/678346752bytes; absolute peak medians C582860800/F590512128 versus NI817446912/817315840bytes. These are separate metrics, not a combined peak. Python arenas/mappings and0.5ms short peaks remain coverage limits. The ~13MB increase against the prior fafa heap measurement remains explicit; that prior heap was not freshly remeasured this round.

Raw formal timing reports inherited production_adopted=false from the prototype template. Actual current source, new-install paths and native/worker identities were independently verified; the separate benchmark-provenance-correction.json records the correct provenance while retaining raw timing reports and their script hash. Do not rewrite samples or imply another measurement run.

The exact-helper component probe on the identical4112 coefficients measured factor~.256s, measured reconstruction~.367s and local induction~.171s. A six-round paired single-prefix DGEMM/DTRMM component screen reduced reconstruction only~20ms/6%; it has no full-task evidence and was not adopted. Pure in-place Eigen, wide panels and other negative screens remain bounded historical evidence. Subsequent compact lower-prefix representation is an isolated, unaccepted certificate experiment; actual original factorization/solve must remain unchanged, every represented entry and product-error bound must be measured, and generic panels must remain supported.

Operator signatures/AST/TypedIR/Excel mathematical formulas are unchanged; ordinary C++ calls perform no Excel generation, and no new Microsoft Excel desktop recalculation is claimed. Full ordinary regression includes existing Excel tests. Native/source/worker quality passes do not prove cross-platform or platform business migration. Other important medium/large loads, full QP consumers and overall NJIT time superiority remain open; no Git operations.

## 隔离873ac进展与未完成检查

873ac7063453a6282fa9795f982f580dedd37f6bb711e0c6b79955f79d087f79为隔离AOT源/包/worker身份；实际项目仍7a6f。20组新增夹具覆盖rank64、dense asymmetric、missed sample、row permutation和small scale的C/F输入/输出，修正后在当前正式7a包上20/20通过。候选526相关通过，原阈值反例、错误、四运输、两代独立结果和输入不变均覆盖；完整12977、原生、SAN/OFF、Eigen-OFF+Accelerate-ON尚未执行，不能套用7a的质量结果。

首轮526用例496通过/30失败：12个shm_open被沙盒禁止，8个新夹具预期错误（置换不保证canonical成功、小尺度大解不应套统一绝对1e-11残差），10个算法复制审计旧公式遗漏新增采样/三角项复制。保留原XML/log/测试；修正测试采用原canonical失败/消息对照及原方程componentwise残差，没有改变运行源码或删除用例。复制断言计入实际新增字节。沙盒外同526复跑全部通过；这不是把环境错误标为通过。

4096完整配对C .8340456/.6770549/.6469671秒、F .9298820/.7712318/.6774509秒（实际7a/隔离候选/NI）；输入/算法/CPU1/输出/保留方式相同，准备和普通入口成本在内，编译排除，原方程及约束残差通过。候选较当前快17–19%，仍比NI慢5–14%。两个完整计时报告保留最初执行脚本哈希；后续仅将8192的“必须库调用一次”错误假设改为允许真实回放，并提前保存ready元数据，数学/计时/输入/保留方式不变。原8192失败控制记录保留；不能据其断言失败称数值失败，也不能省略该负载。

大任务计时与三后端24个fresh申请/RSS任务串行继续；来源和live handle只说明归档时状态，不代表终态通过。正式接入仍取决于扩大质量验证、完整峰值与主要负载影响。下一主问题是8192系数的准入回放原因，仍需在计时结束后按具体拒绝位置诊断；不删除保护取得速度。完整平台QP、其他主要负载和整体目标继续开放。

## 终态测量与隔离20cf审核进展

873ac两组8192六轮计时和24个fresh峰值任务已完成，原running快照保留，终态另存。8192候选仍为NJIT的4.41/4.24倍；4112内存相对NI申请峰值约少123MB、RSS增量约少240MB。来源、算法、CPU1、输出/保留方式及计量口径一致。申请和RSS不相加，RSS含采样盲区；8192峰值未验证。873ac未接入，整体目标未完成。

20cf新方案的alias审核：活动error/growth、full/tail持续独占；64N QR面板在重构结束后才改作alpha/beta，两个512²重构缓冲同期改作DGEMM右侧；源callback已完成该阶段读取，原始LU和rhs不被这些写入覆盖。当前列组内主元只读取活动组状态，批内未来主元行即时更新；批外未来组只有下一批才读取，因此可延后计算。乘子、RHS、严格分离、回代/残差和错误恢复均保留。

首次528相关测试521通过/7失败，两项8192回归通过。6项仍预期n=1008时宽界回放，细分使其现在安全调用库；1项prepared可变输入恢复用逐位断言限制了允许的浮点变化。保留原XML/log/测试，路径预期更新为真实成熟库调用；prepared浮点对照采用已有1e-10/4.9e-5，并新增对实际返回结果的原方程残差。旧结果、输入和错误反例仍精确检查，未放宽奇异/异常/离散要求。修正后同528复跑状态须以其终态文件为准；不得把首次失败记为通过。

候选AOT源码、wheel、worker身份20cf一致；相关、完整Python、原生、真实ASan/UBSan及库关闭路径、完整配对耗时和fresh峰值尚须分别验证。批量正界常规FP64证明及逐单元对照未闭环，不借用7a/873ac的测试通过作为20cf证据。更大真实平台QP及其他主要负载仍开放。

### 20cf已完成检查与当前边界

同528相关复跑全部通过，XML逐项无failure/error/skipped；两项8192 C/F实际库调用、原方程/约束、输入不变及两代独立结果通过。原失败521/7记录保留，未改写。AOT源/包/worker身份未因测试修正改变。

用同一完整LU同时执行256宽未批处理递推与批处理递推：4112/8192分别82240/311296个最终error单元，候选没有任何一个低于参考；最大增加约4.29e-14/4.98e-10。两个证书均通过。这是393536个单元的具体回归证据，数学正界证明仍限定常规FP64模型，不能推为任意数据/厂商的普适结论。

当前完整六轮C4096实际7a/20cf/NJIT .833410/.683343/.825035秒，F .919418/.782577/.801039秒。候选较当前快18%/15%，同轮比NI快17%/2%；F差异不足以宣称显著领先，NI绝对值跨轮有波动，旧873ac结果继续保留。8192完整计时和fresh峰值仍按观察快照进行；live handle只证明记录时进程存活，不证明终态。性能阶段结束后才运行已准备的12979全量、原生9、ASan/UBSan528/9、全库OFF528/9及Eigen-OFF528/9流程，未执行的不计通过。实际源码仍7a，其他重要链路和整体目标继续开放。

### 20cf完整测量终态：大额内存收益，时间仍待扩大

四组六轮完整计时和16个8192 fresh峰值任务已全部终态0；样本、实际source/installed native/worker身份、输入摘要、结果/原方程和CPU1路径核对通过。8192系数C实际7a/隔离20cf/热NJIT为28.710523/4.690125/5.318823秒，F为30.468901/5.395030/5.556103秒。候选成熟库路径真实调用一次，无canonical回放；较实际7a快约82–84%。同轮比NI快约12%/3%，F应视为基本持平，不能据此宣称主要负载全面显著领先。

完整申请峰C/F候选1.639574/1.639639GB，NI均2.147899GB，少约508.3/508.3MB。每布局/后端三个fresh进程，RSS增量中位C候选/NI1.676722/2.688303GB、F1.676198/2.687320GB，少约1.0116/1.0111GB；绝对RSS峰中位C1.822327/2.827567GB、F1.823850/2.826011GB。heap含连续准备、图构造/首次/热执行和两代保留；计时含每次owned readonly输入准备、普通入口/图执行与独立返回，编译排除。RSS采样/Python allocator/mmap盲区保留，两口径不相加。原始峰值差及计量边界不被Planner估计替代。

此解为一列，结果物理上同时C/F连续；输入布局的完整链差异不能解释成输出排布优势。范围仍是构造并求解KKT，未覆盖完整QP、平台请求或其他主要链路。正式实际源码仍7a，20cf仍隔离；已启动同源完整12979/原生及SAN/OFF验收，状态观察另存，未结束项目不得计通过。接入需质量、自审核和来源闭环，整体目标继续开放。

同源20cf原生9/9已终态通过，逐testcase核对无失败/错误/skip。完整Python及SAN/OFF的当前进度见质量观察快照，未据运行中状态计通过。实际源码尚未采用，整体目标仍开放。

### 20cf全量回归与阶段定位

同源20cf完整Python12979和原生9/9已终态0，XML逐项无failure/error/skipped。ASan/UBSan实际扩展已构建，528专项仍在执行；全库OFF及Eigen-OFF尚未完成，不据此宣称全部质量验收通过。当前实际源仍7a，20cf尚未采用。

仅增加时钟与复制计数的隔离f548诊断包，8192系数C/F各执行两次同一普通完整KKT任务，原方程/约束、实际成熟库调用一次及无回放均通过。C分解2.09–2.19秒、证书2.02–2.09秒；F分解2.35–2.37秒、证书2.13–2.14秒，F原方程残差阶段0.34–0.37秒。源callback累计重新生成约1.074GB，C约0.060秒，F约0.373–0.375秒，小矩形不足0.001秒。分解与证书是秒级主瓶颈；不能把全部差距归因布局，也不优先追逐小矩形的毫秒收益。

此阶段与SAN质量并行，仅用于定位，不能替代串行公平性能验收。初次诊断在已完成一次有效C计算后误访问MaterializedResult.audit而失败；原驱动与stderr保留，修正为真实raw.audit后另目录完整重跑。数值源码、输入、计时边界和算法均未因驱动修正改变。下一候选优先评估减少证书重构成本，数学/奇异阈值和独立结果要求保持。其他重要任务及整体目标继续开放。

### 20cf内存安全终态及未采用实验

真实同源ASan/UBSan扩展528项及原生9/9均终态0，XML无failure/error/skipped；不支持LeakSanitizer，未宣称泄漏检查通过。完整12979、普通原生9和SAN结果保持各自范围，不替代未执行检查。

库关闭构建在pip metadata阶段失败：临时依赖目录中的scikit_build_core仅剩namespace/缓存，build_wheel不存在，尚未进入C++编译。原终态/驱动/日志保留。按原目录metadata版本在任务独立build-deps恢复scikit-build-core1.0.3、pybind11 3.1.0、pathspec1.1.1、packaging26.3；未修改全局环境、C++或测试。单独续跑OFF及Eigen-OFF；快照不代表终态，也不重复已通过质量项。

秒级阶段实验暂未产生适合采用的新方案。QR8将8192重构约1.09秒变为1.68秒，因一个额外面板未满足表示准入而进入完整重构；QR16约1.02秒，收益不足以据此推进。重构块256/32约1.49秒、256/16约1.31秒、128/16约2.03秒；128/32在严格主元分离处拒绝，原始负证据保留。上述仅改变校验证书的构造，没有改变原方程LU或移除保护。

分解块128/256/1024约4.00/2.82/2.37秒，当前512约2.29秒；直接成熟DGETRF约2.14秒。只有约0.15秒局部差异，且厂商额外工作区和完整任务未测量，不据此替换原资源可控路径。实验均与SAN质量并行，用于定位，不能当作公平验收；没有接入任何实验源码。当前实际仍7a，20cf待库关闭路径闭环后接入。一般大求解、完整QP和其他主要负载继续开放。

### 20cf关闭配置及测试预期修正闭环

全库关闭528相关及9原生终态通过。Eigen-OFF/Accelerate-ON首轮527通过/1失败，原XML、日志、控制器和原测试保留。诊断实际调用两次Accelerate LU，输出最大绝对尾差4.77e-18/7.81e-18，原方程残差1.95e-14/1.47e-14，输入逐位未改、独立只读结果正常。失败来自测试误将Eigen关闭等同于所有库关闭，以及将该库浮点结果强制逐位比较。

仅修正test_solve_pair_reuse：明确实际库调用的路径和两根复制量4*A+2*(B1+B2)；库路径输出按浮点容差并检查实际返回值的原方程残差。未调用该库时仍逐位比较；状态、异常、输入、负零和旧结果仍精确检查。运行数值源88项与原候选20cf完全一致，未改奇异阈值或保护。

修正后Eigen-OFF相关528、原生9终态0；被改模块在普通、真实ASan/UBSan、全库关闭各39通过。此前完整12979、普通原生9、SAN528/9、OFF528/9证据保持其原测试快照；没有声称测试文字修改后重跑12979全套。以上闭合候选采用的适用质量条件；正式实际源码/包与完整任务复验仍待执行。整体目标未完成。
