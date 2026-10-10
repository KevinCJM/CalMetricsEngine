# MatMul 宽列块：最终源码与正式工具审核

2026-09-30。只读核对最终源码、测试、实际对象、JSON 和已生成日志；没有启动构建、数值、计时或内存任务，也未编辑仓库。本报告对应当前square构建 **93dcfd0800e6670e40af4b80a439f131f9b96043c25d213a2310a51f6c611fbf**；16d2/d2e版报告及完整证据分别保留于 `candidate-16d2/`、`candidate-d2e/`。正式114项时间与51项/306子进程内存测量均已完成；测量完成不等于全部性能或完整内存采用条件通过。

## 静态结论与被否决候选

最终源码、测试及正式计时/堆工具没有发现需要阻断的正确性或公平性缺陷。非重复行C宽块现只准入 **M=K=N**、严格C连续右矩阵 `(N,1)`、且 **N为8的倍数**；矩形、padding或不足完整8列块的C矩阵回到既有SIMD。重复行单独依据 A 行 stride 为零，继续支持原 C/F 兼容布局。测试与执行指南已同步该差异，没有缩减正式114个时间案例或预定51个内存案例。

**529294f4 候选已经被否决**：其 `512×512×512 padded_c_b ordinary` 约16.84 ms，旧版约12.47 ms，回退约35%。候选时间保留于 `timing-candidate/`，中止的内存结果保留于 `memory-candidate/`；身份为 `candidate-build-identity.json`，静态证据保留于 `production-static-candidate/`，本审核的候选版为 `final-code-tool-review.candidate.md`。不能把候选的正确性通过、其他布局加速或不完整内存结果用作最终性能通过证据。

**16d2候选同样未采用**：完整114项配对时间确认127³ CC比值1.1327（11/11轮回退）、33×33×39 CC/FC约1.14/1.13（均11/11轮回退）、512×512×513 CC约1.0234（9/11轮回退）。这些实际回退集中于非整8列，父任务因此新增 `N%8==0` 准入；没有删去这些正式case。完整16d2时间、51项/306子进程内存与旧审核保留于 `candidate-16d2/`，不能当作当前square结果。

**d2e候选也未采用**：实际准入37个regular-C案例中，32768×64×32 FC ordinary/prepared配对比分别2.4911/2.4540，64×64×32768 CC为1.5902，均11/11轮更慢；同一版同长路径51项冷暖heap峰值持平不消除这些时间失败。未改F控制亦出现波动，根因尚未分离，不能归零为噪声。完整114时间、51内存和审核已保存到 `candidate-d2e/`；`timing-stability-review.md`冻结该历史，不是square版本结果。父任务据此只保留方阵regular-C准入，矩形case仍保留在114/51里验证fallback。

## 最终范围与身份

按 `start-manifest.json` 的1475个轮初文件逐项比较，最终有以下8个既有路径发生变化；没有以巨大的 HEAD diff 扩大本轮范围。AGENTS、数学纪律、Excel纪律、CMake、IPC及性能工具均与轮初快照一致。

| 路径 | 最终 SHA256 |
| --- | --- |
| `cpp/include/calmetrics_engine/matrix_backend.hpp` | `cd5972679b13b49a86ed40a54f2292cca2ff60ea3ebedb9559d934210b2cb2d5` |
| `cpp/operators/matrix.cpp` | `df2d948fd4530b2ba0df4f41624a41c11340a218abaecec7a82ca152a591f362` |
| `cpp/operators/matrix_backend.cpp` | `aa30759a0e12710ad6e763fbe95550767cbf025e487aae35ce888ae59d94b3f5` |
| `tests/operator_native_tests.cpp` | `44747b59472a90acd42404b61788317c02f75900053515b2bcfc7b642adc8622` |
| `tests/test_matmul_library.py` | `5db7042b636344ca1ce44b30fe23f962060d89a20ba29e29d30805656585d7e2` |
| `docs/execution-guide.md` | `996b8e7b143aadbaaf9cf3d4f8a35aa5f273e002facd7979a5289451e107bf05` |
| `docs/publishing.md` | `6fc1b34839f0f0d782d41900e17c8b7045a93de6fc341dc1591856fbb54dc3a9` |
| `docs/physical-execution-optimization.md` | `ea1aec6e648f7c79db2bc50b753fe67f32d8eccc26e61b34a266639830508b77` |

旧正式baseline为 `1aec1a6b0d4a07f0caee43b58cd4b339bdcc165f5e615f2190b15da0ba66e86b`；新build为93dcfd08。正式driver使用真实同长度安装目录 `/private/tmp/calmetrics-matmul-wide/square-install-old` 与 `square-install-new`，两者都是54字节真实目录，不是符号链接。

已独立核对当前 `build-identity.json` 的全部生产源hash与square-install-new扩展/worker hash，均匹配93dcfd08；源码只增加regular-C方阵判定，backend helper不变。当前114项时间的old/new身份已逐项核实为1aec/93dcfd08，correctness均true；51项/306个唯一内存child的old/new身份也已逐项匹配1aec/93dcfd08；没有沿用旧campaign。没有import或执行二进制。
## 数学、边界、所有权与审计

- C 4×8 SIMD lanes 是独立列，四个 Row 对每个输出保持初始 +0、升序 k 的乘法与累加；行尾与列尾也完整走 k。严格C、方阵和整8列判定仅改变物理分派，不改变数值helper；先判断repeated-row，再判断整8列，故不误伤重复行。A 可负步长、间隔或零步长；A/B 可共享只读底层，输出仍须独立。
- F 4×4 helper 保留原数学路径。显式 ISA、小矩阵、其他平台或 Eigen-OFF 保留原入口。没有引入 BLAS 线程设置、私有线程池、动态数值缓存或额外输入打包。
- repeated-row 的充分依据是 A 的行 stride 恰为零，不是近似值相同。首行每列完整计算所有乘积，包括 `0*Inf/NaN`；随后 `std::copy_n` 写入本次输出各个不重叠行。输出不是广播视图，输入、前次结果及 prepared 快照均不复用为新输出。
- `(M-1)*N*8` 在经原 shape/容量校验的独立输出范围内计算，进入已有 `algorithm_copy_bytes`。该路径标为 `matmul-repeated-rows`，没有冒称 Eigen；正常 C/F 路径标为 `eigen-array-matmul`。既有 graph 累加/IPC 字段承接复制计数，本轮没有新增协议字段。图和 worker 不保证单独传递 helper 的 ISA 字符串，文档明确此限制。
- **67项Python专项**：保留66项，另加40² FC准入正例。普通C的IEEE/非有限、worker比较边界、负/间隔stride、显式ISA和prepared夹具用40²；共享只读C继续128²。所有F原几何不变；原矩形37/39/40列布局case均保留，按最终predicate强制不命中regular C；重复行仍用非方阵33/37列，确保新条件不误伤它。原容差未放宽。
- 本审核者仅在temp准备square补丁，由父任务完成证据归档后应用；仓库当前Python文件与已审草案逐字一致（SHA见上表）。没有在d2e测量期间改测试，也没有自行启动运行。
- 原生126个组合=原90个矩形组合+36个40²组合，覆盖C/padded-C/F右矩阵、C/F/负/零stride左矩阵和scalar/automatic。F-left scaled-coordinate oracle使用 `a[source_column*rows+source_row]`，与实际列主stride一致，不另跑一次矩阵乘法充当oracle。F路径先检测当前平台可用性；若库可用，square普通C/FC必须真的命中Eigen；矩形、padded或scalar普通C强制不命中。原只读owner/padding、首尾哨兵、0workspace、复制量审计均保留。
- 原浮点容差 `2e-12/2e-14` 未改。old/new bitwise列用于证明本次保序实现，不扩张为普通 MatMul 的全域逐位数学契约；真实 bool/error、IEEE类别和特定比较边界单独校验。

已在当前square114时间记录复核9个ordinary/prepared zero-A案例，实际 `native_chunks.algorithm_copy_bytes` 均等于 `(M-1)*N*8`，包含padded-C和长K。direct不采audit，没有为采集审计额外运行MatMul。

## 最终square静态对象与栈边界

顶层static正文/summary/identity/反汇编现均为93dcfd08；独立核对6个对象实际hash均匹配。d2e原报告和反汇编完整保存在candidate-d2e，未被当前刷新覆盖。

实际matrix对象先于 `0x10dc` 分派零A行stride；严格C/整8列之后，`0x10fc–0x110c` 完成M==K、K==N判断。C恢复自身frame后在 `0x1144` 尾调用；F为 `0x19f4`，repeated为 `0x1a2c`；不满足准入时保留SIMD。matrix320 B、C400 B、F416 B、repeated本地80 B；旧C循环464 B、含selected_isa已知warm本地峰值496 B。backend及simd对象不变，未增加数值工作区、私有线程或结果字段。

static报告准确区分本地frame与完整请求：backend无FMA/浮点spill，但有整数spill；repeated真实调用系统memmove，冷版本查询/系统callee/共同调用者未赋零。不能因C本地frame较小或数值scratch=0而宣称完整RSS不增。方阵限定只是当前经验准入，static正文没有宣称任意方阵必然快。

## 正式工具公平性

### 范围、数据与身份

此前d2e manifest逐项确认原90及顺序保留；当前adapter/cases仍未改变，总数114，额外24项不变。当前square时间记录已完成114项且没有删case；原90+24及顺序保持前轮定义。带padding、负stride、间隔或广播视图从同一CC随机值构造，对old/new/NJIT使用同一生成器，没有根据失败结果删case。三次收窄分派后仍保留矩形、padded-C及非整8列案例。

临时 adapter 同时替换 base和bench的child入口，scalar reference、worker、memory child都会回到同一adapter，继承1aec baseline。preflight在独立进程取identity，新worker再核对冻结build id。当前driver传入同长度square-install-old/new；当前114项时间child身份均匹配1aec/93dcfd08；内存306个唯一child同样已核对身份匹配。529、16d2及d2e均已归档，不复用为square最终身份。

### 时间与真实输出生命周期

ordinary/prepared在可调用体内构造 `MaterializedResult` 并投影全部数值输出；prepared使用独立 `run_snapshot()`。输出不会到timer之后才物化，也没有额外flatten/copy伪造独立所有权。`result=function()` 在新返回前保留旧结果，旧结果的释放包含在赋值范围内。

参考校验、输入hash与高精度抽查在timer外；direct无额外诊断MatMul。三个独立worker每轮仅激活一个，11轮轮转先后并记录wall/CPU；M<256的case每轮5次（也包括部分大矩形）仍是11个统计组，不能当55个独立配对。NJIT编译实际readonly/layout签名，执行 `a @ b`，双方按同核数控制。NumPy threadpool记录不能单独证明NJIT库：现有静态链路为Numba capsule→SciPy→Accelerate NEWLAPACK，没有具体AMX硬件使用证据。

### 冷暖堆与RSS

`run_wide_memory.py` 保留原31项，再加20项预先固定案例，计划51×3后端×2测量=306个独立子进程；validator要求案例/后端/测量组合完整、唯一及build一致。当前driver使用同长square-install-old/new。529中止partial不得视为51/51；16d2完整报告保留原始+8 B；d2e同长路径51项冷暖heap峰值持平，但其时间失败保留。不能由此推断square完整内存结论。

imports/输入/NJIT实际签名compile在epoch前，compile不调用被测数学。graph/prepare/setup、首次数学和库初始化在cold内。连续heap epoch保留工作区，释放首结果后预热，再记录warm；峰值包含setup存活字节，不重复相加。warm赋值自然覆盖两个独立结果短暂共存，没有把库数值缓存预热到计费外。

局限须保留：输入hash可能在epoch前分配再释放整输入大小的bytes，因此不是空分配器原始启动；只测51/114内存。heap与RSS分开子进程；每个案例/后端RSS只有一次campaign，0.5ms进程内采样可能漏短峰值，不能证明栈或物理footprint不增。解释器、输入、监控线程及页面基线会影响RSS，但正增量不能预设全是噪声；需要时做外部多对归因。ASan脚本设置 `detect_leaks=0`，通过不等于泄漏检测通过。

## 安装路径引起的heap差异归因

只读核对 `equal-install-attribution/manifest.json`、6个child的 `memory.json` 及源码：实验明确是**1aec与16d2既有字节**，不是aligned新版本；ordinary/prepared/direct各一对。三个entry的setup/cold/warm全部 `allocated_bytes/live_bytes/peak_bytes/allocations` 差均为0。原不等长安装下，ordinary/prepared setup已有+72 B累计分配和+8 B存活，cold/warm峰值再呈+8 B；direct原本全0。归因实验没有改heap epoch、数值结果投影或仪器，也没有预先执行额外MatMul。

源码 `cpp/native_api_bindings.cpp:826–848` 每次构造EngineAPI都会从 `_native.__file__` 得到worker路径；`cpp/native_runtime.cpp:50–51` 将字符串保留在 `worker_path_`，并不要求真的创建process pool。macOS libc++请求容量与字符串长度有关。控制真实安装路径为等长后，这6个child的差异消失，支持这里的+8 B来源于安装路径，而非新增数值workspace；没有逐个瞬态分配的stack trace，不能宣称已证明+72 B中每次分配来源。

d2e使用53字节、当前square使用54字节的对称真实安装路径，是控制与代码无关的输入因素；并非修改内存门槛、删字段或扣掉测量到的字节。仍保留原51项中的所有正增量和原报告；后来d2e完整51项已证实冷暖heap峰值差全部为0；当前square完整51项也已独立测量，见下节。6-child本身仅是归因，不能单独外推为全部45/51增量已关闭，更不能证明RSS或系统分配不增。文档 `install-path-memory-attribution.md` 已明确这些边界。

## 当前square114项时间：收益与未关闭项

只读按真实输入shape/stride及当前predicate分类：**24项regular C、11项repeated、38项未改F、41项fallback**。统计是11轮的“每轮new/old中位数之比”的中位数；表中毫秒使用正式pooled median，两者不可混为同一比值。11轮仍是同批持久worker，不是11组fresh进程。

- regular C：**20/24配对改善、4/24退化；0/24快于NJIT**。其中3个退化是32²微任务；另一个是 **1024³ CC direct**，old141.628→new144.832ms，配对比 **1.0130**、6/11慢（逐轮比范围0.884–1.190）。不能写“所有中大型C都比旧版快”，也不能把此项先归零为噪声。
- 部分稳定收益仍有价值：512³ CC ordinary/direct/prepared约0.9444/0.9251/0.9367，FC约0.7381/0.7385/0.7351；1024³ FC三个入口约0.6501/0.6632/0.7594。但旧版/新版单进程状态差异仍未全部分离，不用它们掩盖上述例外或NJIT差距。
- repeated：**11/11配对更快且每项11轮全快，10/11快于NJIT**。例512³ C ordinary11.856→0.501ms，NJIT1.295ms；仍有 **64×32768×64 repeated-C** 4.579ms慢于NJIT3.034ms（配对比约1.484；两列总体中位数商约1.509），不能写所有重复行负载领先NJIT。
- 原三个大矩形已走fallback：32768×64×32 FC ordinary/prepared配对约 **1.0084/0.9970**，64×64×32768 CC约 **0.9974**；本轮没有复现此前2.49/2.45/1.59的大退化。此前失败历史仍保留，不能改写为从未失败。
- 未改F控制仍有大退化：**1024³ FF prepared** 158.141→248.558ms，配对 **1.5692，11/11慢**；FF direct205.349→221.725ms，配对1.0866、10/11慢。此前d2e的FF prepared也退化，尚未完成原因隔离。控制组异常不等于新C数学错误，但足以阻止“整套大矩阵无退化/全部通过”的概括。

正退化如实保留；不把较快case与较慢case相抵销，不把CPU≈wall、同长安装或对象不变当作无内存/性能风险的证明。当前时间证据支持限定范围内的部分改进；结合下列RSS正增量，仍不能宣称完整内存不增或联合采用全部通过。历史 `timing-stability-review.md` 仍是d2e报告，本次未覆盖它。

## 最终51项内存：活跃堆持平，RSS不增未证明

独立读取 `memory-final/memory.json`：**51个case、306个唯一case/backend/kind组合**，覆盖旧/新/NJIT×heap/RSS；所有原生child身份匹配1aec/93dcfd08，formal两阶段rc均为0。这是**51/114子集**，不是114项全内存覆盖。

逐项核对51个new−old：setup/cold/warm的 `allocated_bytes`、`live_bytes`、`peak_bytes`、`allocations` 全部精确为0；冷/热独立结果峰值与暖态11次累计分配差也均为0。实际结果及前份独立输出的存活计费不变，同长安装路径的因素没有靠扣减测量值处理。

RSS单次独立进程样本仍有正增量：

| 指标（new−old） | 最小B | 中位B | 最大B | 正值case |
|---|---:|---:|---:|---:|
|总RSS峰值|-51,937,280|+98,304|+26,689,536|27/51|
|请求RSS增量|-50,511,872|-32,768|+25,083,904|19/51|
|基线RSS|-3,751,936|+16,384|+2,031,616|26/51|

冷暖heap逐项相等不解释这些RSS差，也不保证系统memmove/首次触页或完整调用链没有额外资源。每个RSS组合仅一次campaign，采样可能漏短峰值；当前不声明统计等价、不把所有正差当噪声，也不把该结果推广到未测的63个时间case。**完整内存不增仍未闭环。**

## 生成文档一致性核对

只读对照新 `docs/physical-execution-optimization.md` 首节、`docs/physical-execution-evidence/2026-09-30-matmul-wide/README.md`、`formal-summary.json/md` 与原始114/51记录，未发现数值或状态冲突：都标93dcfd08、7129、67专项/126原生组合、各9/9、本轮phase2 4/4；旧d2e3/4及三轮否决证据保留。

两份repo文档准确披露20/24 regular-C改善、0/24快于NJIT、1024CC direct1.0130退化、未改FF prepared1.5692，以及51/114内存范围和27/51 RSS总峰正值；标题及结论均标联合验收未闭环，未用表格中的快例子掩盖退化。时间配对统计与总体毫秒中位数明确区分，M<256的每轮5次说明也包含大矩形，不再误称全部微案例。

数学/API/图协议/Excel公式与预算未改，469/3056是既有真实Excel快照复比，不是新Excel重算；ASan不宣称LeakSanitizer。新证据README属于新增材料目录，8文件范围仅指轮初已存在的路径。证据文件归档及链接最终落盘由父任务执行，本审核未执行归档。

## 当前回归结果与未关闭目标

当前square正式验收由父任务统一串行执行。只读核对新日志：Python **7129 passed**；native、ASan/UBSan、Eigen-OFF各 **9/9**，185算子回放与Excel快照阶段rc均为0；build identity为93dcfd08。新square phase2 **4/4通过**，63点ordinary为4.458µs、配对比1.0。这里是新源码构建的一次独立验收，不是对d2e失败的无改动择优重跑。

历史d2e严格phase2为3/4：63点ordinary Scheduler4.583µs，配对比1.0092490640828011>1.00；历史rc=1保留在candidate-d2e。用户允许微任务轻微劣化，不等于旧严格门槛当时全通过；square新构建的独立验收也不能追溯抹掉该历史失败。

新114项时间与51项/306child内存均已完成并保留全部正负结果。正确性和原四组门禁通过，仍有1024CC direct的实际退化、未改控制的运行状态差异、普通C对NJIT的显著差距以及RSS正增量。因此只能确认本轮实现与局部收益，不能宣称中大型全面领先或完整内存联合验收通过。Excel若只核验历史469case/3056结果快照，不得称为此次Microsoft Excel重算。三轮已否决性能、原不等长路径内存、d2e完整正/负RSS观察均须保留，不能用方阵子集或广播场景收益宣称全部中大型目标闭环。
