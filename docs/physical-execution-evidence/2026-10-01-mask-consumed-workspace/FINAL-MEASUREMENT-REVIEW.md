# Final measurement review: methodology and completed evidence

日期：2026-10-01。先在正式 timing 运行期间审核方法；root 确认所有计量完成后，再只读核验 timing.json、12 份 memory.json、summary.json、cleanup 及 provenance。没有执行数值、编译、测试、基准或等待，没有读取未完成成绩，也未修改计量脚本、结果 JSON、frozen tools 或生产源码。唯一写入为本报告。AST、纯 case metadata 及 JSON 记录一致性检查没有导入数值包或执行 inputs/setup/backend。

## 当前结论

**方法及完成记录审核通过，未发现测量身份、线程、所有权或增量汇总错误。**当前 summary 的限定采用结论与原始数据一致：按消费量分配 masked-order 工作区，显著减少申请堆，并改善相对于旧 native 的完整请求时间。不能扩大为所有算子、所有内存状态均领先 NJIT；全引擎目标仍未完成。

静态定义与完成记录均为30个 timing case、12个 memory case。30个case_id唯一且包含density；12个memory筛选各且仅匹配一个case，覆盖两个4M unmasked、两个masked算子的4M .01/.55/1、16M sparse和4M prepared sparse。按三后端×heap/RSS，共72份 fresh child 报告；36份 heap报告的6个连续阶段共216个stage，overflow全部为0。9个轮换计时round；63点每round5次，其余每round1次；三后端原始样本与median一致。审查者未运行这些定义。

## 身份与后端隔离

- benchmark.py:23–29 固定 baseline `850871ebcff4f6f2e089ae96029f736ccc0c94282cd06824193f839a410906b3`、candidate `ccca6b67a5220ea2134a4743d281b80c5af4c096e7d41c3d5b747bb8de70e331`。CME_OPT_NEW_BUILD如与固定candidate冲突会拒绝，而不是默默接受其他构建。
- 安装目录分别为 `/private/tmp/calmetrics-chol-group-next/install`、当前stage/install。公共 child_env:426–434按后端设置唯一PYTHONPATH，并去掉父进程预加载插桩；只有heap子进程注入探针。默认路径及run_memory显式参数一致。
- 两份已完成identity.json的build、source recomputed及runtime identity相同，source_wheel_runtime_identity_match为true。方法阶段核验的是既有身份文件；完成阶段另逐项核对了实际worker的build/audit，见下方完成记录。
- stage audit 对每个C++实际结果的engine_build_id强制匹配固定后端身份。真正加载不同wheel会报错，不会纳入一个看似配对的成绩。
- 冻结两工具与上一正式stage逐字节一致。下表标识本次方法读取的wrapper和工具字节；冻结最终证据仍需使用实际执行时的字节，而不是后来注释修改后的SHA。执行时顶部继承的Source wrapper注释已在计量完成后重标；原始执行字节完整保存在benchmark-executed.py。provenance.json的执行前后SHA、当前父wrapper SHA、common tool SHA均与实际文件一致；逐行diff仅一条注释，AST完全相同。历史执行身份不以新注释替代。

## CPU与真实库线程控制

- benchmark.py:174的两个C++入口均由 AdaptiveScheduler(cpu_budget=1) 执行。普通调用和prepared.run_snapshot采用同一预算，不给native额外线程或外层并发。
- 公共工具在任何数值包导入前固定五项线程环境；但方法并未把环境变量单独视为公平性证明。
- baseline_threads:120–164加载SciPy库后，再用threadpool_limits(1)限制并记录threadpool_info实际池；macOS≥15额外调用BLASSetThreading(1)、BLASGetThreading确认active==1，退出时恢复并核对原策略。当前order参考为nogil的普通NJIT，没有parallel=True。
- native不会被测试控制器额外套全局线程设置；它须自行服从Scheduler和构建的Eigen无内部并行。完成阶段已检查C++audit实际budget/执行线程字段及NJIT pools/active；证明来自实际记录，而不是仅凭静态配置。
- timing的三个子进程虽共存，但父进程逐个send并等待返回，9轮轮换old/new/njit先后，无同时数值执行。内存driver每个case以及公共工具每个后端/heap/RSS均subprocess.run串行执行。root仍须确保外部没有并行重型测试/构建。

## 完整结果与所有权

- ordinary execute和prepared.run_snapshot返回后，在被测callable内部构造MaterializedResult；公共工具:222–231同时保留raw和已声明的field.values[0]包装。计算、结果分配及公开typed字段封装均在timer/heap/RSS边界内，没有计时后才展开懒结果。
- timing初始化先保留首次raw/数组，第二次返回后逐位核对原结果未变化，并用shares_memory核对输入和前后输出独立。外部校验使用的copy/np.savez在计时之外，不当作结果所有权替代。
- 被测 `result = function()` 在新结果求值完成前仍保留前次result；warm heap/RSS循环同样如此。因此稳定warm峰值包含两次独立结果的交接寿命，不能改写为单输出零所有权开销。
- 冷heap先持有首结果记录cold_peak，再释放该结果；cold_retained/warm_baseline表示剩余图/Workspace等常驻量。该名字不是“首结果在全部warm调用中保留”的证据。此前后端生命周期一致。
- 新旧C++成功结果逐bit相同；NJIT按现有allclose并单独核对finite/NaN/signbit/zero分类。未放宽容差或更换参考算法。失败/所有权回归仍属于正式测试，方法检查不替代它们。

## mask与RSS准备

- 每case固定相同seed，先创建readonly目标源；mask在设置只读标志前，按8192最大块原地生成。这里readonly是返回边界状态，不是说构造期源不可写。
- .55使用原>.45条件，.01使用>.99，1直接填true。密度在首次mask生成时决定，没有在NaN插入后重新抽样或修改mask；每块只有~mask的false位置被np.copyto设成NaN。未消费NaN保持为测试控制，消费值来自原normal输入。
- 概率密度.01/.55不应写成精确消费比例；consumed_count/workspace核验使用执行audit及输入身份，不能仅用density*N。alltrue才有严格count=N。
- inputs与input_identity在fresh memory epoch前执行。当前12个memory case全部random C向量；没有reverse中释放整条向量临时区的问题。mask RNG/反掩码临时数组最多8192项，C输入hash使用memoryview，无全输入复制。未预热数值排序，也未在测量前预留算法Workspace。
- 同seed、shape、dtype、strides、readonly及源/mask SHA在三个后端间记录并核对。完成阶段已将每份memory输入SHA与对应timing case关联，避免误将不同密度拼在一起。

## 申请堆与RSS边界

- measure_heap:33–58为一个连续malloc live追踪epoch，setup、首次调用、释放、五次warmup、warm调用均在其中。reset_peak保留当前live；不能把setup峰与cold/warm峰相加。C++图、prepared绑定、消费count扫描及常驻scratch都不能因为发生在setup就称免费。
- 同口径cold/warm报告各含实际result寿命；warm三次调用至少覆盖前后独立结果交接。trace overflow会拒绝报告，完成阶段已核对全部stage counter无overflow。malloc只记录追踪到的申请字节，不完整覆盖Python arena、mmap、别的进程或全物理内存。
- RSS为另一个全新未插桩子进程；NJIT仅compile(signature)在epoch前，不做reference数值调用。native包导入在外，scheduler/图/Workspace setup在首次cold_through_warm调用内，从cold至warm连续观察。
- 监控线程/psutil先初始化，再记录idle control及rss_baseline；idle_delta没有被减去制造优势。0.5ms采样和调用尾部当前RSS共同形成观察峰，单fresh sample仍有页级和采样限制，不是完整进程树峰值或分配计数，不能称rss_delta==0为零算法内存。
- driver拒绝已存在目标目录；每case/后端/kind保留原始JSON和stderr。合法的负性能数据本身不终止本工具；check=True只对实际测量/校验子进程异常负责。

## 性能政策

两冻结工具没有以time/heap/rss正差或strict false触发采用veto的分支。本轮wrapper也没有速度或内存比较门禁；硬失败用于身份、数值、所有权、输入不变性、追踪溢出或进程错误。

最终报告须同时保留绝对毫秒/字节差、相对差、完整负载及资源约束，以及独立采用理由。几KB内存增长和几ms轻微退化不能独立拒绝。旧strict字段可保留诊断；它们不等于全引擎已领先，也不能隐藏中大型实质时间/内存欠项。不得另设一个未经授权的通用KB/ms阈值。

## 完成记录核验

- 30个timing case全部有9轮正确轮换；每轮三后端样本数相同，raw samples重新汇总的median与timing.json、summary.json逐项相同。各case三后端的完整input metadata及SHA相同；所有new旧C++及NJIT对照canonical_changed_bits为0；independent_results为true。
- 全部旧C++实际audit为850871eb，新C++为ccca6b67；CPU预算、cpu_tokens、native_threads、thread_count、threads_per_process均为1，native_processes为0。NJIT实际OpenBLAS pool.num_threads为1、Accelerate active为1；timing cleanup及24个NJIT memory子进程的恢复记录均为prior=restored=0。
- 72份memory报告与对应timing case的输入、build一致。连续malloc cold/warm原值、216个stage overflow=0、fresh RSS本底/峰/增量、idle_delta=0及summary差值全部对应。各case分开比较，没有用总体平均掩盖负结果。所有summary strict_comparison_is_adoption_veto为false。
- benchmark-executed.py的SHA与本次原始方法审查的960ed612一致。当前benchmark.py仅重标注释，SHA为78be5f14；provenance可追溯，执行代码未变。没有重写原始成绩。

## 真实申请堆收益与RSS限制

下表warm heap是连续epoch的完整请求峰值，包含常驻工作区及独立结果交接；RSS是另一个fresh进程的完整cold-through-warm观测增量。单位MiB（2^20字节），不能混用成同一项“峰值内存”。

| case | warm heap 旧 / 新 / NJIT | fresh RSS增量 旧 / 新 / NJIT |
| --- | --- | --- |
| median_where 4M .01 ordinary | 32.0100 / 0.3297 / 0.6401 | 32.5938 / 0.9063 / 0.6719 |
| quantile_where 4M .01 ordinary | 32.0105 / 0.3302 / 1.0001 | 32.5781 / 0.9375 / 0.6875 |
| median_where 4M .01 prepared | 32.0102 / 0.3299 / 0.6401 | 32.5625 / 0.9219 / 0.0156 |
| quantile_where 4M .01 prepared | 32.0106 / 0.3304 / 1.0001 | 32.5781 / 0.9219 / 1.3281 |
| median_where 16M .01 ordinary | 128.0100 / 1.2872 / 2.5550 | 128.5625 / 1.8594 / 3.8750 |
| quantile_where 16M .01 ordinary | 128.0105 / 1.2877 / 3.9922 | 128.6094 / 1.8750 / 9.5000 |

4M sparse申请堆由约32.01降到0.33MiB；16M由约128.01降到1.287MiB，约99%减少。cold峰值同方向，分别新0.3292–0.3298MiB、1.2866–1.2871MiB。native audit实际工作区4M为335,264字节（41,908个double），16M为1,339,240字节（167,405个double），与消费量一致，而不是density*N近似值。

4M .55 warm heap新17.6048–17.6052MiB、旧约32.01MiB；fresh RSS新18.1875–18.2031MiB、旧32.5625–32.5938MiB。alltrue申请堆与旧实现持平，不能声称它也减少了自身工作区；它仍低于本次NJIT参考的64.0006/100.0009MiB。

**并非所有驻留状态领先NJIT。**4M sparse的fresh RSS新0.9063–0.9375MiB，对照NJIT为0.0156–1.3281MiB；ordinary median/quantile及prepared median分别多0.2344、0.2500、0.9063MiB。原值和正差完整保留，单次页级/采样差不能用来宣称普遍领先或独立否决。尤其0.0156MiB的NJIT观测不是零算法内存。unmasked median的warm heap仍比NJIT多7,876字节，RSS增量多0.5156MiB；不能把整进程RSS本底差当作算法工作区收益。

初始及稳定稀疏调用的结果不证明任意生命周期都维持这些峰值。summary保留Planner/prepared的原source-N上界、显式reserve以及dense-to-sparse复用的high-water限制；这些跨调用负载没有在本轮12case内计量，不关闭其成本。不可强迫缩容来换取单次图表。

## 真实时间收益与未关闭项

以下均为完整owned-result请求、1CPU、9轮配对median，单位ms：

| case | 旧 / 新 / NJIT |
| --- | --- |
| median_where 4M .01 ordinary | 12.1348 / 5.8578 / 5.8706 |
| quantile_where 4M .01 ordinary | 11.7416 / 5.7823 / 5.9151 |
| median_where 16M .01 ordinary | 47.2299 / 23.0895 / 23.1702 |
| quantile_where 16M .01 ordinary | 46.4337 / 22.2933 / 22.5243 |
| median_where 16M .55 ordinary | 148.1606 / 123.5358 / 137.8874 |
| quantile_where 16M .55 ordinary | 183.6092 / 160.3099 / 180.2693 |
| median_where 4M .01 prepared | 11.9729 / 5.9136 / 5.7763 |
| quantile 16M unmasked | 176.7436 / 176.6392 / 139.3775 |

masked 4M相对于旧native节省约5.92–6.28ms，16M节省约23.30–24.62ms；结合数十至上百MiB申请堆减少，这是有效的完整负载收益。sparse时间主要追平NJIT，绝对领先仅0.013–0.231ms；不能宣传成对NJIT显著提速。prepared sparse median仍比NJIT多0.1373ms，不是单独采用否决项。

16M unmasked quantile仍比匹配NJIT多37.2616ms（约26.7%），是中大型实质差距，应保持优化事项未关闭；此轮相对于旧native几乎不变。16M unmasked median为144.6776 vs 138.0342ms，多6.6435ms（约4.8%），保留事实，不单凭该毫秒正差否决本次masked-workspace采用。4M unmasked median的0.3422ms正差及63点的0.0038–0.0047ms正差同样不能当独立门禁。

summary的限定adoption及whole_engine_objective_complete=false与数据一致。此前大矩阵Cholesky/LU/Jacobi欠项未在本轮重新测量或关闭；185个fixture的正确性和其他回归数量不由本次性能记录自动证明。Microsoft Excel为历史快照对照，没有本轮重新计算；macOS arm64结果不能代表Linux/Windows的优化性能。本报告仅确认计量方法、完成记录及限定结论的一致性；代码回归/语义/预算自审核由对应证据独立承担。

## 静态来源SHA-256

| 路径 | SHA-256 |
| --- | --- |
| `benchmark-executed.py`（实际执行字节） | `960ed612d6d80555a398d908cfb0e23f5dc940695a9b0a58ef2707d9d1fa8267` |
| `benchmark.py`（计量后仅重标注释） | `78be5f1488af75f6ce4738363ca344142bd477d11b5af4fd39d5d26801aaec4a` |
| `run_memory.py` | `e7b4a5c2d01a02682825047c783599497f50b93b877ceacddca02146d7ba6120` |
| `tools/check_large_matrix_performance.py` | `2f99bf41adac0926e30aa2045fa0f6bb67ea938826bf6a4c69ed6cf278c10518` |
| `tools/benchmark_memory.py` | `b3862bb22e58f01f874bad0fd8466a3eb63ac4f3803186846184aafb47f4d7f8` |

## 完成证据SHA-256

| 路径 | SHA-256 |
| --- | --- |
| `formal-timing/timing.json` | `8ec4593ae06dc30056a56f1a04519946caf41268601529a4e358350b8d7eea18` |
| `formal-timing/timing-worker-cleanup.json` | `9eabafe5c7ac6ae728ad0f775c392c98b3e03008b29cfc2e2b78fb6dd1df5f96` |
| `summary.json` | `f252bc1539134fe2f47fdebcd02ecf9f69a87389ff8eeae293877c1737acc060` |
| `provenance.json` | `fc08a8a0c20e18910d4a78b873039d1c53a69843e81d5cda8dc214f7c6991fc1` |
