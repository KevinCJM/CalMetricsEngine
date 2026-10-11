# 下一里程碑：完整 compact MC 数学链与三个原语

只读设计，2026-10-03。CodeGraph先定位真实FIRP projection→NAV→fan调用，再核对当前源码、canonical registry/shape/Excel预算。未修改生产，未编译、import Python、运行数值、测试或性能。本文的阶段耗时由root提供，非本代理独立复测。

## 1. 决策与收益来源

下一轮优先完整 compact 链，暂不继续多轮投影微优化。root 的固定 transformed tape、健康输入、同一完整数值输出 profile：C8约3107.5ms，projection约2091.3ms，NAV约84.7ms，fan约930.7ms；C1约390ms，projection约261ms，fan约117ms。编译/输入准备、coverage/random/JSON在计时外。它证明 fan/排序值得作为完整链候选，不证明实际API或全seed验收完成。

真正新增的独立数学能力只有三项：显式初态的任意axis inclusive product；显式初态的任意axis inclusive strict maximum；共享每条axis排序的多概率quantiles。NAV/回撤/fan/概率/贡献由这些与已有算子组成，不能注册一个业务名MC黑盒。私有 compact fusion 是这些逻辑步骤的物理执行，不是第四个数学算子。

年度 moments 不属于当前 `scenario_stress/engine.py::_monte_carlo_results` 链。它在 `strategic_allocation/reference_inputs.py:224` 与 `cma_statistical_kernels.py:183/209`，应作为另外的真实消费者工作包，不能放进本轮MC必需项。其 raw sum×(252/rows)、已有 covariance、shrink、sqrt与安全分母组合优先复用现有原语；零波动correlation原返回NaN，而CME全局correlation会报错，不能直接替代。也不能对P条MC路径构造P×P covariance来冒充年度moments。

## 2. 建议冻结的公开合同

保持旧187项ID与合同。以下新ID暂顺接188–190；名字可依项目约定调整，但参数和数学树必须明确。

| 原语 | 参数与输出 | 数学与错误合同 |
| --- | --- | --- |
| `cumulative_product_axis(values, axis, initial)` | values float64 rank1..3；axis标量整数0..rank-1；initial float64标量；输出同shape/逻辑axes | 每个非axis坐标独立 acc=initial，按axis升序先acc*=current再写包含当前项的acc。没有factor/block小计、reassociation、忽略坏值或金融阈值。IEEE NaN/Inf/±0/overflow传播，不加finite gate。空axis/空outer输出空且不读数值。初版不增加无人消费的per-line initial向量。 |
| `cumulative_max_axis(values, axis, initial)` | 同上；输出同shape/axes | acc=initial，逐项仅当current>acc更新；否则保留acc。NaN current比较为false，NaN initial维持NaN；相等/±0保留原state，Inf按严格比较。空输出不读payload。不是fmax、默认initial0或并行max。 |
| `quantiles_axis(values, probabilities, axis)` | values float64 rank1..3；probabilities float64 rank1，Q个有限[0,1]；axis标量整数；输出同rank，该axis长度改Q | 每条axis line copy一次N值、用成熟std::sort一次服务所有Q，其他line顺序与逻辑坐标独立。NaN-last比较器复用当前排序语义；generic NaN payload/tie位序不扩大为新承诺，混合signed0/NaN物理加速需canonical回退。position=(N-1)*q；lower=floor/upper=ceil；N1或lower==upper直接返回选中值；否则lo*(1-fraction)+hi*fraction。保留Inf，不能强制finite或改成lo+(hi-lo)*f。 |

补充边界必须在开发前写定：

- shape/dtype/rank/axis/initial标量约束独立于上游payload是否失败；未知geometry只能保留unknown位与容量0占位，不能仿造已知rank1。概率值不可用时不伪称验证通过；已知概率列表的每项是独立配置校验，即使outer为空也应按约定检查。
- Q0输出空、不排序，scratch0；N0且有非空Q输出/非空outer时报INSUFFICIENT_SAMPLE；outer0输出空不报样本不足。checked产品计算非axis line数，不能直接 total/N 导致N0除零。
- AST沿现有axis算子要求axis literal；direct接口允许受检的runtime标量axis。product Typed IR要求increments无量纲，输出继承initial语义；max要求values与initial可兼容；概率无量纲，quantile值继承values语义。probability替换轴标为`component`并保留Q映射/provenance，不能继续标作被归约的path/time。
- 输入C/F/合法padding/负stride/零stride只读；kernel按坐标读，Output.set按C/F写。结果独立所有者，direct Python out继续已有C-only合同；graph/native显式F需要真实writer测试。
- prefix canonical无需堆scratch；quantiles只需N doubles/worker，std::sort自身stack有界，不能Q份完整copy。不为小排序开发自研库。与旧global quantile有意区分整数rank+Inf分支，不无授权改旧ID。
- 非最近舍入、signed0、NaN分类与浮点异常承诺沿现有数学纪律，不能因存在SIMD就放宽。并行只能跨独立line；单line递推不可并行归约。

## 3. 最小源码、结果协议与测试落点

建议新增一个窄实现单元 `cpp/operators/axis_primitives.cpp` 和一个Excel单元 `cpp/excel_axis_primitives.cpp`，仅服务这3项，不建general map/loop框架。延用Family::matrix，增加精确3项predicate与prepare/execute入口；不把它们塞进旧rank1 sequence默认验证或未经证明扩展pointwise safe_op。

| 范围 | 最小落点及必要变更 |
| --- | --- |
| catalog/canonical | `cpp/include/calmetrics_engine/operators.def`追加ID；`operators.hpp`3项predicate/声明；`registry.cpp`prepare、structure、execute、F-output许可、metadata/默认值/temporal_dependency；新轴TU共享同一fold/排序数学体 |
| Typed AST | `cpp/typed_ir.cpp`或窄typed helper；必要`typed_ir.hpp`声明。按上述axis/initial/Q/dtype/semantics推导，不改parser语法、不加模板业务名 |
| DAG/capacity | `cpp/compiler.cpp`cost/输入扩展；`cpp/graph.cpp::required_array_capacity`与library workspace获取调用真实known mask。prefix容量=X.size；quantile按Q替换axis checked推导；输入未知保持unknown，独立坏配置仍报错 |
| Planner/runtime | `cpp/planner.cpp`普通图N-double排序scratch与每worker预算；处理Q0/outer0与最大window。普通入口保持唯一canonical；新compact物理proof后续单独接。`graph_bindings.cpp`/typed result writer仅在现有通路缺项时补，不能多加一个结果协议 |
| worker/ownership | 既有typed roots与program encode/decode传递opcode/shape/dtype；检查`native_process.cpp`等opcode边界和snapshot身份。无需新IPC数组形式；测试encode→decode、ordinary/prepared、thread/process、两代结果共存及只读输入pin |
| Excel | `excel_internal.hpp`声明窄recipe/必要export-only域检查；`excel_operators.cpp`路由；新recipe；`excel_budget.cpp`独立符号上界。`CMakeLists.txt`仅加真实TU；覆盖数目/算子manifest与实际operator_count一致 |
| 文件/文档 | 项目相应算子说明、Excel复现合同、physical execution plan及路由按最终差异更新；不机械结项旧完整MC/seed/服务迁移，未测项标待验 |

建议只加两组有消费者的新Python文件：`tests/test_axis_path_primitives.py`覆盖3项公开/math/shape/protocol；`tests/test_mc_compact_graph.py`覆盖真实NAV/fan/summary组合及后续proof。现有native operator/graph/compiler/planner测试target加边界案例；`tests/excel_cases.py`与导出/符号budget测试增加3项和一个小完整MC组合。无需每个分支另造长期benchmark框架。

重要测试：rank1/2/3、axis每位置、F/C/negative/padded、initial非1、空/singleton、N0/Q0、概率0/1/整数rank/非整数rank/+Inf/NaN、strict max ties/signed0、product cancellation/underflow/overflow；上游失败+独立shape/axis/probability错误；direct out拒绝F、graph F独立结果；Planner预算恰界/不足；worker roundtrip/取消/失败后复用；compiled数学树与原FIRP kernel固定period/tape oracle对照。

## 4. 实际 MC 消费合同与有界执行

真实FIRP规模：P100..50000，T1..1200，N=T*P<=2m，N*F<=8m，F<=32、A<=64、C<=32，N0有限正值。`engine.py:843-903`在组合定义顺序逐个coverage→projection→NAV→fan，然后包装JSON；不调用annual moments。

每组合必须返回：5*(T+1) fan；最前min(12,P)*(T+1) NAV样本（含N0列）；14个terminal统计；A个贡献及coverage/limit摘要。全period、全NAV、terminal-P向量没有API消费，不是必需根。数值数组最低payload为 `[17*(T+1)+14+A]*8` bytes/组合（P>=100）；此外steps、Python/JSON封装、参考元数据都另计，不能称0。

逻辑DAG先表达：

```text
shared R=expm1(affine_contract(transformed_tape,K,Nintercept))
period_c = sum_axis(broadcast_like(w_c,R)*R, asset_axis)
nav_after_c = cumulative_product_axis(1+period_c, time_axis, N0)
peak_c = cumulative_max_axis(nav_after_c,time_axis,N0)
dd_c = 1-nav_after_c/peak_c
max_dd_c = max_axis(dd_c,time_axis)
terminal_c = last_time_nav/N0-1
fan_c = quantiles_axis(nav_with_initial,[.05,.25,.5,.75,.95],path_axis)
terminal summaries = quantiles + explicit masked raw sums/counts/divides/max
contrib_ca = ordered T→P sum(weight_ca*R_ca)/P
```

首N0列必须给真实可编译的组合。现有 `broadcast_like` 的values不接受rank0（matrix_foundation.cpp:355/361，typed_foundation.cpp:68）；`stack(N0,N0,0)`得到component轴，也不能直接broadcast到time轴。健康NAV域可用第一time列slice×0+N0再concatenate，且异常图结果不得发布；或者由调用方显式提供可正确声明为static的N0初态列。不要以一句broadcast scalar掩盖现有约束，也无需为此先造额外primitive。

随后新增精确私有proof，仅在compact根、无observer/中间公开根、已识别同一producer和顺序数学体时准入：

- asset return bounded tile一次共享；每组合period asset左fold的product同时更新contribution。
- 每path维护NAV/peak/maxDD，逐step严格递推；每step完成全部P后才sort截面、写5q与首12样本。P必须有barrier，不能拿一个4096-row tile当完整分位数样本。
- 完成time后重复terminal NAV5q复用已写fan最后列，不再额外copy/sort终态NAV；末端对return和maxDD排序统计。所有tail sum、drawdown sum按原path顺序；保留整数q直接值与terminal Inf。
- 3*C*P doubles基础state，P-size sort workspace可顺序复用；terminal return可在死peak状态区重用，不复用已返回根。严格fault协议可能需要额外perpath min/firstfault状态，必须预算，不把3CP当绝对上界。
- C批次大小按预算选择：C8/P10000基础state1.92MB；C32/P50000为38.4MB，可能超过原逐组合数矩阵。不能默认allC最优；小combo batch会重复producer，需要完整时间/内存权衡。
- 公布全period/NAV/asset中间根或observed/isolate/非匹配图时走原通用图且按完整结果预算。不能compact融合后把scratch借给独立根。

## 5. 不能改变的故障顺序与真实内存基线

projection遍历T→P，先严格比较更新minimum（初态Inf，NaN不更新），再nonfinite status1、<=-1 status2；首fault不写该period cell，贡献尚未执行最后/P。NAV遍历P→T，同样先minimum/return gate，然后递推后检查nav有限且>0，status3；terminal nav/N0-1没有额外finite gate。API公开projection/NAV status1/2的bad_step+1和经_json_number处理的minimum；status3仅INVALID_SIMULATED_NAV，raw kernel坏path另算更强验收。

融合的首fault选择必须按 `(portfolio定义序, coverage/projection/NAV stage, 各stage原遍历)`，而非物理发现时间。具体陷阱：同组合晚step的projection失败原本阻止全部NAV，必须优先于已流计算发现的NAV故障；后组合coverage故障不能抢先于前组合NAV故障。先冻结fault board/停止消费规则，再实现stream，不能后补免责声明。

原FIRP没有N*A资产L/R。共享tape<=64MB、Student chi<=16MB；正常每组合period约8N、NAV约8P*(T+1)。Python RHS替换使旧period/NAV与新period或新NAV短暂共存，约3份N矩阵可达高水位（需fresh heap/RSS验证），不是C份常驻或A份资产中间矩阵。source fan只需P-sized几份排序copy，JSON最终保留所有组合compact结果。

对T200/P10000/F4/A32/C8，原N矩阵每份16MB，三份约48MB；compact基础state1.92MB+sort/tile/诊断/独立小根，具备数十MB级机会。这是源码尺寸推导，不是已测节约。保留两代同完整compact结果，heap包括constructor/retained workspace，RSS fresh无DYLD子进程；比较同核、同输出、同healthy/fault范围与同random/JSON排除范围。不能用当前选择全period的128MB C8根基线证明实际API节约。

## 6. 最优开发与验收顺序

1. 冻结上述3合同及partial geometry/Typed axes/Excel数值域；只写canonical数学实现与AST/DAG/Planner/worker归属、真实组合测试。
2. 固定period→NAV→fan/14metrics完整正确性闭环；再接固定transformed tape→贡献→完整compact结果。普通图物化可做oracle，不能作为最终大型采用方案。
3. 按profile把共享projection与NAV/fan串为有界compact proof，保留故障board与独立根。先stdlib sort每line一次并复用5q，后续成熟selection库只在完整链确有收益时筛选。
4. 对完整C1/C8，T200/P10000/N2m/F4/A32测ordinary/prepared、强shared与真实service NJIT；补T5/P50000/N250k/F32、T1200/P1666/N1999200/F4，验证C批次state/重复producer取舍。小KB/ms不作门禁，不能用C8胜出掩盖大型C1实质劣化。
5. 单独关闭available子轴take/scatter、旧seed MT/normal/Cheng流与native LAPACK root相同basis，再完整normal/Student/fault/真实API集成。当前185/187历史Excel或fixed tape证据不能抵扣。

## 7. Excel：逻辑同图，物理预算另算

prefix每条chain有显式initial步骤与升序前状态公式；strict max用受保护IF比较而非MAX忽略NaN。product导出应检查所有可消费前缀的Excel可表示域；内部Inf随后NaN不能吞#NUM，只在导出触发检查，普通native不增加scan。

多q先复用当前透明稳定rank展开，每line只构造一次ranks服务全部Q；不要每q重复N²rank，也不直接用另一个统计函数改变算术树。符号上界可按 `L*[N*(N+sum_cells(N))+Q*(2N+16)+16]` 起步作保守recipe上界，追加既有input/guard/provenance/roots与公式/timeout成本，checked overflow。prefix上界包含L个seed、M项递推及guard/独立输出。具体formula计数由真正recipe反向核对，不把草图公式当已证上界。

Excel以逻辑依赖展开，不因C++ tile复用覆盖已引用单元格。全MC N2m通常在input即接近/超过2m全工作簿预算，新增状态/排序必超；必须预检OVER_BUDGET，不能分sheet规避或自动降为fixed tape。新recipe需要目标Microsoft Excel实际重算、编辑/域/budget测试后才宣称验收。

状态：设计可执行；3个原语、compact proof、Excel、旧seed与真实平台接入均未由本报告宣称完成。
