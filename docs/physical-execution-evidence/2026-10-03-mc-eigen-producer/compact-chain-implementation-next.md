# 下一轮：可执行的完整 compact MC 数学链

2026-10-03，只读设计。CodeGraph先定位FIRP实际调用，再核对当前CME源码、冻结参考体、profile和heap/RSS驱动及JSON。未修改生产、编译、import、运行数值/测试/性能。本文取代前一份next设计中未具体展开的初态列和ES空尾部写法。

## 1. 应解决的真实问题

`profile-complete-reference.py`保留FIRP逐组合 projection→NAV→fan 的数学体、调用次序和compact输出；编译、数据准备在计时外。范围是健康输入、固定已变换随机带，排除coverage、旧seed、协方差root、factor transform、limits、JSON和首故障病例，不能称完整平台迁移或完整MC验收。

| 同一固定带参考 | C1 | C8 |
| --- | ---: | ---: |
| 总时间中位数 | 390.098459 ms | 3107.504417 ms |
| projection | 260.978375 ms | 2091.328793 ms |
| NAV | 10.862792 ms | 84.692832 ms |
| fan/statistics | 116.762625 ms | 930.692916 ms |
| compact数值payload | 27,704 B | 221,632 B |

规模T=200、P=10,000、F=4、A=32、N=T*P=2m，同核预算1。C8瓶颈是重复producer和fan排序；NAV单独已只占约2.7%，不应再次围绕小递推耗时做多轮优化。

内存原始记录：warm requested-heap峰值 **48,903,416 B**；RSS增量 **65,798,144 B**；公共只读输入 **64,003,328 B** 在增量epoch外；前后两代独立compact结果共存。heap累计allocated_bytes=3,134,146,528是cold+预热+重复调用累计流量，不能称单次峰值；cold一次allocated_bytes=391,768,448揭示重复数组分配/复制的量级。RSS的`warm-monitor-control-1`指监控预热，不表示full-size计算已达到稳定缓存状态；heap不等于RSS，Python arenas/mmap不完全覆盖。

原FIRP按组合逐次执行，没有N*A的整块资产L/R。period为8N=16MB，NAV为8P(T+1)约16.08MB；Python右侧替换期间，旧period/NAV与新period或新NAV共存，形成约三份N级矩阵的峰值。源fan每次只复制P元素排序，不长期保留C份全NAV。优化对象是这些约49MB中间生命周期和约392MB/次的分配流量，不是几KB输出头或几ms样本复制。

root报告本轮Eigen投影正式验收C1/C8/Wide分别205/366/65ms，对照NJIT286/657/146ms；完整period结果的内存258.8/256.0MB仍没有显著优势。这是已验收投影工作负载，不能直接代入旧完整链的阶段时间或推算新的总时间。下一轮必须对同一完整compact输出重新配对测量。

## 2. 三个新数学原语足够到哪里

保留当前187个算子合同。建议新ID顺接188–190，最小独立计算语义如下，详见`/private/tmp/calmetrics-mc-compact-contracts-next.md`：

1. `cumulative_product_axis(X, axis, initial)`：float64 rank1..3，标量初态；每条line按axis升序 `acc=initial; acc*=X[i]; out[i]=acc`。IEEE传播，同shape，无金融finite/正值门槛，无reassociation；空输出不读payload。
2. `cumulative_max_axis(X, axis, initial)`：仅`X[i]>acc`更新；相等/signed0保留state，NaN当前值不更新，NaN初态保留。不能用fmax或默认0替代。
3. `quantiles_axis(X, probabilities, axis)`：每条line一次copy和成熟`std::sort`服务全部Q；probabilities为float64 rank1有限[0,1]；该axis替换为Q/component轴。N=1或整数rank直接返回值；其余严格`lo*(1-f)+hi*f`。保留Inf，不改为`lo+(hi-lo)*f`。Q0/outer0空结果scratch0；N0且确有输出则样本不足；概率配置校验独立于输入失败。

**结论：三项足够表达当前平台T>=1、健康固定period/tape的完整NAV、fan、14个统计和贡献，不需要再新增annual moments、MC业务黑盒、通用map/loop框架、reshape或多轴归约。** 但它们不会自动关闭有界物理执行、首故障协议、缺失资产和旧seed链。原始NAV kernel的T0不在当前平台T1..1200合同内；若另外支持该raw空horizon，需要显式初态shape输入或独立shape构造能力，不能借T>=1组合虚称全域已覆盖。

## 3. 必须落地的真实AST组合

以自然布局R[time,path,asset]、period[time,path]为例；若外部period是[path,time]，先借用swap_axes view。axis必须由现有Typed IR可验证的literal声明，金融语义由调用方声明，内部不硬编码NAV字段。新增product的increments无量纲、输出继承initial语义；peak要求initial与values语义兼容；quantile继承值语义，概率无量纲。当前restricted AST不支持列表literal：q5、q_return、q_dd均是调用方绑定的float64 rank1概率向量inputs，依次持有五项(.05,.25,.5,.75,.95)、三项(.05,.5,.95)、一项(.95)。S=min(P,12)由图构建时已绑定的P确定，并将S的整数literal值写入slice参数；示例S是构建占位记号，不是runtime向量/标量输入，P变更时须重新绑定该切片上界。

```text
R = expm1(affine_contract(transformed_tape, K[F,A], intercept[A]))
period_c = sum_axis(broadcast_like(w_c,R)*R, 2)
nav_after_c = cumulative_product_axis(1+period_c, 0, N0)
peak_c = cumulative_max_axis(nav_after_c, 0, N0)
drawdown_c = 1-nav_after_c/peak_c
max_dd_c = max_axis(drawdown_c, 0)                         # [path]
first_time = slice_axis(nav_after_c,0,0,1,1)               # [1,path]
seed = where(finite_mask(first_time),N0,N0)                # [1,path]
nav_full_c = concatenate(seed,nav_after_c,0)               # [T+1,path]
fan_c = swap_axes(quantiles_axis(nav_full_c,q5,1),0,1)     # [5,T+1]
samples_c = swap_axes(slice_axis(nav_full_c,1,0,S,1),0,1)  # [S,T+1]
terminal_nav_c = sum_axis(slice_axis(nav_after_c,0,T-1,T,1),0)
terminal_return_c = terminal_nav_c/N0-1
return_q = quantiles_axis(terminal_return_c,q_return,0)
dd_q95 = quantiles_axis(max_dd_c,q_dd,0)
```

`where(array_mask,N0,N0)`是当前registry/Typed IR真实支持的shape carrier，两分支均直接N0，不存在NaN乘0问题。`broadcast_like`不接受scalar values；`stack(N0,N0,0)`产生component轴，也不能冒充time轴。这修正前报告“健康slice乘0”的不够通用写法。终态singleton `sum_axis`只在验证后的有限正NAV成功域用作降维；这里不能调用只接受rank1的global last。

14个统计按源定义组装：前5项复用fan最后列；return p05/p50/p95；VaR=`maximum(0,-q05)`；ES；loss=`count_true(r<0)/P`；hit=`count_true(r>=target)/P`；avgDD=`sum(max_dd)/P`；p95DD。`length(r)`可给P。avgDD必须原path升序raw sum再/P，不能换global mean的Welford算法。尾部条件包含`r<=q05`的全部ties，不能只取ceil(.05P)项或改为排序后求和。异质金融语义的统计保留named roots，由adapter组装14项，不能强行stack异质单位。

**ES count0的具体安全组合：** 当前`sum_where`全false会抛INSUFFICIENT_SAMPLE，普通DAG的where不是惰性求值，直接`where(count>0,sum_where/count,VaR)`不成立。

```text
mask = less_equal(r,q05)
head = slice_axis(r,0,0,1,1)
zero_one = where(finite_mask(head),0,0)        # 一项path轴数值0
true_one = equal(zero_one,zero_one)           # 一项同轴true
tail_total = sum_where(concatenate(r,zero_one,0),
                       concatenate(mask,true_one,0))
tail_count = count_true(mask)                # 不含sentinel
has_tail = greater_than(tail_count,0)
safe_count = where(has_tail,tail_count,1)     # 当前scalar where真实支持
ES = where(has_tail,maximum(0,-(tail_total/safe_count)),VaR)
```

sentinel保证至少一项被选，不读未选坏值，也不做mask乘数值；分母始终安全。选中元素仍按原path顺序、末尾追加0；zero符号可能在追加0时变化，但不成为公开root，最终source max(0,…)归一为相同零，需用±0/fenv病例核验最终ES。有效MC的terminal可有+Inf（NAV有限但NAV/N0溢出），无NaN，q05同树计算时tail_count本来>=1；count0仍应有明确组合回归，不能借健康域漏掉source fallback。

贡献继续用每asset独立root：`sum(first(slice_axis(w,0,a,a+1,1))*slice_axis(R,2,a,a+1,1))/P`，逻辑flatten T→P→singleton asset，保持原T/P左fold。两次嵌套sum_axis会改变跨time累计顺序。A由已验证模型尺寸确定，最多64，不需要新通用sum_axes算子。

另一个容易遗漏的初态细节：**fan首列不能无条件写N0。** 源quantile对P个相等N0仍在非整数rank执行`N0*(1-f)+N0*f`，任意N0可能有尾差；可以省该列排序，但必须调用同一quantile interpolation数学体。sample首列才直接写N0。

## 4. 真正省几十MB的私有compact闭包

公开DAG继续保留上述可编辑步骤。新增物理proof只识别实际producer→period→scan→fan/summary/sample/contribution消费者链，不注册业务黑盒，也不保留第二套公式。当前投影proof只认period/contribution roots，新增scan/quantile消费者不自动准入；若仅加三个operator然后用普通图执行，N*A的affine/expm1可能各物化512MB，加上多个C*N状态，内存反而会远高于原49MB。

第一版物理执行：

- 同一个有界资产tile生产一次R，复用已验收ordered factor fold和Eigen候选的严格guards；每组合asset左fold的weight-first product同时更新本asset贡献，跨tile不做subtotal归约。
- step升序，path分块；每个组合/path仅保留NAV、peak、maxDD三项state；其每条path的递推顺序与source一致。时序交错改变原P→T全局故障发现顺序，因此诊断另见下一节，不能假装自然等价。
- **一个step必须完成全部P**，再复制当前NAV P元素到同一sort scratch，排序一次服务5个q，写独立fan根及前12 sample根。不能把4096-row tile当整条path分布。自然time/path内部视图与P连续state适合该消费者，不强制输入转C/F或复制整个tape。
- last step的fan已包含5个terminalNAV q，可直接作为同值统计根；末端return和maxDD分别用同一个P scratch。tail sum、DD sum、loss/hit计数仍原path升序，在排序前/独立原state遍历完成，不能按排序顺序累加。
- 不需要保留period[N]、fullNAV[P,T+1]、full asset R[N,A]或独立terminal-P公开结果；外部根仅fan、sample、统计、贡献。已发布根有独立所有者，不能作下一次执行scratch。peak死后可复用为terminal return，但Planner须证明死亡点和sort源alias安全。

预算以combo批次B而非无条件全C：

```text
W = checked(3*B*P*sizeof(double)
            + asset_tile_capacity*sizeof(double)
            + P*sizeof(double)                 # 一份排序copy
            + contribution/metadata/pack/alignment/diagnostic capacity)
R = checked(C*[5(T+1)+S(T+1)+14+A]*sizeof(double))  # 平台P>=100时S=12
```

C8/P10k基础state为1.92MB，较源三N矩阵有几十MB机会，但这不是已测节约。C32/P50k基础state38.4MB，即使T=5源矩阵很小，全C同时驻留也可能更差。按budget选择B，必要时减少batch重算producer；T长且C多时更愿意共享，全路径较大且T短时优先小B。每次执行重新绑定实际P/A/F，禁止缓存旧geometry或跨请求numeric值。不得用强制同时全C来换虚假吞吐。

只有全部相关输出/中间消费被proof识别、没有observer/额外公开R/period/NAV根且ownership/错误策略匹配时准入；otherwise canonical fallback并保留真实大容量预算。普通、prepared、worker必须同一proof预算；计划中N级virtual节点不能仍reserve原缓冲，实际Workspace.capacity、高水位和Audit必须一致。多个统计/贡献root需按真实node ID及消费者关系处理，不能因selected roots省掉已要求的错误/输出。无需新增线程池/多进程框架；先同核1把完整时间和内存闭环，再考虑成熟排序库和跨独立line并行。

## 5. 数学成功域之外的明确缺口

三个原语并不自动重现FIRP错误返回。projection按T→P：先strict minimum更新，再nonfinite status1、<=-1 status2，首坏cell未写且贡献尚未/P；NAV按P→T：先return/min检查，递推后finite/positive status3；terminal除N0没有新增finite gate。融合故障优先级是`(组合定义序, coverage/projection/NAV阶段, 该阶段原遍历)`。同组合晚projection故障优先于更早物理发现的NAV故障；后组合coverage不能抢先于前组合NAV故障。

第一交付可明确只关闭当前冻结healthy固定带数值链；正式API迁移前需要fault board/停止发布/原minimum和坐标协议及对应回归。若要求raw partial arrays一致，额外状态必须按实际协议预算，不把3BP当绝对上界。不能新增泛用reduce/error黑盒掩盖该差异，也不能将各节点IEEE传播改为统一金融约束。

原source不可用资产直接`continue`，不是weight0乘坏收益。adapter需先take可用asset子轴，再恢复A贡献的不可用NaN，扩展物理proof对真实gather/scatter消费的证明；零权重可用资产仍要消费0*Inf并按源故障。旧seed MT19937→polar Box-Muller缓存→Cheng chi-square及normal先全量、再chi顺序，协方差eigh root的basis/sign，属于单独真实调用链工作包。当前三个数学原语和固定带性能不会替代这些验收。年度moments不在当前MC链，不加入本轮必需范围。

## 6. 最小开发顺序与源码落点

| 顺序 | 具体交付 | 验收边界 |
| --- | --- | --- |
| 1 | 冻结3项合同；`operators.def`/`operators.hpp`/registry加准确ID/structure/execute；窄`cpp/operators/axis_primitives.cpp`；Typed IR literal axis/initial/Q推导；compiler、graph容量和Planner workspace接真实known位 | canonical数学、C/F/stride、空/singleton/Inf/±0、独立结构校验、预算恰界、两代结果；先用小完整消费者，不跑大型物化oracle |
| 2 | `tests/test_mc_compact_graph.py`把上述AST编译并执行，核验初态fan、14项、贡献、count0、terminalInf；固定period后再固定tape | 与冻结FIRP数学体同输入对照，确认没有scalar broadcast、eager divide、nested贡献sum等伪代码缺口 |
| 3 | 窄compact metadata/proof/executor复用现有producer和3个canonical数学体；graph接执行/释放；Planner同准入与budget；typed roots/worker用现有协议 | 真实compact消费者闭包；不公开的N级矩阵不分配；实际capacity/Audit、准备缓存和selected/fallback、所有权通过 |
| 4 | 完整链同CPU/同输出重测C1/C8和大P小T、长T、wideF；只在930ms fan确为当前热点时筛选现有成熟`std::nth_element`/Highway selection等多rank方案 | 单line共享unique rank；整数Inf同树；NaN/mixed signed0/未证排序行为回canonical；不能按未测比例承诺加速 |
| 5 | 补缺失资产、原fault board、旧seed/root/transform并接真实FIRP调用和JSON | 分开记录完整数学、接口和业务验收，当前healthy设计不称完整替代 |

跨worker没有理由再造数组协议：既有float64 scalar/rank1..3 typed roots、序列化opcode、readonly snapshot与独立writer即可；检查operator_count、encode/decode边界、ordinary/prepared、native_process身份及同预算，不因新family绕过。CMake只添加实际TU。文档/路由由最终代码差异维护，旧seed等未完成项明确保留。

新增回归宜集中`tests/test_axis_path_primitives.py`和上述MC consumer测试，扩展现有native graph/compiler/Planner targets；涵盖任意axis、C/F/negative/padded、partial known geometry、绑定概率向量及配置独立错误、N0/Q0/outer0、non1 N0、first fan舍入、Inf整数rank、raw path tail ties、ES count0、P<12时S=min(P,12)的样本shape和值、whole graph fallback、public intermediate roots/observer、内存预算不足、失败后prepared复用。结果协议回归含不同shape多root、进程往返及两代不共享。不得把现有投影183/193等已通过结果抵扣新scan/fan闭包。

## 7. Excel并行设计与完整采用门槛

Excel是同一逻辑DAG的另一导出通路，普通C++执行不触发导出、Excel域扫描或工作簿。新增窄`cpp/excel_axis_primitives.cpp`，由`excel_operators.cpp`路由，`excel_internal.hpp`声明，`excel_budget.cpp`给checked保守上界，cases增加3项和小完整消费者。

product：每line独立initial单元格，axis升序每步引用前状态；strict max：明确IF(current>previous,current,previous)，不以MAX忽略NaN替代；quantiles：每line复用一次当前透明rank展开及所有Q的lo/hi/interpolation，整数rank必须显式direct branch，不能EXP/PERCENTILE/SUMPRODUCT式替换改变数学树。初态fan也走同 interpolation。export-only校验所有消费中间值的Excel可表示域，C++有效但含terminalInf/次正规等Excel不能表示时明确拒绝。

prefix符号规划计L个seed、M项更新、guard、输出和全部输入/provenance；quantile从`L*[N*(N+sum_cells(N))+Q*(2N+16)+16]`等保守rank recipe上界起步，再以真实emit逐项反核，包括guard/辅助/公式长度/工作量/工作簿总cell，不能把草图当已证预算。full MC N2m依赖加递推/排序通常远超2m总cell，必须在emit前OVER_BUDGET；不能分sheet规避或自动换算法。目标Microsoft Excel实际重算和数值/预算/域测试完成前不宣称Excel验收。

采用门槛：只比较同一完整输出`fan+14stats+12samples+Acontrib`、同一固定带/排除项、CPU1、编译输入在计时外、fresh probe与fresh RSS子进程。C1/C8 T200/P10k/F4/A32先闭环，追加P50k/T5/F32及T1200/P1666/F4、预算选择B的极端。保留两代独立结果并检查input hash；constructor/retained workspace计入增量，不能预先藏进64MB公共输入。时间和大额内存须均有实测改善，几KB/几ms不作为门禁；若大C state超过参考，减batch再配对，不修改参考合同。

状态：真实收益方向和实现范围已确定；三个原语、完整compact proof、Excel及正式平台迁移尚未由本只读报告实现或验收。
