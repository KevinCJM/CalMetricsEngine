# MetricsFactory → CalMetricsEngine 指标能力审计

> 本文及配套 CSV/JSON 保留补齐前的审计快照。后续新增能力与契约见[详细设计](mathematical-composition-design.md)，实际验证结果见[验收记录](mathematical-composition-acceptance.md)；不将旧分类自动改写为全量指标等价通过。

审计日期：2026-09-21。结论：**当前引擎已有大部分基础数学，但尚不能按 MetricsFactory 的完整契约覆盖全部指标。主要缺口是递推状态、分段/多尺度组织、派生时序的图组合、分组/横截面绑定，以及期权反解。**

本次只分析并保存审计材料，没有修改计算实现、技能或生产服务。

## 范围与证据

- [逐项清单：333 条 CSV](metrics-factory-capability-matrix.csv)：每项包含数学组合、缺口编号、实际源码函数/行号、窗口和契约差异。不同目录的同名指标分别保留，333 不是去重后的唯一名称数量。
- [原生小样本证据 JSON](metrics-factory-capability-evidence.json)：成功组合、失败反例、源码 SHA256、构建身份。
- 已使用 CodeGraph 定位引擎注册表、编译器、Typed IR、执行器及独立 finance API，再读取当前源码核实。
- 引擎是 **118 个 canonical 算子**。另外检查了 `cal_cpr` 等注册表之外的 native finance API；不能仅凭 canonical 名单判定整个引擎能力。
- 检查的原生包为 0.3.0，构建 ID 为 `5d16b88015d839eddaaa3aaddec8e59d931d5c55dc8cb203f5c2209d134bc34b`。按当前 `CMakeLists.txt` 的源文件哈希规则重算后，与当前 C++ 源码及构建配置一致。
- 参考源码：`/Users/chenjunming/Desktop/KevinGit/PyFinance/MetricsFactory`；技能：`/Users/chenjunming/.codex/skills/metrics-factory`。目录索引已针对此源码重新导出到临时目录，未覆盖全局技能。

| 范围 | 当前条目数 | 数学基础可复用/组合候选 | 主要剩余缺口 |
| --- | ---: | ---: | --- |
| 区间标量 | 86 | 81 | 5 项涉及多尺度、分块或跳过缺失后的邻接 |
| 核心时序 | 96 | 71 | 13 项递推契约；12 项派生时序图组合 |
| A 股恐慌度 | 56 | 22 | 4 项 IV 内核；30 项面板/分组/期权链组织 |
| A 股拥挤度 | 69 | 35 | 34 项面板/分组/去重与广播 |
| 配置存在但未接线 | 26 | 不计入可执行覆盖率 | 先明确并接通参考算法 |

这里的“可组合候选”只表示所需数学语义已有基础，**不表示这些指标已在引擎注册、已写好迁移模板，或已通过所有缺失值/窗口/参数的等价验收**。标量指标可按多个区间批量输出，仍不同于一条保持原时间轴的时序根。

技能快照原为 85 个区间指标；当前源码新增 `HurstExponentRS`，实为 86 个，区间映射也增加 `90d/120d/150d`。核心滚动仍为 96 个。26 个未接线项包含 20 个相对历史指标及 `BBI/DMA/DIF/DEA/MACD/ADXSlop-6`，见 CSV；其中两个配置 key 还拼入了说明文字，不能靠名称猜测执行契约。

CSV 分类：`D`=已有归约或字段透传基础；`C`=可用现有数学组合；`S`=递推契约不完整；`B`=缺少分段/选择能力或数值求解内核；`G`=图绑定/组合受限；`U`=参考入口未接线。配方列为说明性伪代码，不能直接当作已发布 DSL。

## 1. 标量指标：哪些无法完整覆盖

### HurstExponent、HurstExponentRS

共同流程是：删除 NaN → 多尺度切分 → 每段统计 → 聚合各段 → 对尺度结果做回归。

- `HurstExponent`：6 个尺度，分段数为 1/2/4/8/16/32；每段长度取整并丢尾。旧实现使用 `range(r-mean(r))/std(r)`，**没有累计均值偏离**。
- `HurstExponentRS`：使用 `range(cumsum(r-mean(r)))/std(r)`；每段至少 8 个观测，过滤无效段，至少 4 个有效尺度。
- `mean/std/cumulative_sum/min/max/log/linear_slope` 均已存在。缺的是 **有效样本选择、非重叠分段视图、多尺度子图执行以及结果归集**（G3），不是缺少标准差或回归。
- 若在外部生成分段、分多次调用原生算子，可以做出部分计算；这不等于现有 AST/DAG 已能执行完整指标。不能把大量 Python 数值循环作为补齐结果。

源码：`period_metrics_cal.py:86`、`:160`。

### CrossProductRatio-1 / -5 / -10

- 指标计算的是 `WW*LL/(WL*LW)`，0 收益归入非负状态。
- 5/10 日版先按原行顺序做 **非重叠** 5/10 行求和、丢弃不足一块的尾部、保留全空块为空，再按每个产品删除空块。
- 1 日版也先跳过 NaN，再统计有效观测之间的转移；不能直接对原数组做相邻比较。
- 缺口是 **跳过缺失的相邻状态访问/转移、块归约及其图组织**（G3）。完整无缺失的 1 日情形可用比较和计数组合，不应称其基础算术缺失。
- 引擎已有 `cal_cpr`，但它按“同类型产品的每日中位数”分类，计算 `same/changed`，与这里的定义不同。

源码：`period_metrics_cal.py:857`、`:893`；引擎 [`cpp/finance/statistics.cpp`](../cpp/finance/statistics.cpp)。

### 其余 81 项

收益、波动、回撤、Sharpe/Sortino、VaR/CVaR、Cornish–Fisher 修正、分位数、Omega、KRatio、净值/成交量回归等所需基础数学已具备。必须按源码构建组合及契约，不能依据同名函数或说明文字直接替换。

例如 KRatio 使用**斜率标准误**；引擎 `regression_standard_error` 返回残差标准差，需再除 `sqrt(Sxx)`。这个修正可组合，属于口径适配，不必新建 KRatio 黑盒。最大回撤的“最后同深谷底”也已有 `last_drawdown_interval` 支持，不是缺失功能。

## 2. 时序指标：重点补齐两类能力

### G1：通用递推滤波的参数、初始化与缺失规则

涉及 13 项：

- `EMA`、`EMADiff`。
- `TRIX`、`MATRIX-3`、`MATRIXDiff-3`、`MATRIX-5`、`MATRIXDiff-5`。
- `KDJ-K-3`、`KDJ-D-3`、`KDJ-J-3`、`KDJ-KD-3`、`KDJ-KJ-3`、`KDJ-DJ-3`。

当前 `recursive_smooth(x,periods,initial)` 有三个限制：

1. `periods` 只接受整数，递推权重为 `1/periods`。EMA 的权重为 `2/(span+1)`，偶数 span 对应半整数 periods，会被拒绝。
2. 遇到非有限值时保留内部状态，但该行**输出 NaN**；参考 EMA 对 NaN 行输出上一次状态。
3. KDJ 第 0 行固定 `K=D=50`，不使用首行 RSV；以后 RSV 无效时 K/D 同时停更。当前直接串联两个 smooth 不满足该规则。

建议新增明确契约的通用递推能力：实数 alpha、固定/首个有效值初始化、是否消费 seed 行、独立 update mask、缺失时输出旧状态或 NaN。已有 `recursive_smooth` 行为应保留。EMA、KDJ、TRIX 在此之上作为可展开组合，不应各自复制一套递推内核。

源码：`rolling_metrics_cal.py:22`、`:176`；引擎 [`sequence.cpp`](../cpp/operators/sequence.cpp)、[`registry.cpp`](../cpp/operators/registry.cpp)。

### G2：派生时序的同轴移位与作用域捕获

明确涉及 12 项：

- `MACR-10-5`、`MACRDiff-10-5`、`MACR-20-9`、`MACRDiff-20-9`、`MACR-40-17`、`MACRDiff-40-17`、`MACR-62-28`、`MACRDiff-62-28`。
- `ADXR-6-6`、`ADXRDiff-6-6`、`ADXR-6-14`、`ADXRDiff-6-14`。

问题不是缺少减法或均线：

- 现有 `lag(x,k)` 返回前 `T-k` 个值的借用视图；`difference` 同样缩短长度。它们不是在前面补缺失的 `shift`。
- 公共时序根要求长度为 T，`x-lag(x)` 会在类型阶段因形状不同被拒绝。
- 原始输入的 shift/diff 已能用 `rolling_apply` 组合，见下节，因此不应把全部 MTM/RSI/OBV 都列为数学不可算。
- 但派生的 `rolling_mean(CR)`、ADX 或 EMA 等结果需要保持全历史状态后再移位。当前 rolling body 不能捕获任意已计算表达式，且禁止嵌套 rolling / recursive_smooth；按通常组合方式放在一个 DAG 中会受阻。
- 多阶段执行可绕过部分限制，但需要显式阶段/所有权/计划支持，不能据此声称单个当前图已经覆盖。

建议增加**同轴 shift/差分契约**，并按需要补充派生值捕获或原生阶段连接；正确计入 liveness、窗口重置与计算预算。TRIX 系列也受此能力影响，表中为避免重复计数归入 G1。

源码：[`compiler.cpp`](../cpp/compiler.cpp)、[`registry.cpp`](../cpp/operators/registry.cpp)、[`graph.cpp`](../cpp/graph.cpp)。

## 3. 恐慌度、拥挤度的额外缺口

### G4：横截面/分组与广播进入原生 DAG

受影响的完整名称在 CSV 的 fear/crowding `G` 行；包括市场宽度、涨跌停聚合、市场/行业成交份额与 HHI、分组 Top-N、持仓集中度、市场两融汇总、期货席位集中度、PCR 等。

- 引擎已有 `sum_asset/mean_asset/std_asset`、矩阵算子和直接算子接口。
- **当前 interval GraphCompiler 只绑定 scalar/time-series，不能绑定 asset vector 或 time×asset matrix**。Typed IR 可以描述矩阵，不等于执行入口已支持。
- 通用分组 offsets/类别、组内归约、归约结果广播回明细，以及时序↔横截面组合尚未形成完整原生图契约。
- 已按日/组准备好的单组数组，可以交给现有 C++ 归约。缺口在完整批量图，不应误写成“引擎不会算市场均值/HHI”。

需要完善矩阵/分组绑定与执行，而非新增“恐慌度算子”“行业拥挤度算子”。字符串解析、单位转换、日期/合约关联和 point-in-time 成员映射仍由上游输入边界负责；数值分组归约应进入 native 图。

### G5：离散去重、排序索引与多字段选择

- `stock_fund_holder_count` 需要按组计算基金 ID 去重数量；不能把记录行数当持有基金数量。需要稳定整数类别与 `distinct_count/unique` 一类语义。
- `option_strike_oi_hhi` 先按行权价聚合，再计算 HHI；缺的是分组组合链，不是平方求和。
- VIX-like 需要配对 call/put 行权价、排序、选择 `K0`、同步选择对应报价、构造相邻 `ΔK`，再跨期限选择/插值。当前没有通用的排序索引与 gather 图契约。
- Top-K **标量和**可以使用分位阈值、严格大于计数和并列补足组合，不是必须新增 sort/top_k；不过返回被选中元素的多字段记录是另一项能力。

### G6：期权正态 CDF 与有界求根

明确涉及 `option_atm_iv`、`option_put_wing_iv_spread`、`option_call_wing_iv_spread`、`option_25d_put_call_skew`。

- 已有 `normal_pdf/normal_ppf`，没有 `normal_cdf/erf`。
- 已有加减乘除、log、exp、sqrt，可组合 Black–Scholes 价格与 Delta；当前没有原生通用有界标量求根/IV 迭代节点。
- 需定义求根边界、容差/迭代上限、不可解与不收敛状态、逐合约错误隔离。参考脚本使用 `[1e-4,5]` 上的 80 轮二分，并没有完整收敛验收；迁移时应先确定保留口径还是发布新的数学契约。
- `option_vix_like_30d`、`option_vix_term_slope` 的数学公式本身不需要 IV。但**当前技能脚本先筛选反解 IV 成功的行再计算 VIX**，所以复现现有输出链也依赖 G6。不能在不说明算法变化的情况下移除这道筛选。

源码：技能 `scripts/run_a_share_fear_indicators.py:638`、`:666`、`:712`、`:767`、`:800`。

## 4. 这些不应列为必须新增的数学算子

以下是实际通过原生小样本验证的组合方向，仍需正式模板与全边界验证：

| 需求 | 现有组合 | 限制 |
| --- | --- | --- |
| 滚动求和 | `rolling_apply(sum_where(x,finite_mask(x)),w,1)` | O(T×W) 通用窗口成本；不是已优化的 O(T) rolling_sum |
| 原始序列同轴差分 | `rolling_apply(sum(difference(x,k)),k+1,1)` | 原始输入可用；不能推广为任意派生状态移位 |
| 原始序列同轴 shift | `rolling_apply(mean(lag(x,k)),k+1,1)` | 同上；早期不足窗口产生 NaN |
| 历史百分位排名 | 窗口最后有效值 + 比较 + count / 有效数 | 最后有效值可由位置序列、max、value_at 得到；保留相等边界 |
| 当前连续净流出天数 | `i-cumulative_max(where(x<0,0,i))`，i从1开始 | i用对全行的cumulative_sum生成；缺失重置；不是区间最长连续长度 |
| DKX 加权均值 | 窗口内生成1…20权重 × B，求和/210 | 源端固定20；缺失贡献和前19行需要专门适配 |
| Top-K 和 | 阈值上方求和 + 剩余名额×阈值 | 必须处理并列、K=1/K≥有效数和空组；不能简单sum(x≥分位数) |
| 修正 VaR/CVaR | mean/std/skewness/kurtosis/pdf/ppf + 多项式 | 冻结源码算法，设精度容限 |
| KRatio | fit + SSE + Sxx + sqrt/divide | 原生残差标准差不等于斜率标准误 |
| Omega / 收益分布积分 | 条件正负偏离求和 | 源码中的排序对等权求和可消去，无需积分内核 |

## 5. 横跨大量指标的契约差异（K1、K2）

**K1：数值和窗口。**这些是迁移必做项，但不能一概归为缺少数学能力。

| 易误替换点 | 源码实际口径 |
| --- | --- |
| TotalReturn / AnnualizedReturn | 默认累计 log return；收益年化按自然日365，风险年化252 |
| AvgPositiveReturn / AvgNegativeReturn | 非对应方向和NaN先填0，分母保留全部行 |
| period DownsideVolatility | 低于MAR的差值，其余填0后算样本方差；不同于恐慌指标的负子集std |
| PriceSigma | 首行即部分窗口；只有1个有效值时std=0 |
| CloseMA / VolMA | 前N−1行NaN；之后允许窗口只有1个有效样本 |
| CCI | 第二次均线处理的是逐时点 `abs(TP_t-MA_t)`，不是另一种窗口MAD定义 |
| DKX | 默认20行权重，即使外层rolling_days是其他值 |
| 回撤 | 最后同深谷底；无回撤为0天；缺失dd填0计入RMS分母 |
| NaN / Inf | 很多源码使用isnan/nan*，引擎finite_mask会同时剔除Inf；二者不能混同 |
| 安全除法 | 当前divide_or_default使用 `abs(den)<1e-12` 且只支持数组；源端常用den==0或1e-6/1e-10，各不相同 |

另外，`where` 是已经计算完子节点之后的选择，不是短路控制流。`where(count>0,mean_where(empty),0)` 仍会先抛空样本错误。可用安全分母、有效掩码和无异常组合解决的应优先组合；需要空集默认值或有效性结果的地方，应增加显式契约，不能靠外层where吞错。隔离模式的错误状态还会向下游传播，不能自动把错误变成业务上的0。

**K2：时间轴与输入边界。**包括首行、间断日期、字段独立缺失、收益/NAV长度、固定日窗与自然月窗、Pandas `pct_change` 的前填充，以及输出日期是否保留。当前 `rolling_apply` 的 min_periods 是被捕获原始输入的共同有限行数，不总等于指标内部条件子集计数。

参考核心区间构造函数还会分别删掉各字段的全空行；直接 rolling 入口仍存在 open/close 位置参数互换，技能 runner 用关键字绕过。本审计采用计算类与安全 runner 的字段含义，不把源端参数错误当成应迁移的数学定义。以上源端问题本次未修改。

## 6. 建议补齐顺序

1. **先冻结迁移契约和小样本**：log/简单收益、窗口行数/有效数、NaN/Inf、零除、误差容限、逐指标参数和输出形状。复用已有 AOT 门禁，不伪造 NJIT 证明。
2. **G1 + G2**：通用递推滤波及同轴 shift/派生值连接，覆盖核心时序的主要阻塞。
3. **G3**：有效选择视图、非重叠块归约、分段/多尺度子图，Hurst 和 CPR 保持为组合模板。
4. **G4 + G5**：矩阵/分组绑定、广播、稳定 ID 去重、排序索引和 gather，补齐恐慌/拥挤的面板链路。
5. **G6**：normal CDF 和有界求根，再组合 BS/IV、Skew、VIX-like。仅求根/递推等不可分离状态保留耦合内核。

新能力还需贯通 registry、Typed IR、lowering、执行器、Planner成本/内存、进程序列化、状态和测试；只加一个 C++ 函数不足以证明整条链路可用。rolling_sum、WMA、rank、top_k 等专用快路径应由基准决定，不能只因指标很多就扩大公共算子表。

## 验证边界

本次完成全量目录/分派核对、计算源码阅读和 41 项原生能力探针，包含成功组合、被拒绝的表达式及不等价反例，不是“41 项指标等价测试全部通过”。参考运行环境检查通过；Numba 缓存定向临时目录，未安装依赖或写正式数据。探针的表达式、输入、误差容限及结果见配套 JSON。

没有运行真实投研数据作业、全量333条数值等价回归或性能验收；没有把“可组合候选”当成可上线清单。分析也不认证数据可得性、复权时点或投资有效性。
