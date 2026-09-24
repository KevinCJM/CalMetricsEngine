# 数学算子参考

[文档导航](README.md) · [使用手册](user-guide.md) · [直接调用契约](canonical-operators.md)

本页覆盖当前 `canonical-native-1` 的 **146 个算子**，不是历史 opcode 的外部映射表。名称、opcode 和重载参数经当前原生 `operators.catalog()` 与 [operators.def](../cpp/include/calmetrics_engine/operators.def) 核对；计算逻辑按 C++ 内核整理。运行包可能与源码不同，使用者应查询所安装包的目录。

## 阅读规则

- S：float64 标量；L：一维 float64 数组；M：二维 float64；T3：三维 float64；A：L/M/T3；F：S/L/M/T3；B：bool 或合法 uint8 0/1 mask；I：int64 数组。n/T/N 是长度符号，不是自动推断金融含义。
- 下表描述直接算子的物理输入输出；进入图后还必须满足名义轴、形状、语义和公开根约束。L 在图中可能是 time series 或 asset vector，不能任意互换。
- 签名中的方括号是可选尾部参数，不是要输入的 Python 语法。回归和协方差等真正重载逐一列出。直接接口还接受 out、workspace、simd、audit 等公共关键字；图公式不能把这些运行参数当数学参数。
- 图的向量、矩阵、张量及异形多输出使用 `outputs[根].values[区间]`，保持各算子的 dtype；内部记录仍须字段投影。结果接口不改变以下数学口径。
- 大多数结果数值是 float64。直接掩码结果默认 uint8；可提供兼容 bool out。图的 bool 时序输出为 NumPy bool。int64 不能任意传给浮点算子；equal/not_equal、排序、选择和状态能力只开放明确的整数重载。
- “SIMD 可选”只表示存在某个重载的显式向量化路径；实际 ISA、布局、长度及重载仍决定是否使用。空白/“—”不表示 Python 回退。

## 共同数学边界

1. 普通归约不自动忽略 NaN/Inf。sum/mean/variance 等可能被非有限值污染；min/max 使用顺序比较，NaN 行为依位置；中位数/分位数排序将 NaN 放末尾，但不删除它。不要把所有函数等同于 NumPy nan*。
2. minimum/maximum 采用 std::min/max：左输入为 NaN 时保留 NaN，右输入为 NaN 且左输入有效时保留左值；相等保留左值。累计极值同理保留顺序。
3. 比较遵循原生浮点规则：涉及 NaN 的相等/大小比较为 false，not_equal 为 true；sign(NaN)=0。普通缺失并不会自动携带执行故障；需要时另建 finite_mask。isolate 捕获的实际故障则沿依赖传播，不能用比较消除。
4. std/variance 使用 Welford；rolling_std 使用滚动和/平方和，不承诺与不同数值算法逐位一致。滚动族明确按有限样本计数，与普通归约的缺失策略不同；min_periods 不是“窗口内至少这么多连续日期”。
5. 明确异常、普通 NaN、状态未知码和有效 0/False 分开。图的错误隔离与 statuses 见[使用手册](user-guide.md#8-错误缺失和结果有效性)。直接 out 发生异常后可能已部分写入。
6. 原始数组不修改；排序、求解允许必要 scratch 复制。lag/transpose/矩阵diag及状态列投影默认借用视图；保留独立结果的规则见[内存契约](canonical-operators.md)。
7. 下面写出的公式用于解释数学逻辑，实际运算顺序、阈值和边界以链接的当前内核为依据；代数等价不保证可无验收替换浮点实现。

## 逐元素、比较与掩码

内核：[源文件](../cpp/operators/elementwise.cpp)；形状/参数校验：[registry.cpp](../cpp/operators/registry.cpp)。

| ID / 调用签名 | 输入 → 输出 | 数学逻辑与边界 | SIMD |
| --- | --- | --- | --- |
| 1 · `add(lhs, rhs)` | F,F → F | 逐元素 x+y。 | 可选 |
| 2 · `subtract(lhs, rhs)` | F,F → F | 逐元素 x−y。 | 可选 |
| 3 · `multiply(lhs, rhs)` | F,F → F | 逐元素 x*y。 | 可选 |
| 4 · `divide(numerator, denominator)` | F,F → F | 逐元素 x/y；任一被执行分母为 0 抛错。 | 可选 |
| 5 · `power(base, exponent)` | F,F → F | 逐元素 x^y；结果非有限抛 DOMAIN_ERROR；图中指数须无量纲标量。 | — |
| 6 · `minimum(lhs, rhs)` | F,F → F | 逐元素 std::min(x,y)；比较顺序影响 NaN，见共同边界。 | 可选 |
| 7 · `maximum(lhs, rhs)` | F,F → F | 逐元素 std::max(x,y)；比较顺序影响 NaN，见共同边界。 | 可选 |
| 8 · `negate(values)` | F → F | 逐元素 −x。 | 可选 |
| 9 · `absolute(values)` | F → F | 逐元素 abs(x)。 | 可选 |
| 10 · `sqrt(values)` | F → F | 逐元素平方根；负输入抛错。 | 可选 |
| 11 · `clip(values, lower, upper)` | F,S,S → F | min(max(x,lower),upper)；上下限为标量，lower>upper 抛错。 | — |
| 12 · `log(values)` | F → F | 自然对数 ln(x)；x<=0 抛错；图中另校验量纲。 | — |
| 13 · `exp(values)` | F → F | e^x；结果非有限抛错。 | — |
| 14 · `reciprocal(values)` | F → F | 1/x；x=0 抛错。 | 可选 |
| 15 · `sign(values)` | F → F | 正值 1，负值 −1，其余 0；当前 NaN 也产生 0，不代表数据有效。 | — |
| 16 · `equal(lhs, rhs)` | F,F → B | x==y，使用原生比较；NaN 不自动变成未知状态。另支持精确 int64 相等性比较；标量配置限安全整数范围。 | 可选 |
| 17 · `not_equal(lhs, rhs)` | F,F → B | x!=y，使用原生比较；NaN 不自动变成未知状态。另支持精确 int64 相等性比较；标量配置限安全整数范围。 | 可选 |
| 18 · `less_than(lhs, rhs)` | F,F → B | x<y，使用原生比较；NaN 不自动变成未知状态。 | 可选 |
| 19 · `less_equal(lhs, rhs)` | F,F → B | x<=y，使用原生比较；NaN 不自动变成未知状态。 | 可选 |
| 20 · `greater_than(lhs, rhs)` | F,F → B | x>y，使用原生比较；NaN 不自动变成未知状态。 | 可选 |
| 21 · `greater_equal(lhs, rhs)` | F,F → B | x>=y，使用原生比较；NaN 不自动变成未知状态。 | 可选 |
| 22 · `logical_and(lhs, rhs)` | B,B → B | 同形掩码逐元素 AND，不做不同形状的标量广播。 | — |
| 23 · `logical_or(lhs, rhs)` | B,B → B | 同形掩码逐元素 OR，不做不同形状的标量广播。 | — |
| 24 · `logical_not(values)` | B → B | 逐元素 NOT。 | — |
| 25 · `where(mask, if_true, if_false)` | B,F,F → F | 按 mask 选择数值分支，允许显式标量广播；图中不是短路求值，依赖故障保守传播。 | — |
| 55 · `divide_or_default(numerator, denominator, default)` | L,L,S → L | 同长序列相除；输入任一非有限输出 NaN，否则 abs(y)<1e−12 用有限 default，其余 x/y。 | — |
| 72 · `normal_pdf(values)` | F → F | 标准正态密度 exp(−x²/2)/sqrt(2π)。 | — |
| 73 · `normal_ppf(probability)` | F → F | 标准正态分位点的分段有理近似；要求概率在 (0,1)，NaN按实现传播。 | — |
| 118 · `finite_mask(values)` | F → B[T] | 逐位置 isfinite；NaN 和正负 Inf 为 false，其他有限值为 true。 | 可选 |
| 119 · `normal_cdf(values)` | F → F | 0.5*erfc(−x/sqrt(2))；−Inf→0，+Inf→1，NaN传播。 | — |
| 125 · `floor(values)` | F → F | 逐元素向下取整，仍返回float64；NaN/Inf传播，不是类别强转。 | — |
| 126 · `cos(values)` | F → F | 逐元素余弦，弧度输入；图中要求无量纲。 | — |

## 归约与统计

内核：[源文件](../cpp/operators/reduction.cpp)；形状/参数校验：[registry.cpp](../cpp/operators/registry.cpp)。

| ID / 调用签名 | 输入 → 输出 | 数学逻辑与边界 | SIMD |
| --- | --- | --- | --- |
| 26 · `sum_where(values, mask)` | A,同形 B → S | 仅读取 mask=true 位置，计算求和；空选择抛错，不等于 filter_apply 的空集默认策略。 | — |
| 27 · `mean_where(values, mask)` | A,同形 B → S | 仅读取 mask=true 位置，计算均值；空选择抛错，不等于 filter_apply 的空集默认策略。 | — |
| 28 · `variance_where(values, mask)` | A,同形 B → S | 仅读取 mask=true 位置，计算样本方差，ddof 固定为 1；空选择抛错，不等于 filter_apply 的空集默认策略。 | — |
| 29 · `std_where(values, mask)` | A,同形 B → S | 仅读取 mask=true 位置，计算样本标准差，ddof 固定为 1；空选择抛错，不等于 filter_apply 的空集默认策略。 | — |
| 30 · `min_where(values, mask)` | A,同形 B → S | 仅读取 mask=true 位置，计算最小值；空选择抛错，不等于 filter_apply 的空集默认策略。 | — |
| 31 · `max_where(values, mask)` | A,同形 B → S | 仅读取 mask=true 位置，计算最大值；空选择抛错，不等于 filter_apply 的空集默认策略。 | — |
| 32 · `median_where(values, mask)` | A,同形 B → S | 仅读取 mask=true 位置，计算中位数；空选择抛错，不等于 filter_apply 的空集默认策略。 | — |
| 33 · `quantile_where(values, mask, probability)` | A,同形 B,S → S | 仅读取 mask=true 位置，计算线性插值分位数，0<p<1；空选择抛错，不等于 filter_apply 的空集默认策略。 | — |
| 34 · `count_true(mask)` | 一/二维 B → S | 统计 true 数量；空掩码为 0。 | — |
| 35 · `max_consecutive_true(values)` | 一维 B → S | 最长连续 true 长度；false 断开，空输入为 0。 | — |
| 36 · `sum(values)` | A → S | 按逻辑行序求和，空输入抛错。 | — |
| 37 · `product(values)` | A → S | 按逻辑行序连乘，空输入抛错。 | — |
| 38 · `mean(values)` | A → S | 顺序增量均值；不自动剔除非有限值。 | — |
| 39 · `min_value(values)` | A → S | 遍历求最小值，空输入抛错；同值不更换首次位置。 | — |
| 40 · `max_value(values)` | A → S | 遍历求最大值，空输入抛错；同值不更换首次位置。 | — |
| 41 · `variance(values[, ddof])` | A[,S] → S | Welford 中心平方和/(n−ddof)；默认 ddof=1，要求整数 0<=ddof<n。 | — |
| 42 · `std(values[, ddof])` | A[,S] → S | 上述方差的平方根，默认 ddof=1。 | — |
| 60 · `median(values)` | A → S | 排序后中间值；偶数样本取中间两项算术平均；算法 scratch 复制，不改输入。 | — |
| 61 · `skewness(values)` | A → S | 校正样本偏度 sqrt(n*(n−1))/(n−2) * m3/m2^(3/2)，mk 为中心 k 阶均值；n>=3 且非零方差。 | — |
| 62 · `excess_kurtosis(values)` | A → S | (n−1)/((n−2)*(n−3))*((n+1)*(m4/m2²−3)+6)；n>=4 且非零方差。 | — |
| 63 · `mean_absolute_deviation(values)` | A → S | mean(abs(x−mean(x)))，中心是均值，不是中位数。 | — |
| 64 · `root_mean_square(values)` | A → S | sqrt(mean(x²))。 | — |
| 65 · `argmin(values)` | A → S | 逻辑行优先展平后的首个最小值位置，零基 float64 标量；不是 int64 数组。 | — |
| 66 · `argmax(values)` | A → S | 逻辑行优先展平后的首个最大值位置，零基 float64 标量。 | — |
| 67 · `quantile(values, probability)` | A,S → S | 排序后在 (n−1)*p 的相邻位置做线性插值；严格 0<p<1，不自动删除 NaN。 | — |
| 77 · `sum_time(values)` | M[T,N] → L[N] | 沿 axis 0 对每列计算求和；图中输出轴为 asset。 | — |
| 78 · `mean_time(values)` | M[T,N] → L[N] | 沿 axis 0 对每列计算均值；图中输出轴为 asset。 | — |
| 79 · `product_time(values)` | M[T,N] → L[N] | 沿 axis 0 对每列计算连乘；图中输出轴为 asset。 | — |
| 80 · `variance_time(values)` | M[T,N] → L[N] | 沿 axis 0 对每列计算样本方差，ddof=1；图中输出轴为 asset。 | — |
| 81 · `std_time(values)` | M[T,N] → L[N] | 沿 axis 0 对每列计算样本标准差，ddof=1；图中输出轴为 asset。 | — |
| 82 · `min_time(values)` | M[T,N] → L[N] | 沿 axis 0 对每列计算最小值；图中输出轴为 asset。 | — |
| 83 · `max_time(values)` | M[T,N] → L[N] | 沿 axis 0 对每列计算最大值；图中输出轴为 asset。 | — |
| 84 · `sum_asset(values)` | M[T,N] → L[T] | 沿 axis 1 对每行计算求和；图中输出轴为 time。 | — |
| 85 · `mean_asset(values)` | M[T,N] → L[T] | 沿 axis 1 对每行计算均值；图中输出轴为 time。 | — |
| 86 · `product_asset(values)` | M[T,N] → L[T] | 沿 axis 1 对每行计算连乘；图中输出轴为 time。 | — |
| 87 · `variance_asset(values)` | M[T,N] → L[T] | 沿 axis 1 对每行计算样本方差，ddof=1；图中输出轴为 time。 | — |
| 88 · `std_asset(values)` | M[T,N] → L[T] | 沿 axis 1 对每行计算样本标准差，ddof=1；图中输出轴为 time。 | — |
| 89 · `min_asset(values)` | M[T,N] → L[T] | 沿 axis 1 对每行计算最小值；图中输出轴为 time。 | — |
| 90 · `max_asset(values)` | M[T,N] → L[T] | 沿 axis 1 对每行计算最大值；图中输出轴为 time。 | — |
| 124 · `distinct_count(identifiers[, mask])` | I[n][,B[n]] → S | 选中整数的不同值数量；缺省全选，负类别不自动当缺失；空选择0；计数必须可由float64精确表示。 | — |

## 滚动窗口

内核：[源文件](../cpp/operators/sequence.cpp)；形状/参数校验：[registry.cpp](../cpp/operators/registry.cpp)。

| ID / 调用签名 | 输入 → 输出 | 数学逻辑与边界 | SIMD |
| --- | --- | --- | --- |
| 50 · `rolling_mean(values, window[, min_periods])` | L,窗口及可选标量 → L | 尾随窗口均值，只计有限值且保留原行位置；min_periods 默认 window。未达到有效数要求输出 NaN。 | — |
| 51 · `rolling_std(values, window[, ddof, min_periods])` | L,窗口及可选标量 → L | 尾随窗口标准差，只计有限值且保留原行位置；min_periods 默认 window。ddof 默认 0，有限数须 >ddof；使用滚动和/平方和。 | — |
| 52 · `rolling_min(values, window[, min_periods])` | L,窗口及可选标量 → L | 尾随窗口最小值，只计有限值且保留原行位置；min_periods 默认 window。未达到有效数要求输出 NaN。 | — |
| 53 · `rolling_max(values, window[, min_periods])` | L,窗口及可选标量 → L | 尾随窗口最大值，只计有限值且保留原行位置；min_periods 默认 window。未达到有效数要求输出 NaN。 | — |
| 54 · `recursive_smooth(values, periods, initial)` | L,S,S → L | s=((periods−1)*s+x)/periods；正整数周期、有限初态；非有限行输出 NaN并保留状态。 | — |

## 序列、递推与索引

内核：[源文件](../cpp/operators/sequence.cpp)；形状/参数校验：[registry.cpp](../cpp/operators/registry.cpp)。 自适应/二阶/Kalman 递推另见 [recurrence.cpp](../cpp/operators/recurrence.cpp)。

| ID / 调用签名 | 输入 → 输出 | 数学逻辑与边界 | SIMD |
| --- | --- | --- | --- |
| 43 · `cumulative_sum(values)` | L → L | 前缀累计和，不跨区间共享累积状态。 | — |
| 44 · `cumulative_product(values)` | L → L | 前缀累计乘积，保持输入顺序。 | — |
| 46 · `cumulative_max(values)` | L → L | 逐前缀最大值，使用 std::max 的顺序比较语义。 | — |
| 47 · `cumulative_min(values)` | L → L | 逐前缀最小值，使用 std::min 的顺序比较语义。 | — |
| 48 · `drawdown_series(levels)` | L → L | x[t]/max(x[0:t+1])−1；要求每点有限且严格正，首点 0。 | — |
| 49 · `new_high_mask(levels)` | L → B[T] | 首点 true；之后仅严格高于此前最大值为 true；要求有限正值。 | — |
| 56 · `first(values)` | L → S | 首元素；空输入抛错。 | — |
| 57 · `length(values)` | L → S | 序列元素数，以 float64 返回；空序列为 0。 | — |
| 58 · `lag(values[, periods])` | L[,S] → L[n−p] 借用 | 返回 x[:n−p]，不是等长填充；p 默认 1，0<=p<n。 | — |
| 59 · `difference(values[, periods])` | L[,S] → L[n−p] | 返回 x[p:]-x[:-p]；p 默认 1，整数 0<p<n。 | — |
| 74 · `last(values)` | L → S | 末元素；空输入抛错。 | — |
| 120 · `aligned_shift(values[, periods, fill])` | L[,S,S] → L | y[t]=fill（t<p）或 x[t−p]；p默认1，fill默认NaN；p>=n合法，输出全填充。 | — |
| 121 · `recursive_filter(values, alpha, initial, update_mask[, seed_mode, emit_policy])` | L,S,S,B[,S,S] → L | s=(1−alpha)*s+alpha*x；有限且 mask=true 才更新，alpha∈[0,1]；seed/emit 默认0/0；细则见递推策略。 | — |
| 122 · `argsort(values)` | L 或 I[n] → I[n] | 稳定升序索引；相等按原位置，浮点 NaN 最后；int64 全范围比较保持精度。 | — |
| 123 · `gather(values, indices)` | L/B/I, I[k] → 同 dtype 长度 k | 按零基 int64 索引读取，允许重复/空选择；负数或越界抛错；必要输出分配，不是视图。 | — |
| 127 · `recursive_filter_adaptive(values, alpha, update_mask, reset_mask[, initial, seed_mode, min_periods, emit_policy])` | L,标量或L alpha,B,B[,S,S,S,S] → L | reset先清空状态/计数；更新 s+=alpha*(x−s)，仅消费的alpha须在[0,1]；初态0、seed1、预热1、emit1；见递推策略。 | — |
| 128 · `linear_filter2(values, b0, b1, a1, a2, reset_mask[, min_periods, bootstrap])` | L,S,S,S,S,B[,S,S] → L | y=b0*x+b1*前一有效输入+a1*前输出+a2*更早输出；默认前两笔有限值作种子；非有限输出重置；不是通用双向IIR。 | — |
| 129 · `scalar_kalman(values, process_variance, measurement_variance[, initial_variance])` | L,S,S[,S] → M[T,2]名义状态 | 一维随机游走：P-=P+Q，K=P-/(P-+R)，xhat+=K*(x−xhat)，P=(1−K)*P-；首个有限值直接初始化；Q>=0,R>0,P0>=0。 | — |

## 矩阵运算

内核：[源文件](../cpp/operators/matrix.cpp)；形状/参数校验：[registry.cpp](../cpp/operators/registry.cpp)。

| ID / 调用签名 | 输入 → 输出 | 数学逻辑与边界 | SIMD |
| --- | --- | --- | --- |
| 91 · `transpose(values)` | M[m,n] → M[n,m] 借用 | 交换两个轴和 strides；默认返回只读视图。 | — |
| 92 · `dot(lhs, rhs)` | L[n],L[n] → S | Σ x[i]*y[i]，两个一维等长输入。 | — |
| 93 · `outer(lhs, rhs)` | L[m],L[n] → M[m,n] | 结果 C[i,j]=x[i]*y[j]。 | — |
| 94 · `matmul(lhs_matrix, rhs_matrix)` | M[m,k],M[k,n] → M[m,n] | C[i,j]=Σ A[i,k]*B[k,j]，检查内维；无批量高维广播。 | 可选 |
| 95 · `matvec(matrix, vector)` | M[m,n],L[n] → L[m] | 每行与向量点积。 | — |
| 96 · `diag(values)` | L[n] → M[n,n]；M → L 借用 | 向量生成对角矩阵（非对角为 0）；矩阵读取 min(rows,cols) 个对角值的视图。 | — |
| 97 · `trace(values)` | M → S | 求主对角线之和，使用 min(rows,cols)，不强制方阵。 | — |
| 98 · `solve(matrix, rhs)` | M[n,n],L[n] → L[n] | Ax=b，部分主元高斯消元/回代；绝对主元<=1e−14 判奇异；复制系数与 RHS 到可修改 scratch。 | — |
| 99 · `covariance(asset_returns)` / `covariance(lhs,rhs)` | M[T,N] → M[N,N]；L,L → S | 样本协方差 ddof=1；单矩阵沿 axis 0，双序列等长，至少 2 个样本。 | — |
| 100 · `correlation(asset_returns)` / `correlation(lhs,rhs)` | M[T,N] → M[N,N]；L,L → S | 样本协方差除以两边样本标准差；零标准差抛错。 | — |

## 回归与共享拟合

内核：[源文件](../cpp/operators/state.cpp)；形状/参数校验：[registry.cpp](../cpp/operators/registry.cpp)。

| ID / 调用签名 | 输入 → 输出 | 数学逻辑与边界 | SIMD |
| --- | --- | --- | --- |
| 68 · `linear_slope(values)` / `linear_slope(x,y)` | L 或 L,L → S | 带截距 OLS 的斜率 Sxy/Sxx；单输入隐式 x=0..n−1。 | — |
| 69 · `linear_intercept(values)` / `linear_intercept(x,y)` | L 或 L,L → S | OLS 截距 mean(y)−slope*mean(x)。 | — |
| 70 · `linear_r_squared(values)` / `linear_r_squared(x,y)` | L 或 L,L → S | 1−RSS/TSS；TSS 必须严格正。 | — |
| 71 · `regression_standard_error(values)` / `regression_standard_error(x,y)` | L 或 L,L → S | sqrt(RSS/(n−2))，要求 n>2；不是斜率标准误。 | — |
| 112 · `linear_fit(values)` / `linear_fit(x,y)` | L 或 L,L → 拟合记录 | 一次 OLS 得 slope/intercept/RSS/TSS/n；有限输入、n>=2、Sxx>0；多个投影可共享一次计算。 | — |

## 投影、状态与事件

内核：[源文件](../cpp/operators/state.cpp)；形状/参数校验：[registry.cpp](../cpp/operators/registry.cpp)。 状态机和事件内核另见 [state_events.cpp](../cpp/operators/state_events.cpp)、[完整状态契约](state-event-contracts.md)。

| ID / 调用签名 | 输入 → 输出 | 数学逻辑与边界 | SIMD |
| --- | --- | --- | --- |
| 104 · `last_drawdown_interval(drawdowns)` | L → 区间记录 | 输入首项须 0、各项有限且在 [−1,0]；取最深回撤，同深选最后谷底。记录 start/trough/recovery/status；无效 −1，无事件 0，有事件 1；未恢复 recovery=NaN。 | — |
| 105 · `interval_start(interval)` | 区间记录 → S | 投影 start 零基位置，不重新扫描；缺失位置保留 NaN。 | — |
| 106 · `interval_trough(interval)` | 区间记录 → S | 投影 trough 零基位置，不重新扫描；缺失位置保留 NaN。 | — |
| 107 · `interval_recovery(interval)` | 区间记录 → S | 投影 recovery 零基位置，不重新扫描；缺失位置保留 NaN。 | — |
| 108 · `value_at(values, position)` | L,S → S | 零基位置取值；位置非有限/非整数/负/越界，或所取值非有限，输出 NaN。 | — |
| 109 · `days_between(start_date, end_date)` | S,S → S | end−start，自然日整数编号；非有限/非整数/倒序输出 NaN；不转换纳秒时间戳。 | — |
| 110 · `require_positive(values)` | S → S | 要求有限且>0，否则抛 DOMAIN_ERROR；返回原值。 | — |
| 111 · `require_nonnegative(values)` | S → S | 要求有限且>=0，否则抛 DOMAIN_ERROR；返回原值。 | — |
| 113 · `fit_slope(fit)` | 拟合记录 → S | 读取 slope 字段，不重新拟合。 | — |
| 114 · `fit_intercept(fit)` | 拟合记录 → S | 读取 intercept 字段，不重新拟合。 | — |
| 115 · `fit_residual_sum_squares(fit)` | 拟合记录 → S | 读取 RSS 字段，不重新拟合。 | — |
| 116 · `fit_total_sum_squares(fit)` | 拟合记录 → S | 读取 TSS 字段，不重新拟合。 | — |
| 117 · `fit_observation_count(fit)` | 拟合记录 → S | 读取 n 字段，不重新拟合。 | — |
| 130 · `state_estimate(state)` | Kalman状态 → L 借用 | 取 estimate 列；图要求 kalman_series 标签。 | — |
| 131 · `state_variance(state)` | Kalman状态 → L 借用 | 取 variance 列，量纲为观测量纲平方。 | — |
| 132 · `state_hysteresis(values, upper_enter, upper_exit, lower_enter, lower_exit)` | L,四个S阈值 → I[T] | 带上下阈值的滞回；初态1，上状态0，下状态2；边界包含等号，阈值不排序；缺失输出−1但保持内部状态。 | — |
| 133 · `state_confirm(codes, confirmation, min_hold)` | I[T],S,S → I[T] | 连续候选确认与最小保持联合递推；切换须计数达标且active_duration>min_hold，不回填过去；−1中断候选但保持活动状态。 | — |
| 134 · `state_continuous(candidate, initial, observed, confirmation, state_count)` | I[T],I[T],L,S,S → I[T,3]名义状态 | 候选持续确认，输出状态/证据/待确认数；initial[0]建立初态；candidate=−1表示没有新提议，非缺失状态；observed必须有限正值。 | — |
| 135 · `continuous_state_values(state)` | 连续状态记录 → I[T] 借用 | 投影 state；编码与更新顺序见[状态契约](state-event-contracts.md#continuous-candidates)。 | — |
| 136 · `continuous_state_evidence(state)` | 连续状态记录 → I[T] 借用 | 投影 evidence；编码与更新顺序见[状态契约](state-event-contracts.md#continuous-candidates)。 | — |
| 137 · `continuous_state_pending(state)` | 连续状态记录 → I[T] 借用 | 投影 pending_count；编码与更新顺序见[状态契约](state-event-contracts.md#continuous-candidates)。 | — |
| 138 · `drawdown_cycle_state(price, drawdown, valid, stress, rebound, exit)` | L,L,B,S,S,S → I[T] | Normal0/Recovery1/Stress2，未知−1；输入回撤和全窗口valid由外部计算；压力、谷底反弹和退出驱动单步转移；缺失重置，见详细状态契约。 | — |
| 139 · `local_extrema(price, left, right, head, tail)` | L,S,S,S,S → I[T]事件 | 有效正值段内，向前严格、向后含等号选最早平台点；后续更强同类点可删除旧点；峰+1、谷−1、无事件0、缺失−2；全样本依赖。 | — |
| 140 · `ps_filter(price, events, min_phase, min_cycle, amplitude)` | L,I[T]事件,S,S,S → I[T]事件 | 依序联合检查交替、端点、最短周期/阶段及幅度豁免；每次删点重启直到稳定；全样本依赖，最坏O(T²)。 | — |
| 141 · `between_events(events)` | I[T]事件 → I[T,2]边界 | 只产生相反事件之间的完整段；缺失断段，同类事件更新候选；未知首尾不补成完整段。 | — |
| 142 · `segment_starts(segments)` | 分段边界 → I[T] 借用 | 先 O(T) 校验完整双列边界、范围和段成员，再借用左列；未知−1。 | — |
| 143 · `segment_ends(segments)` | 分段边界 → I[T] 借用 | 同样先校验完整双列，再借用右列；未知−1；body可读右端点但广播不包含它。 | — |
| 144 · `phase_direction(events, segments)` | I[T]事件,分段边界 → I[T]phase | 谷→峰为0，峰→谷为1，其他−1；phase不是任意state/category编码。 | — |
| 145 · `drawdown_cycle_reference(phases, changes, segments, stress)` | I[T]phase,L,分段边界,S → I[T] | 跨完整波段记录压力；跌幅<=−stress为压力，其后上升为恢复；标记(left,right]并保留共享端点规则；事后结果。 | — |
| 146 · `state_select(condition, when_true, when_false, valid)` | B[T],I[T]或S,I[T]或S,B[T] → I[T] | valid=false输出−1，否则按condition选择状态；数组代码>=−1，标量限[−1,2^53−1]整数；不含确认/滞回。 | — |

## 具名数学组合

内核：[源文件](../cpp/operators/sequence.cpp)；形状/参数校验：[registry.cpp](../cpp/operators/registry.cpp)。

| ID / 调用签名 | 输入 → 输出 | 数学逻辑与边界 | SIMD |
| --- | --- | --- | --- |
| 45 · `cumulative_return(values)` | L → L | 第 t 行为 ∏(1+x[i])−1，i=0..t；具名组合并不识别输入真实金融含义。 | — |
| 75 · `total_return(values)` | L → S | ∏(1+x)−1；输入按逐期小数增量解释，不将输入当 level path。 | — |
| 76 · `annualized_return(returns, periods_per_year)` | L,S → S | 保留先求 total=∏(1+x)−1、再 (total+1)^(periods_per_year/n)−1 的顺序；年频有限且>0，底数有限且>=0。 | — |
| 101 · `portfolio_returns(asset_returns, asset_weights)` | M[T,N],L[N] → L[T] | 复用矩阵向量乘法；不自动归一化权重，不施加投资约束。 | — |
| 102 · `quadratic_form(vector, matrix)` | L[N],M[N,N] → S | xᵀAx；不要求或验证矩阵是协方差矩阵。 | — |
| 103 · `active_returns(lhs, rhs)` | F,F → F | 复用 subtract，x−y；金融口径由调用方保证。 | — |

## 可选参数与配置范围

以下直接摘录当前原生目录的 defaults 字段，保留参数原名供接口核对。它包含默认值和范围提示；更严格的 shape/dtype 检查仍由 prepare/Typed IR 执行。无该字段条目的算子没有隐含可选尾参，重载不等于默认填参。

| 算子 | 原生默认值及配置契约 |
| --- | --- |
| `variance` | ddof=1; finite integer 0<=ddof<n |
| `std` | ddof=1; finite integer 0<=ddof<n |
| `rolling_mean` | min_periods=window; window>=1, 1<=min_periods<=window |
| `rolling_std` | ddof=0, min_periods=window; window>=1, 1<=min_periods<=window |
| `rolling_min` | min_periods=window; window>=1, 1<=min_periods<=window |
| `rolling_max` | min_periods=window; window>=1, 1<=min_periods<=window |
| `lag` | periods=1; lag permits 0, difference requires >0; periods<n |
| `difference` | periods=1; lag permits 0, difference requires >0; periods<n |
| `aligned_shift` | periods=1, fill=NaN; periods>=0 may exceed length; fill finite or NaN |
| `recursive_filter` | seed_mode=0 consume first with initial, 1 seed row0 without consumption, 2 seed first eligible; emit_policy=0 hold, 1 NaN; alpha in [0,1], initial finite |
| `distinct_count` | mask omitted selects every identifier; no implicit missing category code |
| `recursive_filter_adaptive` | initial=0 finite; seed_mode=1 (0 initial consumed, 1 first finite, 2 first eligible); min_periods=1 >=1; emit_policy=1 (0 hold, 1 NaN); consumed alpha finite in [0,1]; update previous+=alpha*(x-previous) |
| `linear_filter2` | min_periods=1 >=1; bootstrap=1 (first two finite values seed outputs), 0 uses zero initial states; y=b0*x+b1*previous_x+a1*previous_y+a2*older_y |
| `scalar_kalman` | initial_variance=1; finite process_variance>=0, measurement_variance>0, initial_variance>=0; no implicit floors or observation warmup |
| `state_hysteresis` | all finite thresholds required; initial neutral=1, upper=0, lower=2; thresholds retain caller ordering |
| `state_confirm` | confirmation,min_hold required integers in [1,2^31-1]; codes>=-1 exact int64 |
| `state_continuous` | all required; confirmation integer [1,252], state_count integer [2,12]; initial[0]>=0; candidate/initial in [-1,state_count-1] |
| `drawdown_cycle_state` | all required; 0<=exit<stress<1, 0<rebound<1; valid is caller-composed complete-window mask; drawdown in [-1,0] |
| `local_extrema` | all required; left/right integer [1,5000], head/tail [0,5000]; strict preceding and inclusive following comparisons choose earliest plateau |
| `ps_filter` | all required; min_phase integer [1,10000], min_cycle [2,20000], amplitude>=0 finite; amplitude exemption is strict > |
| `drawdown_cycle_reference` | all required; 0<stress<1; completed decline<=-stress marks stress and following rise recovery; membership (left,right] |
| `state_select` | all required; scalar branch finite integer [-1,2^53-1], int64 branch values>=-1; -1 is explicit unknown/no proposal; validity stays separate |

## 递推策略与共享求解

- recursive_filter：seed 0 从 initial 开始并消费首行；seed 1 第0行输出 initial 不消费；seed 2 以首个有限且 mask=true 的观测为种子。emit 0 停更时保持输出，emit 1 输出 NaN；未得到 seed 时 NaN。缺失期间保留内部状态。
- recursive_filter_adaptive：reset 在本行处理前清空状态/计数；seed 0 以 initial 参与更新，seed 1 首个有限值作种子，seed 2 首个有限且允许更新值作种子。有限观测计数与 update_mask 独立；达到 min_periods 后按 emit 决定停更输出。与旧 recursive_filter 的 seed 1 含义不同。
- linear_filter2：bootstrap 1 用每段前两笔有限值作输出种子，0 使用零历史；非有限输入输出 NaN并保持历史，非有限计算结果清空历史；reset_mask 可显式让缺失断段。它没有 b2*x[t−2] 参数、反向执行或边界延拓契约。
- scalar_kalman：首个有限值初始化估计，初始方差默认1；缺失行两字段NaN，内部估计/方差保持，期间不累计Q。不是矩阵Kalman。
- 每个区间和每个独立作用域重新初始化，不跨请求保存状态。更详细初始化/重置及组合例子见[递推设计](stateful-series-design.md#2-递推契约)。

状态码、等号、PS 删点顺序、端点归属是算法的一部分，详见[状态事件契约](state-event-contracts.md)。例如 state_confirm 要求 active_duration **大于** min_hold；state_continuous 的 candidate=-1 表示没有新提议；local_extrema 的 event=-1 则表示谷点，不能互换。

## 编译器作用域不是 canonical 算子

rolling_window、rolling_apply、block_apply、filter_apply、group_apply、bisect、segment_apply 由编译器管理，见[作用域使用说明](user-guide.md#6-窗口分组和求根作用域)。interval_tail 仅可在受限 root_bindings 内使用，不是普通算子。完整指标通过这些作用域和上表算子组合；不把每种金融指标注册成新黑盒。

## 查询安装包与可运行示例

```python
import numpy as np
from calmetrics_engine import operators as op

catalog = op.catalog()
assert len(catalog) == 146
spec = op.get("std").spec
assert spec["parameters"] == ["values", "ddof"]
x = np.array([1., 2., 3.])
assert op.std(x, ddof=0) == np.std(x, ddof=0)
print(op.get("matmul").requirements(np.ones((2, 3)), np.ones((3, 4))))
```

```python
import numpy as np
from calmetrics_engine import operators as op

x = np.array([3., 1., 2.])
np.testing.assert_array_equal(op.gather(x, op.argsort(x)), [1., 2., 3.])
np.testing.assert_array_equal(op.lag(x), [3., 1.])
np.testing.assert_allclose(op.aligned_shift(x), [np.nan, 3., 1.], equal_nan=True)
fit = op.linear_fit(np.array([1., 3., 5.]))
assert op.fit_slope(fit) == 2.0
```

catalog() 给出签名、shape_rule、defaults、missing_policy、granularity、temporal_dependency、SIMD 和输入策略等元数据。部分字段是族级摘要；temporal_dependency="unspecified" 不是“已证明因果”。requirements(*args) 才按具体参数给出输出种类、形状和工作区需求；不能拿目录摘要代替实际输入检查。

## 更新与验证

修改公开算子时同步本表及相应详细契约，核对 opcode/名称集合、所有重载参数和默认值。用户示例按当前已安装 wheel 执行；不通过保存旧算法副本证明历史兼容。固定参考、边界和所有权测试仍是数值真相的验证依据。


M0/M1 开发分支的 `iterate` 与三个诊断投影由编译器拥有，不增加 canonical opcode；完整契约与示例见[使用手册](user-guide.md#12-开发分支新增有界迭代与诊断)。张量只扩展通用逐元素/掩码及全数组归约；矩阵代数、按轴归约和时序内核仍按各行指定 rank 校验。
