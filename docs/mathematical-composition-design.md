# 缺失数学能力与原生组合执行设计

日期：2026-09-21。依据：当前 `AGENTS.md`、[指标能力审计](metrics-factory-capability-audit.md)。

## 1. 目标与范围

补齐审计 G1–G6 所需的通用数学和图执行能力，使递推指标、分段统计、分组统计和标量反解能在一个原生请求中组合。完整链路仍是 Python 参数边界 → C++ parser/Typed IR → DAG/CSE → 内存及执行计划 → 原生调度器 → C++ 数值计算。

本设计不将 Hurst、CPR、EMA、KDJ、Black–Scholes、IV 或 VIX 注册为新的指标黑盒，也不修改 MetricsFactory、研究平台或其默认服务后端。迁移全部指标定义、数据关联、PIT 门禁和业务发布需要另行验收。本次新增能力的可运行证明由独立参考测试提供；旧审计保留为变更前证据。

## 2. 授权与兼容边界

1. 新增能力直接对应已确认的缺口，没有引入另一套计算运行时。
2. 旧 opcode 1–118、`lag` 裁短、`recursive_smooth` 整数周期及缺失输出规则均保留。新语义追加 opcode，不重解释旧定义。
3. 紧凑分块结果与同轴时序有不同形状，不自动补齐或伪装成原时间轴；必须继续归约、映射或由调用者明确输出语义。
4. 保留严格报错和逐根隔离两种契约；不会将任意子图错误默认为有效的 0 或 NaN。空集合的缺失返回是显式集合契约。
5. Python 不执行分组、求根迭代、窗口计算或逐节点回调。测试中的 NumPy/Python 参考仅用于验收。
6. 整数类别必须采用真正的 int64；不能转换成 float64 后处理大于 2^53 的 ID。

## 3. 新增基础算子

参数位置、默认值和约束必须同时落入 canonical registry、Typed IR 和原生 prepare；本文与最终公开接口同步。

| 算子 | 唯一职责 / 颗粒度 | 输入与输出 | 关键契约 |
| --- | --- | --- | --- |
| `normal_cdf(x)` | 标准正态分布累积概率；基础算子 | float64 标量或数组，同形输出 | `0.5*erfc(-x/sqrt(2))`；±Inf 对应 0/1，NaN 传播；无需近似反演 PPF |
| `aligned_shift(x,k=1,fill=NaN)` | 向过去取值并保留原轴；基础算子 | 一维 float64 序列，同长度输出 | 非负整数 k；k=0 原值；k≥N 全填充；不引用未来，不修改旧 lag |
| `recursive_filter(x,alpha,initial,mask,seed_mode=0,emit_policy=0)` | 单状态仿射递推；耦合内核 | float64 序列、alpha∈[0,1]、有限初始值、同轴 bool mask；同轴序列 | `s=alpha*x+(1-alpha)*s`；所有 seed mode 均校验 initial；初始化、停更及输出策略显式；每个产品/区间/窗口重新初始化 |
| `argsort(x)` | 生成稳定排序索引；基础算子 | 一维数值或整数序列 → int64 索引 | 同值按原位置稳定；浮点 NaN 放末尾；不覆盖输入；输出代表选择顺序而非新的时间因果性 |
| `gather(x,indices)` | 用受检索引同步选择一个字段；基础算子 | 一维数值/mask/int64 和 int64 索引 → 保持数值 dtype 或 mask 语义的序列 | mask 输出沿用 uint8 存储；越界/负索引失败；重复和空选择合法；输出是必要分配，不声称为视图 |
| `distinct_count(ids[,mask])` | 计算被选整数类别的不同值数量；基础算子 | int64 和可选 bool mask → float64 数量标量 | 默认选择全部；不把负值自行判为未知；缺失/未知由显式 mask 决定；准确比较完整 int64 值，计数须能由 float64 精确表示 |
| `floor(x)` | 逐元素向下取整；基础算子 | float64 标量或数组，同形输出 | 返回 float64，NaN/Inf 传播；不隐式转类别；为 `floor(length(x)/m)` 提供通用分段参数 |

`recursive_filter` 的 seed mode：0 使用 initial 并消费首行；1 第 0 行输出 initial、不消费首行；2 首个有效且被 mask 选中的值成为 seed。emit policy：0 停更行输出已有状态；1 停更行输出 NaN。未得到首个有效 seed 时输出 NaN。有效更新要求输入有限且 mask=true；无限值不会污染永久状态。调用方应按自己的 NaN/Inf 契约构造 mask。

### 算子颗粒度四问

上述算子均只解决表中一个问题；输出可被独立复用。CDF、真实整数去重和稳定索引在原 registry 中没有等价原语；floor 也不能借整数强转替代负数向下取整。对齐移位与旧裁短 lag 的形状不同。递推具有不可分离的前一状态，拆为普通无环节点会丢失状态语义，因此保留一个通用耦合内核，EMA/KDJ 的其他步骤仍是普通图节点。

排序的索引输出可用于同时选择价格、数量、行权价和期限，避免每字段独立排序造成错配。没有新增业务指标算子、单独线程池或不透明 Python callback。

## 4. 编译器拥有的计算作用域

这些是作用域/控制流结构，类似已有 `rolling_apply`，不是隐藏在 registry 中的完整业务指标。body 编译为同一种 `graph::Program`，复用现有 executor、错误状态和资源管理。

| 结构 | 语义 | 空值与边界 |
| --- | --- | --- |
| `block_apply(body,width)` | 从当前作用域首行开始，执行不重叠的完整块；每块 body 返回一个标量，按块顺序收集 | width 为正整数；丢弃不足一块的尾部；不自动删除空行；输出长度 floor(N/width) |
| `filter_apply(body,mask[,empty_default])` | 按原顺序选择 mask=true 的行，在选中数据上计算一个标量 | 空选择默认 NaN，可显式指定空集返回值；不是对 body 的异常兜底 |
| `group_apply(body,keys)` | 按精确 int64 key 分组，每组计算标量并广播到该组所有原行 | 同组保留原行顺序；组间不依赖整数编码的大小含义；输出保持原轴；空输入为空 |
| `bisect(body,lower,upper,tolerance,max_iterations)` | 在显式区间内求 body=0，body 使用局部标量 `solve_x` | 有限区间、正容差、有界迭代；检查端点根、夹根与有限函数值；不夹根/不收敛明确报错 |

求根的 bracket 更新属于不可分离迭代；价格、Delta、观测误差和参数检查仍以可见节点组合。tolerance 是 x 的绝对误差/区间收敛规则，不能混称成价格误差；参考脚本固定 80 次二分与新求解器的停止契约分别记录，不伪装成完全相同的历史算法。

分块与筛选有不同先后关系：CPR-5 必须先在原行轴分块求和，再删全空块；Hurst 必须先删除缺失观测，再按清理后的长度分块。编译器不得交换这两个步骤。

### 捕获、嵌套与形状

- 所有 body 依赖必须显式计入父图；预先计算的派生序列作为被捕获值，不在每个子窗口内重新计算全历史滤波。
- 局部 `solve_x` 只在求根 body 中有效，不会成为外部待绑定参数。
- 子作用域嵌套设置硬上限，并限制节点数、参数范围和展开规模，避免无界递归或求根。
- 紧凑块轴、筛选后的局部轴与原轴区分；不能将其直接作为保持原时间轴的公开 series root。
- `group_apply` 使用完整 group 后再广播，属于分组内全样本依赖。排序、筛选和普通比较不会将其转成因果序列。
- `group_apply` 保留 keys 的 shape、命名轴和值类别；asset 向量分组结果仍是 asset 向量，只能作为中间量，不能伪装成公开时间序列根。
- 严格模式的子图数值异常传播；隔离模式保留无关根结果并记录受影响根状态。非法 dtype、索引、几何和资源错误仍失败关闭。
- 对齐输出的 group_apply 在隔离模式下只把失败组置为 NaN，继续计算其他组，结果状态逐行反映失败；严格模式仍抛错。求根失败不能污染同一批次中其他合约组。
- filter_apply 的空集 NaN 是显式缺失数据，不立即作为中间节点执行错误传播；写出非有限标量根时仍给出缺失状态。这样组合中的 where 可以选择有效分支；body 真正抛出的异常仍传播，不修改旧 eager where 的求值规则。

`GraphCompiler.compile(..., scope_work_budget=100_000_000)` 提供额外的子图工作上限，可设置为 1 到 10^12。计数按子程序节点数和本次处理的观测数累计，是每个原生执行分块的工作保护，不是整个多 worker 请求的总 CPU 配额或计时器；大型合法批次可显式提高上限。上限、嵌套深度和求根迭代次数随编译程序及 pickle/worker 传输保留。超限是资源错误，不得由 rolling 或错误隔离转换为成功结果。

## 5. int64、矩阵与原生输入边界

扩展同一个 Value/Typed IR，追加整数 dtype/kind，不以 double 表示索引。整数只用于显式支持的索引、类别、选择和去重路径；普通浮点数学不得隐式吞下 int64 参数。

矩阵输入声明 time×asset，向量声明 asset。对区间 [start,end) 只切 time 轴，asset 向量保持完整。已有 sum_asset、mean_asset、矩阵/向量算子由相同 registry 执行。公开 float64 标量/同轴时序结果契约保留；矩阵和整数中间结果通过归约/选择得到受支持根，不自动扁平化。

输入布局、dtype、shape、strides、owner 必须跨以下路径保持一致：普通 execute、PreparedGraphExecution、线程分块、进程打包、共享内存、worker 重建。prepared plan 校验实际几何和输入身份，不能只凭元素总数相同复用。Graph Program 序列化升级至 v4，保留 v1–v3 按原契约读取；worker 请求/响应协议使用 v3 精确版本检查，不混用新旧 worker 包，也不猜测旧协议缺少的 dtype/布局。

原 float64 时间序列入口的 C-contiguous 限制保留；新增矩阵、向量、int64 和 mask 接口按各自已验证的 stride 契约绑定。每次复用缓存或 prepared 绑定，还要复核底层 owner 的地址和范围，不能只检查上层 view 的 shape/strides。

日期关联、分类编码及映射版本、复权口径、PIT 成员、期权合约配对的业务规则由调用方输入契约提供。引擎负责计算和受检选择，不擅自补值或猜测映射。

## 6. 内存、成本和执行计划

- 原输入继续只读借用。连续 block 使用切片；排序索引、离散筛选和 gather 的必要输出/工作缓冲区由原生层管理，不能反复复制全部输入。
- 整数中间缓冲必须使用正确 int64 对齐和容量，不能依赖 double 存储的类型别名。将其纳入 liveness、分支局部 arena 和内存预算。
- 矩阵容量按真实元素数计算，避免以时间行数作为数组槽容量。输出形状变换要检查乘法溢出和最大容量。
- 子程序 scratch 按生命周期复用；父捕获值必须活到所有子程序执行结束，不能被外层 slot 回收覆盖。
- `lag`、`transpose` 和矩阵 `diag` 都会借用上游存储，其底层 owner 的最后使用点必须沿借用链传播；分支提取后的局部 arena 遵守同一规则。
- 对物理成本计入排序 NlogN、分组成员访问、子 body 工作、分块数量及求根迭代上界；不能按“一个 scope 节点”当常数工作。
- Typed 数组成本按真实形状补足矩阵/静态向量工作：elementwise/归约按元素数，matmul 按 MKN，covariance 按 TN²，solve 按 N³；共享归约/排序只计一次。保留原成本系数和调度阈值，这些是结构性工作估计，不能冒充实测耗时。
- root/branch 抽取、序列化和 worker decode 保留所有控制节点、捕获映射及输出索引。预算由同一个 native Scheduler 管理。
- 不因新功能建立新线程池或在 body 中调用 Python。退出、错误和 timeout 路径遵循原调度器所有权规则。

## 7. 指标如何展开

| 指标族 | 可见的组合步骤 |
| --- | --- |
| EMA/EMADiff | 根据 span 得到 alpha → recursive_filter(first-valid) → 原值差 |
| KDJ | rolling high/low → 保护分母的 RSV → K 滤波 → 共用 RSV mask 的 D 滤波 → J/KD/KJ/DJ |
| TRIX | 三层 EMA → aligned_shift → 相对变化 → 已有 rolling_mean |
| MACR/ADXR | 原定义的滚动统计 → 派生序列 aligned_shift → 差值/平均 |
| Hurst | filter_apply 删除 NaN → 每尺度 block_apply → 段内范围/std 或累计偏离范围/std → 尺度均值 → 对数回归 |
| CPR | 原轴 block sum → 跳过空块 → 正负判断 → aligned_shift → WW/LL/WL/LW 计数 → 比值 |
| 市场/行业聚合 | 矩阵 asset 归约或 int64 group_apply → 和/均值/份额 → 平方和等组合 |
| 持有基金数 | group_apply(distinct_count(ids,mask),group_id) |
| 期权 IV | 可见 BS 价格公式（CDF/log/exp/sqrt）→ 与报价之差 → bisect(solve_x) |
| VIX-like 选择 | 稳定行权价 argsort → 对各字段同索引 gather → K0/相邻间距/加权和 → 期限插值 |

Hurst 两种历史定义不可混淆；RS 要包括累计路径初始 0、跳过无效段及至少 4 个有效尺度。CPR 不使用引擎原有同名 cal_cpr 的另一种金融定义。KDJ 两级共享停更 mask，不能让 D 在 RSV 缺失时继续追赶 K。

## 8. 验收与证据边界

实施完成后在配套验收记录填写真实结果，未完成前不将本设计当作已经实现的能力。

1. 算子独立参考：CDF 尾部、shift 边界、递推 seed/mask、稳定排序、int64 极值、重复选择、去重、负数 floor。
2. 图组合参考：EMA/KDJ/TRIX、派生序列移位、Hurst 两种、CPR 分块/缺失、分组聚合、持有人去重、BS→IV 反解。
3. 类型与错误：同轴检查、矩阵形状、int64 不隐式转浮点、无效 mask/索引、空选择、无夹根/不收敛、严格及隔离模式。
4. 实际 wheel：single/thread/process、prepared 重复执行和输出快照、readonly/strided 输入、owner 生命周期、输入不变、零 Python fallback。
5. 旧 canonical 118 个算子的固定参考、已有 C++/Python 全量回归；新增用例补在新文件，不重写旧参考结果。
6. 新的内存及成本路径检查，以及现有全负载性能门禁。只有实际测量后才报告更快或性能验收通过；Linux/Windows/其他 CPython/ISA 未运行的检查明确列出。
