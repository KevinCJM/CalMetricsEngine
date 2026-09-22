# Canonical Operators 使用与内存契约

## 当前能力

`calmetrics_engine.operators` 提供 146 个可执行 canonical 算子。全部经由同一 `_native` 扩展进入 C++，不依赖 Numba，不在首次调用时编译，也不回退到 Python 数值实现。

完整名称、签名、形状与数学逻辑见[数学算子参考](operator-reference.md)；整图类型和示例见[使用手册](user-guide.md)，调度条件见[执行指南](execution-guide.md)。本文集中说明直接调用、内存和工作区契约。

本页描述直接算子接口；公式/DAG 和整图执行由原生 GraphCompiler、Planner 与 Scheduler 提供，单个算子不自行创建线程池或进程池。调用方提供名义轴、计量单位、价格基准和时点可得性；Typed IR 校验已声明的类型与轴契约。

## 基本调用

```python
import numpy as np
from calmetrics_engine import operators as op

values = np.array([0.01, -0.02, 0.03, 0.005], dtype=np.float64)

volatility = op.std(values)                  # ddof=1
population_std = op.std(values, ddof=0)
portfolio_gain = op.total_return(values)

operator = op.get("std")                    # 名称解析一次，重复使用 handle
assert operator(values) == volatility
assert len(op.catalog()) == 146
assert op.get_by_opcode(operator.spec["opcode"]).name == "std"
```

`op.call("std", values)`、`op.std(values)` 和 `op.get("std")(values)` 使用同一 C++ 注册项。`catalog()` 返回新建的元数据对象，修改返回的字典不会修改注册表。

注册表版本为 `canonical-native-1`。显式 opcode 1–125 保持既有契约，126–146 是递推、状态与 typed 时序扩展，**不等于旧 Numba 的 BASIC_OPCODES 或旧计划二进制编码**。能力发现使用 catalog，不通过猜测外部 opcode 映射。旧参考值保留为固定回归。

119–125 依次是 `normal_cdf`、`aligned_shift`、`recursive_filter`、`argsort`、`gather`、`distinct_count`、`floor`，详见[数学组合设计](mathematical-composition-design.md)。126–146 详见[递推与状态设计](stateful-series-design.md)及[状态事件契约](state-event-contracts.md)；运行时 `spec` 元数据与原生 registry 是参数入口。分块/筛选/分组/求根/分段是编译器作用域，不另计为 canonical 算子。

## 输入类型与数据准备

数值数组必须是本机字节序、元素对齐的 `float64 ndarray`。掩码接受 `uint8` 0/1 或 NumPy bool。按各算子契约接受标量、一维或二维，只有明确的标量广播，不隐式进行矩阵与向量的轴广播。

`argsort/gather/distinct_count`、整数 `equal/not_equal` 重载和状态/事件接口另接受其契约声明的本机 `int64` 数组；排序索引与类别比较保持精确，不能转为 float64。普通浮点数学仍拒绝 int64 数组。`distinct_count` 返回 float64 数量，并检查数量可精确表示；负类别 ID 只有在显式 mask 排除时才被忽略。

Python 数值标量可使用 float 或可精确表示为 float64 的整数；也支持 NumPy float64 和范围内的整数标量。NumPy bool/uint8 标量按 mask 解释。拒绝 Python list、float32/object/complex 数组、非本机字节序和不支持的 rank，不偷偷转换。

数据入口可以做一次明确的解码、dtype 转换、排序或对齐，进入算子后不得反复整理为连续副本。C/F order、普通切片、负 stride、零 stride 广播视图和 readonly 输入按真实 stride 读取。

重型产品计算推荐调用方准备 product-major 数据：

```python
values = np.array([0.01, 0.02, -0.01, 0.04, -0.02], dtype=np.float64)
offsets = np.array([0, 3, 5], dtype=np.int64)

product = 1
begin, end = int(offsets[product]), int(offsets[product + 1])
view = values[begin:end]                     # 基本切片，不复制数据
assert np.shares_memory(view, values)
result = op.total_return(view)
```

这里 offsets 只用于调用方定位，**并非本阶段新增了 product-offset 批量执行 API**。半开区间 `[begin,end)` 不能越过产品边界；不能把多产品拼接后的数组整体当成一个产品来做 lag、差分或递推。

原有顶层 `cal_*` finance API 继续使用各自的二维 `(observations, columns)`、int32 分组和 int64 纳秒日期契约。不要将这些旧 API 的 dtype/日期约定直接套用到 canonical 算子。

## 预分配输出

```python
output = np.empty_like(values)
returned = op.add(values, 1.0, out=output)
assert returned is output

scalar_output = np.empty((), dtype=np.float64)
assert op.mean(values, out=scalar_output) is scalar_output
```

数组/标量输出的 `out` 必须 shape 和 dtype 精确匹配、可写且 C-contiguous。mask 输出使用 uint8，也可提供兼容的 bool out。状态 tuple 不支持 out。

不允许 out 与任何数组输入的字节范围重叠；检查使用保守内存范围，因此某些实际不重叠的交错视图也可能被拒绝。需要独占输出时应由调用方预分配，而不是请求引擎复制共享输入。

不传 out 时，只分配最终输出和算法确实需要的工作区。**执行抛错后 out 可能已被部分写入，调用方必须丢弃本次结果**；引擎不为事务回滚再复制整块输出。

## 返回视图的算子

`lag`、`transpose`、二维输入的 `diag` 默认返回只读 NumPy 视图，保活源 owner：

```python
matrix = np.arange(12.0).reshape(4, 3)
transposed = op.transpose(matrix)
assert np.shares_memory(transposed, matrix)
assert not transposed.flags.writeable

prefix = op.lag(values, periods=1)            # 源契约：values[:len(values)-1]
assert np.shares_memory(prefix, values)
```

`lag` 不是自动补 NaN 的等长时移。显式传入 out 时可以将视图所描述的结果写入最终目标，这是一项显式输出操作，不是输入规范化复制。

视图只读并不阻止其他别名写入原始数据。**计算进行时以及需要维持结果快照时，调用方不得通过其他别名修改输入**。

## 工作区复用

```python
workspace = op.Workspace()
quantile = op.get("quantile")
need = quantile.requirements(values, 0.95)
workspace.reserve(
    doubles=need["scratch_doubles"],
    indices=need["scratch_indices"],
)
q95 = quantile(values, 0.95, workspace=workspace)
q50 = quantile(values, 0.50, workspace=workspace)
```

工作区可在顺序调用间复用。一个 Workspace 不得同时借给两个运行中的调用，竞争时返回 `WORKSPACE_BUSY`，不会持有 GIL 等锁。不同 Workspace 的独立调用可并发，算子执行阶段释放 GIL，但自身不新建线程。

| 算法 | 工作区含义 |
| --- | --- |
| 普通元素运算、严格顺序基础归约、多数 scan | 无数值 scratch |
| median/quantile/掩码分位数 | 将选中数值复制到连续 `double` scratch 后排序；输入仍只读，复制计入 `algorithm_copy_bytes` |
| rolling_min/max | 最多 `min(window, n)` 个索引的循环队列 |
| covariance/correlation 矩阵 | 均值/标准差等辅助向量 |
| quadratic_form | matvec 中间向量 |
| solve | 可变系数矩阵和 RHS，部分主元消元所必需 |

`median/quantile` 的连续排序 scratch 与 `solve` 的可修改矩阵/RHS 都属于算法必要工作区复制，并单独报告为 `algorithm_copy_bytes`。这不改变 `input_copy_bytes=0` 的绑定层零拷贝事实，也不能被宣传成“整个算法零复制”。输出不会借用随后复用的 scratch。

## 缺失、窗口与金融口径

| 算子/类别 | 重要契约 |
| --- | --- |
| 普通 std/variance | 默认 ddof=1；不会自动去掉 NaN；保持源顺序归约语义 |
| rolling_std | 默认 ddof=0，min_periods=window；只统计有限值，保留原始日期位置 |
| rolling_mean/min/max | 尾随窗口，min_periods 默认 window；有限值不足输出 NaN |
| quantile | 概率严格在 `(0,1)`，线性插值；median 支持奇偶长度 |
| masked reduction | mask=0 的位置不参与数值读取；被选位置仍遵守源缺失语义 |
| divide | 分母为零抛 `DIVIDE_BY_ZERO`，不会自行切到宽松 series 模式 |
| divide_or_default | 非有限输入输出 NaN；有限且 `abs(rhs)<1e-12` 时使用 default |
| drawdown_series | 输入正值 level path，输出有符号回撤；不同于旧 cal_max_dd 的 returns 输入 |
| new_high_mask | 首项为真，之后只有严格创新高为真 |
| recursive_smooth | 非有限位置输出 NaN，递推状态跨缺口保留，不压缩时间轴 |
| value_at | 零基位置；非整数、负数、越界、缺失位置或非有限结果返回 NaN |
| days_between | 两个自然日编号相减；不转换 Unix 纳秒或 datetime64；无效/倒序返回 NaN |
| covariance/correlation | 单矩阵输入按 time 轴（axis 0）计算；样本协方差 ddof=1 |

普通算子采用冻结源数值内核的语义，不替调用方补做整个 Typed DSL 的有限性、轴和金融计量验证。NaN/Inf 是否拒绝、跳过或传播由各算子的契约决定，不能统一删行、填零或启用全局 fast-math。

## 拟合与区间状态

```python
series = np.array([1.0, 1.9, 3.2, 4.0], dtype=np.float64)
fit = op.linear_fit(values=series)           # 隐式 x = 0,1,...,n-1
slope = op.fit_slope(fit)                    # 只读取状态，不重新拟合

# 双输入重载使用源参数名 x、y。
fit_pair = op.linear_fit(x=np.arange(4.0), y=series)

interval = op.last_drawdown_interval(
    np.array([0.0, -0.2, 0.0, -0.2, -0.2, 0.0], dtype=np.float64)
)
assert interval == (2.0, 4.0, 5.0, 1.0)
```

状态使用与源实现一致的 tuple：

- OLS 五字段：`(slope, intercept, residual_sum_squares, total_sum_squares, observation_count)`。
- 回撤四字段：`(start, trough, recovery, event_status)`。无效输入 status=-1，无事件 status=0，找到事件 status=1；未恢复的 recovery 为 NaN。

同深度回撤选择最后谷底。状态投影不遍历原数组；字段数和 native Kind 不匹配时拒绝。tuple 形式用于本阶段源兼容，不等于已完成 DSL 名义类型系统。

## SIMD 和审计

```python
left = np.arange(17.0)
right = left + 1.0
result, audit = op.add(left, right, simd="auto", audit=True)
print(op.available_simd())
print(audit["isa"], audit["vector_elements"], audit["input_copy_bytes"])
```

当前有显式 SIMD 的 18 项是：`add/subtract/multiply/divide/minimum/maximum/negate/absolute/sqrt/reciprocal/equal/not_equal/less_than/less_equal/greater_than/greater_equal/finite_mask/matmul`。

ARM64 使用 NEON，x86_64 使用 SSE2；AVX2 仅在独立 AOT 目标已构建且 CPU/OS 支持时调用。没有 AVX-512 实现。模块初始化和普通代码不带全局强制 AVX 标志。

`simd="scalar"` 禁用显式 SIMD；`auto` 选择支持的指令集。显式请求不支持的 ISA 会报错。非单位 stride、小于向量宽度、顺序敏感的递推/归约可走 C++ scalar 路径；这不是 Python fallback。

元素算子支持非向量宽度对齐的安全 load/store 和尾部处理。矩阵乘法采用 4 行×2 SIMD 向量的寄存器分块，保持 k 归约顺序且不打包复制输入；要求右矩阵沿列单位 stride 和足够的块大小，其余使用 C++ 标量路径。

逐次审计字段包括实际 ISA、实际向量化元素/输出单元数、输入地址、绑定输入复制字节数、算法工作区初始化复制字节数和工作区需求。`vector_elements` 对 matmul 统计向量计算的输出单元，不是浮点运算次数。所有审计属于本次返回值，不共享全局 last-call 状态。

**C++/SIMD 不保证所有操作比 NumPy/BLAS 更快。** 当前矩阵实现强调零输入打包和运算顺序，实测仍慢于系统 BLAS；完整基准及限制见验收记录。

## 数据安全边界

绑定在释放 GIL 前检查 dtype、rank、元素对齐、shape、stride 算术和可知 owner 的内存边界；可识别的越界 as_strided 视图会被拒绝。具有不透明 capsule 或外部 exporter 的数组，其真实分配范围仍依赖 exporter 合法性契约。不能将任意伪造指针当作受保护的输入。

C++ 直接使用 `operators.hpp` 的调用者还必须保证输入指针有效、输出独占、工作区独占；Python 绑定的 owner/out 校验不能自动保护绕开绑定的原生调用。

## 文件与后续集成

- 唯一注册项：`cpp/include/calmetrics_engine/operators.def`。
- 原生契约/入口：`operators.hpp`、`cpp/operators/registry.cpp`。
- Python绑定：`cpp/operator_bindings.cpp`。
- 开发设计：`canonical-operators-design.md`。
- 对照、平台与性能证据：`canonical-operators-acceptance.md`。

当前 native graph executor 直接复用同一套 C++ `lookup/prepare/execute` 和 canonical kernels，并已具备原生 Typed IR、名义轴、部分声明语义校验、编译器作用域和 typed 时序根。业务字段选择、完整金融口径、因果性和知识时点认证仍由调用方负责，不能从直接算子运行成功推断这些业务条件已成立。
