# CalMetricsEngine 用户使用手册

[文档导航](README.md) · [全部算子](operator-reference.md) · [执行与性能](execution-guide.md)

本手册对应当前 146 算子实现。使用预先构建的 AOT wheel；开发构建见根目录 [README](../README.md#install)。导入或请求期间不会编译机器码，也没有 Python 数值回退。GraphCompiler 将公式编译为原生执行图，不属于机器码 JIT。

## 1. 选择入口与职责

| 入口 | 用途 | 输入和输出特点 |
| --- | --- | --- |
| operators | 单次数学运算、直接矩阵输出、out/Workspace 复用 | 按每个算子的 rank/dtype 契约；不自行创建计算线程池 |
| GraphCompiler + AdaptiveScheduler | 多输出共享 DAG、多产品/区间、自动计划与执行 | 显式类型/轴，支持标量、时序、向量、矩阵、三维张量及异形多输出，统一调度和结果凭据 |
| cal_* | 已有财务接口的兼容调用 | 独立的二维数值、int32 分组、日期及窗口契约；不要套用图接口规则 |
| excel.plan / plan_operator + excel.export | 按需生成可独立重算的 Excel 步骤文件 | 原生公式规划、独立快照、全工作簿预算；[接口及验收边界](excel-export.md) |

调用方决定输入是净值、成交量还是某种价格，负责数据来源、口径、日期对齐、指标定义及结果解释。引擎负责其公开数学、类型、执行和内存契约。

**当前兼容边界**：Typed IR 仍支持 semantic_dimension 与 price_basis，并校验部分已声明语义的兼容性。简写 series 默认 dimensionless，不按变量名称猜测金融字段。本文不新增金融约束，也不表示已有语义校验已移除。不能仅凭图运行成功认证金融口径或因果性正确。

## 2. 标量、时序、向量、矩阵和张量

| kind | 数学意义 / shape | 图声明与传入方式 |
| --- | --- | --- |
| scalar | 一个数，无轴 | 声明 scalar，经 parameters 传入；当前参数为 float64 |
| series | 时间序列 T | time 轴，inputs 中的一维 ndarray |
| vector | 静态一维数据 N | 默认 asset 轴，也支持 model/path 等已注册静态轴；不按时间区间裁切 |
| matrix | 二维数据，例如 T×N | 显式 axes/shape；默认 time×asset，按时间区间切行 |
| value | 精确 dtype 的零维数值 | 经 inputs 传入零维 float64/int64/bool ndarray；区别于 float64 parameters |
| tensor | 三维数组 | 显式三条 axes/shape；time 只能在首轴，没有 time 时整块作为静态输入 |
| window | 逻辑上的 T×W 窗口 | 编译器中间态；不要求输入展开矩阵，也不能直接成为公开根 |
| record | 同一次求解的多个字段 | 拟合/区间记录先用投影算子取字段；不能直接成为公开根 |

Kalman、连续状态、完整段边界等共享状态在物理层是 T×2/T×3 数组，在 Typed IR 中带专用名义标签。必须用相应字段投影，不能随意替换成同形普通矩阵。

series 和 vector 都可能是一维，但轴不同。矩阵 `(T,N)` 配向量 `(N,)` 可以 matvec；不能因为数组长度相同就把 vector 当作 series 相加。普通逐元素运算只支持契约允许的标量广播或同轴同形数组，不支持任意 NumPy 广播。

## 3. 输入准备

- 数值 float64、类别/索引 int64、条件 bool；声明为 bool 的数组也接受经校验的 uint8 0/1。不得隐式把大整数变成 double，或把浮点数组变成状态编码。
- 必须提供本机字节序、元素对齐的 NumPy ndarray。list、float32、object 等不自动转换。必要转换在调用方入口完成一次。
- 图的 float64 series 必须 C 连续；typed vector/matrix/tensor、int64 和 mask 支持合法的正/负步长视图。直接算子支持其契约允许的步长，不要套用图的连续性限制。
- inputs 名称必须与编译后 input_names 完全匹配。相同符号维度（例如 T、N）必须长度一致。时间数据的排序、频率与日期对齐仍由调用方保证。
- 数组可以只读。引擎保活输入 owner；计算期间不得通过其他别名修改数据。prepared 可在两次已完成运行之间更新原缓冲内容，但不能改变其地址、dtype、shape、strides 或区间几何。
- 不统一删 NaN、填零或忽略 Inf：各算子的缺失规则见参考表。价格角色、币种、成交量单位等不是从 ndarray 自动识别的。

多产品推荐每个字段一个 product-major 数组：`[产品0全部观察, 产品1全部观察, ...]`。starts/ends 使用该数组的**全局零基位置**，不是自动叠加 product_id 的局部位置；区间为 `[start,end)`。二者必须是等长、连续的一维 int64。可选 product_ids 同形 int64，只提供产品分组/计划信息，不替调用方阻止跨产品区间。

## 4. 最小直接调用

以下所有 Python 代码块均可在已安装当前原生包的环境独立运行。

```python
import numpy as np
from calmetrics_engine import operators as op

x = np.array([1., 2., 3.])
assert op.mean(x) == 2.0
out = np.empty_like(x)
returned, audit = op.add(x, 1.0, out=out, audit=True)
assert returned is out
np.testing.assert_array_equal(out, [2., 3., 4.])
assert audit["input_copy_bytes"] == 0
```

out 必须 shape/dtype 正确、连续、可写且不与输入重叠；异常后不能继续使用部分写入的 out。Workspace 只能供一个正在运行的调用独占。详细规则见[直接算子接口](canonical-operators.md)。

## 5. 声明参数并运行共享 DAG

compile 接受一个公式、根公式序列，或 `{输出名: 公式}` 的具名多输出。参数和数值常量不作为时序输入传入；根的顺序就是结果列顺序。公共 DSL 不允许任意 Python callback、属性、导入、列表推导等。直接 API 可以用关键字；公式中的 canonical 调用使用位置参数。

```python
import numpy as np
from calmetrics_engine import GraphCompiler, AdaptiveScheduler

graph = GraphCompiler({"x": "series", "scale": "scalar"}).compile(
    ["mean(x)*scale", "std(x,0)"])
x = np.array([1., 2., 3., 10., 20., 30.])
starts, ends = np.array([0, 3], np.int64), np.array([3, 6], np.int64)
with AdaptiveScheduler(cpu_budget=2) as scheduler:
    result = scheduler.execute(graph, {"x": x}, starts, ends,
                               parameters={"scale": 2.0})
np.testing.assert_allclose(result.values[:, 0], [4., 40.])
assert result.values.shape == (2, 2)
assert result.offsets is None
assert result.statuses is None  # 默认 raise 策略
```

parameters 也可用连续的一维 float64 数组，顺序依 graph.parameter_names；mapping 更容易审阅。名称/数量必须精确匹配。表达式共享的依赖由 C++ CSE/融合处理，不要为了每个输出分别执行整张图。

### 时序结果

```python
import numpy as np
from calmetrics_engine import GraphCompiler, AdaptiveScheduler

x = np.array([1., 2., 3., 10., 20., 30.])
graph = GraphCompiler({"x": "series"}).compile("rolling_mean(x,2)")
with AdaptiveScheduler(cpu_budget=1) as scheduler:
    result = scheduler.execute(graph, {"x": x},
        np.array([0, 3], np.int64), np.array([3, 6], np.int64))
np.testing.assert_allclose(result.values[:, 0],
    [np.nan, 1.5, 2.5, np.nan, 15., 25.], equal_nan=True)
np.testing.assert_array_equal(result.offsets, [0, 3, 6])
```

| 结果 | values | offsets |
| --- | --- | --- |
| 标量根 | `(区间数, 根数)`，float64（包括标量比较的数值输出） | None |
| 时序根 | `(所有区间长度之和, 根数)`，连续数组 | int64 前缀和，长度为区间数+1 |

默认 `result_format="auto"` 对纯标量和同 dtype、同 time/T 轴的时序保留上表接口；向量、矩阵、张量、裁短序列，以及混合 rank/dtype 的根自动返回 `output_kind="typed"`。可显式指定 `result_format="typed"`，统一使用下面的输出接口。记录、逻辑窗口和带内部状态标签的矩阵仍须先投影字段；本轮没有改变数学算子的返回 dtype（例如 distinct_count 仍返回 float64）。

### 不同形状和 dtype 的多输出

```python
import numpy as np
from calmetrics_engine import GraphCompiler, AdaptiveScheduler

x = np.arange(12., dtype=np.float64).reshape(4, 3)
graph = GraphCompiler({"x": {"kind": "matrix"}}).compile(
    ["mean(sum_asset(x))", "mean_time(x)", "covariance(x)",
     "sum_asset(x)>0", "argsort(sum_asset(x))"], error_policy="isolate")
with AdaptiveScheduler(cpu_budget=1) as scheduler:
    result = scheduler.execute(graph, {"x": x},
        np.array([0, 1], np.int64), np.array([4, 3], np.int64))
assert result.output_kind == "typed"
outputs = result.outputs
assert [o.values[0].shape for o in outputs] == [(), (3,), (3, 3), (4,), (4,)]
assert outputs[3].values[0].dtype == np.bool_
assert outputs[4].values[0].dtype == np.int64
assert outputs[3].values[1].shape == (2,)
assert all(np.all(o.root_statuses == 0) for o in outputs)
```

`outputs[根序号].values[区间序号]` 是该根的只读 ndarray；标量为零维数组，实际形状由 C++ 执行产生。各根保留 `kind`、`dtype`、`axes`；`byte_offsets` 是底层缓冲区中的字节位置，不是日期或索引映射。时序/向量的轴标签也不等于提供了真实日期/资产标签，标签由调用方维护。

- isolate 下 `statuses[区间]` 与对应值同形，`root_statuses[区间]` 汇总该根的状态（取最大错误码，0 表示全部成功）；严格模式的逐元素 statuses 为 None，普通 NaN 仍按算子契约处理。
- `shape_known[区间] == False` 表示上游失败使实际形状不可知：值和状态以空数组占位，必须读取非零 root_status。已知空数组则保留真实 rank/shape 和成功状态，二者不能混淆。
- typed 结果不提供统一 `values/statuses` 数组，`offsets` 为 None。一次执行仍共用一个 DAG；不要为不同输出拆图重复计算。
- `lag(x,k)`、`block_apply` 等动态输出可以公开返回，形状在每次执行中确定。需要保持输入时间长度时仍使用 aligned_shift。

### 矩阵与静态向量

```python
import numpy as np
from calmetrics_engine import GraphCompiler, AdaptiveScheduler

graph = GraphCompiler({
    "x": {"kind": "matrix", "dtype": "float64", "axes": ["time", "asset"], "shape": ["T", "N"]},
    "w": {"kind": "vector", "dtype": "float64", "axes": ["asset"], "shape": ["N"]},
}).compile(["matvec(x,w)", "sum_asset(x)"])
x, w = np.array([[1., 2.], [3., 4.], [5., 6.]]), np.array([.25, .75])
with AdaptiveScheduler(cpu_budget=1) as scheduler:
    result = scheduler.execute(graph, {"x": x, "w": w},
        np.array([0], np.int64), np.array([3], np.int64))
np.testing.assert_allclose(result.values, [[1.75, 3.], [3.75, 7.], [5.75, 11.]])
```

直接 `op.matmul(A,B)` 和图中的矩阵根均可返回矩阵。图的矩阵/静态向量根使用 typed 多输出接口，按输入名义轴决定是否随区间切片。

## 6. 窗口、分组和求根作用域

这些由编译器拥有，不计入 146 个 canonical 算子，不通过 Python 回调计算 body。

| 结构 | 数学行为 |
| --- | --- |
| rolling_window(x,w) | 编译器逻辑窗口，支持相应归约 lowering，不物化 T×W 输入 |
| rolling_apply(body,w[,min_periods]) | 对尾随窗口执行 float64 标量 body；默认完整有限窗口，前 w−1 行 NaN；显式 min_periods 检查联合有限观察数，不压缩原窗口；每窗重新初始化递推状态 |
| block_apply(body,w) | 按完整不重叠块计算；丢弃不足 w 的尾部；输出紧凑块序列，须继续归约等操作后输出 |
| filter_apply(body,mask[,empty_default]) | 保留选中行的原顺序，计算标量；空集默认 NaN；empty_default 不捕获 body 异常 |
| group_apply(body,keys) | 精确 int64 分组，标量结果广播回原组位置 |
| bisect(body,lower,upper,tolerance,max_iterations) | 对局部标量 solve_x 求零点；检查夹根、有限性和有界收敛 |
| iterate(body,initial,tolerance,max_iterations) | 原生有界反馈：body 通过 iterate_x 访问上一轮 float64 状态，保留 rank/shape/轴；残差为最大绝对步长，达到 tolerance 停止 |
| segment_apply(body,boundaries) | 对完整事件段的两端 `[left,right]` 计算标量，广播到 `[left,right)`；未闭合段保持缺失 |

作用域重排可能改变结果，例如先过滤再分块与先分块再过滤不同。动态/静态 captures 见[数学组合设计](mathematical-composition-design.md)。当前 rolling_apply body 必须含区间聚合并依赖数值时序；不能嵌套 rolling_apply、滚动族算子或 recursive_smooth，也没有一个开关可以绕过该限制。

另有 `rolling_apply(body,w,dates,annual[,min_periods])` 上下文形式。`observation_dates`、`annual_risk_free_rate_decimal`、`returns`、`log_returns`、`observation_count` 及 L 形状在该兼容窗口上下文中有专门绑定含义，不能随意重命名/复用为其他数据。普通数学图不需要这些系统字段；集成该上下文前阅读[窗口设计](typed-dsl-rolling-cpp-design-2026-09-20.md#5-rolling_apply)。

```python
import numpy as np
from calmetrics_engine import GraphCompiler, AdaptiveScheduler

graph = GraphCompiler({"x": "series"}).compile([
    "mean(block_apply(sum(x),2))",
    "filter_apply(mean(x),x>2)",
    "bisect(solve_x*solve_x-2,0,2,1e-10,100)",
])
with AdaptiveScheduler(cpu_budget=1) as scheduler:
    result = scheduler.execute(graph, {"x": np.array([1., 2., 3., 4.])},
        np.array([0], np.int64), np.array([4], np.int64))
np.testing.assert_allclose(result.values, [[5., 3.5, np.sqrt(2)]], atol=1e-9)
```

## 7. 递推和整数状态

递推状态属于本次区间/作用域，不是跨请求持久化状态。prepared 复用工作缓冲不意味着第二次请求承接第一次末尾状态。固定 alpha、动态 alpha、初态、更新掩码、重置掩码、预热和缺失输出策略分别有契约。

```python
import numpy as np
from calmetrics_engine import GraphCompiler, AdaptiveScheduler

x = np.array([1., 3., 5.])
filtered = GraphCompiler({"x": "series"}).compile(
    "recursive_filter(x,0.5,0,finite_mask(x),2,0)")
states = GraphCompiler({"x": "series"}).compile(
    "state_select(x>2,0,1,finite_mask(x))")
with AdaptiveScheduler(cpu_budget=1) as scheduler:
    start, end = np.array([0], np.int64), np.array([3], np.int64)
    a = scheduler.execute(filtered, {"x": x}, start, end)
    b = scheduler.execute(states, {"x": x}, start, end)
np.testing.assert_allclose(a.values[:, 0], [1., 2., 3.5])
np.testing.assert_array_equal(b.values[:, 0], [1, 0, 0])
assert b.values.dtype == np.int64
```

状态 -1、事件 -1、无事件 0、有效 False 和执行失败占位不是同一个含义。编码与等号详见[状态事件契约](state-event-contracts.md)。峰谷历史修订、PS 和完整波段需要未来/全样本数据；普通比较不能消除该依赖。引擎局部 temporal_dependency 元数据不代替调用方全图时点认证。

## 8. 错误、缺失和结果有效性

默认 error_policy="raise"：未被作用域兼容规则处理的异常使请求失败；普通 NaN 是否产生异常取决于算子，不等于全局禁止 NaN。普通 rolling_apply（无分段故障上下文）仍保留历史行为：窗口 body 的数值异常转为该窗口的 NaN，因此不能把 raise 理解为任何窗口错误都会抛出；严格 group_apply 的数值异常则继续抛出。需要逐位置故障状态时，应显式选择 isolate。

可在 compile 时选 isolate，保留健康根/区间，并返回与值同形的只读 int16 statuses；typed 结果在各 output 下提供状态及根状态。0 表示成功；非零原因见[平台执行契约](platform-execution-contracts.md#isolation)。dtype、非法几何、资源与进程故障不能变成成功的部分结果。

```python
import numpy as np
from calmetrics_engine import GraphCompiler, AdaptiveScheduler

graph = GraphCompiler({"x": "series"}).compile(
    ["divide(mean(x),0)", "mean(x)"], error_policy="isolate")
with AdaptiveScheduler(cpu_budget=1) as scheduler:
    result = scheduler.execute(graph, {"x": np.array([1., 2., 3.])},
        np.array([0], np.int64), np.array([3], np.int64))
assert np.isnan(result.values[0, 0])
assert result.values[0, 1] == 2.0
np.testing.assert_array_equal(result.statuses, [[2, 0]])
```

isolate 下非有限 float64 输出位置会被标记为不可用。bool/int64 失败输出用 False/0 占位，必须同时查看 statuses，不能将占位理解为有效信号。实际 rolling_apply、group_apply 和 segment_apply 数值故障会保留对应窗口/组/段的位置状态，并沿依赖传播；暂不能精确映射位置的非局部算子保守使依赖节点失败，独立输出保留。未选分支并不是免于编译/结构校验的 Python 短路分支。

## 9. 重复执行与结果所有权

当前 prepare_execution **只接受 single-lane 计划**，自动规划为 thread/process 或传入该类计划都会拒绝。它不是所有并行任务的通用替代入口；并行任务继续使用 execute。

| 方法 | 结果寿命 | 使用场景 |
| --- | --- | --- |
| scheduler.execute | 独立结果 | 普通请求，可保留 |
| prepared.run | 借用复用输出，下一次运行覆盖 | 旧格式返回 ndarray；typed 格式返回包含 outputs/statuses 的 Result |
| prepared.run_audit | values 借用至下一次运行，附 statuses/audit | 即时消费并审计 |
| prepared.run_snapshot | 独立只读结果与状态 | 缓存、历史结果、跨后续调用保存 |

```python
import numpy as np
from calmetrics_engine import GraphCompiler, AdaptiveScheduler

x = np.array([1., 2., 3.])
graph = GraphCompiler({"x": "series"}).compile("mean(x)")
with AdaptiveScheduler(cpu_budget=1) as scheduler:
    prepared = scheduler.prepare_execution(graph, {"x": x},
        np.array([0], np.int64), np.array([3], np.int64))
    retained = prepared.run_snapshot()
    x[:] = 6.0  # 上次计算已完成；只修改原缓冲内容
    current = prepared.run_audit()
    assert current.values[0, 0] == 6.0
assert retained.values[0, 0] == 2.0
assert not retained.values.flags.writeable
```

同 dtype 的 bool/int64 旧格式在 prepared.run 遇到失败时要求改用 run_audit/run_snapshot，避免裸数组隐藏状态。typed 的借用值、实际形状和状态只对应本轮执行；下次运行后不得继续把旧描述与已覆盖的值配对使用。NumPy 的只读标记防止误写，不是安全隔离；禁止修改底层 base 的 dtype/shape/存储。

数据、区间或计划身份不兼容时重新绑定/规划；不要绕过 stale/prepared 检查。共享输出即使底层来自原生映射也会保活 owner；只读标志本身不能证明结果独立。

普通 execute 会在复核当前布局、区间内容和请求策略后复用兼容计划；几何变化时自动重建。显式传入不兼容计划、或改变 prepared 的绑定几何仍报错。计划复用不影响每次结果的独立所有权。

## 10. 常见问题

| 现象 | 首先检查 |
| --- | --- |
| exact native dtype / C-contiguous | 是否误传 float32、list 或图 float64 的非连续 series |
| AXIS/SHAPE_MISMATCH | series 与 vector 是否混用、T/N 是否一致、是否误用任意广播 |
| PUBLIC_ROOT_TYPE | 记录、逻辑窗口、内部状态矩阵须投影为公开类型 |
| typed result: use outputs | 用 outputs[根].values[区间]；不要强转统一数组 |
| RESULT_CAPACITY_MISMATCH | 实际结果超出计划容量；保留图与输入几何报告问题，禁止绕过校验 |
| SEMANTIC_DIMENSION / PRICE_BASIS_MISMATCH | 检查实际声明和数据口径；不要为消除报错伪造标签 |
| 结果被下一次调用改写 | 是否使用了 prepared 借用输出；需要留存用 run_snapshot |
| 没有多线程或 SIMD | 先查看计划与实际审计，不以数组大/节点多直接推断，见执行指南 |

公开能力和解释应与实际安装包的 catalog、graph.metadata()、plan.metadata() 和 result.audit 对照。数学正确性、执行性能与金融业务有效性是三种不同验收。


## 11. 开发分支新增：具名结果与模型载荷

本节对应 M0/M1 开发代码，性能验收状态见 [开发设计](platform-foundation-design.md)，不表示已发布或已超过 NJIT。具名输出自动使用 typed 结果；`result.named_outputs[名称]` 与 `result.outputs[序号]` 指向同一份结果数据。零维 value 输入保留精确 dtype，包括超过 2^53 的 int64；普通 scalar 参数继续使用 float64。

```python
import numpy as np
from calmetrics_engine import GraphCompiler, AdaptiveScheduler, ModelPayload

schema = {"x": {"kind": "tensor", "axes": ["scenario", "path", "asset"],
                "shape": ["S", "P", "N"]}}
graph = GraphCompiler(schema).compile({"values": "x*2+1", "mask": "x>2"})
metadata = {"schema": "example-1", "algorithm": "affine", "algorithm_version": "1",
            "source_identity": graph.fingerprint, "training_options": "{}", "random_version": "none"}
with AdaptiveScheduler(cpu_budget=1) as engine:
    result = engine.execute(graph, {"x": np.arange(24.).reshape(2, 3, 4)},
                            np.array([0], np.int64), np.array([1], np.int64))
    model = result.to_model_payload(metadata)
assert result.named_outputs["values"].values[0].shape == (2, 3, 4)
restored = ModelPayload.from_bytes(model.to_bytes())
assert restored.identity == model.identity
assert not restored.fields["values"].flags.writeable
```

`ModelPayload.from_fields(types, fields, metadata)` 也可从数组建立载荷。最多 4096 个具名字段；字段为静态 rank-0/1/2/3 数组，不接受 time 轴、逻辑窗口或内部记录。建立时复制一次到物理只读共享存储；字段 NumPy 视图保活 owner。可用 `GraphCompiler(model.types)` 编译并将 model 直接作为 inputs；输入名必须完全匹配。混合外部数组时可使用 `model.fields` 的视图，但普通 dict 输入不携带完整模型身份。

六项示例 metadata 均必填非空字符串。身份覆盖字段声明顺序、精确 dtype/shape/字节以及全部 metadata，属于内容标识，不是认证签名。`to_bytes/from_bytes` 使用版本化 little-endian 协议，不使用 pickle；解码会分配新的独立载荷。算法版本、训练选项和随机版本的含义由调用方负责。失败结果和时间序列不能直接导出为模型。载荷改变后需重新规划；旧模型、旧结果不会被覆写。

## 12. 开发分支新增：有界迭代与诊断

```python
import numpy as np
from calmetrics_engine import GraphCompiler, AdaptiveScheduler

expr = "iterate(iterate_x*0.5,x,1e-6,100)"
graph = GraphCompiler({"x": {"kind": "vector", "axes": ["asset"], "shape": ["N"]}}).compile({
    "value": expr, "status": f"iteration_status({expr})",
    "count": f"iteration_count({expr})", "residual": f"iteration_residual({expr})"})
with AdaptiveScheduler(cpu_budget=1) as engine:
    result = engine.execute(graph, {"x": np.full(8, 8.)},
                            np.array([0], np.int64), np.array([1], np.int64))
assert result.named_outputs["status"].values[0] == 0
assert result.named_outputs["count"].values[0] == 23
```

同一 iterate 表达式经 CSE 只执行一次。tolerance 必须有限且 >=0，max_iterations 为 1–10000 的整数值 float64。更新前后状态的 dtype、轴、形状和已有语义声明必须一致；更新可以捕获不同形状的数组，不能传 Python callable。每次调用从 initial 重新开始，不自动承接上次请求。

status/count 为 int64 标量，residual 为 float64。当前产生 0（步长收敛）、1（迭代上限）、5（数值失败）；2/3/4 为后续求解内核保留的无可行起点/已证不可行/局部停止状态，不能据此认定已有这些求解能力。步长收敛不证明最优或可行。数值更新失败时保留最后有限状态；初态本身非有限时没有有效候选，状态为 5。残差此时为 Inf，在 isolate 结果中对应残差根不可用。

结构错误、预算超限、取消和超时仍为执行错误；独立根的隔离行为遵循原契约。每轮复用原生子图和状态缓冲，使用同一 CPU/内存准入，不创建额外线程池。该通用接口尚不等于 QP、SQP、EM 或具体模型算法的实现。

执行层可将逐元素链按缓存块融合，短标量广播链进一步使用 AOT 寄存器融合，并让 typed 根直接写本次独占输出。可证明始终有限的短迭代链复用独占结果缓冲，一般子图保留候选状态以支持失败恢复。这些优化不改变逻辑 DAG、dtype、轴、错误策略或 prepared 的借用／快照契约。只读输入与先前的独立结果均不作为工作缓冲。调度与复制审计见[执行指南](execution-guide.md#9-本轮物理执行优化与审计)。
