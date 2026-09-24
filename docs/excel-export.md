# Excel 公式导出使用说明

[文档导航](README.md) · [数学算子](operator-reference.md) · [复现契约与验收](excel-reproduction-design.md#10-原生实现与验收)

`calmetrics_engine.excel` 是按需调用的导出模块。C++ 根据现有逻辑 DAG 展开公式，Python 仅流式写 xlsx。普通 `operators`、Scheduler、prepared、线程和进程计算不运行此模块。当前 146 个算子和 7 类作用域已通过 Microsoft Excel 16.89.1 的全量夹具验收（391 组、2,533 个结果）；具体构建、版本和适用域见复现契约。新生成文件仍须单独重算，不能把生成成功当作本次结果已验证。

## 1. 安装与调用

安装同源 wheel 及可选依赖：`pip install 'calmetrics-engine[excel]'`。生成文件不要求安装 Excel。以下例子可独立执行：

```python
from pathlib import Path
import tempfile
import numpy as np
from calmetrics_engine import GraphCompiler, excel

graph = GraphCompiler({"x": "series"}).compile({
    "average": "mean(x)",
    "double_average": "mean(x)*2",
    "path": "cumulative_sum(x)",
    "positive": "x>0",
}, result_format="typed", error_policy="isolate")
plan = excel.plan(graph, {"x": np.array([1., 2., 3., 4.])})
with tempfile.TemporaryDirectory() as directory:
    report = excel.export(plan, Path(directory) / "calculation.xlsx")
    assert report["verification"] == "NOT_RECALCULATED"
    assert [r["shape"] for r in report["outputs"]] == [[], [], [4], [4]]
    assert report["cells"] <= 2_000_000
```

同一 `mean(x)` 只展开一次。多输出仍保留各自 dtype、shape、axes 和地址，矩阵/张量按行优先线性展开；`Outputs` 表说明可逆映射。当前一次计划绑定一个完整输入范围，需要多产品/区间时由调用方显式选择范围并分别导出，不能把拼接的不同产品当成一条递推序列。

直接算子入口是 `excel.plan_operator("solve", [matrix, rhs])`。146 个 canonical 算子都登记了展开实现，完整名称与参数沿用算子参考。fit/interval 直接导出包含全部字段；图内仍使用正式投影算子。`coverage()` 是不绑定 Excel 版本或输入的静态实现目录，保持 `implemented_unverified`；特定构建/版本的验收以独立 PASS 报告为准。未来新增 opcode 不会自动标为实现。

## 2. 参数、快照与预算

`plan(graph, inputs, parameters={}, *, max_cells=2_000_000, max_formula_characters=134_217_728, max_snapshot_bytes=67_108_864, atol=1e-12, rtol=1e-10, timeout_seconds=180)`。

`plan_operator(name, arguments, *, max_cells=2_000_000, atol=1e-12, rtol=1e-10, timeout_seconds=180)`。`max_cells` 可以调低，不能超过 200 万；这些是资源上限，不是重算耗时保证。

- 计划复制所用输入并深拷贝子图。构造期间调用方不得通过其他线程或进程修改输入；构造完成后调用方可释放或修改原数组。stride 视图在此边界复制成连续快照，不声称导出零拷贝。
- graph 的名称、dtype、rank、固定/符号维度须精确匹配。Excel 导出允许合法 stride 的时序快照；这是独立导出接口，不放宽普通 Scheduler 的输入契约。
- 输入、参数、快照副本、节点说明、全部辅助公式、结果和核验列一起计入预算。超限抛 `ExcelExportError`，不会缩短窗口、减少迭代或丢弃输出。分表不能增加总预算。
- 先检查输入单元格下界与快照字节，再由独立 C++ 符号规划器根据 dtype、shape、配置标量及作用域次数推导整个工作簿上界；此阶段不生成公式、地址或逐元素引用。准入后再用有界计数遍历确定确切布局/字符数，展开和写出不得超过上界。单次公式字符串最长 8,192 字符，括号嵌套保守限制 64 层，作用域深度 32，布局类参数存在有限上界。
- 工作量估计上限 1 亿、原生 JSONL 与 xlsx 各限制 512 MiB，私有临时目录按批检查合计 1 GiB。检查间隔内存在一个写出批次的余量；大规模并发任务仍应由调用方设置目录配额和外层任务预算。
- `plan.cancel()` 使后续工作协作取消；展开及写出定期检查，失败清理临时文件，不替换已有目标。C++ 参考计算进入现有 Scheduler，取消和超时在参考内核前后检查，不承诺中途硬杀某个内核。硬隔离需求使用调用方任务边界，不能依赖此 API 的协作取消。

`metadata()` 同时返回精确布局计数和 `upper_bound`：单元格、公式单元格、公式字符、快照字节、工作量及成本最大的 10 个顶层节点（作用域含其子图）。公式字符上界采用公式单元格数 × 8,192，因此可能在实际字符数仍较小时保守拒绝；错误明确标记 `symbolic cell/character/work bound`，不能当成实测占用。可减少范围，或为图导出显式提高 `max_formula_characters` 后重新预检；精确布局仍检查实际字符数与单公式限制。其他字段包含输出映射、容差、公式版本、构建身份和计划身份。计划身份是确定性 FNV 标识，用于定位；不是安全签名。自动核验另保留原始 xlsx 与 SHA-256。

## 3. 支持域与展开方式

- 标量、同形数组和契约允许的标量广播保留原语义；没有新增 NumPy 广播。矩阵求解展开带部分主元的消元步骤，不能用伪逆代替奇异失败。
- rolling/block/filter/group/segment/bisect/iterate 使用原生解析后的子图展开。窗口内部重新计算；递推状态逐行保留，迭代按配置最大次数展开并在停止后冻结。过滤/分组展开全部可能的成员数量，分段展开可能区间，因此某些短输入也会产生较多公式。
- period、window、最大迭代数等布局参数必须能从常量、显式标量参数或可静态传播表达式确定。任意数据归约决定布局而不能确定上界时返回 `UNKNOWN_BOUND`。用户在工作簿中修改这些参数后显示 `LAYOUT_STALE`，需要重新导出。
- bool、int64、NaN 和执行错误分别表示。Excel 数值运算只接受绝对值不超过 `999999999999999` 的 int64；不把大整数悄悄转 double。输入 Inf、次正规数拒绝。超出 Excel 有限算术范围的运算尚不提供完整 IEEE-754 模拟，不应据普通样本通过推断所有极值组合已支持。
- 为避免大偏移小方差被十进制输入截断，非整数 float64 通过可见的短整数/二进制幂公式重建。可直接修改 Inputs 的 B 列；C 列保留导出快照。此表达仅恢复输入常量，不读取 C++ 结果。
- `ERROR:…` 与文本 `NaN` 不等于零或空白。图的 isolate 状态遵守现有原生规则，包括逐位置故障和独立根。普通缺失和异常来源分别处理；原生对某些根的非有限结果归为状态 4 的行为也保留。

## 4. 查看、修改与核验

工作表均可见：`Readme`、`Definitions`、`InputsN`、`Nodes`、`StepsN`、`Outputs`、`ResultsN`。Nodes 提供上下文实例、逻辑节点、父节点、dtype/shape 和首末地址；计算步骤引用永久保留，不覆盖已经被引用的单元格。公式计算不依赖 Results 中的 C++ 参考列，不使用宏、Solver、Python in Excel、外部文件或网络。

Results 各列为：名称、Excel 公式值、冻结 C++ 参考值、绝对误差、比较状态。浮点采用导出前确定的 `atol + rtol*abs(reference)`；离散值与错误状态严格比较。默认容差适用于当前夹具，病态问题仍须由调用方按算法验收；不能事后扩大容差掩盖失败。

新文件的公式缓存为 `NOT_RECALCULATED`。在目标 Excel 中强制重算并保存后，才可读回结果。Readme 的证据标记保持导出时状态；表内 `MATCH` 只是计算比较，不能自行升级为服务器验收。编辑输入后显示 `REFERENCE_STALE`；编辑公式后属于用户修改版，应重新生成原生参考并核验。

## 5. 自动化检查

```bash
python -m pytest tests/test_excel_export.py tests/test_excel_parity_tool.py -q
python tools/check_excel_parity.py --engine libreoffice --all-cases --check-edits --output /tmp/excel-lo
```

`.github/workflows/excel.yml` 配置 Linux LibreOffice 差分及证据产物。该工作流尚未在远端运行。测试同时核对原始公式/输入身份、冻结 C++ 值、dtype、逐位置状态和表内比较，不能仅凭可编辑的 MATCH 标签通过。

Windows 安装真实桌面 Excel 和 `pywin32` 后使用 `--engine excel --check-edits`：测试创建独立 Excel 实例、禁用宏/链接更新、执行 `CalculateFullRebuild`、保存读取并关闭。编辑检查另改动数值和窗口参数，分别验证重新计算/旧参考失效和布局失效提示。普通 GitHub hosted runner 不假定装有 Excel。

macOS 可先 `--engine generate --check-edits`，在真实 Excel 强制重算并保存 `conformance.xlsx`，同时重算保存 `input-edits/value.xlsx` 与 `input-edits/layout.xlsx`，再用 `--engine read --check-edits --engine-version 'Microsoft Excel 实际版本'` 校验。`source.xlsx` 是原始身份基准，不能替换成重算文件。测试所需 `openpyxl` 只用于验收，不进入生产导出依赖。
