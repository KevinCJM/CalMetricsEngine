# CalMetricsEngine 0.3.0 Verification

## 已验证

- CMake native build：通过。
- standalone C++ tests：通过。
- Python 3.12 wheel build：通过。
- wheel / sdist 内容检查：通过。
- Twine metadata check：通过。
- 安装真实 wheel 后 pytest：220 passed。
- CPython 3.10 / 3.11 / 3.12 / 3.13 / 3.14：各 220 passed。
- NumPy 1.26.4 + Python 3.12：220 passed。
- standalone C++ ASan / UBSan：通过。
- Ruff lint / format：通过。
- 历史金融数值 golden regression：包含在 pytest 全量测试中并通过。
- strided float64 输入：C/F order、行/列 slice、负 stride 均通过。
- readonly 输入：通过。
- strided int32/int64 输入：通过。
- datetime64[ns] → int64 view：零拷贝路径通过。
- wrong dtype / list / unaligned input：明确失败，不发生隐式复制。

## 架构事实

0.3.0 将项目统一命名为 CalMetricsEngine：

- Distribution：`calmetrics-engine`
- Python import：`calmetrics_engine`
- Native extension：`calmetrics_engine._native`
- C++ namespace：`calmetrics_engine`
- CMake project/targets/options：统一使用 `calmetrics_engine` / `CALMETRICS_ENGINE_*`

输入内存契约从“必要时 normalize/copy 成 C contiguous”改为：

```text
NumPy owner
  ↓
exact dtype + shape + strides
  ↓
PyBind11
  ↓
C++ ArrayView
```

C++ 内核直接按 stride 读取原数组；计算输入不物化副本。

## 尚未宣称完成

- Linux / Windows / Intel macOS 的 0.3.0 CI 尚需真实远端 workflow 结果确认。
- FundInvestmentResearchPlatform 的 Typed DSL / AST / DAG 尚未迁入本仓库。
- NativeExecutionPlan、operator lowering、workspace liveness planner 属于下一阶段。
- PyPI 尚未发布 0.3.0。
