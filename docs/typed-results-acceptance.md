# Typed 多输出结果协议验收

验收日期：2026-09-23。范围仅 CalMetricsEngine；分支 `codex/typed-results`，工作树 `/private/tmp/calmetrics-stateful-series`，基线 `e88f69a953f43ef60d340ca1e72a21f6e2dc30d7`。下文保留初验快照；提交前复审修复和最新验证见末节。远端 CI/Bot 结论以 PR 当前 HEAD 的真实记录为准。

## 初验结果

- scalar/series/vector/matrix 可以在同一个共享 DAG 返回不同 shape/dtype；原有兼容接口保留。
- C++ AST/Typed IR → DAG → Planner 容量 → 原生执行 → worker IPC → NumPy owner 贯通。新增逻辑与数值路径在 C++；Python 仅导出类型和提供验证/测量脚本。
- 每根返回实际形状、逐元素状态及根状态；空结果、未知几何失败、有效 0/False 有明确区别。
- 单线程、多线程、inline/shared worker、Hard Stop、async、prepared 借用/快照均有实际执行覆盖。内置数学算子仍为146个，未修改金融口径或新增指标黑盒。
- 自审核未留下已确认的阻断项；功能测试及原性能门禁通过。此结论是本机引擎验收，不是上层平台的新协议接入验收。

设计见 [typed-results-design.md](typed-results-design.md)，调用方式见 [使用手册](user-guide.md)。

## 初验验证证据

环境：macOS-15.6.1-arm64-arm-64bit，Python 3.12.11，AppleClang 17.0.0.17000603。

| 检查 | 结果 |
| --- | --- |
| 安装后 wheel 全量 Python | 4692 passed，0 failed，0 skipped |
| 新增 typed 结果回归 | 73 项，见 `tests/test_typed_results.py` |
| Release 原生 CTest | 8/8，通过新增 `result_native_tests.cpp` |
| ASan + UBSan 原生 CTest | 8/8 |
| 使用手册 Python 示例 | 9/9 |
| diff 空白检查、改动 Markdown 本地链接 | 通过 |
| 源码与安装 wheel 构建身份 | 一致 |

- 原生 build ID：`4652adc5d6aefda02156516d2fd4688d0af08992708d4c4de76520eb07cc3912`。
- wheel SHA-256：`edb047ec7ca0d3afe74fa5a60ed303424f59149ef4619c82f8765969d09a6ef2`。
- 包：`calmetrics_engine-0.3.0-cp312-cp312-macosx_15_0_arm64.whl`，安装验证目录 `/private/tmp/calmetrics-typed-install`。
- 身份校验按 CMake 的源码文件清单、工具链和编译参数重新计算 SHA-256，再与真实执行返回的 engine_build_id 比较，不只比较 Git SHA。

## 自审核发现与修正

1. **整数根兼容**：`is_numeric()` 仅涵盖 float64，首次扩展根校验误拒绝已有 int64 状态时序。补齐整数判断，保持内部记录必须投影的限制，旧时序用例和新混合根均通过。
2. **prepared 所有权**：只隐藏公开 output 不足以阻止调用方追溯 NumPy base。补齐缓冲地址、dtype、shape、布局和可写性检查；明确借用描述和值仅属于本轮执行，持久保存使用 snapshot。
3. **完整状态码**：worker 校验必须允许原有 7/9/10，不能只接受0–4；新增新旧格式在四通道的逐项回归。
4. **内存预算**：描述槽分配前检查预算；typed inline IPC 同时计入父输出、worker 输出、发送帧、接收帧和解码 payload。容量/对齐算术检查溢出，混合 payload 使用字节拷贝避免不同原生类型别名写入。
5. **进程响应**：对真实 worker 返回包注入错误版本、根数、rank、超容量 shape、状态、dtype/kind 和截断 payload，确认拒绝时未复制到目标输出；v5 旧计划仍可解码执行。

新增覆盖还包括只读/负步长矩阵、精确 int64 极值、0维标量、0轴矩阵、0区间、动态 lag/block、分段失败跨 bool/int64 传播、独立健康根、未知动态长度、超时后恢复，以及释放 result/scheduler/shared bundle 后保留视图。

## 性能

最终 wheel 执行原 `tools/check_phase2_performance.py`，负载与阈值未修改：prepared/batch ≤0.90，普通 Scheduler ≤1.00。较早一轮也为4/4通过；下表取最后内存预算修正后的最终结果。

| 负载 | Prepared/批量 C++ ÷ NJIT | 普通 Scheduler ÷ NJIT | 结果 |
| --- | --- | --- | --- |
| 500×2520×12×16 | 0.499 | — | 通过 |
| 1000×2520×12×16 | 0.536 | — | 通过 |
| 1×1×5 / 63 点 | 0.609 | 0.940 | 通过 |
| 1×1×5 / 252 点 | 0.707 | 0.878 | 通过 |

新 typed 负载：64个区间，每区间252×8，五个输出为资产均值向量、协方差矩阵、转置矩阵、bool序列、排序int64向量；7轮交替测量，共1,214,464字节结果容量。计时包含公开 NumPy 视图和状态物化，比较同一共享 DAG 与逐根分别执行；两者均使用相同数值定义并检查 NumPy 参考结果。

| 通道 | 共享 DAG 中位 ms | 分图中位 ms | 比值 | 最终输出复制 bytes |
| --- | --- | --- | --- | --- |
| process_inline | 9.239 | 24.805 | 0.372 | 1214464 |
| process_shared | 2.982 | 4.391 | 0.679 | 0 |
| single | 4.627 | 4.558 | 1.015 | 0 |
| thread | 2.351 | 2.829 | 0.831 | 0 |

该新负载没有历史性能门槛。单线程共享 DAG 此次略慢于分图；不宣称所有场景都加速。共享内存没有最终 payload 回拷，状态/描述信息传输及输入规范化的必要复制仍单独存在。调用方可缓存一次 `outputs = result.outputs`，避免重复构造包装和状态数组。

## 复现

```sh
CMAKE_GENERATOR='Unix Makefiles' python -m pip wheel . --no-build-isolation --no-deps -w /tmp/cme-typed-wheels
python -m pip install --no-deps --target /tmp/cme-typed-installed /tmp/cme-typed-wheels/*.whl
PYTHONPATH=/tmp/cme-typed-installed python -m pytest tests -q
cmake -S . -B /tmp/cme-typed-native -DCMAKE_BUILD_TYPE=Release -DCALMETRICS_ENGINE_BUILD_PYTHON=OFF -DCALMETRICS_ENGINE_BUILD_TESTS=ON
cmake --build /tmp/cme-typed-native --parallel 4
ctest --test-dir /tmp/cme-typed-native --output-on-failure
cmake -S . -B /tmp/cme-typed-asan -DCMAKE_BUILD_TYPE=Debug -DCALMETRICS_ENGINE_BUILD_PYTHON=OFF -DCALMETRICS_ENGINE_BUILD_TESTS=ON -DCALMETRICS_ENGINE_SANITIZE=ON
cmake --build /tmp/cme-typed-asan --parallel 4
ctest --test-dir /tmp/cme-typed-asan --output-on-failure
PYTHONPATH=/tmp/cme-typed-installed python tools/check_phase2_performance.py --better-root /path/to/BetterSaaTaa --output-dir /tmp/cme-typed-performance
PYTHONPATH=/tmp/cme-typed-installed python tools/benchmark_typed_results.py --lane process_shared --output /tmp/cme-typed-performance/typed-shared.json
```

本次日志与JSON在 `/private/tmp/calmetrics-typed-full.log`、`/private/tmp/calmetrics-typed-ctest.log`、`/private/tmp/calmetrics-typed-sanitize-ctest.log`、`/private/tmp/calmetrics-typed-performance-final/`。POSIX共享内存测试需要允许创建共享区域，受限沙箱中失败不能替代这些实际执行证据。

## 边界

公开数值结果限rank0/1/2；record/window/内部状态矩阵继续先投影。当前算子返回dtype保持原契约，float64计数不会因typed格式改成int64。typed使用区间/产品并行，暂不新增typed DAG分支并行；prepared仍只支持single lane。

协议为 program v6、worker v5、`typed-results-1`，typed凭据为 `cpp-aot-execution-2`。需要调用方显式消费每根类型、shape/status及所有权；FundInvestmentResearchPlatform 的协议适配、快照持久化与发布门禁未在本轮修改。此次未验证其他操作系统/架构的wheel或生产部署。

## 提交前复审修复（2026-09-23）

确认并修复 P2：小输入可能生成超过 IPC 单帧限制的大矩阵，原 Planner 仅看输入大小，导致计算结束后响应序列化失败。现在按实际分块检查完整响应，计入 payload、逐元素状态、shape、根状态与协议头，超限自动使用共享输出；内存预算降低进程数后重新分块检查并估算内存。共享路径的状态与形状仍经 IPC，超限在规划阶段明确拒绝并要求缩小区间批次。保留 256 MiB 单帧上限。

- 新增 13 项 Python 边界回归：小输入/大输出、数值恰好 256 MiB 但含头超限、状态触发超限、分块仍可 inline、预算减少进程数后切换运输与预算边界、共享状态超限早期拒绝。
- 原生测试用真实 worker 响应核对完整大小估计，检查精确帧界限、状态/形状开销和整数溢出。
- 原复现实际执行成功：输入 32,800 字节，两个 4100×4100 矩阵、结果 268,960,000 字节；自动 shared，估计总内存 672,467,244 字节，最终结果复制 0，逐值等于 1。
- 已安装 wheel 全量 Python **4705 passed**，typed 文件 **86 项**；Release 原生和 ASan/UBSan 均 **8/8**。Ruff 0.16.8 全量检查与 `git diff --check` 通过。
- 修正本轮新增导入的排序、失效导入和基准脚本未使用变量；不修改数值路径或既有基准阈值。
- 当前源码与安装 wheel build ID 一致：`72215d1a6b6f6fde75be073ceb0591b20dfb404c96fac2cc069982d3fc0b9cd1`。
- 当前 wheel SHA-256：`3c934f3fda76f68f38b9ad45656384465323aad46bf4d39360ce50f21d75a7f1`。

四组原性能门禁重新执行通过：500产品 0.724、1000产品 0.654；63点 prepared 0.634 / Scheduler 0.973；252点 prepared 0.717 / Scheduler 0.887。均为 C++/NJIT 耗时比，阈值仍为 prepared/batch ≤0.90、Scheduler ≤1.00。日志和结构化结果在 `/private/tmp/calmetrics-typed-fix-performance/`。初验中的新 typed 负载测量为当时快照，本节不把它重标成此次复测。
