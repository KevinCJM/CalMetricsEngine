# 使用文档体系设计与验收

初稿日期：2026-09-21，初稿代码依据 `52b0f90`。提交前于 2026-09-22 按主线 `42fefed`（实现提交 `f84185b`）复核；`canonical-native-1` 仍为 146 个算子。两次验证分别记录如下。

## 1. 问题与目标

现有资料分散在 README、阶段设计和验收记录中。调用者需要理解六类值、算子组合、内存所有权以及原生调度策略，却容易读到早期 118/125 算子、float64/bool 类型或旧协议说明。

本轮建立面向调用者的阅读路径：准备数据 → 声明类型 → 构建图 → 执行 → 检查结果与凭据 → 按实际计划调优。每个 canonical 名称都提供可定位的签名、输入输出、数学职责和边界；复杂状态算法引用唯一详细契约，避免另写一套容易漂移的算法定义。

## 2. 范围与职责

本轮只修改 Markdown 文档，不修改数值实现、调度参数、金融语义校验、API 或测试。金融字段选择、来源、复权真实性、单位对齐和业务解释属于调用方；当前 Typed IR 对已声明 semantic_dimension/price_basis 的校验仍存在，手册必须描述这一事实，不能把职责讨论写成已经删除的能力。

文档中的通用例子默认使用无量纲数值。状态/event/phase 类型、shape、dtype 和递推边界仍是执行契约。既有具名复合算子及 cal_* 接口继续按当前数学口径说明，不改名、删除或重新解释。

## 3. 信息结构

| 文件 | 唯一主要职责 |
| --- | --- |
| README.md | 项目介绍、安装、短示例、统一入口 |
| docs/README.md | 按任务导航，区分当前说明与历史证据 |
| user-guide.md | 数据类型、数学轴、声明、DAG、作用域、结果与错误、生命周期 |
| operator-reference.md | 全量名称、稳定 opcode、真实签名、形状、算法与默认值 |
| execution-guide.md | SIMD、Planner 决策顺序、参数默认值、线程/进程/异步/共享内存与审计 |
| canonical-operators.md | 直接算子特有的 out、Workspace、视图及输入契约 |
| platform-execution-contracts.md | 平台绑定、状态和执行凭据协议 |
| architecture.md | 原生模块职责及实现链路 |
| 阶段设计/验收 | 保留历史范围和证据，在入口标明阅读边界 |
| AGENTS.md | 要求变更时同步用户文档；不复制全量算子表或调度规则 |

手册解释“怎样使用”，参考表解释“每个算子做什么”，执行指南解释“为什么采用该执行方式”。三者用链接复用详细契约，避免把同一状态机完整复制多次。

## 4. 编写规则

1. 区分直接算子、整图 API、现有 cal_* 财务 API，不把其中一个入口的 shape/dtype 契约套给另一个。
2. 区分 scalar、series、vector、matrix、window、record；一维数组不等于同一语义轴，矩阵中间结果不等于公开矩阵根。
3. 算子签名来自当前 op.catalog() 并与 operators.def 核对；数学描述来自唯一 C++ 内核、prepare 校验与相关测试。
4. 明确默认值、边界等号、缺失/错误、借用/独立输出、必要复制及不能隐式广播的情况。
5. SIMD 是内核能力，线程/进程是计划策略，协程是调用方式，共享内存是存储/运输方式；不能列成互斥的五种算法模式。
6. 调度阈值是当前可配置的工作量估计，不是数据点数或普遍性能承诺。计划与本次审计分别反映预计选择和实际执行。
7. 历史验收不重写成当前结果；旧设计加阶段提示，当前用户说明修正过时事实。

## 5. 验收方法

- 对照 operators.def 和已安装原生包目录，检查 146 个 opcode/名称全部且仅出现一次，参考签名覆盖公开重载。
- 检查参考公式、形状和关键默认值；默认调度参数与 planner.hpp 对照，决策顺序与 planner.cpp 对照。
- 执行新增使用手册、执行指南和算子参考中的全部 Python 示例；覆盖直接算子、标量/时序/矩阵图、递推、作用域、错误、快照、线程、进程、异步和共享内存。
- 检查新增与修改 Markdown 的相对链接、章节锚点及代码围栏；git diff --check 检查补丁格式。
- 本轮不改运行逻辑，不以文档示例代替完整数值回归、性能门禁或跨平台验收。

## 6. 初稿验收（2026-09-21）

2026-09-21，本机 macOS arm64、CPython 3.12、已安装 0.3.0 AOT 包：

| 检查 | 结果 |
| --- | --- |
| 三份新增指南中的独立 Python 示例 | 15/15 通过，包含线程、进程共享内存、Hard Stop 与 asyncio |
| 现有算子接口说明中的顺序示例 | 7/7 通过 |
| 算子名称/opcode 与源码及实际原生目录 | 146/146，签名参数和目录默认值一致 |
| 显式 SIMD 目录 | 18 项，与参考表一致 |
| Planner 默认参数 | 9 项，与实际原生配置一致；决策顺序另按源码核对 |
| 修改/新增 Markdown 相对链接及章节锚点 | 12 个文件、118 处链接通过 |
| 补丁格式 | git diff --check 通过 |

运行包的 engine_build_id：`157c5d54eb75ab6c6240dd0fdc6bacc066787ddd26d7fff20544b9e7462fae06`。
文档内容按当前源码核对；本轮没有修改引擎代码，未重新构建 wheel、重跑完整数值回归或性能门禁，也不声明跨平台或平台业务验收。

## 7. 提交前复核（2026-09-22）

以远端 main 的 `42fefed4cf6a3e3ae0bec049b35c7f501de780c5` 为开发分支基线。
该基线已经包含普通窗口/分组的故障溯源及分段投影边界校验；相应补充错误策略和投影说明，并明确矩阵沿时间归约输出 asset 轴、沿资产归约输出 time 轴。

本机 macOS arm64、CPython 3.12，改用该实现的已安装 0.3.0 AOT 包重新执行：

- 三份新增指南的独立示例 15/15，现有算子说明的顺序示例 7/7，合计 22/22 通过。
- 146 个算子/opcode、全部公开签名参数和目录默认值、18 个 SIMD 标签、9 项 Planner 默认值一致。
- 12 份文档的 118 处相对链接/锚点及补丁格式检查通过。
- 现有严格窗口和普通缺失兼容专项 3/3 通过（`tests/test_scope_error_status.py` 中 `strict_rolling or nonexceptional_missing`）。

运行包的 engine_build_id：`c5dfec2ad6bde5290db2cb10718154db2824d2a7279177ddd5b6065d21808dce`；按当前 C++ 源码、CMake/包配置及实际工具链参数重新计算后完全一致。
本次仍为纯文档变更，未重跑本机完整数值/性能门禁；PR 的跨平台必需 CI 与当前 HEAD 的 Bot 审核另外核验，不能以以上结果替代。

## 8. 复现文档示例

示例可在仓库根目录、安装好对应 AOT wheel 的环境复现。新增三个文档的代码块各自独立，已有直接算子说明按阅读顺序共享示例变量：

```bash
python - <<'PY'
from pathlib import Path
import re
import subprocess
import sys

total = 0
for name in ("user-guide", "execution-guide", "operator-reference"):
    blocks = re.findall(r"^```python\n(.*?)^```", Path(f"docs/{name}.md").read_text(), re.M | re.S)
    for body in blocks:
        subprocess.run([sys.executable, "-c", body], check=True, timeout=45)
    total += len(blocks)
blocks = re.findall(r"^```python\n(.*?)^```", Path("docs/canonical-operators.md").read_text(), re.M | re.S)
subprocess.run([sys.executable, "-c", "\n".join(blocks)], check=True, timeout=45)
print("validated Python blocks:", total + len(blocks))
PY
git diff --check
```

进程示例要求运行环境允许创建本地进程和共享内存。命令只验证示例行为；不能将其用作性能对照或完整算子数学回归。
