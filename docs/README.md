# 文档导航

本文档集对应当前源码的 146 个 canonical 算子。引擎版本号相同不代表构建内容相同；部署时同时核对构建身份、算子目录和计划凭据。

## 使用者从这里开始

| 任务 | 阅读入口 |
| --- | --- |
| 了解标量、时序、向量、矩阵；准备输入；构建和运行 DAG | [用户使用手册](user-guide.md) |
| 查找全部算子、签名、输出形状和数学逻辑 | [数学算子参考](operator-reference.md) |
| 了解何时使用 SIMD、线程、进程、协程和共享内存 | [执行与性能指南](execution-guide.md) |
| 直接调用算子、复用 out/Workspace、理解借用视图 | [算子接口与内存契约](canonical-operators.md) |
| 接入逐指标错误、快照和执行凭据 | [平台执行契约](platform-execution-contracts.md) |
| 深入了解状态编码、确认、峰谷及区间边界 | [状态与事件契约](state-event-contracts.md) |

首次接入建议依次阅读使用手册第 1–5 节、执行指南第 1–3 节，再查所需算子。算法数值契约以当前原生实现及其回归测试为依据；发现文档与实现冲突时先核查，不凭文档静默改变数学逻辑。

## 维护与实现

- [编程与数学计算纪律](computation-design-rules.md)：相关设计、开发与审核的必读规则；覆盖依赖、复用、矩阵结构、状态、数值等价及性能证据，不代表待开发能力已经实现。
- [架构](architecture.md)：模块职责、编译执行链路和所有权。
- [文档体系设计与验收](documentation-design.md)：本轮组织方案、维护规则和检查边界。
- [数学组合设计](mathematical-composition-design.md)：分块、过滤、分组、求根及组合算法。
- [递推与状态设计](stateful-series-design.md)：当前递推、typed 输出和错误传播细节。
- [物理执行优化与验收](physical-execution-optimization.md)：缓存块融合、直接根输出、迭代缓冲、张量分块与审计。
- [M0/M1 开发设计与验收](platform-foundation-design.md)：本工作树的三维协议、具名载荷和原生迭代；验收状态以该记录为准。
- [平台自研计算完整承接规划](platform-compute-roadmap.md)：后续算法、类型协议、平台接入及等价验收工作；附[函数盘点初表](platform-compute-inventory.csv)，均不代表迁移已完成。
- [发布](publishing.md)：构建和分发流程；发布前还需遵守当前检出的提交规范。

## 阶段设计与验收记录

以下文件保留当时的范围、决策和实际测量，不能作为最新 API 总目录。历史“118/125 个算子”、旧 dtype/协议版本和当时不支持的能力不代表当前限制。

- 第一阶段：[设计](canonical-operators-design.md)、[验收](canonical-operators-acceptance.md)。
- 原生计算图：[设计](phase2-execution-graph-design.md)、[验收](phase2-execution-graph-acceptance.md)。
- C++ 执行链路：[设计](cpp-first-design.md)、[验收](cpp-first-acceptance.md)。
- Typed DSL 与窗口：[阶段设计](typed-dsl-rolling-cpp-design-2026-09-20.md)。
- 物理 DAG：[设计](planner-physical-dag-optimization-design-2026-09-20.md)、[验收](planner-physical-dag-optimization-acceptance-2026-09-20.md)。
- 指标能力：[变更前审计](metrics-factory-capability-audit.md)、[数学组合验收](mathematical-composition-acceptance.md)、[状态扩展验收](stateful-series-acceptance.md)。
- 性能：[多负载记录](multiworkload-benchmark.md)、[C++/NJIT 对照](cpp-vs-njit-benchmark-matrix-2026-09-20.md)、[0.3.0 验证记录](verification-0.3.0.md)。

历史验收通过仅覆盖其记录的构建、机器、数据和检查范围，不自动证明新修改、其他平台或上层金融业务通过验收。
