# 同 A 双 solve_multi 根：最终静态自审核

审核结论：未发现阻断性代码问题。本结论仅针对本轮共享求解增量，不代表全算子性能目标完成。最终运行验收由主代理执行并完成；本审核仅只读核对其日志，未自行执行数值验证。

审核对象为 identity.json 记录的构建 `295a46460a8c783dd5b764f8eb37b556dd2d1d22997c070308ef4540efe6a3a3`，对照构建 `437269d43960c791fd13984fba0051d3df5a72d3dfbe03a8d96414df6aa030c7`。本次通过 CodeGraph 导航并直接阅读当前源码、已生成日志和 evidence-validation.json；未重新运行身份校验、编译、测试、数值或性能程序。没有把共享工作区的其他存量差异作为本轮新增成果。

## 代码与契约

- `cpp/graph.cpp:778`：资格仍为两个不同、无消费者的终端 solve_multi，共享同一个静态 A 输入；无 scope/isolate，中间仅静态输入绑定。Program::validate 先校验根、父节点、input index、axes 和 schema，资格短路及索引有前置保障。
- schema 现在接受 `order != F`，因此显式 C 与 automatic 均能发现候选；`reuse_solve_pair` 同时检查第一实际输出 order 和第二 `ResultLayout::order` 均为 C。没有把 F 缓冲伪装成 C。现有 `choose_order` 对该 solve_multi 图默认返回 C；明确 F 请求继续原路径。Python 编译入口默认 auto，专项的默认入口加一次 A 复制审计覆盖该真实路径。
- 第二 RHS 读取静态 input 描述，不读取尚未绑定的 scratch.values。第二 prepare 失败仅拒绝共享；第二输出类型、rank、容量和两个不同大小结果区的非重叠检查均保留。现有全图形状前置校验没有绕过。
- `solve_multi_library.hpp` 的单 span 包装保留原平台/规模门槛、single/multi 不同 NaN 主元判据、32 列面板、Eigen 更新和每个 RHS 的乘减顺序。双 span 只共享相同 A 的消元；回代先完整第一根、再完整第二根，未合并或重新结合每个 RHS 的算术。
- `matrix_foundation.cpp::solve_multi_pair` 只对 SINGULAR_MATRIX/NON_FINITE_RESULT 回放。已写第二缓冲先恢复正零，再由图按原节点顺序执行；资源异常不吞掉。零 RHS 不访问其元素，仍保留 A 的主元和回代对角检查。
- 第二 scratch.values 与 completed 仅在完整成功后写入；completed 每区间重置，只跳过第二求解，B2 输入绑定仍正常执行。observer、prebound、isolation、input_failures 保持原路径；没有跨请求数值缓存。
- 两根各写最终独立结果槽，原结果发布循环不变；工作区仍为 N² doubles。旧双根本来就复用该工作区，不能声称峰值减少一个 N² 矩阵。减少的是重复消元及累计一次 A 复制。Planner 原保守预算/工量保持，typed 请求不走 scalar-only DAG branch。

## Native 失败缓冲回归

已确认片段接入真实 `tests/graph_native_tests.cpp:130`，并补齐必要 includes。128 阶、2/3 列 RHS、B2 input 位于两个 solve 之间，覆盖：

1. A 最后主元为零且 B2 含 NaN：错误保持 SINGULAR_MATRIX，第二输出的每个位置仍为正零。
2. A 倒数第二对角为 0.5，B2 对应行第二列为 DBL_MAX：第二回代发生 NON_FINITE_RESULT；第一根完整，第二根该行保留 `12 / +Inf / 10` 的失败前缀，末行已完成。

两个场景将 caller-owned 输出的全部布局字节与 execute_observed 的原节点路径对照，另检查异常代码、输入逐位不变及两端 canary。这样能覆盖 Python 异常时不返回的部分结果缓冲。该 oracle 证明共享路径与当前非共享原节点执行一致；它不是一个独立求解算法，旧新安装包逐位对照仍由基准/专项的其他证据承担。

## 已存在的验收证据与边界

- 已只读核对主代理执行的最终日志：`python-full.log` 为 11129 passed；`focused-final.log`、`focused-off.log` 各 332 passed；`native.log`、`asan.log`、`native-off.log` 分别为 ON、ASan/UBSan、OFF native，各 9/9。`remaining.log` 的各步骤退出码为 0，并以 ALL FINAL CHECKS PASSED 结束；原待完成项目据此关闭。本审核未自行编译或运行这些测试。
- `direct-fixtures.log` 记录 185 个直接算子夹具通过；`excel-snapshot.log` 记录 469 案例、3056 值通过历史重算快照对照，且 newly_recalculated=false，没有新的桌面 Excel 重算。`remaining.log` 还确认主代理的 ON/OFF 分发检查、OFF 身份及最终身份检查完成；`final-identity.json` 仍记录上述 295a464 构建。
- `evidence-validation.json` 记录 8 类、每类 6 轮计时及 24 个 fresh 内存子进程。4096 C 普通完整请求中位数为旧 5.094873 s、新 2.654035 s、强 NJIT 0.693015 s。新实现减少约 2.441 s，但仍约为 NJIT 的 3.83 倍，不能宣称中大型全面领先。
- 新旧 heap 峰值差为 +16 B；4096 F 请求 RSS 增量从 137543680 B 到 138919936 B，增加 1.3125 MiB。原始变化应披露，不把小额堆差当作单独性能否决，也不写成内存完全不增加。
- 原生 heap 记录保留非零 untracked_frees，例如 cold_retained=1、warm_baseline=5、warm=6。不能据此声称完整分配生命周期审计闭环；这些记录不是已定位泄漏证据，亦不能删除或静默归零。
- 已测适用平台仍是当前 Apple arm64/macOS 原生库路径。主代理已完成本机 OFF 回退的专项与 native 验收，但未取得新的 Linux/Windows 执行证据，不能把本机 OFF 通过外推为跨平台运行通过。数学与 Excel 逻辑 DAG 未变，本次静态审核不构成新的 Microsoft Excel 重算验收。

本次仅维护此 stage 审核记录，没有修改生产、正式测试或其他文档文件。
