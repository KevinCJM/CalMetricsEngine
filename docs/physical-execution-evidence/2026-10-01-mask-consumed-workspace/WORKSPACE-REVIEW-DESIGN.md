# Masked order 工作区：只读复核与最小方案

日期：2026-10-01。范围：`median_where` / `quantile_where` 的每次执行消费计数、Workspace 容量、Scratch/TLS 生命周期和 prepared 输入变化。未修改仓库文件，未构建、运行数值、测试或测量。已先用 CodeGraph 导航；规则依据为本轮已读的 AGENTS、computation-design-rules 和 Excel 生成/复现规范。冻结基线由 root 指定为 850871eb；上轮正式证据为 `2026-10-01-cholesky-group-order-linear`。

## 结论

root 采用的首轮 count 方案静态复核未见阻断：在既有全 mask 结构校验中同时计数 C，每次 `execute` 用 C 申请 masked order 的 doubles；Prepared 和 Planner 继续保留全形状 N 上界，Workspace.ensure、reserve、所有权和高水位复用行为不变。

这能消除首次/固定稀疏调用的全 N doubles 分配。dense → sparse 的同一 Workspace/TLS 仍保留旧 capacity；`workspace_bytes=C*8` 只能表示本次请求工作区，不能据此宣称这类序列的保留容量或 RSS 同比例下降。必须用真实 capacity 审计和独立内存测量披露该边界。当前范围不新增 owner 模式或缩容策略。

## 根因及既有释放规则

- `cpp/include/calmetrics_engine/operators.hpp:332–347`：`Workspace::ensure` 仅在 d 超过 doubles.capacity 时先 swap 释放再 resize；较小请求和 d=0 只改 size。indices 也保留 capacity。`capacity_bytes()` 报两份 vector 的真实 capacity。
- `cpp/operator_bindings.cpp:273–315`：未传 Workspace 的 direct 调用使用局部 Workspace，函数结束即销毁；显式 Workspace 由调用方所有，并用 try_lock 保护独占。`398–418` 的公开 reserve/capacity 不能被此次优化悄悄改变；目前没有记录 reserve 意图的字段，不能从容量推断“可释放的自动增长”与“调用方预留”。
- `cpp/graph.cpp:39–70,209`：图执行使用 root_scratch TLS；各 child Scratch 自有 Workspace。`117–141` 在 graph metadata 身份改变、max_window 缩小或 matrix panel_limit 改变时释放 workspace。相同图、相同形状、仅 mask 内容改变，不触发释放。
- `cpp/graph.cpp:154–157` 有 arena 精确容量 swap idiom；`105–115` 的 release_execution_buffers 会同时清除 live arena、values、captures/state 和 children，不能在单个 reduction 中借用。`3023–3025` 的 release_graph_workspace 是安全边界的显式释放入口，不是 masked 单节点缩容入口。首轮 count 无需调用它们。
- `cpp/include/calmetrics_engine/inline_vector.hpp:53–77,126–149`：heap 仍在大 size 区域时 resize 保留容量；退到 inline/empty 时销毁 heap。此处用例是 `cpp/include/calmetrics_engine/native_runtime.hpp:35` 的 audit chunks，数值 doubles 不由 InlineVector 承载。更改 InlineVector 不能解决 masked 数值缓冲保留。

## 当前 count diff 的安全性

当前检查对象：`cpp/operators/registry.cpp:424–455,524–544`。

1. count 只对 masked median/quantile 启用，且只统计 rank>0 的 arg1 mask。`195–211` 已要求数值/掩码 rank 1–3、相同 shape。因此逐字节验证 byte≤1 后 `selected += byte` 得到 C≤N；正常执行的输入数据可用，零尺寸允许空指针。
2. 校验完整扫描所有 logical mask 元素，先 require 再计数。无提前停止、无按数值有效性跳过校验。非法 bool/uint8（含末端字节）在 ensure 和数值 packing 之前仍抛 INVALID_MASK。
3. C-contiguous mask 走原始 uint8 指针，其余 F/负 stride/补边走 `Value::u`。`cpp/operators/reduction.cpp:53–76` 的 packing 使用同一 logical traversal；即使 x 非 C 而 mask 为 C，mask 的 `u(i)` 仍与计数一致。计数不读取 x、不消费被 mask 排除的 NaN/Inf，也不改变原 packing 顺序、排序守卫、分位插值或异常优先级。
4. C 不是 Prepared 的缓存字段。`cpp/native_api_bindings.cpp:824–849` 的 prepared unchanged 检查冻结地址/类型/几何而允许内容改变；每次 execute 重新扫描 mask，因此同地址 sparse → dense 会重新增长到足量工作区。输入在执行中遵守原只读借用契约；两次运行之间可更新内容。
5. N=0 或 C=0：ensure(0) 只把 size 置零；packing 不 append、不解引用空 ordered，随后原 `INSUFFICIENT_SAMPLE`，out.set 尚未执行。原有容量仍保留；fresh Workspace 容量保持零。异常恢复不依赖上次 count。
6. 结构 preflight 仍调用 validator 的默认模式，仅校验、不计数。缺少 payload 的失败派生输入依旧不被读取；图的 isolate/partial-error 路径保持原结构校验和跳过执行行为。未成功执行的节点不保证生成成功 Audit，不能凭异常路径断言 `workspace_bytes` 已公开。

## 请求量、真实容量与 Planner 的组合

| 层级 | 保持/变化 | 验收应解释的量 |
|---|---|---|
| Prepared `scratch_doubles` | 保持 N | 几何最坏上界，不能在 prepare 读一次 mask 后冻结 C |
| `ops::execute` ensure | masked order 改为 C | 当前活跃 doubles size；不等于 capacity |
| direct `workspace_bytes` | 改为 C×8 | 当前申请的算法工作区，不包括调用方 reserve/历史余量 |
| direct `workspace_capacity_bytes` | 保持实际 capacity 计算 | 真实 doubles/indices 保留容量，包括显式 reserve |
| graph `operator_workspace_capacity_bytes` | 保持实际容量累加 | 当前 Scratch 与 child 工作区、已有 scope 状态/元数据；不应要求等于单根 C×8 |
| `algorithm_copy_bytes` | packing 的 C×8，未改 | 实际复制消费量；多根累加，不能与共享 workspace 容量混为一谈 |
| Planner | 保持 N/window 上界 | 准入预算仍需容纳今后同几何 dense mask |

依据：direct 审计在 `cpp/operator_bindings.cpp:258–260`；graph 容量审计在 `cpp/graph.cpp:2901–2930`；实际 packing copy 在 `cpp/operators/reduction.cpp:77–81`。Planner 的 `cpp/planner.cpp:255–283` 对顺序节点共享 Workspace、对 masked order 保留 `window*24` 上界，并单列 unmasked ordered cache；没有因为首次 sparse 放宽准入。

对只有 masked order 的同几何图，历史容量来源 C_old≤N≤window，保持当前 Planner 全形状上界可覆盖 sparse/dense 内容变化。max_window 下降时 Scratch 原释放规则生效；若别的输入保持全局 max_window 不变，则旧 masked N 仍被这个 window 上界覆盖，此时容量不下降是已有保留行为。其他算子的矩阵/模型工作区与 reserve 不是此次优化新增，不能从 masked C 推导全图容量上界。

多指标按顺序共用一份 Workspace：容量取执行历史的最大需求而非求和；较小 masked 节点的 resize 会重用容量，下一较大节点仍 ensure 足量，无 scratch 数值跨算子生命周期。`cpp/graph.cpp:384–427` 的 unmasked median/quantile 多根 ordered cache 是独立的 N 元素缓存，generation 每次更新；此次不得将它当 masked count 缓存、清除它或宣称所有排序内存已缩为 C。

prepared snapshot 在 `cpp/native_api_bindings.cpp:866–874` 直接写入独立输出并冻结只读；Workspace 与返回输出、旧 audit chunks 没有所有权混用。count/resize 不会影响旧 snapshot。各 child Scratch 独立递归计入审计，不应为清理某个失败 masked 节点全局释放父作用域 state/capture。

## 最少有意义回归（建议合并为 4 组，root 串行执行）

1. **fresh 稀疏 + 布局**：新 direct Workspace/新图身份确保无历史 doubles；N 明显大于 C，覆盖 C/F/负 stride/补边中必要两三种路径。原 scalar 位模式、input 地址/内容、out canary；请求 bytes 与复制 bytes=C×8，capacity 实测小于全 N 分配，不把跨 STL 的 capacity 精确相等当新契约。
2. **同一 prepared 的内容变化 + snapshot**：同地址/形状 sparse（含 C<1024）→dense（C≥1024）→sparse→empty→dense；每次实际值正确、从新 mask 重计数、empty 原异常/状态类别、后续恢复、旧 snapshot 位模式/只读/独立。dense 后 sparse 的 capacity 明确仍≥dense 峰值；Planner estimate 不随 mask 内容下降。直接 caller Workspace 的 reserve 复用可放同组单独几行断言。
3. **完整结构校验 + partial error**：首/末非法 uint8/bool 字节，尤其大部分 mask=0 且 x 含排除的 NaN；INVALID_MASK 仍优先于数值消费、out 未改。isolate 图另有健康根，失败根不破坏健康输出；随后修好同一 mask 成功。非法 mask 在新 workspace 上不先分配全 N doubles；旧容量保留不误报为新增分配。
4. **共享 Workspace/形状边界**：两 masked 根使用不同 C，再加一个仍需 N 的 unmasked order 消费者或现有大工作区消费者；容量/结果不被最后小 C 缩坏，copy 审计按真实根次数累加。另用较小 masked shape + 不变的大输入覆盖 global max_window 不下降；若现有 scope fixture 易复用，把相同检查放进一个 child 并保留父/旧 snapshot。无需新写数学实现或大量重复选择算法样例。

检验“首次稀疏”必须在没有相关 TLS 历史容量的执行上下文中进行；不要用已经 dense warmed 的同线程 audit 判定首次优化失败。检验 dense → sparse 则必须复用同一 owner/thread，不能靠重新建图掩盖保留容量。性能/RSS 结论仍需 root 正式串行实测；本文件只给静态安全结论。

## 读取身份

以下 SHA256 在 root 已写 count diff 后读取；这不是冻结 baseline 850871eb 的身份：

```text
operators.hpp      e49b1c4e97e169179f3e28b79b6c91548f010160bc1abc935522c350f434777b
inline_vector.hpp  541850ea3930085601380615911f08332af1987936d7fbc07b8d7403baec8997
registry.cpp       bc8d665ddd45862f71f66f77bb89c0e4bfca43fc1a273b225c004357872b115b
reduction.cpp      0024db411c68609c6f64c6c4478da9496c5b99a93c5846b3d5ad70c38f1aedf0
graph.cpp          8c0285c24eee1c1ca0e8fd09bfda3c0479f3678935ba4624d53bc3be02b7b31f
planner.cpp        6a6597a629cc59361ff2cca3e08a97855d5aaf90aaea1a71e40e5ecb1ddc8cf9
native_api_bindings.cpp fce802fad12e3eca81b6a45abca895fefcdd4adeef76027e6e06328d53c22525
operator_bindings.cpp   24ce31d0f0815e1028b02b5ef487e6da1adf8b9280df8032240ef9ba115e7fbe
```
