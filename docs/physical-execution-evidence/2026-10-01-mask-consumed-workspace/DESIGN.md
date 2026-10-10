# Masked 顺序统计按本次消费数准备工作区

日期：2026-10-01。只读设计，不是实施或验收结果。目标仓库 `/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine`；基线构建 `850871ebcff4f6f2e089ae96029f736ccc0c94282cd06824193f839a410906b3`。先使用 CodeGraph 定位 registry/reduction、graph、planner、binding、runtime/process；稀疏索引未覆盖处再读当前源码。只写此临时文件，未修改仓库、构建、测试或执行数值/基准，共享脏工作全部保留。下列行号为根开始本轮编辑前的基线源码。

## 结论与首轮范围

可复用当前完整 bool 校验扫描，取得本次 masked median/quantile 的消费数 C，再以 C 个 double 执行 `Workspace.ensure`。这不改变选择/排序/插值/缺失/错误/结果所有权，不需要改变 Prepared、公共接口、Planner 或 IPC19。准备和规划仍保留源 N 的安全上界，计数只在本次执行栈内存在。

根已选定首轮最小方案：内部 `validate_data_structure` 可选计数；仅 masked-order 第 1 号参数的现有校验同时累计，C 连续掩码按 byte 访问，其他布局沿 `Value::u`；execute 按 C 准备实际工作区。**保留当前外部 Workspace 和图 TLS 的高水位容量复用契约**。cold 稀疏以及始终稀疏的 warm 请求消除 N 个 double 的首次分配；同图同几何先 dense 再 sparse，历史大容量仍保留，不能宣称这一场景也已回收。

## 根因与现有数据链

| 阶段 | 当前事实及位置 | 本轮处理 |
| --- | --- | --- |
| shape prepare | `cpp/operators/registry.cpp:195–212`：numeric/mask rank 1..3、同 shape；所有 median/quantile/where 的 `scratch_doubles=x.size()`，q 仍须有限且严格 0<q<1 | 维持 N 上界与原参数错误 |
| partial structure preflight | `registry.cpp:424–444,465,506–509`：只读 available payload，所有 mask 完整逐 byte 校验 0/1；失败后仍检查可见独立结构 | 默认不计数；结果及结构错误顺序原样 |
| actual execute | `registry.cpp:512–530`：output/NULL_OUTPUT/order/ISA 检查后 reset audit，再 mask 校验，最后 ensure 原 N | 只在此调用开启 masked-order 计数；校验全部成功后取 C 作为实际 doubles |
| packing/math | `cpp/operators/reduction.cpp:38–115,259–265`：按逻辑 C 顺序只复制 mask true 的值，消费数写入 `algorithm_copy_bytes`；原 nth/sort/±0/NaN guard 与插值不变 | 无需修改；将确保容量从 N 改为本次 C |
| workspace | `cpp/include/calmetrics_engine/operators.hpp:332–346`：double 增长先释放，再 resize；小于等于 capacity 时复用，不释放高水位 | 不改此通用规则；capacity 审计仍报告实际保留量 |
| direct binding | `cpp/operator_bindings.cpp:273–315`：prepare、输出检查、独占 Workspace lease、Scheduler token，再同一 ops::execute | 自然覆盖，无新 caller 参数；调用方显式 reserve 保持 |
| public requirements/audit | `operator_bindings.cpp:241–262,374–389,398–418`：requirements 是形状上界；workspace_bytes 为本次 logical need、workspace_capacity_bytes 为真实容量 | requirements 仍 N；实际 workspace_bytes=C*8，capacity 与 need 分开 |

冻结证据 `docs/physical-execution-evidence/2026-10-01-cholesky-group-order-linear/README.md:10,28,40` 已明确当前 N 预留：1% 消费的 4M masked median/quantile，cold 申请峰约 32.01MiB，相对 NJIT 多约 31.0–31.4MiB。这是实质欠项。按 C 分配仅改变必要排序 scratch；未选数值不读取，不增加 mask 副本或索引数组。可预期节约核心量为 `(N-C)*sizeof(double)`，**具体完整请求 heap/RSS/时间仍需新测量，不能由公式宣称已领先 NJIT**。

## 最小实现约束与证明

1. `validate_data_structure` 保持 TU 内部 static；可以从 void 改为返回 size_t，并增加默认关闭的计数选项。只统计 `median_where/quantile_where` 的 arg=1、mask、非 rank0、available payload。普通 prepare 不额外扫内容；structure-only caller 忽略返回值且默认关闭。
2. 目标 mask 的每个 byte 仅读取一次，先执行原 `require(byte<=1,"INVALID_MASK")`，再累计 byte==1。C 连续分支必须有 rank>0；单例任意 stride/空数组可采用现有 `c_contiguous` 判断，空数组不解引用 null。非 C、负/零/间隔 stride、F、rank3 均沿既有 `u(i)` 逻辑访问。不能把 bool 强转成 true 后丢掉非法 2/255 检查。
3. 非目标 mask 与 gather index 检查仍沿原参数顺序和完整范围执行，不因 count=0 提前返回，不用 prefix true 数量提前分配/发布。shape/type/q 检查的原先后顺序不改变。invalid mask 在任何位置仍先于 x 消费、scratch 分配及结果写入失败。
4. 当 mask 全部有效时，C=Σ[mask.u(i)==1]，且 0≤C≤N。packing 恰好按照相同 mask 与逻辑顺序写 C 次；workspace 大小至少 C，因此原 ordinal 数组及运算序列逐位保持。只计整数 mask，不读取 x，不将 masked-out NaN/sNaN/Inf/±0 纳入分类或准入。
5. count=0 的空/allfalse 请求调用 ensure(0)，再保留 `ordered_stat` 原 `INSUFFICIENT_SAMPLE`，不向 out 发布值。有效 mask 空选择不能隐藏非法后缀 mask 或其他结构错误。原公共 strict-q 规则仍禁止 q0/q1，generic reduce_value 的内部概率边界契约不改。
6. count 只作为 execute 内部局部量，不能写入 Prepared、BoundData、execution_metadata 或 cache。prepared 两次运行间允许同地址 mask 变化，每次重新校验/计数；运行期间不得通过别名修改输入，已有公开依据 `docs/canonical-operators.md:99`、`docs/user-guide.md:43`，因此校验到 packing 的两次读在合法调用下内容一致。
7. `algorithm_copy_bytes` 继续由 packing 的实际 C 得到，不复用预测数冒充真实写入量；`scratch_bytes` 改为本次 C*8，而 capacity 保持真实高水位。非法调用不保证返回部分 audit，沿原失败规则。

## 图、prepared、子作用域及 worker

- `cpp/graph.cpp:2190–2248,2469–2478,2684–2686`：每个实际节点重新构造 args/prepare，再同一 ops::execute；prepared API 只复用绑定/plan，不缓存各算子内容计数。`order_fusion_eligible:461–462` 只含未掩码 median/quantile，masked-order 不进入 source-level OrderedCache，所以无需改缓存语义。
- `graph.cpp:39–95,117–140,174–194`：每线程 root Scratch 拥有 Workspace，子作用域各自拥有 child Scratch；换图、缩小几何、matrix panel limit 改变时释放。**mask 内容变化不改变这些身份**，因此本轮保留 dense→sparse 高水位事实。Scratch.ensure 不提前分配 order 的 N doubles；实际大分配在 registry execute。
- `graph.cpp:1332–1364,1577–1693,1710–1817,1859+`：unresolved、rolling、apply/iterate 的 child execute_impl 使用各自实际 captured view/局部几何；局部 mask 校验所得 C 只对应该 invocation，不能使用父 mask 全量计数、其它窗口/组或未来迭代计数。prebound/observer/isolate 不绕过该统一算子入口。
- `graph.cpp:1394–1408`：失败依赖通过 `validate_structure` 只校验 available payload，不能把不可见 mask 当成全 false；只做结构检查的路径不分配数值 scratch。
- `cpp/native_api_bindings.cpp:805–849,866–873,998–1025`：PreparedExecution 的 unchanged 只允许稳定地址/dtype/shape/strides/区间；每 run 调 engine，snapshot 写独立结果。无新 count 成员，借用输出及 next-run 失效规则不变。
- `cpp/native_runtime.cpp:133–137,210–244`：single、thread chunk、DAG branch 均终至 graph::execute；每个线程 Workspace 独占。process 交给现有 transport。
- `cpp/native_process.cpp:516–528,683–695,779–783`：worker 缓存的是 logical program bytes，接收本次 payload/descriptor 后执行 graph；count 不传输、不缓存。已存 graph audit `operator_workspace_capacity_bytes` 和 copy_bytes 原字段足够；IPC19 不新增字段。
- `graph.cpp:2903–2930` 当前容量审计包括 root/child Workspace 和其它持有缓冲；不能把单个 masked-op 本次 C*8 当成全图常驻/申请峰，更不能用减少后的 logical need 掩盖旧容量。

## Planner 安全上界与未来细化

`cpp/planner.cpp:204–281` 的 scratch_estimate 用 shape/array capacity；masked-order 仅触发 `operator_scratch`，保守另加 window*24，order_sources 的独立排序 cache 仅统计未掩码 median/quantile。`matrix_workspace_doubles` 在 `graph.cpp:956–993` 只计算相关矩阵/library 面板，masked-order 不参与。实际 C≤N，首次分配减少而现有高水位仍受原 N 上界覆盖，所以本轮不改变 admission 或 budget error。

`planner.cpp:312–360` 递归加入 child capacity 和选择缓冲；`486–498,667–668,812–818,882–892` 覆盖并发 worker/branch 和全部运输/结果/预算。保留这些上界既安全也避免 mask 可变时缓存 plan 下估。首轮可能仍因 N 保守预算而拒绝实际稀疏请求：这是**规划精度未来欠项**，不是本次 actual memory 优化失效。

未来若要降低按内容的 admission，必须给每次 native invocation 独立、可消费一次的当前校验证明，且涵盖中间 mask、局部视图、所有子作用域与并发 worker；不能仅扫描顶层 mask 并把 C 存入长期 plan。未经这项具体证据，不改 Planner 或构建新通用 framework。

同几何 dense→sparse 主动释放大容量是独立政策变化：`docs/canonical-operators.md:117` 与 Workspace 注释目前明确足够容量继续复用。首轮保留并专项量化；若未来要求回收，只对本次 masked-order 无用容量建立有文档/测试的局部规则，先释放再按 C 分配以免 old+new 峰；不要全局改 Workspace.ensure，破坏显式 reserve、其它算法或所有 warm 复用。该未来动作不等于可宣称整个请求内存预算已精确化。

## Caller / ABI 与公式影响

内部 validator 的直接 callers 只有 `registry.cpp:465` 和 `:522`。默认关闭计数的 prepare_impl caller 不接结果；execute caller接局部 size_t。因此这个内部返回值变更没有外部 ABI 或协议影响。

公共 prepare/prepare_geometry/validate_structure/execute 与 Prepared/StructureResult/Workspace 类型全部不变。传递覆盖的生产调用点：direct binding `operator_bindings.cpp:314`；graph `graph.cpp:1493,2685`；pointwise fallback `pointwise.cpp:314,477`；Excel 单算子 native reference `excel.cpp:649`；工具 `benchmark_multiworkload.cpp:140,154` 与 native tests 的 execute helper。pointwise 图本身不含 masked reductions，但其统一 fallback 保持兼容。

公共 validate_structure 的 graph shape/failure 调用 `graph.cpp:910,1403,2493`，binding borrowed-mask preflight `operator_bindings.cpp:288`，Excel geometry/failed-formula调用 `excel_budget.cpp:185,212`、`excel_operators.cpp:304,322` 均不取/缓存 C。reduce_value 的 signature/callers 均不变；现有内部 callers 是其它 mean/std/variance/axis reductions，没有新增 masked-order count 依赖。

Excel 仅数学/公式，不依赖 C++ Workspace 物理容量：遵守 `docs/excel-generation-rules.md:91–93` 的物理等价，以及复现契约 `docs/excel-reproduction-design.md:16–18`。数学与展开方案不变，不新增普通 C++ 请求的 Excel 工作；新 C++ 专项及历史快照可复核，不能声称本轮真实 Microsoft Excel 重算。

## 必须验收的边界

- 当前每个 mask 验证都完成：非法 byte 位于首/尾/未选区；同时坏 q/shape、x 非有限、allfalse 的错误优先级；out/input bit 不变。
- count=0/1、C=1023/1024/1025、全 true/稀疏、C/F/非连续/负/零 stride/单例怪 stride/rank3、空数组；消费与不消费 NaN/sNaN/±0/Inf payload，和原标量路径逐位值/错误对照。
- `requirements` 保持 N 上界，默认新 Workspace logical need=C*8；显式 reserve=N 后 capacity 保留 N；小于 count 时沿已有先释放再增长。direct 公共 out/lease/所有权不变。
- prepared 同地址 mask sparse→dense→sparse→allfalse→invalid→sparse：每次 C 和 copy_bytes 更新，快照不受后续失败影响；**dense→sparse capacity 仍 N** 应有明确断言，而不是错误写成 C。
- 多根/共享 mask、内部消费者、observer/isolate、至少一个 child scope 和 process IPC 路径；根/child capacity 审计可解释，未产生 count cache。
- 完整结果、同 CPU、同布局及同旧输出寿命测 cold、固定稀疏 warm、prepared snapshot、dense→sparse 常驻容量；输入构造临时有界，heap 与 RSS 分开。按 AGENTS/D8 判断实质收益，几 KB/几 ms 不单独否决，31MiB 欠项由实测闭环；Planner 仍保守、其它 NJIT 欠项保持开放。

## 基线文件身份

```
79eb2dfd17d2233a59084fa2948471911a491a931452fd6649e4f4ccf030c97c  cpp/operators/registry.cpp
0024db411c68609c6f64c6c4478da9496c5b99a93c5846b3d5ad70c38f1aedf0  cpp/operators/reduction.cpp
e49b1c4e97e169179f3e28b79b6c91548f010160bc1abc935522c350f434777b  cpp/include/calmetrics_engine/operators.hpp
8c0285c24eee1c1ca0e8fd09bfda3c0479f3678935ba4624d53bc3be02b7b31f  cpp/graph.cpp
6a6597a629cc59361ff2cca3e08a97855d5aaf90aaea1a71e40e5ecb1ddc8cf9  cpp/planner.cpp
fce802fad12e3eca81b6a45abca895fefcdd4adeef76027e6e06328d53c22525  cpp/native_api_bindings.cpp
ea260563ec6a30704d20f0ab0bccd21777d8c5f1e1d9c9310cbae13be7a7ab49  cpp/native_runtime.cpp
e56a8b78a635e8c27c1d4b40797ac7d67c1157130c8719510edaff236e9d5e1b  cpp/native_process.cpp
```
