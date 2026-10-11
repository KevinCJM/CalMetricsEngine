# Mask consumed workspace 最终只读审核

2026-10-01。**未发现本轮范围内需要修复的实际问题。**审核源码 `cpp/operators/registry.cpp` 的 count 差异及 `tests/test_order_linear_packing.py` 本轮四个参数化新增项/断言调整；未构建、测试、执行数值或性能，未修改仓库。适用 AGENTS、计算规则 D2/D5/D6/D8 与 Excel 物理等价纪律；CodeGraph 先行，未覆盖段以当前源码补读。

## 身份与审核范围

- 基线 `850871ebcff4f6f2e089ae96029f736ccc0c94282cd06824193f839a410906b3` → 本轮 ON `ccca6b67a5220ea2134a4743d281b80c5af4c096e7d41c3d5b747bb8de70e331`。
- 两份 identity 的 77 个 source_sha256 对照，只有 `cpp/operators/registry.cpp` 改变；本地当前 registry SHA 与新 identity 一致。
- 在内存中仅逆转已审两处 count 改动（validator 和 execute 的调用/实际大小），重建 hash 恰为基线 `79eb2dfd17d2233a59084fa2948471911a491a931452fd6649e4f4ccf030c97c`。因此本轮没有夹带 registry 的其它生产差异。未保存生产历史副本。
- 当前 registry SHA：`bc8d665ddd45862f71f66f77bb89c0e4bfca43fc1a273b225c004357872b115b`。
- 当前测试 SHA：`bdd94be10631cd469b2ad9f4ac2040ca3e85cea652440bd5129fe5ee34872a07`；静态参数展开该文件共 38 项，其中本轮新增同一测试的 2 operator × 2 reserved = 4 项。这不是实际 collection/test 通过声明。

## 已核查的关键行为

1. `registry.cpp:424–455`：计数默认关闭；仅本次 execute 对 masked-order 开启，arg=1 mask 的每个 byte 先 require≤1 再相加。非目标 mask/gather 检查保留原循环及参数顺序。结构预检调用 `:477` 沿默认关闭路径；不可见 payload 在 `:428` 跳过，不伪造消费数供执行使用。
2. `registry.cpp:431–441`：rank0 先走原 scalar bool 校验，不能误走 byte 指针；实际 masked-order rank1..3 已由 prepare 验证。C 连续 mask 使用现有 `c_contiguous`，单例任意 stride 不影响逻辑索引；非连续、负/零/间隔 stride、F 和通用 rank3 保留 `u(i)`。空/null 数据无元素时循环零次，不解引用。
3. `registry.cpp:524–544`：output/null/order/ISA 错误仍先于 mask 校验。完整 mask 校验完成后才选实际 doubles 并 ensure；invalid 末尾 byte 仍在读取 x、分配/改写 Workspace 和写 out 前失败，不发布部分 selected。allfalse/空 mask 的 selected=0，随后 reduction 原 `INSUFFICIENT_SAMPLE`，不是提前成功或新的错误。
4. `reduction.cpp:38–77` 完全未改：同一合法 mask、同逻辑顺序只读取/复制 C 个 x 值，因此所需大小 C 与打包数一致，排序/插值/NaN/±0/Inf 语义和失败前缀不变。计数扫描只读 mask，不新增 masked-out x 或 sNaN 消费。输入运行期间禁止别名改写的公开契约见 canonical-operators:99 / user-guide:43，prepared 两次已完成运行间仍可更新 mask。
5. Workspace.ensure、requirements、Prepared、StructureResult、graph、Planner、worker 全部未改。新 cold 按 C 分配；显式 reserve 或历史 dense 分配的 capacity 继续复用，与 canonical-operators:117 相符。scratch_bytes 报 C*8，capacity audit 报真实保留量，algorithm_copy_bytes 仍由实际 packing 得到。无跨请求 count cache、额外 mask/index 副本、所有权或 CPU token 变化。
6. 图 masked-order 未进入共享 unmasked OrderedCache；observer/isolate/prebound/child/worker 的 canonical execute 入口自然获得同一校验/分配行为。worker 只缓存 program，无 count 传输，IPC19 原字段保持。Planner 继续 N/window 安全上界，并发 worker、子作用域与旧容量仍在已有预算内。

## 测试差异与边界

- `test_order_linear_packing.py:39–71`：既有 C/padded/F/negative/singleton2/singleton3 × sparse/alltrue 回归把本次 logical workspace 断言改为 consumed*8；输入/out/read-only/复制数继续核查。覆盖 C 指针计数与不同 source/mask 布局，不只镜像计算 count。
- `:74–91`：1023/1024/1025 consumed 阈值原值逐位比较保持，新增/调整实际 count 工作区和 requirements 仍按源 N 的断言。未放宽容差。
- `:132–168`：空/allfalse → 原错误、out 不写、Workspace 可恢复；首/末非法 bool byte 与 NaN x → INVALID_MASK，输入/out 不改。
- `:184–214` 新四项：median_where / quantile_where、默认 / reserve=N，顺序 7→N→1→0→1025。成功值对照排序 oracle；本次 need=count*8；capacity=max(过去count,显式reserve)*8。0 请求须错误且 out 不改，后续成功验证保留容量及可用性。明确覆盖 dense→sparse 的高水位保留，没有把保留量误报为本次 C。
- `:217–239`：共享未掩码排序与两个 masked 根共存、两区间、所有字段值/复制数/独立只读结果。
- `:242–279`：prepared 同地址只改当前 mask，1023→1024→1025→全选→7→allfalse 失败→恢复 1024；每次结果和 copy count 更新，所有先前 snapshot 保持独立。内容不缓存，失败恢复沿原规则。

本轮四项并未声称重新覆盖整个 observer/scope/process 组合矩阵；这些入口统一调用链未改变，应由根的完整回归和既有专项验收。代码审查无发现，不代替该验收。

## 必须保留的结果边界

此改动解决的是首次稀疏分配及固定稀疏 warm 的源 N scratch；**dense→sparse 同图同几何仍保留高水位**。Planner/requirements 仍按 N 保守，低内存预算可能继续拒绝实际稀疏请求，这属于后续预算精度问题。只有新完整请求 heap/RSS/时间证据能关闭冻结约 31MiB 欠项，不能把 C*8 推导写成实测或整体领先 NJIT。几 KB/几 ms 不单独否决，旧输出保活、准备临时、进程运输和常驻缓存须继续计量。Excel 数学/公式无变化，本审核不声明新的 Microsoft Excel 重算。
