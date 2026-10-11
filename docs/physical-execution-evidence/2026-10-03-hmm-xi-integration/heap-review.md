# Full HMM heap probe 静态审查

日期：2026-10-03。范围：只读源码与已生成 JSON；没有运行探针、数值、编译或性能测试，没有修改项目文件。

## 结论

`untracked_frees=13` 不能解释成 13 次非法释放、泄漏或 13 个本轮分配漏记。它只表示：探针在 active epoch 中收到非空 `free(pointer)`，却没有找到该地址的**当前 epoch**记录。epoch 前合法分配、epoch 内释放足以产生此计数；未拦截的分配路径也能产生相同现象。现有聚合统计不能证明这 13 次具体属于哪一种原因，更不能给出它们的字节数。

本轮没有静态发现“失败 realloc 把旧记录丢掉”的实现缺陷。也不能因为 table overflow 为零、old/new 的未跟踪释放次数相同，宣称全进程内存已完整追踪或遗漏可抵消。证据应表述为**相同探针下的、已拦截 malloc-family 请求字节观测**，结合独立 RSS、原生分配/复制审计和具体大缓冲生命周期分析；不作为完整内存证明。

## 源码事实

| 位置 | 可证明行为 |
| --- | --- |
| `tools/heap_probe.cpp:25-48` | 只有 active 时登记成功返回的非空分配；记录请求 bytes，不是 allocator usable size/RSS。 |
| `tools/heap_probe.cpp:50-65` | 找不到当前 epoch 记录、且不是带 ticket 的退休操作，才增加 `untracked_frees`；不减 `live_bytes`。 |
| `tools/heap_probe.cpp:68-80` | `ticket_for` 只查询当前 epoch，查不到时返回零。 |
| `tools/heap_probe.cpp:84-96` | begin 递增 epoch、清零计数；snapshot 不换 epoch；reset_peak 仅把 peak 设为当前 live。epoch 前已有对象不被自动纳入。 |
| `tools/heap_probe.cpp:103-112` | realloc 失败且 size 非零，旧记录保留；成功/size 零才退休已识别 ticket，并登记返回的新对象。 |
| `tools/heap_probe.cpp:34-39,59-64` | address reuse 时可退休旧记录；ticket 防止晚到的旧 realloc 删除新对象。带 ticket 找不到记录不会增加 `untracked_frees`，因此这个计数也不是所有退休异常的计数。 |
| `tools/heap_probe.cpp:126-131` | 仅拦截 malloc/calloc/free/realloc/posix_memalign/aligned_alloc 六个 API；没有宣称覆盖 malloc_zone_*、reallocf、映射及全部运行时分配。 |
| `tools/benchmark_memory.py:31-58` | 只 begin 一次：setup → cold → 删除 cold result → 五次 warmup → GC → warm baseline → warm 循环 → 删除结果 → end。 |
| `tools/benchmark_memory.py:59-70` | 自动拒绝 table overflow；不因 `untracked_frees>0` 拒绝结果，也未给未知释放做字节归因。 |

## 本次已有观测

`memory-old.json:22-79` 和 `memory-new.json:22-79` 的计数模式完全相同：

| 阶段快照 | 累计 untracked_frees | 从上一快照新增 |
| --- | ---: | ---: |
| setup | 0 | 0 |
| cold | 12 | 12 |
| cold result 删除后 | 13 | 1 |
| warm baseline | 13 | 0 |
| warm | 13 | 0 |
| warm result 删除后 | 13 | 0 |

所有快照 table overflow 都为零。由此只能确认未知释放集中于首次执行及首次结果释放；warm 的 13 不是热阶段新增 13 次，也不是 setup/cold/warm 三个 epoch 各漏记一批对象。连续 epoch 已消除了“冷结果跨新热 epoch 释放而失配”的人为重置问题。

两个报告使用相同 probe SHA `74421dbfff14b56b730553c7b0b0ce9b38a136de053f132069e1b948decd3382`，以及相同 helper SHA。报告记录的是二进制身份；本轮没有重新建立该预编译 dylib 与当前 `heap_probe.cpp` 的构建来源绑定，因此当前源码分析和所用二进制的完整实现一致性仍是证据边界。

## epoch 前分配为何是合法候选

实际驱动 `/private/tmp/calmetrics-platform-latent-closure/memory.py:55-74,109-120` 先完成导入、源码/二进制身份读取、完整 readonly 输入构造、tiny NJIT 编译/运行以及 threadpool 配置，再进入 `measure_heap`。helper 本身还在 begin 前执行 GC，并创建 ctypes 统计对象。进程中明确存在合法的 epoch 前分配；首次完整执行可导致缓存、运行时或旧对象释放。

这证明“合法 epoch 前分配在 epoch 内释放”是该测量机制允许的来源，**没有证明这 13 个地址就是这些对象**。要具体归因需要地址/分配来源/尺寸/epoch 的证据，当前 JSON 不含这些信息。本轮没有为小额差异增加追踪框架或额外运行。

## realloc 与哈希表审核边界

- failed realloc 不会调用 forget；success 则先按 ticket 退休再登记新地址，未见由这一普通路径直接造成的旧记录丢失。
- realloc 的旧地址来自 epoch 前时 ticket 为零；即使此次 realloc 释放旧对象，也不会调用无 ticket 的 forget。这种释放不计入 `untracked_frees`，所以零计数同样不能证明完整追踪。
- 当前 epoch tombstone 保留 epoch，查找遇到空 pointer 仍继续；遇到旧 epoch 才停止。静态未发现把普通删除 tombstone 当搜索终点的错误。
- table overflow 为零只排除了已拦截 track 操作报告的表容量耗尽；不能排除未拦截入口/出口、运行时 arena 子分配或观察边界交错。
- 内存 hook 与真实 allocator 操作不是同一个原子区间：malloc 之后登记，free 之前退休，realloc 成功后补记。并发地址复用虽有 ticket 保护，计数仍不能等同任意瞬间真实 allocator 存活量。

## 允许和不允许的结论

允许：相同工作量/独立结果/后端配置/探针下，报告冷/热已跟踪请求字节峰值、所有阶段未跟踪释放次数和 table overflow；将大规模数组、工作区、复制字节的代码事实与观察互相核对。当前连续 peak 已包含 epoch 内存活 setup/workspace，不能再叠加阶段峰值。

不允许：称 heap 数据是全部对象/全部分配的完整统计；由 13 的次数上界推导遗漏字节很小；因为 old/new 都是 13 就扣掉/抵消未知字节；由 warm_retained 非零判定泄漏；由 table overflow 为零证明追踪全覆盖。owner/function 在 heap end 后仍有意保留，关闭后的释放不在该 epoch 内。

warm 循环 `result = function()` 会先生成新结果、再释放旧 result，峰值真实包含两份结果交替时的共存；这是双方一致的测量负载，不应误认成探针累计了全部历史结果。RSS 使用独立子进程和 0.5ms 采样，没有 heap probe 注入；它是页级补充证据，会受基线、缓存、采样和运行时影响，也不能单独替代完整分配审计。

本轮应保留未跟踪计数及这些限制，不据此新增 KB 级门禁或反复调整阈值。后续若需要证明中大型链路的显著内存收益，应围绕真实大数组/工作区/结果重叠给出字节与生命周期依据，而非追逐 13 次未知释放的表面计数。
