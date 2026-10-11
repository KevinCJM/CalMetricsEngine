# V70 后续：按实际尺寸诊断与规划缓冲

2026-10-09，本轮为源码事实与下一步设计，尚未实现新的 arena 分配器。

## 当前源码事实

`cpp/compiler.cpp` 的 `Builder::finish` 758–810 行：先计算节点最后一次读取，根保活到结束；反向传播借用视图的 backing 生命周期；按类型使用不考虑 shape 的 first-fit 染色。只有旧槽 `death < node_id` 才可复用，等于也不可复用。borrowed 视图保留 inline storage，本身没有数组槽；eigenvalues 为潜在物理内核预留槽，真正执行原视图时 arena 会跳过。因而不能把问题描述成“所有视图都有幽灵数组”。

`cpp/graph.cpp` 的 `plan_array_arena` 2761 行起：按物理准入去掉独立直接根、已证明融合等；其余每个逻辑 slot 取实际 shape 或融合 cache 的最大容量，最后求和。删除/共享逻辑数组会改变编译染色及 backing 生存期，slot 的最大容量求和反而可能增大。两组 GMM 控制图的增长几乎全部来自这里。尚未取得逐节点容量/时间线 trace，不宣称已证明某一个具体槽造成全部增长。

HMM `loan_latent_workspace` 还会使用 arena 空闲段及本次未来独立输出保存历史，保存/恢复仍活跃 backing，补缺失 NK 历史；原有申请峰是 arena、公开结果、fit reserve 和其他输入准备共同构成。只优化 arena 总量可能使 history reserve/复制增加，不能只看局部表格。

## 下一步实现顺序

1. **先诊断，后替换**：在隔离构建导出本请求各 owned 节点的 dtype、实际 byte extent、birth/death、原 slot、物理 offset/capacity，以及借用 owner、deferred producer 最后消费点和直接根。诊断记录不持有数据，不增加大数组，不进入正式性能测量。闭合区间逐项检查冲突，核对总容量与现有 audit；对原图及两个负控制图找到放大的实际槽。
2. **明确别名约束**：仍持有输入/旧结果/独立根的区域不得充当工作区。借用后代继承 owner 的完整区间。deferred private source 在最终消费者读完前不能释放。原物理融合的同槽就地许可、强制不同槽证明及 latent loan 的区间依赖要列成显式约束，不能把不同 node 的地址任意改成相同。
3. **按尺寸选空闲区**：先做仅诊断的带约束区间装箱，对齐、溢出及 byte/dtype 边界检查；输出当前方案与候选方案的 arena、history reserve、save/restore byte 估算。保留原方案，只有组合成本真实下降时才切换。请求 shape 不明、scope/隔离或无法证明 backing 时保持当前安全执行路径。
4. **最小物理接入**：若 trace 证明确有大额空间收益，再考虑 `node offset/capacity` 与现有 logical slot 分离。读写目标、slot bounds、延迟 cache、views、loan 保存/恢复必须统一使用同一计划；不能只更改 `numeric_slot` 的一处索引。预先记录 HMM 拟合节点，避免 V70 私有融合 O(PN) 扫描。
5. **资格与采用**：数值/离散/错误/非别名、nested views、overlapping closed intervals、多 HMM、门槛边界、prepared 更换 shape、线程/进程/隔离；原生、相关后再全量/SAN/OFF。相同输入和所有输出的重复完整链 time/heap/RSS，joint cost 保留大 HMM history 成功路径；负控制图只是诊断，不能牺牲错误语义换内存。采用后重建实际包再次验证身份和实际路径。

目标是三状态大样本 HMM/GMM 的大额申请峰缺口。几十 bytes 的 audit 差别、几 KB/毫秒的变化不作为门禁；输入准备、全部根和两代保留不删减，随机/停止规则不放宽。既有 Eigen/Accelerate/Highway 数学内核不重写，不新增业务算子。

## 公开方法参考

- [OpenXLA 缓冲分配说明](https://openxla.org/xla/hlo_to_thunks)：参考调度生存期、别名和分配协同检查的方法，GPU 执行描述不能视作本 CPU 引擎的实现。
- [ONNX Runtime allocation_planner.cc](https://raw.githubusercontent.com/microsoft/onnxruntime/main/onnxruntime/core/framework/allocation_planner.cc)：参考追溯原 backing、复用计数、同尺寸和外部输出准入、inplace alias 区间归并。仅分析参考，未复制源码、未引入 ORT/XLA 依赖，也不声称它们可直接替换当前 Planner。

现有结构是 C++ 原生图规划问题；成熟矩阵库不能自动修复上层跨算子的生存期。完整 active-set QP、有效前沿、随机探索与更多 C/F/批次/多核主负载仍未完成。
