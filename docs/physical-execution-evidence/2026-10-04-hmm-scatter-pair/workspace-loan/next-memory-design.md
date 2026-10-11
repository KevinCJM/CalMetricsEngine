# 下一轮大额HMM内存瓶颈：设计假设，未实施

2026-10-04，当前0fe已接入，完整回归正在运行。以下不是功能或性能验收，也不能关闭requested heap高于NJIT27.159 MB的缺口。

## 已定位的链路

冻结500k/4/12数学链的training为375k行。graph_for的train_filter只经slice_axis最后一行→sum_axis成为下一段hmm_filter初值，未作为public根；逻辑hmm_filter仍返回完整历史。7366同一逻辑链的native geometry显示训练density/filter各4.5M doubles（36 MB），使numeric slots0/2各达到36 MB；完整double arena96.002 MB。0fe只改loan pointer/lifetime，并未减少这些逻辑中间结果。这个几何记录为前阶段证据，后续实验须重新由当前AOT实测核对，不能当作新候选峰值。

当前loan借用2个NK arena span与future probability NK，保留live training/classify16 MB。仅缩短下游full-history中间值会减少arena，却同时减少可loan的fit span：HMM仍需3NK时会重新分配缺失矩阵。必须测整个请求，不能只报告少了一个36 MB矩阵。

## 联合候选A：过滤尾行的私有物理闭包

识别same model projections→gaussian_log_density→hmm_filter→最后一行slice→sum这一实际single-use闭包，只在full输出不被root/其他消费者使用、时间/组件轴及步长可证明、严格flat执行且没有observer/isolate/deferred失败时准入。逻辑节点/类型/来源保持；未知或partial/output消费者保持canonical。

复用同一density与filter row数学原语及现有Eigen独立行/状态lane，仅保留有界density block和K状态；逐行先后与原始初值/转移/continue语义保持。不能给一个K缓冲伪造NxK可读Value：跨slice的私有组必须证明几何并在最终消费者直接产生原K结果，内部placeholder只有geometry且绝不被其他节点读取。原density时点的参数/域校验和first-error必须保留；中间节点若无法证明无独立错误，继续原路径，或在私有失败后按原顺序重放canonical并验证行为。

Planner/worker/executor用同一closure与shape证明，public roots选择、process/thread、diagnostic/Excel回放保持完整逻辑图。不新增canonical算子，不改变公开hmm_filter语义，不把某一模板名字写进通用引擎。

## 联合候选B：HMM后向状态checkpoint/replay

复用同一backward row算术，首次倒序保存块边界Beta状态，Xi所需块按原行顺序使用逐块重建的Beta。checkpoint是精确复制的K状态，不做近似/新停止策略。Xi仍按原time/source/target累加；alpha/gamma可按消费次序复用，须证明每一旧alpha在所有Xi消费者结束后才覆盖，likelihood在覆盖前完成，M-step在gamma完整后执行。

保存原最后行/边界、特殊N/K、scalar与library两路、一致的error/partial-state恢复。选择块大小依据同核数完整耗时与heap/RSS，extra backward扫描和copy均计入；对当前约5.7秒/NI11.25秒优势的真实侵蚀必须报告，不以较少capacity掩盖退化。代码保留一个当前数学实现，不引入平行历史fit。

## 验收顺序

先当前0fe全量、真实Sanitizer和三库OFF结束；再隔离实现两个候选并分别/联合对照0fe及同一冻结NI，所有13字段逐位old/new和原1e-9 NI/离散精确、只读/2代/故障恢复/全路径检查。完整setup内存epoch及ordinary计时与first-use准备分别记录，CPU1/必要外层多核相同核数；包括大宽度C/F/负步长与未准入负控制。只有稳定的大额净收益且数学/所有权安全，才接入正式代码。主要矩阵求解/协方差等仍有独立实质性能缺口，不因HMM进展结项整体目标。
