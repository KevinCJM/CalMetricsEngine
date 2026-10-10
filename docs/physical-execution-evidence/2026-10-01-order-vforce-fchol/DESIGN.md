# 三项独立物理优化：设计与验收边界

本轮目标是实质减少中大型完整请求耗时与矩阵占用；几 KB、几毫秒变化不单独作为门禁。数学正确性、异常顺序、只读输入、结果独立所有权及资源预算仍为硬约束。按照 AGENTS.md、D1–D8 和 Excel 纪律执行，不替换公共数学算法或修改金融语义。

1. 单个 median/quantile 及对应 masked 算子：在已复制的消耗值中同时判别 NaN/混合零符号；至少 1024 个安全值使用标准库 nth_element 选上排名，必要时用 max_element 选相邻下排名，仍执行原插值表达式。其余输入继续当前 full sort。无额外缓冲；共享多根排序缓存保持原实现。数学逐位等价证明见 order-selection 子任务设计。
2. 独立终端 exp/log：使用原 vForce 与原正常值准入范围，以最多 65536 元素的实际连续块逐块准入。坏块按原 scalar_math 重放，健康块继续库计算。无消费者/observer/isolate/scope/继承错误的强门槛保证失败结果不可见。直接算子调用保留全输入写前判别和原失败前缀。跨布局继续既有有界 tile；审计只累计实际库调用的元素。
3. 独立明确 F 布局 Cholesky 根：原算法完成后上下三角因素逐位互为镜像，因此改变成功末尾清零的物理三角即可生成相同逻辑 F 结果。保留原算术、对称性/有限校验、panel 与重放、库预算。清零前不改变内部 C 索引。移除 F 输出原先额外的整份 C arena 和最终转置复制；结果仍独占。consumer/scope/isolate/covariance pair 不准入。observer/prebound/input_failures 恢复 C arena 的 metadata 后使用原路径。

AST、DAG 的数学节点、公开算子签名、Excel 公式和过程传输描述均未改变。数学等价用旧 wheel、原 scalar 或原 full-sort 缓存比较；exp/log 既有库浮点容差保持，IEEE 类别、符号、零及特殊输入 canonical bits 单独检查。Excel 无公式变化，原 reference observer 验证不可冒充新的 Microsoft Excel 重算。

统一以本轮源码构建新 wheel，并核对源码、wheel、加载模块与 native worker 的身份。Python 功能/回归、原生、ASan/UBSan、库关闭分别检验。完整负载对比保留 e619f8cd 旧 wheel、实际预热 NJIT、单核预算、普通/独立快照、保留上一结果的所有权成本。时间串行均衡轮转；内存用独立新进程从 setup 起连续测 cold-through-warm 堆峰值，另用新进程测 RSS，禁止 setup 中数值预跑或释放 N² 临时矩阵掩盖请求增量。按严格历史诊断与当前 D8 采用结论分别记录，不把少量正差当作自动拒绝，也不把显著 NJIT 欠项写成已通过。

基准覆盖正常/稀疏/整块异常、C/F 与跨布局、selection 随机/已排序/逆序/全等控制。跨布局 NJIT 比较 vector 与 bounded tiled 两种实现，采用更快对照；全部原始结果保留。Cholesky 新旧全输出逐位比较，并检查三角/正对角及 81 个确定系数重建，不将抽样重建声明为全矩阵残差。

独立审核发现 native F 子图 prebound 降级漏计一份 C arena，属于实质预算错误而非小分配波动。已让 Planner 子图和执行共用降级判据，plan_array_arena 明确区分普通终端与 prebound；F 连续初值需要两份迭代状态，回归同时验证精确预算/低预算拒绝、实际工作区和输入不变。新增 compiler native 回归保留 wire F schema。独立原生 observer 回归另验证 C 描述、F 结果逐位一致及下一次普通执行重新使用零 arena。

## 最小语义与复用纪律

三项修改分别仍只负责既有顺序统计、逐元素 exp/log、Cholesky 分解；没有新增黑盒算子。原有可独立消费的逻辑结果仍能被选取，含消费者的图继续原执行路径，独立根资格不会删除数学依赖。公开 DAG 已能表达这些算子职责，标准库选择、vForce 块与 F 存储属于物理实施细节；再拆成公共节点既不能增加独立数学结果，还会改变错误顺序、工作区或结果所有权。因而本轮保持原公共颗粒度及唯一数值实现。所有权、失败回放、observer、prebound、子作用域与资源预算分别验证，未把图 metadata 中的 F 标记当作真实 F 结果的证明。

## RSS 测量修正

初始驱动在 F 数据生成、masked 输入修改和哈希时释放整块临时数组，这会让 allocator 复用准备期驻留页，掩盖请求 RSS 增量。保留原记录，申请堆数据仍按实际 malloc 请求有效；RSS 用有界行/8192 元素块准备及流式逻辑 C 哈希重测。受影响输入与冻结计时输入 SHA 完全相同，33 个新 fresh 子进程替换原 RSS 口径。每类每后端只有一个 RSS 监控样本，不外推所有驻留状态。
