# 完整中心化面板复用（2026-10-02）

本轮接入已有组件候选，目标是去掉大幅度协方差拒绝后的重复中心化。遵守 AGENTS、计算纪律 D1/D2/D5/D6/D8 和 Excel 纪律；不改变数学或公式版本。

## 复用条件与唯一实现

`guarded` 已计算本次 canonical 均值并完整写入中心化面板。仅首次幅度 >8 拒绝、F-compatible、首段实际 count=N、偶数 D 时直接消费该面板。`count=N` 也覆盖 N<512/D>2048：实际每列按 count 紧密写入，分配的 block 上界不改变有效列 stride。tiny 首先拒绝、奇 D、部分面板、C、DSYRK 后拒绝及所有旧未准入域保留既有路径。

现有 `finish_pair/fused_pair_tile/upper` 使用编译期 Centered 分支，共享唯一固定 Eigen Four/Two 归约树。中心化分支只省去已执行的减均值；首次乘积种子、四位置累加、两包合并、N%4 尾部、除 N−1 与上三角镜像顺序相同。普通入口保留四参数 musttail。Centered 不读取未初始化的输出末行，也不采用 dynamic Eigen reduction，避免改变指针对齐相关的奇 D 边缘行为。

## 所有权、预算与生命周期

面板只属于本次 Workspace；消费前先恢复调用线程 BLAS 策略，消费中不扩容、不重入 execute，完成后释放。输入和先前快照只读，输出直接写本次独立结果。无新矩阵、缓存、私有线程或 IPC 字段。原 `(B+3)*D*8` 面板峰和 Planner 库工作区估算仍存在。保留 attempted 审计和 scratch 峰，释放后 retained capacity 为0，不能把复用说成零分配。Prepared 每次重读当前值；AST/DAG/Planner、错误隔离、Excel 观察器及非独立根的路径契约不变。

## 验收设计

以修改前 a684 wheel 为真实完整请求基线，新 wheel 的全部77源文件及 extension/worker 单独核验。专项覆盖 N 的四种余数、padded F、2050宽矩阵、C/F输出、进程打包、重复输入、tiny拒绝顺序、四舍入模式及已有异常标志。原生新增 canary、输入不变、精确输出与 Workspace 峰/释放断言，ON/OFF和ASan/UBSan均运行；macOS不支持LeakSanitizer。完整 Python、185 direct夹具、469案例历史Excel快照复核；没有新的桌面重算。

12类完整 ordinary/prepared 请求，用同一CPU预算、独立结果、6轮均衡次序比较旧/新C++与预热NJIT。新旧全矩阵逐位一致；NJIT原1e-12差异单独保留计数，不修改容差，不把差异标为正确。3类fresh malloc/RSS计入设置、面板、旧结果保留和库内部申请。输入准备只分配一个矩阵加O(D)临时行，内存测量前只编译NJIT签名，不数值预热。几KB/ms微差不独立否决；大幅度及长相关矩阵相对NJIT的实质欠项仍须保留。

结果以同目录日志和 identity/timing/memory 数据为准。
