# 隔离候选自审：通用求解与连续布局

本阶段保留CME根924源码作为生产基线。候选不新增canonical算子、Python数学体、线程池、跨请求分解缓存或IPC版本；只改4份既有C++文件，数学LU证书与原错误阈值不变。

- 同一LU核心：private arena直接分解；retained只在独占Workspace复制当前系数。B有限校验仍在A前；不准入/库失败沿同一个canonical体，保留实际尝试、复制、残差物化和reserve审计。coeff N²区与有界residual区以offset隔开，取指针在ensure后，原系数与历史结果始终不变。
- 残差：retained原C/F值直接参与signed DGEMM，仅abs面板有界物化；private由仍存活的叶父canonical重建。完整原方程核验后才发布C/F结果，原pivot、singular和离散错误行为不放宽。
- 几何/预算：dense方阵1024–4096、RHS1–64；runtime与Planner共享查询，覆盖copy N²、scratch、ABI向量和vendor估计。reserve是估计，实际请求堆/RSS单独测；OFF不做无用N²复制。显式F private失败回放所需完整workspace有保守预算。
- 连续布局：same-order C仍使用既有canonical；same-order F展平physical span但保持原每项IEEE；跨布局用32×32栈tile调用同一个canonical向量体，再做物理转置写出。不跨项求和或改参数顺序；无完整中间矩阵。private叶父F继续在arena生成F；公共根严格按其指定C/F布局。mixed/strided/小几何/观察/子图prebound/错误继承仍走原路径。
- 类型/所有权：新布局入口先拒绝非float数值输出和不合格producer；公共结果/输入不参与分解写入，两个请求及根间独立存储。两代prepared状态变化、错误恢复、thread/inline/shared原生transport、NaN/Inf/signed-zero、partial32/64面板、RHS1/64均有回归。
- 接口/Excel：逻辑AST/DAG、Registry、dtype/shape/axes、参数、错误和Excel公式/符号预算不变；不增加C++计算时的Excel成本。本阶段没有重新执行桌面Excel。

已验证：最终ac54源码/wheel/worker88源身份一致；710相关回归、normal native9/9及12,658全量通过，零失败/错误/skip；40个完整任务复测结束。初始b458完整30任务及全部失败记录保留，尤其公共F秒级退化和retained路径比NI高约17MB的峰值不得隐去。分解段DGETRF2筛选降低約10.6MB但变慢约88ms，只是ABI诊断，未替代正式库。

尚未满足生产采用/整体结项：SAN/OFF回归及retained峰值缺口。normal全量和40个完整任务已终态通过执行，但retained峰值未优于NI。general复制路径、独立operator入口、更多非DD/更大矩阵、其他常用组合任务的既有未闭环项仍需继续。

最终复核：normal全量12,658、实际SAN相关710/原生9、关闭库OFF修正后相关710/原生9均零失败/错误/skip。normal全量和SAN在仅测试审计断言修改前；修改后normal选中7及OFF全相关710通过，数值源码/normal/SAN/OFF包逐项hash不变。初次OFF709PASS/1FAIL和失败控制器终态原样保留，不改写为成功；根924数值源码仍未采用候选。retained峰值约17MB缺口及主要负载开放状态不变。
