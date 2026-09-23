# Typed 多输出结果协议设计

状态：已实现并通过本机验收，详见 [验收记录](typed-results-acceptance.md)。基线为 main e88f69a；本轮只修改 CalMetricsEngine。

## 目标与兼容

一个共享 DAG 可输出 scalar、series、vector、matrix，各根独立保留 float64、bool、int64、名义轴、实际形状和错误。数学算子、金融口径、状态/窗口语义保持不变。record/window 及带内部状态标签的矩阵仍通过字段投影使用。

GraphCompiler.compile 新增 result_format="auto"|"typed"。auto 对原先可接受的纯标量、同 dtype 对齐时序保留 values/offsets 接口；其他公开数值根自动使用 typed。显式 typed 也适用于原有图，尤其需要保留 bool 标量时。dtype 以 Typed IR 为准，不把原有 float64 的计数/索引归约改成整数；本轮精确 int64 验收覆盖向量、时序和矩阵。原标量接口继续按既有 float64 契约运行。

Typed 结果提供 outputs（顺序与表达式一致）。每个输出有 dtype、kind、axes、values（每区间一个只读 ndarray）、statuses（每区间的逐元素状态）、root_statuses 和 shape_known。零维标量、空数组和未知几何的失败分别表示。未知几何的失败返回空占位数组且 shape_known=false；root_statuses 保留失败原因，不能把空数组或整数零当作成功。禁止把混合结果强转成统一 float64/object 数组。

## C++ 布局与执行

Program 增加每根 dtype/rank 描述，原生计划编码升级 v6；解码保留既有版本。表达式源、root 顺序及 typed 格式参与计划身份与 pickle。语法白名单不扩张。

升级会改变执行图指纹，旧执行计划不能跨版本复用；持久化定义可重新编译，历史快照凭据保持原样。worker 必须使用同一安装包中的版本。

Planner 按每个区间推导各根容量上界（复用现有原生几何规划），区别容量和实际形状。按区间、根顺序排列输出，每根起址按 8 字节对齐；一个底层缓冲区承载不同 dtype，根视图不复制。动态长度（lag、block/group 等）在容量内记录实际形状，不在 Python 再算形状，不重复执行 DAG。输出、状态和描述信息均纳入内存预算，并检查大小溢出。

在预留“区间数×根数”的描述槽之前先检查描述信息预算。typed inline IPC 计入父输出、worker 输出、发送帧、接收帧和解码 payload 的同时存活容量；共享路径沿用共享输出。预算是原生执行存储估计，不保证进程 RSS 或调用者重复物化 Python 视图的总内存。

Executor 沿用同一个节点遍历、CSE、融合、arena 与错误传播，仅扩展根写出。逐区间保留实际 shape、root status 和逐位置 status；使用 strides 读取借用/转置/负步长输入，只把最终结果写入独立输出。失败值不覆盖健康根，结构错误继续抛出。所有 payload padding 及未使用容量初始化，避免 IPC 暴露未初始化内存。

线程按区间写入互不重叠的输出片段；typed 本轮采用已有产品/区间并行，现有标量 DAG 分支并行保持。容量布局依赖几何，不依赖输入数值/参数，因此复用计划时仍允许数值变化。实际形状与容量在写出前校验。

## 进程、所有权与凭据

原生 worker 协议升级，显式校验 typed 模式、根数、rank/dtype、shape、状态数与字节长度。共享路径按布局偏移写入同一父进程拥有的区域；小包路径返回有界连续区间片段。父子各自按同一程序和输入几何推导布局，不传裸指针。失败/超时仍先结束任务再释放映射和 CPU lease。

Planner 按实际分块检查完整响应的 256 MiB 上限，包括容量 payload、逐元素状态、shape、根状态及所有协议头。超限自动改用共享输出，再按该运输路径估算内存；预算减少 worker 后重新分块并检查。共享路径仍通过 IPC 返回状态与形状，若这部分超限则在规划阶段拒绝，调用方需缩小区间批次。协议限制不通过放宽单帧上限规避。

普通 execute 和 run_snapshot 结果独立；typed 根视图持有整个底层缓冲区/共享区域的所有者，关闭 scheduler 或销毁 Result 后仍有效。Prepared.run/run_audit 显式借用，后续执行可覆盖值；typed run 返回携带状态的 Result。run_snapshot 直接写入独立缓冲区。所有公开 typed 视图只读；prepared 不提供公共 output 缓冲接口。NumPy base 仍可被调用者追溯，属于受信任内存边界；复用前检查底层地址、dtype、shape、连续性及可写状态，调用期间禁止更改或并发写入。

Typed 结果凭据使用 cpp-aot-execution-2、typed-results-1，声明每根类型与结果布局；原接口继续 execution-1。上层平台尚未接入新协议，不能据此宣称生产服务迁移完成。

## 验收与自审核

- 不同 dtype/rank/shape 根混合、重复根、共享中间结果、精确 int64 >2^53。
- 矩阵/静态向量、区间切片、转置/负 strides、只读和零轴输入。
- 动态长度、空区间、最小样本、未知几何失败、逐位置错误传播和健康独立根。
- single/thread/process inline/shared、Hard Stop、pickle、陈旧计划/协议、内存预算和溢出。
- prepared 借用与快照、跨轮 shape 变化、所有者释放后视图、并发与超时生命周期。
- 全部 Python、原生、ASan/UBSan；既有四组性能门禁阈值不变，增加 typed 代表负载测量。
- 自审重点：计划容量是否低估、失败时是否读取无效指针、描述信息是否可被外部修改、共享写入是否重叠、旧接口是否回归。
