# MatMul cache 分块：最终源码与回归范围审核

日期：2026-09-30。审核采用本轮 `start-manifest.json`，生产基线为 93dc；不把仓库既有脏修改纳入本轮。只读源码、既有报告及日志，没有运行数值、性能或构建。本报告不代替另一份实际对象/栈审核，也不提前给出完整请求验收结论。

## 结论

**没有发现当前补丁的数学、边界、所有权或线程预算阻断项。** 本轮仍需依最终构建的完整回归、114 项时间、51 项内存及真实栈/RSS证据判断收益。专项 `targeted-oracle-fixed.log` 已记录 106 项通过；这是读取主任务的执行结果，非本审核自行执行。

## 审核身份与范围

| 文件 | 当前 SHA256 |
| --- | --- |
| cpp/operators/matrix.cpp | 3c4a4cc865393c85451a499b07f90f57c72af478199b4968ef5be7bb6db58572 |
| cpp/operators/matrix_backend.cpp | 6d35c9896ec4233ee164cc2c404e506c727c33454e08b253d9647a120d50201e |
| cpp/include/calmetrics_engine/matrix_backend.hpp | 8ac92c958432045dd89e9ff68a668569de753c1d867328726925ba4794a97f28 |
| tests/operator_native_tests.cpp | 441379f95cc7fde596a72d044be77a377cec781d39463e8948d28990ff47b97a |

`tests/test_matmul_cache_library.py` 为本轮新增，包含 39 个参数化用例；专项成功后仅修正模块说明，明确逐位检查是有限夹具证据。最终实际文件身份由归档的 `final-repository/` manifest 冻结。三个生产文件与 `integration-patch-manifest.json` 的 draft SHA 一致；其历史“尚未应用”状态仅描述生成补丁当时，不代表当前是否采用。

AGENTS、computation-design-rules、Excel generation/reproduction 纪律的哈希均与已读版本一致。按 CodeGraph 定位 `prepare→infer_matrix_shape→matrix` 与结果/输入边界，再检查当前磁盘源码；图谱不作为运行证明。

## 数学与访问边界

- `matrix.cpp` 只对 automatic、Eigen/macOS15 ARM64、M/K/N 最小门限、C-compatible B（内步长1、行距至少N）选择新路径。65536/96先做整数除法，再比较K，实际683起准入，避免阈值乘法溢出。零行步长A的重复行路径在前；显式ISA、短K、F右矩阵及其他布局保留现有路径。
- `matmul_c_blocked_bulk` 只重新安排相互独立的输出格，固定32/64/64工作块；每个4×8输出Tile的K仍严格递增。第一panel从+0开始，后续直接载入该输出格上一panel的binary64累积值，再顺序做乘加；没有“每panel从0求和，再合并partials”的新括号。
- 每项均执行A×B，包括0×Inf及NaN。没有零系数跳过、finite预筛、阈值截断或补偿求和。现有编译选项保持 `-fno-fast-math -ffp-contract=off`；实际机器码与活跃栈由另一份对象审核负责。
- 固定Eigen Array/Map表达式同步求值，引用只在当前语句/调用内有效，没有懒表达式逃逸。输入只读；A允许F/负/切片/零列步长。A/B相互别名仍可读，只有输出被限制为独立可写区域。
- bulk范围为4的整数行及8的整数列。edges先计算bulk行的余列及余行全部列，两区域互不相交，且完全覆盖剩余输出。余列/余行的每格仍执行原序完整K。B padding、A stride及out两端canary都有测试。
- 原入口继续校验rank/shape、输出尺寸及数值缓冲跨度。Python边界继续执行输入owner/stride检查和 `OUTPUT_ALIASES_INPUT` 拒绝；本补丁没有绕过。已合法尺寸下，M/N>=32及原有可寻址大小约束使32/64块递增与内部偏移不会新产生size_t回绕。原生低层调用仍须提供原契约规定的有效Value与存储，未扩大为任意伪造指针安全API。
- MatMul既有IEEE值语义不改：非有限输出不是本算子新增的抛错条件；后续比较、sum、divide的错误与离散行为由真实DAG回归覆盖。不同输出格求值先后不改变共享输入，也不引入部分失败恢复的新契约。

## 所有权、成本与预算

- panel状态只存于本次已分配的独立结果。没有额外数值堆、packing、线程池、TLS缓存或跨调用数值复用。每次执行读取当前绑定内容，prepared地址/shape不变也不会缓存旧系数。
- panel结果store/reload是同一输出格的运算状态读写，不是把其他完整缓冲复制进结果，`algorithm_copy_bytes`维持0有明确口径。重复行旧路径仍单独计实际复制字节，未冒用新标签。
- `eigen-block-matmul`标签只在bulk执行后写入；无零尺寸“只打标签不算”的路径。它是direct算子Audit的路径标识，并不是Scheduler已经提供逐MatMul worker计数的声明。
- 库工作在现有调度线程内；没有库内线程或进程全局线程设置。`EIGEN_DONT_PARALLELIZE`与静态Array表达式沿用既有编译环境。
- 源码没有额外数值堆不等于完整请求内存不增。寄存器spill、函数帧、首次代码页、解释器/allocator、保留结果和RSS分别按实际对象与完整请求报告判断；本报告不以零workspace声明总内存通过。

## 回归是否命中真实范围

- direct检查：682/683/684，C/padded C RHS，32/33/64/65输出边界，F/负行/负列/切片/零列stride A；非C RHS及31轴边界验证保留fallback。显式scalar/可用ISA不得打新标签；重复行快路径保持优先及复制计数。
- 可追溯数值夹具：639–641跨后期K panel的FMA反例、分组归约、溢出、0×Inf、NaN×0、subnormal/正负零覆盖bulk和输出边缘。原生测试另覆盖首个/第二个64边界、解析可求解的坐标选择、稠密矩阵及原生canary；Eigen-OFF路径不假装库命中。
- 初次4失败是2类固定随机夹具各C/padded布局的近零参考差异，`test-reference-diagnosis.json`证明新旧93dc及scalar逐位相同，旧代码同样超出NumPy GEMM原容差。仅这两类改用独立`math.fsum`汇总已舍入乘积，输入与rtol=2e-12/atol=2e-14不变；同时检查当前scalar位模式及IEEE。不是为掩盖新代码差异扩大容差，也不把有限样本逐位测试升级为全域公共契约。其他NumPy参考测试保留，原始失败日志不删。
- direct workspace复用、相互别名的只读A/B、拒绝输出别名、独立结果不变、预填NaN输出正确初始化均覆盖。
- `run_snapshot`的两次独立结果检查绑定内容更新、旧结果保留、B根形状和内容、readonly及无输入/前次结果共享；没有声称普通`prepared.run`返回独立快照。
- single/thread/process-inline/process-shared四通道运行pickle后的真实typed DAG，检查CSE、矩阵float、bool、精确int64、divide错误及健康根独立性；两产品位置均检查。实际plan/lane、共享模式与CPU预算上界都有断言。
- worker证明边界：worker接收C矩阵、相同准入形状和同一生产代码；direct/native测试明确验证标签。当前Scheduler不输出逐节点ISA命中计数，因此不能仅凭图结果宣称已观测worker标签。测试注释保留此限制，没有新增审计字段。
- 共享结果测试关闭Scheduler和SharedInputBundle，删除全部输入owner与Result（包含循环遗留的array引用）后仍读取只读结果视图，验证输出owner独立保活。只读flag本身没有被冒充独立所有权。

## Excel及文档边界

这是物理分块变更，未改AST、DAG、dtype/shape、错误、算子版本、IPC或Excel数学展开与预算。没有理由新增Excel公式或把Excel工作引入普通C++路径。历史Microsoft Excel快照的参考回放只能证明对应范围，不是本轮新的Microsoft Excel重算。

当前对执行说明/发布说明的更新及最终性能结论，由主任务结合完整证据完成；本报告不将屏幕原型、106专项或归档完成提升为全算子/全矩阵联合验收。
