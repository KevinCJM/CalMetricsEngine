# V48 自审核：限定接入及实际复验已完成

2026-10-06。审核对象为 build_id `30722fb25c692505f924aec6755d8f3c5a6113507b9b54552fb22a895d8ab365` 的六个 native 文件和两份新增测试。初审基于隔离候选。现已限定接入真实checkout，完整资格和实际复验由v48-qualification-status.json及v48-adoption-status.json记录；原失败记录保留，整体目标仍开放。

- 数学顺序：每列仍按逻辑行升序求和，方差保持两遍中心化、原 ddof 和逐列失败顺序；Eigen 表达式保留减法与后续运算的两个浮点步骤。不更换到一遍二阶矩，不跨请求缓存统计结果。统计库路径只处理 C 连续 float64，N≥32、8≤D≤256；其他受支持布局及显式 scalar 继续现有真实职责的实现。
- DAG 与存活：第一节点必须私有、只有一个消费者、不是根，父值除至多一个显式短向量广播外均需满足原 persistent 证明。先选择原已证明的融合，之后才选择依赖几何的缓存。缓存使用第一节点自己的 slot；编译器的消费者最后使用约束让它在 variance/sqrt 期间持续存活。规划与执行共享 deferred_broadcast_extent 证明。
- 不同区间：arena_shapes 来自最大逻辑窗口的容量上界，不是每一区间的实际形状。广播缓存从已经 canonical prepare 的当前二维视图取值；消费者保持当前实际 shape 并改成零行步长列视图。空当前输出不读取广播行。没有把最大窗口的维度直接冒充当前输出。该行为由空/短窗口、不同请求形状及 prepared 结果独立性检查覆盖。
- 错误：第一节点和消费者在原位置 prepare；仅允许不会提早产生数学错误的 add/subtract/multiply 第一节点。除法只有非零的短广播分母进入库路径，否则在写输出之前回原路径。均值和非有限时方差在任何写入之前回放原八列代码；平方溢出仍逐列返回原错误及 out 前缀。isolate、观察器、上游故障不启用新增延迟缓存。
- 所有权/资源：缓存最多 D doubles，Eigen 动态 Map 没有动态拥有数组；方差两份最多 256-double 栈空间，不占新的 heap scratch。根写入本次独立输出，不修改输入；公开图结果只读，直接算子的既有可写独立输出契约保留。算法复制计入 audit，Planner 既有库预留不削弱；库关闭保持原生完整路径。
- 结果协议/Excel：不改变 AST 或序列化字段、公共算子、dtype/形状或数学公式。metadata 在 finalize 中重建，跨进程路径按原协议；已有 Excel 标准化所用原语依赖不变，普通 C++ 不执行 Excel。
- 测量边界：36 个同核数完整时间请求、36 个顺序 fresh 内存进程。输入独立只读准备、普通入口及两个独立结果轮转都计入。malloc 中位数自身减少 268,435,200 bytes，仍较 NJIT 高 276,934 bytes，视为基本持平；RSS 自身约少 256MiB，绝对 RSS 的 NI 差异不能替代数值工作区收益。C 完整时间 0.1236s、NI 0.0843s，缺口保留；F 为 0.1049s、NI 0.1332s。

本范围暂未发现阻止继续资格验证的问题。正常相关 1,235/native9、全量、ASan/UBSan和库关闭已通过；已限定接入真实六个native文件和两份新增测试，并重建实际wheel/worker、重新对照完整时间及内存。整个引擎、平台完整负载和大矩阵性能目标仍开放。

研究核对：[Eigen lazy evaluation](https://eigen.tuxfamily.org/dox/TopicLazyEvaluation.html)、[broadcasting](https://eigen.tuxfamily.org/dox/group__TutorialReductionsVisitorsBroadcasting.html)、[aliasing](https://eigen.tuxfamily.org/dox/group__TopicAliasing.html)。官网当前重定向开发文档，本候选仍使用项目固定 3.4.0，并已在实际编译器/运行库验证 Map 表达式；未升级依赖。

进一步核对 [ArmPL 官方 macOS 支持](https://developer.arm.com/community/arm-community-blogs/b/tools-software-ides-blog/posts/first-macos-release-of-arm-performance-libraries)：有 C/C++ BLAS/LAPACK、串行及 OpenMP 变体，针对 M1/M2 优化。它仍是现有大矩阵负结果后的下一库候选；本机未安装、未对照，也未采用，不能引用供应商描述宣称本机获益。此前 BLIS 完整负结果保留，不盲目重试。

资格验证现已完成：全量 15,628（0失败/错误/跳过）、正常/ASan+UBSan/库关闭相关各 1,235 和 native 各9通过，三种配置各40个额外转置窗口结果通过。首次ASan专项的1项失败来自macOS框架Python启动：父进程已预加载ASan，但框架启动后重写sys.executable并丢弃DYLD环境，测试派生的launcher子进程晚加载ASan。测试专用bootstrap恢复真实framework可执行路径与子进程preload，并先用ASan全局符号和同一空轴例子核验，再完整重跑同一1,235项。没有改变引擎/测试源码、跳过用例或放宽断言；原failed status、log、XML和执行器均封存在v48-asan-launch-failure-1中。此后同源普通/SAN/OFF检查终结，真实接入、实际重建与复测已完成。实际36次配对和36个fresh内存进程另加同场64次时间复核均保留；C仍慢于NI，F已领先，malloc自身减少约256MiB、与NI基本持平。不删首次wall波动、不宣称整体目标完成。
