# 按轴递推、多分位数与完整消费者链

本轮新增通用数学原语 188–190，原 1–187 的契约不变。签名、类型、IEEE 边界及空轴规则以[算子参考](operator-reference.md#按轴递推与多分位数)、canonical 注册表和[原生实现](../cpp/operators/axis_primitives.cpp)为准。它们不包含金融输入限制或业务模拟黑盒。

## 1. 本轮要解决的问题

已验收的[共享投影](mc-projection-primitives.md)减少了重复资产计算，并显著提速；需要完整 period 输出的负载仍必须保留这些大结果。真实平台的 compact 输出只需要 fan、样本轨迹、14 个统计和资产贡献，源 NJIT 完整固定带参考约 3.108 秒、requested-heap 峰值约 48.9 MB，参见[冻结的完整链设计和证据边界](physical-execution-evidence/2026-10-03-mc-eigen-producer/compact-chain-implementation-next.md)。这些旧参考值不是本轮配对验收结果。

先补齐可编辑数学 DAG，再实现有界消费者闭包。仅有新原语不能证明省内存：普通 DAG 仍可能物化完整资产收益、NAV 等张量。大负载须等有界 proof 与预算贯通后测量，不把单算子微基准当作完整链性能结论。

## 2. 依赖、布局和所有权设计

- prefix 的每条 line 有独立标量初态，沿选定轴升序递推；没有跨 line 状态。空输出不读取数值 payload。数学体由窄 helper 提供，后续物理闭包复用，避免维护第二套公式。
- 多分位数只共享同一次请求、同一 line 的排序。概率仍逐项验证；不同数据、轴或请求不能沿用旧数值。NaN-last 比较器不剔除 NaN，也不对相等值的 payload 顺序作新承诺。
- 输入按合法 signed stride 借用，不强制全量 C/F 转换。prefix 直接写结果；quantiles 每条 line 复制 N 项到一份复用排序缓冲，再直接写结果。公开结果独立，不作下次计算 scratch。
- AST/Typed IR 固定轴、推导 shape/axes/量纲；DAG 几何校验与 Planner 使用同一 scratch 容量。顺序节点复用同一个 Workspace，线程/worker 各自独占，仍服从现有 Scheduler CPU/内存预算。
- typed 根支持 C/F 连续 writer。现有 rank 0–3、多根 dtype/shape 与进程协议承接结果，不增加运输格式，也不创建新线程池。

完整消费者闭包的私有执行由 `compact_path.cpp` 在原逻辑 DAG 上证明，不注册新的业务黑盒算子。共同投影在一批 B 个扫描组内共享；只保留 NAV、peak、maxDD 的 `3×B×P` 当前状态、一份 P 排序缓冲、单行资产收益和贡献状态。当前 B 为 `min(C,max(1,T/3))`，预算与物理工作量都计入实际批次数。fan/样本直接写独立 C/F 根，14 项统计及其他合法单组标量后处理克隆原节点并调用 canonical 子图。

Planner 和执行使用全部实际区间长度验证几何；未知消费者、公开全量中间结果、跨组后处理或不匹配区间按完整 canonical 预算执行。数值故障在原节点位置重放，不因数据异常转入未计预算的大图。私有子图关闭 pair/pointwise 融合时，其预算也关闭相同融合；子图 Audit 与布局同时存活的描述字节计入预算，较小请求释放旧保留容量。普通、shared/inline process 和 Hard Stop 沿同一 Scheduler；prepared 仍只支持原有 single lane。

## 3. 完整固定带数学链

[可执行消费者测试](../tests/test_mc_compact_graph.py)通过 GraphCompiler 构建实际 DAG：固定 tape → affine/expm1 → 权重资产归约 → NAV/peak/drawdown → fan、样本、14 个独立统计根及贡献。先覆盖固定 period，再覆盖共享固定 tape 的多组合。

概率向量是明确输入；T、样本上界 min(P,12) 是本模板构建时的字面配置。fan 初态也采用分位数插值，不能直接填 N0；贡献保持 T→P 的单次左折叠；ES 用末尾零/true sentinel 和安全分母表达，避免 eager where 隐藏除零或空选择失败。异质单位的统计保留不同具名根，不强行 stack。

该成功数值链明确排除旧 seed、coverage、协方差 root、随机变换、limits、JSON 及平台首故障/部分数组协议；这些仍在[完整承接规划](platform-compute-roadmap.md)中，不能因本轮数学测试通过而关闭。

## 4. Excel 并行后端

新增[独立 recipe](../cpp/excel_axis_primitives.cpp)和 checked 符号上界。prefix 显式展开 seed 与每步状态；quantiles 每条 line 共用透明 NaN-last rank 网络，全部 Q 复用，整数 rank 直接返回值。工作簿保存可审计中间步骤，不能按 C++ scratch 规则覆盖。

全工作簿预算包含输入、公式、辅助、状态、输出、参考与来源；预算预检不读取本次 C++ 结果决定展开量。典型 N=2m 的完整展开远超默认 200 万单元格预算，应在写出前拒绝，不通过拆表规避。

Excel 导出专用诊断检查被消费的插值中间项和全部 prefix 状态。C++ 的 Inf/次正规等有效输入若超出 Excel 域则明确拒绝导出，不改变普通 C++ 的数学契约；普通计算不执行 Excel 工作。登记实现、原生参考/预算测试、LibreOffice 复算和 Microsoft Excel 复算分别记录。

## 5. 前阶段验收（compact3，2026-10-03）

2026-10-03，候选构建 `0b0457dc35130e7ac9089c09a8426ce097757954388b4129b532dbb0bec3d6d4` 的 Python 全量 11,986 项通过，包含 130 项私有执行回归；原生及 ASan/UBSan 各 9/9，通过实际原生 compact 体测试。关闭 Eigen/Accelerate/Highway 的原生路径也为 9/9。源码、正式 wheel 与 worker 身份一致；这些证据限本机 macOS arm64。

同 CPU=1、六轮交替配对、全部 fan/14统计/样本/贡献、两代独立输出，NJIT 保留原固定 C 签名、cache/nogil 和 quantile `inline=always`。普通入口大型 8 组合为 1.781 秒，对照 3.074 秒；T5/P50k/F32/A32/C32 为 2.554 秒，对照 4.044 秒。单组合大型为 0.379 秒，对照 0.384 秒，**时间优势仍不明显**；微小用例增加约 0.089 毫秒只作诊断。prepared 结论一致，不把几毫秒差异作为显著收益。

九组 fresh 内存观测各用独立 heap/RSS 进程，共 18 个 PID。请求分配峰值计入构图、plan、工作区及两代结果；共同输入、固定签名编译和 tiny 预热在区间外。ordinary 热峰值 C1 为 **4.627/32.519 MB**、C8 为 **9.100/48.903 MB**（C++/NJIT），显著降低；短周期 32 组合为 **18.397/8.503 MB**，**完整 owner 口径仍有约 9.9 MB 的实质缺口**。短周期 setup 保留约 13.736 MB，后续优先定位图与执行对象，而不盲目增加组合批量。

ordinary 的 RSS 执行增量 C1/C8/短周期分别为 0.573/3.555/10.846 MB，NJIT 为 32.440/65.716/48.480 MB；RSS baseline 在 owner 构造之后，不能用这项较低增量抵消短周期的 heap 缺口。两类指标范围不同，allocator 未追踪释放及 0.5ms RSS 采样盲区保留，不宣称完整分配审计或全平台验收。设计、终态原始数据、数值/所有权验证及限制见[本轮证据](physical-execution-evidence/2026-10-03-mc-compact-integration/README.md)。

新增三项已有配方及符号预算、原生参考对照；LibreOffice 25.8.4.2 的三个新增夹具共 34 值通过补充复算。全空输出保留独立 EMPTY/错误诊断并纳入预算。Microsoft Excel 本轮真实重算尚未完成，历史 185/187 项证据不能抵扣新增三项。

## 6. 前阶段排序与元数据优化验收（2026-10-03）

完整 line 排序复用 Highway 1.3.0 VQSort；自动ISA、至少1024值才准入，NaN、任意负零及非零次正规保持原NaN-last路径，排名和插值不变。Planner每活动worker增加16KiB库栈/TLS保守估算，实际调用计数贯通子图及IPC21。完整恒等分支共享只读metadata/proof，仍校验结构；请求工作区和结果保持独立，原worker预算不因去重降低。

最终构建 `faf875f1`，同单核、六轮配对、原生产固定签名NJIT及全部输出：C1 **0.319/0.385秒**，C8 **1.280/3.074秒**，短周期C32 **2.156/4.015秒**。C8的F输出与prepared snapshot保留收益；tiny多0.091ms只诊断。

12个fresh heap/RSS进程：ordinary热请求分配峰C1 **4.569/32.519MB**、C8 **8.658/48.903MB**；短周期 **16.635/8.503MB**，完整owner内存仍未领先。相较前阶段，短周期setup及热峰各减少约1.76MB；compiler retained仍约8.47MB。原始失败负载、pytest导入成本与allocator/RSS盲区均保留，较低执行RSS不抵消owner缺口。

完整Python套件11963通过/59明确Eigen opt-in跳过，按XML精确补跑59项全通过，合计 **12022个不同用例全部通过**；原生、ASan/UBSan、三库OFF各 **9/9**。首次负零/FZ异常标志与错误捕获长度夹具已修复，保留失败记录。源码/wheel/worker身份一致，仅物理优化，Excel公式不变；新增Microsoft Excel重算及完整随机/业务链仍未关闭。[详细设计、完整原始证据与限制](physical-execution-evidence/2026-10-03-mc-sort-metadata/README.md)。

## 7. ParentSlots 设计与当前验收边界（2026-10-03）

本节保留ParentSlots的b8dc阶段设计和原始验收；最新构建及剩余边界见第8节，不把下列历史数字改写为当前结果。

### 7.1 修改依据与范围

第6节之后的原生逐项释放归因确认：短周期32组合图有 **7405个节点**，编译对象保留 **8,473,791字节**；NodeInfo数组本身占2,949,120字节，主Program、分支Program和compact后处理还分别保存节点描述。全部1,187,045次受追踪分配与释放配平，没有未追踪释放或表溢出。该测量定位编译owner的组成，不是本次修改后的内存收益，也不等同于完整请求峰值或RSS。

本次只优化节点父依赖的内部表示。原固定32槽数组随每份节点描述复制；当前canonical算子最大arity为8，但作用域捕获仍可达到32个依赖，不能通过降低上限或丢弃冷槽节省存储。不新增算子，不修改数学运算、错误与状态、参数绑定、selected roots、调度资源规则或Excel公式。

### 7.2 容器、依赖与所有权

[ParentSlots](../cpp/include/calmetrics_engine/graph.hpp)固定保留32槽的逐值契约，前8槽内联，后24槽由节点独占的`unique_ptr`按需保存。默认全部为0；未分配冷尾的const读取返回0，不申请内存，对冷尾的可写访问才建立完整且零初始化的尾部。实际arity仍由`parent_count`给出，父依赖顺序不变。

复制构造与赋值深复制尾部，赋值先完成新尾部构造再替换旧owner；移动转移尾部所有权。Program、NodeInfo和分支之间不共享可变父槽。恒等metadata准入仍逐项比较全部32槽，包括未被`parent_count`消费的值；未分配的全零尾部与显式保存的全零尾部相等，未用槽的非零差异仍阻止共享。作用域捕获去重、32依赖上限、拓扑检查及嵌套限制保留原规则。

### 7.3 协议与预算边界

[Program编码与解码](../cpp/compiler.cpp)保持原版本和字段顺序，仍写入实际`parent_count`及对应父节点值；[GraphNode、native_nodes和Python绑定](../cpp/graph_bindings.cpp)保留原逐值结果、pickle及独立快照生命周期。节点常量和其他Program/RootOutput字段不变，逻辑DAG仍是Excel展开的唯一依据。

这是内部C++源码和ABI表示的改变：父槽不再是`std::array<uint32_t,32>`，其类型、布局及连续迭代接口不保持原表示。本项目没有据此承诺已发布C++SDK兼容；公开Python和Program序列化协议保持不变，不能将这点扩大为任意外部C++消费者的二进制兼容承诺。

[compact元数据预算](../cpp/compact_path.cpp)在节点数组容量之外，逐节点计入`overflow_bytes()`的真实尾部ownership。即使C++手构Program只写入未用冷槽，也不能按arity推断没有尾部；这项payload计量不等于allocator头部或OS栈/RSS上界。执行Scratch、输入、输出和worker预算继续分别计量，不以内部表示缩小为由共享请求状态或降低既有预算约束。

### 7.4 当前正式验收与剩余边界

最终构建为`b8dc2f16aef13017d8652575f282cc6522992fdabb08e051c68634b76b26cb4d`，本机macOS arm64的完整Python套件 **12,071项通过、无skip**，已包含新增49项；[Python聚焦回归](../tests/test_compact_parent_storage.py)另跑 **49项通过**，不与全量重复相加。覆盖内联/溢出及32依赖边界、33依赖拒绝、raise/isolate、single/thread及inline/shared进程、pickle、节点快照和结果生命周期、只读输入与两代独立结果。原生、ASan/UBSan、三库OFF各 **9/9通过**；[原生回归](../tests/compiler_native_tests.cpp)包含全部槽默认值、深复制与赋值、移动、未用槽identity，以及真实32依赖Program编码往返与执行。源码、正式wheel与worker身份一致。

完整固定带成功数学链使用CPU=1、六轮交替配对、原生产固定签名热态NJIT及全部fan/14统计/样本/贡献，两代结果同时保留。普通入口中位耗时如下；它们复证完整链已有时间优势，不把ParentSlots单独解释为新的数值提速来源。

| 负载 | C++（ms） | NJIT（ms） |
| --- | ---: | ---: |
| T200/P10k/F4/A32/C1 | 318.968 | 384.762 |
| T200/P10k/F4/A32/C8 | 1282.510 | 3069.972 |
| T5/P50k/F32/A32/C32 | 2154.869 | 4015.338 |

C8的F输出和prepared snapshot分别为1280.156/3071.798ms、1286.246/3071.290ms；tiny多约0.090ms仅作诊断。12个fresh heap/RSS进程保留相同owner和结果生命周期，ordinary热请求分配峰值为：

| 负载 | C++（字节） | NJIT（字节） | 比第6节C++减少（字节） |
| --- | ---: | ---: | ---: |
| 大型C1 | 4,429,636 | 32,519,400 | 139,840 |
| 大型C8 | 7,948,801 | 48,903,400 | 708,936 |
| 短周期C32 | 13,821,031 | 8,503,168 | 2,814,336 |

同口径Python阶段的compiler retained从8,473,839降至 **5,659,511字节**；第7.1节的8,473,791字节是此前原生释放归因口径，不能直接替代这个阶段值。短周期完整请求虽减少约2.81MB，**仍比NJIT多5,317,863字节**，实质heap缺口没有关闭。

heap计入构图、plan、工作区与两代结果；共同物理输入、固定签名JIT和tiny预热在区间外。RSS baseline仍在owner构造之后，不能将执行增量称为工作区或用它抵消短周期heap差距；连续epoch中未追踪释放、非拦截allocator与采样盲区继续保留。本轮证据不构成完整分配审计或全平台验收。数学、错误、状态、参数、selected roots和Excel公式不变；**Excel 187–190的Microsoft Excel真实重算、完整随机流及业务请求仍未闭环**。[正式设计、原始验收与测量限制](physical-execution-evidence/2026-10-03-mc-parent-slots/README.md)。

## 8. NodeInfo 不可变类型共享设计（2026-10-03）

### 8.1 设计、字段与所有权

ParentSlots 后的实际原生 owner 分解仍有 2,228,224 B 的 NodeInfo 数组和 298,800 B 的逐节点类型 payload。正式 b8dc wheel 上核对短周期同一冻结图：7,405 节点只有 28 种完整类型说明。它证明说明重复，不把整个 NodeInfo 数组当成可节约量；必须重新测完整请求才能判断是否值得采用。

本项只共享 immutable 类型说明，不共享数值、模型拟合、状态、随机源或结果。`typed::ValueType` 的可变推导及所有数学规则保持，NodeInfo 保存 `shared_ptr<const ValueType>`，`type()`只读访问；默认 NodeInfo 仍读取原 scalar 类型。发布后的编译节点全部持有同次 Builder 的驻留所有者，默认标量静态值仅处理尚未发布的默认对象，不能成为全局图缓存。

Builder 驻留表按 kind、dtype、axes、shape、semantic_dimension、price_basis、record_tag、fields 全部字段哈希，碰撞仍用现有 `typed::same_type` 完整比较。表在编译结束释放；每个节点、GraphNode 快照、复制的 CompiledGraph/Excel Plan 保留其共享所有者，不依赖 Builder 寿命或原图存在。不同编译图没有全局驻留和无限缓存，unique-many 不使用逐项全表扫描。

迭代状态/计数与 group 输出需要改变 dtype、semantic 或 price_basis 时，先完成局部可变 ValueType 的组装，再驻留，只读前驱对象不可修改。CSE 键、节点 ID、输出来源、Program 传输版本和图指纹保持原逻辑，不把“类型相同”当作“计算相同”。Python 属性仍逐次生成独立字典，不能向外暴露可修改的共享对象；C++ NodeInfo 字段类型和布局会改变，不承诺旧 C++ SDK/ABI 接口。

### 8.2 正确性与公开图验收

最终构建为`97fdfe5267e55ea83caffc38b333919bcf4c206dc1d37e57eeec500fb86f247c`，源码、正式wheel与worker身份一致。本机macOS arm64完整Python **12,082项通过，无失败或跳过**，包含新增11项；ParentSlots及类型专项另跑 **60项通过**，不与全量重复相加。原生、ASan/UBSan、Eigen/Accelerate/Highway三库OFF各 **9/9通过**。

[公共回归](../tests/test_node_type_sharing.py)覆盖raise/isolate与single/thread/inline/shared进程、异构dtype/axes/shape、外部类型字典修改、pickle、图释放后的快照保活、两代独立结果及只读输入；同11项先在b8dc旧wheel通过，确认夹具沿用原行为。原生另外验证完整八字段、同值共享、跨图隔离、default、副本生命周期、group/iteration局部修改及512种独立类型。

同一短周期图的7405节点、28种完整类型，旧/新公开metadata、native_nodes及全部类型字典的SHA256均为`94fa298494e245767b2122bec320f08b63512004098877e82a4931e66a0071d0`，图fingerprint相同。它证明公开描述不变，不自行扩大业务或数学验收范围。

### 8.3 完整固定带时间与requested heap

CPU=1、六轮交替配对、冻结生产固定签名热态NJIT、全部fan/14统计/样本/贡献及两代独立结果，普通入口中位值如下。这里仍使用已经变换的固定随机带，图构造单列时间并计入heap；不包含完整请求前段。

| 负载 | C++（ms） | NJIT（ms） | C++ warm heap（字节） | NJIT warm heap（字节） |
| --- | ---: | ---: | ---: | ---: |
| T200/P10k/F4/A32/C1 | 318.138 | 384.430 | 4,352,356 | 32,519,400 |
| T200/P10k/F4/A32/C8 | 1280.240 | 3071.281 | 7,598,449 | 48,903,400 |
| T5/P50k/F32/A32/C32 | 2157.405 | 4017.753 | 12,415,063 | 8,503,168 |

C8的F输出为1282.151/3068.562ms，prepared独立快照为1278.325/3069.126ms；tiny多0.091ms只诊断。完整链已有的时间收益保持，不把几ms跨阶段变化解释为类型共享带来的显著提速。

12个fresh heap/RSS进程保留完整owner与结果生命周期。短周期warm heap相较第7节减少 **1,405,968字节**，仍比NJIT多 **3,911,895字节**；compiler retained从5,659,511降为 **4,253,543字节**。原生拆解的对应值为5,659,463→4,253,495字节，48字节口径差保留；不能混用两种阶段计数。NodeInfo数组减少1,114,112字节，类型payload减少291,856字节，合计与实际峰值减少量一致，不将整份数组当作节约量。

heap连续epoch保留构图、plan、pytest/helper导入、工作区及两代结果；共同输入与微型固定签名JIT预热在区间外。未追踪释放可能来自epoch前的合法owner，不证明泄漏，也不能据此宣称完整allocator覆盖；CPython arena、mmap及非拦截分配器盲区仍存在。

### 8.4 RSS复测与类型多样性负控制

原正式短周期RSS baseline在owner构造后；旧/新绝对peak为 **217,333,760→221,659,136字节**，增加4,325,376字节，保留这项单次观测。随后8个相邻fresh PID补充测量把baseline前移到owner构造前，包含3对旧/新native及2次NJIT，共同输入和预热仍在epoch外。

三对native的新减旧绝对peak为 **+3,227,648、+737,280、−6,258,688字节**；旧范围219,496,448–227,229,696，新范围220,971,008–224,935,936。差异双向且范围重叠，**没有证明稳定的RSS增加或降低**。NJIT两次peak为242,581,504/247,627,776字节；本范围内较低的native peak不证明全负载、全机器的resident-memory优势。不能用RSS中位数或执行增量关闭requested-heap欠项。

4096种独立合法schema负控制的retained为23,876,664→24,040,504字节（**增加163,840字节**），peak为25,486,864→25,865,912字节（**增加379,048字节，约1.5%**）。compiler和graph仍在epoch内并保活，调用方schema在epoch外；单次时间增加1.394ms只诊断，没有执行数值或与NJIT比较。该成本证明不能宣称所有图都因驻留获益。

### 8.5 剩余范围与下一步

本项不改算子数学、Excel公式或预算展开口径，现有Excel回归只复核类型读取与图复制。**短周期heap、稳定RSS优势、完整随机流/业务请求及Excel 187–190的Microsoft Excel真实重算仍未闭环**。固定带成功范围尚未包含coverage、旧随机流、协方差根、因子变换、limits、JSON及按平台阶段排序的首故障prefix。

下一步扩大真实计算请求：先承接原始固定normal/chi带的因子变换与coverage，再核验投影先于NAV、组合先后次序的首错；旧seed和成熟LAPACK根分别冻结契约验收，limits及JSON保持调用方业务解释。使用已有数学原语和可编辑组合，不新增模拟业务黑盒，不继续以几KB/ms的owner微调代替完整链闭环。详见[本阶段全部正负证据、原始记录及限制](physical-execution-evidence/2026-10-03-mc-node-types/README.md)。

## 9. 原始固定带前缀与完整请求扩展

### 9.1 真实缺口与实现顺序

第8节的97fd正式基线从已经变换的带开始。该历史版本的compact producer要求资产`affine_contract`的source为直接输入（`input_axes=2`）；前接rotation、scale和means会走canonical执行并物化完整transformed张量。这是97fd的覆盖边界，不再代表当前源码的唯一准入方式，也不能把该历史时间和内存结果沿用为新前缀的证据。

当前[compact producer匹配与执行](../cpp/compact_path.cpp)已增加已有数学节点的私有原始带前缀，承接rotation、scale和means后进入原compact消费者链。第9.6节记录冻结平台源码、固定签名及Numba版本后的只读原始固定带成功链，以及当前12,267个不同Python用例的等集PASS复验。coverage、按组合/阶段排序的首错、limits/JSON、旧随机流与协方差根继续分别验收。各阶段独立保留实际覆盖，固定原始带重放不包含随机生成和全部服务逻辑。

### 9.2 可编辑的数学DAG与异常差异

公开逻辑图使用以下现有算子组合，私有路径保留其计算树：

```text
rotated = affine_contract(raw, transpose(root), +0)
normal_transformed = add(broadcast_like(means, rotated), multiply(rotated, 1))
student_scale = sqrt((df - 2) / df) / sqrt(chi3 / df)
student_transformed = add(broadcast_like(means, rotated),
                          multiply(rotated, broadcast_like(student_scale, rotated)))
returns = expm1(affine_contract(transformed, transpose(beta), initial))
```

其中transformed选择对应的normal或Student表达式。公开raw轴为`[time,path,component]`、形状`[T,P,I]`，root轴为`[factor,component]`、形状`[F,I]`；I与F可以不同，beta轴为`[asset,factor]`并收缩F。`component`是现有公开轴词汇，不新增innovation轴；原平台协方差方根本身仍为方阵，其成功链验收不能由矩形小样本代替。normal不读取chi或df公式，但保留rotation乘1后再加means。均值不能作为第一层affine初态，也不能预合成root和beta，这两种改法会改变原计算树。chi3是原`[T,P]`的`[T,P,1]`接口视图，轴为`[time,path,factor]`；右对齐广播不能用现有插入长度2轴的stack代替。此视图的零拷贝需核对实际buffer、只读性与寿命。

FIRP异常行为的全域等价仍未闭环。源码和当前Numba lowering显示：原transform签名是`void(float64[3D,C], float64[2D,C], float64[2D,C], float64[1D,C], int64, float64)`，只设置cache/nogil，默认python error model。合法Student df下，实际除零条件是计算得到的`sqrt(chi/df)`为±0，不能只检查原始chi为零；正/负最小subnormal在df=5等合法参数下也可能下溢为±0并抛ZeroDivisionError。`chi/df`仍为负非零或-Inf时，libc sqrt形成NaN；chi=NaN传播，chi=+Inf使scale为0。公共服务在调用前拒绝df<=2，直接内核边界不等于服务边界。normal不消费chi。

CME canonical sqrt对负值返回DOMAIN_ERROR。当前私有路径以形状和完整依赖证明准入，**没有finite-positive的数值准入限制**；Student的subtract/divide/sqrt等scale节点按原拓扑顺序执行完整canonical pass，保留CME对零、负值、NaN、Inf、次正规下溢和df的既有结果或错误。有限正chi是与FIRP成功路径比较的已选范围，不是本实现额外设置的金融或输入约束。CME私有/canonical异常一致不等于FIRP负chi产生sqrt NaN的行为已复现；后续须通过显式可编辑组合或确有缺口的最小数学原语保留源行为，不统一拒绝或修补这些输入。eager依赖不能先执行未选分支的非法sqrt，节点错误顺序也不能冒充投影/NAV的业务首错。

### 9.3 有界前缀融合与成熟库

当前私有不可变前缀证明接受经核对的affine→multiply→add、normal常量1或Student scale及只读广播依赖，并检查全部消费者。公开逻辑节点、根、metadata、Excel可编辑步骤和Program编码不变；中间tensor被选为公开根、有额外消费者、未支持作用域或isolate错误路径时继续走canonical实现。前缀证明、工作量和完整workspace估算已接入原compact预算，worker从同一逻辑图重建证明；进程审计增加原始带工作量字段，不能只凭Planner reason判断实际执行。

通过[共享线性行计算](../cpp/include/calmetrics_engine/affine_projection.hpp)复用现有Eigen独立列计算、同一个ordered fold与系数packing；rotation使用不带expm1的线性行函数，资产projection使用其expm1 epilogue。初态先加、输入收缩轴升序、乘法与加法分开，随后保留rotation×scale再means左加。SIMD沿独立输出展开，不新增水平归约或FMA。成熟库的收益须由实际命中和完整负载证明；已有packed GEMM的M≥1024/K≥256/N≥1024条件并不覆盖F=4/32，不能据“已有GEMM”宣称加速。

raw、root、means、beta和chi视图只读；每个执行的rotation行、`[T,P]` scale、组合状态、系数pack及结果由现有workspace和结果owner持有。scale各节点完成整个pass后复用已死的私有存储，保留数值检查顺序；系数只在本次执行内按需要pack并供多个区间复用，不缓存跨请求数值。上述容量及私有描述信息计入预算，不新增池，也不覆写公开输入或前一代独立结果。内部生成、无额外消费者的独占带是否可原地复用仍需另做liveness与所有权设计。

执行先对**全部interval的结构、实际形状及所有结果slot容量/边界**预检，失败时不写结果。随后按interval处理数值：Student完成该interval的全部scale canonical pass后，才写其compact消费者和结果；不把后续interval的数值检查全局前置。进入已准入的数值执行后，错误按canonical抛出，不能再以回退掩盖失败。这个规则不表示全部消费者的数值错误都可在首个写入前确定，也不关闭平台阶段首错缺口。

同步修复[required_array_capacity](../cpp/graph.cpp)的空数组容量推导：选择父节点形状时先比较rank，只有rank相同才比较size。已知空数组不再因为元素数0小于scalar的1而被降为rank0；未知几何仍按既有规则保留其未知状态。该修复不改变数学广播、canonical执行或Excel公式逻辑；新增通用空轴120项和当前全部Python用例已按第9.6节复验，原始失败与仅修正sum(empty)夹具的记录保留。

### 9.4 多组合的真实时空权衡

97fd从直接transformed输入开始；105226原始带版本以`B0=min(C,max(1,T/3))`为组合batch，只有B0已覆盖全部C个组合时才融合前缀。这两阶段的实测仅适用于各自版本，不是当前短宽路径的证据。

85/046阶段的形状成本规则在全部raw/rotation/means/scale及消费者几何验证后执行。T为interval步数、P为路径数、F为变换后factor数、A为资产数、C为已证明的组合数；从B0开始，当以下条件成立时允许全部C的单batch。当前大状态批次策略保留此准入证明，并按第12节选择实际batch：

```text
B0 = min(C, max(1, T / 3))       // 整数除法
(C - B0) * (3 * P + A) <= T * P * F
```

左侧是扩大batch新增的NAV/peak/drawdown及资产贡献累加器double数，右侧是一次完整transformed张量的double数。A贡献状态不能省略。实现以有溢出检查的extent计算和等价整数除法比较，Planner与执行共用同一shape binder，准入后全部workspace仍计入预算。85/046阶段满足条件时每个step/path行只做一次rotation；当前小状态或长扫描仍保留此行为，大状态短周期按第12节有界重放。成本不满足时走唯一canonical图。选择不读取payload，进入数值执行后的canonical错误不会触发另一种成本回退。

例如T5/F32/A32/C32时，P7的新增状态1643个double大于transformed的1120，拒绝前缀；P17的2573不超过2720，可以通过本成本条件，但仍须通过其他结构、预算和执行校验。真实T5/P50k/F32/A32/C32负载也满足该成本条件。全部组合同时推进约需`3*C*P*8`字节的NAV/peak/drawdown状态，C32/P50k约38.4MB，另有贡献、排序和结果；一次物化`[T5,P50k,F32]`transformed需64MB。这些是设计规模和准入依据，不能记作本轮实测节省或性能收益。

精确fan分位数需要每个组合、每个时点完整的路径分布，不能用任意小tile同时宣称免重算和免存储。本轮需分别验收原本B0=C及新增成本准入的短宽路径，并保留成本拒绝负载。当前actual审计分别记录`compact_path_raw_rows`（各interval的ΣT×P，非C倍）、`compact_path_raw_scale_elements`（normal为0，Student为ΣT×P）及`compact_path_raw_pack_bytes`（实际资产beta和rotation系数packing）。pack计数已包含在`algorithm_copy_bytes`中，是其明细而非新增一份复制；无需pack时为0。实际计数和workspace容量须与Planner估算、输入布局及请求内容变化一起复核。

### 9.5 平台阶段顺序与验收

平台每组合依次执行coverage→全部projection→全部NAV→fan→limits；projection按step/path、资产/factor升序扫描，NAV按path/step。融合若早遇NAV故障，必须保留有界候选并完成其前置projection的合法性扫描，再按组合、阶段和该阶段坐标选首错。projection故障的minimum_return按原step/path前缀及原检查顺序计算；核验NAV内核诊断时另保留其path/step前缀，不能混成统一的扫描顺序，也不能纳入首错之后的极值。原DAG节点重放不等于已证明平台阶段首错。

coverage应保留原gross、abs(w)>1e-15、available==1、ratio+1e-12、covered_net、net/single/gross检查和tiny权重字段语义。不可把全部组合coverage前置而抢先于前组合NAV错误；也不可用零权重跳过已映射资产的0×Inf。投影对unavailable资产的真正跳过需保留，未消费资产不能因极端intercept或NaN而新增错误。limits仍由比较/归约原语组合，金融状态和JSON解释由平台负责；当前MC limits没有路径首触发日期，不能新增该输出语义。

验收同时保留两个入口：只读固定带重放与平台独占、原地变换随机带。只读重放下NJIT所需copy可计入该入口，但不能冒充原平台本来就有的成本；计时外reset、原始带或保留缓冲也计入相应owner内存。随机生成、root构造尚未纳入时继续明确排除。完整请求验收最终须从内部带生成起计入相应分配与复制，不能把共同64MB放epoch外后宣称全请求内存获胜。

normal/Student均验证transformed、returns和全部最终输出；同核数交替计时、fresh PID内存和两代结果。冻结C1/C8/短宽C32及tiny，tiny仅诊断。异常固定样本包括chi±0/负值/NaN/±Inf、正/负subnormal下溢与计算分母为零、normal未消费chi、df服务与内核边界、早NAV溢出加晚projection非法、前组合NAV错加后组合coverage错、mapped零权重×Inf、unavailable极端输入、minimum_return首错前缀及JSON None。另覆盖C/F/非连续/负stride、参数变化、single/thread/process、取消、预算和结果寿命。任何失败都按真实数学、错误、所有权和完整负载证据处理，不能用成功链局部提速关闭本节全域目标。

### 9.6 当前只读原始带重放证据与剩余边界

本节当前证据限定于**immutable readonly raw replay**：固定raw/chi输入，经transform→projection→NAV→fan→14项统计→样本（最多12条）→资产贡献，返回全部选定根。它排除旧随机流、协方差root生成、coverage、limits、JSON包装及平台按组合/阶段选择first-fault，不是完整平台请求验收。

#### 9.6.1 构建、数学与所有权

当前构建为`85d2003fc8c93506d85d561346577e4894097f7aff00cd36d715edd26eff4581`，以`candidate-wide-final/identity.json`核对源码、编译配置、安装二进制和wheel；wheel SHA为`32a2ab82d4dc5a162b0f13e7f0c786a6752c00879812ba1aa7b4731834a803d7`。每份驱动报告在测量前后复核实际身份，不能把旧97fd或105226结果当作此构建的验收。

[原始固定带公共回归](../tests/test_raw_factor_prefix.py)提供`raw_factor_spec`和小型独立参考。当前专项**65/65通过，无skip**；原生、ASan/UBSan、完整禁库OFF各**9/9通过**。专项覆盖normal/Student、I与F不同、C/F及非连续/负stride、多interval、prepared内容和参数变化、两代独立结果、single/thread/process、selected roots、公开transformed根回退、短batch与贡献状态成本拒绝、全batch和预算边界。零/负chi、最小正次正规下溢、NaN/±Inf和df边界核对CME私有/canonical结果与错误，不据此声明FIRP异常等价。首版55项通过的历史记录保留。当前collect与互不重复的PASS记录精确匹配：**12,267个不同Python用例通过，无skip**（未改旧用例12,082、raw专项65、空轴专项120）；不是一次单体全绿运行。首次沙箱运行11,459通过/743失败，允许共享内存后精确重跑743项为631通过/112失败；112均为新增空样本夹具把sum(empty)误写为零，修正夹具后120项全部通过。原生产INSUFFICIENT_SAMPLE契约与构建未改，原始失败和复验分别保留。

NJIT参考从FIRP `backend/scenario_stress/numba_kernels.py`提取transform及原投影/NAV/quantile/fan五个函数，保留函数AST、固定dtype/C布局signature、cache/nogil及quantile的`inline=always`；源SHA为`bb53056653d697c698dab4ac5e716d96dbaec9071157d755c3fbef32e9831b79`，实际使用Numba0.60.0、NumPy1.26.4。不导入平台runtime，完成warm/JIT后禁用新编译。全部最终根核对status、shape及dtype，离散概率精确比较，浮点使用`rtol=2e-12, atol=2e-13`；输入指纹不变，原生审计确认无Python算子或回退、无请求期JIT并实际命中前缀。正式驱动验证旧compact出口值稳定及全部typed根不别名，65公共专项另保存原始named values（含0D scalar）逐旧代核对shape/dtype/有限值IEEE位元和只读寿命；这是组合证据。NaN仅核对分类，prepared多代和释放后的旧metadata未全量保存复核，不能宣称所有非有限位元或metadata逐代稳定已全面验收。

#### 9.6.2 同核数完整成功链时间

下表是CPU预算1、普通Scheduler入口、C布局结果、6轮交替首跑的中位数，单位ms。输入生成、构图与初始warm/JIT在计时外；每次执行保留前一代完整结果。C1/C8为T200/P10000/F4/A32，短宽C32为T5/P50000/F32/A32。97fd列是同一raw replay图在旧引擎上的历史结果，当前C++/NJIT比值只使用85d的配对结果。

| 负载 | 分布 | 97fd C++（历史） | 85d C++ | 同轮NJIT | C++/NJIT |
| --- | --- | ---: | ---: | ---: | ---: |
| C1 | normal | 1614.709 | 349.923 | 406.335 | 0.861 |
| C1 | Student | 1635.133 | 355.248 | 406.887 | 0.873 |
| C8 | normal | 9412.604 | 1310.518 | 3088.058 | 0.424 |
| C8 | Student | 9572.003 | 1325.429 | 3088.493 | 0.429 |
| 短宽C32 | normal | 4302.445 | 860.570 | 4091.820 | 0.210 |
| 短宽C32 | Student | 4214.991 | 858.044 | 4084.319 | 0.210 |

本范围内C8约为NJIT的2.3倍速度，短宽C32约4.8倍；C1的优势约13%–14%，不据此声称所有大型算法均有同等加速。T3/P8/F4/A3/C1小任务的normal/Student C++约0.102ms、NJIT约0.013ms，差约**+0.09ms**，只记录诊断，不作为性能门禁。8份时间记录均正常结束并完成数学/所有权核对，不能把小任务诊断混作8项严格提速门禁。

105226原始带版本仍拒绝短宽全batch：normal为4465.716ms对NJIT4091.943ms，Student为4448.215ms对4082.861ms，分别慢约9.1%和8.9%。这些历史失败保留；本轮85d通过第9.4节的形状成本准入复验，而不是放宽数学或性能要求。

**只读重放的复制成本必须单列。** 原NJIT transform原地改写draws，故本不可变raw replay入口每次在执行内复制完整64,000,000B raw；C++借用同一物理buffer的只读视图，未为两个后端另外复制共同输入。原平台RNG生成的独占raw已经可写，原地transform不支付这一额外重放copy。本表包含copy的收益适用于约定重放入口，不能直接泛化为原平台独占随机带的完整请求提速。

#### 9.6.3 fresh PID请求内存

下表为连续requested-heap epoch的warm峰值，单位B；**完整输入创建、构图/plan owner、cold、5次warm及替换时的两代完整结果均在epoch内**。每个后端/metric使用fresh PID，最终跨后端数值对比及大结果hash在epoch后。列出的peak已包含共同输入，不能再加各阶段peak；累计`allocated_bytes`是多次执行的分配总量，不是同时驻留峰值。

| 负载 | 分布 | C++ warm heap峰值 | NJIT warm heap峰值 |
| --- | --- | ---: | ---: |
| C1 | normal | 68,372,142 | 160,520,504 |
| C1 | Student | 100,383,731 | 176,520,504 |
| C8 | normal | 71,710,243 | 176,906,472 |
| C8 | Student | 103,785,336 | 192,906,472 |
| 短宽C32 | normal | 114,078,041 | 136,527,744 |
| 短宽C32 | Student | 118,351,518 | 138,527,744 |

6对heap在此测量口径下均降低峰值；共同逻辑输入约为normal64MB、C1/C8 Student80MB、短宽Student66MB，额外权重、参数及owner仍由实际记录计入。旧97fd C8 normal的同入口C++ heap峰值4,854,327,635B是**旧CME canonical张量物化**的实测；它不是原平台NJIT曾物化4.85GB的证据。本轮六对heap与一对RSS、另含旧基线的13个wrapper任务共15条fresh-PID metric记录，均正常结束并完成结果/所有权核对。

heap probe没有table overflow，但存在6–8次C++、0–7次NJIT未跟踪free；其中可能包含epoch前分配、epoch内合法释放，不能据此判断泄漏，也不能声称完整allocator覆盖。它不覆盖CPython arena内部子分配、mmap及未拦截allocator，因此该表是requested-heap证据，不能冒充全进程物理内存或纯workspace上界。

RSS仅测C8 normal的一对fresh PID：C++ baseline/peak为129,531,904/206,061,568B，增量76,529,664B；NJIT为133,709,824/327,467,008B，增量193,757,184B。baseline在完整输入和owner构造前，绝对峰值含解释器和库；heap与RSS来自不同PID，不相加，RSS增量也不是workspace。0.5ms采样及idle control不证明精确捕捉全部瞬时峰值；一对观测不证明各负载、各OS或反复运行的稳定RSS优势。

#### 9.6.4 未关闭项与证据归档

当前实现是已有数学DAG的私有执行优化，不新增金融算子或改写logical公式。FIRP负chi的sqrt NaN行为、按组合/阶段及扫描坐标的first-fault、coverage、旧随机流、协方差root构造、limits/JSON和完整随机请求仍需独立复验。Excel继续展开logical DAG，普通C++不触发导出；Excel 187–190的Microsoft Excel真实重算及原始带组合的完整预算/重算证据仍未闭环，不能沿用既有185项历史PASS。

本轮原始脚本、源码/二进制身份、当前测试覆盖核对、65项和120项专项及native/ASan/OFF日志、配对时间、heap/RSS与历史失败从[原始证据](physical-execution-evidence/2026-10-03-mc-raw-prefix/README.md)进入。归档只保存本节实际范围，不能关闭第9.5节的完整平台请求目标。

## 10. 独占随机入口：历史候选设计（未实现）

### 10.1 目标、已确认事实与验收边界

本节保留此前探索的历史候选设计，**六个拟定RNG原语、生成轴规划、私有RNG融合和Excel配方均未实现**。当前以已有算子和组合优化为主；新增算子须直接服务于优化目的，不能为扩大功能覆盖而新增。下列候选尚未进入实际开发或完整链验收，不作为本轮应补功能。复用成熟原生库、减少重复计算、大块复制及存活内存，重点取得秒级耗时和大额内存收益；已有190个算子及其编号不变。第9节的85d生产证据保留原适用版本。独立AOT筛选已经核对NumPy 1.26.4的MT19937原生源和Numba 0.60.0的分布薄层；它证明所选内核在筛选样本中的序列兼容，不证明Typed IR、Planner、进程协议或完整owner内存已验收。原生RNG实验收益只作诊断，不代替本轮完整链验收。

历史候选曾考虑由引擎内部生成随机数，以消除完整raw带；这不是当前入口或开发承诺。**本轮完整owned对照的双方均使用同一NJIT平台prep RNG**，计入实际生成带、chi及其owner生命周期，再比较后续已有算子和组合的执行；不替换其中一方的随机算法，也不给NJIT额外增加64MB只读重放copy。缓存、图、结果及库工作区同样计入。原生RNG单内核实验不能用来宣称该完整入口已经提速或节省内存。

已读取的FIRP `backend/scenario_stress/numba_kernels.py`事实如下：固定签名`seeded_factor_draws_kernel`先seed，然后按time→path→factor生成**全部normal**；Student再沿time→path在同一随机流生成**全部chi-square**，其值为`2*standard_gamma(df/2)`。normal的chi是1×1占位。随后原地transform全部draws，再按组合顺序进行coverage、完整projection、完整NAV、fan和limits。空normal输出仍执行seed；F=0且T、P非零时，Student的chi生成不能因为没有normal而跳过。

原始序列来自Numba的NumPy随机状态，不是NumPy `Generator`的新分布接口。复用成熟MT原生实现，同时保留uniform53、polar normal和Cheng/GS gamma的薄层顺序；不得以统计分布相同替代固定seed的序列相同，也不得静默改用`std::normal_distribution`、`std::gamma_distribution`或其他随机算法。

### 10.2 历史备选API与可编辑组合

以下六项是历史候选API，均未实现、未获当前新增授权，不是现有190个算子的缺口清单或待办。它们曾用于讨论如何保持producer链可编辑，只表达确定性的随机状态转换、分布和投影，不包含MC业务、协方差root、金融df限制或完整情景黑盒。以下签名、协议和约束均属于该备选方案，不改变现有canonical opcode及1–190编号。

| 拟定签名 | 数学语义和输出 |
| --- | --- |
| `rng_seed(seed)` | uint32域整数seed，建立算法版本固定的初态；返回内部nominal RNG block，draw数量为0 |
| `rng_normal(state, spare, count, word_budget)` | 从显式checkpoint生成count个标准normal；返回values和成功后的checkpoint所在的内部block |
| `rng_gamma(state, spare, alpha, count, word_budget)` | 从显式checkpoint生成count个标准gamma，shape为alpha、scale为1；返回同样的block |
| `rng_state(block)` | 投影独立的`int64[635]`控制状态，静态`component`轴、state语义 |
| `rng_spare(block)` | 投影缓存normal的float64标量；与整数控制状态一起构成完整checkpoint |
| `rng_values(block[, d0[, d1[, d2]]])` | 投影float64 draws；无dims时为component向量，1/2/3个dims分别声明component、time/path、time/path/component形状 |

历史候选中，`seed`为有限整数，范围0至2³²−1，不将非法seed掩码后接受。`alpha`须为有限正标量；alpha>1还须保证Cheng setup中的`sqrt(2*alpha-1)`及`alpha+sqrt(2*alpha-1)`有限，不能宣称所有有限正shape均已复现。当前筛选在取词前对此拒绝code3，包含count=0；例如alpha=1e308仍有限，但2*alpha溢出会使原接受表达式失效。候选API的算术域约束及对应INVALID_PARAMETER不能用无尽抽样或budget耗尽替代。gamma保留源算法的scale=1乘法步骤；其他scale由普通multiply组合。`count`、各dim和`word_budget`拟为编译时有限非负整数字面配置，仍受现有精确整数、地址、乘积和请求资源上界限制；它们不是任意长度输入数组，也不是为推导shape构造的占位张量。dims乘积必须精确等于count，空轴仍保留声明rank。

下面仅示意历史候选的构图关系，不能在当前引擎执行。T×P×F、T×P及预算须先由调用方检查并展开成字面整数；不是新增一套赋值DSL语法：

```text
seed_block = rng_seed(seed)
normal_block = rng_normal(rng_state(seed_block), rng_spare(seed_block),
                          literal_normal_count, literal_normal_word_budget)
raw = rng_values(normal_block, literal_T, literal_P, literal_F)
gamma_block = rng_gamma(rng_state(normal_block), rng_spare(normal_block), df / 2,
                        literal_chi_count, literal_gamma_word_budget)
chi3 = multiply(rng_values(gamma_block, literal_T, literal_P, 1), 2)
```

normal请求不构造gamma分支。Student显式依赖normal的完整next checkpoint，确保全normal先于gamma，而不是逐行交错normal/chi。之后继续使用第9.2节的rotation→scale→means→asset affine→expm1及既有消费者图；不合并root×beta、不将means放进rotation初态。covariance root、coverage和limits由各自可编辑图或上游适配负责。

### 10.3 历史候选checkpoint表示

历史候选checkpoint拟采用**635个int64控制字段加一个float64 spare标量**，不是当前生产协议。每个int64槽只保存uint32范围内的值，64位计数用低/高32位limb表达，避免经过double参数、packed carrier或Excel时损失整数精度。拟定schema v1如下：

| 控制向量槽位 | 内容 |
| --- | --- |
| 0 | magic `0x434d4552` |
| 1 | 状态schema版本1 |
| 2 | 算法版本ID1：`mt19937_uniform53_np_polar_cheng_v1` |
| 3 | MT pos，0至624 |
| 4 | has_spare，0或1 |
| 5–6 | 已消费MT uint32 word总数，低/高limb |
| 7–8 | 已返回normal总数，低/高limb |
| 9–10 | 已返回gamma总数，低/高limb |
| 11–634 | MT key[624]，原顺序 |

`rng_seed`设置pos=624、has_spare=0、全部计数=0、spare=+0。缓存命中时只清has_spare，不清spare的旧值，保留Numba的checkpoint行为。Gamma不读取或改变normal缓存，必须原样携带它。加载检查完整schema、算法、pos、flag、limb范围、key几何及非全零key；计数相加使用checked逻辑，不静默回绕。

独立筛选的**632槽bit checkpoint不是生产协议**：其0/1为magic/version，2–625为MT key，626/627为pos/flag，628/629为cached-double的IEEE低/高bits，630/631为MT word计数。当前`load`无论has_gauss是否活动都验证cache finite，gamma和空输出也经过同一loader；这与后述仅在真实消费时验证的备选deferred规则有区别。147项独立native/ASan检查覆盖该632协议的chunk、边界、alias和checkpoint事务，不是635候选协议或engine-native验收。

635候选协议增加算法ID及normal/gamma计数，缓存float直接保留，不将cached-double bits混入整数向量；它未实现，新增字段、packed投影、独立spare及loader规则不能继承632协议的PASS。该方案要求C++内存与IPC保留float64位元；Excel尾差按浮点容差验证，flag、MT字、接受判定和全部计数仍要求精确。这个表示决定不证明Excel目前能在所有域重现相同分支。

借用现有GMM/eigen的nominal packed-matrix模式承载内部block，拟定float64矩阵shape为`[1,636+count]`、tag为`rng_block_v1`、具名字段为values/state/spare：前635项是精确可表示的控制整数，第635项是spare，其后为draws。它是明确版本化的内部carrier，不宣称现有`ValueKind::record`已支持任意异构数组record。

`rng_values`可借用draw区域的只读视图并延长block的last_use；`rng_state`须转换并写入真正的int64缓冲，禁止double/int64指针reinterpret；`rng_spare`复制标量。nominal block不能直接作为公开根，必须先投影。公开state与values结果具有各自独立owner；序列化的checkpoint可使用现有静态ModelPayload输入，不把nominal carrier或时间字段直接塞入ModelPayload。后述split/stream执行可省carrier物化，但必须提供与canonical布局、dtype、错误和结果所有权相同的证明。

### 10.4 连续抽样、纯依赖与有限预算

Normal沿用uniform53的两次uint32取词、polar拒绝条件`r²>=1或r²==0`，返回第二个normal并缓存第一个，最后保留源`0+1*value`步骤。gamma的alpha>1保留Cheng：先接受范围内u1，再抽`1-U`的u2，两个接受条件保留源短路顺序；alpha=1保留指数变换，0<alpha<1保留GS。MT seed、twist、temper、uniform53及分布表达式不能用fast-math、FMA或“等价”代数变换改写后免验收。

- count=0不取随机word、不推进pos/cache/计数，返回原checkpoint和正确rank空values；独立参数和结构校验仍执行。当前筛选严格验证spare finite，即使has_spare=0、gamma或空输出；候选协议若保留这一loader，它就是完整状态合法性约束。“仅在真实消费时检查spare数值”的deferred备选与此不同，需要独立固定非活动NaN/Inf、gamma和空输出契约、版本及回归，不能称为当前筛选行为或已验收优化。
- 奇数normal数量留下缓存；下个normal块可零word成本消费它。normal→gamma→normal混流保留同一个cache与MT流。顺序chunk的values拼接、最终控制字段和spare须与一个完整块一致。
- 同一显式state、spare、全部参数和算法版本确定同一结果。两个消费者使用同一checkpoint表示显式fork/replay，不表示隐式连续抽样。CSE仅在完整纯依赖相同且所有权成立时合并；不同seed/stream/checkpoint不可因为count或shape相同而合并。
- prepared只保存计划、描述和私有scratch，不保存跨请求可变随机cursor。重复执行同一seed/checkpoint应重现序列；推进流由调用方显式传入next checkpoint。线程、进程和区间不得自行加worker ID/reseed。
- word_budget计本块允许新增的MT uint32 word数；uniform53每次需2个，normal cache命中需0个。候选逻辑先检查完整取词所需资源，再推进内部状态；当前筛选逐word检查，不继承其检查粒度作为候选协议PASS。成功的next checkpoint只在完整块成功后发布；失败、取消或超时不改调用方state/spare，不向外发布半checkpoint。筛选已观察到uniform请求count5、budget4失败后out保留前两个有效值，后3项仍是sentinel；失败后必须废弃整个私有out并禁止成为公开结果。这不是外部output buffer逐字节原子写入的承诺，输入与next state事务性也不等于out未被部分写入。

原polar/Cheng拒绝采样的理论最坏取词数无界。这里的word_budget是**显式、版本化的新资源契约**：保留预算内的原算法和序列，超预算明确终止，不能声称它与旧无界算法在失败行为上完全相同。计数overflow及非法几何必须明确拒绝；拟定`RNG_BUDGET_EXCEEDED`按现有资源预算的整请求失败策略处理，不伪装成NON_CONVERGENCE或数值isolate状态。

shape/dtype/schema等独立检查不因上游数值故障跳过；已知geometry在失败传播时保留，未知payload不读取。非法seed、alpha和状态数值使用规范化的INVALID_PARAMETER及现有raise/isolate规则；spare非有限值按最终声明的严格loader或deferred规则验证，两者不得混称。公开values/state/spare的shape不能因为producer失败而丢失。实现前用错误用例固定这些新原语契约，不修改旧算子的错误分类。

随机sample消耗word数量不固定，不能按sample数量简单discard后并行分块。并行或replay只能从实际前置转换得到的完整checkpoint开始；position pass自身的取词、成本和scratch也要计入执行审计。chunk budget允许分开配置，完整块等价验收必须确保各块预算足够，并比较总实际取词，不能把不同预算失败位置称为序列不一致。

### 10.5 历史方案涉及的AST、Planner与进程协议

历史方案会涉及新增canonical family及Typed IR推导，literal配置进入节点完整身份；packed tag和投影几何沿用现有严格解析方式。它需要Compiler dispatch、borrow-aware last_use、array capacity和direct-root writer识别新family。候选`rng_values`的空轴使用rank优先几何，不把state的635项或scalar的1项误当成draw shape；这不扩张现有广播契约。上述改动均未实施，不在当前范围。

当前Planner有temporal输入时按其geometry确定可用范围；没有temporal输入时`typed_geometry`使用请求ends的最大值，并不存在由随机producer的literal dims决定的生成time域。历史方案需要独立的**生成time extent**，由compiler、Planner、request geometry和worker统一验证，不能把既有无temporal输入规则当作生成域验证。其候选约束如下：

1. canonical完整生成域为已声明T/P/F，先生成全部normal，再生成全部gamma；interval消费其声明time视图，不重新seed，也不按interval交错抽样。
2. 同一产品/request中相同纯producer可共享一次生成或等价checkpoint执行；不同产品/流的checkpoint与所有权独立。返回next state代表已声明完整block的转换，不是某个消费者恰好读取的slice。
3. start/end按相应生成time域检查，多个time源需显式兼容；没有声明生成轴的旧图仍使用原输入长度规则。不能伪造大dummy input、全局放宽interval上界或以零observations漏计工作量。
4. cost与memory直接计literal count、word_budget上界、packed/投影结果、scratch、checkpoint和保留容量；预算不能只依据输入observations或逻辑values尺寸。selected roots与全部实际interval仍先做独立结构和容量检查。

当前int64输入/结果、rank 0–3 C/F roots及Program v7/v8已有承载字段，不需要新增Python数值层或通用异构wire。worker v22并不已包含RNG审计字段：若增加实际取词/position pass/生成与缓存峰值审计，需同步升级请求/响应版本、bindings、worker身份与传输回归。旧worker或未知opcode必须明确拒绝，不能静默降级。Program格式保持或升级须以实际字段变化决定，不能机械因为新增opcode改版。

### 10.6 历史候选物理计划与独占证明

此前考虑过以下三类物理计划，均依赖尚未实现的RNG数学路径和投影，不是当前交付顺序或后续任务。候选选择依据完整工作量和峰值，不把“全C状态”与“每组合重做producer”作为最终二选一；保留第9节ordered fold和可编辑源节点。独占证明、复制及存活内存的要求仍可用于审查已有算子的物理优化。

| 计划 | 主要存储与工作量 | 适用证明与剩余成本 |
| --- | --- | --- |
| 直接RNG流→共享rotation→asset expm1→全组合消费者 | 不物化完整normal带；每TP行一次rotation与资产投影，保留`3*C*P`组合状态及必要chi/scale、排序和结果 | 私有闭包且完整full-batch可承受；T5/P50k/C32的三组状态约38.4MB仍必须计入 |
| graph-owned带原地复用→资产缓存→小组合batch | 私有独占normal带可变换/重用，重型前缀仅一次，batch状态较小 | old values已死且owner唯一、无公开中间根/其他消费者；容量不足或输出别名风险必须拒绝复用 |
| checkpoint replay→共享资产缓存→小组合batch | 不保留完整normal带，保留`T*P*A`资产缓存及checkpoint/chi；重型rotation/资产投影仅一次 | 比较取词position pass与缓存成本；A8时缓存可为16MB，A32时64MB仍是实存，不能漏计 |

候选Student共享流须遵守全normal在先、全gamma在后：可先用位置扫描得到normal结束checkpoint，生成chi/scale，再从原checkpoint重放normal供rotation。位置扫描保留polar的接受和取词顺序；缓存需按源语义求出最后一对的spare，不能只保存pos。gamma顺序、最终next checkpoint与normal/gamma计数保持canonical语义；额外扫描取词是物理工作量，不计成公开流被重复推进。独立筛选的skip/checkpoint能力只是候选证据，未接入生产完整图。

双游标的**逻辑流计数与实际工作预算必须分开**：最终checkpoint只表示一次normal→gamma转换的word数，不能反映另一个游标重放normal所做的工作。设实际normal取词为Wn、gamma为Wg，skip→gamma→normal replay的总取词工作为`2*Wn+Wg`，成功公开流仅推进`Wn+Wg`；cache命中和空块按实际0词计。Planner需为全部position pass、replay、gamma分别计上界，再做全请求工作预算准入，执行审计另计总实际word，不能只读最终checkpoint关闭资源验收。若私有计划的额外工作上界超预算，在执行前选择已计预算的canonical计划；进入数值执行后不以资源失败切换路径。不得把物理重复推进写入公开流计数，也不得因公开计数较小而漏计双游标资源。

原地复用只针对**本次图内部独占、已证明dead的owner**，不修改外部借用的只读输入、ModelPayload、checkpoint或前一代结果。先完成读取当前factor行所需的独占scratch，再覆盖旧带；A≤F仅是必要容量条件之一，不是充分所有权证明。A<F前移压缩若vector仍保留原capacity，内存仍按原capacity计，不能将逻辑缩短直接称为已释放64MB。合法C/F/strided借用输入保留原布局；只有真正内部自有的buffer才按所选高性能连续布局生成/重用。

私有proof检查完整RNG state/spare依赖、normal/gamma顺序、values视图、rotation/scale/means及所有消费者。公开values/state中间根、额外消费者、未证明isolate/作用域、布局或故障语义，均使用已计预算的canonical路径或保守materialization；不得先写结果后临时换算法。proof不读数值决定数学分支，私有scratch与输出仍独立。多个interval复用只读producer/checkpoint，不能重新推进已发布状态。

精确fan需要各时点完整路径分布，小tile不会凭空消除这项存储。选择模型同时计RNG额外扫描、真正存活的chi/scale、asset cache、组合状态、排序、图、两个结果世代及库scratch。审计应能核对实际MT取词、位置扫描、normal/gamma产出、TP producer行、缓存/物化字节、复制和retained capacity；预期重型rotation为T×P一次而非C倍。细小buffer不能掩盖新增的大缓存。

### 10.7 历史方案的失败顺序与行为边界

RNG原语是通用数学能力，不能自行嵌入平台coverage、组合状态或业务报错。原平台顺序仍是root检查/构造→生成→全部transform→逐组合coverage→全部projection→全部NAV→fan→limits。新资源budget是明确的新失败边界；它不能被私有融合推迟到NAV后，也不能假称旧无界源已有同一budget错误。

融合检测到NAV错误时保留有界候选，完成其所属组合的前置projection合法性扫描，再按组合、阶段和该阶段坐标选首错；前组合NAV错误优先于后组合coverage/projection错误。minimum_return保留projection的time/path前缀，NAV诊断保留path/time前缀。mapped零权重仍计算已映射资产，unavailable资产真正不消费；不能为“优化校验”读取未消费资产。

内部gamma值合法不自动证明scale及所有后段合法。原Student transform在`sqrt(chi/df)`计算为±0时有默认python error-model除零，负非零sqrt产生NaN；CME canonical DOMAIN_ERROR等与源的差异仍需第9.2、9.5节的逐域证明。不得把生成seed样本全部成功视为故障协议全域等价，未证明的路径继续明确保留迁移缺口。

本轮不新增RNG，也不替换covariance root：FIRP的Numba linalg链使用LAPACK对称eigh、下三角，现有CME有界Jacobi不承诺相同符号、重复特征空间基或非对称读取规则。owned对照使用同一已冻结root时须明确不计其构造；不能静默换Cholesky。coverage、limits、金融状态及JSON保持上游职责；此处保留真实顺序及候选风险，用于防止已有算子优化改变已涉及的行为，不形成扩大开发范围的要求。

### 10.8 历史候选的库来源与潜在改动面

独立筛选的MT底层使用固定NumPy 1.26.4源的`numpy/random/src/mt19937/mt19937.c/.h`及其真实依赖，没有接入生产构建。候选不使用标准库实现的私有序列化格式作为协议。其public几何是uint32 key[624]+pos，与Numba `_random.c`的seed/twist式一致。NumPy MT目录的`LICENSE.md`保留Jean-Sebastien Roy、Matsumoto/Nishimura及Wada等原始声明；分布薄层来自Numba 0.60.0 `cpython/randomimpl.py`，相关Numba BSD、CPython/PSF、NumPy第三方声明及修改说明均属于潜在引入时的许可约束。

历史方案的潜在部署约束是按原始来源、版本、source SHA、实际复制文件和license清单固定依赖，CMake、sdist、wheel和worker携带相同源/许可与构建身份。所需MT C代码是AOT native依赖，不要求运行时导入NumPy/Numba，不允许请求期编译、下载、TLS全局随机流或Python回退。jump文件是否进入构建按真实依赖核对；不能仅因源归档存在就宣称已实现跳转或并行sample分块。

下列是当时评估的潜在工程影响，**不是本轮待办或后续开发计划**。所有名字是历史拟定模块，不是当前能力；保留它们是为了说明候选不能仅靠替换随机内核就接入现有图：

1. **producer数学链影响**：会涉及RNG原生/Typed IR模块，`operators.hpp`、registry/catalog、typed/compiler dispatch、literal配置、capacity和borrow liveness、direct operator int64边界，packed投影及独立结果owner，以及CMake/包/license清单。只有API登记不能证明该方案可执行或有优化收益。
2. **私有流式/缓存与生成轴影响**：会涉及Planner/request geometry、graph生成依赖和worker重建，compact proof及affine行消费者的liveness/cost，以及native bindings/IPC审计。实际64MB物化消除条件、重复前缀次数和全部owner均需要独立证据，不能从筛选内核推导。
3. **执行安全与Excel影响**：会涉及四种lane、预算、取消、快照和两代结果，以及bounded RNG Excel recipe与符号上界。公式、MS Excel复算及owned计算链是不同验收层，候选尚未完成这些接入。

该候选即使只接入API也仍可能物化大normal带，不能据此宣称节约内存；物理计划不能改变纯依赖、可编辑逻辑或checkpoint接口。历史方案仅考虑Python薄绑定，不将随机、图、Planner、内存调度或Excel数学移入Python。

### 10.9 历史候选的Excel有限展开约束

六项候选没有当前Excel recipe，也不增加现有190个算子的Excel覆盖承诺。历史方案要求Excel使用同一版本化逻辑DAG，仅对**显式bounded RNG契约**规划：count/dims/word_budget必须已知，条件拒绝采样展开到该word上界，成功分支继续原polar/Cheng算法，超预算分支显示相应失败。不能先看当前seed的实际接受次数再决定展开量，因为用户编辑seed/alpha会改变拒绝次数。

旧无界抽样算法没有有限符号最坏上界，导出应在写出前拒绝`UNKNOWN_BOUND`；不能为了Excel自动给旧算法加隐式budget或换成逆CDF等算法。全工作簿超过默认200万单元格仍明确拒绝，预算包含MT状态、辅助位运算、全部拒绝分支/状态、spare、values、校验、后续数学和结果，不通过多表绕开预算。

MT uint32和计数limb在Excel中精确；seed递推的`1812433253*word`可超过2⁵³，必须分16位limb等精确组合，不能直接用double乘法取MOD声称整数等价。uniform53保留两word拼接顺序。normal/gamma浮点尾差可按已批准容差检查，但接受条件、cache flag、word数和MT状态必须精确；log/exp尾差若改变拒绝分支就是序列差异，不能用最终allclose掩盖。

不可使用RAND/RANDARRAY、冻结输出值或C++本次结果替代公式。未有recipe、未证明分支兼容或超Excel数值域时须明确拒绝，coverage登记不能仅随新增operator_count自动声称已实现。bounded配方的原生检查、符号预算、LibreOffice与Microsoft Excel真实重算分别验收，尤其编辑seed/alpha、奇数cache、混流、chunk和超预算；未完成这些验证时标未验证。普通C++执行仍不运行Excel。

### 10.10 历史方案需要的证据（非本轮门禁）

下列是历史候选若获另行授权后需要的证据，保留用于解释未实现和未证明的边界，**不是本轮门禁、缺口清单或开发承诺**。源兼容、图执行、物理优化与Excel须分层，不把独立内核筛选计为生产通过：

- **数学与checkpoint**：MT seed/twist/temper、uniform53 words、polar/Cheng/GS接受轨迹、float bits及635控制字段/spare；seed上下界、count0/1/奇偶、alpha三分支、cache/no-cache、mixed chunk、计数carry、malformed state、budget边界、失败不发布next state。检查同一checkpoint fork、完整依赖CSE和Prepared重复输入重现。
- **真实公开链**：六原语AST/Typed IR/DAG及投影shape/dtype、生成time和多interval、public中间根canonical、四种执行lane、IPC/pickle/ModelPayload、C/F/strided输入、selected roots、独立结构错误传播、取消及两代只读独立结果；原生、sanitizer、禁库路径和完整Python回归另计。
- **涉及的阶段错误**：冻结真实生成签名、版本和抽样顺序；生成/budget/transform失败、early NAV加late projection、mapped零权重、unavailable极端值、chi下溢除零及minimum_return前缀。coverage/root/JSON未进入候选入口时明确排除；进入时按原调用顺序适配，不能以成功seed宣称已覆盖。
- **候选同核时间**：原平台owned RNG→原地transform→完整projection/NAV/fan与候选对应完整入口交替配对；保留同CPU、库线程、热NJIT固定签名、全部统计/样本/贡献及两代结果。构图/首次/warm分别记录；RNG位置扫描、root和上游包装纳入范围或明确排除，不能删对照成本。tiny只诊断，中大型须获得实质收益。它与本轮双方使用同一NJIT prep RNG的对照不同，不能混用证据。
- **完整owner内存**：fresh PID连续计量输入构造、图/plan、首次、至少5次warm及两代完整结果；owned normal/chi/cache/checkpoint、retained capacity、库scratch、RHS求值时仍存活的旧period/NAV变量、结果staging和worker均计入。requested heap、RSS/mmap/allocator覆盖分别报告，不相加、不以计时外reset或只读重放copy制造优势。
- **物理实际工作**：rotation和资产projection共享TP行一次，避免C倍重算；实际word、位置扫描及缓存/复制字节与Planner一致。公开所有权负控制、成本拒绝、budget/取消失败均走已声明路径，不允许数值失败后隐式切换未计预算的执行。
- **Excel**：有限上界、两后端数值/整数离散状态、编辑和chunk连续性及MS Excel真实复算；旧无界UNKNOWN_BOUND和超预算明确拒绝，不能把unsupported拒绝登记为成功复现。

本节六项API、635协议、生成轴、RNG私有计划和Excel配方均为**历史未实施候选，不排入当前或后续开发计划**。独立RNG筛选不关闭第9节历史证据边界。本轮只对已有算子和组合的实际优化报告同核时间、完整owner内存、复制及重复计算变化；完整owned对照双方保留同一NJIT平台prep RNG，筛选收益只作诊断。

## 11. 普通输入缓存与执行所有权

### 11.1 问题与优化边界

原普通 `AdaptiveScheduler.execute` 将带强输入 pin 的 BoundData 留在隐式缓存中。调用完成且调用方释放输入后，旧 raw/chi 仍存活；准备下一批输入时出现两批大数组重叠。这是完整 owned-input 负载中的真实常驻成本，不能从计时或内存 epoch 中移走缓存来制造优势。

本轮仅修改 PyBind 输入所有权管理，保持 190 个算子、数学实现、公开接口、Typed IR、DAG、Planner、worker 和 Excel 公式不变。只复用地址与布局描述及兼容执行计划；不缓存数值、参数结果或输出。普通请求仍读取当前内容，结果保持独立。

### 11.2 强执行引用与可失效元数据

- **首次绑定**：按原校验顺序建立 BoundData，从已校验 pin 复制本次执行的强引用；完成引用分配后才发布缓存。不能第二次读取 mapping 来取得 owner，因为通用 mapping 与 dict 子类可能返回不同数组。
- **缓存匹配**：一次取得当前值，先强持有，再验证 identity、data、dtype、rank、shape、strides、实际分配几何及 owner 边界。命中时将同一对象交给执行；匹配失败仍按原流程重新绑定，不能把匹配失败时取得的另一数组当作新绑定的 owner。
- **缓存生命周期**：普通 BoundData 放弃数组及其分配 owner 的强 pin，仅保留描述和不解引用的 identity。未注册 NumPy weakref，避免引入 resize 限制。分配器地址重用也必须通过当前对象的完整校验；地址相同不等于数值相同。
- **dtype 描述**：支持的 float64/int64/bool/uint8 dtype metadata 可以引用任意 Python 对象。普通绑定放弃 pin 时，从已校验 dtype 的数值字符串重建无 metadata 描述，避免间接保留大数组；命中缓存后直接复用该描述，不重复构造。调用方 dtype、字节序、类型、几何及 exact-dtype 检查不变；Prepared 不执行这项释放。
- **本次执行**：局部 owner 集合同时保活输入、starts/ends、产品索引和数组参数，跨两次 GIL release、计划复核、原生任务完成、结果封装及异常展开。引用只在持有 GIL 的调用线程释放，原生 worker 不持有 Python 对象。
- **缓存替换与回调**：匹配时先局部强持有被检查的 BoundData；动态 Python base 回调或其他调用替换缓存后，本次仍执行原绑定。控制数组 owner 分配失败时清理本次已发布的强缓存，不能留下意外常驻 pin。
- **显式生命周期**：Prepared 强持有原绑定，几何/底层分配变化仍报 `PREPARED_INPUT_CHANGED`；shared/model 路径保留其原协议。执行期间修改输入或强制 resize 仍违反既有只读契约，强引用不是内存写隔离。

数值内容不跨请求共享；工作区仍由原生 Scheduler 独占。省去旧 input owner 不表示省去当前输入、必要递推状态、排序 scratch、图/plan 或两代独立结果；这些成本都必须保留在验收中。

### 11.3 验证范围

[公开生命周期回归](../tests/test_binding_cache_lifetime.py)覆盖成功与数学失败后回收 view/base、Prepared 保活、当前值更新、同几何新输入、dtype/shape 与底层 resize、两代 typed 结果独立、动态 mapping/dict 子类，以及校验回调清空 mapping/嵌套替换缓存。已有线程、进程、共享内存、关闭/超时与模型回归另行复测；Python base 回调是确定性的边界测试，不冒充跨 GIL worker 屏障测试。

另有八项大 owner 负控制覆盖四种支持的数值输入及 starts/ends、产品索引、数组参数中的 dtype metadata。旧构建八项均因 metadata owner 未回收而失败；修改后检查当前值、命中缓存、调用方 metadata 不变及两代独立结果，并确认输入与 metadata owner 均可回收。这是移除可任意扩大的间接常驻内存，不是为几 KB 微调数学路径。

完整成功负载的双方使用同一冻结平台 fixed-signature NJIT 随机输入准备函数，均计入其分配与运行时间；没有额外 readonly replay copy，亦不调用实验 RNG。随后执行旋转/normal 或 Student 缩放、资产投影、NAV、fan、14 项统计、样本和全部贡献，并精确核对离散结果、逐项浮点容差、状态、只读及旧结果独立性。普通 C++ 绑定/计划/执行/结果封装在计时内；静态模型 fixture 与构图在计时外，但纳入连续内存 epoch。协方差根生成、coverage/limits、JSON 和平台阶段首错不在本范围。

验收计入图、全部当前输入、常驻状态及保留结果；requested heap 与 fresh-process RSS 分开，不把阶段峰值相加，也不将 malloc 盲区记成零。源码、wheel 和 worker 身份与原始测试/性能输出见[本轮记录](physical-execution-evidence/2026-10-03-binding-owner-cache/README.md)。单组合提速幅度和短宽 C32 内存差距分别披露，不能由 C8 的优势关闭整体目标。

## 12. 短周期大状态批次（2026-10-04）

### 12.1 依赖、取舍与实际物理计划

046 的 T5/P50k/F32/A32/C32 完整 owned-input 成功链同时驻留32组 NAV、peak、maximum，三组状态占38.4 MB。每个组合的递推和贡献独立；readonly raw、rotation、means、scale、资产系数共享。无需新增数学算子，可以按组合批次重算确定性的原 producer，降低同时驻留的状态；不重复 RNG，也不修改输入或先前输出。

保留第9.4节全部前缀及消费者准入证明。准入后仅在 `T<C` 且 `C*(3*P+A)*8>=16 MiB` 时取 `B=min(C,4)`，其他形状保留全部组合共享一次 producer。16 MiB 是避免为小状态增加重算的当前形状分派条件，不是新输入限制、资源预算或通用性能门槛。`3*B*P+B*A` 状态、排序line、rotation行、scale、pack和child Scratch继续按实际容量纳入同一Planner；raw与asset工作量均乘 `ceil(C/B)`。CPU、结果协议、IPC22及公开配置不变。

每组仍按 time→path→asset 更新贡献和 NAV/peak/maximum；factor fold、weight-first product、排序与分位数公式不变。scale先按原拓扑完整执行一次，保留定义域检查先于消费者写出；coefficients/rotation只打包一次。post按既有group及原逻辑节点映射重放错误，child Scratch独立，失败后下一次正常请求重新初始化本次状态。未证明闭包、public中间根、scope/isolate等路径保持原canonical行为。两个批次之间没有共享可变递推状态，没有隐藏线程池。

两个隔离候选与046只差 `compact_path.cpp`。每批4组（c0ba665e）normal/Student完整链为1.497/1.510秒，冻结热态NJIT为4.202/4.212秒；热requested heap为77.155/81.483 MB。每批2组（28de24c6）为2.102/2.118秒，heap为74.754/79.082 MB。2组只比4组再省2.401 MB，却多约0.6秒，因此选择4组；不是按微小正差强行判性能失败。

4组Student三次独立PID的完整setup/cold/warm/replacement RSS增量为112.067/108.298/108.184 MB，NJIT为133.972/133.677/140.050 MB；中位数108.298/133.972 MB。这些隔离数据计入双方同一冻结NJIT RNG分配、普通入口、全部输出和两代结果；排除构图计时、协方差根、coverage、limits、JSON及平台阶段首错。RSS采样与requested heap覆盖不同，不能用较低RSS关闭heap剩余差距，也不称完整分配审计通过。正式策略另增加大状态形状条件，必须用正式wheel复证，不能直接沿用隔离版本作为最终验收。

### 12.2 契约与验证范围

[大状态公共回归](../tests/test_raw_state_batching.py)对normal/Student、C/F、single/thread/shared-process逐根IEEE对照public canonical AST；验证八批实际旋转计数、scale只做一次、两interval、预算、readonly输入、两代独立结果。三类错误核对类型和消息，并验证正常请求恢复。[原生回归](../tests/raw_prefix_native_cases.hpp)用无payload几何验证16 MiB上下界、贡献状态及重复工作量入账。已有小状态全batch、负stride、nonfinite、prepared和成本拒绝回归保留。

仅改变有依赖证明的物理批次及其预算，逻辑DAG、算子数学和Excel展开不变；普通计算仍不执行Excel，也不保留导出中间值。公式覆盖由适用回归复核，本轮没有新Microsoft Excel重算。正式构建7abae5b2全量12,319、实际ASan扩展52及三类原生各9/9通过，31个串行性能任务正常结束。短宽normal/Student约1.496/1.511秒对NJIT4.195/4.211秒；requested heap较046减少33.607 MB，仍比NJIT多4.627/6.955 MB。Student三次fresh PID RSS中位数107.790/133.186 MB，少约19%。全部数据与范围见[正式记录](physical-execution-evidence/2026-10-04-mc-state-batching/README.md)；C1提速幅度、中大型整体目标及其他主负载欠项继续开放。未提交或推送。
