# 矩阵与数组基础：设计与验收

2026-09-25。状态：14 个算子及跨层实现已完成，本机正确性回归通过；时间和内存严格双优验收未通过。范围为 M2 的矩阵与数组基础；不包括随机源、QP/模型训练或平台服务切换，也不代表 M2 整体完成。保留进入任务前的执行优化和提交规范修改，本轮未提交或推送。

## 数学契约

在原 146 个 opcode 后追加能力，不改变旧 `solve`、归约、广播与错误契约。新增算法全部在 C++ 执行；Python 只绑定输入/结果。遵循 computation-design-rules 的 D1–D8 及 excel-generation-rules 的 X1–X10。

| 算子 | 职责与输入输出 |
| --- | --- |
| `cholesky(A)` | 严格对称有限 float64 方阵 → 下三角 L，A=LLᵀ；不加扰动，不截断负特征值 |
| `solve_multi(A,B)` | 方阵 A 与二维多 RHS B → X；一次部分主元消元，求 AX=B；沿用绝对主元 <=1e-14 判奇异 |
| `triangular_solve(A,B,lower=1,unit_diagonal=0)` | 显式上下三角求解，B 为向量或矩阵；仅消费指定三角和非单位对角，忽略另一三角 |
| `symmetric_eigh(A,tolerance=1e-12,max_sweeps=32)` | 对称有限矩阵的有界循环 Jacobi 分解；最大绝对元素缩放；每轮最大归一化非对角值 <= tolerance 停止；上限仍未收敛报错 |
| `eigenvalues(decomposition)` / `eigenvectors(decomposition)` | 同一 nominal 分解的值/向量投影；值升序稳定排列，向量为列，每列最大绝对分量的首个位置取正 |
| `take_axis(x,indices,axis)` | rank 1–3、float64/int64/bool；沿指定轴按非负 int64 下标选择，保留次序与重复 |
| `slice_axis(x,axis,start,stop,step)` | 严格 0<=start<=stop<=轴长，step 正整数；半开区间；不自动夹紧越界，不接受 Python 负下标 |
| `concatenate(a,b,axis)` | 同 dtype/rank，其他轴匹配；沿已有轴连接两个数组 |
| `stack(a,b,axis)` | 同 dtype/shape 的 rank 0–2 值，在新 component 轴堆叠两个值，结果最高 rank 3 |
| `broadcast_like(x,template)` | 同 dtype，右对齐广播到模板形状；只能从长度 1 或相同长度扩展；模板确定形状，模板的计算失败仍传播 |
| `sum_axis(x,axis)` / `mean_axis(x,axis)` | float64 rank 1–3 沿一个轴归约并移除该轴；sum 空轴为 0，mean 空轴在有输出时为短样本错误；NaN/Inf 按算术传播 |
| `swap_axes(x,axis1,axis2)` | rank 1–3，交换轴及 strides；支持非连续、负/零 strides 输入 |

分解的原生载荷为 n×(n+1)：每行第一项为一个特征值，其余为该特征向量分量。Typed IR 标记 `symmetric_eigh`，仅专用投影接受该标记；这不是可混用的业务矩阵。直接调用的两投影借用共同载荷，图根结果继续独立拥有；不要求调用方解析载荷。空方阵可分解为空结果。tolerance 限定 [1e-15,1e-3]，max_sweeps 整数 [1,128]；预算耗尽使用 NON_CONVERGENCE，不伪造成功。重特征值比较特征子空间与重建残差，不以任意特征向量的逐项相同作验收。

轴参数在 AST 中必须是字面整数，使 dtype/rank/具名轴在编译期确定；slice 的边界和步长也要求字面整数。直接算子接口仍每次校验实际参数。Typed IR 保留金融维度/价格基准，矩阵求解和分解显式推导输出维度，禁止把有名轴默默当作位置相同。

结构错误始终抛出，包括上游失败时可独立检查的 dtype/rank/轴/几何/下标约束。矩阵非有限、非对称/非正定、奇异、未收敛分别保留数值失败类别；isolate 仅污染依赖根。没有把 0 当缺失。数组重排只处理实际选中元素，但沿用通用算子对上游执行异常的保守依赖传播。

## 执行与内存

- 复用 canonical registry、AST/Typed IR、DAG/CSE、Planner、worker 和类型化结果；追加 opcode，已有 opcode 不改号。
- 多 RHS 共用一次消元；同一 `symmetric_eigh` 的投影通过 CSE 共用分解，输入/参数/版本不同不共享。prepared 每次重算变化数值，不按地址缓存分解。
- 类型化图中的相邻 `symmetric_eigh→eigenvalues`，若分解只有该消费者且不是公开根，健康、无观察器的执行可以省去死去的特征向量和完整载荷。缩放、矩阵旋转、停止及升序值逐位保持；先复制全部输入再写 N 元素目的地，支持死 arena 复用。共享向量、错误隔离或观察器执行完整分解；Excel 仍观察原逻辑节点，不受物理裁剪影响。
- 完整分解 n≥128 时用标准库稳定总序索引及循环行置换整理 eigenpair；相同值按原行号排列，不改变向量位模式。N 个 `size_t` 索引计入 Workspace/Planner；行缓冲借用已经死去的缩放矩阵。独立值路径不申请该索引；Planner 仍保留未融合/隔离的保守容量。
- slice、swap、broadcast 和分解投影采用受检借用视图；编译器延长底层存活期。普通图根和快照物化为独立结果，直接算子借用结果保活原输入。take/concat/stack 分配必要输出；求解/分解使用独占 Workspace，全部 scratch 计入 Planner 和审计。
- take_axis 与轴归约按逻辑行一次计算非归约轴地址，内循环沿原有 stride 读取；减少坐标除法，不改变归约顺序或先行结构校验。选择结果仍物化，尚未实现选择与归约的物理融合。
- 本轮原生内核不新增外部 BLAS 或私有线程池；独立批次仍由现有 Scheduler 按成本与内存预算调度。分解内部保持单线程，不能以超过 NJIT 核数获取优势。不宣称所有新内核使用显式 SIMD。
- 数据和时间轴只由声明决定；不替调用方构造日历、定义金融指标或认证因果性。输出最高 rank 3，正步长 slice 的限制显式保留。

## Excel

所有新算子同时实现可见公式展开；分解/消元按真实步骤展开，特征分解按最大 sweeps 预算并在收敛后冻结。数组重排使用可追溯引用，动态下标以公式选择，不冻结为当前值。预算先于生成，按带溢出检查的多项式上界覆盖迭代、输入、中间与输出，不读取 C++ 参考值计算结果。普通运行不执行任何 Excel 工作。

## 验收计划

1. NumPy/SciPy 独立参考及矩阵重建、正交性、方程残差；覆盖空/单元素、奇异/不正定/非对称、病态/非有限、参数边界和未收敛。
2. 数组所有 dtype、rank、各轴、重复/越界下标、空轴、负/零 strides、只读、别名、跨次结果与 owner 寿命。
3. AST/IR/序列化、共享投影、多个根、失败后独立结构校验、single/thread/process/shared，以及预算低于必要容量的拒绝。
4. Excel 覆盖清单、符号预算 >= 实际占用、公式依赖、真实重算与预先冻结容差。Microsoft Excel、LibreOffice 与公式解释器证据分别报告。
5. 安装最终 wheel 后完整 Python/native/sanitizer 回归，核对源码/构建身份；保留旧四组与 M1 七组性能门禁。
6. 新负载在计时前固定小/中/大矩阵、多个 RHS、分解共享以及连续/反向三维数组。同 CPU=1、热态 NJIT、独立输出，普通/prepared 分开交替配对；记录中位数及 95% 上界。独立子进程 RSS 和 malloc 追踪分开测量，包含工作区/保留结果，测不到的内存优势记为未证实。未达时间和内存双优不能宣称迁移验收完成。

## 验收结果

### 构建与正确性

本机 macOS arm64、CPython 3.12、AppleClang 17，Release AOT wheel。最终源码与安装包的 `engine_build_id` 均为 `e14910c73188f8a841f8a36431dd1cd576382d4b4874e459f117d4f8543805d2`；wheel SHA256 为 `928537c3504d5d616b93ce1a1e34e83ac916f9d7b85e89e0838604a22a474430`。构建包含进入任务前已存在的未提交运行时优化；不能将整棵脏工作树的差异都归为本次实现。

| 验证 | 最终构建结果 |
| --- | --- |
| 完整 Python 回归 `pytest tests tools -q` | 6,181 项通过 |
| Release 原生测试 | 9/9 通过 |
| ASan/UBSan 原生测试 | 9/9 通过；macOS 使用 `detect_leaks=0`，不表示做过泄漏检查 |
| 新算子差分与不变量 | NumPy/SciPy 参考、重建/正交性/方程残差、重根子空间及尺度覆盖通过 |
| 跨层契约 | AST/Typed IR、CSE 投影、序列化、单线程/线程/进程/共享内存、独立根与快照所有权通过 |
| 输入边界 | float64/int64/bool、正/负/零 strides、只读、空数组、结构错误、失败隔离与预算拒绝通过 |
| Ruff 与补丁格式 | 新 Python 文件检查、`git diff --check` 通过 |
| 使用文档示例 | 最终安装包执行 26 个 Python 代码块通过 |
| 文档一致性 | 12 份文档、154 处本地链接/锚点及 160 个算子名称、opcode、签名入口核对通过 |

自审核已修正并回归：bool 借用视图的字节 stride、借用掩码输入的非法字节校验、矩阵非有限值与非对称的错误优先级、大外层尺寸下空输出的提前结束，以及形状加法溢出的边界。未发现其他已确认的正确性问题；本机结果不替代 Linux/Windows 或平台业务迁移验收。

### Excel 独立核验

- 最终构建在 LibreOffice 25.8.4.2 实际重算 **431 组、2,714 个结果**，全部通过；另有 2 组输入编辑检查通过。公式预算、最大迭代展开及不引用 C++ 结果单元格的检查也通过。
- 本轮早期构建的新增夹具在 Microsoft Excel 16.89.1 实际重算 **28 组、159 个结果**通过。最终工作簿与该次源工作簿 **16,802 个单元格**逐项一致，比较包含公式、输入、冻结的 C++ 参考和输出映射，仅排除导出 ID 与构建 ID 两个说明字段。
- 最终构建重开 Microsoft Excel 时 macOS 已锁定，未进行最终构建的桌面重算；上述是本轮真实重算加等价核对证据。最终独立生成的 Microsoft Excel 文件仍保留 `NOT_RECALCULATED`，没有改成 PASS。

### 性能与内存

采用固定的 16 个负载：3/16/64 阶矩阵与 8 个 RHS，以及小/大、连续/反向三维数组。矩阵对比普通图、prepared 独立快照、直接算子三个入口；数组比较普通图与 prepared，共 44 组。CPU=1，OpenBLAS/Numba/OMP 线程均为 1；NJIT 提前编译，31 轮交替配对。速度通过条件为 C++/NJIT 耗时比的 bootstrap 单侧 95% 上界 <1。直接算子的局部胜出不能代替图入口通过。

内存用独立后端子进程，分别测准备期保留量、首次执行与热态分配峰值、100 次调用的 RSS。计时不注入 malloc 追踪器。两侧先加载运行库，避免只把首次 import 计入 C++；图/工作区的准备成本仍单独记录并计入峰值门槛。malloc 不能覆盖 CPython arena 内部子分配及所有映射，RSS 零增量或持平记为未证实，不算优势。这里的 RSS 是单轮采样，不能据一个小的页级差值宣称稳定内存优势。

| 门禁 | 结果与边界 |
| --- | --- |
| 新负载时间 | 8/44；普通图 2/16，prepared 2/16，直接算子 4/12 |
| 新负载 malloc 峰值 | 15/44，含准备期保留量与首次/热态峰值 |
| 新负载 RSS 增量 | 本轮观测 4/44 达到脚本条件；其余未通过或无法证实 |
| 新负载联合条件 | 本轮 1/44，仅 `triangular_solve_64` 直接入口；整体未通过 |
| 原四组 Phase-2 门禁 | 4/4，阈值未改；63 点普通 Scheduler 比值为 1.000，仅满足原 <=1.00 门槛，未证明严格更快 |
| M1 七组时间/内存联合门禁 | 1/7，仅 `tensor_large`；其他缺口保留 |

代表性普通图耗时比（越小越好）：

| 负载 | 中位数 | 95% 上界 |
| --- | ---: | ---: |
| Cholesky 3 阶 | 1.530 | 1.552 |
| Cholesky 16 阶 | 1.384 | 1.401 |
| Cholesky 64 阶 | 1.878 | 1.892 |
| 对称特征分解 3 阶 | 0.923 | 0.932 |
| 对称特征分解 16 阶 | 2.069 | 2.075 |
| 对称特征分解 64 阶 | 5.574 | 5.599 |
| 多 RHS 求解 64 阶 | 1.657 | 1.712 |
| 三角求解 64 阶 | 0.873 | 0.874 |
| 小数组选择与轴归约 | 3.474 | 3.488 |
| 大数组选择与轴归约 | 4.751 | 4.768 |
| 大数组反向 strides | 4.984 | 4.991 |

普通图的热态 malloc 占用口径为“准备后保留量 + 首次执行后保留量 + 热态峰值”，单位字节：

| 负载 | C++ | NJIT |
| --- | ---: | ---: |
| Cholesky 64 阶 | 73,018 | 107,072 |
| 对称特征分解 64 阶 | 142,386 | 142,352 |
| 大数组选择与轴归约 | 1,070,754 | 8,512 |
| 小数组选择与轴归约 | 22,434 | 832 |

本轮已把 take/归约的坐标解码移至每条逻辑行，连续大数组耗时比由初测 8.133 降至 4.751，但仍未胜过已融合的 NJIT 参考。矩阵较大规模仍落后于 NJIT 调用的 LAPACK；数组链中选取结果、图缓冲容量及重复访问仍是明确欠项。不能靠增加核数、删输入检查或将根输出改成借用快照消除这些失败。

后续优先项：① 为 `swap_axes → take_axis → sum_axis/mean_axis` 增加受检的物理融合并缩减实际所需 arena；② 对矩阵求解/分解做缓存布局与分块优化，独立评估原生库方案的契约、发布及统一线程预算；③ 继续降低小图绑定/分配成本，并用相同负载复测。它们是后续工作，本轮未宣称已实现。

### 原始证据与复现

原始结果位于 [matrix-array-evidence/2026-09-25](matrix-array-evidence/2026-09-25/README.md)，包括源码/工具链/wheel 身份、全量测试日志、逐轮时间样本、分配与 RSS、原四组和 M1 门禁、Excel 报告。初测与修正 import 口径前的诊断结果单独保留，不混入最终通过数。CodeGraph 已同步到当前新增内核，用于导航，不替代运行证据。

先按项目构建说明安装当前 AOT wheel，确认源码构建身份；在仓库根目录执行：

```bash
python -m pytest tests tools -q
ctest --test-dir /private/tmp/calmetrics-matrix-native --output-on-failure
ASAN_OPTIONS=detect_leaks=0 ctest --test-dir /private/tmp/calmetrics-matrix-asan --output-on-failure
python tools/check_excel_parity.py --all-cases --check-edits --engine libreoffice --output /private/tmp/matrix-excel-check
# 以下性能命令顺序运行，避免并发构建、测试或桌面重算干扰。
c++ -std=c++17 -O2 -dynamiclib tools/heap_probe.cpp -o /private/tmp/matrix-heap.dylib
python tools/check_matrix_array_performance.py --heap-probe /private/tmp/matrix-heap.dylib --output /private/tmp/matrix-performance.json
python tools/check_phase2_performance.py --cpu-budget 4 --repeats 41 --output-dir /private/tmp/matrix-phase2
python tools/check_foundation_performance.py --output /private/tmp/matrix-m1.json
```

原生测试目录须事先按 CMake 的 `CALMETRICS_ENGINE_BUILD_PYTHON=OFF`、`CALMETRICS_ENGINE_BUILD_TESTS=ON` 配置并构建；ASan/UBSan 另加 `CALMETRICS_ENGINE_SANITIZE=ON`。malloc 适配器目前仅支持 macOS；其他平台不能沿用其内存通过数。矩阵测量脚本输出 JSON 不等于通过门禁，须检查每组 `time_pass/heap_pass/rss_pass/joint_pass`；本轮结论为功能正确性通过、性能验收未闭环。
