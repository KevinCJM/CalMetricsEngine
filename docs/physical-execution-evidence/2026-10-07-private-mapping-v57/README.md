# V57：私有算术与列重排融合的实际验收

2026-10-07，最终G版已接入当前CalMetricsEngine工作树，未提交、推送或部署。完整13输出压力链比实际V56少约61MiB申请峰值，耗时基本保持；申请峰值仍比NJIT高约16MiB，**整体优化目标未完成**。

## 改动与证明

- 两段私有算术复用现有Eigen/标量实现，最多保留一个已证明存活、独立拥有的同形C数组和一个短广播快照。检查原slot写入和另一操作数的视图别名，省去第一段完整矩阵。
- 私有二维列`take_axis`可直接供独立`scatter_rows`根或既有scatter/argmax消费。证明数值源与int64列索引底层存活，复用canonical填充、索引检查和离散决策，不保存重排矩阵。
- Planner与执行使用同一几何条件；最终F根目的地不可用时重新采用原路径。原位置结构/dtype错误、readonly输入、独立C/F根、Prepared重读及两代结果保持。未知形状、其他轴/秩、隔离、观察器及子作用域保留原职责。
- HMM训练图暂不启用这两项新私有消除：少一点arena可能丢失免费第三块history、增加逐轮beta复制。保留V56已验收路线，后续联合规划arena/history成本。
- 修改4个引擎C++文件及4个测试文件；无新算子、依赖、数学公式、线程池或IPC协议。Excel仍按相同逻辑图导出，不因C++计算触发Excel。本轮不作新的Microsoft Excel实机验收声明。

[算术设计](raw/design.md.txt)、[列重排设计](raw/scatter-companion-design.md.txt)、[最终F根证明](raw/v57f-final-root-proof.md.txt)、[HMM成本修复](raw/v57g-history-decision.md.txt)、[自审核](raw/self-review-v57g.json)均保留。

## 资格、身份与负结果

| 验证范围 | 结果 |
| --- | --- |
| 最终G版正常Python全量 | 16,100通过，0失败/错误/跳过 |
| ASan/UBSan相关Python | 1,362通过，0失败/错误/跳过 |
| 关闭Eigen/Accelerate/Highway相关Python | 1,362通过，0失败/错误/跳过 |
| 三种构建原生 | 各9/9通过 |
| 当前目录重建后正常相关Python/原生 | 1,362通过、9/9；不是第二次全量或SAN |

新增Python202项（算术95、列take/scatter107），并新增原生float64/int64/bool首错及完整输出前缀检查。覆盖操作数顺序、C/F/反向/零stride、短样本、别名/中途写入拒绝、公开F根、坏索引/重复行、非有限值、Prepared变化、未消费坏数据、single/thread/inline/shared、旧结果所有权及恢复。

[正常资格](raw/v57g-quality-normal.json)、[SAN资格](raw/v57g-quality-asan.json)、[库关闭资格](raw/v57g-quality-off.json)、[正常原生](raw/v57g-normal-native.json)、[实际目录复核](raw/v57g-actual-quality.json)。93份源码/构建配置及编译特征一致，最终资格与实际目录构建ID均为`db14f515428e1f822a87b304a65b9c2c0ad6da134c54c71787f7edc6c590e1e1`；[候选身份](builds/candidate-v57g/identity.json)、[实际wheel/worker身份](builds/actual-v57/identity.json)、[精确落入范围](raw/v57g-adoption.json)。接入前322份既存源码/测试/规范文件全数保持；最终仅改变5个既存代码/测试文件、新增3个测试文件，另同步3份主题文档。

早期D版全量有2个明确失败：F回填根清零私有源，以及HMM复制超出原history成本上界；SAN/OFF各有同一个F失败。原生探针还发现错误dtype首错变化。E版“只检查typed程序”的解释和修复未解决F问题，属于失败假设。F版检查最终根目的地解决F；G版保留HMM历史路线解决复制回退。缺少PYTHONPATH、错误的测试夹具/arena对比、启动器路径及实际构建准备器替换失败一并保留；未放宽旧断言。[D版全量失败](raw/companion-quality-normal-full.xml)、[首错探针](raw/native-take-storage-diagnosis.log)、[准备器修复](raw/v57g-runner-prepare-failure.json)。B/D早期性能只能说明各自实验，不能替代最终G版资格。

## 完整请求性能

同一seed、1,398,104行×8维、K16、训练1,048,578行、4迭代，普通Scheduler与强预热NJIT。计入独立只读输入准备、标准化/初始化/训练/推断/排名/回填/状态映射、13输出和两代结果保留；CPU1/库线程预算1，定义及编译不计入。构建、回归与CodeGraph同步结束后测量，完整数组在计时之后逐元素对照。

| 六轮交替fresh完整请求 | 实际V56 | 最终G版 | 预热NJIT |
| --- | ---: | ---: | ---: |
| 时间中位（秒） | 2.118238 | 2.081958 | 7.322157 |
| malloc暖峰值（bytes） | 932,617,522 | 868,304,738 | 851,456,928 |
| RSS暖绝对峰值（bytes） | 1,126,596,608 | 1,062,666,240 | 1,421,246,464 |
| RSS相对初始基线增量（bytes） | 834,027,520 | 769,884,160 | 1,090,830,336 |

六轮为每backend各6个fresh时间进程；malloc/RSS则每backend各一个独立fresh进程，内含三代请求，不能写成每backend三个fresh峰值进程。约36ms时间差仅作诊断，采用依据为64,312,784bytes（61.33MiB）申请峰值改善。RSS包含不同基线驻留，不能代替申请峰值比较；仍有16,847,810bytes（16.07MiB）NJIT申请峰值缺口。[全部配对与峰值](raw/v57g-paired-state.json)。

当前目录重建再作各3个fresh时间进程交替配对，中位C++/NI为2.084427/7.254333秒，CPU约100%；另各fresh malloc/RSS进程、三代请求，malloc重现868,304,738/851,456,928bytes。C++输出逐字节匹配旧V56，NI沿用4.9e-5浮点预算、离散结果精确，13项完整输出全部通过。[实际目录确认](raw/v57g-actual-confirmation.json)。

numeric arena204,126,000→139,813,216bytes；mask arena22,369,664bytes未缩减；根复制2192bytes，当前`algorithm_copy_bytes`报告584,422,064bytes，不宣称全算法零复制或全面复制审计闭环。

## 调研与剩余工作

参考[Eigen索引](https://libeigen.gitlab.io/eigen/docs-nightly/group__TutorialSlicingIndexing.html)、[别名规则](https://libeigen.gitlab.io/eigen/docs-nightly/group__TopicAliasing.html)、[BLAS/LAPACK后端](https://libeigen.gitlab.io/eigen/docs-nightly/TopicUsingBlasLapack.html)及[ExecuTorch生命周期规划](https://docs.pytorch.org/executorch/stable/compiler-memory-planning.html)、[OpenXLA缓冲分配](https://openxla.org/xla/hlo_to_thunks)。新版文档是方法依据，不能当作本机Eigen3.4性能证据，也不据此引入第二套图运行时。

本机Numba0.60的`ensure_blas`导入SciPy的cython_blas；SciPy1.17.1配置为Accelerate，NumPy1.26.4配置为OpenBLAS，二者不能混同。[本机依赖事实](raw/installed-blas-research.json)、[内存规划调研](raw/lifetime-memory-research.json)。CodeGraph现有索引已同步并读到`execute_impl_body → scatter_take_pair → validate_scatter_indices/scatter_values`；索引不是运行时资格证据。

下一步：处理状态比较掩码，联合评估原slot容量、真实生命周期、别名和HMM历史借用成本，再验证重要大矩阵和其他平台完整链。大矩阵NJIT退化、整链内存优势与完整分配/复制审计均保持开放，不把本次GMM压力链当作全平台验收。

[SHA清单](manifest.json)覆盖约10MiB原始证据，[收尾检查](raw/v57g-closeout-validation.json)验证源码身份、未受影响文件、证据哈希与文档链接；代码与历史设计以文本快照归档，不加入当前调用链或制造新CodeGraph源码副本。
