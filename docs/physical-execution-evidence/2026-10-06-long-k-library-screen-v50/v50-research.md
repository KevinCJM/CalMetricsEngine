# V50：公开库与方法筛选

2026-10-06。本轮实际引擎保持 V48，没有修改 C++、安装新生产依赖、提交或推送。这里的性能判断仅针对本机单核、高量级长 K 数学请求。

## 官方资料与采用边界

- [Apple DGEMM](https://developer.apple.com/documentation/accelerate/cblas_dgemm)支持 C/F 存储顺序与转置参数。结合当前源代码，测试平衡树中间块直接使用 F 布局；不能从接口支持推断性能。本机候选反而更慢，否决。
- [Apple DAXPY](https://developer.apple.com/documentation/accelerate/cblas_daxpy)定义原地更新 Y = alpha X + Y。现有树的两个相加位置使用独立、连续的 numeric scratch；alpha=1，长度 rows*cols 有界且可用原 LP64 接口表达。候选仍调用同一个 Accelerate DGEMM，仅以成熟库替换两个 Eigen 相加位置。
- [ATLAS 缓存分块说明](https://www.netlib.org/atlas/developer/atlas_contrib/node11.html)强调矩阵形状、拷贝成本与缓存复用需要实测。本机只读系统查询得到性能核 L2=12 MiB；原计划提供九个 2 MiB 树块，候选改为九个 1 MiB 块。仅缩块与缩块加 DAXPY 均更慢，不采用。没有硬件 miss 计数，不能把容量对比解释为已证实的缓存原因。
- [OzBLAS](https://github.com/mukunoki/ozblas)提供 Ozaki 分拆的准确、可复现 BLAS；项目 README 说明其实现用于研究展示，不推荐直接用于实际应用。本轮不将它引入生产，也未做本机性能资格。
- [ReproBLAS](https://github.com/willow-ahrens/ReproBLAS)提供分桶式可复现累加，[LIBXS Ozaki](https://libxs.readthedocs.io/en/stable/samples/libxs_ozaki/)提供分拆/CRT 与不同设备路线。它们仍为资料候选，没有本机 AOT、误差/错误、Scheduler、峰值或完整时间证据；公开速度不作为本机采用结论。

不重复已闭合的 BLIS firestorm、ArmPL serial DGEMM、减少归并深度或 DGEMM beta 分组实验。本轮新增的 DAXPY 保留 beta=0 的独立乘积和原显式平衡树；它不是此前分组累加方案。

## 数学、复用及所有权

D1/D5/D6：没有数值 CSE、跨请求缓存或共享可写结果。两个归并点的 X 与 Y 位于不同的当前 scratch 槽；仍在原位置将 current 加入 older。alpha=1 的乘法对有限 binary64 精确，保持原逐元素相加语义、归并深度和误差预算。原输入域、分块、向上界、近零 canonical 回放、取消检查和整根晚拒绝重算均未改。树片段保留 C 布局，最后按原规则复制到独立 C/F 结果。

D8：私有控件确认两代结果不共享、输入只读且未变、保护区未被覆盖；DAXPY 与原版控件全数组相同。108 次计时均使用单核预算、预热后计算、独立只读输入准备与两代结果；最后一次计时之后全部 worker 先停在 CHECK 屏障，再做重核验。math.fsum 抽查使用已舍入的 binary64 乘积，不是 MPFR 精确乘积证明。

计时 ABI 链接合格 V48 的 input-bound 实现，保留计划及工作区限制；它没有实际 Graph/Planner/worker 的候选集成，也不具备全部公开故障策略。显式 scratch 容量相同只能说明该项没增，不能证明隐藏库分配或新峰值相同。

X1/X9/X10：仅优化物理数学执行，公共算子、AST/DAG/IPC 协议及 Excel 公式不变。没有新 Excel 函数或重算验收，本轮不能据此宣称新的 Excel 资格。

## 下一项资格

仅 DAXPY 候选保留待验收：先审查 actual helper 调用关系，再完成完整 AOT 引擎、typed Graph、Planner、process transport、错误回放/取消/所有权、正常及 SAN/OFF 回归和四布局 fresh malloc/RSS。若实际入口的收益或内存条件不成立，继续否决。当前一般大矩阵仍慢于 NI，其他重要完整链和整体目标开放。
