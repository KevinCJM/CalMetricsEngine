# 原生库调研与本机调用事实：尚未采用新库

2026-10-06。遇到多轮没有大额时间或内存收益，先调查成熟库与公开方法，再做有限、同条件的完整请求对照。公开性能图、架构支持、小输出结果都不能替代引擎实际链路及峰值验收。

本机是M1 Max（8性能核、2能效核），没有SME或SME双精度。Numba实际通过SciPy cython_blas；本地该二进制otool依赖Apple Accelerate。此前threadpoolctl仅列出NumPy的OpenBLAS池，不能据此宣称NJIT的矩阵后端就是OpenBLAS；原报告不重写，附加调用链与源/二进制hash。已封存4096单核样本的进程CPU时间约等于wall，VECLIB上限仍为1。库枚举本身不是全部工作线程的运行证明。

优先实测[BLIS 2.1](https://github.com/flame/blis/releases)的Apple firestorm与small/skinny控制，采用单线程静态AOT，避免错误落入generic参考内核。[BLIS构建说明](https://github.com/flame/blis/blob/2.1/docs/BuildSystem.md)明确区分配置、CBLAS与线程；同一预算下是否有益仍未知。[ArmPL 26.01官方发布](https://developer.arm.com/community/arm-community-blogs/b/servers-and-cloud-computing-blog/posts/what-s-new-in-arm-performance-libraries-26-01)支持macOS AArch64标准数值接口，列为第二候选，本机尚未安装或验收。[OpenBLAS](https://github.com/OpenMathLib/OpenBLAS)作为另一个原生FP64对照，不能把已经在NumPy加载当成引擎已完成接入。

[LIBXSMM](https://github.com/libxsmm/libxsmm/blob/main/README.md)针对小块并主要依赖JIT；只列为固定prepared形状候选，不能引入请求期编译或把小块成绩外推为4096完整矩阵。[OzBLAS](https://github.com/mukunoki/ozblas)用于参考准确/可重复方法，作者说明其为实验实现，不直接接入。[M4 SME贡献](https://github.com/flame/blis/pull/921)仍是开放PR且本机无对应指令，排除作为本机加速依据。

当前仍使用已验V33。大矩阵过慢还包括输入扫描、精度界强制分块、重复packing与Eigen归并；换库未必消除这些费用。先固定来源和版本，在原五组高量级/非整块条件下计入完整准备、两代独立结果、空间/复制/线程，并保留负结果。通过后才贯通AST/DAG/Planner/worker与真实源代码。没有新canonical算子、系统包安装、发布或整体达标声明。


官方固定提交已在隔离目录静态构建，按有效宏及运行入口核验单线程/无池；[152项原控制](blis-qualification/README.md)已通过，后续完整25/150与100fresh已经闭合，五组均未改善Accelerate同计划，未采用该替换。初版构建验证器失败保留在blis-build-status.json，供应商构建实际均退出0；没有系统安装或引擎接入。


[完整库筛选与下一方法边界](blis-qualification/closed-prescreen/README.md)保留全部负结果。进一步参考[GEMMFIP原论文](https://arxiv.org/abs/2302.08417)：packing融合首次计算是公开的减少数据移动方法，尚未验证本机或当前精度分块适用性，不直接复制实验原型成为生产能力。
