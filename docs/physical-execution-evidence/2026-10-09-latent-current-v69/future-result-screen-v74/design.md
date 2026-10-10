# V74：本次未来独立结果区借给 GMM 后验

2026-10-09，实际a87f2c25不先改变。只改graph.cpp的物理工作区选择，数学内核latent.cpp保留实际源码，既有成熟Eigen计算及单次E步不重写。V72/V73有界重算各30个完整fresh进程通过数学/所有权，但约9MB节约伴随GMM/HMM实质耗时增长，不采用。

## 依据与改动

当前loan_latent_workspace已会借本次未来独立scatter矩阵存放HMM历史，但先要求NK>=2Mi和至少一个arena span，并保存所有仍活跃arena输入。GMM3的NK工作区约9MB，而活跃ND可超过9MB，arena save/restore无法省空间。

GMM full posterior只有一个NK矩阵；当前13根链的未来probabilities scatter矩阵已经按结果协议独立分配，且capacity>=NK。先尝试只借这个未来区，不触碰arena/输入，仍调用同一canonical fit body、相同更新顺序/停止/参数。只保留原3K短scratch，没有重复E，没有ND save/restore。数组准备、所有根、两代结果与最终完整shape仍保持。

未来span选择复用旧helper规则：typed严格入口、无isolate/scopes、fit destination为C；未来root id>fit、operation/numeric/scatter_rows、float64/rank2、直接root destination匹配；本row/root capacity与byte bounds/alignment、fit output非重叠。新增当前metadata中没有任何<=fit物理writer指向这个root的检查，排除提前写入的区域；root region仅本次所有者，不能借输入、前次结果或已写入根。

只对full posterior GMM、NK>=1Mi且prepared scratch>=NK准入；已经用bounded replay的GMM不强制恢复full posterior。原arena/HMM loan仍保留原门槛和save/restore，复用同一个未来span选择lambda，避免维护两套规则。prepared、observer、输入故障及deferred error的现有caller禁用条件不改变。未来根尚无可见内容，strict失败不返回本次结果，故不需恢复其旧值；隔离失败仍走原安全路径。

固定实际shape是scratch准入，不是数值缓存身份；每次fit仍重算，model输出仍原owned slot。必要分配是当前结果区和3K小工作区，没有新大数组/输入打包；scratch过大旧capacity释放，不能转移到常驻缓存声称消失。原public order/strides与以后scatter写入不改变。

## 资格与边界

先核对93源/配置、191测试、真实编译、wheel/native/worker身份；原生9/相关回归701及普通/不同shape/boundary/F/strided/错误/多fit/独立根/prepared/线程/进程专项。再五组原完整链current/candidate/warm NJIT、CPU1、六owned readonly准备、13根/两代保留、独立fresh time/heap/RSS；全部输出/NaN、离散/实际迭代、数学不变量、input/old hash及非别名。不能只按prepared scratch推断完整peak。

候选若有实质收益，补重复组、全量/SAN/OFF，再重建实际包复验，才采用。D1/D2/D5/D6/D8，X1/X6/X9/X10适用：物理存储复用，同逻辑/数学/错误/所有权，不改AST/算子、Excel recipe/预算；普通请求不执行Excel，无本轮Microsoft Excel重算。实际仍未采用，剩余arena缺口、完整QP/前沿/探索与其他主要负载继续开放，目标未完成。
