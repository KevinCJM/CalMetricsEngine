# V59 大矩阵诊断与未采用的 Highway 候选

日期：2026-10-07。本轮完成研究、隔离实验和验收记录；**没有接入生产 C++ 优化，也没有性能闭环**。当前正式源码仍是 [V58](../2026-10-07-state-decision-v58/README.md) 的 `6a4114d3736f7967e011c5e7619172eb093685461cbacd3cbda74191de02aadb`，382个既有源码/配置/测试/规范文件在记录文档前逐SHA不变，平台项目未修改。候选为 `961a5036ceaceb72cfc7727708595aeb0ddd58f358807d2304092ca4b651387e`；它通过局部回归，不代表发布资格。

## 研究与设计

- [Highway 1.3.0 官方接口说明](https://github.com/google/highway/blob/1.3.0/g3doc/quick_reference.md)建议把横向归约移出关键循环。隔离候选复用项目已打包的1.3.0：流式读取连续C/F物理行，累积有限性、绝对最大/最小与符号，再在行外归约；第二版采用四组独立寄存器，减少同一累积器的依赖。没有手写新的数学算子或线程池，不增加库依赖/heap数组。任何拒绝都重新进入既有Eigen分类器，因此零、负零、NaN、越界、padding及旧fallback路线保留。两版分别留档，实际目录中没有第二套实现。
- [DGEMM 官方说明](https://www.netlib.org/lapack/explore-html/d7/d2b/dgemm_8f_source.html)允许beta=0时输出未初始化。当前Graph普通typed入口仍先整块清零；原生回归还要求失败时未执行的区域为正零，因此本轮没有删除清零。下一候选必须证明异常前缀、取消、独立输出及原生调用正确，不能把BLAS约定套给整个Graph。
- 当前NumPy1.26.4使用OpenBLAS，SciPy的cython_blas动态库直接绑定Accelerate；Numba的BLAS接口通过SciPy执行。见[本机依赖及源码证据](blas-provenance.json)；不能用NumPy配置冒充NJIT后端。此处确认的是本机安装版本，不外推其他平台。

## 测量范围和结果

CPU1，双方同数据、float64、相同只读owned输入副本、独立只读输出、两代结果；普通Scheduler对已预热且固定签名的NJIT。图/JIT定义和源夹具生成不计入计时。四组为131072×1024乘1024×512（long）及512×1024乘1024×131072（wide），C/C/C与F/F/F；没有为了计时变更F输出或转C。三组独立进程轮换启动/运行顺序，各4/3/3次，共120次计时。下表为三组中位数的中位数，单位秒，不能取最快值宣称胜出。每组数据见[self-review](self-review.json)。

| 形状与布局 | 当前C++ | 候选四组累积 | 暖NJIT |
| --- | ---: | ---: | ---: |
| long-cc | 0.8460 | 0.8299 | 0.7781 |
| long-ff | 1.1960 | 1.1446 | 1.0958 |
| wide-cc | 1.1833 | 1.1609 | 1.0655 |
| wide-ff | 0.8260 | 0.7489 | 0.6682 |

候选没有稳定的大幅完整请求收益，也未全面领先NJIT，**不采用**。这是收益尚不足的筛选结论，不是因几KB或几毫秒设否决门槛。计时存在调度/首轮驻留页波动，CPU与wall原始样本分别保留；不把三组样本等同于跨机器统计证明。第一版单累积器也未获得稳定优势，保留其单组筛选，未混入第二版结果。

组件诊断使用独立带计时标记构建，完整输出/校验未变，但不作为采用性能。long-CCC排除预热后的中位：输入检查0.107724秒、DGEMM0.531746秒、结果检查0.0328994秒，512MiB输出清零0.056797秒。只测此几何；不是所有矩阵的成本比例，也没有把未计时的输入准备忽略。见[分段诊断](component-diagnosis.json)。这组实际和候选都为单次DGEMM，不是高数值幅度的分段路线。

## 内存、数学与所有权

每backend各一个fresh malloc进程及一个fresh RSS进程，各三代请求。暖态申请峰值：当前C++/候选均2,156,010,256bytes，NJIT2,156,003,744bytes。6512bytes差只作诊断；候选**没有内存改善**。相同输入副本与两代结果本身占2,151,677,952bytes；源夹具在epoch外。startup malloc拦截不覆盖全部Pythonarena/mmap/栈，不能当RSS；RSS的import/源夹具基线不同，绝对/增量另报，不用绝对RSS差声称收益。见[memory-summary](memory-summary.json)及逐backend原始stages。

当前/候选numeric/mask arena、operator workspace、root copy、boundary/output/algorithm copy均0，512MiB根直接写独立结果；库准入保守reserve12,713,984bytes不是实测峰值。调用者准备仍复制两输入约1GiB，不能称完整请求零复制，也不能期待消除不存在的512MiB根复制。见[分配诊断](allocation-diagnosis.json)。

计时/峰值阶段检查shape、C/F连续、只读、输入不共享、两代不共享及旧输出三点不变；完成后各64点独立math.fsum误差最大约5e-18。是抽样数学核验，没有宣称大矩阵全元素或所有权逐字节差分。734项现有回归补充完整较小夹具、NRN/特殊值、布局、形状、prepared、内部/隔离、thread/inline/shared及两代所有权验证。

## 自测试、自审核与边界

- 最终候选原生9/9、相关Python734/734，无失败/错误/skip。见[测试记录](candidate4-quality.json)、[原生XML](candidate4-native-permitted.xml)、[Python XML](candidate4-related.xml)。
- 最初沙箱原生7过、2项shm_open权限失败，保留[初次XML](initial-sandbox-native.xml)。随后在允许共享内存的本机环境复验9/9；不能把初次失败删除或当作数值bug。
- 未完成候选全量/SAN/OFF资格，没有接入当前C++，故未进行发布资格/平台迁移/Excel桌面重算。正式源码93份source/config与当前V58wheel/worker身份核对不变；本轮没有重新宣称V58全量已运行。
- 没有新增算子、修改AST/DAG、Planner、IPC、金融语义、Excel公式或结果协议；正式执行指南与用户/Excel契约无需对应行为改写。本轮仅更新既有优化设计与本证据。
- 后续优先评估组合图的大中间数组，以及保持失败前缀的重复清零消除；仅输出本体占主导的单乘法不能作为“整体迁移收益已达到”的证据。其他主要平台链与完整复制/分配审计仍开放。

原始数据、构建身份、脚本和候选差异只作为不可执行证据留档；没有安装它们到生产入口。未提交、推送或发布。[manifest](manifest.json)给出逐文件SHA，审计可复核所有测量而非只看摘要。
