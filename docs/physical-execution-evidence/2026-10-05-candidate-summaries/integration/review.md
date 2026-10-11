# 私有矩阵统计消费者：实现与自审核

本轮只改变已有 `matmul → sum_axis/mean_axis/variance_axis` 组合的物理执行。
未新增公开算子，不改变逻辑 AST/DAG、接口、金融定义或 Excel 公式。遵守
AGENTS、计算纪律 D1–D8，以及 Excel 纪律 X1/X6/X9/X10。

## 数学与准入

对于 W∈R^(N×K)、R∈R^(T×K)，定义 Y=W Rᵀ，
μ_j=Σ_t R_tj/T，C=(R−μ)ᵀ(R−μ)。采用同一中心化样本方差：

```text
mean(Y_i) = w_i μ
sum(Y_i) = T w_i μ
variance(Y_i, ddof) = (w_i C w_iᵀ)/(T-ddof)
```

只有非公开、输入/稳定借用视图支持的 matmul 才准入；所有直接消费者必须是
axis=1 的上述三个统计。完整下游闭包只允许终端 float64 向量、正有限常量乘法
及至多一次 sqrt。额外明细、比较/排序/状态及任意未证明表达式恢复原矩阵路径。
图本身不含作用域。Observer、延迟故障、isolate、输入故障和 prebound 子程序
保持原执行；Excel observer 获得真实 Y，不会读物理占位描述。

共享几何证明限制 32≤N,T≤1,000,000、0≤K≤128，且形状完整、收缩轴一致。
其余形状保持既有执行。动态/非法 ddof 不重排统计，但仍可用有界 canonical
回放，在原 variance 节点检查参数及短样本错误。

成熟库快路径要求 macOS15+、Accelerate、可证明的正 C/F 权重 strides、
FE_TONEAREST、有限且 |input|≤1e50。R 可按通用 strides 读入有界面板。
按常规 FP64 操作误差模型估计 dot/中心化/Gram/二次型的保守包络，传递完整
下游正缩放及 sqrt 放大后不得超过 4e-5；这是实现准入，不是结果舍入。
包络包含 canonical 和库路径的折叠差，使用最大 |R|、最大行 ℓ1(W)、
γ((16T+16K²+256)u) 及64倍余量。该工程包络不声称证明所有厂商实现的形式精度。
大偏移、超大放大、非有限、未知舍入方式和不合格 strides 选择 canonical。
负/非有限二次型以及 mean→sqrt 的不确定零边界逐行回放，不截负方差、不加正则。

## 内存、执行与错误

Accelerate 使用 DSYRK/GEMM/GEMV/DOT，当前调用线程限制单线程并恢复，服从
Scheduler CPU预算，无私有线程池。R中心化面板最多512观测；候选积最多1024行。
统计缓存每组2N doubles，共用辅助区上界为：

```text
T + 2K + K² + min(512,T)K + min(1024,N)K doubles
```

Planner/执行共享几何和容量推导，checked 加法；缓存/offsets按本次容量释放或重建。
另保守计入1 MiB厂商资源估计，它不是显式分配，也不是厂商内部严格上界。
audit 使用已有库计数/复制/arena/workspace字段，不新增wire协议。
typed 工作量仍保守覆盖数值拒绝后的完整回放，不能把快路径作为最坏上界。

canonical 回放每行最多4096观测一块，调用当前标量 matmul，再按原观测顺序
累计 total 和两遍中心化 numerator；不临时创建 N×T。K=0直接写零，避免空指针算术。
NaN/Inf/溢出在producer缓存，不提前移动消费者错误；逻辑消费者先prepare，再执行
原短样本及finite检查。独立错误优先顺序、isolate健康根、动态ddof仍按原契约。
扫描、库块和回放块检查cancel/deadline。没有跨请求数值缓存；prepared本次输入
变化后重新计算。旧结果与输入只读且不能当作工作区。

## 已验证与边界

同源普通完整 Python 13,074项，无失败/错误/skip；新增63项包含其中。
普通原生9/9；实际ASan/UBSan扩展相关254及原生9/9；三库OFF相关254及原生9/9。
ASan与普通库参数不同，但89个数值/配置文件一致；没有全量SAN/OFF、LeakSanitizer
或x86/Linux/Windows新验收声明。

新增覆盖 C/F/反向/稀疏步长/广播、输入不变、空收缩维、多个统计消费者/producer、
独立公式与非负性、取消恢复、观察器、错误顺序、动态ddof、四运输、两区间两代、
live prepared输入、isolate及预算/容量收缩。Excel诊断参考与真实逻辑图相符，但
`NOT_RECALCULATED`；没有本轮Microsoft Excel重算。

公平对照包含流式NJIT及采用相同共享统计量算法的NJIT；只测给定候选的均值与
标准差数学完整链，不代表随机采样/修复、EWMA、下行风险、VaR/CVaR、回撤、
离散筛选、Pareto、完整前沿或平台服务。编译排除，双方CPU1、owned readonly
输入准备、普通Scheduler/结果返回及两代独立结果计入实际范围。

初次性能驱动的for-loop局部a/b额外保留了一代旧视图，且C++共享根owner与NJIT
独立数组owner会多保留不同字节；原始三组计时及16个内存任务保留在
negative-retention，修正驱动后全部复测。不得用原测量关闭内存验收。

仍开放：相同算法NJIT与原生之间的小毫秒差；候选统计相对流式NJIT的额外scratch；
一般strides数值回放的中大型性能；完整风险/选择消费者及其他主要模型/模拟/求解。
本阶段重大收益是消除既有大矩阵和重复计算，不据此宣称全引擎目标已达成。

正式接入补证：九份代码/测试按预像接入，实际目录正式重建包另验254项/原生9/9，源码、扩展与worker身份c605一致。24例变化窗口补验通过。正式CPU1六轮及12个fresh峰值任务的范围、全部结果与残留3.602MB申请峰差见README和acceptance.json；不能关闭整体目标。
