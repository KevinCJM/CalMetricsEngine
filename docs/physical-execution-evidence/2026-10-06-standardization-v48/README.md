# V48：标准化图的短向量保存和有序 Eigen 归约

2026-10-06，macOS arm64/M1 Max、CPython3.12。本轮已进入当前未提交源码；没有提交、推送或部署。整体性能目标仍为 **PROGRESS**。

## 做了什么

只修改6个native文件、新增2份测试（351项）。其他290个既存文件及原已验166份测试保持不变。公共算子、参数、AST/IR格式、worker协议及Excel数学公式不变。

- 私有、单消费者的add/subtract/multiply第一节点，可保存一个显式broadcast_like短向量，继续要求其他父值持久。保存发生在原逻辑位置和canonical prepare之后，使用第一节点自己的存活slot，不能引用已经被variance/sqrt覆盖的均值backing。规划和执行使用同一二维几何证明；已有持久父值融合优先。
- 复用固定Eigen3.4.0的Map和惰性表达式融合两次算术，直接写本次独立C结果；F继续已有列路径。短分母为零时，在库写入前回原路径，保持原错误前缀。
- C连续float64矩阵、N≥32且8≤D≤256的非显式scalar轴0和/均值/方差，复用Eigen按连续行归约。每列仍按原行顺序累加，方差保留两遍中心化、ddof和逐列错误。方差使用两份最多256-double栈Map；有序和非有限时写入前回放原实现。F、其他步长/宽度、显式scalar及库关闭保持各自当前职责的原生执行。

设计和审核见 [设计](v48-design.md)、[自审核](v48-self-review.md)。源代码和补丁以`.txt`保存于`sources/`，测量/资格执行器以`.py.txt`保存于`drivers/`，均为惰性证据，不构成第二套运行实现。

## 质量与实际接入

| 检查 | 结果 |
| --- | --- |
| 冻结候选全量 | 15,628通过，0失败/错误/跳过 |
| 正常相关/native | 1,235 / 9通过 |
| 全引擎ASan+UBSan相关/native | 1,235 / 9通过 |
| Eigen/Accelerate/Highway均关闭，相关/native | 1,235 / 9通过 |
| 实际目录重建，相关/native | 1,235 / 9通过 |
| 额外转置窗口审核 | 上述四种配置各40个结果；不同长度、空窗口和两代所有权通过 |

覆盖运算与操作数顺序、C/F/反向、空/短/零几何、NaN/Inf、平方/求和溢出、原out失败前缀、共享根、多个请求形状、prepared当前输入、四种执行路径与独立结果。执行模式不能用于替代数值证明。

实际source/wheel/worker build_id为 `30722fb25c692505f924aec6755d8f3c5a6113507b9b54552fb22a895d8ab365`；93项native/build输入与冻结候选一致，实际二进制SHA另外核对，不能把同源误写成二进制完全相同。ASan与OFF拥有各自构建身份。[接入/复验](v48-adoption-status.json)、[资格原始记录](v48-qualification-status.json)、[实际执行器记录](v48-actual-replay-status.json)和XML/log一并保留。

第一次ASan专项1234通过、1失败：框架Python启动后重写sys.executable并丢弃DYLD环境，测试子进程经launcher晚加载ASan，在算子调用前SIGABRT。测试专用bootstrap恢复真实framework可执行路径及子进程preload，先核验全局ASan符号和同一空轴用例，再完整重跑同一1235项；没有修改引擎/测试、跳过用例或放宽断言。[原失败](v48-asan-launch-failure-1/v48-asan-related.log)和原XML/执行器保持不变。

早期r0的通用8值span对完整C请求无实质提速，已移除；r0/r1/r2原wheel身份、片段和负结果保留。构建拼写、scalar-broadcast和直接算子所有权等测试期修正也保留原始记录，不把修正后的通过反写到旧报告。

## 完整时间和内存

固定N=1,048,576、D=32、seed20261006，普通Scheduler/预热NJIT，双方CPU预算1。每次都计入独立只读输入准备、执行和独立结果返回，保留两代输出；夹具生成、定义构建和AOT/JIT编译不计。NI复用其均值来计算两遍方差，不人为增加重复扫描或验证费用。新增缓存不跨请求保存数值。

候选36次配对、实际36次配对及因wall/CPU差异补做的同场64次配对均终结；原始样本不删除、不以追加复测覆盖首次数据。下面是同场64次完整请求的中位数（秒）：

| 布局 | 原V47 | 冻结候选V48 | 实际V48 | 预热NJIT |
| --- | ---: | ---: | ---: | ---: |
| C | 0.24036 | 0.13097 | 0.12990 | 0.09682 |
| F | 0.38217 | 0.10299 | 0.10820 | 0.13546 |

实际相对原版本减少约46.0%/71.7%耗时；F相对NI减少约20.1%，C仍慢约34.2%。此前实际36次中位数C/F为0.16794/0.10986，原V47为0.25193/0.40545，NI为0.09133/0.14564；部分wall显著高于CPU时间。追加同场比较中实际/候选C为0.992、F为1.051，未显示同源实际构建稳定地变慢；不会据此删除首轮波动或把CPU时间替代wall门禁。[首次实际时间](v48-actual-time-status.json)、[同场复核](v48-actual-time-recheck-status.json)及每backend原始结果可复核。

候选和实际各36个顺序fresh内存进程，分别测malloc及0.5ms RSS采样，每backend/layout/metric有3次独立进程；不与构建/测试/正式计时并行。

| 指标，C/F一致 | 原V47 | 实际V48 | NI |
| --- | ---: | ---: | ---: |
| warm完整malloc峰值（bytes） | 1,074,021,256 | 805,586,056 | 805,309,122 |
| 数值arena（bytes） | 268,435,968 | 768 | 不适用 |

自身申请峰值减少268,435,200 bytes（约256MiB），NI与实际相差276,934 bytes，按当前政策视为基本持平，不能称为大额领先NI。每请求复制短向量256bytes，根直接写268,435,456bytes，根复制0。调用方256MiB输入准备虽然不属于绑定审计的input_copy_bytes，仍完整计入请求时间/峰值。

实际RSS峰值C/F约1.1673/1.1676GB，原V47约1.4358/1.4361GB；NI约1.2109/1.2113GB。RSS包含source fixture、imports、JIT常驻及运行库等基线，是补充页级观察；绝对RSS的NI差异不能代替数值工作区/分配收益证明。malloc不覆盖部分Python arena、特殊zone、mmap或栈；追踪器无table overflow也不代表完整分配审计。[实际内存](v48-actual-memory-status.json)保留逐次样本、范围和盲区。

## 还差什么

C标准化仍存在额外均值扫描及有序归约成本；通用大矩阵FF、其他主要完整链及对NI的大额内存优势仍未全部闭环。当前结果只证明这个既有算子组合的接入与收益，不证明平台服务已替换NJIT、业务验收或跨平台CI通过。下一步优先减少整矩阵重复读取、评价成熟库的实际大矩阵收益；不以小幅KB/ms变化单独否决或结项。

用户指南和算子参考不改：公开名称、签名、默认值、shape/dtype、错误及所有权不变。执行指南和物理优化记录同步物理准入及测量范围。Excel仍展开同一逻辑DAG，原公式和预算不变，本轮不重跑Microsoft Excel或宣称新的Excel验收；已有相关生成/预算测试属于全量回归的一部分。

参考：[Eigen惰性求值](https://eigen.tuxfamily.org/dox/TopicLazyEvaluation.html)、[广播](https://eigen.tuxfamily.org/dox/group__TutorialReductionsVisitorsBroadcasting.html)、[别名规则](https://eigen.tuxfamily.org/dox/group__TopicAliasing.html)。官网现在重定向开发版文档，本实现仍固定3.4.0。ArmPL仍是本机大矩阵候选，尚未安装、对照或采用；此前BLIS完整负结果不重试为已获收益。
