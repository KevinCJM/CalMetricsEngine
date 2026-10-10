# V63 D：保留既有跨布局库路线的隔离候选

2026-10-07。C完整正常回归通过后进一步静态审核发现：elementwise_pair的早选只检查matrix/scalar外形，选择后却因不连续/不同布局被内部Eigen scalar拒绝，再走通用逐单元循环。这个早返回绕过了原有Eigen column/tile路径，可影响F输入/C结果、pitch列和反向F行。功能探针已确认两根图可触发scalar pair，尚未得到大矩阵退化的实测；本设计不先宣称退化量级。

在独立source-layout-guard中抽取唯一的same-order源资格选择器，返回0/1或不准入。前置选择和实际Eigen执行共享该资格，资格不符继续已有column/tile或其他canonical路径。同序MatMul根链仍采用C已有成熟DGEMM/Eigen组合，没有加入新算子、参数、IPC字段、审计接口、缓存或线程。

原除法右标量非零条件保持；未通过资格不提前计算或写数据。数学仍是原来两步有序算术，不将标量移入BLAS alpha，不改NaN/Inf/零符号、错误前缀和所有权。布局保持调用者要求，不能借测试将所有结果统一转换为C或F。

质量中的C源码、三个C构建及C最终测量保持不变，D只是另一个隔离候选。先完成C的24进程两根跨布局完整请求对照与原48进程MatMul链；若确有大额退化，再构建D、验证其关联/完整/SAN/OFF、同源完整性能及内存，并与C和V61保留的原负载配对。不能将C完整成绩或未执行的D设计冒充D资格。几KB/ms只作诊断，不单独阻断或宣称成功。实际源码仍为V61，整体目标未完成。

## 大矩阵跨布局对照已完成

C的24个fresh进程已完成并逐单元核验两代两个独立根。输入4096×8192、256MiB F矩阵；一次完整普通请求含owned只读准备复制、scalar pair和transpose两个各256MiB根，CPU=1。四组轮换中位数再取中位数：C结果由V61的0.136316秒变为C的0.270366秒（1.9834倍，增加0.134049秒），证明前述早返回确有退化；F结果0.311256→0.317577秒仅增加6.32ms，是同布局控制诊断，不单独设置门禁。申请峰均约1,342,189,710bytes，C候选仅增加16bytes，不能称为内存退化。两根普通请求而非pair-only微基准，尚不能外推平台所有负载。

D另新增144项Python跨布局/stride/两步operand顺序/NaNInf/零符号/两代所有权用例，以及24个原生布局×结果顺序×除法错误前缀类。它们尚未执行；C已通过的16370正常、2792 SAN（未开启泄漏检查）、OFF 2400通过/392按库资格skip、各native9，只属于C。D构建和所有后续测量必须等C串行任务结束，保持对照稳定。

## 公开方法依据

复核[Eigen storage orders](https://libeigen.gitlab.io/eigen/docs-3.4/group__TopicStorageOrders.html)与[Map/stride](https://libeigen.gitlab.io/eigen/docs-3.4/classEigen_1_1Map.html)：遍历方向要匹配布局；Map可借用现有内存并显式表达outer/inner stride。网页当前为3.4.1，项目仍是已有3.4.0，本轮没有依赖升级。D由此恢复已经测过的既有Eigen packet/tile路线，保留明确要求的C/F结果，不再自己重写矩阵转排数学。资料只说明机制，实际收益须由D完整请求实测给出。
