# V62：矩阵与标量完整链隔离实验（未接入）

日期2026-10-07；实际目录仍为V61，候选D仅在 `/private/tmp/calmetrics-matmul-scalar-chain-v62-20261007/source`。
本记录不表示性能目标完成，不表示提交、推送或部署。原AGENTS及377份受保护文件在记录文档前逐SHA未改；实际93份source/config与V61已安装wheel/worker一致。研究入口见[公开来源](research.md)。

## 修改与依据

复核Eigen官方BLAS/LAPACK委托、惰性表达式与别名，以及Netlib DGEMM beta=0约束。现有Eigen3.4与Accelerate已具备底层能力；本轮没有升级库或新增依赖。Nightly文档只提供方法指导，不是已升级版本的证据；无法读取的Apple正文和3.4镜像不充当数学证明。

隔离候选通过私有storage region让单消费 `matmul(a,b)*0.5+0.25` 直接写独立C/F根，复用既有DGEMM；标量两步用成熟Eigen Array Maps融合，并在写入小块时检查最终数值分离。输入检查、保守向外舍入证书、非适用数据回原数学、含糊符号/零值重放、初始typed清零及逻辑准备顺序仍保留。不是新增算子，也没有Python数值回调、新线程池或新IPC协议。

准入仅为有限normal有界输入、nearest、最多8步add/subtract/multiply和静态可证明独占的rank2根。输入最大幅度及累计误差按原绝对预算0.000049证明；没有降低预算或改变离散断言。divide、额外消费者/根、负常量的negate逻辑、作用域和其他未证明路径保留原路线。具体边界见[最终设计](design-guarded.md)及[数值证书](design-library.md)；未知shape/子作用域Planner包络尚未完整验收，不能据此接入。

## 完整请求时间

CPU1，同一owned readonly输入准备、普通Scheduler或已预热相同固定签名NJIT；请求每次含两份输入副本，结果为同样要求的C或F布局，保留两代独立只读输出。编译、fixture、预热及最终逐数组校验不计入计时。输入为8192×1024和1024×8192，根输出512MiB。每布局/后端3个fresh进程，各3次暖请求，三组轮换启动顺序；总54次计时。

| 布局 | V61秒 | 候选D秒 | NJIT秒 | 候选/NJIT |
| --- | --- | --- | --- | --- |
| C | 6.379005 | 0.703530 | 0.630910 | 1.1151 |
| F | 6.310471 | 0.696521 | 0.633501 | 1.0995 |

以上为fresh进程中位数的中位数。大幅优于旧C++，但本几何仍比NJIT慢约63–73ms，不宣称中大型计算全面领先。不能因为几KB或几ms设置硬门禁；本候选尚有完整质量资格缺口和总体性能目标未达成，决定暂不接入。

## 完整内存与审计

每布局/后端各一个独立fresh malloc进程和一个独立fresh RSS进程，各三代请求。malloc观测期排除fixture/模块导入/JIT，表溢出为0；RSS增量相对请求前驻留基线，0.5ms抽样，不能与申请峰值或总RSS混算。

| 量 | V61 | 候选D | NJIT |
| --- | --- | --- | --- |
| C/F申请峰值bytes | 1,747,464,724 | 1,214,259,180 | 1,744,830,992 |
| C RSS增量bytes | 1,680,850,944 | 1,153,875,968 | 1,694,236,672 |
| F RSS增量bytes | 1,680,834,560 | 1,153,908,736 | 1,693,007,872 |

候选相对NJIT申请峰值少530,571,812bytes，约506MiB；相对V61少533,205,544bytes。这是本完整fixture的实测收益，不能推广所有图。候选每请求audit有1次DGEMM，numeric arena/root copy/algorithm copy均为0，vendor reserve12,582,912bytes；原V61 numeric arena536,870,912、algorithm copy4,361,027,584bytes。输入准备仍真实复制128MiB，vendor reserve是预算而非等额实测分配，不称全链零复制或全面复制审计完成。

每个最终进程在测量结束后检查最后两代全输出，按32行块比较相同布局NJIT参考，绝对预算0.000049、相对0；候选最大尾差0、原V61约5.55e-17，所有权和只读断言通过。较早screen仅为单组诊断，不能混入上述轮换结论。

## 质量、失败与未完成项

最终D新增Python78/78，原生9/9，零fail/error/skip。新原生头含16类C/F、分块/未分块、操作数左右与故障发生位置夹具，覆盖除法输出前缀和输入不变。两次初始native错误预期已保留：既有row-batch只提交完整tile，而未分块writer会留下部分前缀；此前将两者按同一oracle比较不正确。最终测试分别使用各自既有契约，没有修改既有row-batch实现。初次新增Python60通过/18失败只存在工具输出，没有独立原始XML，本记录不冒充已有完整原始日志；问题涉及负常量negate fallback、单步arena未开启融合及fallback arena的错误预期。

候选修复单步链的unused arena、排除divide存储借用并保留原生首错/前缀；negative scalar参数仍测试有效路径，negative literal不被折叠成另一逻辑图。失败日志和修正版脚本全部保留，没有删掉原数值或错误断言。

本轮没有候选完整回归、ASan/UBSan、库关闭、跨平台或真实Excel重算；不能搬用V61历史成绩当本轮资格。未知shape/子作用域Planner库空间包络、resource/timeout/cancellation全部分支、其他平台代表链和全面分配/复制审计继续开放。候选不接入正式代码，下一步先完成这些契约证明与完整链成本定位，再决定是否采用；整体优化目标仍未完成。

## 证据与复现边界

- [最终汇总](summary.json)、[30个fresh测量进程原始结果](final-measurements.json)、[自审核](self-review.json)。同名30份 `.log` 保存各进程原始输出。
- 最终[Python XML](quality-guarded-r3-targeted.xml)、[native XML](quality-guarded-r3-native.xml)与三个版本的失败/成功日志均保留。
- `builds/` 保存A/B/C/D身份、安装及构建输出；`baseline-actual-v61-identity.json` 保存实际身份。没有wheel或机器码重复副本。
- `candidate-d-source/`、`baseline-source/`、A/B/C快照及patch均是带 `.txt` 后缀的实验文本，不能参与当前构建或CodeGraph代码解析。实验脚本也为 `.py.txt`，执行前须恢复隔离文件名并核对其记录的同源V61依赖和本机probe/control，不能声称本记录可脱离基线独立构建。
- `manifest.json` 校验归档文件内容；`closeout.json` 记录本轮实际源未变及文档检查。正式源码仅为现有唯一V61实现。

本轮只同步现有优化设计与该实验记录。公开算子、AST/DAG、用户/执行/Excel说明没有本轮实际行为变化，因此无需重写；成熟库与多轮失败后调研要求已在计算纪律中，无需重复修改AGENTS。
