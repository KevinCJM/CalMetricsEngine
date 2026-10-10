# 保序Cholesky行面板自审核

范围：只优化既有cholesky物理实现及明确工作区预算，不新增业务/逻辑算子，不改函数签名、数值/错误/结果/Excel契约。工作树此前已有大量未提交修改，旧GEMM和其他算子保持；以本轮start-manifest及source-build identity辨别范围。未提交、推送或触碰研究平台。

## 数学与依赖

- 完整Eigen LLT/LAPACK历史候选在抵消/病态矩阵上改变原运算分组，因此仍不采用；新的小幅内存政策不是数学豁免。
- 每个输出元素继续按k递增做独立乘法和减法；Eigen只将互相独立的列作为SIMD lane。8步表达式不变成先求和再减；不使用fast-math/FMA。
- 16行的先前列因子只依赖已结束行；同一8列内因子按u递增求出。begin是16的倍数，且16能被8整除；最后不满16行仍有完整前缀组，尾行remaining>=1。
- 前缀各行因子互相独立，镜像写入已经完成行的未用upper部分；row_finish只处理begin..i，仍使用原canonical除法/sqrt实现，避免维护一套平行数学。
- finite-before-exact-symmetry检查仍在计算前完成。面板重排可能先见后行overflow；仅DOMAIN_ERROR/NON_FINITE_RESULT触发canonical行序重放，资源/形状/结构失败不吞掉。重放从只读输入重建全部已消费lower/upper值，不读未初始化/过期镜像。

## 内存、预算与所有权

- 输入Value按原strides读取，包括C/F、padded、负stride、zero stride；不改调用者数据和布局。输出继续实际C连续、独立拥有，没有结果借用workspace或新增隐式输出copy。
- 数值面板16*(n+17)double，因素表128double，全部来自既有独占Workspace；Eigen Map表达式不拥有堆数组，没有完整n²中间矩阵。必要双下三角复制明确审计。工作区不能共享给并发调用。
- checked shape bound避免n+17、乘16及字节数溢出；registry在显式非auto时不分配新面板。unknown shape以capacity作保守阶数，预算不凭实际数据内容压低。
- 将已有matmul panel budget扩展为顺序matrix panel max；所有子作用域/worker仍在原递归预算中独立计入。任一panel上界变化时释放旧数值workspace，覆盖最大arena面积不变的变阶。其他solver已有保守通用workspace预算维持。
- 无新线程/全局BLAS状态/JIT/Python回调；单核库工作使用当前Scheduler线程。大负载变快不等于应启用多进程。

## 验证与边界

- 原型两轮配置、全部慢样本均保留；不针对每个shape挑最快配置。生产固定R16/K8，n>=1024；编译平台未验证者走原路径。
- 专项第一次有2项测试预期错误：既有iterate会保留最后有限状态，诊断status=5，并非把整个root标为失败。依既有test_iteration_protocol修正测试，增加直接失败root与健康root对照；未改变生产迭代逻辑，原失败日志保留。
- 24项新增专项已通过。原生/ASan/EigenOFF、全量Python、185fixture、源/wheel身份与既有Excel已重算快照对照结果以最终summary为准，不将运行中检查写成通过。
- 正式性能使用实际安装包完整普通Scheduler/Prepared快照，与旧版和NJIT单核配对，保留公开输出物化和上一独立结果；输入/编译/图构造不混入暖时延。内存另用fresh child，从setup贯穿cold和两次warm，heap与RSS分开。
- 几KB/ms增量不独立拒绝；本轮重点是4096矩阵秒级收益，新增有界工作区诚实计量，不能宣称内存零增长或全部算子已胜过NJIT。Excel数学未变，本轮只验证历史真实重算快照，不声明新Excel重算。
