# 多右端保序求解优化：2026-09-30

范围：以75229d17构建为基线，仅在已验证平台将 solve_multi 的大矩阵消元接入 Eigen fixed16×2/panel32。最终构建4e12b6bb；完整SHA和wheel/源码身份见source-identity.json。没有提交、推送或平台接入声明。

结果：Python7062、原生/ASan+UBSan/Eigen-OFF各9/9、185原语回放通过。104组时间、33组内存子集（198 child）已执行；新路径48组比旧C++快13.5%–40.6%，但0/48领先NJIT。冷/热活跃堆33/33与旧版一致，数值调用方栈752B持平，完整RSS和首次OS检查系统栈不增未证明。旧性能门禁3/4，63点普通入口比值1.0002202，保留失败。联合验收未通过。

主要证据：

- acceptance-status.json：分项结论和未关闭条件。
- final-acceptance/timing：104组原始轮次、完整输出核验与构建身份。
- final-acceptance/memory：33组冷/热堆及RSS；不是104组全内存覆盖。RSS单pair，不作统计不增证明。
- final-static-review.md、inline-production-review.md：数学、错误/所有权及正式编译对象栈审计。
- python-refined.log：含129阶末面板失败四lane传播的最终全量；python-final.log为加强该夹具之前的一次全量。
- phase2：原始门禁结果，未修改阈值。
- excel-snapshot：469组/3056值对历史真实Microsoft Excel重算快照复比，newly_recalculated=false。输入/公式不变，只允许Readme构建身份变化。
- candidate-evidence.tar.gz：未采用的原型、编译命令、逐位回归和静态证据，包含首次动态4生产候选的已知栈回退；不能当最终版本验收。Jacobi没有生产改动。

正式工具 tools/check_solve_performance.py 显式传入旧/新安装目录；本轮33组子集选择及命令见run_final_solve_acceptance.py。direct计时不额外求解收集audit；普通/prepared完整物化输出，prepared使用独立快照。所有计时与构建/回归串行执行；CPU预算1，原四组门禁按原预算4。

微任务容忍与旧严格门禁分别报告；活跃堆、数值栈与RSS分别报告。无需因本次物理优化改数学签名、公式或预算。尚需继续大矩阵对NJIT性能和完整内存验证，不能据此声称总体目标达成。
