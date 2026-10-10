# 同 A 双 solve_multi 共享已有 Eigen 消元

2026-10-03，macOS arm64，Git HEAD d17b371；本地未提交/推送。
旧构建 `437269d43960c791fd13984fba0051d3df5a72d3dfbe03a8d96414df6aa030c7`，新构建 `295a46460a8c783dd5b764f8eb37b556dd2d1d22997c070308ef4540efe6a3a3`；构建身份不是 Git SHA。

仅同 A、两个独立终端 solve_multi、静态 rank-2 float64 输入且两个实际输出均为 C 时，复用现有 Eigen ordered elimination。固定一个/两个 RHS span，保持 A 主元/更新树和各 RHS 原运算树；完整回代第一 RHS 再完整回代第二 RHS。复用已有 N² 工作区，每个根直接写独立结果；只复制一次 A，不拼接 RHS、不新增 pivot/矩阵/跨请求缓存或线程。auto schema 只排除明确 F，运行期仍检查实际 C。[完整设计](design.md)和[静态自审核及 root 验证状态](self-review.md)记录准入与失败处理。

第二节点投机校验拒绝时不提前报错；共享数值失败先恢复第二写入区为原零，再按原节点顺序重放。scope/isolate/observer/prebound/故障继承、非终端、临时输入、不同 A、第三根及 F 结果不共享。AST/逻辑 DAG/数学/Excel公式、IPC与公开 API 未改。typed 不走 dag_branch；Planner 保留原两个求解的保守工作量、完整工作区与输出容量预算。

## 完整请求计时

8类、每类6轮平衡换序、CPU=1，实际旧/新 wheel 和已预热 NJIT。普通请求含绑定/规划、执行和全部两个独立只读 C 输出物化；保留前一次结果。prepared 只预备计划/绑定，仍重读输入并完整重算。强 NJIT 在调用内拼接 RHS、一次 np.linalg.solve，再复制成两个独立 C 输出；不拿两次弱参考替代。旧/新结果逐位相同，NJIT 保留原1e-11并独立检查残差/后向误差1e-12、输入hash和所有权。

单位ms，中位数；c/f是输入布局，输出均C：

| 请求 | 旧C++ | 新C++ | NJIT |
| --- | ---: | ---: | ---: |
| 32x1plus3_c_ordinary | 0.025 | 0.024 | 0.011 |
| 512x3plus5_c_ordinary | 14.430 | 7.901 | 2.542 |
| 2048x8plus32_c_ordinary | 699.604 | 376.270 | 92.766 |
| 2048x8plus32_f_ordinary | 732.592 | 391.169 | 67.608 |
| 4096x8plus32_c_ordinary | 5094.873 | 2654.035 | 693.015 |
| 4096x8plus32_f_ordinary | 5256.714 | 2760.407 | 574.129 |
| 2048x256plus257_c_ordinary | 1170.512 | 780.242 | 132.077 |
| 4096x8plus32_c_prepared | 4952.665 | 2621.636 | 672.792 |

4096 C普通 **5.095→2.654秒**，省2.441秒；F普通省2.496秒，prepared省2.331秒，中大型较旧版省33–48%。但4096 C仍为NJIT的 **3.83倍**、F为4.81倍、宽RHS为5.91倍，整体目标未完成。32阶相对NJIT多约12微秒，仅诊断。几KB/ms正差不单独作为门禁；中大型实质差距不靠该政策豁免。

## fresh内存

24个独立子进程：4负载×旧/新/NJIT×heap/RSS；12次heap共12个连续epoch、72个阶段快照。输入、导入与JIT在epoch前；请求准备、首次数值分配、工作区复用、5次热身与3次保留结果替换仍计量。RSS各3次，报告增量，不能混成总RSS。

单位MiB：

| 请求/后端 | 冷申请峰 | 热申请峰 | RSS增量 |
| --- | ---: | ---: | ---: |
| 4096x8plus32_c_ordinary/old | 129.267 | 130.516 | 131.203 |
| 4096x8plus32_c_ordinary/new | 129.267 | 130.516 | 131.203 |
| 4096x8plus32_c_ordinary/njit | 133.544 | 134.794 | 146.516 |
| 4096x8plus32_f_ordinary/old | 129.267 | 130.516 | 131.172 |
| 4096x8plus32_f_ordinary/new | 129.267 | 130.516 | 132.484 |
| 4096x8plus32_f_ordinary/njit | 133.544 | 134.794 | 146.656 |
| 4096x8plus32_c_prepared/old | 129.267 | 130.516 | 132.516 |
| 4096x8plus32_c_prepared/new | 129.267 | 130.516 | 132.453 |
| 4096x8plus32_c_prepared/njit | 133.544 | 134.794 | 145.406 |
| 2048x256plus257_c_ordinary/old | 40.032 | 48.047 | 48.766 |
| 2048x256plus257_c_ordinary/new | 40.032 | 48.047 | 48.906 |
| 2048x256plus257_c_ordinary/njit | 49.684 | 57.700 | 77.875 |

所有新旧冷/热申请峰差 **16字节**，照实记录而不否决。4096 C普通RSS增量相同；F普通多 **1.3125MiB**，prepared少64KiB、宽RHS多144KiB，不外推为普遍增长或节约。旧双根已串行复用N²工作区，不能宣称本轮峰值减半。4096 C算法复制 **269746176→135528448字节**，正好省一份128MiB A；这是复制流量节约，不是128MiB峰值节约。

表溢出均0；old/new各阶段表外释放计数相同：setup/cold=0，cold_retained=1，warm_baseline=5，warm/warm_retained=6；NJIT为0。Python arena子分配、mmap/栈、厂商缓存及这些表外释放仍限制完整生命周期结论，不能记为零或推断泄漏。探针SHA256 `74421dbfff14b56b730553c7b0b0ce9b38a136de053f132069e1b948decd3382`。内存证据支持本轮取舍，不构成全引擎内存全面领先。

## 验证与维护

- Python全量 **11129通过**；ON/OFF五文件专项各 **332通过**，新增39例覆盖成功逐位结果、复制审计、主元/阈值、零列、7类故障、排除条件、prepared与thread/process/shared所有权。
- 原生、ASan/UBSan、三库OFF原生各 **9/9**。两项 caller-owned 失败缓冲回归与原逐节点 observer 路径逐字节比较，含边界canary和输入位模式。Mac ASan使用detect_leaks=0，不声称LSan或厂商库插桩。
- 185直接算子scalar/auto夹具通过；历史Microsoft Excel469例/3056值复核通过，**无新桌面重算**。本次不改变数学或Excel逻辑。
- 77项源/构建输入、ON/OFF wheel/installed/worker身份与包内容核对通过；最终测试源单独冻结，生产补丁包含本轮5个生产文件及native/Python回归，不混同此前共享工作区的完整Git差异。
- 首次15项失败的focused.log、初版identity-first.json与构建日志保留：默认auto资格误被排除，所有数值仍一致；修复schema排除显式F并保留实际C守卫后重新构建，最终专项332通过。没有改容差/门槛或删除失败案例。
- evidence-validation.json只校验测量记录、hash、轮次/复制/线程/内存阶段；不是采用判决。remaining.log与最终身份记录另证明root串行执行完成；自审核代理仅做静态审阅。

已同步执行指南与物理优化总览。公开类型/算子数学/布局/所有权/下载契约不变，其余用户指南、参考表和AGENTS无需重复修改。实际命令见[commands.md](commands.md)。CPU利用率、线程扩展及IPC分项性能本轮未重测；thread/process功能回归不能代替这些性能证明。

本目录仅保存小型脚本、原始日志、JSON、设计/审核及差异；wheel/install、大数组与xlsx保留stage，不进入证据目录。脚本中的本机路径和旧构建快照是实际记录；异地复现须显式准备相同源/旧新包/工具链与历史Excel快照，选择新stage/output，不能直接覆盖证据或把新源码冒充旧快照。manifest.json记录所有冻结文件SHA256（自身除外）。未提交/推送，整体目标继续开放。
