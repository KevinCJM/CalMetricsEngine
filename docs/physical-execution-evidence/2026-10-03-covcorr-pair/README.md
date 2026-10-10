# 协方差与相关矩阵配对输出验收（2026-10-03）

本轮采用已有 Accelerate DSYRK、Eigen 面板和 canonical 归一化，复用同源协方差结果。大矩阵配对输出消除一次重型计算及 F 根临时方阵；全算子性能目标仍未闭环。仅本地修改，未提交或推送。

## 实现及数学边界

- 恰好两个终端 typed float64 矩阵根，按现有节点执行顺序为 covariance(x)、correlation(x)，父节点相同。两根均不得有内部消费者；反序、重复、额外根、不同父节点、isolate、作用域不准入此次库扩展。
- covariance 候选同时满足原协方差与归一化相关矩阵误差证书，原 1e-12 容差、canonical 对角、符号/零、局部及密集重放不变。配对路径仍使用协方差源幅度限制，并未继承单根相关矩阵的扩大范围。
- 两个 F 对称根各自直接写最终不重叠结果区，完成后公布真实 F strides。已完成协方差的 C/F 连续存储均可复用；第二个独立结果仍需要一次 D² memcpy，计入算法复制，不声称全链零拷贝。
- 配对 C 输入的中型面板限制为 1024 行，复用已有分块，不新增数学实现或缓冲。F 输入保留完整面板及已有 centered fallback；Planner 继续保守预留。observer/prebound 保留原数学路径，observer 仍可用对称 F 根存储。
- 只有物理执行改变：公开算子、AST/DAG、dtype/shape、IPC、错误与结果所有权、Excel 公式不变；无新依赖、缓存或私有线程。详细理由见 [DESIGN.md](DESIGN.md)，准确任务差异见 [implementation.patch](implementation.patch)。

## 对等完整调用计时

实际上一阶段 wheel `59224ee281be4fb2b0d01a62b17cf6b3a5f562d39a5c310eebac8ab3be04d076` 对比最终 wheel `a129d662535bc5e6ebe20f2dc27e4ce7baa7376bfe8d38b23486a0f284d96142`。macOS arm64，普通 Scheduler、1 CPU、9轮均衡次序，两个独立结果完整物化并保留上次结果。图编译及 NJIT 签名预热不入计时；普通入口绑定、规划、执行及返回纳入。NJIT 用 np.cov 一次并直接写第二个相关矩阵，不强加 C++ 的复制步骤；F 结果使用自身对称矩阵的转置视图。

| 场景 | 旧 C++ ms | 新 C++ ms | NJIT ms |
| --- | ---: | ---: | ---: |
| small63_c | 0.009209 | 0.009875 | 0.003791 |
| small63_f | 0.011042 | 0.010667 | 0.005333 |
| medium512_c | 2.451417 | 1.011042 | 0.875625 |
| medium512_f | 5.398959 | 1.176625 | 0.809750 |
| medium2048_c | 608.316875 | 99.040791 | 125.598083 |
| medium2048_f | 1245.485666 | 116.329709 | 111.988292 |
| large4096_c | 5350.311625 | 631.999042 | 807.389250 |
| large4096_f | 10625.571417 | 658.837042 | 742.940333 |
| tiny2048_f | 1245.588042 | 117.535292 | 112.177750 |
| scaled2048_f | 1254.276875 | 488.019333 | 112.930750 |
| negative1024_f | 155.911000 | 77.354375 | 17.823167 |
| dense_replay_f | 292.439209 | 193.886959 | 58.323709 |

全部12类新旧/NI原1e-12对照、独立抽样协方差/Pearson、不变输入、结果独立/只读、对称性通过。新旧对角逐位相同且符号/零分类相同；其他元素仅声称原容差内等价。4096 F约16倍旧版提速、比NI少约11%耗时；4096 C比NI少约22%。scaled、negative、dense replay仍有实质NI差距。微小正差及约4–5ms的中型差异记录，不单独作为否决门禁。

## 内存与审计范围

12个 fresh 子进程；2048 C/4096 F × 旧/新/NJIT × heap/RSS。cold包含初始化与首次调用，warm保留上一份双输出直到本次结果返回；连续生命周期峰值，不相加独立阶段峰值。MiB=2²⁰ bytes。

| 场景 | 后端 | cold申请峰 MiB | warm申请峰 MiB | RSS请求增量 MiB |
| --- | --- | ---: | ---: | ---: |
| medium2048_c | old | 64.010210 | 128.010939 | 128.812500 |
| medium2048_c | new | 92.056316 | 156.057077 | 156.750000 |
| medium2048_c | njit | 102.000454 | 166.000668 | 172.140625 |
| large4096_f | old | 512.010210 | 768.010939 | 768.843750 |
| large4096_f | new | 278.103191 | 534.103952 | 558.953125 |
| large4096_f | njit | 390.000454 | 646.000790 | 652.281250 |

4096 F cold申请峰比旧版少约234MiB、比NI少约112MiB；RSS请求增量分别少约210/93MiB。原生审计的 D² arena 与最终根复制均为0；必要独立协方差复制128MiB计入algorithm_copy。2048 C相对旧C++申请峰增加约28MiB，是库/面板真实成本；结合约6倍提速采用，不能称零增长。相比NI申请峰少约10MiB。

初版完整C面板的申请峰比NI多约6MiB，随后复用1024行分块减少16MiB，完整计时基本不变。初版真实源码身份、全部计时/内存、10755项全回归及负证据保存在 [initial/review.json](initial/review.json) 和 initial/；两次数据不混合、不覆盖。

**分配追踪未完全闭环：** 四个native heap进程在cold峰阶段均无untracked free，后续cold保留阶段1次，warm基线4次、warm及最终保留5次；大F的NJIT仅在warm峰之后最终释放有1次。所有阶段无表溢出。重复观测不能证明无泄漏或定位原因；warm测量带此限制，不宣称完整生命周期分配审计通过。原始明细见 [validation.json](validation.json)。探针不覆盖所有Python arena/mmap/栈/进程树，RSS每后端场景一个fresh监控样本；两者不能互相替代。库工作区预留是已有估计，不是供应商严格上界。

## 验证与自审核

- Python全量 **10771通过**；专项 **448通过**。原生、ASan/UBSan、三库OFF原生均 **9/9**；OFF专项 **307通过/141预期跳过**。
- 185直接算子夹具scalar/auto、469案例/3056值历史Microsoft Excel重算快照复比通过；**没有新的Excel应用重算**。ASan/UBSan不覆盖供应商库，detect_leaks=0。
- 新增90项配对回归及原生observer检查，覆盖C/F/填充输入、C/F/auto输出、真实single/thread/process-inline/shared路径、精度证书失败、旧结果存活、错误恢复、预算、所有消费者排除及1024/1025/2048/2049分块边界。
- 根资格在finalize、工作区和实际执行共用；typed根不会进入仅支持scalar的DAG分支调度。复制前检查完成状态、几何、有限性、非重叠，错误时不破坏独立健康输出。
- 77项当前源码/编译参数与wheel、安装extension/worker身份一致，ON/OFF包检查和Ruff通过。最终OFF构建 `a53f4ad6fb1c031abb0fc9ccc5f65bd3aa8dd4aac3a26cf92c3a61a0c0fc11f5`。自审核详见 [review.json](review.json)，复现命令见 [commands.md](commands.md)。
- 文档影响：同步execution-guide、physical-layout-design和physical-execution-optimization；AGENTS及D8已满足用户小幅退化政策，不重复维护；算子/Excel数学契约未改，无需改公式或registry。CodeGraph先查但未定位新增私有函数，后以当前源码和全链测试核实。

剩余事项：特殊输入NI差距、完整分配追踪限制及全算子中大型收益仍开放；本机证据不是跨平台、上游业务迁移或部署验收。
