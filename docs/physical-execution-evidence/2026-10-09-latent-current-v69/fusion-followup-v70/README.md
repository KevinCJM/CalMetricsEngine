# V70：HMM 私有融合候选与 arena 负实验

2026-10-09 本地完成筛选，实际源码/native/wheel/worker仍为 `a87f2c25`，候选为 `658a558a`。本候选未采用，不能将结果写为实际引擎已获得本轮收益。

候选只改 `cpp/graph.cpp`：按真实 HMM 几何允许现有私有算术融合，保留大 HMM history 准入及 take/scatter 限制，不改数学内核、输入、图定义或13个输出。原生9/9、7个相关Python文件701/701通过，无skip；首次沙箱共享内存2项失败保留，允许共享内存后相同命令通过。

候选/当前/预热NJIT五组各time/heap，共30个fresh完整进程；另2个smoke。相同六份owned readonly准备、CPU1/库线程1、普通完整链入口、13个独立输出和两代保留；编译、静态fixture/定义及事后核验排除。全部输出、NaN位置、离散/实际迭代、数学不变量、输入和旧结果hash、非别名与配对input hash通过。

| 完整链 | 当前/候选/NJIT秒 | 候选申请峰较当前 | 候选申请峰较NJIT |
| --- | --- | --- | --- |
| HMM 500k×4、12状态、最多4轮 | 4.1427 / 4.1584 / 8.9147 | 相同 | 少6.62MiB |
| HMM 250k×4、3状态、最多60轮 | 0.8229 / 0.8223 / 1.7566 | 少5.72MiB | 多9.10MiB |
| GMM 1,398,104×8、16状态、最多4轮 | 2.0649 / 2.0630 / 7.2957 | 相同 | 少15.93MiB |
| GMM 500k×4、3状态、最多60轮 | 0.8222 / 0.8238 / 1.8968 | 相同 | 多20.06MiB |
| HMM 750k×4、3状态、最多60轮 | 2.4704 / 2.4656 / 5.2638 | 少17.17MiB | 多27.22MiB |

这是单组fresh时间（各进程3次请求中位）和申请峰，尚无候选RSS或重复组，不宣称新的稳定提速或全面通过。60是最大轮数；训练75%、seed17为明确可配置大样本负载，平台实时训练60行/quantile默认、HTTP/PIT/data链、F/批次/多核未覆盖。当前较NJIT优势包含历史优化。

另4个实际包完整进程把固定有限fixture的训练标准化改为借用全量标准化结果，全部13输出核验通过。GMM16/GMM3申请峰反而增加44,738,432/2,999,168bytes，几乎全部是numeric arena增长；workspace、直接结果和root copy不变。该控制图不采用，也未证明一般故障隔离/首错等价。

源码确认编译器借用视图本身没有数组slot，backing生命周期会延长；shape无关first-fit染色后，各slot按最大真实容量预留。共享/删除数组未必省峰值。逐节点trace和带别名约束的尺寸规划尚未实施，不能把全部增长归到某个具体节点。

采用前须预计算HMM节点，避免每个私有匹配重复整图扫描的O(PN)风险；补门槛/未知shape/多HMM/错误/传输专项、全量/SAN/OFF、重复time/heap/RSS，并重建实际包验证。随后联合arena与history save/restore成本处理剩余缺口。没有公开数学、算子或Excel recipe/budget变化，没有本轮Microsoft Excel重算、平台迁移或Git提交推送。完整QP/前沿/随机探索与其他主负载仍开放，整体目标未完成。

[候选设计](design.md)、[自审核](self-review.md)、[后续尺寸与别名设计](arena-followup-design.md)、[完整对照](comparison.json)、[负控制图](shared-comparison.json)、[闭合复核](closeout-validation.json)、[原始筛选](screen.json)。
