# HMM 完整有限输入链路：保序交错与验收

2026-10-03，macOS arm64，CPython 3.12，CPU预算1。仅本地修改，未提交或推送。

## 设计与依赖

生产变更仅 `cpp/operators/latent.cpp`：统一的时间块helper保留每行source→target的原logaddexp树、每个xi单元的时间升序累加及 `std::exp/std::log1p`。alpha/beta/em/log_a只读，xi完整产生后才消费；前后向递推、M-step、floor、停止和错误位置不变。非scalar、N≥128、K≥3使用128行块，其余用同一helper的Block=1。Planner和prepare共用 `latent_scratch`，取得指针前预留最多128 doubles；不在拟合中resize，不引入全量矩阵、输入pack、数值缓存或线程。

成熟vForce exp/log1p候选仅做隔离筛选，因浮点尾差的严格停止/离散风险未采用；保留原标准库函数的交错控制约快3倍，但正式采用依据为下方完整链路。逻辑图、算子数185、AST/Typed IR、IPC及Excel公式不变。详见 [XI设计](XI-DESIGN.md)、[完整链路设计](DESIGN.md)和[静态审核](review.md)。

原生DAG保留训练标准化、显式Park–Miller初始化、GMM预训练/共享拟合、样本外过滤、原winner后的组件排序映射，以及状态/score/置信度/概率回填。两侧返回相同13个独立输出，含9个额外数值审计值；每次重读输入并重新拟合。参考只依赖引擎自包含的 `tests.latent_reference`，未导入FIRP生产源树。

## 完整时间证据

旧/新/NJIT三个驻留进程，每次只执行一个请求，6轮轮换顺序。JIT和初始全量校验在计时外；计时包含保留旧结果时创建独立新结果及释放旧结果。ordinary包括绑定/校验/规划/封装；prepared使用 `run_snapshot`。图编译准备单列于内存报告，不声称成本消失。输入为固定生成的有限矩阵，75%训练、25%分类，K≤12、state_count=K；不是市场数据或真实服务验收。

| 模型/形状/入口 | 旧C++ ms | 新C++ ms | NJIT ms | 新/NI |
| --- | ---: | ---: | ---: | ---: |
| HMM 20000×4，K12，ordinary | 6749.667 | 3771.521 | 7335.106 | 0.5142 |
| HMM 同形，prepared | 6727.716 | 3767.966 | 7332.279 | 0.5139 |
| HMM 2048×4，K12 | 237.646 | 134.903 | 263.509 | 0.5119 |
| HMM 20000×1，K3 | 125.215 | 92.012 | 149.703 | 0.6146 |
| HMM 120×1，K3，3轮 | 0.2044 | 0.1934 | 0.1922 | 1.0062 |
| GMM 20000×4，K12控制 | 576.113 | 576.196 | 1331.881 | 0.4326 |

大HMM实际93轮、预训练30轮，上限200。新/NI配对bootstrap95%区间 ordinary `[0.5128,0.5190]`、prepared `[0.5116,0.5156]`；其余原始样本、迭代数、审计和区间在各JSON及 [summary.json](summary.json)。这是固定主机/构建/样本证据，不是普遍倍数保证。小控制多约1.2微秒只诊断；未改历史阈值。

六类新旧13输出的dtype/shape/所有字节一致，独立NI浮点按原1e-9、状态/顺序/迭代等精确比较；概率和未分类行另验不变量。每后端NPY快照hash保留于JSON。实际请求一处Python→C++、0 Python数值/worker回调、0 fallback、0请求期编译；118个逻辑节点结构共享，单CPU lane，无额外进程/共享内存。

## 内存与采用边界

复用既有heap/RSS工具，每后端每指标各fresh子进程，总6个；相同13个独立输出，共3042512字节。公共输入构造及相同tiny NJIT预热在双方epoch外；heap包含图构建、首次执行、5次预热及2次热调用。RSS在独立子进程测首次执行及结果替换；基线含已经构造的图/owner，idle delta均0，不相加不同阶段峰值。

| 观察 | 旧C++ | 新C++ | NJIT |
| --- | ---: | ---: | ---: |
| 冷请求申请峰 MiB | 11.913 | 11.914 | 6.723 |
| 热请求申请峰 MiB | 14.816 | 14.817 | 9.625 |
| RSS请求增量 MiB | 16.172 | 16.328 | 38.609 |
| 两次热调用累计申请 MB | 6.095 | 6.095 | 1274.758 |

本次新增申请峰1016字节、RSS多160KiB，只诊断。C++既有迭代缓冲复用显著减少累计申请，但堆峰比NI高约5.19MiB，不能把累计量写成峰值节约，不能宣布时间/内存全面领先。一次RSS观测不足以证明普遍优势。主收益为旧版约44%秒级提速，同NI约1.95倍，在D8下阶段采用。

图有约4.68MB numeric arena、0.24MB mask arena、4.32MB operator workspace；必要标准化/索引/模型初始化复制约1.003MB，root_copy2032字节、直接结果写入3040480字节，input_copy0。这些是原生审计范围，不等于无分配或完整OS内存统计。

heap仅覆盖所用probe拦截的malloc家族；Python arena内部、mapping、allocator zone及epoch前对象有盲区。native cold累计12次unknown free、删除首结果后13，热阶段无新增；不能据此判非法释放，也不能证明遗漏字节很小。NI最终释放有3次unknown free，表overflow全为0；[探针静态审核](heap-review.md)明确计数含义及未建立本轮probe源码/二进制绑定的限制。probe/helper SHA保留，未修改追踪工具或据此设置KB级门禁。

## 自审核、自测与身份

- Python11179通过；ON/OFF模型专项各390，新增50项覆盖127/128/129和255/256/257/258、K门槛、C/F/负stride/padded、4舍入模式、停止差值nextafter、极端有限/非有限、错误码/失败out、恢复及独立结果。
- 原生、ASan/UBSan、三库OFF原生各9/9；native内部尾部canary保持vector元素合法存活，覆盖N128/129/257/258、K3/12、scalar/auto，输出哨兵/逐位/只读输入/指针稳定。
- 185直接夹具、ON/OFF wheel分发检查通过。历史Microsoft Excel469例/3056值对照通过，公式/输入仅身份单元格变化；**没有新的Excel桌面重算**。
- 77项当前源码/编译配置、wheel/extension/worker及实际audit身份一致：`2abcc292fd55b1d41af8aad343e083b848a6e7ee182208ce953e7046cf55389f`。ON wheel SHA `273204b312d72ada3f490b1c847466d7332021bc8c0e619f142a098e7376c51f`；OFF身份另列。

旧包295a464按历史manifest与实际binary核对；它相对当前源码仅latent.cpp不同，memory-old如实报告current-source匹配false。不将旧包冒充当前源码。测试之外没有更改math/default、operator目录/用户接口，用户指南与算子表无需机械修改；执行指南、模型说明、路线图和物理优化记录已同步。

首次pair驱动的0D ndarray字节view错误和沙箱shm_open限制失败均保留；修复脚本后、具共享内存权限的390项全部通过。此前与隔离screen重叠的时间不作为采用证据。本轮六类重新串行配对，无数值/编译/heap/RSS任务并行运行。独立 `bench.main` 保留初始结果的偏差已审查，该入口未用于正式证据；正式pair与memory只调用其数学/setup函数。

## 复现与剩余工作

此目录冻结脚本/JSON/log SHA；原安装与轮子在 `/private/tmp/calmetrics-hmm-xi-integration`，旧包在 `/private/tmp/calmetrics-solve-pair-integration`。脚本为本机历史证据，含显式stage路径；跨机器需重建对应manifest的AOT包、配置路径，并重新测量，不能直接沿用数值。

```bash
python /private/tmp/calmetrics-hmm-xi-integration/pair.py --label fresh-hmm-run
CME_INSTALL=/private/tmp/calmetrics-hmm-xi-integration/install python /private/tmp/calmetrics-platform-latent-closure/memory.py --backend ordinary --model hmm --identity-manifest /private/tmp/calmetrics-hmm-xi-integration/identity.json --output fresh-memory.json
```

先完成数值时间进程，再运行memory或构建；日志/输出不覆盖。平台动态finite行、knowledge-time、quantile长排序同值、retrospective、Markov、state_count>K、其他模型、跨平台CI及真实服务接入尚未由本轮验收。整体性能与完整分配审计未结项；M6固定带资产投影/有界融合及M3/M4求解前沿继续开发，目标不缩小到这条已通过的链路。
