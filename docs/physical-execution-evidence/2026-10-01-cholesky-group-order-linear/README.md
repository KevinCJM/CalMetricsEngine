# Cholesky 四行共享与顺序统计打包（2026-10-01）

两项物理优化已采用，**全算子中大型时间与内存领先 NJIT 的目标仍未完成**。修改未提交或推送；范围为本机 macOS arm64，非跨平台或平台业务验收。

构建：`2ebc5d22f827d240a99a0507b8153213b9fd9216290c22e0bec01f96285567a0` → `850871ebcff4f6f2e089ae96029f736ccc0c94282cd06824193f839a410906b3`。77 个源文件重算、wheel 与安装 extension/worker 均一致，生产差异仅两个文件，见 [identity](identity-final-confirmed.json)、[差异身份](source-changes.json)和[本轮补丁](production-changes.patch)。保留前一阶段共享工作区，不提交其他任务内容。

## 变更和边界

- Cholesky 保留 16 行/8 因子面板，在因子完成后用固定 Eigen 数组，让四个独立行共享源读取；共同三角前缀之外继续原尾部。每个元素仍按原顺序分别乘减，true division、sqrt、完整输入校验、错误重放及 C/F 成功清零不变。Workspace 上界不变，没有新增数值堆。实际 Release `update_group` 帧为 720 B，固定数组逻辑容量 320 B，两者不可混淆；反汇编含分别 `fmul/fsub`，没有 FMA。见 [栈记录](stack-frame.json)。
- 顺序统计只对非标量、逻辑 C 连续的源及可选掩码直接读指针；其余布局继续逻辑访问。先判断是否消费再读值，分类/选择/排序/插值不变；复用打包消费数填审计，省去重复 mask 扫描。**仍按源 N 预留工作区**，稀疏掩码内存欠项未解决；没有内容缓存或新增公开接口。
- AST/DAG、Planner、IPC19、所有权、准入平台和 Excel 数学/公式均沿原契约，不新增普通请求的导出计算。准备复用不缓存过期 mask，独立快照不受下次失败影响。详见[当前设计](DESIGN.md)、[代码自审核](FINAL-CODE-REVIEW.md)及[测量审核](FINAL-MEASUREMENT-REVIEW.md)。审核记录注明审核当时状态，最终运行结果以 [validation](validation.json) 为准。

## 完整请求时间

真实旧/新安装 wheel 和预热 NJIT，单 CPU，9 轮平衡顺序。普通入口计入检查、规划、执行与全部结果封装；prepared 返回独立 snapshot。新结果返回前保留上一结果，消费字段完整且输入不变。NJIT实际记录 Accelerate active=1、OpenBLAS threads=1；native audit 为一个 CPU token/线程，退出后恢复参考进程原设置。

下表单位 ms；全部 35 类原始轮次、CPU 时间、输入/输出身份见[计时数据](formal-timing/timing.json)。新旧 C++ 完整结果逐位一致，NJIT沿原容差及有限性/符号/零检查；Cholesky另检查完整三角、对角和81个确定系数重建，不宣称全矩阵重建残差。

| 完整负载 | 旧 C++ | 新 C++ | NJIT |
| --- | ---: | ---: | ---: |
| Cholesky 4096，C输入/F结果，普通 | 1825.688 | 1232.325 | 402.083 |
| Cholesky 4096，F输入/F结果，普通 | 1961.322 | 1352.265 | 477.177 |
| Cholesky 4096，C输入/F快照，prepared | 1860.791 | 1238.384 | 409.027 |
| 16M masked median，55%消费，普通 | 237.133 | 145.390 | 136.225 |
| 16M masked quantile，55%消费，普通 | 272.576 | 180.029 | 177.467 |
| 4M masked quantile，55%消费，普通 | 61.579 | 38.791 | 43.484 |
| 16M unmasked quantile，普通 | 201.530 | 174.181 | 137.113 |
| 4M masked quantile，1%消费，普通 | 29.996 | 11.748 | 5.654 |

4096 阶 Cholesky 减少约0.59–0.61秒、约31%–33%，但仍明显慢于该 NJIT；16M 非掩码分位数仍有约37ms差距。只采用相对旧 C++ 的明确收益，不据此宣称 NJIT 目标闭环。512 阶及63点请求保留独立结果；小规模几微秒/几毫秒差不单独否决，大批量累积效应须按实际负载评价。当前35类中最大新旧正差约0.25微秒，属于本轮观察，不外推其他负载。

本轮 NJIT Cholesky 为 `np.asfortranarray(np.linalg.cholesky(a))`，满足同一指定F输出；其布局物化计入时间和内存。这是该参考实现的证据，未另实现 NJIT 原地布局优化，不能外推为所有布局或最优实现的内存优势。

## 完整请求内存

13类、78个 fresh 子进程，分别量 malloc-family 申请峰及 RSS。持续 heap epoch 覆盖 setup、cold、首次结果释放、warmup、warm与常驻工作区；warm保留上次输出。记录各阶段峰值，**不可相加**。输入生成/掩码/哈希临时最多64KiB；13类内存不含 reverse 模式的整条复制。RSS为单 fresh 页级采样，保留 idle control；不是 malloc总量、process-tree峰或泄漏结论。详见[摘要](summary.json)、`memory/*/memory.json` 和原始日志。

- 新旧申请堆峰基本持平（逐项差16B），不将这类差额写成显著节约。4096 Cholesky新cold/warm约128.511/256.511MiB，比**本轮指定F的NJIT参考**各少约127.49MiB，主要源于前阶段直写结果，非本轮新增收益。
- F输入4096 Cholesky的RSS请求增量相较旧版多64KiB；C输入及prepared分别少96/32KiB。保留原始值，按D8不以这些页级小差独立否决。
- **实质欠项**：1%消费的4M masked median/quantile，新cold申请峰仍约32.01MiB，NJIT约0.64/1.00MiB，多约31.0–31.4MiB；源N工作区也产生显著RSS增量。并非几KB正差，后续应将实际消费数贯通已有工作区准备及预算，保留每次mask变化与失败校验，不缓存过期计数。
- 申请追踪六阶段均无table overflow。覆盖malloc族，不覆盖未跟踪mmap、Python arena内部或完整进程树；RSS基线、峰和增量分别报告，不以较低总RSS替代请求增量优势。

## 功能、构建和公式复核

| 验证 | 结果 |
| --- | --- |
| 完整Python | 10383通过，新增50项回归 |
| 原生 / ASan+UBSan / 双库OFF原生 | 各9/9；不含LeakSanitizer声明 |
| 双库OFF相关Python | 410通过 |
| 冻结直接算子夹具，scalar/auto | 185项全部通过 |
| 历史Microsoft Excel快照复比 | 469案例、3056数值通过；本轮无新Excel重算 |
| 完整请求时间 / 内存 | 35类正确性通过；13类78进程完成，不代表全部性能目标通过 |

首次新测试的绝对导入造成collection失败，已改为项目现有相对导入并复跑通过；原失败日志保留。计时脚本最初来源注释混淆wrapper和工具hash，随后只改注释，AST不变；[执行版本](benchmark-timing-executed.py)和[修正记录](benchmark-provenance-repair.json)保留原字节。

`screening/`、根目录 `screen*.json` 和 `timing.json` 是隔离ABI初筛，不是正式Scheduler验收。初筛58夹具/232调用全部逐位一致；其中NJIT计时只设置环境变量、没有正式线程查询，**不作为公平NJIT性能证据**。正式结论使用 `formal-timing/` 和 `memory/`。完整输出 `.npz` 只作当前case交换，未归档每项完整数组；每项摘要与全值验证保留在JSON。

## 复现与采用纪律

`benchmark.py`、`run_memory.py` 复用冻结共用工具；install路径和构建身份明确锁定本次两包。运行先准备相应源码身份的wheel，再按脚本参数提供实际安装目录。heap库及工具链指纹保留于validation/identity；二进制、完整输出数组和临时build树不存仓库。历史快照只读，不改缓存公式值。[文件清单](manifest.json)可用 `python verify_manifest.py` 复核。

按[AGENTS](../../../AGENTS.md)和[D8](../../computation-design-rules.md#10-d8用对等的完整负载证明收益)，严格 `time/heap/rss` 比较字段只是诊断，几KB/几ms增加不能自动作为否决。当前采用依赖完整旧C++显著收益及契约通过；大矩阵时间和稀疏mask内存的实质NJIT欠项保持开放。其他既有 LU、Jacobi、混合GEMM、协方差回退及潜变量/张量性能欠项没有在本轮复验或关闭。
