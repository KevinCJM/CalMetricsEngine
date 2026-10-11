# 协方差完整中心化面板复用：已采用，整体 NJIT 目标未完成

2026-10-02，本机 macOS arm64。真实新构建 `5b31418d52e9804122fe901bb645f5291bf181a1a2fba17e77cb0fee0489a1c9` 对照 a684 wheel；77项来源中只修改三个 covariance C++ 文件。未提交或推送。

## 改动与自审核

复用首次幅度拒绝前已经生成的完整 F 中心化面板，沿唯一现有 Eigen 固定归约树计算。没有再实现一份数学内核；输入、原均值/乘积归约顺序、样本除数、尾部和镜像保持逐位一致。先恢复 BLAS 调用线程，再消费 Workspace，最后释放。无新增矩阵、缓存、私有线程或 IPC 字段。普通、prepared、C/F 根与实际进程传输均有回归。完整设计见 [DESIGN](DESIGN.md)，本轮精确差异见 [implementation.patch](implementation.patch)。

独立自审核两次均未发现剩余可操作问题；清理两个失效 Map aliases，并补了原生 canary/面板峰/释放检查。`count=N` 已证明面板覆盖全输入，也测试 N=35/D=2050 的 block>N 几何。公开数学、AST/DAG、Planner 上界、Excel 公式均不变；执行指南已同步复用条件。

## 完整请求耗时

单 CPU，真实旧/新 wheel、预热 NJIT `np.cov`，同输入、同指定结果布局、独立结果和保留寿命。12类负载、6轮轮换，所有新旧矩阵逐位相同。以下为中位数，单位 ms：

| 输入/输出与入口 | 旧 C++ | 新 C++ | NJIT |
| --- | ---: | ---: | ---: |
| 2048² F/scaled，C输出，普通 | 630.742 | 475.797 | 84.397 |
| 2048² F/scaled，F输出，普通 | 620.749 | 471.506 | 108.551 |
| 2048² F/scaled，F输出，prepared | 634.158 | 475.302 | 112.573 |
| 2048² C/scaled，C输出，普通 | 603.384 | 601.134 | 93.770 |
| 2048² F/normal，F输出，普通 | 86.306 | 84.615 | 110.372 |
| 32769×256 F/correlated，F输出，普通 | 189.073 | 189.251 | 54.912 |

已采用路径少约149–159ms/24%–25%。小63×8新旧正差约0.0003–0.0006ms，不独立否决；C/scaled与长相关矩阵没有解决，scaled F仍实质慢于NJIT。不能用正常域领先替代全域结论。原始轮次和 CPU 时间保存在 [timing.json](timing/timing.json)。

五个 scaled 用例保留旧 C++ 与 NJIT 的严格1e-12差异：513×257 F有36格；2048² F的三个入口各2644格，最大绝对差约4.54e-9；2048² C有972格，最大约6.98e-10。这些差异新旧C++相同，报告 `njit_strict_pass=false` 和 `correctness=false`，未放宽容差。原生逐位一致、Excel/契约回归与NJIT严格一致是不同结论。

## 内存

3类负载、18个fresh子进程，54个heap阶段无追踪溢出或未追踪释放。计入图准备、面板、内部申请和仍存活的旧结果；输入生成只用一个矩阵和O(D)临时行，内存前不做数值预热。

- 2048² F普通/prepared与C普通：新旧cold/warm申请峰均仅增加 **8B**，无新增数值工作区，接受。
- 新cold/warm申请峰约 **64.054/96.055MiB**，同输出布局NJIT约 **102.000/134.001MiB**；约37.95MiB优势来自此前实现，不是本轮新增节约。
- fresh RSS新旧增量：F普通−96KiB、F prepared−128KiB、C普通+96KiB。均保留原数值，页级差异不独立否决。每类/后端一个RSS新进程观察，不能外推所有驻留状态或并发规模。

详细数据见 [summary.json](summary.json) 与 [memory.json](memory/memory.json)。Planner估算、申请峰和RSS不是同一指标；不能称整个请求零分配。

## 验证与复现

- Python全量 **10414通过**，协方差专项 **358通过**。
- 原生ON、双库OFF、ASan/UBSan各 **9/9**；OFF专项 **226通过/132跳过**。macOS不支持LeakSanitizer，未声称已检查泄漏。
- 185直接算子夹具分别覆盖scalar/auto；历史Microsoft Excel重算快照469案例/3056值通过，**没有新桌面重算**。
- [identity.json](identity.json) 与 [identity-off.json](identity-off.json) 校验全部77构建来源、wheel和extension/worker；当前源码匹配ON身份。两个公共基准工具逐字节沿用前一阶段，hash在 provenance 中关联。
- 首次基准传了小写output_order，在计时前退出；只修正基准包装器为C/F并重跑，原脚本与失败日志保存在 `initial-layout-argument-failure/`。全部正式计时/内存子进程正常退出，NJIT线程策略恢复。

本地完整 wheel/install 在 `/private/tmp/calmetrics-cov-centered-integration-next`；旧 install 在 `/private/tmp/calmetrics-order-short-tail-next/install`。基准源码、原始JSON/日志、精确patch及变更源快照已冻结；大 `.npz`、wheel和完整工作簿留在临时目录，输入可按脚本种子重建。源码快照仅供证据审查，不参与生产构建。

```sh
python3 benchmark.py --mode timing --output /private/tmp/cov-centered-timing
CME_COV_MEMORY_CASES=1 python3 benchmark.py --mode memory --heap-probe /private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib --output /private/tmp/cov-centered-memory
```

实际执行使用 `/Users/chenjunming/Desktop/myenv_312/bin/python3.12`，安装路径须与identity一致。若从冻结目录重跑，先把同一已验wheel安装到该目录的 `install`，并保留所引用旧identity/install；不要把缺失依赖当成已有验收失败。整体目标继续开放，下一步仍需处理协方差剩余域、Cholesky/LU和其他重型差距。
