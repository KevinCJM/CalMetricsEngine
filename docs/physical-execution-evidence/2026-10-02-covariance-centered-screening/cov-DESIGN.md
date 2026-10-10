# 协方差首段中心化面板复用：隔离候选

本候选遵守 AGENTS 与计算纪律 D1/D2/D5/D6/D8。仅在 tmp 编译，未改生产，也未运行数值、测试或基准。编译成功不代表数值正确、收益成立、完整任务验收或平台接入通过。

## 已确认的源码边界

`covariance_accelerate.cpp:68–102` 已先计算所有 canonical means、再完整物化本段各列、最后检查中心化幅度。中型 `32<=N<=2048, 256<=D<=2048` 使用 N 行完整面板；长序列/大宽矩阵只使用 512 行，不能当作全量中心化输入。原拒绝分支先恢复 BLAS 线程，销毁面板，再通过 execute 重算 means 和每个产品的中心化减法。

候选只处理首次 `abs(centered)>8` 拒绝，且要求 F-compatible、首段覆盖 N、偶数 D。tiny、非有限/过幅原始输入、全零、非 nearest rounding、线程准入失败、DSYRK 后的密集回放、部分面板、C 布局、奇数 D 均保持原路径。没有放宽 DSYRK 适用域、误差阈值或 guard。

已有 F 内核用 Eigen 固定 Four/Two、2×2 单元组，首次产品作为种子，然后四位置步进；两对链合并、可选 two packet、horizontal sum、最后奇数尾值，最后除 N−1。候选仅从现成 centered 面板读取产品，保留该归约树、乘/加顺序、非 FMA 编译、逐元素对称 bit copy。拒绝时原输出还未写 DSYRK 或任何 covariance 单元。全部输入有限且受原幅度准入，不引入新的错误；线程恢复仍在复用消费者前。fenv 标志及非 nearest 路径仍待实测，不凭静态推理宣称全通过。

D 偶数限定用于避免 canonical 奇数边缘的 dynamic Eigen sum 受指针对齐影响。N 模 4 的所有尾部保留原固定 packet 树。

## 所有权、预算与唯一实现

面板/means/norm/total 唯一归属于 request Workspace。消费者开始后不 ensure、不扩容、不调用重入 execute，也不写借用输入；完成所有结果与对称 copy 后销毁面板。最大现有面板仍 `(N+3)*D` doubles；原 Planner reserve 不变。没有新增全矩阵 scratch、缓存、结果或私有线程。初始面板分配峰值没有消失，是否节省总时间/内存必须分别测量。

隔离候选复制固定 packet helper 只为早期 ABI 对照。若值得生产接入，应参数化唯一现有 helper 或让其接受显式 centered mode，不能把这份副本并列塞进生产。普通 C++ 请求不触发 Excel，算法契约未改；仍须后续数值与适用 Excel 状态复核。

## 构建/对照与准备的验证

ccca 三个协方差源以及 registry/header 六个相关来源逐文件匹配，见 cov-provenance.json。但原 wheel-build 已被 root 重建为 a684，故冻结的 cov-operators-baseline.a 是 **a684 archive 的 unchanged covariance component**，不是完整 ccca archive。两个 ABI 都链接同一冻结 archive，唯一差异为拒绝路径消费现成 centered 面板。原 ccca wheel 仍是后续完整请求的真实历史基线。

ABI 两端都执行 prepare、域检查、原库准入、分配、完整独立 C 结果写入；计时保留旧结果直到下一独立结果完成。代码调用实际 BLAS ThreadLimit，并验证 single 及恢复；原本没有库调用的路径只执行无 private pool 的 Eigen。NJIT 已预热并返回同样完整独立 C 结果。复用原 frozen screen 的 adversarial cases 与三个独立 worker 的轮转计量，原始负结果保留。这里只是组件筛选，不能替代普通 Scheduler、F 根布局、prepared、进程、完整所有权与 malloc/RSS 测量。

cov-native-check.cpp 延续 canonical execute oracle，加上 N 的四种 packet 余数、偶/奇 D、完整/部分面板边界及 reuse flag；复用路径要求 bit exact、输入/canary 不变、fenv flags 与 rounding 一致。cov-screen.py 直接比较 baseline/candidate ABI 位模式及保留结果，报告实际 threadpoolctl/Numba 信息和原生 admission/caller threading 状态。未运行任何这些检查。

可由 root 串行运行：

```sh
/private/tmp/calmetrics-cov-centered-fallback-next/cov-native-check
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 /private/tmp/calmetrics-cov-centered-fallback-next/cov-screen.py --mode check --baseline-library /private/tmp/calmetrics-cov-centered-fallback-next/cov-baseline.dylib --library /private/tmp/calmetrics-cov-centered-fallback-next/cov-candidate.dylib --install /private/tmp/calmetrics-mask-consumed-next/install --output /private/tmp/calmetrics-cov-centered-fallback-next/cov-correctness
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 /private/tmp/calmetrics-cov-centered-fallback-next/cov-screen.py --mode timing --baseline-library /private/tmp/calmetrics-cov-centered-fallback-next/cov-baseline.dylib --library /private/tmp/calmetrics-cov-centered-fallback-next/cov-candidate.dylib --install /private/tmp/calmetrics-mask-consumed-next/install --output /private/tmp/calmetrics-cov-centered-fallback-next/cov-timing
```

小型几 KB/ms 正差不单独否决；中大型 NJIT 实质欠项不能因一次候选改善就关闭。原 scaled covariance 与 NJIT 的历史 reference difference 必须保留，不能调整容差掩盖。
