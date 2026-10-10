# Highway 顺序统计：2026-10-02 局部采用

本轮在 Apple arm64 接入静态 Highway 1.3.0 VQSelect，完整请求随机大样本明显提速，整体全算子目标仍未完成。构建 `c9f75aed3f4d38fd44caf893b9faba83619c1a094f006870ce6062f814a238ec`，旧版为 `5b31418d52e9804122fe901bb645f5291bf181a1a2fba17e77cb0fee0489a1c9`；源码、wheel、扩展与worker身份见[最终身份](final-identity.json)。只改5项原生构建输入；保留共享工作区原有修改，未提交推送。

## 设计和边界

[设计](DESIGN.md)记录依赖、数学、所有权及验证方案。[精确差异](implementation.patch)相对于本轮开始的工作区，不等同于整个Git差异；[源码快照](sources/cpp/operators/reduction.cpp)是已测实现。仅在自动ISA、消费量≥1024、无NaN/混合正负零且支持float64时用VQSelect；其他情况保留既有选择/排序。邻排名与插值算式不变，掩码只消费已选中值，input只读，独占原工作区，无私有线程池。显式scalar/ISA不被库自动派发覆盖。

共享多根排序缓存未改；无新算子、AST/DAG/Planner或IPC19协议变化。direct `isa=highway_vqselect`标记实际调用，图不新增ISA字段或库计数，未将消费数量冒充精确SIMD工作量。Excel公式、预算和数学契约保持原样。固定上游archive/校验和/许可证随源码分发，静态链接不依赖Homebrew；其他平台继续原实现，未外推跨平台收益。

## 完整请求时间

[原始9轮](timing/timing.json)、[脚本](benchmark.py)使用实际旧/新安装包、普通Scheduler、独立结果、1 CPU，seed=20261001；三种顺序各3轮，输入逐轮核对。NI先编译再预热，只有NI受库/TLS线程控制，native遵守自身Scheduler预算。16/16旧新及本轮NI结果逐位一致；[结束记录](timing/timing-worker-cleanup.json)确认NI TLS恢复。保留全部wall/CPU离群样本。

|计算|旧版 ms|新版 ms|NJIT ms|
|---|---:|---:|---:|
|16777216 quantile random q=0.95|157.790000|60.189541|137.264250|
|16777216 median random|143.837667|69.202209|136.624375|
|16777216 quantile random q=0.05|160.208458|56.499458|220.369291|
|4194304 quantile random q=0.95|37.056542|15.716000|55.884042|
|16777216 quantile_where random q=0.95|128.683959|87.392417|177.490542|
|63 quantile random q=0.95|0.006083|0.006250|0.001500|
|63 median random|0.006209|0.006084|0.000792|
|1023 quantile random q=0.95|0.054708|0.046875|0.011250|
|1023 median random|0.057458|0.055000|0.014042|
|1024 quantile random q=0.95|0.012000|0.014042|0.009458|
|1024 median random|0.018500|0.017708|0.012459|
|1025 quantile random q=0.95|0.024209|0.022292|0.015042|
|1025 median random|0.020417|0.017417|0.008375|
|4194304 quantile sorted q=0.95|6.092500|17.988042|18.942625|
|4194304 quantile reverse q=0.95|6.330458|19.028917|19.409042|
|4194304 quantile equal q=0.95|6.136292|5.520667|18.943625|

16M高分位数少97.600ms/61.9%，比NI快约2.28倍；中位数少74.635ms/51.9%，比NI快约1.97倍。**4M sorted/reverse分别增加11.90/12.70ms**，新版约18/19ms；不隐去这一回退，不宣称所有分布均显著领先。按用户当前口径，短请求的这些绝对变化不独立否决大样本随机路径收益；真实持续批量仍须按累计耗时和预算判断，不外推未经测试的批量吞吐。

## 内存

[12个fresh子进程](memory/memory.json)、[比对验证](memory/memory-validation.json)：2计算×3后端×heap/RSS。每heap包含setup、首次调用、5次warmup、3次warm测量；RSS包含setup及首次请求，未先运行数值oracle。36个heap阶段无追踪溢出/未匹配free。

- Quantile旧/新cold峰128.008862/128.008854MiB，warm128.009420/128.009413MiB；新旧均−8B，属于微差，不宣称有实际节约。NI约272.000805MiB，既有128MiB工作区优势被保留。
- Median旧/新cold峰128.007444/128.007437MiB；NI128.000477MiB，差约7KiB，不作为否决。
- Quantile新旧RSS增量+32KiB；median−48KiB。新版median RSS比NI多0.594MiB，完整内存并非所有口径严格领先。几KB/页级变化照实保留，不按单个正差否决。

heap不覆盖Python arena子分配、mmap或全部调用栈；RSS为单次fresh进程的采样峰，不能充当分配计数。原型动态库首次TLS另有+600B，不直接套用到静态引擎，[组件筛选](screening/RESULT.md)与正式证据分开。

## 回归与审核

- [Python全量](python-full.log)10428通过；[专项](focused.log)132通过，ON强制验证库路径，包含四个算子、极值/微小值/Inf、masked、prepared变化、真实线程/进程/共享内存及结果独立性。
- [原生](native-test.log)、[ASan/UBSan](asan-test.log)、[三库OFF原生](off-native-test.log)各9/9；原生新增四舍入模式、异常flags、输入/输出canary和逐位检查。macOS不支持LeakSanitizer，因此关闭该项，不声称泄漏扫描通过。
- [OFF专项](off-python.log)132通过不跳过；[OFF身份](identity-off.json)与独立wheel匹配。
- [185直接夹具](direct-fixtures.json) scalar/auto通过；[历史Excel快照](excel-snapshot-report.json)469案例/3056值一致，输入和公式未变，**没有新Microsoft Excel重算**。
- [链接依赖](link-dependencies.txt)和[导出符号](exported-symbols.txt)无Homebrew动态依赖或公开hwy符号；wheel及sdist许可证/固定archive由check_dist验证，源和工具语法、Ruff、差异空白检查通过。

自审核修正新增原生测试的非法q端点为相邻合法浮点值；生产数值口径未改。独立只读审核确认本轮调度预算、结果、内存、构建身份与实际路由，无阻断发现。协方差、Cholesky、LU及有界Jacobi等既有性能欠项没有关闭；本轮不宣称平台完整替换验收。
