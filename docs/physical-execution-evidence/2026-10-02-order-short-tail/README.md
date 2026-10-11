# 单根高分位数短尾选择：局部采用，整体目标仍开放

2026-10-02；本机 macOS arm64、CPython3.12。只改既有标准库选择分支，不新增数学算法、第三方依赖、工作区或并行。见 [设计](DESIGN.md)、[精确生产差异](reduction.cpp.patch)、[回归差异](test_order_selection_library.py.patch)。新构建 `a6841b56ae9e25c011f991df43385f47c1b5fa7285d0ec804cdf7f71366d75a2`，基线 `ccca6b67a5220ea2134a4743d281b80c5af4c096e7d41c3d5b747bb8de70e331`；[77项源码及extension/worker身份](source-identity.json)只差 reduction.cpp。未提交或推送。

## 实现、自审核与边界

现有 count≥1024、无NaN、无混合±0守卫内，高q需要相邻排名时选lower，再扫短尾最小值补upper；其余保留原upper/max-prefix。中位数、原插值表达式、Infinity、NaN payload/zero tie回退、错误及只读所有权不变。两调用入口统一修改，共享多根仍完整排序；AST/DAG/Planner、IPC19、源N预算、高水位容量、消费copy审计不变。

独立静态审核未发现生产阻断；发现新增测试用重复键会掩盖邻排名错误，已改为固定随机唯一键。完整Python10395通过；最终唯一键专项8通过，关闭两库专项118通过；[原生](native.log)、[ASan/UBSan](asan.log)、[双库OFF原生](off-native.log)各9/9。macOS未运行不支持的LeakSanitizer。完整回归的汇总与最终测试强化分别保留，不宣称改fixture后又重跑整套。

185直接夹具 scalar/auto全部通过；[历史Excel快照复比](excel-snapshot-verification.json)469案例/3056值一致，公式/输入未变。没有新Microsoft Excel重算。初次verifier缺少输出argv、拒绝覆盖旧快照均有原日志，最终在新目录重跑通过；不是数值失败。正确性、身份、错误/所有权和资源预算仍硬约束。

## 完整请求时间

[32类、9轮均衡完整请求](timing/timing.json)使用实际新旧wheel与warmed NJIT，单CPU、独立结果同等保留；新旧完整结果逐位相同。普通4M q=.95 **41.357→36.095ms**，NJIT51.683ms；16M **171.263→151.682ms**，NJIT133.262ms，仍慢18.421ms/13.8%，该族目标未闭环。16M 55%mask **154.539→143.328ms**，NJIT173.158ms。低q、中位数、短输入、稀疏mask、sorted/reverse/equal、prepared控制均保留，不用平均掩盖负值。

[初次16M筛选](screen-16m/timing.json)独立保留171.085→152.127ms、NJIT129.591ms，不与后续配对数据混算。没有为几毫秒继续调参；仅采用现有std的少量代码减少大数组扫描，下一优先级仍是更大的重型差距。strict小差异不作独立采用否决。两轮worker均[正常结束并恢复参考线程](timing/timing-worker-cleanup.json)；native没有外施线程限制。

## 内存与采用结论

[4类、24个fresh子进程、72个heap阶段](memory/memory.json)计入setup、冷暖工作区和保留结果；追踪表overflow0。新旧cold/warm申请峰均**增加8B**，如实保留，依[当前D8](../../computation-design-rules.md#10-d8用对等的完整负载证明收益)不是独立门禁。未新增数学scratch；4M/16M申请峰仍约32.009/128.009MiB，对应NJIT68.001/272.001MiB。本轮相对NJIT的大额堆优势来自已有路径，不是本次新增内存节约。

fresh请求RSS新旧相等或−32/64/128KiB；每后端每类单样本，准备驻留、Python arena/mmap等盲区仍保留，不宣称所有内存状态领先。只对这处物理优化局部采用；全算子目标及重型分解、协方差等实质欠项保持开放。

## 复现

保留 [最终wrapper](benchmark.py)、[计时执行版本](benchmark-timing.py)、byte-identical共用tool与[配置/初次失败说明](provenance.json)。计时版本和最终版本仅差可选memory子集选择；不改变数值路径。更新安装位置为匹配身份的wheel后运行：

```sh
python benchmark.py --output /tmp/order-short-tail-timing --mode timing
CME_ORDER_MEMORY_CASES=1 python benchmark.py --output /tmp/order-short-tail-memory --mode memory --heap-probe /path/to/heap_probe.dylib
python verify_manifest.py
```

脚本中的本机路径及历史快照为本次运行定位信息；没有归档wheel、dylib或整套dirty源码，不把hash清单称为独立Git可重建发布证据。仅本地arm64，未跑新跨平台CI或平台业务集成。完整正负差值见 [summary.json](summary.json)。
