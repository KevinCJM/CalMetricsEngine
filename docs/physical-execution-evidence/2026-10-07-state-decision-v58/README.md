# V58D 私有状态链物理分块验收（2026-10-07）

已进入当前 CalMetricsEngine 源码并重建复验；整体性能目标仍未完成。未提交、推送或部署。

## 做了什么

三份引擎文件贯通 finalize、Planner、arena 和执行：同源、输入 valid、标量分支的私有多级 state_select 按4096点计算，复用 canonical 比较/状态内核，直接写独立 int64 根。保持原 AST/DAG、逻辑节点、数学/错误/所有权及 Excel 展开；观察、隔离、作用域、未知几何或生命周期无证明时走原路线。另改既有原生测试、加两份测试，新增126项 Python 案例和10种原生故障。

本轮查阅 [MLIR loop fusion](https://mlir.llvm.org/docs/Passes/#-affine-loop-fusion) 和 [elementwise fusion](https://mlir.llvm.org/docs/Passes/#-linalg-fuse-elementwise-ops)，用于缩小私有中间缓冲的设计参考；没有引入第二个编译器或新的数学实现，也不凭文献宣称本机收益。

## 质量与失败边界

最终 D 同源正常全量16226项、ASan/UBSan及库关闭相关各1488项，三种构建原生各9/9通过，均无失败/错误/skip。当前实际目录另通过相关1488项和原生9/9，这是实际入口复验，不是第二次全量或SAN。当前 source/wheel/worker身份 `6a4114d3736f7967e011c5e7619172eb093685461cbacd3cbda74191de02aadb` 与最终资格候选一致。

C虽曾通过全量/SAN/OFF，但自审核被否决：geometry-only准备误把真实空指针当缺失内部描述符，双故障原路径NULL_INPUT而候选INVALID_STATE_CODE。D恢复比较的完整准备，选择阶段只豁免经过证明的内部条件/前态，其余真实父值先保留空指针检查；十种原生故障复核首错与输出前缀。早期编译、测试夹具/协议、沙箱共享内存和C错误探针失败全部保留。没有放宽原测试、数学容差或小任务门槛。

## 完整任务对照

1398104行×8维、K16、训练1048578行、4迭代的标准化/拟合/推断/排序/回填/状态压力链；CPU及原生库1核，相同独立只读输入准备、13项输出及两代保留，NI已预热，编译/图定义不计时。三组交替fresh进程，各三次计时；时间、malloc和RSS分别测，期间不并行其他测试/构建。

|完整请求|V57|V58D|暖NJIT|
|---|---:|---:|---:|
|时间中位（秒）|2.0941|2.0999|7.4885|
|暖态malloc峰值（bytes）|868,304,738|834,750,250|851,456,928|

申请峰值较V57少33,554,488bytes（约32MiB），该几何较NI少16,706,678bytes；不把这个局部约2%优势说成所有负载的大额领先。毫秒差只作诊断，采用依据是32MiB完整请求峰值减少且没有明显耗时退化。RSS绝对及增量单独保存于数据中；不能用Planner/arena数字替代峰值。

当前目录重建后另三轮fresh配对中位C++/NI为2.1240/7.3807秒，独立fresh malloc重现834,750,250/851,456,928bytes。所有13项C++输出对既有完整C++参考逐字节一致，NI离散结果精确、浮点在原4.9e-5预算内；校验在测量后进行。源、wheel、worker身份及测试SHA保留在build/raw记录。

mask arena由22369664bytes缩至8bytes；numeric arena由139813216至128628384bytes，包含整数arena。额外tile4096bytes。input/root/state复制审计仍需按字段范围解释；algorithm_copy_bytes仍报告584422064bytes，不是全算法零复制。

## 还差什么

重要大矩阵相对NJIT的退化、其他平台完整链、HMM整链收益和全面分配/复制审计继续开放，不能宣称整体完成。此次没有新算子、公开API/wire或数学契约变化，相关operator、typed、Excel及AGENTS无需对应重写；同步执行指南、潜变量说明与物理优化记录。CodeGraph和文档链接的收尾核对另见最终maintenance记录。

## 可复核证据

- [设计及D修正](raw/design.md.txt)，[C否决](raw/self-review-c.json)，[D自审核](raw/self-review-d.json)。
- [正常全量](raw/v58d-quality-normal.json)，[ASan/UBSan](raw/v58d-quality-asan.json)，[库关闭](raw/v58d-quality-off.json)，[实际复验](raw/v58d-actual-quality.json)。
- [候选15个测量进程](raw/screen-d.json)，[收益判断](raw/d-benefit-review.json)，[实际10个确认进程](raw/v58d-actual-confirmation.json)。
- [实际应用范围](raw/v58d-adoption.json)，[完整文件SHA](manifest.json)。全部255份原始证据/脚本/源码快照保持不可执行文本，未归档巨大NPZ或编译缓存。

收尾核对：234个本地Markdown链接通过，324份范围外受保护文件不变；93份实际source/config、测试SHA、最终完整链逐项校验及malloc无表溢出均通过。CodeGraph已同步并复读实际新符号；[维护核对](raw/maintenance.json)、[同步](raw/codegraph-sync.log)、[状态](raw/codegraph-status.txt)、[当前符号](raw/codegraph-current-symbols.txt)。
