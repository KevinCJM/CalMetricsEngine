# V47：私有组合投影共用分位点选择（2026-10-06）

当前实际项目已接入并复测这一物理优化。只修改 `cpp/operators/matmul_statistics.cpp`，新增 58 项回归；数学、类型、错误、结果所有权、AST/DAG/Planner/进程和 Excel 协议不变，没有新增算子或依赖。相关原有工作区文件保持不变，未提交、推送、全局安装或部署。

## 做法与复用依据

原有有界 DGEMM 投影每条时序先按 canonical 顺序算均值和二次偏差，再完整排序以取分位数。所有有效分位数消费者（包括重复列与不同输入节点）请求同一数值概率时，改用既有 `partition_quantile_line`，复用 Highway 1.3.0 VQSelect/标准库 nth_element 及相邻排名补齐。各输出沿用原凸组合插值。不同概率、特殊数值、较短样本与不适用条件仍按原 helper 处理，统计不读取重排后的时序，prepared 每次重读实际概率。

官方依据：[VQSelect 接口契约](https://raw.githubusercontent.com/google/highway/master/hwy/contrib/sort/vqsort.h)、[已有 1.3.0 发布](https://github.com/google/highway/releases/tag/1.3.0)。master 页面用于方法查阅，不表示升级依赖；实际使用固定构建缓存与源码身份。运行时独立 helper 证明见 [记录](v47-selection-runtime-proof.json)，图审计没有新增 VQSelect 次数或 ISA 字段。

## 质量与完整请求

一次同源 Python 全量 **15,277** 项，无失败/错误/跳过；普通、全引擎 ASan/UBSan、关闭 Eigen/Accelerate/Highway 的相关测试各 **643** 项，native 各 **9/9**。实际项目新 wheel 再验相关 643 和 native 9，不能称为第二次全量。SAN 不检查泄漏，跨平台与 Microsoft Excel 桌面重算未执行。

计时与峰值分开、顺序运行；固定输入 N=131,072、T=2,520、K=30。完整请求计入独立只读输入准备、普通 Scheduler、全部组合投影、年化均值/样本标准差/插值分位数，以及保留两代独立只读结果。NJIT 预热完成，固定单核；核验实际 SciPy capsule/Accelerate NEWLAPACK 并按调用线程设置/恢复库预算。参考使用有界 64/512 行 DGEMM 及排名选择，没有全量 N×T 矩阵，也不以较慢的完整排序 NJIT 当基线。

[候选审核和原始对照](v47-prescreen-review.json)包含六轮配对的 48 次计时及 32 个 fresh 峰值进程；[实际接入复测](v47-adoption.json)包含三轮 24 次配对及当前 C++ 的 8 个 fresh 峰值进程。逐位置数据、原始日志、XML、各 build identity 与生产 audit 一并保留。初次 driver 的错误区间夹具在 [失败记录](initial-time-attempt/failure-review.json)保留：零有效样本、全部终结；修复 driver 后新目录完整复测，不混入有效对照。

具体中位数和峰值见 [最终审核](v47-review.json)。实际收益是秒级耗时下降；相对原 C++ 峰值基本持平，不能宣称本轮有大额内存节约。malloc 不含部分 Python arena/zone、mmap 和栈，RSS 采样可能漏掉短峰值；没有完整 process-tree/泄漏审计。

## 范围与剩余项

本请求使用调用方已给定的候选权重，只证明现有投影及三统计组合计算。没有证明 RNG、分组修复、多轮探索、有效前沿或平台服务已迁移，也没有新增这些能力。多个不同分位点继续原排序；其他大矩阵的重要 NJIT 缺口、主要完整链和大额峰值目标仍开放，整体 goal 未完成。

实验与实际安装路径仍位于 `/private/tmp/calmetrics-quantile-selection-20261006`，所有脚本以 `.py.txt` 保存为惰性证据，便于区分惰性证据与唯一当前实现。源码唯一当前实现位于项目 `cpp/`；临时 wheel 未替代全局环境。重跑前先在独立临时目录恢复脚本并按 identity 核对源码、测试、编译器/依赖及二进制，保持测试/构建与性能测量不并行。旧基线的隔离源码路径由记录给出，不能直接拿当前源码冒充旧 build。
