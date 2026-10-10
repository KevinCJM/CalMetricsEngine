# 大矩阵原生库优化证据

**历史候选记录（`a5669c9c…`）**：下文的“当前”指本历史轮次，不指最新源码。Accelerate GEMV及其构建开关已在后续保序Eigen实现中移除，旧命令只适用于对应历史构建。本轮ordinary/prepared计至返回原生Result，逐字段NumPy懒包装在计时/堆区间外；“typed结果封装”不能解读为完整公开输出物化。后续已修正口径并重测三个后端，见[续轮证据](../2026-09-29-ordered-libraries/README.md)。本目录保留原数据与当时的复现说明。

本机 macOS 15.6.1 / arm64、AppleClang 17、CPython 3.12.11、NumPy 1.26.4、Numba 0.60.0。生产接入系统 Accelerate GEMV，Eigen 3.4.0 仅提供无分配误差界计算；其他平台/禁用构建保留原生路径。设计和完整限制见[物理执行优化](../../physical-execution-optimization.md)。本目录不是发布、跨平台或全部性能验收。

- 基线构建：`e55cb4a1659f30b0061b2d3ee1644af81f58df66d90960973aeb6c155df5eba3`。
- 当前构建：`a5669c9ce006ac9be9919e46088d67d9cd875122c6dc24a34bf199dc63c7a46d`。
- wheel SHA-256：`47ebc91d13f740bcba20196e42dd8a033c14fcd6cef3c09f93090cd031072dd7`。
- Python 6,661 项，专项 111 项；原生、ASan/UBSan、Eigen/Accelerate-OFF 各 9/9；185 个算子回放通过。macOS 未开启 LeakSanitizer。
- 29 组旧/新/NJIT数值对照通过，18 组 matvec 相对旧版均有显著时间改善（1.49–5.25 倍）；全部 29 组仍慢于 NJIT。Cholesky 未采用库候选。
- 固定额外活跃堆 8–24 字节，RSS 不稳定；**严格“内存不增”和中大型领先 NJIT 尚未通过**。原四组严格性能门禁 3/4，63 点微任务增加约 0.166 μs，阈值未改。

## 证据和口径

| 文件 | 用途及限制 |
| --- | --- |
| `source-identity.json`、`packaging-binary-parity.json` | 从源码、工具链和开关重新生成身份；最终文档打包后的扩展/worker与实测安装包逐字节一致 |
| `timing.json`、`timing-summary.json` | 29 组、三个独立后端、11 轮轮换；普通入口包括 typed 结果封装，prepared 用独立快照。bootstrap按配对轮聚类，不把每轮5次当成独立轮次 |
| `memory.json` | 29×3×2=174 个独立子进程；连续堆和RSS分开，输入预先存在，包含图/工作区建立、首次库调用及保留旧结果后返回新结果 |
| `timing-worker-cleanup.json` | NJIT限制当前线程库策略并恢复的记录；生产路径自身负责线程预算，不能用外部设置替它证明 |
| `dispatch-final.json`、`dispatch_final.py` | 18组最终auto/改进后scalar比较，验证实际库分发有收益；不代替普通DAG或NJIT比较 |
| `final-status.json`及日志 | Python、回放、Excel快照、性能门禁的真实退出码；最初比较脚本失败及修复后的完整成功均保留 |
| `native-checks.json`和对应日志 | 原生、sanitizer、两库关闭构建；没有跨平台或泄漏检查通过声明 |
| `all-operators.json` | 185个算子scalar/auto回放，不是全算子NJIT性能验收 |
| `excel-ms-snapshot-report.json` | 469组、3,056个结果对照已实际重算的Microsoft Excel 16.89.1快照，公式/输入未变，newly_recalculated=false |
| `phase2/` | 原4组严格门禁；63点失败保留。比值是配对比值中位数，不等于独立中位数相除 |
| `probe*`、`chol*`、`matvec-probe*`、`tiled-memory.json` | 筛选用AOT原型及样本，**不是最终生产结果**；矩阵乘法库的额外堆、Cholesky无块候选的数值问题均未掩盖 |
| `contract-screen.json`、`contract_screen.py` | Cholesky固定容差失败与未保护GEMV的强消去反例；生产Cholesky未变，GEMV已加误差保护 |
| `start-manifest.json`、`task-scope.json` | 本轮开始哈希与实际差异，保护已有未提交修改 |

计时无malloc注入，与本任务构建/回归串行。相同输入、CPU=1、独立结果；NJIT使用真实NumPy矩阵函数并预编译精确签名。内存阶段只提前编译、不执行数值函数，首次数值库调用在epoch内。普通入口的矩形矩阵轴是time×asset，向量是asset。进程输入可能转成C布局，专项测试另验证worker内部transpose的F视图实际调用库并回传非零计数。

malloc探针不覆盖已有CPython arena、全部mmap或进程外资源；RSS每组合仅一次独立child，不能据此宣称统计稳定。堆观测包括审计对象和库线程元数据，新增8–24字节也明确保留。直接接口、ordinary、prepared的绝对内存不可混用。当前所有公开数学和Excel公式不变；不关闭历史潜变量/M1/其他矩阵性能欠项。

## 复现

路径按机器调整；基线与当前包版本同为0.3.0，必须核对完整build identity。当前源码及仓库固定Eigen归档支持离线构建，Accelerate来自系统SDK。`CALMETRICS_ENGINE_USE_ACCELERATE=OFF`禁用新路径；本轮原生回退测试同时关闭Eigen和Accelerate。

```bash
PYTHONPATH=/path/to/new-install CME_EXPECT_EIGEN=1 CME_EXPECT_ACCELERATE=1 python -m pytest tests tools -q
python tools/check_large_matrix_performance.py \
  --old-install /path/to/baseline-install --new-install /path/to/new-install \
  --output /tmp/large-matrix-comparison --mode all \
  --heap-probe /path/to/heap_probe.dylib
```

工具自动在独立内存子进程启用探针，父进程不可注入malloc探针。`--case-filter matvec_32768`只运行新增矩形组。脚本固定随机种子并记录输入SHA；原始样本不依赖网络。`final_checks.py`等保存实际本机运行目录，Excel快照验收依赖已有workbook并先核对其SHA，缺少原件不能宣称复现。

依据：[Apple线程局部策略](https://developer.apple.com/documentation/accelerate/blassetthreading(_:))、[vDSP内部线程说明](https://developer.apple.com/documentation/accelerate/vdsp-library)。没有将vDSP私有并行假定受BLAS策略控制。
