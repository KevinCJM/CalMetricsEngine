# 协方差与相关矩阵的原生库优化证据

本机 macOS 15.6.1 / arm64、AppleClang 17、CPython 3.12.11、NumPy 1.26.4、Numba 0.60.0；Eigen 为仓库固定的未修改 3.4.0 归档。设计、自审核、范围与未通过项见[物理执行优化](../../physical-execution-optimization.md)。本目录不是发布或跨平台验收。

- 基线是本任务开始时的 `df59950324aa9c52d45bfc6f11f03f97c132def603c8e381ef239ea4b6daffb6` wheel；不是再往前的 `027e8c87…` 构建。
- 最终构建：`e55cb4a1659f30b0061b2d3ee1644af81f58df66d90960973aeb6c155df5eba3`。
- 最终 wheel SHA-256：`e07cbd7e3dfa862530c23096eb8a99b41dcb39655e15830500f970cfcbe84692`。
- Python 6,550、原生/ASan/UBSan/Eigen-OFF 各 9/9、185 个算子回放通过。macOS LeakSanitizer 未启用。
- 27 个旧 C++/新 C++/NJIT 数值对照通过；64 列 C 布局协方差约 18.72 倍加速，相关矩阵约 19.86 倍加速，但仍慢于 NJIT。
- 10 组活跃堆均低于旧 C++；RSS 方向不稳定，**未宣称完整内存验收通过**。原四组严格性能门禁本次 4/4，历史失败记录保留。

## 证据口径

| 文件 | 用途及边界 |
| --- | --- |
| `source-identity.json` | 从实际源码、工具链、选项重新构造 CMake 哈希，与新 Excel 原生参考导出的 build identity 一致 |
| `packaging-binary-parity.json` | 最后仅同步随 wheel 安装的 Eigen 使用说明；扩展及 worker 与全部测试/性能所用安装包逐字节相同 |
| `timing.json`、`timing-summary.json` | 三方独立子进程，轮换顺序、保留前一次结果；11 个完整配对轮次的原始纳秒值、审计及配对比值 bootstrap 单侧 95% 上界。微型负载每轮 5 次聚类，不把 55 次当成 55 个独立轮次 |
| `memory.json`、`memory_checks.py` | 输入预先存在且相同；图编译、Scheduler、常驻工作区、冷执行、保留结果与新结果均在连续堆 epoch 中。RSS 独立子进程采样，包含基线、总峰值、增量和 idle 控制 |
| `phase2/`、`phase2.log` | 原阈值、CPU=4、同准备程度的四组门禁。比值是配对比值中位数，不等于独立时间中位数相除 |
| `final-status.json`、`native-checks.json`、对应日志 | 实际命令退出码；完整回归、原生、sanitizer、禁用 Eigen 构建与验证 |
| `all-operators.json` | 185 个原算子的 scalar/auto 回放；不是 185 个算子的 NJIT 性能验收 |
| `excel-ms-snapshot-report.json` | 469 组、3,056 个结果对照 Microsoft Excel 16.89.1 已重算快照；逐格公式/输入不变。不是新一次 Excel 重算 |
| `probe.cpp`、`probe.py`、`probe-*.json` | 筛选原型，尚未包含生产参数/错误/所有权检查。LLT/GEMM 定义 `EIGEN_STACK_ALLOCATION_LIMIT=0`，让临时数值存储显式出现在堆探针中，不能借栈隐藏工作区。未将这些原型接入生产 |
| `backend-disassembly.txt` | 本机 Release 的 NEON 指令静态证据；运行是否命中库仍看 `eigen_covariance_calls` |
| `start-manifest.json`、`task-scope.json` | 进入任务的文件哈希与本轮范围核对；原有其他未提交内容未删除或覆盖 |

计时没有 malloc 注入，不与本任务的构建/回归并发。协方差对照使用 CPU=1，OPENBLAS/VECLIB/OMP/NUMBA/MKL 环境上限均为 1；当前生产 Eigen 适配器不调用 BLAS、不创建池，也不改调用方线程设置。未声称采集了全部硬件计数器或系统级 CPU 利用率。

堆探针不覆盖已有 CPython arena 内部细分、全部 mmap 和进程外资源。禁止在 epoch 前先执行被测大图：调用线程的图工作区不随 Scheduler.close() 自动释放。零 RSS 增量不是内存优势；本轮总 RSS/增量方向不一致，不能借更低的活跃堆关闭 RSS 问题。旧的其他矩阵/M1/潜变量失败项本轮未复验，不因此关闭。

## 复现

构建使用当前源码和已安装的离线构建依赖；所有测试使用安装后的 wheel。`CALMETRICS_ENGINE_USE_EIGEN=OFF` 覆盖仍受支持的原生路径。路径需按机器调整；当前 wheel/基线包同为 0.3.0，必须核对完整 build identity，不能只看版本号。

```bash
PYTHONPATH=/path/to/new-install CME_EXPECT_EIGEN=1 python -m pytest tests tools -q
python tools/check_covariance_performance.py \
  --old-install /path/to/baseline-install \
  --new-install /path/to/new-install --output /tmp/covariance-comparison
PYTHONPATH=/path/to/new-install DYLD_INSERT_LIBRARIES=/path/to/heap_probe.dylib \
  python tools/check_covariance_performance.py --memory heap --backend new \
  --case '{"name":"covariance","n":32768,"d":64,"layout":"c","entry":"ordinary"}' \
  --heap-probe /path/to/heap_probe.dylib --output /tmp/covariance-memory
```

RSS 将 `--memory heap` 改为 `--memory rss`，并移除 `DYLD_INSERT_LIBRARIES`。性能脚本根据固定种子重新生成相同数据；NJIT 是预热后的 `np.cov/np.corrcoef`。独立结果口径覆盖 direct、ordinary 和 prepared.run_snapshot；普通借用 prepared.run 不拿来冒充独立输出。

`final_checks.py`、`native_checks.py`、`memory_checks.py` 保留本机实际目录和调用顺序。Excel 快照仍在原临时验收目录，脚本先校验原始 workbook 的 SHA-256，缺少它不能宣称已复现该快照核验。
