# 行连续大矩阵优化证据

2026-09-29，本机macOS15.6.1/arm64、AppleClang17、CPython3.12.11。最终生产构建为 `3304d2724a8dce1611c5d16eae4ca74d652d993439f49f5c9827532e456eb6f0`，基线为 `81768857…`。完整设计和结论见[物理执行优化](../../physical-execution-optimization.md)。

采用Eigen R8/R1借用Map的行连续MatVec，保持列累加顺序与独立结果。普通大矩阵入口相对基线约1.56–2.70倍加速；29/29冷/热活跃堆持平，所审查栈帧240 B不增。**C布局仍未全面超过NJIT，完整RSS联合验收未闭环**。最终Python6,875项、native/ASan/UBSan/Eigen-OFF各9/9、185原语回放、原四组性能门禁4/4通过。没有提交或推送。

## 证据与边界

- `source-identity.json`：逐源文件SHA、最终wheel身份；重建后的扩展和worker与实际测试安装包逐字节相同。最终源代码是3304；`candidates/`里的其他构建均不是采用版本。
- `acceptance-status.json`、`native-checks.json`和测试日志：3304按顺序构建/完整验证。最后增加5项所有权回归后，`python-final-complete.log`全量为6,875；MatVec专项204项。首次沙箱共享内存限制与4项新夹具维度错误保存在候选日志，不计作通过。
- `matrix-comparison/`：29组形状/布局/入口，旧/新/NJIT轮换11轮，CPU=1，完整公开输出；独立174个heap/RSS子进程。输入/编译前置，首次实际数值执行、workspace与双结果共存进入内存epoch。malloc不覆盖所有Python arena、mmap或栈。
- `phase2/`：原四组门禁的原始记录，阈值未改。500/1000比值0.756/0.755，63/252点ordinary比值0.973/0.969；这不代表全部185项性能通过。
- `matvec-r8-r1-disassembly.*`、`matvec-review.md`：最终3304的静态地址/指令/栈证据。R8主循环是8个独立标量累加器；局部R1乘积与F路径含NEON。所查函数无FMA、未见直接数值栈spill。不能仅凭Eigen名称推定SIMD或端到端收益。
- `excel-snapshot-report.json`：469组/3056值与既有真实MicrosoftExcel16.89.1重算快照一致；输入/公式未变，newly_recalculated=false，**不是新一次Excel重算**。重现需原workbook及匹配SHA，不能用合成缓存值替代。
- `rss-attribution-review.md`、`final-rss/`：4组×11对、88个child的外部分阶段RSS归因。请求增量中位下降，但C2048总峰中位+64KiB，不能宣称RSS全部不增。每个原始child报告压缩在`raw-reports.tar.gz`，摘要没有替代原始样本。
- `previous-round-rss/`：a566→817的3组×11对、66个child归因；GMM冷计算后external固定+32KiB，11/11一致，仍开放。与本轮生产改动分开，不拼接成相同测量样本。
- `rss_stages.py`、`rss-stages-design.md`：归因脚本和SDK观察器边界。父进程采样，记录阶段/实际间隙，独立释放、关闭、GC；暂停和快照改变时序，因此仅用于归因，不能自动把原RSS门禁改成通过。
- `candidates/`：冻结原型/原始筛选、初版栈问题、noinline和Cholesky未采用证据。完整LLT逐项误差失败；保序Cholesky更慢；noinline未明显提速且F链栈增64 B。独立原型比生产更快的原因仍未证明。设计探索记录不覆盖最终3304实现。
- `start-manifest.json`、`task-scope.json`：任务范围核验，仅改变5个存量文件并新增本证据目录，其他既有文件哈希保持。

## 复现

以对应构建的独立AOT安装运行，禁止凭版本号使用另一个系统包。保留完整输出物化和普通/direct/prepared独立结果；性能测试不与构建、回归或其他数值任务并行。

```bash
PYTHONPATH=/path/to/final-install CME_EXPECT_EIGEN=1 python -m pytest tests tools -q
python tools/check_large_matrix_performance.py \
  --old-install /path/to/baseline-install --new-install /path/to/final-install \
  --output /tmp/dense-comparison --mode all --heap-probe /path/to/heap_probe.dylib
```

原生构建脚本、Excel快照脚本和归因脚本保留实测机器路径；重现时须替换路径并核对完整构建身份。归因脚本默认测上一轮，测本轮应显式指定old/new安装与build ID、C/F四组case；参数见`--help`及manifest。禁用Eigen验证使用 `CALMETRICS_ENGINE_USE_EIGEN=OFF`。本机macOS未开启LeakSanitizer，不能据此声称其他操作系统验收通过。
