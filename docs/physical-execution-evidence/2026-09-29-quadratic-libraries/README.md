# 大矩阵二次型原生库复用证据

2026-09-29，macOS15.6.1/arm64、AppleClang17、CPython3.12.11。旧构建3304d272…，本轮75229d1773803cf459a8dd7a63931223ad182ed7344e43fbd3548bd5d9b85c9a。完整设计与分项结果见[物理执行优化](../../physical-execution-optimization.md)。

二次型保持Aᵀx后点积的原顺序，复用既有Eigen MatVec与N doubles工作区。2048阶C普通入口22.40→0.88ms，NJIT2.88ms；F布局仍未超过NJIT。22/22冷热活跃堆持平，完整RSS/系统调用链栈不增未证明，因此不宣称联合验收全部通过。

## 原始证据

- `source-identity.json`：全部计算源码、wheel、安装扩展与worker身份。`task-scope.json`核对本轮范围，保留其他脏工作；测试文件不参与生产构建身份，最终测试/工具hash另在`closeout-checks.json`。
- `python-final.log`：6967项，含92项二次型专项和4097块边界。`affected-native-checks.json`、三份`*-affected-test.log`：补129阶原生夹具后native/ASan/UBSan/Eigen-OFF各9/9。早期6966项、初始native与沙箱失败日志保留，最终以追加回归后的记录为准。
- `quadratic-final-timing/`：永久工具22组最终11轮普通/direct/prepared完整结果时间；`quadratic-comparison/`：同源码第一次筛选及132个独立heap/RSS子进程。输入、编译前置，首次数值计算与设置工作区进入内存epoch。原始样本不拼接为一轮，堆不覆盖所有arena/mmap/栈。
- `phase2/`、`phase2.log`：四组原有严格门禁4/4，不等于全部算子联合验收。`all-operators.json`仅185原语scalar/auto数值回放。
- `excel-snapshot-report.json`：469组/3056值对原真实MicrosoftExcel16.89.1快照一致；newly_recalculated=false，不是新重算。
- `quadratic-static-review.md`和相应反汇编：生产数学/所有权/审计审核，无功能发现；冷态平台系统栈边界未闭环。该审核在追加4097/129夹具之前，末尾测试hash为当时快照，不冒充最终测试身份。
- `quadratic-harness-review.json`、`quadratic-tool-permanent-review.json`：完整结果计时/所有权/内存epoch与永久工具AST等价检查。NumPy同公式对照不是独立的分解残差证明。
- `rss/`：4组×11对，共88个独立child，全部raw report在`raw-reports.tar.gz`；`rss-compact-summary.json`及`rss-quality.json`为索引，未替代原始数据。请求增量中位−16/0/0/0KiB，总峰中位+96/−352/−256/+208KiB，不能判不增。
- `rss_stages.py`：基于SDK真实struct的外部观察；阶段暂停改变时序，归因不是完整性能门禁。direct不可变标量也为完整输出；direct审计没有build-id字段，因此只在END后用同一已加载二进制的普通图诊断核验身份。`rss-diagnostic-failures.tar.gz`和旧失败日志保留修复前假设错误，不算通过。
- `candidates/`和`matvec-candidates.tar.gz`：未接入的编译边界候选、LLVM IR/指令/时间和材料hash。设计README为实施前准备，实测以比较JSON和冻结复核为准。未证明有效的根因，生产MatVec仍为3304；候选生产代码只存在此实验归档，不参与构建。
- `remaining-priorities.md`：下一步有界Jacobi、多右端求解、宽协方差；旧部分harness未将懒输出包装放进计时的问题保持披露，不将历史数字当当前完整验收。

## 复现

使用匹配构建的独立安装，CPU=1，基准不得与构建或回归并发：

```bash
PYTHONPATH=/path/to/new-install CME_EXPECT_EIGEN=1 python -m pytest tests tools -q
python tools/check_quadratic_performance.py --old-install /path/to/3304-install --new-install /path/to/7522-install --mode all --heap-probe /path/to/heap_probe.dylib --output /tmp/quadratic-comparison
```

原生、Excel快照和RSS脚本保留实测路径，换机器须显式替换路径并核对manifest身份。RSS脚本默认案例沿用旧轮次，复现本轮须指定四个quadratic_form_2048_*案例和old/new完整build ID；不要复用已含结果的目录。本机macOS未启用LeakSanitizer；没有跨平台、平台服务接入或发布验收。本轮未提交、推送。
