# 保序原生库与元数据优化证据

2026-09-29，本机macOS15.6.1/arm64、AppleClang17、CPython3.12.11。范围：Eigen保序矩阵向量乘法、固定特征维EM M-step、单chunk元数据、完整输出测量。**不是全算子时间/内存验收通过**。

- 基线构建：`a5669c9ce006ac9be9919e46088d67d9cd875122c6dc24a34bf199dc63c7a46d`。
- 最终构建：`81768857d7a1e1385f59535ac0688765518bacaeeee9eed8087fa7030b1a1b5b`。
- wheel SHA256：`cb18e618cc5e7799b78f4b1a4c1eb34327f2313b2fb1216b2c34e1f8cdf64dc4`；wheel内扩展/worker与安装后实际测试文件逐字节相同。
- Python6,823项；native、ASan/UBSan、Eigen-OFF各9/9；185算子scalar/auto回放通过。未启用LeakSanitizer，未做跨平台实测。
- 29矩阵+24完整EM结果三方对照及24特征维筛选通过原容差，318个独立child测量heap/RSS。53/53组冷/热活跃堆不增；RSS尚未证明稳定不增。
- 原四组严格门禁3/4，63点ordinary比值1.010，未放宽阈值。行连续matvec、Cholesky与RSS/全算子目标继续开放。

完整设计、数字和限制见[物理执行优化](../../physical-execution-optimization.md)。当前公开数学签名、停止语义与Excel公式/预算不变；本轮未提交或推送。

## 文件与测量边界

| 文件 | 说明 |
| --- | --- |
| `comparison-summary.json`、`matrix-comparison/`、`em-comparison/` | 原始11轮、相同输入SHA、CPU/墙钟、完整逐字段结果、形状/dtype、审计及独立heap/RSS；summary不是另一次测量 |
| `em-dimensions/` | n=32/2048×d=4/5/6/8/9/16×GMM/HMM，最终fixed/scalar准入；只有时间对照，不能当成额外内存验收 |
| `source-identity.json`、`packaging-binary-parity.json` | 源码/工具链/开关身份，wheel与安装二进制一致 |
| `simd-evidence.json` | 最终二进制SHA、局部符号/地址与NEON指令；检查范围内无FMA。不是对整包所有函数的指令审计 |
| `acceptance-status.json`、`final-tools-status.json`、测试日志 | 最终按顺序运行的返回码；phase2失败如实保留，其他阶段通过 |
| `native-checks.json`、`native_checks.py` | 三种native构建配置及9/9结果；macOS无LeakSanitizer |
| `metadata-design.md`、`latent-design.md` | 依赖/顺序/所有权/异常安全设计与候选筛选记录；最终准入以当前源码及主验收文档为准 |
| `all-operators.json` | 185原语scalar/auto正确性回放，不是185项NJIT性能验收 |
| `excel-ms-snapshot-report.json` | 469组/3056结果对已有真正重算MicrosoftExcel16.89.1快照；仅导出/构建身份变更，newly_recalculated=false |
| `phase2/` | 原四组严格门禁样本和摘要；不改微任务阈值掩盖失败 |
| `em-permanent-ast-parity.json`、`permanent-em-smoke/` | 永久EM工具与实测私有入口的计算/初始化/测量AST一致；差别是路径、说明和sys/Path导入顺序；六组小用例入口smoke |
| `candidates/` | 被拒绝的动态Eigen候选、未收紧维度候选及当时失败记录；不是最终结果。dynamic-em-raw-result还漏计懒输出包装，禁止与最终数据拼接 |
| `probe*` | 最初仅用于筛选的独立AOT原型，不能替代生产图验收 |
| `start-manifest.json`、`task-scope.json` | 任务开始状态与本轮文件范围；保护已有大量未提交工作 |

ordinary/prepared在被测callable内部物化每个真实输出并保留raw Result；direct/NJIT返回同样完整投影。数组不压平或复制来伪造所有权，EM保留int64迭代计数和bool停止字段。计时包含上一结果释放，heap热峰包含新旧结果并存；两类指标没有混成单个成本。

计时无malloc注入，所有本任务构建/测试/基准串行。独立heap epoch从图/工作区建立前开始，输入/导入/精确NJIT签名编译在epoch前，首次数值函数仍在epoch内。CPU=1，库不自建线程池；NJIT原生库被约束同核数，退出恢复线程局部设置。RSS每组合仅一次独立child，33/53组本次总峰不高于旧版，不能据此宣称统计稳定；malloc不覆盖已有CPython arena、全部mmap、线程栈或进程外资源。53/53活跃堆持平/下降不等于完整内存联合验收。

## 复现

使用与身份对应的独立AOT安装目录，不使用系统同版本包猜测身份。当前源码可离线构建，Eigen归档固定在仓库；关闭库用 `CALMETRICS_ENGINE_USE_EIGEN=OFF`。历史Accelerate构建开关已删除。

```bash
PYTHONPATH=/path/to/new-install CME_EXPECT_EIGEN=1 python -m pytest tests tools -q
python tools/check_large_matrix_performance.py \
  --old-install /path/to/old-install --new-install /path/to/new-install \
  --output /tmp/matrix-comparison --mode all --heap-probe /path/to/heap_probe.dylib
python tools/check_em_library_performance.py \
  --old-install /path/to/old-install --new-install /path/to/new-install \
  --output /tmp/em-comparison --mode all --heap-probe /path/to/heap_probe.dylib
```

父进程不能注入heap probe。工具自动为独立内存子进程启用探针；先编译探针并运行其回归。`--case-filter`可选择形状；不得据筛选子集宣称全量通过。保存的driver含原机器路径，应替换为本机目录。Excel快照比对需已有原始workbook并核对其SHA；缺少原件不能宣称重新复现，更不能将其写成新一次MicrosoftExcel重算。
