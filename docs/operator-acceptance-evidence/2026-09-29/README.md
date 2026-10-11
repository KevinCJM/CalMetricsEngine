# 2026-09-29 算子复验证据

结论见[全量复验](../../operator-acceptance.md)。本目录保留成功与失败；性能脚本成功退出有时只表示测量完成，以逐项字段为准。

| 文件 | 用途 |
| --- | --- |
| `source-manifest.json`、`evidence-manifest.json` | 构建/安装包一致性、工作树范围、来源和证据哈希 |
| `summary.json`、`time-memory-comparisons.csv` | 总结与 248 项情景/矩阵入口的逐项时间、内存和正确性结果 |
| `operator-coverage.json`、`all-operator-smoke.json` | 全部 185 算子的实际成功调用来源和 scalar/auto 回放 |
| `python-final.log/xml`、`native-tests-unsandboxed.log`、`asan-ubsan-tests-unsandboxed.log` | 最终回归；未启用 LeakSanitizer |
| `excel-ms-*`、`excel-lo-*` | 分别记录真实 Microsoft Excel 与 LibreOffice 的公式/输入身份、值及编辑对照 |
| `latent-performance.json`、`matrix-performance.json` | 配对原始时间、独立内存测量及审计；含扩展负载 |
| `matrix-performance.log`、`matrix-complete.log` | 首次被特征向量差异中断的运行，以及保留该失败后完成的全矩阵运行 |
| `eigen-diagnostic.py/log` | 默认精度失败、残差及显式更高精度诊断；不放宽现有门槛 |
| `m1-performance.json`、`phase2-*.json`、`scaling.json` | M1、旧四组、同 CPU=1/2/4 扩展对照 |
| `simd.json`、`operator-memory.json`、`heap-probe-test.log` | 实际 ISA/复制/工作区，以及内存工具回归 |
| `runs.json`、`run_checks.py`、`run_followup.py` | 串行执行参数、顺序、退出码和耗时；机器路径是本次执行记录 |

本机 Python 为 `/Users/chenjunming/Desktop/myenv_312/bin/python3.12`，安装包路径 `/private/tmp/calmetrics-latent-install`；原始工作簿及中间文件在 `/private/tmp/calmetrics-reaccept-20260929`。临时文件清理后需重新生成与重算，不得合成 Excel 缓存或替代已有验收结论。

独立复现时先安装身份匹配的 wheel，设置 `PYTHONPATH` 和 BLAS/Numba 线程预算。下列命令均从仓库根目录执行，输出目录按本机实际位置设置；有计时的命令串行运行，不与构建、全量测试或 Excel 重算并行。

```bash
python -m pytest tests -q
python tools/check_all_operator_cases.py --collect --fixtures /tmp/cme-cases.pkl --output /tmp/cme-coverage.json
python tools/check_all_operator_cases.py --fixtures /tmp/cme-cases.pkl --output /tmp/cme-replay.json
clang++ -std=c++17 -O2 -dynamiclib tools/heap_probe.cpp -o /tmp/cme-heap.dylib
clang++ -std=c++17 -O2 tests/heap_probe_tests.cpp -o /tmp/cme-heap-test
DYLD_INSERT_LIBRARIES=/tmp/cme-heap.dylib /tmp/cme-heap-test
python tools/check_latent_performance.py --extended --heap-probe /tmp/cme-heap.dylib --output /tmp/cme-latent.json
python tools/check_matrix_array_performance.py --extended --heap-probe /tmp/cme-heap.dylib --output /tmp/cme-matrix.json
```

仅加载覆盖工具在本机生成的可信 pickle。Excel 复现用 `tools/check_excel_parity.py --engine none --check-edits --output <目录>` 生成，在目标 Microsoft Excel 实际重算并保存后执行 `--engine read --check-edits`；编辑夹具也须分别打开、重算和保存。LibreOffice 用 `--engine libreoffice --check-edits`。具体已执行命令见 `runs.json`，其他参数以工具 `--help` 为准。
