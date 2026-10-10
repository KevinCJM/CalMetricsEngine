# 矩阵与数组基础原始证据

结论、范围与复现命令见[设计与验收](../../matrix-array-foundation.md)。本目录对应 2026-09-25 本机 macOS arm64 测量；`gate-summary.json` 明确记录 `performance_accepted: false`。

| 文件 | 用途 |
| --- | --- |
| `source-manifest.json` | 最终源码逐文件哈希、编译身份及 wheel 哈希；与安装包身份一致 |
| `evidence-manifest.json` | Git 基线、脏工作树说明、测试/测量工具及证据哈希 |
| `python-tests.log`、`native-tests.log`、`asan-ubsan-tests.log` | 最终构建完整回归；ASan 未启用 LeakSanitizer |
| `doc-examples.log`、`documentation-check.log`、`codegraph-sync.log` | 示例、文档与索引核对；不是性能或业务迁移证明 |
| `matrix-performance.json`、`gate-summary.json` | 新增 44 组对照的最终样本、审计、分配与 RSS，以及通过数 |
| `phase2-*.json`、`m1-performance.json` | 原门禁结果及原始数据；保留未通过项 |
| `matrix-before-stride-optimization.json` | 本轮寻址优化前的诊断基线，不能与最终构建结果混用 |
| `diagnostic-before-import-fix.json` | 最终二进制的早期测量，其中第一项 C++ 准备成本错误包含首次 import；仅供追溯，不作为最终门禁证据 |
| `excel-libreoffice.json` | 最终构建全部夹具实际重算及输入编辑检查 |
| `excel-microsoft.json` | 本轮早期构建的新增 28 组在 Microsoft Excel 实际重算 |
| `excel-microsoft-final-generated.json` | 最终 Microsoft Excel 目标文件仅生成，状态保持 NOT_RECALCULATED |
| `excel-final-formula-equivalence.json` | 两构建的 16,802 单元格一致性核对和明确证据边界 |

工作簿不作为源码提交；本轮本地文件分别位于 `/private/tmp/calmetrics-matrix-lo-acceptance/`、`/private/tmp/calmetrics-matrix-ms/` 和 `/private/tmp/calmetrics-matrix-ms-acceptance/`。报告记录重算文件哈希，等价报告另记录源文件哈希。临时文件可能被系统清理；届时应按设计中的命令重建并重新验收，不能伪造新的重算记录。

首次与热态分配分别保留；RSS 为后端隔离进程内采样，受到页粒度、分配器复用和操作系统调度影响。新负载本轮仅直接三角求解 64 阶同时满足三个条件，不表示所有新算子、平台算法或所有内存口径已严格双优。
