# 情景识别数学基础证据

结论与契约见[设计及验收](../../latent-model-operators.md)。本目录只证明当前 macOS arm64 本机构建的已测范围；不表示平台服务已切换，也不表示全部负载达到时间/内存双优。

| 文件 | 证据边界 |
| --- | --- |
| `source-manifest.json` | C++/构建配置逐文件哈希、工具链、安装包身份和 wheel 哈希 |
| `evidence-manifest.json` | 本任务与原脏工作树的范围区分、参考来源和测试工具哈希 |
| `python-tests.log`、`native-tests.log`、`asan-ubsan-tests.log` | 最终构建完整回归；macOS 未启用 LeakSanitizer |
| `planner-before-fix.log` | 新增 K>D 测试在旧构建复现预算低估；修复由最终回归覆盖 |
| `latent-performance.json`、`gate-summary.json` | 144 组对照的原始配对时间、申请峰值、RSS 与严格通过数 |
| `phase2-*.json`、`m1-performance.json`、`matrix-performance.json` | 原有门禁复测，阈值与失败项保留 |
| `excel-libreoffice.json` | 最终构建全目录真实重算、公式身份与输入编辑检查 |
| `excel-microsoft-generated.json` | 最终 Microsoft Excel 目标文件仅生成，锁屏阻止 UI 验收，保持 NOT_RECALCULATED |
| `doc-examples.log`、`documentation-check.log`、`codegraph-sync.log` | 文档示例、链接/算子目录与索引核对 |

性能对照固定同样的输入、初始参数、最大轮数和独立结果所有权，预热 NJIT，CPU=1，交替配对 31 轮，以比值的单侧 bootstrap 95% 上界 <1 判定时间优势。原四组门禁维持其 CPU=4 与既有中位数阈值，不能混用两种判定。

新增内存测量按后端使用独立子进程；同一后端的各负载顺序执行。malloc 探针分别记录准备、首次及热态申请，并将准备/首次保留量计入热态峰值；只覆盖探针拦截的分配家族，不代表 Python arena、mmap 或其他进程的完整内存。RSS 是预热后的增量，以 0.5ms 采样 100 次执行，不包括预热前峰值；页级噪声、分配器复用与零增量不能证明内存优势。联合通过要求时间、冷/热申请峰值、正 RSS 增量都严格优于 NJIT；RSS 单独通过不能抵消保留堆内存失败。

工作簿位于 `/private/tmp/calmetrics-latent-v8-lo/` 和 `/private/tmp/calmetrics-latent-v8-ms/`，不作为源码保存。报告记录实际文件哈希；临时文件被清理后必须按设计命令重建、重新重算，不能生成替代的验收结论。
