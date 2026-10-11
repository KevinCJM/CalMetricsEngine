# 2026-09-29 性能与内存优化证据

本目录对应最终安装 wheel，工作树基线 `d17b371abbd692e3ac839c7ba191a44d11054d7f`，包含既有未提交功能。设计、具体收益和未达标项见[优化与验收记录](../../physical-execution-optimization.md)。原始对照来自[同日优化前复验](../../operator-acceptance-evidence/2026-09-29/README.md)，其失败与阈值未改。

- 引擎构建 ID：`027e8c87b424b29daa09ee63f50d5535315cd2a5a4a0b6e20dcfa5fcb8ecdd7f`。
- wheel SHA-256：`536a76b9c205b920559126c75da034126a270e9261e0a3e5ccbf7133e85d0891`。
- 环境：macOS 15.6.1 arm64、AppleClang 17、Release、Python 3.12.11、NumPy 1.26.4。
- 结果：Python 6,375；原生和 ASan/UBSan 各 9/9；185 个算子 scalar/auto 回放通过。原四组门禁 4/4，M1 联合门禁 1/7，同核数速度检查 15/18。矩阵扩展对照仍为 53/56，三个默认 128 阶特征向量比较失败保留，不能宣称全面验收通过。

## 数据与口径

- `source-manifest.json` 包含实际构建身份、源码和包文件哈希；`tool-hashes.json` 固定测试/基准工具；`scope.json` 与 `start-manifest.json` 证明本任务只修改列明文件，其他任务的既有内容保持原样。
- `latent-performance.json`、`matrix-performance.json` 保存全部入口、数值比较、原始时间、冷/热 allocator 与独立进程 RSS 数据；`before-after.csv` 连接优化前、当前及同轮 NJIT 数据，不把前后两轮独立中位数伪称为配对统计。
- `m1-performance.json`、`phase2/`、`scaling.json` 保存原有门禁及相同 CPU 上限对照；失败不删除。`runs.json` 保存每项命令和真实退出码。总调度脚本会继续执行后续项目，因此其完成不表示每项门禁通过。
- 数值、shape/dtype、初始化、迭代次数和独立输出契约对齐；NJIT 预热后计时。所有基准依次执行，不与本任务构建/回归并发；malloc 探针只注入独立内存测量，不注入计时或生产 wheel。
- 堆测量从准备前开始同一追踪 epoch，热态保留准备/TLS，并包含上一次独立结果直到下一次返回；它不覆盖 CPython 已有 arena 内细分申请、全部 mmap/shared-memory 或独立 worker 事件。RSS 小于分辨率或为零时不能判严格优势，低总 RSS 也不能代替增量优势。
- `simd.json` 的进程累计高水位、`operator-memory.json` 的预分配后稳定性检查是辅助诊断，不是 NJIT 联合门禁，也不能证明请求零分配。

## Excel 证据边界

`excel-ms-snapshot-report.json` 为新 C++ 参考结果与既有 Microsoft Excel 16.89.1 真实重算快照的对照：469 组、3,056 个结果通过。`verify_excel_snapshot.py` 验证历史文件哈希，并逐格检查当前公式/输入，仅允许每组 Readme 的导出身份及构建身份更新。`newly_recalculated=false`；未制造缓存，也没有将此检查声称为本轮 Microsoft Excel 重算。

`excel-libreoffice-report.json` 为 LibreOffice 25.8.4.2 本轮实际重算，469 组、3,056 个结果及两项输入编辑通过。两条证据分别保存，LibreOffice 不替代 Microsoft Excel。工作簿留在报告命令所列临时目录，仓库只存清单、哈希和检查脚本。

## 复现

在仓库根目录安装当前 AOT wheel，使用 `PYTHONPATH=/private/tmp/calmetrics-opt-install` 指向实际安装包。`run_acceptance.py` 顺序记录全部工具参数、CPU/BLAS 环境和退出码；路径为本次运行的真实位置，在其他机器复现时需替换安装目录、Python、探针、原始算子样本和 Excel 快照位置。

构建命令为 `pip wheel . --no-build-isolation --no-deps --no-cache-dir --config-settings build-dir=/private/tmp/calmetrics-opt-build -w /private/tmp/calmetrics-opt-wheels`，构建前已有 CMake、Ninja、scikit-build-core 和 pybind11。完整构建日志为 `build-6.log`。

```bash
PYTHONPATH=/private/tmp/calmetrics-opt-install python -m pytest -q --junitxml=/tmp/python-final.xml
cmake --build /private/tmp/calmetrics-latent-native --parallel 4
ctest --test-dir /private/tmp/calmetrics-latent-native --output-on-failure
cmake --build /private/tmp/calmetrics-latent-asan --parallel 4
ASAN_OPTIONS=detect_leaks=0 ctest --test-dir /private/tmp/calmetrics-latent-asan --output-on-failure
clang++ -std=c++17 -O2 -dynamiclib tools/heap_probe.cpp -o /tmp/heap_probe.dylib
```

上述原生构建目录分别已配置 Release 与 ASan/UBSan；macOS 未开启 LeakSanitizer。记录只适用于本机构建，不声明 Linux/x86/Windows、平台集成或业务验收。中间候选原始记录保留在 `/private/tmp/calmetrics-opt-20260929/candidate-4`、`candidate-5`，未混入本目录的最终数据。没有提交或推送。
