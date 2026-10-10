# 成熟原生库接入证据

设计、算子候选、采用范围和未达标项见[物理执行优化](../../physical-execution-optimization.md)。本目录是当前脏工作树的本机验证，不是提交、发布、跨平台或平台业务验收。

- 基线构建 `027e8c87…`；最终构建 `df59950324aa9c52d45bfc6f11f03f97c132def603c8e381ef239ea4b6daffb6`。
- 最终 wheel SHA-256：`26882d82995d02f2b80bef7e2a4f6437b3af26a5d817451392e0e2bb4078bfd9`。
- Python 6,416、原生/ASan/UBSan/Eigen-OFF 原生各 9/9；185 算子回放通过；构建无编译警告。
- 512 阶普通 DAG 约 4.61 倍加速、活跃堆峰值降低 12.44%；RSS 未稳定改善，仍明显慢于 NJIT/LAPACK。没有全量性能通过结论。
- 四组原门禁 3/4；63 点普通调用慢约 0.042 μs，保留严格失败。矩阵对照 53/56、联合 0/56；M1 联合 1/7。

## 记录说明

| 文件 | 内容与边界 |
| --- | --- |
| `environment.json`、`source-identity.json`、`tool-hashes.json` | 软件环境、工具链/源码与 wheel 身份、工具哈希；固定的 Eigen 归档哈希也纳入构建校验 |
| `start-manifest.json`、`task-scope.json` | 进入任务时的 438 个文件哈希和本轮修改清单；没有删除既有文件，其他任务的内容保持原样 |
| `eigen-comparison.json` | 最终普通入口、同输入、独立结果、CPU=1 三方轮换的原始纳秒样本及实际调用审计；没有 malloc 注入 |
| `eigen-parity-and-timing-summary.json` | 新/旧 C++ 逐位比较、冻结 NJIT 的 `rtol=atol=1e-9` 对照、残差与正交误差、输入哈希、新/旧耗时比的 bootstrap 单侧 95% 上界 |
| `eigen-memory.json` | 单独子进程的连续活跃堆及 RSS；准备、常驻工作区、前一次保留结果与本次结果均进入热态峰值 |
| `eigen-pre-final-*.json` | Eigen 初次构建的观测；随后仅修正注释及许可证安装位置。保留 RSS 方向不稳定的原始证据，不替换成最终样本冒充复现一致 |
| `accelerate-*.json`、`comparison-candidate.json` | 未采用的 Accelerate 候选以及包含 257 阶压缩回退的早期候选；不是最终生产结果 |
| `probe-*.json`、`candidate.cpp`、`probe.py` | Cholesky/LU 原型筛选。仅有限良态输入，未实现全部生产校验；其堆包含数据准备，不能与普通 DAG 堆数横向混用 |
| `jacobi-timing.json`、`eigen-probe-timing.json` | 早期单内核定位，不能把布局收益全部归因于库、也不能代替最终普通入口测量 |
| `matrix-gates.json`、`m1-gates.json`、`phase2/` | 原阈值门禁、失败负载、原始样本及内存结果；不是只收录通过样本 |
| `final-check-status.json`、日志 | 各真实退出码。首次 Phase-2 因 NJIT 线程环境不匹配未计时，第二次被沙箱 `shm_open` 阻止，第三次完整跑完并产生 63 点严格门禁失败 |
| `offline-build-binary-parity.json` | 源码目录与 sdist 离线构建的扩展、worker 二进制逐字节相同；两份 wheel 容器自身哈希不同，不声称容器逐字节相同 |
| `eigen-rotation-disassembly.txt` | 实际 Release 适配器的 NEON 指令；这是静态内核证据，实际 API 调用另由审计计数证明 |

堆探针不覆盖已有 CPython arena 内部细分、全部 mmap 或其他进程；本轮重点负载是单进程。库适配器自身不打包或创建池，但整图仍有必要缩放复制和独立输出。RSS 记录同时保留总峰值、热态基线、增量和空闲控制；零增量不是内存优势，堆减少也不是 RSS 达标。所有性能项目串行，不与本任务的构建/回归并发。

`eigen-comparison.json` 的 16/64 阶为 33 个样本，128–512 阶为 11 个样本，640/768 阶为 5 个样本。`new_old_time` 的分子是最终 Eigen 构建，分母是旧 C++；它不代表 NJIT 比值。新增大样本仍有默认 Jacobi 与 NJIT 的逐向量对照失败；旧 C++ 与新 C++ 逐位相同，未改默认精度来消除失败。

## Excel 证据

`excel-ms-snapshot-report.json`、`verify_excel_snapshot.py` 记录 469 组、3,056 个结果与既有 Microsoft Excel 16.89.1 真实重算快照的一致性。逐格验证公式与输入，只允许 Readme 的导出/构建身份更新；`newly_recalculated=false`。此次没有重新操作 Microsoft Excel，也没有对超预算大矩阵作 Excel 验收声明。

## 复现

从当前源码或 sdist 构建 AOT wheel；构建依赖须已安装，Eigen 归档随 sdist 提供。工具/脚本内的安装目录、Python、堆探针、原始算子样本和 Excel 快照路径是本次真实位置，在其他机器须替换。基线 wheel 的完整哈希见 `environment.json`，不要用另一个同版本 wheel 代替它。

```bash
PIP_NO_INDEX=1 python -m pip wheel . --no-build-isolation --no-deps --no-cache-dir -w /tmp/cme-library-wheels
python tools/check_dist.py /tmp/cme-library-wheels
PYTHONPATH=/path/to/installed-wheel CME_EXPECT_EIGEN=1 python -m pytest tests tools -q
cmake -S . -B /tmp/cme-library-native -DCALMETRICS_ENGINE_BUILD_PYTHON=OFF -DCALMETRICS_ENGINE_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cme-library-native --parallel 3
ctest --test-dir /tmp/cme-library-native --output-on-failure
```

ASan/UBSan 构建另加 `-DCALMETRICS_ENGINE_SANITIZE=ON`，macOS 运行设置 `ASAN_OPTIONS=detect_leaks=0`；无 Eigen 构建另加 `-DCALMETRICS_ENGINE_USE_EIGEN=OFF`。这些模式的独立配置、构建和日志均在本目录。

`run_final_checks.py` 顺序运行最终配对时间、内存、四组门禁、矩阵及 M1；门禁失败后仍继续，脚本完成不等于全部通过。四组门禁需双方同为 4 核并允许本机共享内存；其他矩阵比较为 CPU=1。`eigen-compare.py` 在独立子进程测量请求本身，不把控制进程 IPC 计入请求时间；NJIT 已完成首次编译。`summarize_eigen.py` 使用同一输入输出文件检查前后等价，原始大数组保留在临时目录，不打包进仓库。

库源码为[固定的 Eigen 3.4.0 上游归档](https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz)。接口及线程原则参考 [JacobiRotation](https://libeigen.gitlab.io/eigen/docs-3.4/classEigen_1_1JacobiRotation.html) 与 [Eigen 并行控制](https://libeigen.gitlab.io/eigen/docs-3.4/TopicMultiThreading.html)；在线 3.4 文档可能跟随维护版本更新，本次编译源码以仓库固定归档及哈希为准。
