# 第一阶段 Canonical Operators 验收记录

## 交付范围

在原 CalMetricsEngine `cd2791e` 基础上增加 **118 个真实可执行 canonical C++ 算子**，不修改原 8 个 finance 算法，不修改 BetterSaaTaa，不迁移其 AST/DAG 或业务服务。版本继续沿用尚未发布的 0.3.0；本记录不代表已创建 Git tag、已提交/推送或已发布 PyPI。

| C++ family | 注册项 |
| --- | ---: |
| elementwise | 29 |
| reduction | 39 |
| sequence | 11 |
| rolling | 5 |
| matrix | 10 |
| regression | 5 |
| state | 13 |
| composite | 6 |
| 合计 | **118** |

另外有 4 个回归兼容名称被标为 regression family，但也声明了 primitive/fitting composition；因此总计 **10 个组合/兼容入口**，并非 118 套互不复用的数学实现。

注册项来自唯一 `operators.def`；Python 模块、命名调用和 get/call 都委托该 C++ 注册表。原生代码提供 lookup/prepare/execute，后续 DAG 执行器可以直接复用，不必从 C++ 回调 Python。

## 源契约与独立对照

源：BetterSaaTaa HEAD `03530df` 的 `backend/cal_indicators/`，冻结文件哈希记录在 `tests/data/canonical_reference.json`。工作区其他改动未纳入本任务。

通过 `tools/capture_canonical_reference.py --source-root <明确源目录>` 实际执行源 NJIT 内核，组合名称调用源 primitive lowering 语义，得到 **438 个用例，其中 86 个预期失败**。每个 canonical 名称至少一个真实成功用例。未使用新的 C++ 结果反向生成 oracle。

测试覆盖：118 名称与显式 opcode、源参数名/arity、scalar/auto、矩阵轴、窗口与缺失、同深度回撤、OLS 投影、C/F order、负 stride、非连续只读输入、out 对象/地址一致、输入未修改、返回视图生命周期和算法工作区复制计量。

测试 fixture 固定在本项目内。普通测试和已安装软件包不需要原仓库、Numba 或网络。

另外新增**独立 NumPy / Pure-Python reference**，不导入 CalMetricsEngine 数值实现、BetterSaaTaa 或 Numba，也不读取 fixture 的 expected 作为计算输入。全部 118 个算子均至少有一个独立 reference 路径；352 个成功 fixture 分别在 scalar 和 auto ISA 下执行，共 **704 个数值交叉验证 case**，再加 1 项 118/118 覆盖检查，共 **705 passed**。`normal_ppf` 使用 Python `statistics.NormalDist.inv_cdf` 作为独立高精度算法，因此允许约 2e-9 的算法近似差异；其余 reference 使用 NumPy 或可读的纯 Python 数学定义。

## 已执行的功能与内存验证

- Python 3.12 当前完整测试：**4034 passed**，包括原有回归、冻结 NJIT oracle、独立 NumPy/Pure-Python parity、SIMD/内存/接口契约和 Phase-2 graph runtime。
- C++ standalone：finance、operator、graph 三个 CTest suite **3/3 passed**；operator suite 直接执行全部 118 项，graph suite 独立验证 native DAG。
- ARM64 standalone ASan/UBSan：三个 suite **3/3 passed**。
- Python 3.11 独立解释器、带 ASan/UBSan 的实际 `_native` 扩展：**3272 passed**。
- Python 3.12 Homebrew framework 解释器的 ASan 预加载未生效，导入时出现 interceptor 初始化错误；该环境不计为绑定层通过，改用上述 Python 3.11 独立解释器完成绑定层内存检查。
- 本机禁用了 LeakSanitizer，未据此声称完成泄漏检测。ASan/UBSan 验证的是所覆盖的内存访问和未定义行为路径。
- workspace 并发复用明确返回 WORKSPACE_BUSY，独立调用的只读数组并发测试通过；Phase-2 process/thread/shared-memory 调度另见 `phase2-execution-graph-acceptance.md`。
- 共享内存 ndarray 直接借用验证通过。初次写入共享存储属于测试的数据准备，不算作执行阶段零拷贝的证明范围。

### 输入与算法工作区

数组绑定只借用输入地址/stride，审计中的 input_copy_bytes 为 0。地址检查、readonly 输出、owner 生命周期、输入未变和 out 重叠拒绝均有测试，不只凭 copy=False 宣称零拷贝。

median/quantile 将选中的数值一次复制到连续 `double` scratch 后排序，并通过 `algorithm_copy_bytes` 报告；这样保持 caller input 只读且改善 cache locality。solve 同样初始化可修改的系数/RHS 工作区并报告 `algorithm_copy_bytes=(n*n+n)*8`。这些都不能与绑定层 `input_copy_bytes` 混淆。

原生直接调用者必须保证指针有效、输出不重叠和 workspace 独占；对于不透明 exporter/capsule，绑定无法独立证明真实分配范围，仍依赖 exporter 契约。

## SIMD 实现与真实验证边界

当前有 **18 个显式 SIMD 算子**：17 项元素/比较/mask 及 matmul。未将顺序敏感归约、递推、排序和状态投影标为已向量化。

| 平台/ISA | 实际证据 |
| --- | --- |
| ARM64 scalar + NEON | 本机真实执行；Python 与 standalone 测试通过 |
| x86_64 scalar + SSE2 | 本机交叉编译并在 Rosetta 执行 standalone 全量算子/边界测试通过 |
| x86_64 AVX2 | 独立源文件编译通过；当前 Rosetta 未暴露 AVX2，未执行该路径 |
| AVX-512 | 未实现 |
| Linux/Windows | 已接入现有 CI 构建/测试入口，本次没有远端运行结果 |

AVX2 标志只作用于对应 translation unit；初始化/基线代码不强制 AVX。运行时检查 CPU 与 OS 的向量寄存器支持，显式请求不支持 ISA 时失败，不能强行启用以伪造验证。

## 性能记录

命令：

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python tools/benchmark_operators.py --output .build-bench/operators.json
```

本机 macOS ARM64、Python 3.12，15 次重复取 warm-call 中位数。输入、out 和必要 workspace 在计时前准备；计时包含 Python→C++ 边界验证。只对当前单算子调用有效，不推断平台启动时间、整图收益或多进程吞吐。

| 任务 | C++ scalar | C++ auto/NEON | NumPy reference |
| --- | ---: | ---: | ---: |
| add，1,000,000 连续元素 | 3.622458 ms | 0.390958 ms | 0.347583 ms |
| multiply，1,000,000 连续元素 | 3.508583 ms | 0.390166 ms | 0.366750 ms |
| finite_mask，1,000,000 连续元素 | 1.388500 ms | 0.365750 ms | 0.128292 ms |
| add，1,000,000 元素、stride=2 | 3.530583 ms | 3.529959 ms（scalar lane） | 0.610917 ms |
| matmul，32×32 | 0.007750 ms | 0.003708 ms | 0.001208 ms |
| matmul，128×128 | 0.299583 ms | 0.177417 ms | 0.014583 ms |
| matmul，256×256 | 2.635667 ms | 1.413375 ms | 0.101000 ms |

连续百万元素上，NEON 相对本项目 scalar：add 约 **9.27×**、multiply 约 **8.99×**、finite_mask 约 **3.80×**。但 NumPy 仍分别约快 11%、6% 和 2.85×；stride=2 当前没有 SIMD fast lane，NumPy 明显更快。矩阵 SIMD 相对本项目 scalar 有提升，但 **256×256 NumPy/BLAS 约快 14×**，因此当前 matmul 不应作为 BLAS 替代。

这些结果说明 SIMD 实现是真实生效的，但“C++”本身不等于“比 NumPy 更快”；下一阶段应优先优化调度/融合、减少多节点往返，并对矩阵路径考虑系统 BLAS/backend dispatch，而不是重复造高端 GEMM。

### 内存与 steady-state 验证

`tools/benchmark_operator_memory.py` 在**每个场景独立子进程**内预先分配输入、`out` 和 Workspace，先 warm 一次，再测 20 次 steady-state 调用。当前结果：

| 场景 | 中位时间 | Workspace | steady-state RSS 高水位增长 | Python tracemalloc peak |
| --- | ---: | ---: | ---: | ---: |
| add 1M | 0.382895 ms | 0 B | 32,768 B | 3,576 B |
| rolling_min 1M, window=252 | 15.665729 ms | 2,016 B | 131,072 B | 3,576 B |
| quantile 1M | **72.206542 ms** | 8,000,000 B numeric scratch | 163,840 B | 3,580 B |
| solve 256×256 | 1.953000 ms | 526,336 B | 32,768 B | 3,576 B |
| covariance 5000×64 | 15.281375 ms | 1,024 B | 65,536 B | 3,576 B |
| matmul 256×256 | 1.403396 ms | 0 B | 32,768 B | 3,576 B |

所有 118 个算子的成功 fixture 审计中，**最大 `input_copy_bytes = 0`**。median/quantile 使用连续 numeric scratch 并明确报告算法复制；rolling min/max 的 scratch 按 window 而非历史长度增长，covariance/correlation 保存列统计，quadratic_form 保存一个向量；`solve` 也报告可修改消元工作区的初始化复制。`lag`、`transpose` 和二维 `diag` 可直接返回只读 view。

RSS 32 KiB 的增长属于进程高水位粒度/allocator 行为，不能解释成输入复制；精确输入复制与算法工作区仍以 native audit 为准。20 次 warm steady-state 中 Workspace capacity 均保持不变。

## 完整多产品 × 多区间 × 多指标对比

已新增 `tools/benchmark_multiworkload.py` + `tools/benchmark_multiworkload.cpp`，直接比较 BetterSaaTaa 真实 `CompiledNumbaBatchPlan` 与 CalMetricsEngine C++ Operator Registry，而不是逐算子微基准。

这是 **Phase 1 Operator Registry 的历史基准**：500 产品 × 2520 日 × 12 区间 × 16 指标时，结果与 NJIT 零误差，但当时 C++ 逐指标执行比 fused NJIT 慢约 19%–25%，1000 产品规模慢约 26%–30%。该结果只说明单纯 C++ 化算子不足以赢过 fused DAG；Phase 2 的 Native DAG + reduction fusion 已取代这一路径，当前性能结果见 `phase2-execution-graph-acceptance.md`。

同时，BetterSaaTaa 本次运行时计划编译约 10 秒，并带来数百 MB 级进程 RSS high-water 增长；CalMetricsEngine 为 AOT，无请求/启动时 JIT。完整方法、线程扩展曲线、内存边界与限制见 `docs/multiworkload-benchmark.md`。

该 Phase-1 结果解释了为什么需要 Native DAG/CSE/统一调度。Phase 2 已完成这些执行侧能力；新的整图结果见 `phase2-execution-graph-acceptance.md`。

## 可重复的原生检查命令

```bash
cmake -S . -B .build-operators-native -DCMAKE_BUILD_TYPE=Release \
  -DCALMETRICS_ENGINE_BUILD_PYTHON=OFF -DCALMETRICS_ENGINE_BUILD_TESTS=ON
cmake --build .build-operators-native --parallel 4
ctest --test-dir .build-operators-native --output-on-failure

cmake -S . -B .build-operators-sanitized -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCALMETRICS_ENGINE_BUILD_PYTHON=OFF -DCALMETRICS_ENGINE_BUILD_TESTS=ON \
  -DCALMETRICS_ENGINE_SANITIZE=ON
cmake --build .build-operators-sanitized --parallel 4
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
ctest --test-dir .build-operators-sanitized --output-on-failure
```

Python 绑定层 sanitizer 测试必须用支持预加载的解释器，并安装带相同 sanitizer 的本项目扩展，不能只运行普通 wheel 后宣称通过。

## 安装产物终验

`python -m build --installer uv --outdir .build-operators-dist` 已完成 sdist → 从该 sdist 重建 wheel。`tools/check_dist.py` 和 Twine strict metadata 检查通过，随后安装实际产物测试：

| 本机环境 | 安装来源 | 结果 |
| --- | --- | --- |
| CPython 3.10.20 | 发布候选 sdist 构建安装 | **4034 passed** |
| CPython 3.11.13 | 同一 sdist 构建安装 | **4034 passed** |
| CPython 3.12.11 | 从 sdist 重建的 wheel | **4034 passed** |
| CPython 3.13.11 | 同一 sdist 构建安装 | **4034 passed** |
| CPython 3.14.6 | 同一 sdist 构建安装 | **4034 passed** |
| CPython 3.12.11 + NumPy 1.26.4 | 同一 CPython 3.12 wheel | **4034 passed** |

这些都是本机 ARM64 环境，不代表 Linux、Windows 或其他 macOS 版本已验证。候选产物位于 `.build-operators-dist/`；本机 wheel tag 为 `macosx_15_0_arm64`，不以 CI 配置中的 macOS 11 目标代替真实产物标签。

本轮实际发现并修复源码包包含范围问题：scikit-build-core 默认/classic 规则中 include 优先于 exclude，原 `docs/**` 会强行带入本地私有目录。已将包含范围收窄为源文件扩展名、测试夹具和顶层 Markdown；私有配置留在本地，最终 archive 检查确认不包含它，也禁止 `__pycache__`/pyc 混入。未上传过含私有配置的中间产物。

wheel 检查确认包含新的 operators 入口且只有一个 `_native` 扩展；sdist 包含完整 C++ registry/SIMD 源码、独立测试、冻结 fixture 和必要文档。没有新增 Numba 或编译器运行时安装依赖。

官方打包规则依据：https://scikit-build-core.readthedocs.io/en/latest/reference/configs.html#sdist-include

## 未纳入本阶段

DSL/AST/Typed DAG 迁移、整图 NativeExecutionPlan、CSE/liveness、统一 Scheduler、持久线程池/进程池、跨进程 Hard Stop、服务部署和 PyPI 发布。本阶段的接口和测试为后续集成准备，不等于这些能力已完成。
