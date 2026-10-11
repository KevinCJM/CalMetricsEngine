# CalMetricsEngine 构建与 PyPI 发布

本仓库当前发布单一 Python distribution：`calmetrics-engine`，Python import 为 `calmetrics_engine`。

## 首次配置

PyPI / TestPyPI Trusted Publisher：

| 字段 | 值 |
| --- | --- |
| Owner | `KevinCJM` |
| Repository | `CalMetricsEngine` |
| Workflow | `publish.yml` |
| Environment | `testpypi` / `pypi` |
| Distribution | `calmetrics-engine` |

仓库不保存 PyPI Token。正式公开发布前由所有者确认 LICENSE 与 PyPI 包名权限。

## 本地验收

```bash
python -m pip install ".[dev]"
python -m build
python tools/check_dist.py
python -m twine check --strict dist/*
python -I -m pytest tests -q --import-mode=importlib
```

完整的 NJIT 参考对照需安装 `.[test,njit-test]`，并设置
`CALMETRICS_TEST_EXPECT_EIGEN=1`。Numba 不属于引擎运行依赖；其当前二进制分发
不覆盖 Intel Mac 和部分旧 libc 平台，不能据此缩减引擎的平台支持。
这些平台的 wheel 正确性测试显式设置 `CALMETRICS_TEST_REFERENCE_BACKEND=python`，
解释执行同一份冻结参考函数体，保留全部输入、断言和容差。此模式只用于测试参考，
不是引擎运行时回退，也不是 NJIT 性能证据。

参考后端默认仍为 `njit`；缺少 Numba、未知模式或禁用 JIT 均直接报错，不自动回退。
性能工具拒绝 Python 参考模式。source、NumPy 最低版本、Linux 最新 NumPy 和 sanitizer
验收继续安装 `njit-test` 并使用 NJIT。所有 wheel 的移植性测试使用明确的 Python
参考模式；后端模式与测试结果一同记录。

安装后的包验证：

```bash
python -m pip install --only-binary=:all: "calmetrics-engine==0.3.0"
python -c "import calmetrics_engine; print(calmetrics_engine.build_info())"
```

## CI

`build.yml` 构建并测试：

- CPython 3.10–3.14。
- Linux manylinux/musllinux：x86_64、aarch64。
- macOS 11+：x86_64、arm64。
- Windows：AMD64。
- NumPy 1.26 最低兼容验证。
- C++ standalone tests。
- ASan / UBSan。

wheel 验收按五个平台和五个 CPython ABI 拆成 25 个任务；Linux 每个任务同时验证
manylinux、musllinux 及宿主机最新兼容 NumPy。每个 wheel 仍执行完整测试集。
五个原有受保护的 `Wheels` 检查名保持不变；只有全部 ABI 任务成功时才通过，
失败、取消或跳过均不通过。单任务上限为 180 分钟，用于包含大型参考计算的全量验收。

公开 wheel 使用 baseline CPU，不允许强制 `-mavx*`、`-march=native` 或 fast-math。

默认构建使用随源码固定的 Eigen 3.4.0 上游归档；CMake 校验 SHA-256 后仅在构建目录解压，不下载依赖、不读取本机 Eigen。头文件以 `EIGEN_MPL2_ONLY`、`EIGEN_DONT_PARALLELIZE`、`EIGEN_FAST_MATH=0` 编译。wheel 随附许可说明，sdist 包含原始归档和许可；`tools/check_dist.py` 校验这些内容。`CALMETRICS_ENGINE_USE_EIGEN=OFF` 测试无 Eigen 的受支持路径。

`softmax_axis` 的受控向量指数同样复用该归档；构建回归须覆盖 count-adjusted 数值准入边界、极值下溢、scalar/Eigen-OFF、不同axis/strides及失败行不提前写出。没有新增链接依赖、工作区或进程协议。

当前行连续/列连续矩阵向量乘法、行/列连续右矩阵的自动矩阵乘法、二次型的转置投影、EM特征更新与多右端保序分块消元复用同一Eigen归档；前一未发布候选的Accelerate开关/链接已移除，构建不依赖系统BLAS线程控制API。平台范围暂为实测macOS15+arm64，其他平台与Eigen-OFF使用原生路径；最低macOS目标仍为11。构建身份包含源码与Eigen开关，扩展和worker须同源（内部进程协议11）；跨平台发布仍须对应CI验证，不能以本机实测替代。各路径的时间与内存验收状态以[物理执行优化](physical-execution-optimization.md)为准。

大K、C兼容右矩阵的 `matmul` 另使用固定Eigen Array缓存分块，保留逐k累加与重复行优先复用；它没有启用Eigen默认GEMM、私有线程或新的库依赖。构建和测试须同时覆盖准入边界、特殊stride、尾块、IEEE行为及Eigen-OFF。数学、AST/DAG、进程协议、Excel公式与预算保持原契约，完整请求性能和内存欠项仍单独记录。

大矩阵 C/F 输入另有受预算的 Eigen pack/gebp 路径：M/N≥1024、K≥256 时使用有界完整K面板。固定3.4.0内部traits适配限定在 `matmul_packed.cpp`，静态断言版本、Index和NEON packet尺寸，该编译单元禁用unity/LTO并保留 `-ffp-contract=off`。升级Eigen或工具链须重新检查链接符号、实际指令、尾块和异常数值，不能仅凭库版本兼容推定等价。Planner须覆盖 `(320*K+72)*8` 字节面板及并发/子作用域副本；包、worker和源码身份一致，Eigen-OFF保持可构建。未新增公开算子、线程池或Excel公式；具体准入和审计见[执行指南](execution-guide.md)。

独立非负 MatMul 根复用同一可关闭的系统 Accelerate，限定 macOS 15+ arm64、实测尺寸/尺度及独立最终浮点结果；其他请求保持保序内核。库工作区准入估算为 `8MiB+512*K*8`，不是厂商硬上界，必须另测库内部冷/暖分配。新增 DGEMM、零填充及 reserve 审计使 IPC 双向协议升为15；扩展和 worker 必须同源构建，不能混装旧 worker。测试覆盖数据变化重读、C/F/padding、single/thread/inline/shared、预算和不准入路径，以及 Eigen/Accelerate 双关闭构建。具体范围与未闭环性能见[专项验收](physical-execution-evidence/2026-09-30-nonnegative-gemm-integration/README.md)；数学签名、Excel公式与单元格预算不变。

同源协方差/相关系数直接根可共享本次已物化的协方差核心；编译与回归须覆盖重复根、错误隔离、独立结果、prepared数据更新、线程/进程通路及Eigen-OFF。此优化复用既有内核和结果槽，未新增库依赖、协议字段或Excel数学规则。

协方差的 C 兼容布局另使用 Eigen 4×4 特征块与 128 行时间块，F 兼容布局对有限、有界且非重叠列使用 8 列均值扫描及 2×2 列对融合，保留固定 Eigen 的归约分组。仍使用同一固定 Eigen 归档；没有新增动态库、打包矩阵或私有线程。AppleClang 对 C/F 入口及完成的 F 准入、均值设置阶段显式要求尾调用，避免已经结束的栈帧与数值核叠加；升级编译器时应复核实际对象的栈与指令，不能仅凭源码推断内存收益。回归覆盖 32 行/8 列准入、128 行分块边界、四观测包和特征尾块、末行均值寿命、异常及特殊 strides。

## 发布

版本唯一来源是 `pyproject.toml`。Release tag 必须严格等于 `v<version>`。

- TestPyPI：在版本 tag 上手动运行 `Publish distributions`，选择 `testpypi`。
- PyPI：发布 GitHub Release，或在版本 tag 上选择 `pypi`。
- 所有 build/test job 成功后才允许 privileged publish job 上传。

## 计算引擎发布约束

发布前必须同时验证：

- wheel 中只有一个 `calmetrics_engine._native` 原生扩展。
- Python 层不会隐式复制或转换计算输入。
- strided / readonly NumPy views 可直接进入 C++。
- dtype 不匹配明确失败。
- 数值结果与金融历史契约回归一致。
- 没有运行时 JIT、Python fallback 或 host-specific ISA 强依赖。

Cholesky有界行面板复用Eigen公开Array/Map，限定macOS15+arm64自动ISA及n≥1024；不调用完整LLT来静默更换浮点分组。发布复核应覆盖原scalar逐位一致、病态/错误优先级、非整块尾行、负步长和只读输入、Planner工作区、进程/共享内存结果独立所有权以及Eigen-OFF。新面板/因素表必须通过Workspace预算；禁止Eigen私有线程或运行期编译，继续保留严格浮点编译设置。
