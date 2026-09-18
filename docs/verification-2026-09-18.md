# 架构升级自审核与验收

日期：2026-09-18。原始基线 `d1ced04`；改造版本 `0.2.0`。所有改动保留在本地工作区，未提交、未推送、未上传 PyPI/TestPyPI。

## 1. 已完成

- setuptools 七扩展改为 `scikit-build-core + CMake + pybind11`，仅产出 `my_ctools._core`。
- Python API、绑定、纯 C++ 内核、公共线程/日期/数组视图分层。
- 恢复历史统计函数，保留八个函数和七条子模块导入路径，保留 `cal_std_mean_module` 兼容属性。
- 移除全局 AVX/POSIX 专属构建假设；统一 int32/int64 与纯整数 UTC 日期算法。
- 统一数组 dtype/维度/内存布局/对齐/长度/整数范围校验；只读输入、按需转换、合规数组不复制；输出由 NumPy 持有。
- 明确零行/零列行为；修正会写出数组边界的滚动窗口起点。
- per-call 线程数、线程异常 join 后传播、数值段释放 GIL。
- CI 构建、测试、sdist/wheel 校验、NumPy 最低版本、原生及绑定层 sanitizer 配置。
- PyPI/TestPyPI OIDC 发布工作流，发布 job 不执行源代码构建，只接收已验证产物并上传；版本标签与 metadata 一致性校验。

## 2. Python 实测

以下均为本机 macOS ARM64；包从普通 wheel 或 sdist 构建安装到隔离虚拟环境。运行使用 `python -I -m pytest tests -q --import-mode=importlib`，不是 editable install。

| Python | NumPy | 构建/安装方式 | 结果 |
| --- | --- | --- | --- |
| 3.10.20 | 1.26.4 | 从 sdist 构建并安装 | 234 passed |
| 3.11.13 | 1.26.4 | 从 sdist 构建并安装 | 234 passed |
| 3.12.11 | 2.5.3 | sdist→wheel，安装 wheel | 234 passed |
| 3.12.11 | 1.26.4 | 同一 CPython 3.12 wheel | 234 passed |
| 3.13.11 | 2.5.3 | 从 sdist 构建并安装 | 234 passed |
| 3.14.6 | 2.5.3 | 从 sdist 构建并安装 | 234 passed |

234 项是每个环境执行的同一套测试，不代表每个版本新增 234 个独立测试。包含 30 个实际旧 C++ 输出样本 × 3 种线程配置（90 项），以及输入、空数组、日期、所有权、并发和安装完整性测试。

## 3. C++ 与 sanitizer 实测

- ARM64 原生 C++ 可执行文件：ASan + UBSan 通过。
- x86_64 原生 C++ 可执行文件：交叉编译成功，在 Rosetta 下运行通过。不是 x86_64 Python wheel 或真实 Intel 机器验收。
- 原生测试包括 datetime64[ns] 覆盖范围内逐日公历 round-trip、月末/闰年/溢出、线程异常传播、空数据与短滚动窗口边界。共 213552 条检查，绝大多数是逐日循环断言，不应宣传成 213552 个独立业务测试。
- 完整 Python 绑定和内核的 ASan + UBSan 构建：Python 3.11.13（非 Framework）、NumPy 2.4.6，**234 passed**。
- 绑定层 sanitizer 测试关闭 LeakSanitizer，只验证地址与未定义行为；不据此宣称没有内存泄漏。sanitizer wheel 存在被忽略的 `.build-artifacts-sanitized/`，不在发布目录。

过程中的环境问题：Homebrew Framework Python 3.12 的 sanitizer 运行中报告运行库加载太晚，测试在导入时中止。随后改用现有隔离的非 Framework Python 3.11，确认 ASan 已预加载，再完整重测通过。没有修改系统 Python、安全设置或发布 wheel。

## 4. 回归差异排查

首轮历史回归有 6 个近零标准差差异，来自新旧测试编译时浮点融合设置不同。对两代代码统一 `-fno-fast-math -ffp-contract=off` 后，差异消失；没有放宽原定容差，也没有手工修改期望数值。旧统计 SIMD 代码选择其已有 scalar fallback，以排除不同硬件向量归约顺序的影响。

方差公式仍为历史 sums/squared sums 算法，其近常数、大偏移数值稳定性问题没有在本次偷偷更改。streak negative 模式、首行区间分母、max_dd bfill 和 rolling tail+1 等金融口径也保留，详见设计文档。

## 5. 打包与静态检查

已通过：

```text
python -m build --installer uv          sdist→wheel
python tools/check_dist.py             包结构、metadata、依赖与源文件完整性
python -m twine check --strict dist/*   分发元数据与 README 检查
ruff check src tests tools              Python 静态检查
ruff format --check src tests tools     格式检查
git diff --check                        diff 空白检查
```

本机 pip 的隔离构建受与项目无关的索引/代理设置影响；改用 build 的 uv installer 后成功，不修改任何全局 pip 配置。

普通发布候选产物：

```text
dist/my_ctools-0.2.0.tar.gz
dist/my_ctools-0.2.0-cp312-cp312-macosx_11_0_arm64.whl
```

已检查 wheel 中只有一个 `_core` 扩展、完整公共/兼容 Python 模块和 `py.typed`，没有 C++ 源码、测试或私有虚拟环境。旧的 macOS 15-only 中间 wheel 已清除。部署目标标记 macOS 11，不代表实际在 macOS 11 系统上跑过测试。

YAML 已解析；cibuildwheel CLI 已实际解析出以下 35 个构建标识：Linux x86_64 10 个、Linux aarch64 10 个、macOS x86_64 5 个、macOS arm64 5 个、Windows AMD64 5 个。仅验证配置解析，不宣称 GitHub 工作流已成功执行。

## 6. 自审核结论与发布前剩余事项

本地架构升级、安装和上述测试完成，未发现当前测试覆盖内的阻塞问题。下列事项仍须分别完成，不能省略：

1. 远端实际跑通 Linux/Windows/macOS 全部 wheel 矩阵。没有执行远端 CI，也没有 Windows/Linux 实机测试。
2. 所有者确认 PyPI 包名管理权限、许可证，以及 GitHub environment / Trusted Publisher 配置。本次不代选许可证，不保存发布 Token。
3. 所有者审核并授权提交/推送/版本发布后，先 TestPyPI 演练，再正式 PyPI 上传；未上传时不能宣称远端新版已可安装。

本次不包含 BetterSaaTaa 的 NJIT 内核迁移、不修改该应用、不测量其启动时间。该应用只有实际调用本包替代相应 NJIT 后，才会消除那些内核的 JIT 编译步骤。
