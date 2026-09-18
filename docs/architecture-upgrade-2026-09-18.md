# My_C_Tools 架构升级设计

日期：2026-09-18。基线：`d1ced04`；统计函数追溯至 `1bf35be^:my_ctools/cal_std_mean.cpp`。

## 1. 目标与范围

将现有指标库升级为可通过 PyPI 分发的跨平台预编译 Python 包，供 BetterSaaTaa 等项目独立依赖。保留八个历史公开函数及七条历史子模块导入路径。不迁入 BetterSaaTaa 业务、不新增投资算法、不发布远端版本。

| 四问 | 本次回答 |
| --- | --- |
| 是否属于当前需求？ | 构建、平台兼容、API 边界、模块化、测试和发布流程属于；金融计算口径变更不属于。 |
| 不改是否阻塞？ | 缺失源文件、强制 AVX、Unix 日期 API、Windows long 位宽、非法数组越界均阻塞可靠分发。 |
| 能否最小改造？ | 保留已有计算过程，拆出公共基础设施；一个扩展、轻量兼容模块，不建立插件框架或大型线程池。 |
| 如何验证与回退？ | 固定 Git 基线做差分回归；新增边界测试；sdist→wheel→干净安装；旧版本不覆盖，新版本采用 0.2.0。 |

## 2. 原项目事实

- `setup.py` 使用七个独立扩展，引用已删除的 `cal_std_mean.cpp`；`__init__.py` 仍导入两个统计函数，当前 checkout 不能完整构建。
- 全局 `-mavx/-fPIC/-O3` 不适用于所有目标编译器和硬件。
- 多个文件直接调用 `gmtime_r`；周期数组用 `long`，在 Windows 与 Unix 上位宽不同。
- 指针按 C-contiguous 访问，但未验证 shape、stride、长度、对齐。
- 各函数自行创建线程，没有统一异常回收；计算时未释放 GIL。
- 滚动收益在窗口覆盖全样本时可能从 `ret[-1]` 写入；最大回撤在零行时访问 `cum[0]`。
- 无自动测试；CI 仅构建 CPython 3.8 Linux wheel；仓库没有许可证文件。
- 没有 `.codegraph/` 索引，本次不擅自生成索引。

## 3. 分层结构

```text
src/my_ctools/        稳定 Python API、输入标准化、历史导入兼容
        ↓
cpp/bindings.cpp     dtype/维度/长度/对齐校验、GIL、NumPy 输出
        ↓
cpp/kernels/         纯 C++ 数值算法，无 Python/NumPy 头文件
        ↓
cpp/include/my_ctools/  数组视图、UTC 日期、可控并行、内核声明
```

只生成 `my_ctools._core` 一个扩展。C++ 以静态目标链接入扩展，无单独运行时库、无 OpenMP、无 BLAS 新依赖。`src` 布局防止源码目录掩盖安装包缺漏。版本只在 `pyproject.toml` 维护，由构建系统传入 C++。

## 4. Python API 与内存契约

- 保留旧名字、参数顺序、返回 tuple/dict/数组布局。`cal_std_mean` 历史上只返回样本标准差；`cal_std_mean_simd` 返回 `(2, N)`，第一行为均值。后者保留历史名字，但不承诺手写 SIMD。
- 增加 keyword-only `n_threads=1`；`0` 表示自动，默认单线程避免服务多 worker 叠加过量线程。调用之间没有全局线程池或可变线程配置。
- 普通 API 接受实数 array-like；标准化为 native-endian、对齐、C-contiguous float64。已经符合要求的 NumPy 输入不复制；F-order、切片、非 float64 或未对齐输入按需复制。
- 类型数组统一 int32；日期和索引统一 int64。拒绝整数溢出和小数索引。日期支持 int64 纳秒，或精确 `datetime64[ns]` 视图；拒绝 NaT。
- `_core` 是严格内部接口：不自动转换 dtype/stride；适合未来经验证的生产 ndarray 路径。不作为稳定外部 ABI。
- NumPy 输出先分配，由 C++ 直接写入；输入只读。只有数值计算期间释放 GIL，创建 Python 对象、校验与打包时持有 GIL。调用方不得并发修改输入。
- 零列返回空结果；零行有明确定义而非越界：统计/CPR/滚动输出 NaN，最长段输出 0，最大回撤输出 NaN、空日期和未恢复哨兵 1000000。
- 结果周期与索引统一 int64。计算不启用 fast-math；NaN、并列极值和恢复哨兵保持旧口径。

## 5. 线程和日期基础设施

线程数量受列数和上限限制；单线程直接执行，不创建 OS 线程。线程异常捕获并在所有线程 join 后回到 Python；线程创建失败也先回收已创建线程。日期使用纯整数公历算法，不依赖平台的 `gmtime/timegm` 范围、时区或 locale；支持纳秒时间戳覆盖的公历日期和月末截断。

## 6. 硬件与构建支持

采用 `scikit-build-core + CMake + pybind11 + cibuildwheel`。CPython 3.10–3.14 常规 GIL 构建；不宣称支持 PyPy、free-threaded、GPU 或 32 位。

| 平台 | wheel 目标 | 策略 |
| --- | --- | --- |
| Linux glibc | x86_64 / aarch64 | manylinux2014，glibc >= 2.17 |
| Linux musl | x86_64 / aarch64 | musllinux_1_2 |
| macOS | x86_64 / arm64 | 分别构建，最低 macOS 11 |
| Windows | AMD64 | MSVC，64 位 |

Windows ARM64 暂不列为已支持；之后必须在真实 ARM64 Python/NumPy 环境测试后加入。其他平台可尝试 sdist 自编译，但不在承诺矩阵中。

不使用 `-march=native`、`-mavx*` 或 `/arch:AVX*`；默认使用目标架构基础指令集与编译器优化。不是一个二进制兼容全部硬件，而是每个 OS/CPU/Python ABI 对应 wheel。SIMD runtime dispatch 留作有性能证据后的独立需求。

## 7. 构建和发布链

- `python -m build` 先构建 sdist，再从 sdist 构建 wheel。
- wheel 测试在源码目录之外运行；同时测试 NumPy 1.26（Python 3.10–3.12）与 NumPy 2.x，其他 Python 按 NumPy 可用版本执行。
- CI 的 wheel 矩阵安装后运行完整 pytest；sdist 检查文件清单和元数据。
- 发布工作流复用同一构建工作流；GitHub Release 的版本必须与 metadata 一致；所有构建和测试通过后才进入受保护的 PyPI environment。
- PyPI/TestPyPI 使用 Trusted Publishing，不在仓库保存 API Token；由所有者配置项目名、仓库、工作流及 environment。
- 本次只准备发布能力，不创建 Release、不打 tag、不 push、不上传 PyPI。
- 原仓库没有授权许可证；不擅自替所有者选择 MIT 等许可证。公开分发前由所有者确认许可证与包名权限。

## 8. 回归口径与已知历史行为

差分测试编译 Git 中旧实现，统计函数从删除前恢复。为跨指令集比较，旧统计代码选择其已有的 scalar fallback；新旧构建统一禁用 fast-math 与浮点融合，不修正旧公式。比较八个函数的返回布局、数值、NaN、日期和周期，容差固定为 rtol=2e-12、atol=2e-14。危险输入不执行旧代码，改用新边界测试。历史 variance 使用 sums/squared sums，近常数或大偏移数据可能存在消减误差；这不代表新版本已改为数值稳定的方差算法。

保留并记录以下行为：largest/longest 的 `negative` 模式仍按原实现选择正值段；longest 在段从第零行开始时以最后一行累计值作分母；max_dd 的 bfill 与全 NaN 口径不变；rolling 保留原 tail+1 规则，但起点至少为 0。上述金融口径需另开需求审议，不能借架构升级偷偷更改。

## 9. 验收与限制

必须验证：本机源码构建、sdist 重建、普通 wheel 安装、八个 API、历史子模块导入、dtype/shape/stride/对齐/空数组/异常、线程一致性、只读输入、结果生命周期、多个 Python 版本。跨平台 CI 配置完成不等于其已运行；验收报告严格区分本机实测和待 CI 验证。实施结果、测试环境和遗留限制见 `docs/verification-2026-09-18.md`。

## 技术依据

- pybind11 官方构建指南：https://pybind11.readthedocs.io/en/stable/compiling.html
- scikit-build-core：https://scikit-build-core.readthedocs.io/en/stable/guide/getting_started.html
- cibuildwheel 架构与测试选项：https://cibuildwheel.pypa.io/en/stable/options/
- PyPI Trusted Publishing：https://docs.pypi.org/trusted-publishers/using-a-publisher/
