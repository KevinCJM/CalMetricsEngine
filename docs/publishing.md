# 构建与 PyPI 发布

本次改造准备的是可发布的软件包和发布流程。**没有提交、推送、创建标签、创建 GitHub Release，或上传 PyPI/TestPyPI。** 本地 `pip install .` 已不依赖 PyPI 发布；远端 `pip install my-ctools==0.2.0` 只有在该版本正式上传后才适用。

## 1. 所有者首次配置

1. 确认 PyPI 上 `my_ctools` 项目的管理权限。没有项目时，可以创建 Pending Trusted Publisher；它不会提前保留包名。若名称已被他人占用，先解决名称/权限，不在代码里绕过。
2. 原仓库没有许可证文件。所有者确认授权许可证后，再加入相应 LICENSE 和 `pyproject.toml` 的许可证元数据；本次没有擅自选择许可证。
3. 在 GitHub 仓库建立 `testpypi` 和 `pypi` 两个 environment。为 `pypi` 设置发布审批人及受保护的版本标签；限制能够修改发布工作流的人。
4. 分别在 TestPyPI 和 PyPI 配置 Trusted Publisher：

| 字段 | 配置 |
| --- | --- |
| Owner | `KevinCJM` |
| Repository | `My_C_Tools` |
| Workflow filename | `publish.yml` |
| Environment | 分别为 `testpypi` / `pypi` |
| Distribution name | `my_ctools` |

不需要在仓库里保存 PyPI Token。OIDC 的 `id-token: write` 只授予最终上传 job；构建、测试 job 无发布权限。上传在顶层 `publish.yml` 执行，`build.yml` 只负责可复用的构建与测试，避免把受信任发布者配置成可复用工作流。

## 2. 本地验收

建议 Python 3.12 独立虚拟环境：

```bash
python -m pip install ".[dev]"
python -m build
python tools/check_dist.py
python -m twine check --strict dist/*
```

`python -m build` 的默认顺序是 sdist→从 sdist 构建 wheel。安装这个 wheel 后，在源码路径之外运行完整测试；不能仅用 editable install 宣称 wheel 验收通过。

本机存在自定义 pip 源或代理且隔离构建受影响时，可以使用 `python -m build --installer uv`。不要把本机配置写入本仓库。构建目录和环境均被 `.gitignore` 排除。

`tools/check_dist.py` 检查：包名与版本、NumPy 运行时依赖、历史 Python 模块齐全、仅一个 `_core` 扩展、sdist 包含所有构建材料、没有私有环境/缓存混入。它不替代平台运行测试。

## 3. CI 矩阵

`build.yml` 覆盖 CPython 3.10–3.14 的以下 wheel：

- Linux manylinux2014：x86_64 / aarch64；Linux musllinux_1_2：x86_64 / aarch64。
- macOS 11+：x86_64 / arm64。
- Windows：AMD64。

常规 GIL CPython 构建，不包括 free-threaded、PyPy、Windows ARM64、32 位及 GPU。cibuildwheel 使用原生架构 runner，安装产物后在隔离的测试目录执行 pytest。额外 job 验证 NumPy 1.26.4（Python 3.10–3.12）、源码包重建、C++ 原生测试、ASan/UBSan。

分发以每个目标平台的基础指令集为底线，不使用 `-mavx*`、`-march=native`、`/arch:AVX*` 或 fast-math。新增平台必须同时添加构建与运行验证，不能只增加文件名标签。

## 4. TestPyPI 演练与正式发布

更新版本的唯一来源是 `pyproject.toml`。版本一经上传不可覆盖；修改后发布新的版本号。

经过代码审核、测试、许可证确认后，由所有者提交并创建对应版本标签，例如 `v0.2.0`。然后：

- 演练：手动运行 `Publish distributions`，选择该**版本标签**，`index=testpypi`。
- 正式：发布该标签的 GitHub Release，或手动选择该标签、`index=pypi`。

选择普通分支时不能发布。标签必须严格等于 `v` 加元数据版本；不允许用 `v0.2.1` 标签发布 `0.2.0` 的包。发布 workflow 重新执行同一套完整构建和测试；只有所有 job 成功才进入对应 environment 的审批/上传。未成功的矩阵不会被部分上传。源代码包和修复后的平台 wheel 作为 `dist-*` artifacts 交给最终 job。

发布以后在新的虚拟环境验证：

```bash
python -m pip install --only-binary=:all: "my-ctools==0.2.0"
python -c "import my_ctools; print(my_ctools.build_info())"
```

TestPyPI 演练时先从正常 PyPI 安装 NumPy，然后仅从 TestPyPI 获取本包，避免把测试索引作为全部依赖的混合来源：

```bash
python -m pip install "numpy>=1.26,<3"
python -m pip install --no-deps --only-binary=:all: --index-url https://test.pypi.org/simple/ "my-ctools==0.2.0"
```

## 5. 维护边界

当前改造不证明 Windows、Linux 或每一版 macOS 已实际运行通过。只有工作流的真实成功记录才能作为相应平台验收证据；本机报告单独记录。没有真实发布结果时，不能宣传新版已经能够从 PyPI 下载。

新增数值算法时：Python API 不承载业务编排；绑定负责数组契约和 GIL；纯 C++ 内核负责计算；测试同时包含参考实现和异常边界。把 BetterSaaTaa 的 NJIT 内核迁入是后续独立任务，不包含在本次架构升级中。

## 官方参考

- PyPI Trusted Publishing：https://docs.pypi.org/trusted-publishers/using-a-publisher/
- 配置 Trusted Publisher：https://docs.pypi.org/trusted-publishers/adding-a-publisher/
- 首次创建项目：https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/
- 最小权限和审批：https://docs.pypi.org/trusted-publishers/security-model/
- 可复用工作流限制：https://docs.pypi.org/trusted-publishers/troubleshooting/
- cibuildwheel 选项：https://cibuildwheel.pypa.io/en/stable/options/
