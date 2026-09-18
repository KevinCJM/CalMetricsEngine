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

公开 wheel 使用 baseline CPU，不允许强制 `-mavx*`、`-march=native` 或 fast-math。

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
