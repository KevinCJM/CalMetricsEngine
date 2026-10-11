# 成熟矩阵库可行性实验

2026-09-29，macOS 15.6.1 arm64、Python 3.12.11、AppleClang 17。生产引擎仍为上一轮构建 `027e8c87b424b29daa09ee63f50d5535315cd2a5a4a0b6e20dcfa5fcb8ecdd7f`；本次没有替换生产算子，也未提交/推送。[结论及接入设计](../../physical-execution-optimization.md)。

## 证据范围

- `matrix_library_probe_v1.cpp` 对应第一组直接调用 `dpotrf/dgetrf+dgetrs/dsyevd` 的可行性实验。该源码从本轮编译输入还原保存；第一版临时二进制已被第二版替换，没有第一版二进制身份声明。
- `matrix_library_probe.cpp` 在相同入口追加 mode=3 的 64 阶分块 Cholesky，底层调用 `dpotrf/dtrsm/dsyrk`。当前二进制和源码哈希见 `probe-checks.json`。原型为独立 AOT C++/PyBind 扩展，直接链接系统 Accelerate，未调用 Python/NumPy/SciPy 完成数值。
- `library_probe.py`、`library-timing.json`、`library-memory.json` 为第一组全部样本；三种算法、16/128/512 阶、21 轮轮换顺序。`blocked_probe.py` 及对应 JSON 为第二组：Cholesky 五个规模、25 轮轮换顺序；二者不同轮，不合并计算同轮比值。
- 当前引擎使用普通 DAG 入口；原型是直接 PyBind 入口。均含必要校验、输入复制和独立结果；原型不包含 DAG/Planner/worker 开销，因此不代表生产接入后的完整收益。
- NumPy/OpenBLAS、SciPy/Accelerate、Numba 与 OMP 的线程环境上限均为 1；原型另用 macOS 15 的 `BLASSetThreading(SINGLE_THREADED)`，退出时恢复当前线程原设置。当前引擎 Scheduler CPU=1。没有建立私有线程池。
- NJIT 先编译预热。计时顺序运行，malloc 探针不参与计时；堆测量在独立子进程按现有 `benchmark_memory.py` 连续 epoch 覆盖准备、冷调用和保留前次独立结果的热执行。没有将堆峰值当作 RSS，也没有本次 RSS 联合门禁结论。
- 原型全部 9 个矩阵案例通过既有 `rtol=atol=1e-9`；分块实验通过 `rtol=atol=1e-11` 及残差检查。`probe-checks.json` 另验证只读输入不变、结果保留、失败诊断和异常后的线程设置恢复。**不是完整边界回归、跨平台验证或新的生产全量验收。**
- `dsyevd` 与现有 bounded Jacobi 是不同算法；本实验不证明 `max_sweeps/tolerance` 或退化问题下的向量选择等价。OpenBLAS/Eigen/oneMKL 仅作官方资料评估，未执行本机库间性能比较。

## 复现入口

从仓库根目录运行。脚本内路径为实际实验路径 `/private/tmp/calmetrics-followup-20260929`，安装包为 `/private/tmp/calmetrics-opt-install`。在其他机器复现需替换这些路径和 Python/include 位置；原型要求 macOS 15 及以上。编译时包含 Python 和 pybind11 头文件，并链接系统 Accelerate：

```text
clang++ -std=c++17 -O3 -DNDEBUG -ffp-contract=off -shared -undefined dynamic_lookup
  -mmacosx-version-min=15.0 -I<PYBIND_INCLUDE> -I<PYTHON_INCLUDE>
  matrix_library_probe.cpp -framework Accelerate -o matrix_library_probe<EXT_SUFFIX>
```

`library_probe.py` 用第一版源码单独编译运行；`blocked_probe.py` 使用当前源码。Python 路径需同时包含对应扩展目录和实际安装引擎；参考负载来自仓库 `tools/check_matrix_array_performance.py`，堆探针来自 `tools/heap_probe.cpp`。二进制、采样缓存留在临时目录，仓库只保留实验源码、原始数据及身份清单。
