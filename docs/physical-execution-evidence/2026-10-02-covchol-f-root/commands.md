# 本轮执行与复现

主代理在 CalMetricsEngine 内串行构建、运行数值和性能。Python：`/Users/chenjunming/Desktop/myenv_312/bin/python3.12`；CMake/CTest：`/opt/homebrew/bin`。分支 codex/submission-policy 的共享脏状态被保留，无提交、推送或 Git 基线改变。

ON 使用 `/private/tmp/calmetrics-covchol-next/wheel-build`，Release/macOS15、Eigen/Accelerate/Highway 全部 ON；OFF 使用同目录旁的 off-build，macOS11、三库全部 OFF。pip 的无隔离构建路径包含已有本机依赖，`PIP_NO_INDEX=1`；没有请求期编译或下载新依赖。精确工具链/参数/77源SHA/二进制在 identity.json 和 identity-off.json。

```sh
env PATH=/private/tmp/calmetrics-perf-followup-deps/bin:/opt/homebrew/bin:/usr/bin:/bin PYTHONPATH=/private/tmp/calmetrics-matmul-deps:/private/tmp/calmetrics-perf-followup-deps PIP_NO_INDEX=1 CMAKE_BUILD_PARALLEL_LEVEL=3 MACOSX_DEPLOYMENT_TARGET=15.0 /Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pip wheel . --no-deps --no-build-isolation -Cbuild-dir=/private/tmp/calmetrics-covchol-next/wheel-build -Ccmake.define.CALMETRICS_ENGINE_USE_EIGEN=ON -Ccmake.define.CALMETRICS_ENGINE_USE_ACCELERATE=ON -Ccmake.define.CALMETRICS_ENGINE_USE_HIGHWAY=ON -w /private/tmp/calmetrics-covchol-f-next/wheels
# --no-deps 安装至本轮 install；OFF 改 build-dir、macOS11、三个 OFF 与 wheels-off/install-off。
# PYTHONPATH 指向对应 install，OPENBLAS_NUM_THREADS/OMP_NUM_THREADS/VECLIB_MAXIMUM_THREADS/NUMBA_NUM_THREADS=1。
# ON 三个 CME_EXPECT_* = 1；OFF 三个 = 0。
python -m pytest -q
python -m pytest -q tests/test_covariance_cholesky_reuse.py tests/test_cholesky_f_root.py tests/test_covchol_accelerate.py
# OFF 专项不含需要实际 Accelerate 的 test_covchol_accelerate.py。
ctest --test-dir /private/tmp/calmetrics-covchol-next/wheel-build --output-on-failure
cmake --build /private/tmp/calmetrics-covchol-next/asan-build --parallel 3
ASAN_OPTIONS=detect_leaks=0 ctest --test-dir /private/tmp/calmetrics-covchol-next/asan-build --output-on-failure
ctest --test-dir /private/tmp/calmetrics-covchol-next/off-build --output-on-failure
python tools/check_dist.py /private/tmp/calmetrics-covchol-f-next/wheels
python tools/check_dist.py /private/tmp/calmetrics-covchol-f-next/wheels-off
```

基准复现时将 benchmark.py 和 tools/ 复制到全新临时目录；准备身份匹配的旧新安装，显式指定 --old-install/--new-install。旧实际安装 `/private/tmp/calmetrics-signed-gemm-next/install`；新 `/private/tmp/calmetrics-covchol-f-next/install`。输出 timing 固定放脚本旁 timing 子目录；memory 通过这些已校验的输出 SHA 作事后核对。不要在归档目录或已存在结果上重跑覆盖记录；历史未提交 wheel 留在本机临时目录，未承诺清理后仍能获得旧二进制。

```sh
# 原始严格运行在 scaled=1e8 与 NJIT 不一致时停止，正常8组证据已经保存。
CME_OPT_NEW_BUILD=5d658a12626a65ea524e82d680c17c04b792bc3e1067f194e18cae2a2704ed8f python benchmark.py --output ./timing --mode timing
# 仅显式诊断模式允许已知的 scaled NI 不等价继续计时；不放宽 new/old bits。
CME_OPT_NEW_BUILD=5d658a12626a65ea524e82d680c17c04b792bc3e1067f194e18cae2a2704ed8f CME_COVCHOL_NJIT_DIAGNOSTIC=1 python benchmark.py --output ./timing-scaled-fixed --mode timing --case-filter scaled
CME_OPT_NEW_BUILD=5d658a12626a65ea524e82d680c17c04b792bc3e1067f194e18cae2a2704ed8f CME_COVCHOL_MEMORY=1 python benchmark.py --output ./memory --mode memory --heap-probe /private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib
```

原严格计时和12内存子进程使用 benchmark-before-njit-diagnostic.py 及 tools-before-diagnostic/check_large_matrix_performance.py（内存运行时 wrapper 已增加诊断开关，但未开启；测量本体相同）。首次诊断计时使用 benchmark-before-evidence-fix.py 和该原工具，当前完成文件已写 false；最新 benchmark.py/tools 仅把严格状态提前到每组记录创建时，修复中途异常可能遗留 true 的问题，复测输出为 timing-scaled-fixed。

两个工具版本的测量、CPU 限制、独立结果与计时范围相同，SHA 分别随文件清单保留。基准外对9负载/10批次核对 old/new bits、所有新布局搬运字节、全部构建/输入/输出 SHA；内存36阶段无表溢出或未跟踪释放。heap probe SHA 为 `74421dbfff14b56b730553c7b0b0ce9b38a136de053f132069e1b948decd3382`。RSS 覆盖 setup-through-warm 单进程，不能替代 malloc 或整个 worker 树。
