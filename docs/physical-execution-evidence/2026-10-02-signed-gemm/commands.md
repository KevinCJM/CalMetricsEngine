# 本轮执行口径

所有构建、测试、时间和内存进程由主代理串行执行；只读代理不运行数值。工作目录为 CalMetricsEngine，Python 为 `/Users/chenjunming/Desktop/myenv_312/bin/python3.12`。下列路径是本机本轮证据，不是发布环境。

```sh
# ON：复用已配置的 Release/macOS15 构建目录；三库开关均ON。
env PATH=/private/tmp/calmetrics-perf-followup-deps/bin:/opt/homebrew/bin:/usr/bin:/bin PYTHONPATH=/private/tmp/calmetrics-matmul-deps:/private/tmp/calmetrics-perf-followup-deps PIP_NO_INDEX=1 CMAKE_BUILD_PARALLEL_LEVEL=3 MACOSX_DEPLOYMENT_TARGET=15.0 /Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pip wheel . --no-deps --no-build-isolation -Cbuild-dir=/private/tmp/calmetrics-covchol-next/wheel-build -w /private/tmp/calmetrics-signed-gemm-next/wheels
# wheel --no-deps 安装至本轮 install 后，使用 PYTHONPATH 指向该安装目录。
# 强制ON专项的期望开关：CME_EXPECT_EIGEN=1 CME_EXPECT_ACCELERATE=1 CME_EXPECT_HIGHWAY=1。
python -m pytest -q
python -m pytest -q tests/test_nonnegative_matmul_accelerate.py tests/test_matmul_transpose_direct.py tests/test_matmul_packed_library.py
ctest --test-dir /private/tmp/calmetrics-covchol-next/wheel-build --output-on-failure
# Sanitizer 为三库ON、BUILD_PYTHON=OFF、RelWithDebInfo/macOS11。
cmake --build /private/tmp/calmetrics-covchol-next/asan-build --parallel 3
ASAN_OPTIONS=detect_leaks=0 ctest --test-dir /private/tmp/calmetrics-covchol-next/asan-build --output-on-failure
# OFF：同pip命令改用 off-build、wheels-off、macOS11，并显式传入三个开关。
# -Ccmake.define.CALMETRICS_ENGINE_USE_EIGEN=OFF
# -Ccmake.define.CALMETRICS_ENGINE_USE_ACCELERATE=OFF
# -Ccmake.define.CALMETRICS_ENGINE_USE_HIGHWAY=OFF
# 安装到 install-off 后，三个 CME_EXPECT_* = 0，运行相同专项及OFF CTest。
```

回归及时间/内存对照进程在启动前将 OPENBLAS_NUM_THREADS、OMP_NUM_THREADS、VECLIB_MAXIMUM_THREADS、NUMBA_NUM_THREADS 设为1；性能脚本另设置 MKL_NUM_THREADS=1。性能本体不使用env强制库功能开关；实际路由通过native audit验证。

```sh
CME_OPT_NEW_BUILD=f90ddcc0b42dddba625903abf18e8860303b8e747548dd14e8b2e2d1ed246a57 python /private/tmp/calmetrics-signed-gemm-next/benchmark.py --output /private/tmp/calmetrics-signed-gemm-next/timing --mode timing
CME_GEMM_MEMORY=1 CME_OPT_NEW_BUILD=f90ddcc0b42dddba625903abf18e8860303b8e747548dd14e8b2e2d1ed246a57 python /private/tmp/calmetrics-signed-gemm-next/benchmark.py --output /private/tmp/calmetrics-signed-gemm-next/memory --mode memory --heap-probe /private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib
```

必须先完成timing；memory审计在测量后读取timing中已校验的完整结果hash。old实际安装 `/private/tmp/calmetrics-highway-integration-next/install`；new为本轮install，两个wheel/源/原生二进制的完整SHA在身份JSON中。九轮顺序均衡，结果独立所有权，ordinary公开包装和必要布局转换在计时内；NI编译和预热在计时外。

heap probe SHA256 `74421dbfff14b56b730553c7b0b0ce9b38a136de053f132069e1b948decd3382`；公共harness和内存工具SHA硬校验在benchmark脚本内。堆与RSS在不同fresh子进程采样；setup、cold及warm保留峰分别报告，不相加。

重现时把脚本及tools复制到新的临时目录，准备所需旧/新安装包并用`--old-install`、`--new-install`指定；输出timing固定放该新目录的timing子目录。不要在冻结证据目录内直接重跑，以免覆写历史证据。原生构建输入仅3项变化，测试和文档差异另见本轮增量；源码身份不等于Git提交身份。
