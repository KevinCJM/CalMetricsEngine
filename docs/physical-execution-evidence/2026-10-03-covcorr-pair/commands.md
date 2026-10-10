# 本机复现条件

仓库为CalMetricsEngine；Python为 `/Users/chenjunming/Desktop/myenv_312/bin/python3.12`。所有编译、回归、时间和内存任务串行。本轮stage为 `/private/tmp/calmetrics-covcorr-final`，基线安装为 `/private/tmp/calmetrics-correlation-range/install`。不同构建证据分开保存。

```bash
export PATH=/private/tmp/calmetrics-perf-followup-deps/bin:/opt/homebrew/bin:/usr/bin:/bin
export PYTHONPATH=/private/tmp/calmetrics-matmul-deps:/private/tmp/calmetrics-perf-followup-deps
export PIP_NO_INDEX=1 CMAKE_BUILD_PARALLEL_LEVEL=3
PY=/Users/chenjunming/Desktop/myenv_312/bin/python3.12
STAGE=/private/tmp/calmetrics-covcorr-final

MACOSX_DEPLOYMENT_TARGET=15.0 "$PY" -m pip wheel . --no-deps --no-build-isolation \
  -Cbuild-dir=/private/tmp/calmetrics-covchol-next/wheel-build \
  -Ccmake.define.CALMETRICS_ENGINE_USE_EIGEN=ON \
  -Ccmake.define.CALMETRICS_ENGINE_USE_ACCELERATE=ON \
  -Ccmake.define.CALMETRICS_ENGINE_USE_HIGHWAY=ON -w "$STAGE/wheels"
"$PY" -m pip install --no-deps --target "$STAGE/install" "$STAGE"/wheels/*.whl
PYTHONPATH="$STAGE/install" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$PY" -m pytest -q
ctest --test-dir /private/tmp/calmetrics-covchol-next/wheel-build --output-on-failure
cmake --build /private/tmp/calmetrics-covchol-next/asan-build --parallel 3
ASAN_OPTIONS=detect_leaks=0 ctest --test-dir /private/tmp/calmetrics-covchol-next/asan-build --output-on-failure

MACOSX_DEPLOYMENT_TARGET=11.0 "$PY" -m pip wheel . --no-deps --no-build-isolation \
  -Cbuild-dir=/private/tmp/calmetrics-covchol-next/off-build \
  -Ccmake.define.CALMETRICS_ENGINE_USE_EIGEN=OFF \
  -Ccmake.define.CALMETRICS_ENGINE_USE_ACCELERATE=OFF \
  -Ccmake.define.CALMETRICS_ENGINE_USE_HIGHWAY=OFF -w "$STAGE/wheels-off"
"$PY" -m pip install --no-deps --target "$STAGE/install-off" "$STAGE"/wheels-off/*.whl
ctest --test-dir /private/tmp/calmetrics-covchol-next/off-build --output-on-failure
PYTHONPATH="$STAGE/install-off" CME_EXPECT_EIGEN=0 CME_EXPECT_ACCELERATE=0 CME_EXPECT_HIGHWAY=0 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$PY" -m pytest -q tests/test_covcorr_accelerate.py tests/test_covariance_reuse.py tests/test_correlation_accelerate.py \
  tests/test_covariance_accelerate.py tests/test_covariance_f_root.py

CME_OPT_NEW_BUILD=a129d662535bc5e6ebe20f2dc27e4ce7baa7376bfe8d38b23486a0f284d96142 \
  "$PY" "$STAGE/benchmark.py" --output "$STAGE/timing" --mode timing
CME_OPT_NEW_BUILD=a129d662535bc5e6ebe20f2dc27e4ce7baa7376bfe8d38b23486a0f284d96142 CME_CORRELATION_MEMORY=1 \
  "$PY" "$STAGE/benchmark.py" --output "$STAGE/memory" --mode memory \
  --heap-probe /private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib
PYTHONPATH="$STAGE/install" "$PY" "$STAGE/verify_direct_fixtures.py" "$STAGE/direct-fixtures.json"
"$PY" "$STAGE/verify_excel_snapshot.py"
"$PY" tools/check_dist.py "$STAGE/wheels"
"$PY" tools/check_dist.py "$STAGE/wheels-off"
"$PY" -m ruff check tests/test_covcorr_accelerate.py
"$PY" "$STAGE/verify_identity.py" --build /private/tmp/calmetrics-covchol-next/wheel-build \
  --install "$STAGE/install" --wheels "$STAGE/wheels" --output "$STAGE/final-identity.json"
"$PY" "$STAGE/validate_evidence.py"
git diff --check
```

复用的ASan目录已配置RelWithDebInfo、sanitizers ON、BUILD_PYTHON OFF、三库ON、macOS11。重跑须使用新输出路径、恢复匹配的旧/新wheel身份并保留已有证据；不要覆盖原日志。旧安装是上一阶段相关矩阵源范围扩展构建，不是任意当前main。

heap probe SHA256为 `74421dbfff14b56b730553c7b0b0ce9b38a136de053f132069e1b948decd3382`；各输入、输出、源码、compiler、wheel及安装extension/worker身份在JSON。历史Excel验证依赖已有重算快照，不能称为本轮Excel应用重算。大xlsx、NPZ、wheel、构建目录未复制到仓库。

初版完整面板构建与记录保存在initial目录，最终构建将配对C面板限为1024行；两批原始结果不混合。分配探针的untracked free限制在验收正文及validation.json单独披露。
