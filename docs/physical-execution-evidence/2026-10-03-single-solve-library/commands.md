# 本机复现口径

仓库 `/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine`，Python `/Users/chenjunming/Desktop/myenv_312/bin/python3.12`，stage `/private/tmp/calmetrics-single-solve-next`。脚本内旧wheel安装位置是上一阶段路径；迁移环境需按已记录构建身份恢复旧/新wheel，不能把任意当前构建当成旧版。所有编译、数值回归、性能与内存任务串行运行。

```bash
export PATH=/private/tmp/calmetrics-perf-followup-deps/bin:/opt/homebrew/bin:/usr/bin:/bin
export PYTHONPATH=/private/tmp/calmetrics-matmul-deps:/private/tmp/calmetrics-perf-followup-deps
export PIP_NO_INDEX=1 CMAKE_BUILD_PARALLEL_LEVEL=3
PY=/Users/chenjunming/Desktop/myenv_312/bin/python3.12
STAGE=/private/tmp/calmetrics-single-solve-next

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
  "$PY" -m pytest -q tests/test_solve_library.py tests/test_solve_grouped_library.py tests/test_single_solve_library.py

CME_OPT_NEW_BUILD=2e75f2eabe97084aa63ff03be5351437007dab62dc6d855946460d1f24d16666 \
  "$PY" "$STAGE/benchmark.py" --output "$STAGE/timing" --mode timing
CME_OPT_NEW_BUILD=2e75f2eabe97084aa63ff03be5351437007dab62dc6d855946460d1f24d16666 CME_SOLVE_MEMORY=1 \
  "$PY" "$STAGE/benchmark.py" --output "$STAGE/memory" --mode memory \
  --heap-probe /private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib

PYTHONPATH="$STAGE/install" "$PY" "$STAGE/verify_direct_fixtures.py" "$STAGE/direct-fixtures.json"
"$PY" "$STAGE/verify_excel_snapshot.py"
"$PY" tools/check_dist.py "$STAGE/wheels"
"$PY" tools/check_dist.py "$STAGE/wheels-off"
"$PY" -m ruff check tests/test_solve_library.py tests/test_single_solve_library.py
"$PY" "$STAGE/verify_identity.py" --build /private/tmp/calmetrics-covchol-next/wheel-build \
  --install "$STAGE/install" --wheels "$STAGE/wheels" --output "$STAGE/final-identity.json"
"$PY" "$STAGE/validate_evidence.py"
git diff --check
```

这些命令说明本轮构建/验收条件；重跑需使用新输出目录，保留旧证据。ASan构建复用已配置的RelWithDebInfo、sanitizer ON、BUILD_PYTHON OFF、三库ON、macOS11目录。历史Excel验证依赖既有已重算快照，不能替代新Excel应用重算。

heap probe SHA256：`74421dbfff14b56b730553c7b0b0ce9b38a136de053f132069e1b948decd3382`。计时包装脚本、通用工具及内存工具字节保存在本目录；真实输入和结果hash、编译器、wheel、native source hashes见JSON。完整工作区未被暂存、提交或推送。
