# 实际执行与复现

项目根目录：`/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine`；stage：`/private/tmp/calmetrics-covariance-short-integration`。

构建、专项、计时、内存、全量回归与后续构建/检查全部串行，原始日志分别保留。

```bash
export PATH=/private/tmp/calmetrics-perf-followup-deps/bin:/opt/homebrew/bin:/usr/bin:/bin
export PYTHONPATH=/private/tmp/calmetrics-matmul-deps:/private/tmp/calmetrics-perf-followup-deps
export PIP_NO_INDEX=1 CMAKE_BUILD_PARALLEL_LEVEL=3
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
PY=/Users/chenjunming/Desktop/myenv_312/bin/python3.12
STAGE=/private/tmp/calmetrics-covariance-short-integration
MACOSX_DEPLOYMENT_TARGET=15.0 "$PY" -m pip wheel . --no-deps --no-build-isolation \
  -Cbuild-dir=/private/tmp/calmetrics-covchol-next/wheel-build \
  -Ccmake.define.CALMETRICS_ENGINE_USE_EIGEN=ON \
  -Ccmake.define.CALMETRICS_ENGINE_USE_ACCELERATE=ON \
  -Ccmake.define.CALMETRICS_ENGINE_USE_HIGHWAY=ON -w "$STAGE/wheels"
"$PY" -m pip install --no-deps --target "$STAGE/install" "$STAGE"/wheels/*.whl
"$PY" "$STAGE/verify_identity.py" --build /private/tmp/calmetrics-covchol-next/wheel-build \
  --install "$STAGE/install" --wheels "$STAGE/wheels" --output "$STAGE/identity.json"
PYTHONPATH="$STAGE/install" "$PY" -m pytest -q \
  tests/test_covariance_library.py tests/test_covariance_tiles.py tests/test_covcorr_accelerate.py \
  tests/test_covariance_reuse.py tests/test_correlation_accelerate.py \
  tests/test_covariance_accelerate.py tests/test_covariance_f_root.py
export CME_OPT_NEW_BUILD=526ac72ce21b936ff94dd71a1f4078fd9f4a6521115fe8d3755887a526c3a9f1
"$PY" "$STAGE/benchmark.py" --mode timing --output "$STAGE/timing"
CME_CORRELATION_MEMORY=1 "$PY" "$STAGE/benchmark.py" --mode memory --output "$STAGE/memory" \
  --heap-probe /private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib
"$PY" "$STAGE/run_remaining.py"
"$PY" -m ruff check tests/test_covcorr_accelerate.py
git diff --check
```

`run_remaining.py`先运行全部Python、ON原生构建/9项测试，再顺序调用`finish_checks.py`：ASan/UBSan、三库OFF构建/9项原生/7文件专项、OFF身份、185原语夹具、既有Excel快照、ON/OFF包交付、最终77项身份与证据核对。ASan为macOS11/RelWithDebInfo/Python OFF，`ASAN_OPTIONS=detect_leaks=0`；ON Release macOS15、OFF Release macOS11。不宣称厂商库插桩或LeakSanitizer。

计时普通入口完整物化全部输出；14类各6轮，old/new/NJIT平衡换序，1 CPU。fresh内存3类×3后端×heap/RSS，共18个子进程；heap含首次执行、既有5次热身及3次保留结果的warm，RSS3次，父子进程串行。既有脚本会拒绝覆盖部分证据，复现须使用新的stage/输出目录并设置对应实际新build ID。

旧包为`/private/tmp/calmetrics-covariance-ordered-strides/install`。探针SHA256：`74421dbfff14b56b730553c7b0b0ce9b38a136de053f132069e1b948decd3382`；历史Excel核验使用原有真实重算快照，没有启动新的桌面重算。
