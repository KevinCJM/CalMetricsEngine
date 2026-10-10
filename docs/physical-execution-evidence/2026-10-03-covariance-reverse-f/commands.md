# 负向 F 本轮复现条件

工作目录 CalMetricsEngine；CPython `/Users/chenjunming/Desktop/myenv_312/bin/python3.12`，stage `/private/tmp/calmetrics-covariance-reverse-f-tiled`。构建、数值回归、计时及内存串行，不并行争用CPU。旧版安装 `/private/tmp/calmetrics-covariance-reverse-f/install`，baseline.json固定77项身份；不以当前main替代。

```bash
export PATH=/private/tmp/calmetrics-perf-followup-deps/bin:/opt/homebrew/bin:/usr/bin:/bin
export PYTHONPATH=/private/tmp/calmetrics-matmul-deps:/private/tmp/calmetrics-perf-followup-deps
export PIP_NO_INDEX=1 CMAKE_BUILD_PARALLEL_LEVEL=3
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
PY=/Users/chenjunming/Desktop/myenv_312/bin/python3.12
STAGE=/private/tmp/calmetrics-covariance-reverse-f-tiled
MACOSX_DEPLOYMENT_TARGET=15.0 "$PY" -m pip wheel . --no-deps --no-build-isolation \
  -Cbuild-dir=/private/tmp/calmetrics-covchol-next/wheel-build \
  -Ccmake.define.CALMETRICS_ENGINE_USE_EIGEN=ON \
  -Ccmake.define.CALMETRICS_ENGINE_USE_ACCELERATE=ON \
  -Ccmake.define.CALMETRICS_ENGINE_USE_HIGHWAY=ON -w "$STAGE/wheels"
"$PY" -m pip install --no-deps --target "$STAGE/install" "$STAGE"/wheels/*.whl
"$PY" "$STAGE/verify_identity.py" --build /private/tmp/calmetrics-covchol-next/wheel-build \
  --install "$STAGE/install" --wheels "$STAGE/wheels" --output "$STAGE/identity.json"
export CME_OPT_NEW_BUILD=315171f9725b2c03070d2f525713f566f56dc2d719c52ca0ec6f0884c040fd94
"$PY" "$STAGE/benchmark.py" --mode timing --output "$STAGE/timing"
CME_CORRELATION_MEMORY=1 "$PY" "$STAGE/benchmark.py" --mode memory --output "$STAGE/memory" \
  --heap-probe /private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib
PYTHONPATH="$STAGE/install" "$PY" -m pytest -q
cmake --build /private/tmp/calmetrics-covchol-next/wheel-build --parallel 3
ctest --test-dir /private/tmp/calmetrics-covchol-next/wheel-build --output-on-failure
"$PY" "$STAGE/finish_checks.py"
git diff --check
```

12类完整普通请求，6轮轮换old/new/NJIT顺序，每种后端各首发两轮；小案例每轮5次，其余1次。图编译和NJIT编译预热在计时外；完整Scheduler执行/结果物化在内，保留上一份结果。内存2048/4096×3后端×heap/RSS共12个fresh子进程。heap包含首次调用、既有5次热身、3次带旧结果的warm峰；RSS3次带旧结果，监测线程预热/idle对照。保留所有负向结果，不用性能门限布尔值覆盖实际差异。

专项646项先于性能执行，finish_checks.py随后顺序执行ASan/UBSan、三库OFF构建/原生/专项、185直接夹具、历史Excel快照、包检查及最终身份与证据一致性。ON Release/macOS15；ASan RelWithDebInfo/macOS11/Python OFF，detect_leaks=0；OFF Release/macOS11/Python ON。没有新Excel应用重算，不宣称vendor库插桩或LeakSanitizer通过。

初次专项18项进程测试失败：测试预期误用本地负F精确对角线，既有IPC实际打包为正向F。修正为对应worker原canonical，并保持原始完整矩阵1e-12检查；生产和容差未变。原失败日志保留。后续扩展已有测试覆盖负F局部重放、payload拒绝、zero与prepared恢复，最终全量重新执行。

探针SHA256 `74421dbfff14b56b730553c7b0b0ce9b38a136de053f132069e1b948decd3382`；源码、输入、输出、wheel/extension/worker身份见JSON。大binary/NPZ/build目录不复制入证据目录。复现时使用新输出目录，不能覆盖冻结证据。

本轮先测未分块版本44c762相对原始c9a5的12类6轮；其计时、原失败和身份完整保留于initial/。发现4096与NJIT仍有明显差距后，仅重排原helper/面板的8特征遍历，无数学/内存契约变化，再按以上命令测试最终315171。最终计时的old是44c762，不是原始c9a5；两批样本不混合，内存也是44c762/315171/NJIT三方fresh比较。
