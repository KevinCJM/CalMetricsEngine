# 本轮复现

CME根目录，CPython `/Users/chenjunming/Desktop/myenv_312/bin/python3.12`。stage `/private/tmp/calmetrics-covariance-ordered-strides`，旧安装 `/private/tmp/calmetrics-covariance-reverse-f-tiled/install`。构建、回归、计时和内存全部串行。

```bash
export PATH=/private/tmp/calmetrics-perf-followup-deps/bin:/opt/homebrew/bin:/usr/bin:/bin
export PYTHONPATH=/private/tmp/calmetrics-matmul-deps:/private/tmp/calmetrics-perf-followup-deps
export PIP_NO_INDEX=1 CMAKE_BUILD_PARALLEL_LEVEL=3
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
PY=/Users/chenjunming/Desktop/myenv_312/bin/python3.12
STAGE=/private/tmp/calmetrics-covariance-ordered-strides
MACOSX_DEPLOYMENT_TARGET=15.0 "$PY" -m pip wheel . --no-deps --no-build-isolation \
  -Cbuild-dir=/private/tmp/calmetrics-covchol-next/wheel-build \
  -Ccmake.define.CALMETRICS_ENGINE_USE_EIGEN=ON \
  -Ccmake.define.CALMETRICS_ENGINE_USE_ACCELERATE=ON \
  -Ccmake.define.CALMETRICS_ENGINE_USE_HIGHWAY=ON -w "$STAGE/wheels"
"$PY" -m pip install --no-deps --target "$STAGE/install" "$STAGE"/wheels/*.whl
"$PY" "$STAGE/verify_identity.py" --build /private/tmp/calmetrics-covchol-next/wheel-build \
  --install "$STAGE/install" --wheels "$STAGE/wheels" --output "$STAGE/identity.json"
export CME_OPT_NEW_BUILD=ecf16686c121d959600299583b9706e67bb337fc5cbaec42ebf09040ba627b2e
"$PY" "$STAGE/benchmark.py" --mode timing --output "$STAGE/timing"
CME_CORRELATION_MEMORY=1 "$PY" "$STAGE/benchmark.py" --mode memory --output "$STAGE/memory" \
  --heap-probe /private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib
PYTHONPATH="$STAGE/install" "$PY" -m pytest -q
ctest --test-dir /private/tmp/calmetrics-covchol-next/wheel-build --output-on-failure
"$PY" "$STAGE/finish_checks.py"
git diff --check
```

实际首批计时使用diagnostic/initial-benchmark.py，完成前5例后，第6例放大16384倍输入触发旧C++/NJIT既有容差不匹配（新旧C++逐位一致）。该失败原样保留；不放宽容差，也不将它作为合格NI计时。后续脚本明确把该例改名为offset_source1024、采用加16384来验证同一准入拒绝分支的普通协方差尺度；使用 `CME_ORDERED_REMAINING=1` 续跑7例到timing-rest。5+7原样合并timing-all，未覆盖或选择重跑已完成计时。最终脚本可在新stage完整重跑12例；失败放大案例通过verify_njit_mismatch.py独立复核，依赖首批保留的old/new.npz与输入/hash记录。复现历史失败时，将冻结初版脚本与tools复制到新的同级stage再运行，不直接覆盖本目录。

12例每例6轮，old/new/NJIT循环先发，每后端两轮首发；小例每轮5次，其余1次。普通Scheduler绑定/规划/执行/结果物化在计时内，编译/预热在外，保留上份独立输出。fresh内存为scaled2048和32769×256 dense replay，三后端各heap/RSS独立进程；cold设置/调用、既有5次热身及3次保留结果的warm，RSS3次。4096本轮没有fresh内存样本。

最终finish_checks.py顺序执行ASan构建/原生、三库OFF构建/原生/7文件专项、185直接夹具、历史Excel快照、ON/OFF包身份及全证据核对。ON Release/macOS15，OFF Release/macOS11；ASan RelWithDebInfo/macOS11/Python OFF，detect_leaks=0。不宣称LeakSanitizer、vendor插桩或新Excel重算。源码/extension/worker身份及探针范围有独立JSON；大wheel、NPZ、xlsx与build不复制进证据。

探针SHA256 `74421dbfff14b56b730553c7b0b0ce9b38a136de053f132069e1b948decd3382`。assembly.json与ordered-strides-assembly.txt来自ON matrix_backend.cpp.o的otool反汇编；静态向量指令证据不等于动态执行指令数量。Ruff发现并修正受影响测试的既有import顺序，未改变算法。
