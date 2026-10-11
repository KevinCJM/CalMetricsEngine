# 串行复现

Repo `/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine`；stage `/private/tmp/calmetrics-solve-pair-integration`。

先构建最终 ON wheel 并装入新的 install：

```bash
export PATH=/private/tmp/calmetrics-perf-followup-deps/bin:/opt/homebrew/bin:/usr/bin:/bin
export PYTHONPATH=/private/tmp/calmetrics-matmul-deps:/private/tmp/calmetrics-perf-followup-deps
export PIP_NO_INDEX=1 CMAKE_BUILD_PARALLEL_LEVEL=3
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
PY=/Users/chenjunming/Desktop/myenv_312/bin/python3.12
STAGE=/private/tmp/calmetrics-solve-pair-integration
MACOSX_DEPLOYMENT_TARGET=15.0 "$PY" -m pip wheel . --no-deps --no-build-isolation \
  -Cbuild-dir=/private/tmp/calmetrics-covchol-next/wheel-build \
  -Ccmake.define.CALMETRICS_ENGINE_USE_EIGEN=ON \
  -Ccmake.define.CALMETRICS_ENGINE_USE_ACCELERATE=ON \
  -Ccmake.define.CALMETRICS_ENGINE_USE_HIGHWAY=ON -w "$STAGE/wheels"
"$PY" -m pip install --no-deps --target "$STAGE/install" "$STAGE"/wheels/*.whl
"$PY" "$STAGE/verify_identity.py" --build /private/tmp/calmetrics-covchol-next/wheel-build \
  --install "$STAGE/install" --wheels "$STAGE/wheels" --output "$STAGE/identity.json"
PYTHONPATH="$STAGE/install" "$PY" -m pytest -q tests/test_solve_pair_reuse.py \
  tests/test_solve_library.py tests/test_solve_grouped_library.py \
  tests/test_single_solve_library.py tests/test_matrix_array_foundation.py
export CME_SOLVE_PAIR_OLD_BUILD=437269d43960c791fd13984fba0051d3df5a72d3dfbe03a8d96414df6aa030c7
export CME_SOLVE_PAIR_NEW_BUILD=295a46460a8c783dd5b764f8eb37b556dd2d1d22997c070308ef4540efe6a3a3
"$PY" "$STAGE/benchmark.py" --mode timing --output "$STAGE/timing"
"$PY" "$STAGE/benchmark.py" --mode memory --case-filter solve_pair_4096 \
  --heap-probe /private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib --output "$STAGE/memory4096"
"$PY" "$STAGE/benchmark.py" --mode memory --case-filter solve_pair_2048x256 \
  --heap-probe /private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib --output "$STAGE/memory-wide"
"$PY" "$STAGE/run_remaining.py"
"$PY" "$STAGE/validate_evidence.py"
"$PY" -m ruff check tests/test_solve_pair_reuse.py
git diff --check
```

本机进程/共享内存和 malloc/RSS 探针需允许相应系统权限。构建、数值、计时、内存互不并发；静态审阅独立。计时8类、各6轮交替、同CPU=1、两个输出都在调用内物化并保留上一独立结果。强NJIT在调用内拼接RHS、一次solve、两份C独立复制；prepared也每次重算，不缓存分解。

Fresh memory 4096 C/F/快照与2048宽RHS，共24个独立子进程（4负载×3后端×heap/RSS）；RSS各3次。既有heap连续epoch、5次热身与3次保留结果替换保持；输入及NJIT编译按脚本明确口径计量。探针SHA256 `74421dbfff14b56b730553c7b0b0ce9b38a136de053f132069e1b948decd3382`。不要将表外释放、Python arena、mmap/栈/厂商缓存盲区记为零。

run_remaining依次运行全Python、ON原生，再运行ASan/UBSan、三库OFF wheel/原生/5文件专项、ON/OFF源与包身份、185直接夹具、历史Excel快照和包内容。ASan部署macOS11、RelWithDebInfo/PythonOFF，`detect_leaks=0`；未宣称厂商库插桩或LeakSanitizer。Excel只复核已重算的469例/3056值同公式快照，没有新桌面重算。

首次构建和 focused.log 保留15项复制次数失败：默认auto schema未准入，而进程已解析C的schema能准入。所有数值检查通过；改为排除显式F并在运行期验证实际C后重新构建，focused-final.log为正式专项。初版wheel/identity另存wheels-first/install-first/identity-first，不用于最终验收。

脚本和日志不覆盖既有输出；重跑应选新的stage/输出位置并核对实际新build ID。构建身份不是Git SHA，未提交/推送。
