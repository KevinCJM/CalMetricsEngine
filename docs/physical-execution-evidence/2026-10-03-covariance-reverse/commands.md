# 本轮复现条件

CME根目录；CPython `/Users/chenjunming/Desktop/myenv_312/bin/python3.12`。编译、回归、时间、内存任务串行。stage `/private/tmp/calmetrics-covariance-reverse`，旧安装 `/private/tmp/calmetrics-covcorr-final/install`；源码身份见baseline/identity JSON，不用当前main替代旧wheel。

```bash
export PATH=/private/tmp/calmetrics-perf-followup-deps/bin:/opt/homebrew/bin:/usr/bin:/bin
export PYTHONPATH=/private/tmp/calmetrics-matmul-deps:/private/tmp/calmetrics-perf-followup-deps
export PIP_NO_INDEX=1 CMAKE_BUILD_PARALLEL_LEVEL=3
PY=/Users/chenjunming/Desktop/myenv_312/bin/python3.12
STAGE=/private/tmp/calmetrics-covariance-reverse
MACOSX_DEPLOYMENT_TARGET=15.0 "$PY" -m pip wheel . --no-deps --no-build-isolation \
  -Cbuild-dir=/private/tmp/calmetrics-covchol-next/wheel-build \
  -Ccmake.define.CALMETRICS_ENGINE_USE_EIGEN=ON \
  -Ccmake.define.CALMETRICS_ENGINE_USE_ACCELERATE=ON \
  -Ccmake.define.CALMETRICS_ENGINE_USE_HIGHWAY=ON -w "$STAGE/wheels"
"$PY" -m pip install --no-deps --target "$STAGE/install" "$STAGE"/wheels/*.whl
PYTHONPATH="$STAGE/install" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$PY" -m pytest -q
ctest --test-dir /private/tmp/calmetrics-covchol-next/wheel-build --output-on-failure
export CME_OPT_NEW_BUILD=27dd5ead862e95ffdfe7bf9704b11716de0495d9939f521ec2dab3630f1fc79d
"$PY" "$STAGE/benchmark.py" --output "$STAGE/timing" --mode timing
CME_CORRELATION_MEMORY=1 "$PY" "$STAGE/benchmark.py" --output "$STAGE/memory" --mode memory \
  --heap-probe /private/tmp/calmetrics-reaccept-20260929/heap_probe.dylib
"$PY" "$STAGE/finish_checks.py"
git diff --check
```

实际首批脚本在5个配对案例完成后，单输出检查把ndarray误当输出列表而失败。diagnostic保存原脚本、错误与前5组结果；只用已有output_arrays统一检查形状，无生产修复或容差修改。最终脚本支持全部13组；实际续跑 `CME_REVERSE_REMAINING=1` 至timing-rest，完成余下8组，timing-all.json按原记录串接，未覆盖已完成数据。重新复现可直接跑修正后的13组，必须使用新的输出路径并调整证据验证位置，不覆盖冻结目录。

finish_checks.py在上述工作全部结束后，顺序运行最终专项、ASan构建/9项原生、三库OFF wheel/9项原生/专项、ON/OFF包身份、185直接夹具、历史Excel快照与证据检查；命令参数在脚本内完整保留。ASan复用RelWithDebInfo、Python OFF、三库ON、macOS11目录，detect_leaks=0；OFF Release/macOS11，ON Release/macOS15。不宣称LeakSanitizer或vendor库插桩。

heap probe SHA256 `74421dbfff14b56b730553c7b0b0ce9b38a136de053f132069e1b948decd3382`。native source/wheel/extension/worker、输入/输出hash及线程审计见JSON。历史Excel核验依赖原重算快照，不能称本轮应用重算。大wheel、xlsx、npz与build目录不复制到仓库。


## 最终有符号地址修复

上述测量原样来自27dd5e候选。最终stage为 `/private/tmp/calmetrics-covariance-reverse-final`，构建 `c9a5ef102069a8942e2e8a06261140b14f278d45f988f779db0000df881b045c`；采用相同ON/OFF构建参数重新打包。`verify_text.py` 比较修复前保存的算子.o以及独立安装的两个扩展/worker的完整__TEXT,__text机器字流，三个均逐字节相同。原采样对象在重建前记录，旧安装保持独立未替换。

`verify_final_requests.py`在最终安装环境运行全部13组两次完整普通请求（保留上一份独立结果），比较原实测候选的完整输出hash、输入hash、重要审计和新构建身份。最终重新跑全量pytest、ON原生，然后顺序运行本目录finish_checks.py的专项/ASan/OFF/夹具/Excel/包身份验证。最终validation.json明确区分原计时/内存与最终运行证据，未把机器码验证当成新的计时。原始正负成绩及追踪限制原样保留，未为相同机器码重复计时/测内存。
