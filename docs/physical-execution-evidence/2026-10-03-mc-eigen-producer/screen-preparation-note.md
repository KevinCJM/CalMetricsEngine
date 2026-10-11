# width8 Eigen producer筛选

只写/tmp：candidate-affine_projection.cpp是当前生产77bb7f快照的精确候选；reference.py是原冻结NJIT参考的未改副本；bench.py是原完整投影driver的副本，只增加mirror身份定位和可选Eigen实际命中断言。原生产与已执行历史driver未改。未运行镜像准备、编译、import、数值或性能。

候选guard：Eigen可用、FE_TONEAREST、非空row、F>0、A>=8、F*A<=16384。没有凭猜测加大N门槛；small是可测诊断，正式阈值待完整测量。非nearest/禁库/窄A/F0/pack超上界继续scalar职责。空A仍只写period零，不读tile。系数左乘、factor升序、initial先入、product具体Array独立舍入，同一个ordered_fold；std::expm1、dense/sparse消费者与输出writer未改。

现有geometry-only transpose无法证明runtime布局，因此所有candidate shape保守预留F*A doubles；实际K资产stride!=1才pack一次execution。pack位于最大workspace的尾部，当前row的tile+3Cstates不会触及该区；同请求static K可以重用，下一请求重新copy。unit资产stride直接借用，可正/负factor stride。

Counters：`vector_elements`累加实际Eigen Array执行的独立asset lanes（不乘组合数，不声称是硬件SIMD指令数）；`algorithm_copy_bytes`计实际一次F*A*8 packing；workspace容量包括reserved-but-unused pack。没有修改生产audit/IPC。

root串行执行顺序：

```sh
sh /private/tmp/calmetrics-mc-projection-integration/eigen-screen/prepare-mirror.sh /private/tmp/calmetrics-mc-projection-integration/eigen-screen/source
```

接着在mirror使用与baseline相同的wheel/compiler/flags/config构建与安装，生成新的完整identity manifest。`verify_identity.py`使用`CME_MATMUL_REPO_ROOT`指定mirror；不要复用baseline manifest。这里不提供第二套未经确认的构建参数，以root当前可复现构建命令为准。

完整bench默认安装路径可用--install覆盖，示例（占位identity路径由root填实际新manifest）：

```sh
CME_SCREEN_SOURCE_ROOT=/private/tmp/calmetrics-mc-projection-integration/eigen-screen/source CME_SCREEN_EXPECT_EIGEN=1 /Users/chenjunming/Desktop/myenv_312/bin/python3.12 /private/tmp/calmetrics-mc-projection-integration/eigen-screen/bench.py --install /private/tmp/calmetrics-mc-projection-integration/eigen-screen/install --identity-manifest /private/tmp/calmetrics-mc-projection-integration/eigen-screen/identity.json --n 2000000 --steps 200 --factors 4 --assets 32 --combinations 1 --reference shared --cpu 1 --rounds 6 --output /private/tmp/calmetrics-mc-projection-integration/eigen-screen/large-one.json
CME_SCREEN_SOURCE_ROOT=/private/tmp/calmetrics-mc-projection-integration/eigen-screen/source CME_SCREEN_EXPECT_EIGEN=1 /Users/chenjunming/Desktop/myenv_312/bin/python3.12 /private/tmp/calmetrics-mc-projection-integration/eigen-screen/bench.py --install /private/tmp/calmetrics-mc-projection-integration/eigen-screen/install --identity-manifest /private/tmp/calmetrics-mc-projection-integration/eigen-screen/identity.json --n 2000000 --steps 200 --factors 4 --assets 32 --combinations 8 --reference shared --cpu 1 --rounds 6 --output /private/tmp/calmetrics-mc-projection-integration/eigen-screen/large-eight.json
```

identity helper来自原latent阶段memory.py，screen driver在load后仅将其ENGINE指向明确mirror；保留source全部hash、actual_names和实际native SHA验证，以及before/after冻结核验。另记录driver/helper/candidate SHA；不修改原helper或原bench。`CME_SCREEN_EXPECT_EIGEN=1`用于应命中的大case；fallback case不设此标记。

先完整C1/C8筛选，再决定是否投入专项回归。初次stdlib/Eigen差分需四舍入模式（3种fallback）、C/F/padded/negative K/X/initial、F0/A0/emptyouter、NaN/Inf/0×Inf、asset7/8/9/15/16/17尾块、packcap以下/恰等/超上界、多个interval不同tile容量和静态input同shape改内容、readonly/out/scratch canary与独立旧结果。source冻结的首fault前缀协议不因healthy投影bench通过而关闭。

当前scope是完整fixed-tape投影消费子链，仍非NAV/fan/seed完整MC。root现有完整链profile已指向compact数学链的大额内存/耗时机会；若Eigen收益不实质，不无限调width或微小case。
