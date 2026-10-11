# V50：长 K 定位、成熟库归并筛选

2026-10-06，macOS arm64，CPU1。实际源码与扩展/worker仍为 V48，build id 为 `30722fb25c692505f924aec6755d8f3c5a6113507b9b54552fb22a895d8ab365`。93份 native/build 源、168份已验测试及294份其他既存文件 SHA 保持一致；本轮只追加文档与隔离实验记录。没有修改实际引擎、提交、推送或部署。

## 做了什么

用 CodeGraph 核对 `matmul_partitioned.hpp` 的实际树路径，参考[官方库与公开方法](v50-research.md)，编译独立 AOT ABI：原版、F树、原版插桩、DAXPY归并、512×256树块及小块加DAXPY。没有替换 DGEMM 乘法内核或调整0.000049预算。

三轮64组私有布局/stride/数学/保护区/所有权控制及4组实际激活的2051×8192×2057不规则几何通过，共196组；36个零、负零、NaN/Inf及超域拒绝检查通过。这些不是新增 pytest 回归或完整 native 资格。第一次私有链接漏掉既有 Highway 静态依赖，失败日志保留；补齐链接后成功，没有修改实际源。

## 结果

完整数学请求为 M4096、K8192、N4096，固定±1024混合符号数据。计入两个独立只读输入的准备、计算、独立结果返回及两代结果保留；图定义/编译/预热不计时。每批4轮，后端交错顺序，最后重核验另加屏障。候选仍为私有 ABI，实际 V48 Scheduler 只作基线；不能把候选数字当作已经进入真实图的性能。

| 输入/输出 | 原私有版秒 | DAXPY私有候选秒 | 实际V48秒 | 暖NJIT秒 | 私有收益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FF→F | 2.1958 | 1.9150 | 2.2676 | 1.3297 | 12.8% |
| CC→C | 2.4255 | 2.1079 | 2.4215 | 1.3129 | 13.1% |
| CC→F | 2.3679 | 2.0602 | 2.4248 | 1.6701 | 13.0% |
| FF→C | 2.4736 | 2.1848 | 2.4740 | 1.7525 | 11.7% |

FF→F另一批原版/DAXPY为2.1852/1.8960秒，收益13.2%；仍慢于同批NI1.3931秒。两批收益不能与不同批次的基线混算。F树候选为2.4237秒，对照原版2.2038秒，约慢10%；小块候选2.3482、小块+DAXPY2.0839，对照原版2.1852及DAXPY1.8960，均否决。

插桩原版的中位时间：input domain0.0358秒、norm/plan0.0327秒、DGEMM1.3080秒、显式归并0.6815秒、最终复制0.0347秒、finish0.0100秒。calculate为2.0380秒，含DGEMM/归并/复制/finish，不能重复相加。计划chunk64、128片、深度7、9槽、1024×256块，共8192次DGEMM；没有近零回放。这里是私有诊断，不是实际 Scheduler 的精准阶段分解。

六批共108次计时，全部 worker 已终结。完整样本及局限见[v50-summary.json](v50-summary.json)，各批原始stdout、stderr、状态与结果不删样本。旧版、各候选头及基线源码按惯例保存为 inert `.txt`，避免历史代码进入构建或CodeGraph源码索引；没有归档库二进制或新增生产包。

## 还差什么

DAXPY只是保留候选，尚未完成完整引擎自审核、正常/SAN/OFF全量资格、真实Graph/Planner/worker链或fresh峰值内存。显式scratch容量同为20,972,096 bytes，不等于库隐藏分配或总峰值相同。本轮没有大额内存收益证据，四种一般大矩阵仍慢于强NI。下一步按[资格设计](v50-research.md)验证该候选；整体目标保持开放。

## 复现范围

在隔离目录恢复基线93份源码后，以原93份SHA和当前V48构建选项重建静态operators、highway_order与hwy依赖。私有wrap使用C++17/O3、`-fno-fast-math -ffp-contract=off`、macOS部署目标15、原Eigen3.4.0和Accelerate，无请求期JIT或额外线程池。`screen_wrapper-layout.cpp.txt`、`screen_wrapper-merge.cpp.txt`与最终`screen_wrapper.cpp.txt`分别对应三个私有库；库SHA保留在[身份记录](private-library-identities.json)。`screen.py`通过`CME_V50_LIBRARY`选择库，`run_screen.py`完成交错请求和CHECK屏障，`run_layout_controls.py`串行运行其余三种布局。原临时绝对路径是当时的执行环境，不能直接视为可移植部署步骤。
