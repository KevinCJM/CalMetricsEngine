# V61：固定轮数HMM检查点与成熟DGEMV

2026-10-07，已接入当前未提交源码。只覆盖现有HMM的私有执行优化；整体性能目标继续开放，未提交、推送或部署。

## 实现与数学

零容差且N≥4096、16≤K≤128、1≤D≤32、NK≥8,388,608的HMM，在图未选择更便宜的免费完整history借用时，可以只保留一块NK：完整密度校验后原地转为beta，保存每128行的alpha检查点，再重放128行alpha与136行density；xi仍按时间升序累加，最后一个消费者结束后才将beta覆盖为gamma。工作区为 `NK+4K²+4K+512+ceil(N/128)*K+265K` doubles。原始log概率下限、gamma、中心化M步和严格停止不变。正容差与K8–15保留原路线；Planner不知道运行时容差，继续保留完整history包络及较短区间上界，不能按紧凑工作区降低未知参数的预算准入。

Apple arm64/macOS15+、Accelerate开启、非scalar、nearest且有界数据准入时，前向行按 `exp(alpha-max)` 与转移缓存做DGEMV，后向行做相应矩阵向量乘法，复用既有vForce。缓存是 `exp(log(max(A,1e-12)))`，不重新归一化；logA须在[log(1e-12),0]、行操作数绝对值不超过1e8，不适用行在写入前回到有序log数学。仅固定轮数使用该浮点重排，既定绝对误差预算0.000049及离散/错误断言不变。scalar/OFF/非nearest可使用相同检查点存储，但不执行新的BLAS路线；没有对应平台性能承诺。

beta边界、最后一行和xi/gamma生命周期经过两次脏缓冲重放核验。密度重放的136行保留原8行分组和全局尾部。全量校验和似然检查顺序保持，正容差不执行新因式分解；Planner形状输入不读不存在的第七参数。graph/registry共用完整history尾部函数，先比较已驻留loan的真实成本。直接Python audit补公开已有BLAS计数，worker/IPC不新增字段。缓冲区及根只属于本次执行，不写输入或旧结果。

## 调研与未采用证据

参考[hmmlearn原生前后向](https://raw.githubusercontent.com/hmmlearn/hmmlearn/main/ext/_hmmc.cpp)、[Netlib DGEMV契约](https://www.netlib.org/lapack/explore-html/d7/dda/group__gemv_ga4ac1b675072d18f902db8a310784d802.html)和[HMM检查点研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC2430973/)。采用方法及已有Accelerate，不引入hmmlearn依赖，也不移植其下限/停止/错误行为。设计在代码前形成，见records/design*.md。

- A仅检查点：7.6876→9.0431秒，减少约63MiB但慢17.6%，未采用。
- B/C成熟DGEMV：完整时间出现秒级改善，继续资格测试，没有仅凭E步微基准接入。
- 初轮新测试发现缺直接audit键，修复实际serializer；D另发现K8百万行LL差7.23088e-5，保留失败并拒绝K8准入，没有修改0.000049断言。
- plan()不接受parameters；补充测试改用真实API，检查未知容差完整包络、短区间和prepared两代快照。OFF原批814+失败1，修复后单项通过，不能把原批改记815全绿。
- 下一矩阵研究DSYEVR仅记录方法；未实现或测量，迭代/错误契约仍待解决。

## 同源资格与实际接入

normal完整Python16284通过、SAN相关815通过，各native9/9；OFF原批814通过，唯一测试构造错误修复后单项1通过，native9/9。native新辅助头含560种几何×两次脏重放，Scalar/Auto、C/F/负步长/补边、独立alpha/beta/xi参考和canaries。Python新增24项，包括大形状、严格停止邻点、K8拒绝、动态容差/短窗口、四运输和独立所有权。实际目录另58项和native9/9通过。完整normal XML、SAN/OFF正负XML和JSON都保留在records。

actual/normal build ID `69be4ac9b4a216a212a3b4e4b0b20d8fc2cff1f37585817287dbe6d9f3bcff72`，93份source/config相同；SAN/OFF的构建flags/identity另列。copy前375原文件逐SHA保护，本轮只移入6份C++、修改1份native主测试、添加1份native辅助头和1份Python测试。qualified-source保存93份数值/build文件及本轮测试/参考为文本。identities含wheel/native/worker摘要；某些build缓存后来复用，历史identity路径不是现时原生测试证明。

## 完整请求性能

CPU1，输入准备、ordinary执行和独立输出全部计时；保留两代结果，编译/fixture生成排除。轮换启动顺序，同一几何同算法/固定两轮、相同C/F数据。full8不是全部平台情景服务：输出为4模型投影、3诊断及独立sum。

| N-D-K-layout | fresh组数 | V60秒 | V61候选秒 | 暖NJIT秒 |
| --- | --- | --- | --- | --- |
| 262144-8-32-F | 3 | 7.6095 | 5.1340 | 18.8631 |
| 524288-8-16-C | 3 | 3.8558 | 2.8638 | 10.9179 |
| 1048576-8-16-C | 1 | 7.7118 | 5.7358 | 21.7175 |
| 262144-32-32-C | 1 | 7.8399 | 5.5229 | 22.5584 |

实际目录fresh确认5.1415秒，对候选全输出逐字节相同，对NI满足原绝对0.000049。候选最终180个输出比较最大误差2.94987112e-05，离散精确。额外N/D数值筛选在其他测试并行时执行，只记数学资格，不记时间资格。13输出N500000/D4/K12/4轮固定及正容差完整链保留旧更便宜路线，约3.70/4.23秒，毫秒差异不作为收益。

| 完整8输出训练内存 | V60 | V61候选 | 暖NJIT | 实际V61重建 |
| --- | ---: | ---: | ---: | ---: |
| requested warm peak bytes | 151116912 | 84583792 | 285269146 | 84583792 |
| fresh RSS增量bytes | 134889472 | 68321280 | 268763136 | 68403200 |

各backend独立fresh子进程、三次请求的完整准备/输出分配，malloc probe table_overflows=0。RSS只按进程基线增量，采样与厂商工作区盲区保留；不能与requested相加。audit仍记真实检查点/beta搬运，不是全算法零复制；混合prepared工作负载可保留更大的先前scratch。三组/单组实测不冒充全部几何、跨平台或完整统计联合验收。

## 复现与证据

records保存设计、实现/构建/测量脚本文本、命令、日志、JSON、XML及所有失败；qualified-source的.cpp/.hpp/.py均以.txt归档，避免CodeGraph把历史实验当当前代码。array-artifacts.json保存逐项比较过的完整NPZ哈希与大小，不把大数组或构建缓存提交进项目。manifest.json覆盖归档文件，最终文档/CodeGraph与保护核对另保存在closeout.json。

```bash
python -m pytest tests/test_hmm_alpha_checkpoints.py tests/test_hmm_state_lanes.py -q
ctest --test-dir <当前原生build> --output-on-failure
# 先关闭构建/测试任务，再用records中的train_benchmark.py.txt还原脚本执行：
python train_benchmark.py --backend cpp --install <已核对identity的安装目录> --n 262144 --d 8 --k 32 --iterations 2 --layout F --rounds 1 --metric time
```

SAN使用实际ASan运行时，OFF关闭成熟库；全量/相关命令及flags见records/run_quality_final.py.txt、run_quality_qualified.py.txt和identities。benchmark直接导入冻结的引擎测试参考，不运行时依赖平台源码。没有新的真实Microsoft Excel重算；逻辑公式和预算未改变，当前大NK本就高于默认200万格。公开AST/DAG、算子、dtype/布局/所有权与CPU预算未变。

重要MatMul退化、其他主要平台链、正容差提速和全面分配/复制审计继续开放；本轮不关闭整体目标。
