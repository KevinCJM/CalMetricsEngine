# V68: private coefficient layout, contiguous packing and LU certificate optimization

2026-10-08 candidate milestones; R5 actual replay closed 2026-10-09 local time. R1–R4 were isolated against V66 199d1135. Qualified R5 a87f2c25 is now in the actual numerical source, wheel and worker; no commit/push, platform migration or goal completion.

## Completed facts

- R1 637547d4: native9 passed; Python1103 passed/4 exact-copy-counter assertions failed after mistakenly counting logical concat output as algorithm scratch. Failure and changed source retained.
- R2 22e2d84b: corrected original counter semantics. Initial new-test import collection failure retained; relative import correction changed no engine bytes. Same-source native9 plus1115 related tests pass. All56 fresh complete measurements pass. Forced F layout produced unstable C-input gains; exact heap remained equal.
- R3 c6b9366e: choose F only for positively classified F-dominant persistent leaf bytes, keep C construction for C-dominant/unknown/reverse/broadcast/tied layouts and non-nearest rounding. Four new mixed/padded-layout tests initially used the wrong helper; original test/four failures retained. After correcting only the helper, native9 (same binary reused) and1119 related tests pass. Three rotated groups complete48 fresh timing processes, plus12 heap and12 incrementalRSS, with4 separate smokes.

| actual dimension / input layout | actual199d seconds | R3 seconds | strongest warmed NI seconds | R3 heap bytes | NI heap bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
|4112 C|0.740450|0.719196|0.587431|288293620|406531696|
|4112 F|0.766331|0.742844|0.577349|288326516|406531696|
|8192 C|5.240699|5.178219|5.517239|1103672500|1612008496|
|8192 F|5.395891|5.020614|5.408482|1103738036|1612008496|

Timing is the median of three process medians, each of three measured calls; this is not six formal paired groups. Every request includes the same independent readonly owned input preparation, full graph/library work and result return with two retained independent generations. Both direct-final F and C NI fillers are warmed and timed; their stronger layout-specific result determines the separate memory reference. CPU budgets and library thread caps are one; compilation, fixtures and post-epoch full parity are excluded equally. Residuals, all outputs, input hashes, old output hashes, nonalias and source/binary/driver/probe guards pass. Max full error3.228528555609955e-13; one library call and zero numerical fallbacks/callbacks/request compilation per eligible large request.

R3 F8192 gains6.95percent against actual and7.17percent against this epoch's strongest NI. It removes512MiB of counted private coefficient transpose, while exact measured heap remains equal to actual. Existing~508MB heap advantage over NI is retained, not newly created by R3. C construction remains physically unchanged; small timing differences are not claimed as a new C algorithm speedup. Medium4112 remains22.4percent C/28.7percent F slower than NI, so overall objective remains open. Do not mix R2/V67 NI times into the R3 ratios.

## Next measured candidate

R4 fe574461 was built after the R3 timings finished; native9 and23-file Python1119 qualification pass. It packs the existing64×N certificate |L| rectangle in column-major order, reads the F factor contiguously and calls the same FP64 DGEMM with matching NoTrans/leading dimension. Formula, positive error bounds, output precision, ownership and64N workspace size stay unchanged. R4 complete performance has finished:60 fresh timing processes plus12 heap and12 incrementalRSS. CPU/wall ratios are about1 for all variants. Median4112 C/F times are0.663301/0.676095s against strongest NI0.557422/0.505286s. Median8192 C/F times are4.630489/4.465385s against strongest NI5.600570/4.514631s. F8192 gains7.75percent versus actual, but only1.09percent versus NI in this epoch; the incremental packing benefit is limited and medium still regresses. Exact heap equals R3/actual; NI heap remains1612008496B at8192. This is insufficient for adoption or goal completion. Preserve the R3 epoch separately rather than reusing its faster-looking NI ratio. Full/SAN/OFF and actual source rebuild/replay are still required before any candidate adoption.

[Design and strict boundaries](r3/design.md), [complete numerical comparisons](comparison.json), [primary library research](library-research.json), [baseline/reproduction boundary](reproduction.json). Standard mixed-precision DSGESV was researched but not substituted: its simultaneously live float matrix adds about256MiB at8192 and it does not supply the engine's original error certificate. No blanket mixed precision or change to singular/error contracts is authorized by that research.

Normal cached/prepared, isolated/scoped/shared/public coefficient or later-consumer paths retain their existing admission/execution behavior. No AST/IR/operator/Planner resource contract, transport schema, public result layout or Excel formula/budget change. No Microsoft Excel recalculation is claimed. At the R1–R4 stage, major platform chains, longK/multi-use matrices, full copy/allocation audit and actual adoption remained open. R5 adoption and repeated actual performance are closed below; the broader chains and audit remain open.

## R5：已进入实际代码，完整复验通过

R5 a87f2c25保留完整FP64 LU、原主元/错误路径归纳、原方程全残差及canonical回放。以限定常规FP64 GEPP的保守componentwise后向界初始化局部误差，删除原LU重构/QR和一次完整源扫描；gamma(8N)为本项目推导的保守界，不能自动推广至Strassen、降低精度或新后端。成熟库调度、数学算子、原1e-14奇异边界、独立结果及输入只读契约均保留。R3输入驱动C/F与R4连续校验打包纳入同一R5源版本。

完整60个fresh时间进程（三组轮换、每进程三次）、12申请峰、12独立增量RSS及4独立冒烟通过。CPU1，同owned readonly准备、普通图入口、构造/求解/返回与两代独立float64结果；两种预热直接最终构造NI择强，编译/静态fixture及事后检查同样排除。8192 C/F三进程中位为3.535250/3.390467s，强NI为4.844752/4.620900s，省时27.03%/26.63%。申请峰1,097,413,812/1,097,479,348B，NI1,612,008,496B；增量RSS585,793,536/584,187,904B，NI1,077,297,152/1,077,657,600B。相对旧实际源码的本次新增申请峰收益仅约6.26MB，约515MB的NI差额含先前已实现的私有系数复用，不全归因于R5。

4112 C/F为0.459594/0.467424s，强NI0.544536/0.503777s，省时15.60%/7.22%；各三组均领先，但F中型收益仍有限。最大全输出尾差3.228528555609955e-13；全方程、输入/旧代hash、非别名和source/binary/driver/probe守卫通过。合格请求实际一次LU，零回调、数值回退与请求期编译。F私有系数只省内部转置，不要求输入转换；必要owned准备及原方程读取成本真实计入。

同源正常完整Python16,808及native9通过，ASan/UBSan关联5,522及native9通过，OFF关联5,521通过/1项舍入环境skip及native9通过；ASan未启用泄漏检测。各资格文件包含实际命令/测试范围，33文件方案已与脚本核对。2份C++及3份测试已接入实际目录，337份其他受保护文件未变。实际目录重建后93个source/config、191个测试及native/wheel/worker身份与合格候选一致，另native9/关联5,522通过。完整16,808资格按相同源/配置/编译及测试哈希复用，不能称实际目录又跑一遍完整或SAN。

实际目录复验于2026-10-09本地完成：36个时间进程（三组轮换、每进程三次）、8申请峰、8独立增量RSS，共52个完整请求；4个smoke另列。8192 C/F实际中位3.523471/3.362939s，该epoch强NI4.818062/4.578301s，省时26.87%/26.55%；申请峰较NI分别少490.76/490.69MiB，增量RSS少467.02/463.13MiB。4112 C/F省时15.47%/6.67%，中型F收益仍有限。输入准备、完整构造/求解/返回及两代独立结果真实计入；所有输出、原方程、只读、旧代hash、非别名和源/二进制/driver/probe检查通过，合格请求一次LU、零回退/回调/请求期编译。该actual epoch与隔离84测量分别保留，不混用计时。小smoke约1.068ms对NI0.361ms，差约0.707ms只作诊断。

这是完整KKT构造/求解的实际验收，不是完整active-set QP或前沿/随机探索验收。尚无本轮平台迁移、Microsoft Excel重算、提交或推送；其他重要平台链、一般长K/多用途矩阵和完整分配/复制审计仍开放，整体目标未完成。

| 验证 | 当前状态 |
| --- | --- |
| 正常完整Python | passed |
| ASan/UBSan原生及33文件关联 | passed |
| 关闭原生库原生及33文件关联 | passed |
| 实际目录重建原生及33文件关联 | passed |
| 实际目录完整请求复验 | passed |

[详细设计](r5/design.md)、[自审核](r5/self-review.json)、[隔离完整性能](r5/screen.json)、[实际52项完整复验](r5/actual-screen.json)、[实际比较](r5/actual-comparison.json)。
