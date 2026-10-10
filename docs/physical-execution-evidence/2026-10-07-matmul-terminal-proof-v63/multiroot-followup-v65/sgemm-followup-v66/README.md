# V66：有界私有SGEMM已接入当前代码

2026-10-08，采用恢复完整数组的R2，3份C++与76项回归已进入当前工作区。源码/wheel/worker身份199d1135…一致，保护326份其他文件；[实际接入与验证](array-lifetime-r2/actual-adoption.json)闭合本路径。整体目标未完成，源码未提交、推送或部署。下方R1数据保留原版本范围，不代替R2验收。

## R1 已核验事实

- [正常全量](qualified-r1/full-qualification.json)：16,778/16,778、原生9/9；[SAN](qualified-r1/asan-qualification.json)：4,807/4,807与9/9，未启用LeakSanitizer；[OFF](qualified-r1/off-qualification.json)：4,806通过、1项已明确的macOS舍入环境跳过，与原生9/9。
- [30个完整请求](qualified-r1/measurement-comparison.json)：8192×1024乘1024×8192、两个原逻辑根、两代独立float64结果、CPU1、同数据和强预热NJIT。全部输入准备、分配、复制、调度、执行、结果返回纳入计时；编译、fixture、图构造、参考核验不计。三组轮换fresh时间进程，申请峰和增量RSS另测。

| 布局 | V65秒 | R1秒 | NJIT秒 | R1比NJIT快 | C++申请峰字节 | NJIT申请峰字节 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C | 0.759191 | 0.471418 | 0.645631 | 26.98% | 2,288,002,088 | 2,281,702,136 |
| F | 0.750185 | 0.464163 | 0.643283 | 27.84% | 2,288,002,088 | 2,281,702,136 |

申请峰与V65持平；本负载没有显著NJIT内存优势，不把页级RSS变化当大额节约。全部单元、跨根/跨代独立所有权、只读输入与旧结果hash通过，最大绝对尾差4.757864e-9；完整误差预算仍为0.000049。

[外部调用拦截](qualified-r1/real-sgemm-smoke.json)证明安装后的真实图路径执行SGEMM，未把拦截器计入性能。大负载float打包与double扩展合计603,979,776字节，已计入完整耗时；并非算法零复制。[30个异常检查点试验](qualified-r1/checkpoint-probe-normal.log)覆盖C/F、输入/打包/库调用/扩展、canary、线程恢复；它不等于标准语义的证明。

R1源码/wheel/worker身份全部核对：正常15a38d4d…、SAN fb416116…、OFF aa0ee05a…。原始共享内存sandbox拒绝、所有权测试oracle修复及原始失败XML全部保留；修正后完整回归无失败，不把旧的failed qualification.json改成通过。

## R2 补强和后续

审核发现逐元素placement恢复不足以清晰声明完整数组的可索引对象及新指针。修订版先非分配placement构造完整double数组，保持GEMM字节不被初始化覆盖；仅经character表示读取原float字节，倒序赋值，使用new返回的数组指针。异常也先恢复完整数组再填零。原布局、图资格、所有权、线程预算、误差与回放规则不变。

依据为[C++17草案的basic.life、basic.types和初始化规则](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/n4659.pdf)及[非分配placement array的CWG2382修正](https://cplusplus.github.io/CWG/issues/2382.html)。这属于本项目的具体生命周期分析，不将运行测试当作所有标准语义的证明。新checkpoint probe改用byte-owned原始存储并检查返回的新数组指针；原probe保留为R1历史证据。

[R2正常全量](array-lifetime-r2/full-qualification.json)16,796项全部通过；正常/SAN相关各4,807项和原生各9/9，OFF相关4,806通过/1项既定环境跳过与原生9/9；三模式另各18项边界通过。[60个检查点](array-lifetime-r2/checkpoint-qualification.json)覆盖正常/SAN两模式的输入、打包、库后、倒序恢复、canary及线程恢复，未启用LeakSanitizer。SAN/OFF是同源专项，并非全量或跨平台性能证明。

[隔离30个完整请求](array-lifetime-r2/measurement-comparison.json)三组轮换C/F为0.475823/0.489458秒，强NI0.659750/0.645250秒；比NI快约28%/24%，比同轮V65快约39%/36%。[实际目录资格](array-lifetime-r2/actual-qualification.json)另执行原生9/9与4,825项相关；全量16,796项通过源码、compiler、测试冻结与身份核对复用，不称再次actual全量。

[实际20个完整请求](array-lifetime-r2/actual-measurement-comparison.json)由12个轮换计时、4个malloc与4个RSS进程组成；每计时进程预热后3次，申请峰/RSS各3代。数据、数学、输出与CPU1条件沿用上述完整驱动；包含双方相同128MiB独立只读输入准备、普通入口、真实分配/复制、执行、返回及两代独立结果。NJIT只算一次BLAS乘积、融合原尾链、复用必需输出，签名已预热；fixture、图构造、JIT、编译、测后核验排除。

| 布局 | 实际C++秒 | 同轮NJIT秒 | C++耗时减少 | C++申请峰字节 | NJIT申请峰字节 | C++/NI增量RSS字节 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| C | 0.487038 | 0.643972 | 24.37% | 2,288,002,088 | 2,281,702,136 | 2,233,974,784 / 2,227,240,960 |
| F | 0.474931 | 0.686609 | 30.83% | 2,288,002,088 | 2,281,702,136 | 2,233,909,248 / 2,227,273,728 |

三组时间均快于同轮NI；全部单元、输入、跨根/跨代所有权与旧结果hash通过，最大绝对尾差4.757863492077519e-9。申请峰与V65相同，约6.3MB的NI差及页级RSS波动如实记录，不能宣称本负载明显少于NI内存。四个必需512MiB结果加128MiB输入已占2.125GiB，不能覆盖旧代或改变float64输出来虚构节约。

OFF第一次18项补测缺少模式环境标识，8通过/10失败；相同源码/容差和正确OFF环境复跑18项全部通过，[原始错误与修正](array-lifetime-r2/boundary-off-config-correction.json)保留。摘要曾取错尾差字段，修正读取max_full_absolute_error后报告4.76e-9，30份原始结果/门槛未改。缓存清理仅删除闲置build中间文件，安装包、wheel、源码及证据保留；三个资格build路径再次ctest需重新构建，[清理事实](array-lifetime-r2/cache-cleanup-latest.json)保留。

复查当前graph.cpp确认：GMM私有训练算术保留与状态链tile已在V57/V58实现，不能重新把V56约77.4MiB缺口说成当前缺口。V58历史完整链申请峰834,750,250bytes对NI851,456,928bytes约低2%，尚不足以代表全平台大额优势；需按当前199d身份复验主要负载再选下一修改。重点仍为长K矩阵、一般求解的已知差距以及完整平台链和分配/复制审计，不能把本双根矩阵的提速推广到这些链。总体目标保持开放。

本轮采用同源物理实现；公共算子、公式、IR/IPC与结果类型未改，普通C++请求不执行Excel。512×256×512双根组合的[默认Excel预算预检](array-lifetime-r2/excel-budget-probe.json)返回OVER_BUDGET，未生成/重算工作簿；不称大矩阵Excel验收或将参考入口一概称为canonical。现有默认核验容差与公式纪律未放宽。执行指南更新适用边界；用户/算子/Excel指南无需改写未发生的接口或公式变化。
