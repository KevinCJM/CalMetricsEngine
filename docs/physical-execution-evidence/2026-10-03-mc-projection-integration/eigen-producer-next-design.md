# 下一候选：Eigen 独立 asset lanes 的保序 affine producer

2026-10-03。只读设计，生产保持冻结；本代理未运行数值、编译、测试、时间或内存任务。先用 CodeGraph 定位当前生产实现、Planner 入口、canonical fold 与 Eigen backend，再核对构建和预算源段。适用纪律：AGENTS 与 computation-design-rules 的数学契约、D1/D2/D5/D6/D8，以及 Excel X1/X10。

## 1. 目标与已知证据边界

root 提供的已完成测量：dense 主 case 约481ms，对强 shared NJIT 约659ms；原 service NJIT约2074ms。大型单组合约325ms，对 shared NJIT约287ms，仍慢约13%。medium约21.1ms对22.9ms的差异只有约1.8ms，只作诊断。本文不重跑或独立确认这些数字，未从单组合差距推断某一具体函数占比。

下一候选仅加速现有 private closure 的 producer：相同 row 内，沿互相独立的 asset lanes 使用成熟 Eigen Array packet 做乘法/加法。factor仍升序，显式 initial仍是首项，`std::expm1`仍逐元素执行。消费者、dense/sparse links、跨tile贡献左折叠和根输出保持原实现。不是改成 GEMM，也不接 vForce。

收益是假设：减少相同X因子载入与标量 coefficient access，让 affine的F×A运算使用packet。若当前耗时主要在libm expm1，收益上界有限。必须对完整单组合、多组合请求验证；不得以 isolated affine 或“Eigen已链接”称采用通过。

当前冻结身份：

```text
cpp/affine_projection.cpp
77bb7f37ae21abdf065078a2397da0d3ed88029363bc1f3d37f489bb2699c665
cpp/include/calmetrics_engine/affine_projection.hpp
152484867013eb55af3db4616f5764c9b651fd109da145374e81b5f688f41075
cpp/operators/affine_contract.hpp
1afc6798ff05a26e1bf3bca7768083f0e29b57b8925ca15cdd79982d293da38e
```

## 2. 同一个数学体，而非第二套数值实现

`cpp/operators/affine_contract.hpp:9–20` 已是泛型 `ordered_fold`，每个factor具体构造 `const Accumulator product`，然后 `accumulator += product`。可直接以固定 `Eigen::Array<double,8,1>`（另测16而非先永久加多套dispatch）作为Accumulator：

```text
accumulator[lane] = explicit_initial[asset+lane]
for f ascending:
    concrete_product_Array = coefficient_packet[f,asset:asset+W] * X[row,f]
    accumulator += concrete_product_Array
for lane ascending:
    tile[row,asset+lane] = std::expm1(accumulator[lane])
```

- `coefficient_at(f)` 返回只读 Map 或具体固定Array；`value_at(f)` 返回该row/f的double。系数左乘顺序保持。
- product必须具体求值为Accumulator类型；不可返回持有临时Map/Array的悬空lazy expression，也不可写成让后端融合为pmadd的乘加表达式。
- 每个lane的factor链仍为 `initial → +K0*X0 → +K1*X1 ...`。没有水平归约、factor分块小计、零系数跳过或尾项重排。
- 初态scalar用固定Array fill，初态vector按实际signed stride逐lane读取。不能把initial最后加上。
- 完整W lanes走固定Array；余数走同一ordered_fold的double实例。固定Array是栈/寄存器对象，禁止Dynamic Array或一次row一个heap对象。
- 继续逐asset调用std::expm1。Affine后的库向量指数是另一个数值候选，不捆绑进本次采用判断。
- factor空轴返回initial，不形成X/coefficient pointer；asset或外轴为空不读数据、不形成空tile指针。

物理重排只跨互相独立的asset；不改变贡献的step→path累加，也不改变period的asset升序初态0归约。IEEE非有限值继续传播，不提前抛“asset失败”。特别保留 `0 * Inf`，不能利用系数/权重零值跳过。

## 3. 必要的布局处理：只打包小系数，不改大输入/结果布局

FIRP beta是C连续[A,F]；其只读transpose作为K[F,A]时，asset stride通常是F。动态strided Eigen Map不自动具有contiguous packet load，因此“直接Map转置beta”不保证真正SIMD。

候选先分两类：

1. 真实K的asset stride等于1：每个factor的packet可以直接Unaligned Map。factor-row stride可正/负或有padding，由已验证bounds承担安全性；不要求整块K是C布局。
2. 其他合法strides：每次执行将小K按[F,A]打包一次，所有row和所有combo共享。逐值copy保持原bits/轴顺序；不缓存数值内容，不按地址/shape认为系数未变。负stride/padding也通过原只读view访问。

建议首候选pack硬上界128KiB，即最多16384 doubles，overflow-safe检查 `F==0 || A<=limit/F`。真实平台F<=32、A<=64时最多16KiB；这是设计尺寸推导，不是已测工作区。泛化引擎超过此cap继续当前scalar职责路径，不能拆factor小计改变初态/顺序。

不重排或copy全X；不改根结果C/F；不把F布局视为不连续。必要pack是小系数访问优化，需完整计成本。若同请求多个区间共用同一static系数，可在execute本次生命周期内pack一次；无跨请求mutable cache。已有proof `plan_affine_projection:251–260` 只允许static input或transpose(static input)，满足该共享前提。要在全部区间geometry admission完成后开始packing，失败/取消不得留下半次结果被当成功。

## 4. 预算的重要陷阱与最小解决

当前 `bind_geometry:104–118`：numerical=false时，operation仅有shape并设为contiguous strides；numerical=true才对transpose/slice调用prepare取得实际借用strides。因此**Planner不能拿几何pass生成的K stride断言runtime无需pack**。

首候选建议保守预留pack容量：仅根据已知F/A与cap计算，并对eligible shape一律预算F*A doubles；runtime若可直接借用则不copy，但已保留的workspace仍诚实计量。这样不必为小系数布局另建一般stride框架。后续若需要消除reserve，再补不读取payload的真实view-stride传播，并单独审核，不先扩大范围。

现有workspace是 `tile_capacity*A + 3*contribution_count` doubles（`bind_geometry:142–143`）。新候选追加F*A pack reserve，layout建议tile → totals → weights → divisors → pack；offset全部经现有add/mul checked arithmetic。最大workspace在执行前由所有区间计算并一次ensure。fixedArray栈高水位也记录编译确认，不称“没有任何scratch”。

- `affine_projection_workspace_bytes:321–326` 与 runtime `maximum:351–374` 共用同一reservation规则；构建禁Eigen时仍可保守reserve或精确fallback，但两者必须一致。
- Audit workspace报告实际vector.capacity+descriptor容量；algorithm_copy_bytes累加实际pack F*A*8（一次execution，不一次row），直接借用为0。reserved却unused的pack不能从workspace占用扣掉。
- existing descriptor容量计算维持；新增任何shape/path proof字段计sizeof/actualcapacity，不把planned size当capacity。
- 没有额外全R/L、产品矩阵、private pool或mmap；out仍是根独立所有者，输入pin持续到所有任务结束。
- kernel scratch不能借用已返回root或shared input，也不得跨并发任务复用同一个可变pack。

## 5. 准入与确切源码依赖

首候选保留现有strict typed/no scopes/no isolate/no observer whole-closure proof。producer无法准入只使用同一canonical原生scalar路径；消费者不因候选改变。

- 构建：CMakeLists.txt:96–109已有离线校验的Eigen3.4.0归档、PRIVATE include和 `CALMETRICS_HAVE_EIGEN_ROTATION`。affine_projection.cpp同属calmetrics_engine_operators（:45–65），无需新运行依赖或联网构建。
- helper落点：private affine_projection.cpp中固定Array helper；复用affine_contract.hpp模板。若以后direct affine也采用，再移到受控backend并共享helper，不能预先建设无人消费框架。
- 与现有matrix_backend.cpp:8–11一样，include前明确 EIGEN_DONT_PARALLELIZE、EIGEN_MPL2_ONLY、EIGEN_FAST_MATH=0。既有CMake :269 no-fast-math/no-contraction保留。
- 起步只让实际已验证目标平台与ISA使用packet；其他平台、Eigen关闭、small/窄A、超packcap继续现有scalar。不能把Eigen编译时ISA当运行时支持证明。
- 最保守首轮加入FE_TONEAREST guard，其余3种rounding直接scalar并测fallback；不要因为packed add/mul理论同舍入就未经验证扩大域。若未来开放其他round模式，独立冻结逐位证据。
- private graph入口目前没有Isa参数；不得用direct API simd='scalar'声称private graph必走scalar。需要可辨识的实际candidate hit证据（对照构建或真实audit），保持scalar参考职责。
- 已有eigen_matvec/eigen_density计数不应因名字近似被错误复用。采用时最小新增一个eigen_affine计数，贯通graph Audit/binding/merge/IPC，并同步protocol版本；或者使用经明确文档化且语义准确的既有通用计数。只增加“源码编译了Eigen”标签不足以发现runtime hit。

选择A>=8、大N等阈值只作为screen假设；正式准入阈值来自完整请求测量，不能凭本设计固定。先宽度8对照当前dense；若收益不足再查phase成本，不为KB/ms长期调宽度。

## 6. 必须验证的风险与代表用例

数值：factor0/1/4/32；asset0/1/7/8/9/15/16/17/32/64；pack cap以下/恰等/超上界；Fma cancellation witness、NaN payload/±Inf、signed0/subnormal/overflow和0系数×Inf；极小/极大initial；near-zero expm1及−1/target等边界nextafter。每lane维持标量mathbody不自动证明packed NaN payload和异常标志完全等同；须按已声明契约分离字节、IEEE分类和离散结果。

布局/边界：beta C/F、transpose、负factor或asset stride、padding；初态scalar/参数/strided vector；X rank2/3 C/F/padding/负stride；W余数、tile最后1row；empty F/A/outer不可读null；4round模式。无需全笛卡尔，覆盖不同guard与kernel分支即可。

图/所有权：选单period、仅contribution、重复period/contribution、非matching weights、partial/sparse/dense；typed C/F roots；输入不改、旧独立结果跨重复执行保持；workspace/output canary与ASan；预算不足必须在写出前拒绝；prepared相同shape改coeff/initial内容要读本次值。取消/异常/worker后再正常执行不带脏pack。

完整性能：当前dense vs候选 vs等准备、同核数的shared NJIT和真实service NJIT。保留root指定大N2M/F4/A32/C1及C8；再测合法N250k/F32和真实长T用例。不增加核数或减少输出/retained lifetime取胜。必要pack/copy/ordinary planning、实际direct输出字节、heap遗漏限制与RSS补充完整报告，冷热分别保存identity。

采用条件：完整大任务有实质时间收益且无大额内存/预算问题；小任务轻微KB/ms变化不作否决。若large single组合仍实质慢于强NJIT，保留失败样本并继续完整链优化，不用多组合胜出平均掩盖。此候选仅关闭producer性能问题，不代表NAV/fan/随机流/平台迁移完成。

## 7. Excel影响

只有物理lane/小系数packing变化，logical affine/expm1契约和Excel展开不改。按X1/X10复核该等价依据与适用fixture，不在普通C++请求计算Excel或保留导出中间值。若试验导致任何新数学版本、NaN/零/阈值改变，不能以“只是SIMD”豁免formula和canonical版本联动。

当前结论：可准备上述小范围候选；未实现、未编译、未测试，未宣称加速或采用。root决定下一阶段是否实施。
