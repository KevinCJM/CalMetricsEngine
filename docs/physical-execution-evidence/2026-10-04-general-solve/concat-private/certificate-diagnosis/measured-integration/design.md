# Existing terminal solve_multi: measured local LU integration candidate

当前状态（2026-10-05）：6707在既有20cf校验证书上，仅增加Eigen有界布局复制，已接入实际源码并正式重建/验收。数学、校验、错误、所有权和Excel公式不变；设计/依赖/预算/原始正负筛选见[矩形复制自审核](layout-copy/self-review.md)，正式完整链与峰值见[验收](layout-copy/acceptance.json)。全引擎目标未完成；以下按形成阶段保留。

2026-10-04. Original baseline root/wheel fafa; measured tiled7a6f has now been adopted and formally rebuilt after the validation recorded in review.md. Whole-goal acceptance remains open.

Use existing input-backed private concat trees and existing mature LU/BLAS, no public operator or Python runtime logic. The new entry is only for independent terminal typed float64 roots without scopes, observers, isolated/deferred errors, internal prebound execution or input failure. The public prepared snapshot/audit APIs enter the ordinary native executor and can use this path; this is distinct from its internal prebound flag. Existing DD/arithmetic routes stay unchanged.

Preserve the original 1e-14 singular threshold and B-before-A/finite diagnostics with conservative coefficient/RHS/back-substitution induction. The initial local radius is measured P*A-L*U plus a conventional FP64 product roundoff envelope, rather than an assumed vendor LU backward-error envelope. The local shadow induction retains strict pivot separation and 1e-10 admission safety margin; rejection reconstructs the full original C coefficients from persistent input-backed leaves and executes the canonical native solver. No new numerical exception is exposed by admission.

Reconstruct L*U in 512 square blocks, skipping structurally zero L/U blocks, using DGEMM on off-diagonal borrowed factors and masked diagonal blocks. This avoids a second N² allocation. Explicit scratch is N*K+4*N*ceil(N/512)+67*N+3*512² doubles; pivot/permutation vectors and conservative vendor reserve are budgeted separately. All buffers belong to the synchronous call/Workspace; inputs and retained results remain immutable.

Extend the existing concat leaf-copy lowering to checked rank2 rectangles only. Full rank1/2/3 copies retain current behavior. Native synchronous callback+context regenerates coefficient rectangles for both factor and original-equation residuals, and the full C coefficient on replay. No context is retained after the call. C/F/negative/zero strides use the same byte-copy routine.

Mathematical qualification: measured reconstruction removes a LU-algorithm backward-error assumption, but conventional double-precision BLAS product stability remains an explicit model, as in existing guarded products. A backward residual is not a universal forward-error guarantee for ill-conditioned systems. Terminal float roots require finite safe results plus the independent original-equation residual. Four-decimal user precision applies only to floating values; discrete consumers and uncertain pivots replay. No change to logical AST/DAG/Excel formulas, outputs, or errors.

Performance admission: compare same owned readonly input preparation, ordinary Scheduler graph, independent output return and retained generations versus fafa and fully warmed NJIT under CPU1, serial rotating repeated rounds, continuous requested heap and three fresh RSS processes per layout. Include attempted guard/replay work. A substantial gain over fafa is stage progress; any important NJIT slowdown remains explicitly open and whole goal incomplete. The fully reconstructed isolated guard currently costs 0.55s at4112 plus0.24s factor, so no claim of NJIT superiority yet. Pure Eigen factor1.53s and other reconstruction layout variants are negative evidence, not adopted runtimes.

Validation: all existing guarded/private/DD tests, concat rank/stride/geometry/error tests, four original counterexamples, non-DD KKT, safe/library fallback domains, same-source native+binding+ASan/UBSan+OFF, four transports and two generations. Root adoption only after suitable correctness/ownership/full-chain checks. No Git mutations.

## Wide reconstruction experiment

The measured v2 integration passed 490 related tests. CPU1 complete-task repeated medians at4112 are C: fafa2.4474s, v2 .8336s, NJIT .7443s; F: fafa2.4958s, v2 .9441s, NJIT .7589s. These are substantial baseline improvements but do not establish the NJIT time goal. The original isolated peak-memory tests and later actual-root results are recorded separately in review.md.

Keep the same measured initial error, pivot/finite induction, source ownership and replay. Reconstruct one 512-row by N-column factor product per row panel, accumulating each masked diagonal U tile and its borrowed complete upper strip in two DGEMM calls. This replaces many separate output-column tile products with wider mature-library calls; it neither omits entries nor samples residuals. The same conventional gamma(4N) product envelope includes all dot-product terms and block accumulations. Source regeneration now checks 64-row by N-column rectangles in final permutation coordinates.

Explicit scratch becomes N*K+4*N*ceil(N/512)+67*N+2*512²+512*N doubles, with the existing pivot/permutation and vendor reserves. At4112 this adds roughly14.7MB over v2; it is not free memory. Measure the real complete-task peak against both baselines before adoption. The bounded panel has no retained state and does not change public layout, output or Excel formulas. Portability tests must expect canonical execution when native libraries are disabled, while retaining exact errors and excluded-consumer values.

## 后续隔离候选873ac：测量表示与固定Eigen包（尚未接入）

当前实际源/正式包仍7a6f；候选只改solve_accelerate.cpp，不改AST、DAG、算子、输出、错误或Excel公式。完整原始系数仍由原DGETRF2/DLASWP/DTRSM/DGEMM分解，DGETRS求解。以下表示仅减少校验证书的矩阵重构工作。

对512行严格下三角前缀采样至多32列，复用DGEQR2/DORG2R形成Q，再以DGEMM计算C=QᵀL。不假设输入低秩或Q精确正交；每个L前缀条目均检查Q*C与原条目的差值及gamma(4r)正乘积界，得到逐行半径e_L。半径不够紧、非有限或QR失败的面板采用完整原分块重构，不省略条目。压缩前缀重构采用Q*(C*U)，对角贡献由DTRMM计算；原完整LU因子不修改。

原始P*A与重构仍逐项核验。表示误差上界为e_L(i)*Σ_k max_tile|U_kj|；嵌套产品的舍入界使用明确的常规FP64 BLAS模型及Q绝对值、C绝对值与Upper包络的正产品，覆盖各前缀、DTRMM和加法。gamma(4N+4r+4*512+64)、向上膨胀和原underflow余量保守叠加。局部严格主元/原1e-14阈值、RHS和回代边界仍按原依赖证明，成功发布前仍验原方程残差；这不是对任意厂商误差模型或病态前向误差的普适证明。完整数学审核、SAN/OFF及扩大边界验收尚未闭环。

error/growth状态改为列块优先，使独立8行连续。固定Eigen8×8和8×1表达式执行相同逐p递推，有限/幅度检查合并为abs<=1e100；标量尾部及Eigen关闭时的原生标量递推保留。Q/C、正包络和状态均复用原reserve，固定向量只用调用栈；无第二份完整系数、请求外数值缓存或竞争线程池。512三块及64N缓冲仍独占，所有输入、旧结果和callback所有者保持原寿命。

复制审计显式增加采样列及因子三角项的复制，量可接近一个N²扫描；它是算法移动，不是输入绑定复制，也不增加完整矩阵分配。普通C++不触发Excel，公式/预算/签名无需改变。实际峰值须由完整分配追踪和fresh RSS验证，不能仅凭reserve相同宣称内存不增加。

孤立512/128分块、动态Eigen广播、列状态和固定包试验保留各自来源；128与动态广播没有显著收益，未进入实际源码。固定包单次4112组件重构约.20秒、前向/回代合计约.12秒仅作定位。完整4096实际7a/隔离873ac/热NJIT六轮C .834/.677/.647秒，F .930/.771/.677秒，仍落后NI约5%/14%。8192系数当前和候选均触发canonical回放，真实秒级成本必须继续计量；不能以四千阶局部改善关闭该主要负载。

## 8192拒绝位置与后续20cf隔离设计

873ac的四组完整计时及24个fresh峰值任务已终态通过测量流程；不表示性能目标通过。8192系数C实际7a/873ac/NJIT为29.7097/22.3894/5.0754秒，F为30.5673/23.0496/5.4340秒，双方C++均一次尝试后canonical回放。4112申请峰值候选C/F417.818/417.851MB，对照NJIT541.309MB；fresh RSS增量候选437.715/434.602MB，对照678.117/678.674MB。8192尚未做fresh峰值，不能按缩放推算为已测。

精确873ac helper、同一原始8192系数的诊断在p=7680拒绝：主元绝对值.9700，下界.9558，竞争行8176的因子约-.5199，但块半径52.1266。单一512列包络混入后续大幅值Schur列。只按幅值细分后该点通过，却在p=8165的接近相等主元处拒绝；加入近同值列细分仍因约5e-6半径超过约2.7e-6分离度拒绝。最后行面板完整重构也不足。保留全部负结果，不删除保护或改奇异阈值。

最终隔离方案20cfaf3ed42593a7596230f80fd8621261042d64620e366b37a251939e82711a将校验基础列宽改256，最多增加32个连续列组。列尺度跨64倍、当前或前一列的最大下三角LU乘子接近1（1e-4仅为细分启发式）时按预算细分；预算用尽仍使用宽组保守界。每个实际列有唯一单调column_tile，重构、包络、主元、传播、回代全都按该映射；所有条目仍测量，原1e-14错误阈值及严格分离检查不变。

仅对同一活动列组内最多32个主元批处理。活动组误差、严格主元判断、RHS，以及批内未来主元行保持原逐p顺序；这些是后续计算真正读取的依赖。批外独立行的未来列组不被当前主元读取，可将正仿射递推E'=cE+alpha*E_p+beta*U_p+gG+tiny展开后，以两次DGEMM更新。系数和权重来自已完成的活动递推；未来主元行在其主元时点的状态随后固定，不使用未计算的数据。

c=(1+u)*inflate，weights(q)=inflate*c^(count-1-q)。批外结果为c^count*E旧+Σweights(q)*(carried(i,q)*E_q+changed(i,q)*U_q)+(uG+tiny)*Σweights(q)。两个正乘积、权重/幂/和、原顺序递推和最终加法的常规FP64舍入，以gamma(16*count+128)、向上nextafter及原underflow余量保守覆盖。全部非负，因而最终幅度检查也覆盖中途增长；非有限/超界仍回放。此模型需独立逐单元正界对照及质量/性能验证，不把一次诊断作为普适证明。

显式reserve为N*K+4N*(ceil(N/256)+32)+67N+3*512²+516N doubles，含有界列索引、pivot/permutation及512N厂商保守reserve。alpha/beta复用已死的64N面板，DGEMM右侧复用两块已死512²缓冲；没有新增完整矩阵、数值缓存或线程池。原始LU/DGETRS、发布前原方程残差、结果布局/所有权、AST/DAG/Excel公式和算子注册均不变；普通执行不触发Excel。候选只替换现有私有校验证书实现，实际源码仍7a。

组件诊断8192的未批处理传播约2.216秒，批处理约.567秒；factor2.260、growth.331、reconstruction1.080秒，均为一次组件定位，不是完整任务胜出或内存验收。新增两项8192 C/F回归检查实际库路径、方程/约束、输入不变和两代独立结果。下一步完整配对、峰值及full/native/SAN/OFF，之后才能评估实际采用。
