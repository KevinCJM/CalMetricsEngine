# V40：完整引擎 HMM 检查点候选（尚未采用）

2026-10-06，接续V39。遵守计算D1/D4/D5/D6/D8及Excel X1/X9/X10；不改公开算子、参数、输出形状、数学公式、迭代停止或Excel契约，不改平台，不提交推送。实际V35r2的93份数值源和301份接口/测试/工具冻结。实验完整源放在隔离目录，不能当作已进入当前工程。

## 物理方案

N*K >= 2,097,152、N>=128、K>=3且无已证明独占的LatentFitScratch loan时，训练保留em和alpha两个N*K矩阵。完整后向遍历先检查全部数值，保存每128行的beta端点；按正序恢复129行beta（含右边界），先xi后gamma，gamma覆盖不再被未来块消费的alpha前缀。xi初始化一次、每个单元仍按时间升序累加，lse/libm/运算表达式、M步有序均值与中心方差、floor、错误及停止规则保持原契约。额外后向计算可能变慢，必须实测。

资源：small=2*K*K+4*K+(N>=128&&K>=3?min(N-1,128):0)；compact_beta=(131+floor((N-1)/128))*K。常规预算2*N*K+compact_beta+small；非候选沿用3*N*K+small。所有乘加用原checked helpers；直接入口、Planner、Graph和执行使用同一规则，无局部vector和隐藏分配。已有loan仍可使用三个完整矩阵，small改用独立共享函数，避免从compact预算减3*N*K造成下溢。loan只在reserve<当前预算时成立，并保留现有save/rebind/异常恢复/独立本轮未来结果的所有权证明。

## 验证顺序

原实际wheel独立生成完整模型oracle，与候选检查阈值两侧、128余数、C/F/反向、SIMD/scalar、全部舍入模式、迭代与停止、错误后的外部output前缀和恢复。真实Graph检查直接根和所有投影、prepared、独立旧结果、Planner内存拒绝与传输。沿用现有latent、xi、tail、loan及Graph/Planner相关回归，不调松预算或精度。V39私有E步132+132+132及alias48只说明局部，不代替本次完整引擎。

正确性通过后串行比较实际V35r2、隔离V40和预热的加强NJIT。模型训练负载与原13输出完整业务链分别报告，两边计入相同独立只读输入准备、真实调度/结果返回，上一代保留到下一代返回。相同数据、算法、CPU1、数学参数和输出；编译/静态fixture/一次性定义/预热不计。独立fresh-process requested-live heap和基线RSS增量，不加总分阶段峰值，不把逻辑尺寸当实测内存。任何明显整链退化均须评估，不以局部省内存宣称采用。若性能不合适，保留失败证据并停止晋级。

## 成熟库与外部资料

已有Eigen/Accelerate路径保持。V39已查hmmlearn官方log/scaling说明及公开HMM checkpoint论文；黑盒训练器不能保证本项目log-domain、floor和停止契约，暂不替换。此候选改执行存储顺序，没有另写新的数学算法或扩张算子。Excel仍展开原逻辑数学过程，预算/公式/接口未改，本次不宣称新Excel重算或全量验收。
