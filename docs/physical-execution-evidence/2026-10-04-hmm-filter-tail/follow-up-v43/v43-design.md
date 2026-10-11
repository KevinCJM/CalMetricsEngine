# V43 隔离候选：消费密度后复用 beta 历史，按块重放 density

范围：已有 HMM EM 的物理执行，不改模型、接口、AST、逻辑 DAG、Excel 公式、参数下限、停止判断或 Scheduler 核数。生产数值源仍为 V42。隔离候选构建 ID 为 `8d2c3f545ab2b7b68872d5e9c080d3d6b34a01dff763e78c4634e5000da06df1`。

## 原生库与公开实现核对

2026-10-06 查阅 Eigen Array 官方说明及 hmmlearn 当前原生源码：

- https://libeigen.gitlab.io/eigen/docs-nightly/group__TutorialArrayClass.html
- https://raw.githubusercontent.com/hmmlearn/hmmlearn/main/ext/_hmmc.cpp
- Apple vForce 入口 https://developer.apple.com/documentation/accelerate/vforce-library 本次可见页要求 JS，Markdown 工具读取失败；本次不把该失败读取当作新的功能证据。当前实际已链接 vForce 与 Eigen 的行为以构建、源码和执行计数为证据。

Eigen 适用于保留各分量运算顺序的数组表达式；hmmlearn 是归一化、停止和数学实现的比较参考，并未替换、下载或绑定其内核。本候选重用项目现有唯一 density/Eigen 实现、vForce exp/log1p 以及有序转移 tile，不新增依赖。

## 生命周期和数学纪律

1. 初始完整 density E 和前向 alpha A 使用原数学表达式。
2. 后向每行先按原 `((logA+E)+beta-next)` 与 lse 顺序完成全部状态。仅在每个源状态均已消费 E[i+1] 后，才把 beta-next 写入该行。
3. 使用已被 forward 消费、M 尚未使用的 initial/effective 两个 K 缓冲保存 beta-next/current。终点初始化为精确 +0，最终 E[0] 写 beta[0]。
4. E 此时是完整 beta B，A 未改变。xi 每 128 行重放密度；输入、均值、方差和 logs 仍属于本轮 E 步，M 尚未执行。
5. 重放调用唯一 density_rows：使用原始 N 选择 Eigen 路线，实际行数决定循环。保留有符号 stride、特征归约顺序与异常检查；实际附加库调用全部计数。
6. xi 沿原 s,t 归一化顺序、沿升序时间求和。不把 E+B 预相加，不反向加 xi，不用全局 likelihood 替代逐行归一化。
7. 对 alpha/E/logA 的范围检查在 E 覆盖前完成，beta 的范围在唯一后向递推中检查。若任何部分不符合 vForce 条件，整个 xi 使用原有序数学路径。
8. gamma、M、似然、严格停止判断和输出生成沿原顺序。

几何统一计划：N>=4096、K>=3 时 `2NK+2K²+4K+512+128K` doubles；其它 HMM 为原三矩阵计划。直接 API、registry 和 graph loan 使用同一内部矩阵数规则。完整规划在取指针前完成溢出检查；graph loan 仅使用证明独占的当前 arena/未来结果，恢复活值，不借旧结果。

## 本轮验证与证据边界

- 初次原生测试因沙箱 shm_open 被拒绝，原日志保留；相同测试在可用共享内存权限下 9/9 通过。
- 初次相关回归 782/783 通过。唯一失败为旧测试硬编码 3NK 内存下界；改为精确两矩阵及重放尾部预算，低预算拒绝断言保留。之后完整所选 783 项通过。
- 实际 V42 wheel 对照候选：12 个布局/维度/边界样本及 4 种舍入，共 16 组完整模型全部逐位一致，输入和舍入环境不变。它不证明所有输入均逐位等价。
- 第一次性能驱动未开始计时：残留 V35 的“原基线 vForce=0”断言。V42 已使用 vForce，修正为两套 native 路径均使用相同正调用计数；失败状态、快照、驱动保留，不算有效性能观察。
- 正式交替测试实际 V42、隔离 V43、已预热独立 NJIT；完整训练 8 输出和完整识别 13 输出；CPU1、相同 6 个 readonly 自有输入准备、两代独立 readonly 结果保留。基准期间不跑构建、测试或重扫描。
- 工作区下降约一个 NK 仅是理论/计划指标。图 loan 已可能复用同一空间，不能称为完整请求峰值收益。需分别测训练和完整识别的连续 malloc epoch / 新进程 RSS。
- 当前候选没有完整分配/复制审计或生产接入闭环。后向逐行复制 beta 到历史的流量需要明确记录；宽特征重放成本以及已能免费借用第三矩阵的情形需要额外性能复核。若没有收益，拒绝候选或按真实成本限定路线。
- 数学/Excel 依赖与公式没有变化，不能据此宣称新增 Microsoft Excel 桌面重算验收。

状态：实验，不作为生产已实现能力或总目标完成证据。实测结果另写 review，原失败记录不覆盖。
