# V70：小状态大样本 HMM 的私有标准化融合

2026-10-09，实际源码/包仍是 a87f2c25。隔离候选只修改 graph.cpp，不改数学内核、公共算子、定义图、输入、输出及平台代码。

## 事实与优化范围

V69 完整四组 13 根链的三组 fresh 时间与独立申请峰/RSS通过，但 3 状态 HMM/GMM 申请峰分别高于同核预热 NJIT 14.82/20.06MiB。统一将训练/OOS gather 改成 slice 的控制图导致大 GMM arena 增加约40MiB，不采用。当前 arena 按逻辑 slot 的最大实际 shape 保留；视图延长 backing 生命周期，不能只按删掉一次 copy 判断收益。

当前 Program::finalize 发现任意 HMM 即禁止新私有算术来源融合，而实际 history loan 只在 NK >= 2,097,152 时准入。V69 HMM3 的训练 NK=562,500，没有该 loan，却仍物化私有 ND 标准化中间值。候选保留现有 private-source 编译证明，运行几何阶段对每个 HMM 检查原 history 门槛；只在全部已知训练 NK 未达门槛时允许这类算术融合。大 HMM 保留现有 history route；take/scatter 新融合仍按旧 HMM 限制，不扩大范围。共享同一常量，避免执行和融合门槛漂移。

## 数学、依赖、所有权和错误

- 复用既有相邻/保活证明、单消费者、未选择为根、无 scope、原逻辑 slot 未被中间 writer 覆盖、精确相同 shape 与同坐标读后写检查；未知/零/不符合拟合维度的几何不准入，原节点继续验证及执行。
- 不缓存数据，不改随机流、初始化、EM 算法、正停止容差、迭代数、排序或状态。现有元素级融合按原 subtract 后 divide 顺序执行，仍保留原 logical nodes 和校验位置。
- 输入、旧代及独立根不是工作区。普通、prepared、线程/进程、错误隔离各自原准入条件保留；Planner/arena 和执行共同使用 deferred_pair_geometry。新的几何判断使用整数除法避免 NK 乘法溢出。
- 不降低 history loan 门槛：GMM3 当前仍需保存的训练 ND 可能比 NK scratch 更大，单独放宽门槛不能证明节约。GMM 默认申请峰不因本候选自动关闭。
- 遵守 D1/D2/D5/D6/D8，优先保留已用的 Eigen 原生数学；本轮没有新的底层数学需要寻找替代库。参考 OpenXLA 的生命周期/别名/缓冲分配说明只借鉴分析方法，不引入 XLA、ORT 或请求期编译：https://openxla.org/xla/hlo_to_thunks 。

## 验证和采用要求

先构建独立 AOT wheel/worker，核对 93 源/配置、真实编译参数和二进制身份；actual 不先改变。先原生9项及相关数值/物理图/工作区测试，再同实际包/预热NJIT执行V69四组完整链，并加入较大3状态HMM验证ND收益。每请求相同六份owned readonly输入，普通接口，CPU1/库上限1，全部13根和两代结果保留；编译、静态定义/fixture及事后检查排除。

完整所有输出与NaN位置检查，离散状态/顺序/实际迭代精确，浮点4位尾差政策、概率/方差数学不变量、输入和旧结果哈希、跨输入和跨两代非别名；申请峰与增量RSS独立fresh计量。未完成多组性能、全量/SAN/OFF、异常和传输专项以及actual重建/复验前不采用。

X1/X6/X9/X10：逻辑图、数学与状态未改变，无新算子/Excel recipe或预算变化；物理 arena 不作为公式来源。普通请求不执行Excel；本轮不声明再次Microsoft Excel重算。候选若采用，在现有验收文档登记适用数值回归及未验证项，用户类型/算子手册无需改变。

## 尚未完成

这是隔离候选。GMM3 内存、一般按真实尺寸分配的 arena/history 综合成本、F/批次/多核，完整 active-set QP/前沿/随机探索及平台服务仍开放。目标未完成，无提交、推送或迁移授权。

## 本轮执行结果与状态

原生9/9及7个相关Python文件701/701通过；首次沙箱共享内存两项失败日志保留，允许共享内存后相同命令通过。候选/current/NJIT五组各time/heap，共30个fresh完整测量和2个另列smoke全部通过13根及所有权核验。三状态HMM 250k/750k的申请峰较当前减少5,999,640/17,999,640bytes，耗时基本持平，申请峰仍高于NJIT。大HMM和两个GMM的arena/workspace不变。

4个共享全量标准化控制图完整进程通过核验，但GMM16/GMM3申请峰增加44,738,432/2,999,168bytes，不采用。编译器实际上跳过borrowed视图的数组槽，并延长backing寿命；问题涉及first-fit逻辑染色和slot最大容量，不是视图本身幽灵分配。逐节点trace仍未实施。

目前只是一组time/heap筛选，没有候选RSS、完整/SAN/OFF或边界/传输专项。私有匹配重复整图扫描的O(PN)风险须在采用前移入预计算metadata。实际引擎仍为a87f2c25，候选不采用。详见[自审核](self-review.md)、[下一步arena设计](arena-followup-design.md)、[完整对照](comparison.json)及[共享控制图](shared-comparison.json)。
