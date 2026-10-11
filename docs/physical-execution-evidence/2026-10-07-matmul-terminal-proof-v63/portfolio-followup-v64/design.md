# V64：组合收益兼容算子复用已验收的矩阵向量库

2026-10-07，隔离设计。实际目录尚未采用，性能尚未测量。

## 代码事实与范围

平台 `backend/optimizer.py` 的 `explore_portfolios_kernel` 为每个权重候选调用 `portfolio_returns_kernel`，随后计算收益/风险指标。该内核按资产列升序累加矩阵与权重的乘积。这里只优化引擎现有 `portfolio_returns` 及其已有 DAG 组合，不实现或声称已经替代平台完整的随机生成、约束修复、Pareto 筛选或优化流程。

引擎注册契约已经声明 `portfolio_returns` 等价于 `matvec`；其直接实现仍调用普通 `matvec_value`。`matvec` 和 `quadratic_form` 已使用 `library_matvec`，因此本候选只在 `sequence.cpp` 为非 scalar 路线调用同一个库资格选择器，不复制实现，不增加算子、依赖、线程、IPC、Planner 或工作区。

现有库按 C 连续行或 F 连续列使用 Eigen 借用视图：不同输出行独立 SIMD，资产列的乘法、加法及初始 +0 顺序不变，关闭 FMA 收缩。128 行、8 列以下或不满足既有真实 stride 的输入继续原普通路线。负权重、零系数仍消费原输入，不跳过 NaN/Inf。输入只读、结果独占、显式输出拒绝 alias 的已有边界保持。

## 公平的完整请求

使用一个普通 CPU1 Scheduler 请求：若干不同权重向量共享同一收益矩阵，每个根计算 `std(portfolio_returns(r,w),1)*sqrt(252)`，返回每个候选的风险标量。根全部在同一图中执行，不用 Python 逐次调用算子对照 NJIT 的一次完整循环。

默认 512 个候选、8192 个时点、128 个资产。输入按同一规则 owned readonly 准备：复制矩阵和完整权重矩阵，借用各权重行；C/F 矩阵均保留调用者布局。双方都计入这一准备、计算、分配和结果返回，保留两代完整只读结果。图定义、fixture、NJIT 编译/预热不计时。NJIT 按相同资产顺序生成每个完整收益向量，再按 Welford 样本标准差及年化因子计算；一核，未使用 fastmath。全部根对照原绝对预算 0.000049，并核验旧结果、非别名和输入不变。

顺序测量：V63 未优化兼容算子、现有等价 `matvec` 图作为潜力对照、V64 候选、预热 NJIT。计时 fresh 进程轮换启动，申请峰与 RSS 分别在独立进程测量，不能用规划预算冒充真实内存。若只改善几 KB/ms，或没有实质完整链收益，就拒绝候选，不反复扩展全量资格来制造进展。

## 验收与研究依据

先执行布局、阈值、特殊数、独立结果、显式 alias、错误与图传输测试；确认实质收益且内存不增加后，才执行完整正常/SAN/OFF及实际同源构建。直接复用的库资格不会改变公开数学或 Excel 公式，不能宣称已进行本轮真实 Excel 重算。V63 正在串行资格检查，不与其构建/测试并发跑性能。

公开依据：[Netlib BLAS](https://www.netlib.org/blas/) 区分矩阵向量 Level 2 与矩阵乘法 Level 3；[Eigen 高效乘积表达式](https://libeigen.gitlab.io/eigen/docs-3.4/TopicWritingEfficientProductExpression.html) 与 [Map/stride](https://libeigen.gitlab.io/eigen/docs-3.4/classEigen_1_1Map.html) 支持复用既有布局和借用内存。项目依赖仍为现有 Eigen 3.4.0；网页 3.4.1 不是本轮升级证据。这里保留既有有序矩阵向量实现，未经证明不改为可重排的 BLAS 点积。
