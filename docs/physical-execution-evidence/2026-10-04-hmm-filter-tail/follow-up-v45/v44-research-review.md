# V44 原生库资料与实现审核

2026-10-06。当前工作树仍为 V42，V44 保持隔离，以下是源码审核，不代替回归或实际性能验收。

## 主资料

- [Eigen Aliasing](https://libeigen.gitlab.io/eigen/docs-nightly/group__TopicAliasing.html)：重叠矩阵块赋值会读到覆盖后的值；`noalias` 只有真实不重叠时才有效。页面重定向至 nightly，不据此升级现有 Eigen 版本或声称版本性能。
- [Apple vForce](https://developer.apple.com/documentation/accelerate/vforce-library)：官方搜索结果列出批量双精度 `vvexp`；网页抓取只有 JavaScript 外壳。当前签名、启用条件与数值证据依据本项目编译的系统头文件和 V42/V44 测试，不能把外壳当完整 API 验收。
- [hmmlearn 原生实现](https://raw.githubusercontent.com/hmmlearn/hmmlearn/main/ext/_hmmc.cpp)：读取公开 C++ 前向/后向实现，作为递推结构参考；没有安装、整体替换或宣称其数学契约与本项目完全一致。

hmmlearn 的 `compute_log_xi_sum` 使用完整序列 log probability 归一化并在 log 域累计。本项目保留原每行 s/t 归一化和普通浮点按时间升序累计；不能只因两者都是 HMM 就整体替换后声明完全等价。当前借鉴限于底层方法与生命周期审查。

## 本轮决定

继续使用当前 Eigen、Apple vForce 与唯一有序递推。性能问题涉及实际缓冲寿命：三块历史已存在时保留它们，避免重算；确实可避免新分配时才选两块并重放密度。这是现有数学逻辑的物理执行选择，不是引入另一套 HMM 算法。

审核了四个 C++ 文件：统一几何规划；小尾部按 planned 而非借用矩阵数计算；registry 合法别名检查；图根容量/对齐/重叠和活值保存/错误恢复；同一次 backward 在消费后写入 beta；M 步前重放本轮密度；library 有界证明；真实 beta 复制计数。未发现新的数学、结果寿命或输入修改问题，尚待全量、sanitizer、库关闭与峰值内存证据。

源码中的 scalar、库关闭、小形状路径仍承担当前支持的运行条件；它们不是历史备用版本。没有新增算子、依赖、全局线程设置或 Python 数值回调。
