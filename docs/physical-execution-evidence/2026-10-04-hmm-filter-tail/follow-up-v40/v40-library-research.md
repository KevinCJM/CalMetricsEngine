# V40后续成熟库线索（非采用证据）

2026-10-06。完整训练实测：实际V35r2 5.5408235004秒，严格有序checkpoint候选6.7195711669秒，加强NJIT10.6037980206秒（同输入准备、CPU1、八个模型结果、两代保留，六轮）。checkpoint多一次后向计算，不能仅以工作区减少判定优化成功；未做分阶段归因，不把推测当已量测热点。

Apple官方[vForce](https://developer.apple.com/documentation/accelerate/vforce-library)提供架构适配的向量超越函数；[vvlog1p](https://developer.apple.com/documentation/accelerate/vvlog1p%28_%3A_%3A_%3A%29?changes=_6)给出双精度log(1+x) API，[vvexp](https://developer.apple.com/documentation/accelerate/vvexp(_:_:_:))给出双精度指数API。搜索取得官方说明；正文打开/Markdown点击部分遇到JavaScript或工具content-type错误，不宣称读取全部实现、误差保证或任何本机收益。现有C++根exp/log已使用vForce，HMM xi仍在每个状态对/时间行调用std::exp和std::log1p；可研究复用成熟库处理独立时间行的向量超越函数，而保留状态对顺序与每个xi单元的时间顺序。

采用前必须核对现有宏、CPU预算、SDK/架构、FE_TONEAREST、finite/特殊IEEE边界与fallback，评估模型迭代/停止、分类、排序等离散消费者的影响；允许四位小数后的合理浮点尾差，不能以尾差吞掉离散/错误变化。局部vForce快不能证明整个模型或13输出完整链快，必须完成实际完整图和fresh峰值内存。完整libm计算顺序并非所有浮点库替换的硬要求，但不可忽略具体版本化契约与离散行为。

checkpoint另外需要补充已有Workspace被其他算子写入后复用的N%128=1控制：末行beta端点不能依赖vector首次创建时自动清零。V39私有core每次新建零初始化vector，现有V40大形状未命中该余数；该问题已在本轮后续动态复现：初版N174849/K12、median污染后两轮似然最大差1.176576；实际V35r2不受影响。V40r1显式写末行+0，四舍入/C与F/两种污染8项通过，连同完整引擎新增34、相关651、native9/9完成。修正版训练6.735秒仍比实际5.552秒慢，完整13输出5.714/5.727秒基本持平，保持暂不采用。生产源码未引入checkpoint。
