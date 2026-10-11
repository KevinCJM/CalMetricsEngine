# V71：原生节点容量诊断

2026-10-09，实际仍为a87f2c25，无生产代码采用。独立C++诊断器调用同份graph.cpp的结构形状与arena逻辑，30组离线结果结束；五负载×原图/两负控制图×first-fit/声明尺寸best-fit，闭合区间无同槽冲突。11组容量与之前实际runtime audit逐byte匹配；numeric_arena_bytes合计double及int64，mask单列。诊断没有运行数值内核，也不是新的性能验收。

定位到dense-view大GMM的具体增长：原take55紧邻scatter56时extent为0；控制take53之后多出slice54，违反现有相邻证明，take53物化44,739,328bytes。整体numeric arena因此增加41,943,120bytes。视图本身仍不分配，fit有界replay未丢失；不采用控制图，不自动放宽错误/依赖证明。

简单声明尺寸best-fit也被否定：原GMM16/GMM3 numeric arena增加19,573,456/2,000,000bytes，HMM仅省微小容量。声明尺寸未扣除实际融合/cache/直接根，也有record表达式尺寸未知；不能把区间合法当成内存优势。仍需依据真实物理extent、别名/延迟依赖与history整体成本设计。

本轮仅隔离诊断、源码分析与负实验；93源/配置、191测试、实际native/worker和链接archive hash保护。没有实际代码/公共契约/Excel recipe变化，无本轮Excel重算或平台迁移，无提交推送。下一步独立筛选既有GMM replay的K3/D4大样本准入，完整QP/前沿/探索等主负载仍开放，整体目标未完成。

[设计](design.md)、[自审核](self-review.md)、[逐节点/槽位对照](diagnostic-comparison.json)、[身份与复现命令](preparation.json)。
