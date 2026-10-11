# V71：实际容量诊断与静态尺寸槽位选择

2026-10-09，实际源码/包冻结a87f2c25。继V70正负实验，先用独立C++诊断器调用实际graph.cpp分配逻辑，保留原slot/形状/融合证明和闭合生命周期，逐节点记录真正的物理extent，再比较按尺寸选择空闲槽的候选。诊断不运行数学内核、不分配样本数组，不作为完整性能收益。

输入是已安装实际包的原生图检查接口导出的节点、父边、opcode、类型、槽位、最后读取、根和input axes。Python只写出这些已确定描述，没有生产数值或计划逻辑。Native所需array payload不可用，仅结构规划读取shape；不用伪数据执行数学。诊断图包括五组原链及两个旧负控制图，固定有限数据的控制图仅用于理解分配，仍不采用其数学重写。

诊断graph.cpp在每轮节点容量max更新前临时保存旧count，将当前slot计数置零，记录该节点真实extent，并以max恢复旧count；最终layout必须与已有实际audit逐byte相等。只记录小型metadata，不保留数组。其他真实源码/93配置、191测试、实际native/worker和已链接native archive的hash保护；诊断可执行文件另有hash，不能声称它是已安装actual wheel。

## 候选合法性与职责

目前仅诊断：沿编译器已经包含borrowed backing的原生last_use区间，在不同dtype池中选择空闲槽。相等端点不可复用；根/借用后代保活不缩短。已知声明尺寸按最小capacity增长、再最小匹配容量选槽；未知表达式保守按原first-fit。所有融合metadata在新slot后重新finalize，复用既有依赖/私有/别名证明，不能保留旧槽证明套新地址。

若诊断支持生产候选，只在C++编译器做槽位选择，不改变公共AST/算子/数学/输入输出，实际shape仍由现有Planner和executor共同规划真实容量。未知尺寸不猜值；固定shape只作cost选择，不成为数值复用身份。输入、旧输出、独立根不能做scratch。scope/多根/分支/错误隔离/进程序列化仍遵守原闭合生命周期；若无法证明性能收益，不采用。

底层数学继续复用现有Eigen/Accelerate/Highway，无新数学内核，无新业务算子。适用D1/D2/D5/D6/D8，X1/X6/X9/X10：物理slot不成为Excel公式来源；不变逻辑与数学无需修改Excel recipe/预算，普通请求不执行Excel，无本轮Microsoft Excel重算宣称。

## 验证顺序与剩余资格

1. 原生trace与五组实际arena audit逐byte匹配；各owned闭合区间互不重叠；定位两个负控制图的具体槽和容量变化。
2. 若有大额减少，再隔离构建93源/配置身份一致的AOT wheel，先相关native/Python与特殊lifetime/unknown shape/borrowed views/roots/多scope/错误/transport检查。
3. 相同完整原图、六份owned readonly输入、CPU1、13根/两代保留、暖NJIT，独立fresh time/heap/RSS与原actual三方对照；joint arena/history fit reserve及复制量一起检查，不把局部arena预测当成最终peak。
4. 重复时间组、全量/SAN/OFF、实际重建/复验之后才采用。当前目标仍未完成，完整QP/前沿/探索与其他主要负载继续开放。

公开方法参考沿用V70的[OpenXLA说明](https://openxla.org/xla/hlo_to_thunks)和[ORT分配器源码](https://raw.githubusercontent.com/microsoft/onnxruntime/main/onnxruntime/core/framework/allocation_planner.cc)，仅分析参考，未引入依赖/复制实现。
