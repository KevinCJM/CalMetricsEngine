# V71 自审核与结论

2026-10-09。30组原生离线容量诊断结束：五个负载、三种逻辑图、两种槽位选择，闭合生存区间均无同槽冲突。11组原图/既有控制图的double+int64容量逐byte等于之前实际包audit；`numeric_arena_bytes`本来合计这两池，mask另列，不能把整数池遗漏后声称trace不符。

原代码、93源/配置、191测试、实际安装native/worker及四个链接archive按hash保护。新诊断器是隔离C++工具，不是已安装wheel；未运行数值内核、没有新time/heap/RSS或完整算法验收，也未重新执行原native/Python测试。已有V69/V70记录仅提供匹配的已结束runtime audit，不作为V71新的性能进程数。

## 真正定位的负实验

- dense-view大GMM：原图take55紧邻scatter56，take55物理extent为0；控制图take53与scatter55中间首次需要创建slice54，该节点并非input/constant/parameter，违反现有take/scatter consecutive证明。于是take53物化5,592,416 doubles（44,739,328bytes），整体arena增加41,943,120bytes。不是lost GMM posterior replay，也不是视图本身分配；有界fit workspace仍保持。当前不自动放宽证明，不采用控制图。
- 原图简单static-size best-fit：GMM16合计numeric arena由128,628,384增到148,201,840bytes；GMM3由22,000,376增到24,000,376bytes。HMM只省几十/千余bytes，不值得采用。声明尺寸不等于物理extent，独立根、私有融合/短cache以及未知record维度必须一起纳入成本。只有逻辑区间无冲突不能证明有峰值收益。
- shared-standardized图保留旧训练gather容量，且训练/预测posterior跨槽染色导致另一个44.74MB大槽，不能仅凭ND除法被删除认定省内存。逐slot完整列表在diagnostic-comparison.json，不推测未执行图的时间或数学等价。

## 后续动作

本诊断与static-size方案不进入生产。下一步先验证既有GMM有界replay用于K3/D4的大样本是否能省掉约9MB工作区，同时保持完整耗时优势；更一般的arena重排需要物理extent、别名/延迟依赖及history联合成本，再做实际完整测试。不能用本次离线预测代替采用资格。

逻辑图、公开数学、数据类型、算子、所有权与Excel recipe/budget无变化；无需修改用户/算子/Excel手册，无本轮Excel重算、平台迁移、提交或推送。目标仍开放。
