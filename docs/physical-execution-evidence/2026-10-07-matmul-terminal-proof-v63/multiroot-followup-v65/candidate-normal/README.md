# V65候选：正常资格完成，SAN/OFF与实际接入待完成

后续自审核发现原生混合声明布局的Planner预算缺口；本旧候选虽然通过正常/SAN测试，仍不得接入。已形成[预算修复版及新版本证据](../budget-safe/README.md)，本文件以下为修复前阶段记录。

这是隔离候选，实际源码仍为V64。93份生产source/config，普通Python全量16720项、关联292项、补充原生9/9均通过，无失败、错误或跳过。全量Python后仅追加原生测试header，生产及Python源未改；原生9/9重新通过。候选采用既有DGEMM/Eigen并借用最后逻辑根，单个紧密行省去预清零；未新增算子或改变公开数学/Excel。

| 布局 | 旧C++秒 | 候选C++秒 | 同轮强NJIT秒 | 候选/NI |
| --- | --- | --- | --- | --- |
| C | 6.4842 | 0.7316 | 0.6535 | 1.1194 |
| F | 6.7478 | 0.7697 | 0.6623 | 1.1623 |

8192×1024×8192、两根与两代保留，CPU1，相同owned readonly输入准备和所有输出，普通Scheduler。当前[24个fresh结果](candidate-large-results-lazy.json)全部结束，全部单元及旧结果保护通过，绝对尾差0。旧C++数据是前一epoch冻结基线；同轮新/NI各有三组独立时间进程，每组3次。单根控制仍保留，不能以局部内核替代完整请求。

malloc暖峰两布局均由旧2,821,207,536降到候选2,288,002,088bytes，少533,205,448bytes，约508.5MiB；强NI2,281,702,136bytes，仍多约6MiB。numeric arena512MiB降至0，DGEMM为每行一次；库工作区真实计量。fresh RSS增量C/F候选2,227,585,024/2,233,925,632bytes，NI均2,227,273,728bytes，旧版分别2,754,576,384/2,752,036,864bytes。malloc与RSS分别测量，不相加、不以总RSS代替增量。

**候选仍慢于强NJIT12%至16%，没有低于NI的大额内存收益，整体目标未完成。** SAN构建已启动，尚无SAN/OFF验收或实际源/wheel接入；当前不能宣称采用、迁移或全部平台链通过。

[自审核与失败解释](self-review.md)、[普通资格](qualification-shared-roots-normal.json)、[最新原生XML](candidate-final-native-r2.xml)、[数值/所有权关联XML](candidate-targeted-lazy.xml)、[设计阶段](../design.md)、[状态与剩余项](progress.json)。所有修正前的测试假设、失败、诊断、源码和构建身份保留，没有删除反例或改动数值/性能预算。无本轮真实Microsoft Excel重算、跨平台、部署、提交或推送声明。
