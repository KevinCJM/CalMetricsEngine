# 保序 LU 列分块筛选：未采用

2026-10-03，macOS arm64，基线构建295a464；77项当前生产输入hash未改变。本轮没有提交/推送。

四个候选复用现有32列主元面板与R4×W8 Eigen运算，只改变完成面板之后的独立尾部访问：128/256/512列条带，及32行×128列宏块。原每元素运算树、主元、single/multi不同NaN政策、即时RHS更新、回代、尾行与工作区不变。见design.md、candidate-review.md和四份patch；没有接入生产。

同一probe.cpp、同AppleClang参数分别编译基线和四个候选；1880/1880逐位/错误/失败buffer记录通过，其中包括376项基线自对照。覆盖single/multi/双span、四舍入、主元/阈值、非有限/溢出、零列、负/非连续步长和canary；调用前后输入不变。舍入模式已恢复。基线、strip128、block32隔离ABI反汇编无FMA，完整对象保留stage，不能代替正式wheel或栈上界验收。

## 4096阶C输入、8列RHS

仅一个负载，10轮平衡换序，每后端保留前一独立输出至下一调用返回。计时包含ABI结果分配/canary、输入复制、新建并初始化N²工作区、完整LU及回代；这是隔离筛选，与Scheduler热工作区和完整NJIT对照不同。

单位ms，中位数：

| 版本 | 耗时 | 相对基线 |
| --- | ---: | ---: |
| old | 2356.924 | +0.000 |
| strip128 | 2616.042 | +259.118 |
| strip256 | 2494.887 | +137.963 |
| strip512 | 2421.394 | +64.470 |
| block32 | 2402.573 | +45.649 |

没有候选带来重大收益，故不增加生产遍历逻辑。此结论是缺少优化收益，而非把几KB/ms正差当独立门禁。四个结果和全部原样本保留，不用小幅正差推称重大退化，也不按平均隐藏最慢候选。更小源片段未表现为更快；尚无硬件计数，不能把具体原因断言为cache/TLB。

本轮未做候选的完整Scheduler、强NJIT、heap/RSS、sanitizer或新Microsoft Excel重算，未证明平台负载改进；当前生产维持295a464及前一轮证据。下一步针对宽RHS复用现有保序回代helper，并定位真实平台重复计算/复制链路，以秒级完整负载和大额内存为验收重点。

实际编译命令/源码/库SHA见build-command.json；运行脚本screen.py与原结果JSON保留。headers/dylibs/反汇编仍在/private/tmp/calmetrics-solve-next-screen；异地复现需按baseline-identity.json准备精确295源，运行prepare.py生成条带，并按block32.patch生成另一候选，再以记录选项分别编译，不可把新源码冒充基线。manifest.json保存本目录小型证据hash。
