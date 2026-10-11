# Matmul RSS 归因复测分析

只读分析已完成的66个子进程，不执行数值/构建。旧4e12、新1aec，3案例各11次AB/BA配对；每个进程13阶段。原31项/186份heap与RSS结果完整保留；本次补充归因，**不证明完整内存不增加，不替换原测量**。

## 配对RSS差

以下均为新−旧，单位MiB。request delta先减各自backend_import基线；peak是观测绝对峰差。中位数不能抵销正差案例。

| 案例 | request delta 中位数 [最小,最大] | 正差 | peak 中位数 [最小,最大] | 正差 |
| --- | --- | --- | --- | --- |
| 1024³ CF direct | -0.0781 [-8.0938,+8.0938] | 2/11 | -0.1406 [-8.5156,+6.8594] | 5/11 |
| 32768×64×32 FF ordinary | -0.0156 [-15.9688,+16.1719] | 5/11 | +1.8594 [-15.8281,+15.8750] | 8/11 |
| 512³ CC ordinary 对照 | -0.1094 [-4.0156,+6.0781] | 4/11 | -0.5000 [-4.9219,+8.1094] | 4/11 |

原单次三个案例request delta差依次为+15.8594 MiB、+16.0781 MiB、+10.0781 MiB。这些单次增加仍保留；新的暂停式复测不能将它们判为噪声。原31/31冷、热requested-byte活跃堆峰相同，但该事实不等于zone计量或RSS相同。

## 关键状态净差

下表为配对 `[(新阶段−新backend_import)−(旧阶段−旧backend_import)]` 的中位数，单位KiB；完整11对范围和正差次数见JSON。各列分别取中位数，不能简单相加。

| 案例 / 阶段 | zone in-use | zone allocated | internal | reusable | footprint | RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 1024³ CF direct / warm_two_results | +93.16 | +0.00 | +8064.00 | -8192.00 | +8015.81 | -80.00 |
| 1024³ CF direct / results_released | -8096.05 | +0.00 | -32.00 | +0.00 | -32.00 | -80.00 |
| 1024³ CF direct / gc_after_close | -8096.05 | +0.00 | -32.00 | +0.00 | -32.00 | -80.00 |
| 32768×64×32 FF ordinary / warm_two_results | +93.42 | -1024.00 | +0.00 | +0.00 | -16.06 | -16.00 |
| 32768×64×32 FF ordinary / results_released | -33.33 | -1024.00 | -64.00 | +400.00 | -80.12 | -16.00 |
| 32768×64×32 FF ordinary / gc_after_close | -33.22 | -1024.00 | -64.00 | +400.00 | -80.12 | -16.00 |
| 512³ CC ordinary 对照 / warm_two_results | -31.89 | +1024.00 | +32.00 | +0.00 | +48.06 | -112.00 |
| 512³ CC ordinary 对照 / results_released | -30.47 | +1024.00 | +0.00 | -208.00 | +0.00 | -112.00 |
| 512³ CC ordinary 对照 / gc_after_close | -27.97 | +1024.00 | +0.00 | -208.00 | +0.00 | -112.00 |

1024³ direct的warm_two_results阶段，internal净差中位数 **+8,257,536 B（7.875 MiB）**，footprint **+8,208,192 B（7.828 MiB）**，reusable **−8,388,608 B（−8 MiB）**，RSS **−81,920 B（−80 KiB）**；internal/footprint各6/11为正，RSS仅2/11为正。单个输出本身为8 MiB。此组相反方向、接近一个输出大小的变化，与分配器/VM对复用页的分类和重新计费状态不同相符，不能据此断言新增8 MiB数值scratch或称RSS已证明持平。

结果释放后，该案例internal与footprint净差中位数均为−32 KiB，reusable中位差回到0，持续至关闭/GC；但internal范围−208至+128 KiB、footprint−256.1875至+192.25 KiB，仍有2/11正差，故“归零”只能表述为中位数回到接近基线，不能说每对精确为零。zone in-use中位净差约−7.906 MiB，不能直接解释为引擎节省这部分活跃对象：zone统计包含整个进程arena/cache状态，并受测量前分配影响。

大矩形FF结果释放后，internal/footprint中位净差约−64/−80.125 KiB，2/11为正；reusable中位+400 KiB且范围约−16.06至+16 MiB，RSS正差仍5/11。CC对照没有进入新增F路径，释放后internal/footprint中位为0，但各5/11仍为正，RSS范围仍约−4.02至+6.08 MiB。对照也显示明显页保留变化，这是归因线索，不是替优化路径豁免RSS的依据。

## 观测有效性与盲区

- 66/66进程、858/858阶段快照：Mach返回0、mach_count=93、valid_mask=0x7fff0、page_size=16,384 B；15个wire数据字段全部有效，无None。源码/观察器身份与manifest匹配，编译命令显式`-arch arm64`。全部子进程通过END后完整输出、只读/所有权与构建核验；33对通过原1e-11及有限值逐位核验。
- 父进程与被测进程PID全部不同。共258,965个RSS样本，已将792次相邻阶段handoff并入全局间隙核对；时间戳无倒退。
- 名义间隔0.5ms；真实最大全局间隙 **5.260334ms**（1024³CF direct，第5对old，warm_released内部）；最大handoff **2.647208ms**（同案例第2对old，warm_released→warm_one_result）；最大单次观察调用 **4.205333ms**。短峰仍可能漏采，不能以0.5ms配置声称0.5ms最大分辨率。
- “cold”是readonly输入构造、输入hash和native导入之后的**首次数值调用**，不是allocator初始状态。`input_identity`的`tobytes()`曾分配并释放完整输入副本，可能留下可复用页；快照按此真实边界解释，不能将其移出上下文。
- malloc in-use是zone视角，allocated是容量；两者差不是RSS。internal、reusable、external与footprint具有不同/可能重叠的核算边界，不直接相加。生命周期高水位未相减伪装请求峰值。每阶段20ms暂停和原生快照也改变寿命时序；首样本前及END尾部没有连续时间戳证明。

可用于正文的结论：**分阶段复测支持“输出页复用与VM计费状态差异参与了RSS波动”的解释，尚未证明唯一根因或完整内存不增加。1024阶direct两结果期约+8 MiB internal/footprint伴随−8 MiB reusable，释放后中位净差接近基线；原单次RSS增加及所有正差配对继续保留。**

精确字节、阶段范围、正差次数、每进程最大采样gap和文件hash见`rss-attribution-analysis.json`。
