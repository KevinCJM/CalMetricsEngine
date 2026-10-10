# HMM/EM工作区复用：已接入，阶段回归通过，整体目标未完成

2026-10-04。正式87源与正常AOT构建 **0fe773087525162b7a2c110d8bd551b1b053d59900cae3106452c3b06e87aa3c**一致；本轮从7aba基线接入9个原生文件，测试均已改为仓库内独立可复现，不依赖临时bench模块。未提交、推送或修改平台代码。

[设计](design.md)与[自审核](review.md)：大样本fit在原时点保存活跃内部owner，借用独占arena及本次未来scatter独立结果，调用同一latent数学实现后恢复。公开Prepared in-process保持canonical，ordinary明确授予独立新结果许可；worker仅借用私有inline/本次新建SharedRegion。内部接口需AOT重建，公共190算子/IR/IPC22/结果协议与Excel公式不变，Planner保留canonical预算，实际复制如实计数。

## 完整数学链与峰值

同一finite/fixed realtime/random init/K states分支；CPU1，六轮轮换old/new/热NJIT，完整13根及两代独立结果。标准化、初始化、GMM预训练、HMM训练/续接预测、排序映射/回填和普通绑定/plan/return计时内；fixture与构图独立准备。模型浮点按原1e-9、离散精确，old/new全部13输出逐位一致。

| 负载 | 7aba / 0fe / 热NJIT |
| --- | ---: |
| 20k×4×12，EM200上限 | 3.749 / 3.742 / 7.306秒 |
| 500k×4×12，EM5上限 | 5.724 / 5.713 / 11.247秒 |
| 500k热requested heap峰值 | 403.668 / 299.673 / 272.514 MB |
| 三次fresh RSS增量中位数 | 449.135 / 325.534 / 366.002 MB |
| 三次fresh RSS绝对峰值中位数 | 573.948 / 451.658 / 491.913 MB |

MB为十进制。新版本较7aba申请峰少 **103,994,980B（约26%）**，RSS增量中位少 **123,600,896B**。较NJIT的RSS增量中位少 **40,468,480B**，但requested heap仍多 **27,158,644B**。时间优势主要来自已有内核，本轮几毫秒变化只诊断，采用依据是大额内存收益。

实际Workspace 108,003,808→16,008,512B，aggregate numeric arena 117,002,464→105,002,464B（含int64）；保存/恢复令algorithm copy 25,002,880→77,011,200B，增加52,008,320B已经计入时钟与heap。容量不是整个请求峰值，不能称零复制。

全部12个内存任务正常结束：每后端1个fresh heap进程、3个fresh RSS进程。连续内存epoch包含full readonly fixture构造、构图、第一次/热执行和两代结果；tiny编译/预热在外。原memory报告的一个notes字符串误写fixture在外，实际construct和measurement_scope证明在内，保留原文件与SHA并在review明确纠正。盲区包含Python arena、mmap、未拦截分配器、未跟踪释放和RSS采样；未宣称完整分配审计。

## 正确性与接入状态

- 隔离专项696项、实际扩展ASan/UBSan54项通过，无skip，旧原始文件保留。
- 12个old/new大宽度子进程（6种模型/布局）全部正常结束；300k×32×8的GMM/HMM C/F/负步长，全部模型字段、诊断和posterior逐位一致，只读hash及两代所有权通过；C/F实际Eigen density计数均为2。
- 仓库内新增60项通过：保留原47个scatter、7个完整loan链并增加6个大宽度case。最初沙箱有2项共享内存创建被拒绝，其余58项通过；本地shm许可后60项全部通过，没有skip或修改数值契约。
- 正式源码/wheel身份匹配，完整Python **12,379项通过、零失败/错误/skip**（60项新增包含其中，不重复相加）。原生、原生ASan/UBSan、三库OFF各 **9/9**；真实ASan/UBSan扩展 **60项**，三库OFF真实扩展另 **60项**全部通过，无skip。OFF构建身份0899271a与相同87源/实际二进制匹配，显式EXPECT_EIGEN=0检查没有Eigen调用。macOS不支持LeakSanitizer，未计为通过；全部终态见integration/acceptance.json及实际XML/log。

初始690通过/6失败保留：3项公开Prepared权限假设不成立，修正原生入口而不是减弱所有权；3项Python int参考需np.asarray。大宽度脚本首次误将internal模型record作为public root，二次误用诊断函数名，均在数值执行前按原接口修正；没有放宽容差、算法或预算。

复现用[7aba已验收87源/tests/tools/vendor](../../2026-10-04-mc-state-batching/README.md)，覆盖本目录sources中的9个原生文件与3个测试文件；candidate-owned和integration/production身份列出源及实际wheel/worker SHA。drivers含本机受控路径，需重绑定环境与重验身份；NPZ大型原始数据保留临时目录，seed/输入SHA/逐位检查记录可重建。历史初始源、测试与反例不参与当前生产执行。

Excel是同逻辑图的独立后端，物理loan不对应工作表覆盖，不改公式/预算；全量Python执行适用导出回归，本轮没有启动Microsoft Excel，不外推新的真实重算PASS。

剩余：requested heap仍高于NI27 MB，其他主要负载、完整分配审计、动态缺失/available-time/quantile ties/retrospective/Markov/state_count>K及真实平台全请求未完成。整体目标active。

[下一轮大额内存设计](next-memory-design.md)只记录训练过滤尾行与后向checkpoint/replay的联合候选，尚未实现或计入收益；不能用纸面容量关闭27 MB差距。

原生OFF首轮错误继承EXPECT_EIGEN=1，graph测试要求被禁用的库而失败；原log/XML与终态记录保留。复验不仅去掉错误要求，还显式EXPECT_EIGEN=0重新跑9/9，保持原断言，并对相同源码的OFF扩展运行60项loan/布局/transport回归。没有修改计算代码、测试断言或skip来消除失败。
