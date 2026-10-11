# V51：成熟 DAXPY 的长 K 树归并接入

任务于2026-10-06启动，2026-10-07收尾。记录目录沿用启动日期。结论为阶段进展；整体 NJIT 时间/内存领先目标保持开放。

本轮从 Apple 官方 DAXPY 与 BLAS 线程文档复核已有成熟库。只将独立 MatMul 根的两处连续 scratch 归并改为 `cblas_daxpy(alpha=1)`，补32项实际 Graph 回归。原平衡树、DGEMM `beta=0`、.000049 准入、近零回放、取消、失败后完整覆盖、CPU预算与独立 C/F结果不变，没有新增算子、依赖、API或Excel公式。CodeGraph已更新，实际调用仍由当前独立根进入。

| 正确性资格 | 结果与边界 |
| --- | --- |
| 同源全量覆盖 | 15601通过、59显式Eigen开关跳过；原记录保留。精确59项另补验通过，共15660唯一测试通过，并非第二次全量命令 |
| 相关正常 / ASan、UBSan | 各2484通过；新32项涵盖四种输入布局、C/F输出、四条传输路径和两代独立结果 |
| 库关闭 | 2157通过、327明确库专属跳过 |
| native正常 / SAN / OFF | 各9/9 |
| 观察DAXPY调用的私有探针 | 每个normal/SAN模式48组（32树路径）、144非最近舍入拒绝、2取消；不是整图验收的替代品 |
| 当前目录重建 | 源码93份、wheel和worker身份与完整候选一致；另验native9和最近邻124项 |

首轮 sanitizer 子进程启动参数和 OFF 库验收开关配置错误，以及全量 Eigen 开关跳过，均保留原日志。修正仅作用于测试启动器；对应受影响任务重跑/精确补验后通过，没有以删除测试或放宽数学容差消除失败。详见`launcher-attempt-1/`、`v51-normal-quality.json`、`v51-normal-eigen-completion.json`及`v51-qualification-closure.json`。

所有计时串行、同一CPU1预算，普通执行入口与独立只读输入准备计入；NJIT预热，双方返回相同 C/F 结果并保留两代。昂贵核验开始前其余进程停在屏障，没有并发构建或测试。

| 4096×8192 乘8192×4096，输入/输出布局 | 旧实际V48 / 候选 / 同场NJIT（秒，中位数） | 较V48节约 |
| --- | --- | --- |
| CC → C | 2.3818 / 2.0743 / 1.3003 | 12.9% |
| CC → F | 2.3457 / 2.0141 / 1.7112 | 14.1% |
| FF → C | 2.4654 / 2.1968 / 1.8902 | 10.9% |
| FF → F | 2.1826 / 1.8775 / 1.3170 | 14.0% |

候选共6组108次计时，包括两组K4096未改路径控制；控制约7–8ms差异只作诊断，不设门禁。实际目录确认另有36次计时和8个fresh峰值进程，完整原始样本保留，下面的结项摘要由完成记录生成，不预填结果。

候选48个fresh峰值进程包括12个malloc子进程（每个3次请求），以及每后端/布局3次独立RSS复验，共36个RSS子进程。四种布局malloc峰值均为旧/新826350992bytes，NJIT811598240bytes；没有新的大额内存改善，且高于NJIT约14.8MB。RSS从源夹具和运行时初始基线单列增量；重复范围与旧版重叠，不能将NI多加载的运行时总量当作新增算法节约。malloc探针不覆盖全部Python arena、zone、mmap和栈；0.5ms RSS采样也可能漏瞬时峰值，两者不冒充完整分配审计。

正式采用的依据是完整长K请求稳定节约约0.27–0.33秒且malloc峰值不增加，符合成熟原生库的阶段采用规则。大矩阵仍慢于强预热NJIT，其他主要完整链、全面分配/复制审计和整体目标继续开放。未提交、推送、全局安装、合并或部署。

数学设计与自审核见`v51-design.md`及`v51-self-review.md`。`baseline-v48-identity.json`、三套候选identity和`actual-v51/identity.json`保存源码/编译器/安装二进制hash；数值源和测试快照以`.txt`保存，避免成为第二套当前实现或污染CodeGraph。`manifest.json`覆盖保留文件的SHA256。

公开依据：[Apple DAXPY](https://developer.apple.com/documentation/accelerate/cblas_daxpy(_:_:_:_:_:_:))、[Apple BLASSetThreading](https://developer.apple.com/documentation/accelerate/blassetthreading(_:))。官方Markdown读取在网页工具中因content-type受限，临时目录经公开URL下载后复核；不把文档中的一般说明当成本机速度证据。

复现时按identity的源码hash和编译器参数恢复对应快照、使用已有固定版本vendor档案并安装到独立临时目标；运行器中的绝对临时路径需相应调整。先运行资格，再串行计时，最后fresh内存；历史失败记录不能作为当前通过结果。普通生产执行不执行Excel，此轮不宣称新增Microsoft Excel现场验收。

实际目录复核已结束：

- actual-long-CC-C: 旧实际 / 新实际 / 暖NI = 2.3893 / 2.0844 / 1.3123秒；较旧版省12.8%。
- actual-long-FF-F: 旧实际 / 新实际 / 暖NI = 2.2113 / 1.9000 / 1.3312秒；较旧版省14.1%。

当前C/F两组malloc均826350992bytes，NI811598240bytes；RSS确认值和盲区见actual-v51-performance.json及原始记录。全部36次确认计时、8个fresh确认进程已终结。

实际build_id：`edbaffe144a36608bdafe1f300880c60be09751f0a1f94093e9e35fea0c021c2`。原93份源仅一个header变更、168份已验测试不变、新增1份32例测试，294份其他既存文件保持原hash。
