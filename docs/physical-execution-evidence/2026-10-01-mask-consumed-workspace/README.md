# 2026-10-01 掩码顺序统计：按本次消费量准备工作区

仅修改 `cpp/operators/registry.cpp` 一处执行入口。旧构建 `850871ebcff4f6f2e089ae96029f736ccc0c94282cd06824193f839a410906b3`，新构建 `ccca6b67a5220ea2134a4743d281b80c5af4c096e7d41c3d5b747bb8de70e331`；77项构建源仅此文件变化，76项不变。共享脏工作树，本轮未提交或推送。采用规则沿用 [D8](../../computation-design-rules.md)：几KB/几ms不能单独否决，原始严格比较与当前采用结论分开。

## 改变与合法性

`median_where/quantile_where` 在已有完整mask结构校验中统计本次选中数量；C连续mask直接读byte，其他步长保持逻辑访问。通过全部结构校验后按该数量请求scratch，不缓存输入内容，不加一次扫描，不读未消费的数值。排序/选择、NaN/正负零、原插值、空样本及错误顺序不变。没有新增算子、数学库、线程或IPC字段；AST/DAG、子作用域、observer、prebound与worker仍经过同一执行入口。

Prepared/requirements/Planner继续给出源N保守上界，覆盖内容从稀疏变密集及历史保留容量。显式reserve和dense→sparse复用继续保留高水位；本次请求量下降不等于该历史容量已释放。结果独立所有权不变。详见 [设计](DESIGN.md)、[容量审核](WORKSPACE-REVIEW-DESIGN.md)和[源码自审核](FINAL-CODE-REVIEW.md)。

## 完整请求时间

实际旧/新安装wheel对照已预热NJIT，单CPU，9轮轮换顺序，保留独立完整输出。共30类顺序统计负载；全部新旧结果逐位一致，NJIT沿既有2e-12/2e-14及IEEE分类核验。prepare与ordinary分别测量，没有省输出或增加核数。

| 请求 | 旧C++ ms | 新C++ ms | NJIT ms |
| --- | ---: | ---: | ---: |
| median_where 4M / 1% / ordinary | 12.135 | 5.858 | 5.871 |
| quantile_where 4M / 1% / ordinary | 11.742 | 5.782 | 5.915 |
| median_where 16M / 1% / ordinary | 47.230 | 23.090 | 23.170 |
| quantile_where 16M / 1% / ordinary | 46.434 | 22.293 | 22.524 |
| median_where 4M / 1% / prepared快照 | 11.973 | 5.914 | 5.776 |
| median_where 16M / 55% / ordinary | 148.161 | 123.536 | 137.887 |
| quantile_where 16M / 55% / ordinary | 183.609 | 160.310 | 180.269 |

上述稀疏时间仅接近NJIT，不能写为显著领先。prepared中位数比NJIT多约0.137ms可接受；未改变算法的4M中位数控制样本新旧多约0.423ms，照实保留而不独立阻断。未改的16M无掩码quantile仍为176.639ms，NJIT139.378ms，实质欠项不关闭。完整成绩与绝对变化见 [summary.json](summary.json)、[原始轮次](formal-timing/timing.json)。

## 内存

12类完整请求分别对旧/新/NJIT测连续setup→cold→warm malloc峰与fresh RSS，共72个fresh子进程、216个heap阶段，追踪表无溢出。工作区、接口/计划与保留旧结果均计入连续峰；输入在请求前已存在，三后端输入内容/形状/dtype/strides相同，生成/校验不用整块临时数组。malloc覆盖与RSS覆盖不同，不相加或相互替代。

| 工作区负载 | 旧C++ cold / warm MiB | 新C++ cold / warm MiB | NJIT cold / warm MiB |
| --- | ---: | ---: | ---: |
| median_where 4M / 1% | 32.009 / 32.010 | 0.329 / 0.330 | 0.640 / 0.640 |
| quantile_where 4M / 1% | 32.010 / 32.010 | 0.330 / 0.330 | 1.000 / 1.000 |
| median_where 16M / 1% | 128.009 / 128.010 | 1.287 / 1.287 | 2.555 / 2.555 |
| quantile_where 16M / 1% | 128.010 / 128.010 | 1.287 / 1.288 | 3.992 / 3.992 |
| median_where 4M / 55% | 32.009 / 32.010 | 17.604 / 17.605 | 35.190 / 35.190 |
| quantile_where 4M / 55% | 32.010 / 32.010 | 17.605 / 17.605 | 54.985 / 54.985 |

1%负载相对旧C++分别少约31.680/126.723MiB，并消除原相对NJIT的全N工作区申请差距。未显式预留且固定稀疏的ordinary/prepared cold/warm均受益；dense→sparse仍保留高水位，没有据此宣称该序列RSS下降。

16M稀疏fresh RSS增量新median/quantile约1.859/1.875MiB，NJIT3.875/9.500MiB。部分4M稀疏新RSS比NJIT多0.234–0.906MiB；prepared quantile反而少0.406MiB。单类单后端只有一个fresh样本，存在分配器/准备驻留差异，不宣称所有驻留状态领先，也不因正差无限追查。保留全部正负值、idle control与盲区，见 [测量审核](FINAL-MEASUREMENT-REVIEW.md)及summary。未测进程树/PSS/mmap全覆盖。

## 验证与身份

- Python **10387通过**，新增4项参数化回归并扩展原测试，覆盖实际请求/保留容量、reserve、dense→sparse、空/非法mask、任意步长、prepared内容更新和独立快照。
- 原生、ASan/UBSan、双库OFF原生各 **9/9**；OFF受影响Python **147通过**。macOS不支持LeakSanitizer，保留初次unsupported失败，不冒称leak检测通过。
- 185直接算子夹具scalar/auto复比；469案例/3056值历史Microsoft Excel重算快照复比通过。公式和输入不变，仅B4/B9构建身份不同；**本轮没有新Microsoft Excel重算**。
- 源码、wheel、extension、standalone worker身份一致；OFF构建单独核验。运行方式和库版本/线程记录在原始数据中。[validation.json](validation.json)、[identity.json](identity.json)、[manifest.json](manifest.json)。

[benchmark.py](benchmark.py)、[run_memory.py](run_memory.py)与冻结common tool用于复验；测量实际wrapper bytes另存benchmark-executed.py。之后只重标一条继承的SHA注释，可执行代码没有变化，前后及父文件hash见provenance.json。适用范围为本机macOS arm64的这些负载，库OFF正确性不证明其他平台优化性能。

当前采用该守约物理优化；全算子中大型时间和内存明显优于NJIT的目标仍开放。先前Cholesky、LU、有界Jacobi和混合符号大GEMM差距本轮没有重测或关闭；不以几个微小正差阻断，也不隐藏这些实质欠项。
