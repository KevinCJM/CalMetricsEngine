# V60：复用 vForce 的固定轮数 HMM 状态扫描

2026-10-07，macOS arm64、本机 Release、CPU1。实际目录已接入一个 C++ 文件与34项新回归；未提交、推送或部署。整体性能目标仍未完成。

## 研究与取舍

参考 [Apple vForce 官方接口](https://developer.apple.com/documentation/accelerate/vforce-library)、[hmmlearn 原生实现](https://raw.githubusercontent.com/hmmlearn/hmmlearn/main/ext/_hmmc.cpp)及[HMM 线性内存研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC2430973/)。沿用已链接 Accelerate，没有 Python 数学回调或新增依赖。

最初 exp/log 批量版和保留 scalar log 的 exp-only 版都在既有 `test_strict_stopping_neighbors[4097-12]` 失败，各另20项通过。停止轮数改变会放大为模型差异，不能用浮点尾差许可忽略。两个候选的源码、身份和失败记录保留，旧断言与契约未改。最终版本仅准入 `tolerance==0` 的固定轮数拟合；正容差保持原扫描，现有 xi 优化不变。没有因这一增量路径缩小总体负载目标。

## 实现与自审核

私有 `library_hmm_row` 将 K 个独立状态的 K 项指数聚为一个 `vvexp` 调用。前向按目的状态、后向按源状态分组，维持原左结合加法、升序求和、scalar log、emission 加法及参数更新。只复用 E 步前闲置的 xi K×K、当前历史行和已有 K 临时区；无新增工作区、输入打包、复制、线程、缓存或别名。所有库调用计入已有 `accelerate_vforce_calls`。

准入：Apple arm64/macOS15+、非 scalar、nearest rounding、N≥4096、8≤K≤128及零容差。计算表达式超过 DBL_MAX/32 时，在改写 alpha/beta 前拒绝该行的库路径并执行现有有序扫描；这是物理准入，不是新错误。既有 loan/2NK+密度重放、滚动 beta 复制/复位、gamma/xi、中心化方差、下限与模型发布顺序不变。可移植、小形状、非 nearest 和正容差路径继续具有实际用途。

Excel 的固定轮数 EXP/LN 和 E/M 组合数学没有变化，维持原公式、展开预算和普通请求惰性导出。全量测试包含导出回归，未进行本轮 Microsoft Excel 实机重算；AST/DAG、Planner、dtype、进程协议和公开算子均无新语义。

## 正确性与构建

- 正常完整 Python 16260，ASan/UBSan 相关 791，关闭库相关 791；三个构建原生均9/9，零失败、错误和skip。新增34覆盖准入边界、C/F/反向/带padding、轮数/分类/排序、极端概率、四舍入模式、复用/canary、输入只读、两代结果及 thread/inline/shared/prepared。
- 首轮新31+既有43共74通过，后补概率/对称3项通过；最终34完整进入全量及SAN/OFF。没有把初始74写成当时已执行77。
- 实际目录重建后另跑77项针对性检查和原生9/9；这是实际执行路径确认，不是第二次完整或sanitizer验收。当前93份source/config与资格候选一致，wheel/worker ID为 `43311a4b2baedc23731b83be95cfc9b8812f5e6f9cb9633bef6c2b23ccfea898`。
- 262144×8/K32/F/2轮训练全部8输出，及500000×4/K12/C/4轮固定识别全部13输出对原实现和暖NJIT核对：离散精确，最大浮点差分别2.22e-16和1.92e-12。13输出包含训练、标准化、预训练、过滤、重排/回填、分类和参数。对称/平局另检验离散决定。默认K32十三根图的既有绑定展开预算失败保留，没有改变编译上限或据此宣称该负载通过。

## 完整任务性能与内存

计入双方相同的独立只读输入准备、普通执行及结果返回，保留前一代输出；JIT/图编译、夹具生成及测后比较不计入计时。库/NI线程均1，未与本任务构建/测试并发。13根固定版本在图和NI参考两侧同时显式设HMM容差0，GMM预训练不变，不冒充平台默认容差。

| 完整请求 | 原C++秒 | V60秒 | 暖NJIT秒 |
| --- | ---: | ---: | ---: |
| 262144×8/K32/F、训练2轮、8投影（首组三次计时中位） | 9.5332 | 7.7910 | 19.6943 |
| 524288×8/K16/C、训练2轮、8投影（首组三次计时中位） | 4.7846 | 4.0311 | 11.3573 |
| 500000×4/K12/C、固定4轮、13输出（三次fresh交替组中位） | 4.2675 | 3.7352 | 9.1947 |
| 同形状、默认正容差、13输出（三次fresh交替组中位） | 4.2710 | 4.2692 | 9.2305 |

训练另fresh全输出组为9.1306/7.5658/18.9593秒；固定13根首个全输出组4.2997/3.6921/9.1041秒。实际目录另fresh时间/内存确认见 `raw/actual-sample-summary.json`，不能把各组最优拼成一组比值。8根NI独立 `sum(w)` 使用有序Python适配，重型E/M为NJIT；K≤32的适配量不构成秒级收益，但边界如实保留。

262144×8/K32/F/2轮同完整训练、各backend一个fresh malloc和一个fresh RSS进程、各三代：原/V60申请峰值均 **151,116,912 bytes**，NI **285,269,146 bytes**。V60无新的堆节约，已有history复用的约128MiB/47%优势保留。RSS增量原/V60/NI为134,676,480/134,791,168/268,533,760bytes；原/V60的约112KiB差仅诊断，不作门禁。各自post-import baseline不同，不比较绝对RSS；单次内存采样不宣称跨机普遍结论，malloc表无overflow。

申请堆初始轮因上一回合中断，完成了三个heap与原实现RSS后停止；原candidate RSS日志为空，未记通过。确认无相关进程后，仅补齐两个缺失RSS，使用 `*-resumed.*` 文件，不覆盖中断证据。

实际wheel复测首个训练时间子进程成功，外层脚本误从顶层读取 `accelerate_vforce_calls` 而触发KeyError；现有协议将它放在 `native_chunks`，没有改引擎协议。保留初始脚本和失败JSON，修正读取及精确新调用数校验，复用该已完成测量/输出，只补齐其余时间、heap和RSS。不能把外层失败写成首轮已全面完成，也不把它当成C++数值错误。

## 复制审计与未完成

8投影 V60 算子workspace134,272,000bytes、numeric arena13,312、algorithm copy134,230,272、root copy12,569；13根loan链分别16,023,872、77,002,464、77,011,328、2,032bytes。新扫描不增加任何这些量，但整算法不是零复制；输入准备副本也在完整内存测量内。

现有大矩阵退化、其他平台完整链、正容差扫描加速及全面分配/复制审计继续开放。下一大额内存候选是非loan的 alpha 历史检查点，研究/设计见 `raw/research-and-review.md`；尚未实现或验收，不能把约63MiB几何估算写成实测。已采用的 emission→beta 不能再次当作新节约，也不重复采用已因秒级重算退化被拒绝的V39/V40 beta检查点。

## 复现与证据

`raw/` 保存完整方案、初始失败、构建身份、全部计时/峰值JSON、log/XML、自审核、只读采样、接入守卫及脚本；`source/` 保存资格源码、Python调用/参考/测试和本地vendor归档。代码快照追加 `.txt`，避免CodeGraph把历史证据当实际实现。复现时复制到隔离目录并删除**文件名末尾** `.txt`，保留相对目录，按身份中的相同离线构建配置构建/安装；运行 `train_benchmark.py.txt`、`fixed_chain_benchmark.py.txt` 或 `chain_benchmark.py.txt` 时明确传 `--install`。测试与参考来自快照，需将脚本的原root路径指向复现根；完整工作区源码必须与身份中的93份hash核对。

大数组NPZ未复制进仓库，可由固定种子、冻结参考和 `run_parity.py.txt` 重建；测后逐元素比较的报告保留。原实现复现用相同源码，替换为 `raw/actual-before-latent.cpp.txt`，以 `raw/actual-before-identity.json` 核对。资格JSON的 `actual_adopted:false` 表示测试当时尚未接入，当前接入与实际确认单独列出，历史状态不重写。
