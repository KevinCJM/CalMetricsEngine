# 接入自审核与证据边界

2026-10-04，正式87源与0fe77308隔离版本逐项SHA一致，修改9个原生文件及3个仓库测试文件。数学内核只换内部scratch指针，所有算术、密度库调用、递推/停止及错误检查继续使用同一个latent实现；190算子/AST/Typed IR/逻辑DAG/IPC22/Excel数学与预算保持。

D1/D4/D5：拟合之前扫描未来边和根，追溯全部受支持view的materialized backing；保存仍存活、可证明C连续且完整界限位于当前exclusive arena的owner范围，拒绝重叠或不确定证明。每个arena-backed fit参数验证全部strided可达区间（负/零步长和transpose），一次预留保存、当前fit输出、缺失矩阵及小数组后取得指针，fit期间不得resize。成功/异常均复制回live值及fit输出。外部输入、已写根、旧独立结果不借用；future scatter root必须在fit后、rank2/float64、独立root destination，slot字节/对齐/容量与fit目标不重叠，生产者最终完整覆盖结果。

Native ordinary ingress明确授予新结果复用权限，公开Prepared默认false并非依靠Graph嵌套prebound_inputs猜测；thread捕获随任务完成结束，worker的inline与每请求新建SharedRegion具有独立所有权，失败不发布失败root。RAII恢复thread-local许可；Observe/DeferErrors/isolate/未知输入/scopes不准入。内部C++函数签名改变，需AOT重建，不承诺外部旧SDK二进制兼容。原Planner保留canonical scratch上界，尚未调低预算。保存/恢复复制增加52.008 MB，已含计时与heap，不能宣称零复制。

scatter→argmax保留原scatter时点的索引唯一/边界检查；只允许相邻、single-use、rank2/axis1和input-backed索引，computed索引避免与integer结果slot重用。prepare_geometry仅消费已验证geometry的私有argmax，保留空输入填充、有限值、first tie与错误契约；布局/owner证明不足继续canonical，不新增算子。

正常隔离专项696、真实扩展ASan/UBSan54通过；仓库内测试去除临时bench导入、增加6个大宽度模型case，并通过60项（2项共享内存原沙箱拒绝，允许本地shm后60全部通过；未skip或改契约）。大宽度额外12个old/new子进程对照全部模型字段/诊断和posterior逐位一致，C/F实际eigen_density_calls=2，负步长保持原native路径。完整13根的两代旧/新逐位核验、只读hash、NI 1e-9浮点/离散精确及错误恢复证据保留。

D8：6轮rotating CPU1完整数学计时，图/静态fixture独立准备；标准化、sample初始化、GMM预训、HMM训练和续接预测、回填、全部13输出及普通绑定/plan/return实际计入。12个连续内存任务包括3个heap fresh PID（每后端1个，重复替换3次）及9个RSS fresh PID（每后端3个）。原始fixture构造/graph setup/cold/warm/2代独立输出均在内存epoch，tiny NJIT编译/预热在外。memory原报告有一条notes误写fixture在epoch外，实际construct与measurement_scope证明在内；保留原文件，不修改测量SHA，本文纠正。malloc追踪有Python arena/mmap/非拦截分配器/未跟踪释放盲区，RSS含运行时/JIT基线并采样，均未关闭完整分配审计。

X1–X10：物理arena/root借用不改变逻辑公式、参数、形状、状态或预算，Excel继续在独立下载请求展开并永久保留有消费者的单元格。适用Excel导出回归随全量Python执行，未修改recipe；本轮不启动Microsoft Excel，也不宣称187–190新增范围真实重算通过。用户指南/算子参考签名与数学无变化，无需复制修改；执行指南增加实际复用和Prepared/worker条件，索引及物理验收更新本轮范围。

采用依据：热requested heap较7aba减少103.995 MB、RSS增量中位减少123.601 MB，5.713秒数学链保留约49%的对NI时间优势；RSS较NI少40.468 MB，但requested heap仍多27.159 MB。该候选可按大额内存收益分阶段采用，不能表示整个目标完成；其他主要负载、未覆盖情景、分配审计和真实平台接入仍开放。正式全量与core/SAN终态以integration/acceptance.json及实际XML/log复核。

正式全量12379项无skip通过；native/ASan core各9/9通过。OFF首轮继承EXPECT_EIGEN=1而要求被禁用的库路径，graph测试失败；原log/XML保留。移除该错误环境要求后OFF9/9通过，不修改任何测试断言；后续仍需显式EXPECT_EIGEN=0验证OFF不会执行Eigen，真实ASan60当前运行中。

最终补验：显式EXPECT_EIGEN=0的OFF9/9通过，原断言要求packet计数确实为0；相同87源的OFF AOT扩展60项也通过，无skip，source/native/worker身份已核对。真实SAN扩展60项通过，无skip；完整Python12379及各native/coreSAN/OFF9/9均通过，原失败不覆盖。所有数值/构建进程已终态，当前未关闭whole-goal，尚未提交推送。
