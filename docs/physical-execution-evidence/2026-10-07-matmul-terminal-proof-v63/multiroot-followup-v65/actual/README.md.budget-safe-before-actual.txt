# V65：混合根布局预算修复与完整请求复验

本阶段为隔离候选，尚未接入实际目录；整体性能目标未完成。build_id 为 `072cba9feee2de6f3c84d4535eacfb60e0b78e3331bc17ad79036391279f7272`。实际目录93份source/config仍匹配V64的 `74bd1460606d2a6a8e4bfab8e2a4725cf46945c2420ef61b29c41945f4a146bf`。

## 自审核发现与修复

修复前候选对原生Program中分别声明C/F的两个矩阵根仍生成共享借用metadata。Planner据此省掉arena，执行发现实际布局不同则恢复canonical arena。独立复现显示planned=0、actual=524288bytes；在更大输出下该缺口按乘积尺寸放大，因此不能接入修复前候选。失败不是几KB性能门禁，而是资源准入与真实分配不一致。

修复在Program::finalize的共享链准入中检查各根声明的order。声明不同则保持canonical metadata和arena，输出接口与布局仍支持；动态raw-native非法/混合slot仍保留既有执行资格检查。重新链接相同复现：planned=actual=524288bytes、根值32.25/15.875不变。新增automatic/C/F的6种不同声明配对×根顺序正反，共12组native预算/结果回归。同声明C/F的原有优化继续使用成熟Accelerate/Eigen，不新增数学算子、依赖、池、公共AST/DAG/schema或Excel逻辑。

## 当前新版证据

- 原生9/9与关联Python4749项通过，无fail/error/skip；关联包含原有逐位置/作用域错误、类型/所有权和线程、inline/shared进程路径。
- 24个fresh完整进程通过：8192×1024乘1024×8192，两条标量尾链返回两个矩阵；C/F连续布局、CPU1、同owned readonly准备、普通Scheduler及已预热最强NJIT，均保留两代独立结果。fixture/graph/JIT不计入，输入准备、实际分配、执行与返回计入。
- 三组轮换计时的组中位再取中位：C++ C/F为0.749400917/0.741356042秒，NJIT为0.646770916/0.654579秒，C++仍慢15.87%/13.26%，不能宣称明显优于NJIT。
- 两布局申请暖峰C++均2,288,002,088bytes、NJIT均2,281,702,136bytes。相对前一epoch旧C++基线减少533,205,448bytes，约508.5MiB；C++仍多6,299,952bytes，此库资源如实保留，不因该小额差异单独否决。旧/新不是本轮同时配对计时，主要性能比较为本轮CPP/NJIT同条件配对。
- 增量RSS C/F：C++为2,227,732,480/2,227,568,640bytes，NJIT为2,233,614,336/2,227,355,648bytes；页面与allocator差异只作诊断，不能据此宣称显著NI内存优势。全部两代全单元核验最大误差0，readonly、无输入/根间/跨代别名及旧结果保护通过。
- 本新版全量回归正在运行，尚无最终计数。旧候选的正常16720项及本轮ASan/UBSan原生9/9、关联4749通过属于旧build_id，只作为版本化证据；不能替代新版SAN/OFF资格。漏检的布局预算问题已单独保留，无旧绿灯接入。

## 还差什么

新版全量、SAN/OFF；实际源码接入、同源wheel/worker与实际完整请求确认；更强NI时间/内存优势，以及其他重要平台计算链。未提交、推送、合并或部署。本阶段只证明此共享矩阵任务，不外推到全部情景/归因/有效前沿/随机空间业务链。普通C++请求不执行Excel，未做真实Excel验收。

[预算修复差异](budget-fix-diff.patch.txt)、[复现前](mixed-root-budget-probe.json)、[复现后](mixed-root-budget-probe-fixed.json)、[全部测量](budget-safe-large-results.json)、[配对汇总](budget-safe-measurement-comparison.json)、[源码与测试冻结](budget-safe-snapshot.json)及[当前进度](progress.json)分别保留。
