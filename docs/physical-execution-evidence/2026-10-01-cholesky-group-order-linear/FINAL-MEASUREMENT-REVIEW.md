# 本轮最终测量脚本只读复核

结论：没有发现会使当前 35 项计时、13 项专门内存测量设计失真的执行问题。冻结哈希文字不一致已由 root 修复并保留原执行字节。35 项计时的 correctness 均为 true；13 项/78 个新进程内存测量正在执行，本文不宣称已完成。未执行数值、构建或测试，未修改脚本/仓库。

下文 stage 行号对应保留的 `benchmark-timing-executed.py` 原计时版本；修复后的 benchmark.py 仅多一行来源注释。

## 需修正的溯源文字

- 原 `benchmark.py:21` 标注 common source SHA256 为 `9d5fe138…`，实际为 `2f99bf41adac0926e30aa2045fa0f6bb67ea938826bf6a4c69ed6cf278c10518`。root 已在当前 benchmark.py:21–22 分别注明 source wrapper 与 common tool 的 SHA，保留 `benchmark-timing-executed.py` 和 `benchmark-provenance-repair.json`。静态 AST 相同；原执行 SHA 为 `6b159a335470939b39705b81a3a7626940bb86e14a01c46da366369a4dfd49f9`、修复后 SHA 为 `be19be8fd9b20d36eadff1ab0708587e27d6526c093269904896ce9222997ae5`。stage 与仓库同名 common tool 字节相同，byte-identical 说明成立，此项已关闭。
- `tools/benchmark_memory.py` 本次 SHA256：`b3862bb22e58f01f874bad0fd8466a3eb63ac4f3803186846184aafb47f4d7f8`，stage 与仓库相同。

## 完整结果及同 CPU

- `benchmark.py:174–177` 的普通入口调用完整 Scheduler.execute；prepared 调用 run_snapshot。两者在 callable 内构造 `MaterializedResult`。工具 `:222–231` 在计时/heap/RSS 内读取全部 declared outputs，同时保活 raw result，没有在 callable 外遗漏延迟 ndarray/scalar 包装。
- 工具 `:346–359` 在计时前比较两次完整结果、检查输入未变和独立所有权；stage `:183–211` 对旧/新完整输出做 uint64 位对照，新 C++ 必须 changed_bits=0，NJIT另核对浮点、有限性、NaN、符号和零。Cholesky 重建只抽样 81 个系数，报告明确注明范围，不能写成全矩阵重建检查。
- `.npz` 是每个当前 case 的完整输出交换文件，后续 case 会覆盖；JSON保留每项输入/输出摘要与验证结果，不应宣称每项完整数组都被永久保留。
- 工具 `:120–164` 对 NJIT参考进程设置并读取 Accelerate active=1，加载 SciPy 后再限制/记录所有已发现 pools；native 进程使用自身 cpu_budget=1。它不是仅靠环境变量推断线程。
- 已只读查看完成的 `formal-timing/timing.json` 共 35 条，correctness 均为 true：NJIT记录 Accelerate active=1、OpenBLAS num_threads=1；old/new audit 均 cpu_tokens=1、native_threads=1、lane=single。原始 CPU/wall 样本已保留，cleanup 报告三个 worker 均正常 stopped，NJIT Accelerate restored=0 等于 prior=0。
- 工具 `:467–476` 串行激活三个 backend，9 轮轮换首位；每轮旧结果仍在 RHS 调用期间保活，原始 wall/process CPU 样本完整记录。

## 输入与内存口径

- stage `:69–74` Cholesky 仅分配一块目标矩阵，原地写随机数和对称三角；abs/sum 为 O(N) 行临时。`:110–125` 输入 SHA 对 C 直接读 buffer、F 每次最多一行复制，没有释放整矩阵生成/哈希临时来掩盖正式 RSS。
- 掩码 density 在 `:88–104` 生成时确定，先生成实际 mask 再写非消费 NaN；RNG、反掩码等临时均最多 8192 元素。1%/55%/全真各自有 density ID，不修补已插入 NaN 的 mask。
- `run_memory.py:20–32` 的 13 个完整请求过滤 ID 对当前 case 列表唯一，prepared/C/F/density彼此分开。`:43–44` 拒绝复用已有证据目录；性能退化不使驱动删去报告或停止后续项。
- `benchmark.py:83` 的 reverse 输入仍会复制后释放一整条数组；当前 13 项正式内存列表没有 reverse。上述 bounded-RSS 结论仅适用于这 13 项，未来测 reverse 内存需另修生成方式。
- 工具 `:385–412` 各 case/backend/heap/RSS 在新进程执行；输入、import 与 NJIT实际签名 compile 在 epoch 前，无数值基准 warmup。native graph/setup/prebound准备、首次库工作区均纳入测量。它是首次请求内存，不是包含依赖导入及 NJIT 编译的冷进程总内存。
- memory 工具 `:33–52` 持续 heap epoch 覆盖 setup、cold、首次结果释放、warmup、warm执行和释放后的常驻值。warm `result=function()` 在新结果返回前保留上次完整结果，最多两份结果；RSS `:100–103` 同样如此。stage 只把 repeats 改为3，没有截断输出或把数值 warmup移到epoch外。
- heap stage peaks 已包含当时 live setup/workspace，不能相加；完整阶段峰值应从对应 setup/cold/warm_baseline/warm峰值取最大。RSS 记录 baseline、peak、delta、idle-control并保留页级观察属性；不等同 malloc量，也不能只因正差就判泄漏。

## 严格诊断与采用结论

这四个脚本保存原始时间/内存证据，不把严格 time/heap/RSS诊断 flag 自动转为采用结论。最终 root 分析仍需逐项保留冻结失败，报告相对值与绝对 ms/bytes、输入规模和预算；几 KB/几 ms变化不能单独否决，显著中大型时间/内存缺口也不能用平均值或小差异规则掩盖。静态脚本复核不代表总体 NJIT目标已完成。
