# 重型固定分组筛选（2026-10-02）

本轮只筛选私有固定物理 tile：生产四行×八列，对照 G8W4、G16W2、G8W8。没有接入候选。77 个生产身份文件与两份生产 header 均保持基线字节，build 仍为 `ccca6b67a5220ea2134a4743d281b80c5af4c096e7d41c3d5b747bb8de70e331`，依据 [identity-baseline.json](identity-baseline.json) 与 [freeze-provenance.json](freeze-provenance.json)。未提交、推送。

Chol G8W4 在 4096 C/F 输入仅节约约 17.6/25.3 ms（1.5%/2.0%），没有选作重点接入；LU 三个案例均未提速，生产实现保留。几 KB 或几 ms 的变化不能独立构成失败门禁；本轮没有关闭中大型时间、内存目标，也不声称整体领先 NJIT。

| 最终局部计时（ms，中位数） | 原分组 | G8W4 | NJIT |
| --- | ---: | ---: | ---: |
| Chol 4096 C 输入、F 输出 | 1194.626 | 1177.002 | 379.361 |
| Chol 4096 F 输入、F 输出 | 1300.667 | 1275.331 | 434.782 |
| LU 2048×8 C 输入 | 319.251 | 331.839 | 81.817 |
| LU 4096×8 C 输入 | 2399.072 | 2479.432 | 622.879 |
| LU 4096×8 F 输入 | 3499.774 | 3531.364 | 642.418 |

Chol 的 NJIT 为 `np.asfortranarray(np.linalg.cholesky(a))`，包含指定 F 输出的布局物化，只是该局部案例参考。LU 原生输出为 C，NJIT 输出为 F，其检查只证明逻辑结果对照；以上 LU 列不能用作同输出布局的采用验收。G16W2/G8W8 的 LU 原始样本也全部保留在 [lu-probe-timing.json](lu-probe-timing.json)。

最终证据分工：

- [screen.json](screen.json) 与 [chol-correctness-all.log](chol-correctness-all.log)：58 fixtures、464 ABI calls、348 候选比较行，覆盖全部三候选、C/F 输出、失败类别和完整输出 buffer；全部通过。
- [timing.json](timing.json)、[chol-timing-checked.log](chol-timing-checked.log) 与 [threads-restored.json](threads-restored.json)：实际只运行 4096 C/F、old/G8W4/NJIT 六轮，每次原生调用检查 status，warmup/final 逐位对照。脚本默认七案例不代表本次执行范围。
- [lu-probe-screen.json](lu-probe-screen.json) 与 [lu-correctness.log](lu-correctness.log)：4144/4144，四种舍入模式，成功值、错误及失败前缀；非 NaN 位严格一致，NaN 类别一致，指定 RHS 失败前缀另查复制位。
- [lu-probe-timing.json](lu-probe-timing.json) 与 [lu-timing.log](lu-timing.log)：三个案例、每后端十轮，轮序轮换/反转，独立结果保留与替换，逐次原生 status 和 warmup/final 结果核对。JSON 保留 NJIT 实际单线程池、Accelerate active=1/restored=prior 的证据。

[初始七案例 JSON](diagnostic/chol-timing-pre-review.json)、[日志](diagnostic/chol-timing-pre-review.log) 和 [当时脚本](diagnostic/chol-screen-pre-review.py) 仅为诊断历史：当时计时 callable 丢弃原生 status，已由最终 checked harness 修正；不能将初始七案例充作采用验收或最终 checked 结果。本轮没有运行 malloc/RSS、完整 Scheduler/Planner 性能、新 wheel/全量回归或 Excel 重算。

复现使用 [最终 Chol 脚本](chol-screen.py)、[最终 LU correctness 脚本](lu-screen.py)、[最终 LU timing 脚本](lu-timing.py) 和本目录候选 header/probe。将所需文件复制到独立临时目录，保持 `historical/prototype-fixtures/` 相对路径；依 [freeze-provenance.json](freeze-provenance.json) 的实际 Chol 编译命令、checked Python body，以及 LU 有效参数重建并运行，替换临时 stage 路径。原生包必须匹配基线身份。实际脚本从记录的 CME 仓库路径导入既有线程工具，应核对 [工具快照](support/tools/check_large_matrix_performance.py) 与其 [导入依赖](support/tools/benchmark_memory.py) 的 SHA；本轮只使用线程控制，没有调用内存测量函数。不要在冻结目录写入复现输出。

[lu-probe-build.json](lu-probe-build.json) 原样保留编译时记录，其 `numeric_execution=NOT_RUN` 是当时状态，脚本 SHA 也早于最终 harness 修订。最终执行范围见上述 JSON/log；当前 harness SHA 与编译时 SHA 在 [freeze-provenance.json](freeze-provenance.json) 分列。dylib 只记录实际 SHA，不复制二进制、大段汇编或重复设计报告。

[manifest.json](manifest.json) 记录全部冻结文件的字节数和 SHA256；复用已有 [verify_manifest.py](verify_manifest.py)，可用 `python3 verify_manifest.py` 做纯文件校验。
