# 混合符号 GEMM 数值守卫筛选：不接入

2026-10-01，macOS arm64、单 CPU，基线 wheel 为 `6f02e6bb69d2d59a62781c357eda16f9604ae9eee2377af383b3b56b29c2bf6f`。本阶段仅隔离候选，生产代码没有变化。此说明是在试验完成后整理；不将候选筛选写成引擎验收。

## 问题与候选

混合符号乘法存在抵消，裸 DGEMM 在近零结果上不能证明沿用既有逐格归约的容差及正负/零分类。本候选保留原 `dot_value` 作为不确定格的重放，并借用成熟 Accelerate DGEMM/vDSP：有界结果 tile，按 K 分段；Cauchy 上界估计绝对积和，将 canonical/库误差随部分和推进。不能满足原 `2e-12/2e-14` 或符号/零保证的格重算。受控指数、有限性和布局完整预检先于任何写入；不修改生产容差，也不改金融含义。

tile 128/256、K 段 16/32/64/128，候选显式 workspace 为 395,264 / 1,576,960 B；这是向量申请字节，**不是完整 heap/RSS 峰值**，厂商内部 workspace 未完整测量。有限样本与标准浮点误差模型不是所有工具链上的厂商数学证明，异常时线程恢复也未达到生产验收；这些缺口不能省略。

## 结果与取舍

99 个筛选通过：C/F 四组合，随机、抵消、稀疏、负零、缩放、极小结果，四种 K 段；另有尾部非有限/越界值在写入前拒绝的 canary。1024 阶六候选也通过原容差和数值分类对照。

| 1024×1024 输入 | 当前完整普通 C++ | 热 NJIT | 最快候选 | 候选 / 当前 |
| --- | --- | --- | --- | --- |
| CC | 96.293 ms | 8.321 ms | 337.253 ms | 3.50 |
| FF | 95.907 ms | 8.569 ms | 380.627 ms | 3.97 |

八轮均衡轮换，输出独立并保留；Numba/系统 BLAS 均限制一线程。六候选重放约 10.9%–40.1% 的结果格，逐格回读和守卫推进抵消了库收益。当前取舍是**不接入**，理由是中等规模已有 241–285 ms 实质退化，与几 KB/几毫秒门禁无关。继续细抠 workspace 没有解决主要矛盾；完整 heap/RSS 测量未开展。

首次 512 CC 的候选 25.076–60.430 ms，原 C++ 11.453 ms、NJIT 1.198 ms。随后 NJIT 与原 C++ 的 1024 近零对照有 4/1,048,576 格超过原容差，最大绝对差 `9.95e-14`，初次退出 1。最终将该 **NJIT 参考差异**单独记录，候选仍须通过原容差/分类；不把参考的不同归约结果作为新实现 oracle，也不宣称 NJIT 全部同语义通过。

最终 1024 CC/FF 测量完整；2048 CC 在已确认明显退化后主动中断，进程确认退出 130。2048 没有完整六候选中位数，不能写成通过或失败的全量验收。原始失败/中断日志、样本与完整已完成结果保留。

## 复现、自审核与边界

`candidate.cpp` 为隔离 ABI；`screen.py` 为测试参考，不属于产品 Python 数值执行路径。脚本中的数组由测试持有，不能据此宣称任意 C 调用方的地址、别名或缓冲区范围都已验证。初次执行中整理过脚本排版，初次源码身份未冻结；以最终运行与 `identity-closeout.json` 分别限定，不借当前哈希证明初次文件原样。

```bash
/usr/bin/clang++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off -mmacosx-version-min=15.0 -dynamiclib -Icpp/include docs/physical-execution-evidence/2026-10-01-mixed-gemm-guard/candidate.cpp /private/tmp/calmetrics-covchol-next/wheel-build/libcalmetrics_engine_operators.a -framework Accelerate -o /private/tmp/calmetrics-mixed-gemm-guard.dylib
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 docs/physical-execution-evidence/2026-10-01-mixed-gemm-guard/screen.py
```

复现脚本包含原计划 2048 控制，可能耗时很长；此记录只引用实际完成的形状。`summary.json`、`timing.json`、`correctness.json`、`initial-screen.log`、`screen.log` 与编译/身份文件为依据。没有更新生产 AST/DAG/Planner 或 Excel，未重跑全量功能回归，不将上一阶段 9,966 项作为候选验收。整体优化目标开放，未提交或推送。
