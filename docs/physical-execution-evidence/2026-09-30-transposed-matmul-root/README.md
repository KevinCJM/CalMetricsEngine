# 私有转置 MatMul 根：直接写独立 F 结果

2026-09-30，本地性能阶段；修改未提交或推送。本轮只改变物理结果存储，不改变 MatMul 的升序 K 乘加、转置坐标、错误和 Excel 公式。依照计算纪律 D1/D2/D5/D8，先证明单消费者依赖，再让已有 Eigen/普通 MatMul 核直接写最终结果；没有增加数学内核、线程或进程。

## 设计与自审核

仅对 typed、单根、非错误隔离、无子作用域的 `transpose(matmul(a,b))` 准入；乘积只被该转置消费，根为 float64 rank-2，输出策略是 `F` 或 `auto`。其他根、显式 C、共享乘积、隔离和作用域沿用原路径。`auto` 对此结构选 F。最终 `(N,M)` 的 F 连续字节与乘积 `(M,N)` 的 C 连续字节顺序相同，因此将 MatMul 原有 C 写入器直接指向本次独立结果；转置仍是借用视图。Planner 不再给该乘积和视图保留整块 arena，根复制为零。输入不被改写，结果与后续调用不共享所有权。失败时结果尚未发布，原错误路径不变。

实现自审核重点：必须检查所有图内父边使用次数、单根和作用域边界；不能把共享 MatMul 写入某个根；child 的 `Output.order` 必须保持 C，根布局必须为 F；非准入图仍分配中间结果。现有 Eigen packed MatMul 工作区及复制字节审计保留，不把根直写称为整个 MatMul 零拷贝。Excel 仍按逻辑坐标生成，公式和单元格预算没有变化；本轮没有重新启动 Microsoft Excel 重算。

## 最终 wheel 本机实测

macOS arm64，固定只读 2048×2048 F/F 输入，显式 F 根，CPU 预算 1，七次热完整请求；输入和图在计时前准备，每次产生独立结果。旧/新结果的 C 逻辑顺序 SHA-256 都为 `5688b40d41ade0765b43bf75844a29bdc94172fe10e347358dd2ce97e01f8b34`。[旧](timing-old.json)/[新](timing-new.json)中位数 **753.799 / 753.585 ms**；这组顺序测量显示基本持平，不把不足 1 ms 的变化作为性能门禁或显著提速声明。[已预热同核数 NJIT](timing-njit.json) 为 **83.873 ms**，乘法核约慢 9 倍，整体时间目标仍未达到。旧/新 `numeric_arena_bytes` 与 `root_copy_bytes` 分别为 **32 MiB → 0**，`direct_output_bytes` 为 **0 → 32 MiB**；已有 Eigen 面板 `operator_workspace_capacity_bytes=5,243,456`、算法打包 `569,901,056` 字节未变。

独立 fresh 进程的连续 malloc 探针，epoch 包含图与 Scheduler 建立、首次完整结果，以及保留上次结果的暖执行。冷峰 **72,363,028 → 38,808,596 B**，暖峰 **105,917,910 → 72,363,478 B**，两者均少 **33,554,432 B（32 MiB）**；[旧](memory-old.json)、[最终 wheel](memory-new-wheel.json)、[NJIT](memory-njit.json)原值及[复现脚本](benchmark.py)在本目录。新 C++ 的冷/暖峰还分别比 NJIT 低约 1.04 MiB；NJIT 已编译精确只读签名，但其独立测量不包含 C++ 图建立，因此不能把这个差值推广为所有请求的内存优势。RSS warm 请求增量仅为页级采样，不据其 32/64 KiB 差异判性能。探针不涵盖全部 mmap、栈或 Python arena。

专项 38 组覆盖 C/F 四种输入组合、auto/F 根、single/thread/process-inline/process-shared、旧结果独立、Eigen 大形状逐位及 IEEE 对照、Planner 实际预算，以及显式 C/隔离/多根不准入。最终 wheel 的 Python 全量 **9249 项通过**；原生、ASan/UBSan、双库关闭各 **9/9**。原生首轮沙箱 `shm_open` 权限失败后，在允许共享内存的环境复跑通过。[Python](python-full.log)、[原生](native-ctest.log)、[sanitizer](asan-ctest.log)、[双库关闭](off-ctest.log)日志保留真实结果。另在双库关闭的当前 Python 扩展/worker 执行[37项图与进程专项](off-python-targeted.log)通过；唯一排除的是专门断言 Eigen 打包复制计数的大矩阵测试，不把它计为双库关闭通过。wheel 构建与包内容检查通过，源码增量构建与最终 wheel 的 `engine_build_id` 同为 `f8509c481644af890b14c20936092d781d8f3dda388f02c203436a47e3a98264`；wheel SHA-256 为 `a093fc46d6c829568ec0e8d0e46175525f2f893a15ed41351f0ea12004b3d424`。最终二进制文件的字节哈希不同，不宣称两个构建逐字节相同。

[独立数值复核](numeric-comparison.json)：2048 阶随机输入相对 NJIT 的既有 `rtol=2e-12, atol=2e-14` 逐项口径有 **67 个元素超差**；新旧 C++ 的结果哈希相同，所以这是原 MatMul 数值差距，并非本轮布局改变。没有因此放宽容差或改变离散下游行为。本轮只完成这一结构的显著内存改进；全算子性能与数值承接目标继续开放。
