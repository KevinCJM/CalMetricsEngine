# Production cache-blocked MatMul static review

结论：未发现阻断问题。当前三个生产文件逐字匹配已审核的 `integration.patch` 草案；新对象兑现了预定的尾调用、保序 SIMD、零新增数值工作区。这里的通过仅指静态审查，不代表整请求时间、堆、RSS 或完整验收通过。

## 范围与身份

- 仓库：`/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine`。
- 原基线：`/private/tmp/calmetrics-matmul-wide-square-build`；Build ID `93dcfd0800e6670e40af4b80a439f131f9b96043c25d213a2310a51f6c611fbf`。
- 新构建：`/private/tmp/calmetrics-matmul-cache-build`；Build ID `18977f51909481ec8e88ee73fd24b88ccc83743cc2e82d01c25c2d459547f4a0`。
- 补丁 SHA-256：`1340e50811b49f43335e653c7eb5e924e6883660200b6f2c9accdd65f405ccad`。逐文件起始、草案和当前 SHA 见同名 JSON；三个当前 SHA 都与草案相同。
- 原始反汇编、符号和 undefined 记录：`final-static-{old,new}-{matrix,matrix_backend,simd}.{asm,nm,undefined}`。`ltmp0` 是 offset 0 的 `library_matmul_f`，通过 nm 确认，已纳入栈统计。
- 未运行编译、数值或性能；仅读取源文件、构建规则、Mach-O 对象和现有证据，并反汇编/核对散列。

| 对象 | SHA-256 |
|---|---|
| `matrix.cpp.o` (old) | `0639b05d1548caf10a51e809e47d949c7570dca9761c6b2d62af0e4f4a3759d2` |
| `matrix_backend.cpp.o` (old) | `749654023ecccee37619a541149e4169c05b0d8ff4d9a7a2e3972f7700691118` |
| `simd.cpp.o` (old) | `aa88df3d0aedbe59917c3d0f53b67389f80b30ebb4feb062e54bcd2e3fe48ddb` |
| `matrix.cpp.o` (new) | `7cbd080e490b02dc2b4eb49c27e8f178b25b52e570eaaf9971b7ab32dd23a66b` |
| `matrix_backend.cpp.o` (new) | `a0829a0add978bd7d841da6cef93ae935d2ef307115c3e380aaf454c3468bfc9` |
| `simd.cpp.o` (new) | `aa88df3d0aedbe59917c3d0f53b67389f80b30ebb4feb062e54bcd2e3fe48ddb` |

两版均为实际 arm64 对象，使用 AppleClang 17.0.0.17000603、C++17、`-O3 -DNDEBUG -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off`、最低 macOS 11 和固定 Eigen 3.4.0。旧构建显式带 `-arch arm64`，新构建未显式设置该 flag，但实际对象与元数据都为 arm64；不能声称编译命令字面完全相同。原 SIMD 对象 SHA 完全相同。完整对象规则在 JSON。

## 分派、边界与所有权

`matrix.cpp:175–200` 仅向 EigenON / Apple arm64 / macOS >=15 / automatic ISA 开放。先处理零行步长 A 的 repeated 路径，再判断 C-compatible B (`stride[1]==1, stride[0]>=N`) 和 `K > 65536/96`，即 K>=683；M、N 均至少 32。K682、小矩阵、显式 scalar/neon 等 ISA、F B、未覆盖 B 布局继续原路径。重复行优先次序未变，padded C B 在新路径准入。声明、定义、调用的条件编译一致。

阈值先除常数再比较，不计算可能溢出的 K*96；它是固定工作集启发式，不是实际硬件缓存查询。原 prepare / 输入边界仍负责 shape、inner compatibility、byte span、输出容量和所有权校验，新 helper 没有绕过这些校验，也没有套用 solve-only 工作区加法规则。

`matrix_backend.cpp:124–213` 的 bulk 固定 I32/J64/K64，微块 4×8。full rows/cols 分别向 4/8 下取整，32/64 宏块都是微块倍数；每个 tile 的 4 行、8 列加载与存储都留在逻辑区间内。右侧尾列只覆盖完整行，底部尾行覆盖所有列，两者与 bulk 不相交。A 通过已校验的 signed/zero stride 读取，B 正向 unit-column/padded-row；无临时 Value、owner 复制、输入打包或附加数值数组。A/B 可以共享只读底层存储，独立输出必须与输入不重叠，这仍是既有前提。

指针算式均由合法逻辑坐标组成；负 A 步长不会被转成 size_t 步长。现有维度/ptrdiff byte-span 约束也使固定 32/64 的循环递增远离 size_t 上界。静态证明以现有有效 span / lifetime 契约为前提，不能证明任意外部伪造指针的真实分配有效。

## 数学与成熟库复用

每个输出仅有一条从 +0 开始、按 k 递增的 binary64 乘加序列。后续 K panel 读取本输出上次存下的 accumulator，继续下一项；不是独立 panel 部分和。零系数仍真实相乘，0×Inf、NaN、溢出、次正规数和有符号零不被扫描/跳过。尾部是同样 +0 / ascending-k 原序 fold。边缘优先仅重排独立输出的执行次序，没有改变原内核的错误检查接口。

实际 bulk 热循环 `0x2854–0x28f8` 是 16 个 NEON accumulator 寄存器的独立列 SIMD，显式 `fmul.2d` 后 `fadd.2d`；三个新对象中均未见浮点 FMA 指令。K0 时置 +0 向量，非首 panel 读取先前输出；各 panel 结束后一次写回。新 helper 未见数值栈 spill、outlined evaluator 或 allocator call。四个源码固定 Array 不是 heap workspace；在本对象中它们确实寄存器化。

实质复用的是固定 Eigen 3.4 的公共 Array/Map 借用视图与逐元素表达式 SIMD，项目代码控制分块与独立输出遍历。不是调用完整 GEMM，也没有 Eigen 内部 traits patch、pack 缓冲或隐藏线程池。既有 MatMul 公共容差没有收紧或放宽；局部原序证明与有限值 bitwise 回归不应升级成跨编译器、全域 NaN payload / fenv 的新契约。

## 实际活跃栈和调用链

帧大小包含 `sub sp` 和前置索引 push，未漏算 callee-saved 区。

| 函数 | 旧帧 B | 新帧 B | 新对象地址 |
|---|---:|---:|---|
| `matrix` | 320 | 320 | 0x3f4 |
| `library_matmul_f` | 416 | 416 | 0x0 |
| `library_matmul_c` | 400 | 400 | 0xf8c |
| `library_matmul_c_blocked` | — | 48 | 0x1f54 |
| `matmul_c_blocked_edges` | — | 128 | 0x1fbc |
| `matmul_c_blocked_bulk` | — | 240 | 0x2698 |
| `library_matmul_repeated_rows` | 80 | 80 | 0x2a08 |
| `simd_matmul` | 144 | 144 | 0x14b8 |
| `selected_isa (warm)` | 32 | 32 | 0x1dc |

- `matrix` 在 `0x1120 add sp,#0x140` 恢复 320 B 后于 `0x1124 b library_matmul_c_blocked` 尾调。原 C/F/repeated 分派也仍先恢复帧再尾调。
- 新 driver 对 edges 是正常调用，最多 48+128=176 B；在 `0x1fb4` 恢复自身 48 B 后于 `0x1fb8 b bulk` 尾调。bulk 240 B 不与 driver/edges 共存。
- 因而该算术分派范围内，引擎自有活跃栈峰值为 `max(320,48+128,240)=320 B`。被替换的旧 square C 范围为 max(320,400)=400 B；旧一般 SIMD 为 320+144=464 B，若包含 warm selected_isa 则为 496 B。
- 新 edges 反汇编中的 bzero 只在 `cbz x27` 的 K=0 分支（0x200c、0x2340）可达，当前 K>=683 准入无法进入；bulk 没有任何 call。
- 原 F 416 B、原 C 400 B、repeated 80 B 不变；matrix 原有 d8–d11 ABI 保存存在，不能把整份 matrix 说成“无任何浮点栈访存”。

**边界：320 B 不是整请求或全机器栈上界。** matrix 在系统版本判断时仍以 320 B 帧调用 `___isPlatformVersionAtLeast`，其系统 callee 栈未纳入；repeated 的 memmove 系统栈不能假定为 0；cold ISA 初始化、异常运行库以及上游 prepare/execute/graph/Python/worker 帧也未纳入。当前证据是具体对象中新增算术子链的完整帧叠加及旧路径比较，不是泛化所有平台的上界。

## 原生回归审查

新增原生测试只增补于 `tests/operator_native_tests.cpp`，当前 SHA `441379f95cc7fde596a72d044be77a377cec781d39463e8948d28990ff47b97a`。移除本轮两块新增用例可重构 start-manifest 原始 SHA，原有 126 组合及既有 F/solve 测试均保留。

13 个几何/布局用例覆盖 K682/683/684、M/N31 下界、4/8 尾部、B padding、A 负行/负列/sliced/F/零列步长/repeated、稠密非坐标输入；8 个 IEEE 用例覆盖 K63/64/65 和 127/128/129 的消去、overflow、non-FMA cancellation、0×Inf、denorm、−0 与 NaN。每项 scalar/auto 都执行，有限值逐位及非有限分类/符号对照；另有独立坐标 oracle、输出前后 canary、原输入字节不变、计划和实际 workspace/copy audit。真实可用性由既有 F control audit 判断，不在 EigenOFF 或非目标系统硬断言库标签。repeated 要求原标签和真实 `(M−1)*N*8` 输出复制计数。

仅静态审阅这些测试；不把它们当成本轮已执行结果，也不把原生测试当 Python DAG/CSE、prepared rebind、进程传输、CPU budget 或整请求独立所有权的覆盖。那些需要父任务相应链路结果。

## 内存、审计及剩余验证

新路径没有 heap pack、数值 scratch、数值 TLS、私有线程池、缓存或额外 audit 字段；同线程执行，不改变 Scheduler/CpuBudget。新标签 `eigen-block-matmul` 与原 `eigen-array-matmul` 都是 18 字节，避免已知标签长度变化的元数据混杂。标签等长及局部零分配不等于整请求 heap/RSS 通过。中间 panel 写入/读取真实输出，是额外输出流量，copy audit 为 0 也不能称为“无内存流量”。

尚需父任务独立完成/汇总：本机全量及 ASan/UBSan、EigenOFF/跨平台构建、图与进程所有权、Excel 相关验收、同核数 NJIT 对照与公开入口时间/heap/RSS。当前栈结论只适用于记录的对象/编译器；其它编译器没有 Clang musttail 强制时需要重查生成物。固定工作集阈值能否持续改善相应形状取决于实测，静态审查不作性能通过结论。
