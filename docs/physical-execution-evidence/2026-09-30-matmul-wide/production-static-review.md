# MatMul 宽列块：方阵准入后的实际对象静态审核

2026-09-30。仅检查已构建 arm64 对象、符号、配置与反汇编；没有重新构建、导入/执行数值引擎或启动计时。顶层 `new-*.asm` 和符号表来自 `/private/tmp/calmetrics-matmul-wide-square-build`；旧对象仍来自 `calmetrics-matmul-build`。身份、精确源码/对象/二进制 SHA 与实际旗标见 `production-static-identity.json`；机器可读结论见 `production-static-summary.json`。

历史证据没有覆盖：529 候选在 `production-static-candidate/`，16d2 在 `candidate-16d2/static/`，上一轮 d2e734 的原始反汇编与 static 文件在 `candidate-d2e/`。本次逐项核对 d2e snapshot manifest 中相关11个文件 SHA，全部保持原样。

## 最终准入与反汇编证明

非重复行 C4×8 只准入 **M==K==N、严格 C 布局的 B、N%8==0**，仍要求 auto/Eigen/Apple arm64/macOS15+、各维>=32。矩形、padding 与尾列回到原 SIMD；F 右输入保留上一轮 4×4 库路径。零 A 行 stride 的精确重复行路径位于方阵限制之前，仍支持符合 C/F 兼容布局的矩形、padding 与尾列。方阵限定是经验验收范围，不表示已证明任意方阵必然更快。

实际 `matrix.cpp.o` 指令：

- `0x10dc cbz x8,0x19f8`：零 A 行 stride 先进入 repeated-row 分支。
- `0x10e4/0x10e8` 分别载入 B.stride[0] 到 x8、N 到 x9；`0x10ec cmp x8,x9`、`0x10f0 b.ne 0x19bc` 强制严格行距。
- `0x10f4 tst x8,#7`、`0x10f8 b.ne 0x19bc` 检查整8列；此前已证明 x8==N。
- `0x10fc ldp x10,x9,[x19,#0x10]` 载入 M、K；`0x1100 cmp x10,x9`、`0x1104 b.ne 0x19bc` 检查 M==K。
- `0x1108 cmp x9,x8`、`0x110c b.ne 0x19bc` 检查 K==N。
- 不满足时 `0x19bc` 检查 F 兼容，否则进入既有 SIMD 分派；没有跳过计算或改变结果数学。

当前全部 cpp/CMakeLists/pyproject manifest 对照 d2e734，**仅 `cpp/operators/matrix.cpp` 改变**。`matrix_backend.cpp.o` 和 `simd.cpp.o` SHA 与上一版完全相同。native 测试另从90组合扩展为126组合，测试文件新 SHA 已独立记录；测试不属于引擎 build-ID 源清单。

## 本地活跃栈与尾调用

| 层级/路径 | 旧 | 新 |
| --- | ---: | ---: |
| `matrix` 自身 | 320 B | 320 B |
| 已准入 C 数学循环活跃本地 frame | 320+144=464 B | 400 B |
| C 含 warm ISA 选择的本地峰值 | 320+144+32=496 B | 400 B 数值路径 |
| 矩形/padded/尾列 C 非重复行 | 464 B 数值 / 496 B 含 ISA | 464 B 数值 / 496 B 含 ISA |
| F helper/数值活跃本地峰值 | 416 B | 416 B |
| repeated-row helper 本身 | 原数学路径 | 80 B，另有未知系统 memmove 栈 |
| `simd_matmul` / `selected_isa` | 144 / 32 B | 144 / 32 B |

三个库入口均在 matrix 恢复自身 frame 后尾调用：C `0x1140 add sp,sp,#0x140 → 0x1144 b library_matmul_c`；F `0x19f0 → 0x19f4`；repeated `0x1a28 → 0x1a2c`。所以不能把 matrix320 B 再与 C400/F416 叠加。回退 `0x1568 bl simd_matmul` 保留 matrix frame；旧相应普通调用在0x1524。SIMD 内部 `0x14ec bl selected_isa` 保留144 B，因此464不是整个 warm 本地链峰值。

这些数字以 matrix 为起点；共同 execute/graph/PyBind 外层没有计入。系统版本查询/冷初始化、异常运行时和 memmove 不在这些对象中，不能将系统 callee 视为0 B。重复行完整峰值仍未知，正确表示是 `max(已知分派链,80 B+memmove未知系统链)`，不能宣称整条请求仅80 B。

## 数值 spill、FMA、分配与线程边界

backend对象逐字节与 d2e734 一致，因此此前逐指令审核直接适用；本次另核对原始反汇编与符号映射：

- F/C/repeated helper 的浮点FMA指令数均0；热循环是分开的 fmul/fadd，独立输出仍按递增 k 累加。
- 三者浮点/向量寄存器对 sp/x29 的 load/store 均0，除建立frame指针外没有栈派生数值指针。固定Eigen累加器未落成数值栈数组。几何/指针整数 spill 存在，不能称整个内核无spill。
- C4×8 内环没有 outlined Eigen evaluator 调用；输出在累加后写出。
- F/C 唯一调用引用为 bzero，仅位于 K=0 路径。C 中 K 在0xfcc进入x26，清零块由0x13ac/0x1728/0x1be8的cbz控制；K>=32准入排除它们。F指令字节与旧1aec相同。
- repeated 的 bzero@0x20a8由K==0分支控制；memmove@0x2138实际执行，复制已完成首行到独立输出行。
- 三个数学helper没有直接malloc/new/aligned allocator引用或新增线程入口；matrix既有异常分配引用仍在。不能从直接分配引用为0推出完整请求heap/RSS为0，亦不能据此证明系统memmove没有内部资源使用。

`llvm-objdump` 将 backend 地址0函数显示为 `ltmp0`；`nm -nm` 将同地址映射为 private external `library_matmul_f`，本次未漏算。`production-static.json` 共10个函数记录，包含旧/新F与ISA选择。F helper指令字序列SHA不变，SIMD整个对象SHA不变。邻近的MatVec240 B、covariance368 B和MatrixRotation0 B frame也因backend对象不变保持原值；不据此重新宣称它们的运行验收。

## 构建身份与配置

- 旧引擎：`1aec1a6b0d4a07f0caee43b58cd4b339bdcc165f5e615f2190b15da0ba66e86b`。
- 新引擎：`93dcfd0800e6670e40af4b80a439f131f9b96043c25d213a2310a51f6c611fbf`，依据当前全部cpp源码、CMakeLists/pyproject和实际CMake配置静态重算，并核实出现在 `square-install-new/.../_native.cpython-312-darwin.so` 字节中。
- `calmetrics-matmul-wide-square-wheels` 的wheel内native二进制SHA与安装文件相同；没有通过import或运行引擎取得此结论。
- 新旧使用等长54字符的 `square-install-old/new` 目录；这属于运行测量边界控制，不从静态证据推断运行速度。
- 实际 `/usr/bin/clang++`，AppleClang17.0.0.17000603，CMake及对象架构arm64。
- 正式旗标：`-O3 -DNDEBUG -std=c++17 -mmacosx-version-min=11.0 -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off`。SIMD TU既有重复ffp-contract保留记录。define为`CALMETRICS_HAVE_EIGEN_ROTATION=1`，Eigen3.4.0 include来自各自build/_deps。没有把原型显式-arch旗标冒充正式ninja旗标。

| 对象/二进制 | SHA256 |
| --- | --- |
| old matrix.cpp.o | `b0ead9b7e8bc5b128378fca0f0905e256c8cf43a99bec5dc7f87c86e2f34b746` |
| new matrix.cpp.o | `0639b05d1548caf10a51e809e47d949c7570dca9761c6b2d62af0e4f4a3759d2` |
| old matrix_backend.cpp.o | `ce177a68dcee412b37ac53aa4d7e9e9cca72bf7d47019a7314ecfaf0a25e01e4` |
| new matrix_backend.cpp.o | `749654023ecccee37619a541149e4169c05b0d8ff4d9a7a2e3972f7700691118` |
| old/new simd.cpp.o | `aa88df3d0aedbe59917c3d0f53b67389f80b30ebb4feb062e54bcd2e3fe48ddb` |
| new installed native | `5cb379738bca52028f082b897647e4001e6d3a9a358641f7a299f255bf1d633e` |
| new wheel | `bde011f93a5c181c3ad209fdfaee4da34860917b067cb4a407743589d5e48ab6` |

本次只验证静态身份、准入、调用链与对象边界；数值、完整请求时间、heap/RSS及Excel结论由父任务独占验收流水提供。矩形地址/长步长敏感性的假设仍在 `rectangular-attribution-review.md` 中保持未定因，不用方阵收窄掩盖原失败记录。
