# Ordered Eigen producer：源码、ISA 与运行证据边界

日期：2026-10-03。本页只整理已存在的源码/构建/反汇编/JSON，没有执行编译、测试或数值任务。

## 1. 唯一数学体与准入

生产 `cpp/affine_projection.cpp` SHA：`ff0119105ae53af372cb74f4426ef517f588a3404bf5eb481def15f451011003`。
共同 `cpp/operators/affine_contract.hpp` SHA：`1afc6798ff05a26e1bf3bca7768083f0e29b57b8925ca15cdd79982d293da38e`。

- `affine_projection.cpp:26`：非空 row、F>0、A≥8，`F <= 16384/A` 避免先乘溢出；库编译可用，执行还要求 `FE_TONEAREST`（:461）。
- `produce_asset_row`（:33）在固定 width8 Array 内跨独立资产 lanes 更新。两条路径都实例化 `ordered_fold`（`affine_contract.hpp:13`），factor 升序、initial-first、系数左乘，具体 product 后相加，剩余资产使用同一 scalar 数学体。
- `std::expm1`仍逐资产调用；没有 vvexpm1、GEMM、零系数省略、factor subtotal 或新并行框架。
- geometry bound（:209）保守预留 F×A doubles，最多 128 KiB；实际 K 资产 stride≠1 才 pack 到最大 workspace 尾部（:467）。多区间复用仅限同次静态 K；下一次 prepared/ordinary 执行重新读内容。metadata、reserve 与 actual capacity 保留原 Planner/audit 计量。
- 未能证明 closure、额外 observer、isolate/scopes 使用 canonical；非最近舍入、禁库、窄资产、F0、空轴或 cap 超限使用 scalar duty 或既有空输出逻辑。

数学、别名、pack 尾区和失败后的恢复证明见[静态复审原文](eigen-production-review.md.txt)。这里只记录来源与推论，不保存第二份可执行数学 C++ 实现。

## 2. 实际生产 object 指令

[production-affine-isa.json](production-affine-isa.json)保存根代理读取的元数据，完整 [反汇编](production-affine-disassembly.txt)与 [编译命令](compiler-command-eigen.txt)保持原文。

| 证据 | SHA-256 |
| --- | --- |
| 实际生产 wheel-build 的 `affine_projection.cpp.o` | `3d229599bb270c585e44241014190fb865b0ef9aa3d72dcc6291a210d637bbf2` |
| 生产反汇编文本 | `25e75427b93a6f2ac5fa799f35dd72fcb96a97cd486261c676d9873e1720ee63` |
| 生产编译命令文本 | `950dd790bee3450adf796e90fe96b8b5f45b1d1505fa0d357ea7c7748a0a9894` |
| 生产 `build.ninja`，仅记录 hash | `31b8d8652e5b4f9152d4644677f8d3a210b9cd5d36a17f628952519fadc6f744` |
| Eigen 3.4.0 原始库归档 | `8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72` |

object 原位为 `/private/tmp/calmetrics-covchol-next/wheel-build/CMakeFiles/calmetrics_engine_operators.dir/cpp/affine_projection.cpp.o`。实际编译命令包含 `CALMETRICS_HAVE_EIGEN_ROTATION=1`、C++17、`-O3 -DNDEBUG -fno-fast-math -ffp-contract=off`；使用当前 AppleClang 17、arm64、macOS deployment target 15.0。Eigen 不另开线程。

指令报告记录 40 条 double-vector `fmul.2d/fadd.2d`，FMA 指令列表为空。在反汇编第 4,083–4,091 行、地址 `0x3f5c…0x3f7c`，四条独立乘法后四条加法，循环回边至 `0x3f54`，对应八个独立资产 lanes 的 factor fold。没有因看到一个 SIMD 指令就宣称所有算子或 expm1 使用 SIMD。

镜像 [反汇编](screen-affine-projection-disassembly.txt)为另一个原路径的 object 输出，object SHA 同上，但文本 SHA 为 `1b8bfbf8b4c4f046c68b17c4af1811face90d45f5244d083e83370c92a36a508`。不同文本路径前缀没有改变 object 字节。whole native/worker/wheel SHA 仍不同，按[两份 manifest](README.md#1-身份与历史范围)分开，不能从 object 相同推出整包相同。

## 3. Runtime 分支命中与完整请求分别验证

`vector_elements`计实际进入 Eigen Array 数学体的独立资产 lanes，不是硬件指令数。生产六组 JSON 的 N2M/F4 cases 为 64M，N250k/F32 case 为 8M，N252 case 为 8,064；不乘组合数。packing 对应 1,024 B（F4/A32）或 8,192 B（F32/A32），arena 0，正常根完整返回。[生产计时表](README.md#4-采用后的生产完整请求复测)给出相同 CPU、完整独立输出、强共享 NJIT 的测量结果。

采用后的 273 联合专项与新增 16 传输/预算专项检查 A8/9/17 packet/tail、带符号 stride、四舍入、cap/空轴、prepared 内容变化、canonical IEEE/status、实际 thread/process/shared 命中和总预算边界。ON/ASan/OFF 原生均覆盖 A17。[验证日志与源码文本](README.md#5-测试与审核终态)是运行证据，本页没有重新执行它们。

三层证据分别支持源码数学顺序、当前 arm64 object 的独立 SIMD 指令，以及实际分支/完整请求收益；这不是 CPU 采样、跨平台 directed-rounds/worker round 证明、全引擎性能承诺或完整 MC 验收。
