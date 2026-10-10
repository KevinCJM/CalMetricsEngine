# Matmul final-object static review

Read-only inspection of the actual old solve-final-build and new matmul-build arm64 objects. No rebuild or numerical execution. New installed build ID string was verified in the binary: `1aec1a6b0d4a07f0caee43b58cd4b339bdcc165f5e615f2190b15da0ba66e86b`.

## Result

No source adjustment is required by this static review. The admitted Eigen F-right kernel is a leaf with a 416-byte frame. `matrix` keeps its original 320-byte frame and fully releases it before the actual tail branch. All nine `_bzero` call sites in the kernel are dominated by K==0; admitted K>=32 cannot reach them. No residual Eigen evaluator calls, FMA instructions, or heap allocation references occur in the successful arithmetic kernel.

| Local function | Old frame | New frame |
|---|---:|---:|
| matrix |320 B|320 B|
| simd_matmul |144 B|144 B (object identical)|
| selected_isa |32 B|32 B (object identical)|
| library_matmul_f |—|416 B|
| library_matvec |240 B|240 B|
| library_covariance_upper |368 B|368 B|
| MatrixRotation::apply |0 B|0 B|

The old F-right request entered `matrix(320) -> simd_matmul(144) -> selected_isa(32)` before rejecting the right-hand layout: a 496-byte warm local peak. Its later generic arithmetic retained only matrix's320-byte frame. The new arithmetic416-byte frame is not summed with matrix320: the object contains `add sp,sp,#0x140` at0x10fc and `b library_matmul_f` at0x1100. It therefore fits below the previous496-byte local request peak, while the numerical phase alone is96bytes larger than the old generic phase.

The new platform admission runs before that tail branch. Compiler-rt's `__isPlatformVersionAtLeast` own frame is64bytes, but its `__availability_version_check` and cold initialization callees have not been completely unfolded. Consequently this report does **not** claim a416-byte full process/OS/cold-call peak, nor replace measured heap/RSS acceptance. Error paths keep existing C++ exception allocations. Outer Python/graph/execute frames are outside this narrowly changed-object audit.

## Exact identity

| Artifact | SHA-256 |
|---|---|
| old matrix.cpp.o | `43b8c9506f03ac9bdc1fb2f81037c58a15dac5f8170ce5eff078d758195403e9` |
| old matrix_backend.cpp.o | `e30026bb6712fffc4a32158bf0907bd836829a336ea722d912abf4b8520f4217` |
| new matrix.cpp.o | `b0ead9b7e8bc5b128378fca0f0905e256c8cf43a99bec5dc7f87c86e2f34b746` |
| new matrix_backend.cpp.o | `ce177a68dcee412b37ac53aa4d7e9e9cca72bf7d47019a7314ecfaf0a25e01e4` |
| old:CMakeFiles/calmetrics_engine_operators.dir/cpp/operators/simd.cpp.o | `aa88df3d0aedbe59917c3d0f53b67389f80b30ebb4feb062e54bcd2e3fe48ddb` |
| old:libcalmetrics_engine_operators.a | `d0794384450c63ad9326ed8915d37b8b95acd8c41487c4da106b580458a3e218` |
| new:CMakeFiles/calmetrics_engine_operators.dir/cpp/operators/simd.cpp.o | `aa88df3d0aedbe59917c3d0f53b67389f80b30ebb4feb062e54bcd2e3fe48ddb` |
| new:libcalmetrics_engine_operators.a | `763b0479884b5377eb8778aedddc1e2faed8cda55f0e2c79bcb8c33bdec56984` |
| cpp/operators/matrix_backend.cpp | `334a00aee151b3875409ab5cfb047ba8e2e9c3d68d572a5dc5c5ce768dd44edc` |
| cpp/operators/matrix.cpp | `5d938d6c04eda0e747d8cfcc9dd916ab7bc127a3669ffd11f53011958e329c44` |
| cpp/include/calmetrics_engine/matrix_backend.hpp | `ed7483cffd42fddf36865ba530ced886249aeafde2352db552c9a1380350a049` |
| _native.cpython-312-darwin.so | `024f8cfe6858971d6dfd86e48fbda6eb8f71912a278d96bd7174188fde40c983` |
| calmetrics_engine-0.3.0-cp312-cp312-macosx_15_0_arm64.whl | `d18621c8cfc65a8be41d060c3762356374abdad1e365510e182c6da92709fc36` |

The three current source files still exactly match the frozen reviewed draft. `matmul-formal-chain.json` records instruction addresses and boundaries; `matmul-formal-static.json` retains per-function frames and references; `matmul-formal-identity.json` retains complete paths, sizes, hashes and draft equality. The disassembly, symbol and undefined-reference text files are saved alongside them.

No claim is made about numerical parity, full correctness, elapsed time, runtime heap/RSS, or Microsoft Excel recalculation by this read-only review. Those remain the parent's serialized acceptance runs.
