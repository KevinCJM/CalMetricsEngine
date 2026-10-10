# Column variance library design — 2026-09-30

Scope: replace only the existing contiguous rank-2 axis-0 variance inner loops with Eigen 3.4 fixed-width unaligned maps. No public signatures, financial semantics, formula expansion, graph roots, scheduling or output ownership changes.

Dependency: each output column depends on every row. Keep row order, eight-column validation groups, two-pass population/sample formula, and scalar division `(n-ddof)`. No horizontal Eigen reduction, FMA/reassociation, alternate variance algorithm, private threads or input packing. Admit only existing measured macOS 15+ arm64 library builds, automatic ISA, n >= 32, width >= 8. Width tails, forced scalar, Eigen OFF and other platforms retain the existing scalar algorithm.

Implementation: extract the contiguous variance block from the large latent dispatcher into a private non-inlined function shared by scalar and Eigen paths. Use the original three 8-double arrays as Eigen Map backing, avoiding additional numerical allocations. The output is written only after the same finite checks; prior written output prefixes on failure remain identical. Model operators/Excel expansion are unchanged.

Evidence before adoption: compare the immediately preceding production archive/wheel 146cd294a29a431e6ef82be35f549ba89e3573a68753a686c25407bdac288e84. Native prototype: 408 boundary/layout/invalid cases bitwise identical including error output prefixes; 9 paired screening workloads. Follow with final ordinary/direct/prepared public variance and complete standardization graph (standardized matrix, mean, scale), CPU=1, six balanced backend orders, independent retained outputs, original NJIT tolerances, cold/warm tracked allocations and RSS. Inspect final object stack frames and no FMA in the changed routine. Run full regression, native/sanitizers/Eigen OFF and unchanged performance gates.

The index bitmap candidate is deferred: the historical partition/scatter fixtures are monotonic and bypass sorting already. No claim that replacing sorting fixes those old regressions.
