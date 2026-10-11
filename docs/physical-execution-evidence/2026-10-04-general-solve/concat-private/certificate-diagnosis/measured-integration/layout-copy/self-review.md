# Existing concatenation rectangle copy: native library candidate

终态：6707已接入实际源码、正式包与worker；普通/SAN/OFF相关各274及原生各9通过，实际重建包另验274/9，正式完整链与fresh峰值终态通过。见[实际结果](actual-result.md)及[验收](acceptance.json)。以下保留候选形成阶段的依赖/别名/数学审核与当时待办，不表示这些采用检查仍未执行。

2026-10-05. Scope is three existing C++ execution files and one engine regression
module. Root source is still 20cf until qualification and guarded adoption. No
new operator, mathematical rule, AST/Typed IR contract or upstream edit is made.

## Dependency, correctness and lifetime

The existing deferred concatenation proof remains authoritative: float64 leaves
are backed by this call's persistent input or stable borrowed views; non-public
intermediate concatenations have one consumer. Each logical node still validates
its type, axis and shape at its original position. Public/shared/arithmetic leaves,
observers and incompatible error modes keep their original materialization path.

The new helper is reached only after bounds and rectangle intersection validation.
Its source and destination are disjoint, addressable native spans. The source has
row stride one and positive, non-overlapping column stride. The destination is
the original C scratch/result rectangle with positive row stride. Both logical
dimensions are at least 32; 32-square tails are clipped before maps are built.
Eigen's signed indices inherit the caller's validated addressable array geometry.

Eigen 3.4.0 unaligned column-/row-major Maps perform a plain assignment to the
original target. There are no arithmetic reductions, temporary matrices, dynamic
allocations, cached values, private threads or result-owner changes. Thus payload
bits, coordinates, signed zeros, NaNs and infinities must be preserved, not merely
compared with a floating tolerance. No numeric result is reused across calls.
The helper borrows both spans synchronously and retains neither after returning.

macOS 15+ arm64 with Eigen enabled uses the library copy; other platforms, Eigen
OFF and unqualified rectangles retain the existing live portable copy. C-backed,
negative/zero-stride, small, rank-3 and incompatible cases are unaffected. Input F
layout is preserved. Only the already-needed internal C target is written; this
does not impose a new C input or output contract. Explicit F result ownership and
transport still use the existing implementation.

## Evidence and adoption condition

The diagnostic screen compared the existing loop, bounded vDSP pack/transpose,
Eigen Maps and register transpose. Across 224 small offset/tail/padded cases,
all checked IEEE payloads and destination sentinels matched exactly. On the
536-MB full matrix, seven rotating rounds measured Eigen at about 0.097 seconds
versus the old loop at 0.304 seconds. These are component diagnostics only; vDSP
and hand transpose are not adopted.

The installed candidate includes 32 new real-engine cases crossing the 31/32
boundary, padded/offset F views, C/F results, single/thread/inline process/shared
process lanes, two intervals and two retained generations. All uint64 payloads,
input immutability, readonly and independent ownership are checked. Together with
existing concat/private solve/error tests, the initial 274 cases passed.

Four complete CPU1 tasks, six rotating old/new/warmed-NJIT rounds each, include
owned readonly input preparation, ordinary Scheduler graph execution and
materialized independent result return. 8192 F old/new/NJIT medians are
5.411993/4.793669/5.400359 seconds; 8192 C are 4.901804/4.831929/5.342334.
The C path has no intentional change, so its small difference is not attributed
to this optimization. All old/new outputs are bitwise equal; largest NJIT output
tail difference is 3.23e-13 and independent KKT residual checks passed.
This is construction/solve of KKT, not complete active-set QP or platform frontier.

Eight fresh complete F memory jobs passed: two continuous malloc probes and three
RSS processes per old/new backend. Requested peaks are 1,639,639,239 versus
1,639,639,223 bytes, allocation counts equal. A 16-byte difference is diagnostic,
not a benefit claim. RSS increment medians are 1,679,917,056 versus 1,675,165,696
bytes; do not treat sampled page variation as a new large memory optimization.
Keep allocator/mmap/short-peak blind spots and do not add heap and RSS numbers.
Prior actual20cf/NJIT large memory savings remain historical measurements;
the new candidate has not yet had a fresh NJIT peak comparison.

Before adoption, require affected ordinary native, real ASan/UBSan and all-library
OFF binding/native regressions, unchanged baseline preimages, and source/package/
worker identity. After adoption, rebuild actual root and recheck real graph and
complete task performance. Do not reuse old full-suite counts for this candidate.

## Excel and public documentation

This is a byte-preserving physical copy, so logical formulas, axis mappings,
budget, errors and selected outputs do not change. No Excel dependency, retained
intermediate or export work is added to ordinary C++ requests. No new Microsoft
Excel recalculation is claimed. Current execution guidance should describe the
library eligibility; user types/operator signatures do not need revision.

The whole goal stays open: one 11-percent task improvement does not prove broad,
substantial benefits across platform frontier/candidate evaluation, repeated QP,
general dense solves and the other major model/simulation chains.
