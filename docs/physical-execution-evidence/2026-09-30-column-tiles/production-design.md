# Eigen storage-order arithmetic tiles — design 2026-09-30

Baseline0ddf9e verified against all current C++/build sources. Previous turn is progress, not blocked.

Scope: improve existing F-column two-operation mixed-graph fusion by using fixed Eigen matrix tiles for contiguous row output. Preserve input/operand/operation order, IEEE behavior, admission, zero-denominator failure prefix, original fallback and independent C-order output. No new public operator or business semantics, no changes to AST/Planner/transport/Excel rules.

Evaluate fixed2x4,4x4,8x4,4x8,8x8 tiles before adoption. Library Array/Map expressions own arithmetic and storage-order assignment; project controls bounds. No input packing, heap workspace, private threading or cross-request caching. Inspect actual compiler frames/spills; reject unbounded or unnecessary memory regressions. Keep all failed candidates and timing controls.

Native ABI screening covers old/new exact values/status/output prefix across operands/layouts/nonfinite/tails. Only supported candidates proceed to public old/new/frozen NJIT with equal CPU, all materialized independent outputs and retained prior result; memory separately measures allocation/stack/RSS. Full meaningful regression and original gates follow production changes. Ordinary numerical calls do not execute Excel.

Screening evidence: all5 initial matrix tiles matched7200 old/new ABI cases. 8x8 was slower and used1696B own stack; 4x4 inline edges used384B versus prior320B, so not adopted. Split fixed4x4 bulk/edges version matched1440cases and improved representative big arithmetic by24–41%. Bulk208B+driver48B gives256B normal own stack; edges176B plus scalar_math48B and driver tail dispatch also below prior368B. Final production object must verify these figures.

New192 public parameter/shape cases and192 native canary/padded-input cases cover exact blocks, row-only, column-only and both tails across operation orientations. The fixed72 public requests from previous turn remain unchanged, including direct/prepared and C/reverse controls; all8 memory cases retained.
