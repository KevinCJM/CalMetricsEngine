# Highway order selection integration, 2026-10-02

Status: implementation/acceptance in progress. Baseline build 5b31418d52e9804122fe901bb645f5291bf181a1a2fba17e77cb0fee0489a1c9. ABI screening is not complete-engine acceptance.

Use unchanged upstream Highway 1.3.0 VQSelect only for the existing single-rank selection: consumed count >=1024, no NaN or mixed signed zeros, automatic ISA, runtime HaveFloat64. Explicit ISA/build-OFF/unavailable platforms keep std::nth_element, exceptional ties keep std::sort. Packing, exact rank, short-side neighbor scan and median/quantile arithmetic remain unchanged. Shared multi-root full-sort cache remains independent. This is one physical implementation of the same four operators, not a new logical operator.

Pass Isa through execute -> reduction -> ordered_stat; reduce_value defaults to scalar. Direct audit isa=highway_vqselect identifies use; do not fabricate vector_elements or change IPC. Library uses runtime CPU detection, no private threadpool/JIT, and per-thread pivot entropy. Pivot randomness cannot change exact selected values on the admitted domain. Reuse exclusive count-sized workspace and preserve input/independent result ownership. TLS/stack is measured separately, not claimed zero.

Bundle unchanged upstream archive with SHA256/licenses, statically link hwy core and vqsort.cc/vqsort_f64a.cc. No Homebrew runtime dependency. Flag defaults ON but initially enabled only on measured Apple arm64; all other platforms keep the existing native implementation. Hash flag into build identity, include archive in sdist, no runtime download/build. AST/DAG/Planner/worker/result/Excel mathematical contracts/formulas unchanged.

Verify installed ON/OFF wheels, explicit scalar, boundaries/NaN/zero/Inf/exponents, threads/process/shared-memory ownership, prepared changes, original fixture/Excel snapshot parity, native ASan/UBSan, source/wheel identity. Benchmark actual old/new Scheduler requests vs warmed NJIT with same inputs/core budget/owned outputs, plus cold/warm allocations and incremental RSS. Few KB/ms differences are diagnostic, not vetoes; correctness, ownership and budgets stay hard. All unresolved NJIT gaps remain open.
