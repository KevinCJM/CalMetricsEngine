# ABI screening, completed 2026-10-02

Homebrew Highway1.3.0 dynamic library; this is a component comparison, not the installed engine. Prepared metadata files retain their original status; executed-identity.json records the actual checks.

168 direct cases, 20 failure cases, 192 concurrent outputs, 480 adversarial cases and 48 four-rounding/flag comparisons passed. Nine balanced rounds, N=16M: q95 std153.126→VQ75.228ms (NJIT207.967); median163.255→83.749ms (NJIT143.881). Seed20261002 differs from the subsequent complete-engine seed20261001; do not mix timings.

12 fresh serial memory children passed. Std/VQ use 128MiB scratch; VQ first-use retained allocator overhead +600B. RSS +48/80KiB versus std. NJIT q95 heap272MiB and median128MiB. Few-byte/page differences are diagnostic, not vetoes. malloc excludes Python arena suballocation, mmap and full stack; RSS is sampled, not an allocation count. Production static-link acceptance is in the parent README.
