# V37: primary-source research and evidence boundary

Reviewed 2026-10-06. This is a source review, not a performance certificate.

- [Eigen external BLAS/LAPACK](https://eigen.tuxfamily.org/dox/TopicUsingBlasLapack.html)
  documents external BLAS level 2/3 and LAPACK substitution, including Accelerate,
  OpenBLAS and MKL. Enabling substitutions does not establish identical IEEE,
  errors or ownership contracts. The current candidate continues the already
  qualified explicit Accelerate DGEMM and Eigen maps/reductions.
- [OpenBLAS official repository](https://github.com/OpenMathLib/OpenBLAS)
  supports ARM64/macOS and architecture-specific configurations. Apple VortexM4
  support is described as preliminary, with SME SGEMM-related kernels listed.
  This is not evidence for a faster FP64 replacement on the current host.
- [BLIS kernel guide](https://github.com/flame/blis/blob/master/docs/KernelsHowTo.md)
  documents architecture-specific microtiles, arbitrary C row/column strides,
  packed A/B micropanels, and fused kernels to avoid redundant memory traffic.
  It also distinguishes memory-bound vector operations from compute-intensive
  level-3 operations. This motivates inspecting materialization/merge traffic,
  but does not certify our exact cache geometry. The previously pinned BLIS
  experiment remains rejected; changing a package name is not a new hypothesis.

The measured private phase profile points to explicit partial-result merging,
not primarily the full input guard, in the 4096 square case. The full source
shows a separate SIMD guard limit of 4096 despite a root admission limit 8192.
V37 therefore tests a different tile aspect ratio and safe guard alignment while
preserving existing numerical and ownership rules. The cache benefit remains a
measurement-based hypothesis: no hardware cache-counter claim is made.

Do not infer that loaded NumPy OpenBLAS is the backend of warmed NJIT matmul.
The installed Numba-to-SciPy binding was separately inspected and resolves to
Accelerate. New paired tests explicitly control the actual calling-thread BLAS
mode and restore it, with equal inputs, layouts, retained outputs and CPU1.

If this candidate does not materially close the complete-request time/memory
gap, the next investigation is to reduce repeated library-result merges using
documented GEMM accumulation, with a deterministic error analysis and original
canonical replay before publication. That proposal is not implemented or
qualified here. Do not substitute probability, vendor advertising or another
hardware's speedups for our actual complete-engine acceptance.
