# Eigen 3.4.0

Unmodified upstream release archive, used at build time for the header-only
`Eigen::JacobiRotation` / `Map` / `Array` implementations. There is no Eigen runtime download,
dynamic library, or dependency on TensorFlow/NumPy/SciPy headers.

- Source: https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
- SHA-256: `8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72`
- License: primarily MPL-2.0; upstream notices and per-file notices are preserved.
  The engine defines `EIGEN_MPL2_ONLY`, `EIGEN_DONT_PARALLELIZE`, and
  `EIGEN_FAST_MATH=0`. It does not enable Eigen BLAS/OpenMP dispatch.

CMake verifies the archive, extracts it inside the build directory, and includes
only the headers needed by the native rotation, covariance, ordered matrix-vector, and EM adapters. The full unmodified archive
keeps source releases reproducible offline and makes upstream provenance checkable.
The wheel includes license notices; the source distribution also includes the archive.
`CALMETRICS_ENGINE_USE_EIGEN=OFF` exercises the supported native-only path.

To update the dependency, verify the new release and license, change the pinned
archive/checksum together, then rerun correctness, allocation, performance and
distribution checks. Do not edit extracted headers or silently use host-installed Eigen.
