// Isolated research ABI. No production dispatch or numerical guard is provided.
// The harness owns valid independent output and valid read-only input buffers.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

// Conservative byte-representability check only; it does not establish the
// allocation's actual extent or ownership. Those are harness responsibilities.
bool span_fits(std::size_t rows, std::size_t cols, std::ptrdiff_t rs,
               std::ptrdiff_t cs) noexcept {
  if (!rows || !cols)
    return true;
  if (rs < 0 || cs < 0)
    return false;
  const auto limit = static_cast<std::size_t>(
      std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
  const auto ur = static_cast<std::size_t>(rs);
  const auto uc = static_cast<std::size_t>(cs);
  if (ur && rows - 1 > (limit - 1) / ur)
    return false;
  const auto rmax = (rows - 1) * ur;
  if (uc && cols - 1 > (limit - 1 - rmax) / uc)
    return false;
  return true;
}

API_AVAILABLE(macos(15.0))
bool restore_threading(BLAS_THREADING previous) noexcept {
  // Read back even if the setter reports failure. Either failure is fatal to
  // this experiment; never hide a contaminated persistent worker as success.
  const int status = BLASSetThreading(previous);
  const auto restored = BLASGetThreading();
  return status == 0 && restored == previous;
}

} // namespace

// Returns 0/1 for the SDK's MULTI/SINGLE enum, -1 if unavailable.
// This function is diagnostic and is not a thread-count measurement.
extern "C" __attribute__((visibility("default"))) int
cme_blas_thread_state() noexcept {
  if (__builtin_available(macOS 15.0, *))
    return static_cast<int>(BLASGetThreading());
  return -1;
}

// variant: 0=whole matrix, 1=64x64, 2=128x128, 3=256x256 output tiles.
// Every tile always uses full K. There is no K partition or partial-sum merge.
// Return 0: completed, 1: unsupported/rejected before writing output,
//        2: single-thread admission failed but previous state was restored,
//        3: previous threading state could not be restored; output may be written.
// Code 3 is an experiment failure, not permission to silently run a fallback.
extern "C" __attribute__((visibility("default"))) int cme_blas_gemm(
    int variant, const double *a, std::ptrdiff_t ar, std::ptrdiff_t ac,
    const double *b, std::ptrdiff_t br, std::ptrdiff_t bc, std::size_t m,
    std::size_t k, std::size_t n, double *out) noexcept {
  if (variant < 0 || variant > 3)
    return 1;
  if (!m || !n)
    return 0;
  if (m > INT_MAX || k > INT_MAX || n > INT_MAX || !out)
    return 1;
  if (!span_fits(m, n, static_cast<std::ptrdiff_t>(n), 1))
    return 1;
  if (!k) {
    std::fill_n(out, m * n, 0.0);
    return 0;
  }
  if (!a || !b)
    return 1;

  CBLAS_TRANSPOSE ta, tb;
  std::ptrdiff_t lda, ldb;
  if (ac == 1 && ar >= static_cast<std::ptrdiff_t>(k)) {
    ta = CblasNoTrans;
    lda = ar;
  } else if (ar == 1 && ac >= static_cast<std::ptrdiff_t>(m)) {
    ta = CblasTrans;
    lda = ac;
  } else {
    return 1;
  }
  if (bc == 1 && br >= static_cast<std::ptrdiff_t>(n)) {
    tb = CblasNoTrans;
    ldb = br;
  } else if (br == 1 && bc >= static_cast<std::ptrdiff_t>(k)) {
    tb = CblasTrans;
    ldb = bc;
  } else {
    return 1;
  }
  if (lda > INT_MAX || ldb > INT_MAX ||
      !span_fits(m, k, ar, ac) || !span_fits(k, n, br, bc))
    return 1;

  if (__builtin_available(macOS 15.0, *)) {
    const auto previous = BLASGetThreading();
    if (previous != BLAS_THREADING_MULTI_THREADED &&
        previous != BLAS_THREADING_SINGLE_THREADED)
      return 1;
    const int set_status = BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
    const auto selected = BLASGetThreading();
    if (set_status != 0 || selected != BLAS_THREADING_SINGLE_THREADED)
      return restore_threading(previous) ? 2 : 3;

    const auto tile = variant == 0 ? std::max(m, n)
                                  : std::size_t{32} << variant;
    for (std::size_t i = 0; i < m;) {
      const auto mi = std::min(tile, m - i);
      for (std::size_t j = 0; j < n;) {
        const auto nj = std::min(tile, n - j);
        const double *const ai = a + i * static_cast<std::size_t>(ar);
        const double *const bj = b + j * static_cast<std::size_t>(bc);
        double *const cij = out + i * n + j;
        cblas_dgemm(CblasRowMajor, ta, tb, static_cast<int>(mi),
                    static_cast<int>(nj), static_cast<int>(k), 1.0, ai,
                    static_cast<int>(lda), bj, static_cast<int>(ldb), 0.0, cij,
                    static_cast<int>(n));
        j += nj;
      }
      i += mi;
    }
    return restore_threading(previous) ? 0 : 3;
  }
  return 1;
}
