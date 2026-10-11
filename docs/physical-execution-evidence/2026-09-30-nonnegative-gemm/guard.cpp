// Research-only numerical admission wrapper around the existing Accelerate ABI.
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
extern "C" int cme_blas_gemm(int, const double *, std::ptrdiff_t, std::ptrdiff_t,
    const double *, std::ptrdiff_t, std::ptrdiff_t, std::size_t, std::size_t,
    std::size_t, double *) noexcept;
namespace {
bool admitted(const double *p, std::size_t rows, std::size_t cols,
              std::ptrdiff_t rs, std::ptrdiff_t cs) noexcept {
    if (!p || rows > INT_MAX || cols > INT_MAX) return false;
    if (!((cs == 1 && rs >= static_cast<std::ptrdiff_t>(cols) && rs <= INT_MAX) ||
          (rs == 1 && cs >= static_cast<std::ptrdiff_t>(rows) && cs <= INT_MAX)))
        return false;
    const bool columns = rs == 1;
    for (std::size_t outer = 0; outer < (columns ? cols : rows); ++outer)
        for (std::size_t inner = 0; inner < (columns ? rows : cols); ++inner) {
            const auto i = columns ? inner : outer;
            const auto j = columns ? outer : inner;
            const double value = p[static_cast<std::ptrdiff_t>(i) * rs +
                                   static_cast<std::ptrdiff_t>(j) * cs];
            if (!std::isfinite(value) || std::signbit(value) ||
                (value != 0.0 && (value < 0x1p-128 || value > 0x1p128)))
                return false;
        }
    return true;
}
}
extern "C" __attribute__((visibility("default"))) int cme_nonnegative_gemm(
    int variant, const double *a, std::ptrdiff_t ar, std::ptrdiff_t ac,
    const double *b, std::ptrdiff_t br, std::ptrdiff_t bc, std::size_t m,
    std::size_t k, std::size_t n, double *out) noexcept {
    if (!out || m < 32 || n < 32 || k < 32 || k > 4096 ||
        !admitted(a, m, k, ar, ac) || !admitted(b, k, n, br, bc)) return 1;
    return cme_blas_gemm(variant, a, ar, ac, b, br, bc, m, k, n, out);
}
