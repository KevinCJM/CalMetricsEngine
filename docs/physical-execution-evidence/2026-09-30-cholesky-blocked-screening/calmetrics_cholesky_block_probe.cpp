#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

extern "C" int cholesky_block_probe(const double *a, int n, int block,
                                     double *out) {
    if (!a || !out || n < 1 || block < 0) return 1;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (!std::isfinite(a[std::size_t(i) * n + j]) ||
                a[std::size_t(i) * n + j] != a[std::size_t(j) * n + i])
                return 2;
    std::copy_n(a, std::size_t(n) * n, out);
    const auto previous = BLASGetThreading();
    if (BLASSetThreading(BLAS_THREADING_SINGLE_THREADED) != 0) return 3;
    int info = 0;
    char triangle = 'U';
    if (block == 0) {
        dpotrf_(&triangle, &n, out, &n, &info);
    } else {
        for (int offset = 0; offset < n; offset += block) {
            const int count = std::min(block, n - offset);
            const int remaining = n - offset - count;
            auto *diagonal = out + std::size_t(offset) * n + offset;
            dpotrf_(&triangle, &count, diagonal, &n, &info);
            if (info) break;
            if (remaining) {
                auto *panel = out + std::size_t(offset + count) * n + offset;
                cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans,
                            CblasNonUnit, count, remaining, 1.0, diagonal, n,
                            panel, n);
                cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, remaining,
                            count, -1.0, panel, n, 1.0,
                            out + std::size_t(offset + count) * n + offset + count,
                            n);
            }
        }
    }
    BLASSetThreading(previous);
    if (info) return 4;
    for (int i = 0; i < n; ++i)
        std::fill(out + std::size_t(i) * n + i + 1,
                  out + std::size_t(i + 1) * n, 0.0);
    return 0;
}
