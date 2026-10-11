#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {
struct SingleThread {
    BLAS_THREADING previous;
    SingleThread() : previous(BLASGetThreading()) {
        if (BLASSetThreading(BLAS_THREADING_SINGLE_THREADED) != 0) throw 1;
    }
    ~SingleThread() { BLASSetThreading(previous); }
};
}

extern "C" int covariance_blas_probe(const double *x, int n, int d, int layout,
                                     int block, double *out) {
    if (!x || !out || n < 2 || d < 1 || block < 1 || (layout != 0 && layout != 1)) return -1;
    try {
        std::vector<double> means(d, 0.0);
        for (int j = 0; j < d; ++j) {
            double mean = 0.0;
            for (int i = 0; i < n; ++i) {
                const double value = layout == 0 ? x[std::size_t(i) * d + j]
                                                 : x[std::size_t(j) * n + i];
                if (!std::isfinite(value)) return -2;
                mean += (value - mean) / static_cast<double>(i + 1);
            }
            means[j] = mean;
        }
        std::vector<double> panel(std::size_t(block) * d);
        SingleThread threads;
        for (int first = 0; first < n; first += block) {
            const int count = std::min(block, n - first);
            if (layout == 0) {
                for (int i = 0; i < count; ++i)
                    for (int j = 0; j < d; ++j)
                        panel[std::size_t(i) * d + j] = x[std::size_t(first + i) * d + j] - means[j];
                cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                            d, count, 1.0, panel.data(), d, first == 0 ? 0.0 : 1.0, out, d);
            } else {
                for (int j = 0; j < d; ++j)
                    for (int i = 0; i < count; ++i)
                        panel[std::size_t(j) * count + i] = x[std::size_t(j) * n + first + i] - means[j];
                // Column-major upper occupies the lower half of row-major out.
                cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                            d, count, 1.0, panel.data(), count, first == 0 ? 0.0 : 1.0, out, d);
            }
        }
        const double denominator = static_cast<double>(n - 1);
        if (layout == 0) {
            for (int j = 0; j < d; ++j)
                for (int k = j; k < d; ++k) {
                    const double value = out[std::size_t(j) * d + k] / denominator;
                    out[std::size_t(j) * d + k] = value;
                    out[std::size_t(k) * d + j] = value;
                }
        } else {
            for (int j = 0; j < d; ++j)
                for (int k = j; k < d; ++k) {
                    const double value = out[std::size_t(k) * d + j] / denominator;
                    out[std::size_t(j) * d + k] = value;
                    out[std::size_t(k) * d + j] = value;
                }
        }
        return 0;
    } catch (...) { return -3; }
}

extern "C" int cholesky_blas_probe(double *matrix, int d) {
    if (!matrix || d < 1) return -1;
    try {
        SingleThread threads;
        char triangle = 'U';
        int info = 0;
        // Col-major upper is row-major lower, hence returns row-major L.
        dpotrf_(&triangle, &d, matrix, &d, &info);
        if (info != 0) return info;
        for (int row = 0; row < d; ++row)
            std::fill(matrix + std::size_t(row) * d + row + 1,
                      matrix + std::size_t(row + 1) * d, 0.0);
        return 0;
    } catch (...) { return -3; }
}
