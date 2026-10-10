// Screening-only ABI. No production operator, graph or error contract changes.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_MPL2_ONLY
#define EIGEN_FAST_MATH 0
#include <Eigen/Core>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {
struct ThreadLimit {
    BLAS_THREADING previous = BLASGetThreading();
    bool active = BLASSetThreading(BLAS_THREADING_SINGLE_THREADED) == 0;
    ~ThreadLimit() { if (active) BLASSetThreading(previous); }
};

// Same public Eigen expressions as covariance_c_means and gather_means<8>.
// Do not add these as a second production mean implementation.
EIGEN_DONT_INLINE void means_c(const double* x, Eigen::Index n, Eigen::Index d,
                              std::ptrdiff_t stride, double* target) {
    using Row = Eigen::Array<double, 1, Eigen::Dynamic>;
    Eigen::Map<Row> means(target, d);
    means.setZero();
    for (Eigen::Index i = 0; i < n; ++i) {
        Eigen::Map<const Row> row(x + i * stride, d);
        means += (row - means) / static_cast<double>(i + 1);
    }
}

EIGEN_DONT_INLINE void means_f(const double* x, Eigen::Index n, Eigen::Index d,
                              std::ptrdiff_t stride, double* means) {
    using Tile = Eigen::Array<double, 8, 1>;
    const auto full = d - d % 8;
    for (Eigen::Index j = 0; j < full; j += 8) {
        Tile mean = Tile::Zero();
        for (Eigen::Index i = 0; i < n; ++i) {
            Tile row;
            for (int lane = 0; lane < 8; ++lane)
                row[lane] = x[(j + lane) * stride + i];
            mean += (row - mean) / static_cast<double>(i + 1);
        }
        Eigen::Map<Tile>(means + j) = mean;
    }
    for (Eigen::Index j = full; j < d; ++j) {
        double mean = 0.0;
        for (Eigen::Index i = 0; i < n; ++i)
            mean += (x[j * stride + i] - mean) / static_cast<double>(i + 1);
        means[j] = mean;
    }
}
}

extern "C" int covariance_syrk_screen(const double* x, int n, int d,
    int layout, std::ptrdiff_t leading, int block, double* result) {
    if (!x || !result || n < 2 || d < 8 || block < 1 || block > 512 ||
        (layout != 0 && layout != 1) || leading < (layout ? n : d) ||
        leading > PTRDIFF_MAX / (layout ? d : n)) return -1;
    try {
        // Nonfinite/extreme cases are explicit prototype rejection, not a
        // substitute for the canonical operator's partially healthy outputs.
        const int outer = layout ? d : n, inner = layout ? n : d;
        for (int i = 0; i < outer; ++i)
            for (int j = 0; j < inner; ++j) {
                const auto value = x[std::ptrdiff_t(i) * leading + j];
                if (!std::isfinite(value)) return -2;
                if (std::abs(value) > 1e4) return -4;
            }
        std::vector<double> means(d);
        if (layout) means_f(x, n, d, leading, means.data());
        else means_c(x, n, d, leading, means.data());
        std::vector<double> panel(std::size_t(block) * d);
        ThreadLimit threads;
        if (!threads.active) return -3;
        for (int first = 0; first < n; first += block) {
            const int count = std::min(block, n - first);
            if (!layout) {
                using Row = Eigen::Array<double, 1, Eigen::Dynamic>;
                const Eigen::Map<const Row> mean(means.data(), d);
                for (int i = 0; i < count; ++i)
                    Eigen::Map<Row>(panel.data() + std::size_t(i) * d, d) =
                        Eigen::Map<const Row>(x + std::ptrdiff_t(first + i) * leading, d) - mean;
                cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, d, count, 1.0,
                    panel.data(), d, first == 0 ? 0.0 : 1.0, result, d);
            } else {
                using Column = Eigen::Array<double, Eigen::Dynamic, 1>;
                for (int j = 0; j < d; ++j)
                    Eigen::Map<Column>(panel.data() + std::size_t(j) * count, count) =
                        Eigen::Map<const Column>(x + std::ptrdiff_t(j) * leading + first, count) - means[j];
                // Col-major upper occupies row-major lower.
                cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, d, count, 1.0,
                    panel.data(), count, first == 0 ? 0.0 : 1.0, result, d);
            }
        }
        const double denominator = static_cast<double>(n - 1);
        for (int j = 0; j < d; ++j)
            for (int k = j; k < d; ++k) {
                const double value = result[std::size_t(layout ? k : j) * d + (layout ? j : k)] /
                                     denominator;
                result[std::size_t(j) * d + k] = value;
                result[std::size_t(k) * d + j] = value;
            }
        return 0;
    } catch (...) { return -3; }
}

extern "C" int covariance_syrk_threading() { return static_cast<int>(BLASGetThreading()); }
extern "C" int covariance_syrk_set_threading(int value) {
    return BLASSetThreading(static_cast<BLAS_THREADING>(value));
}
