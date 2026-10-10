#include "calmetrics_engine/operators.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

namespace calmetrics_engine::ops {

std::size_t covariance_cholesky_accelerate_scratch_doubles(std::size_t dimension) {
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL)
    if (dimension < 2048 || dimension > static_cast<std::size_t>(INT_MAX) ||
        dimension > static_cast<std::size_t>(PTRDIFF_MAX) / 513)
        return 0;
    return 513 * dimension;
#else
    (void)dimension;
    return 0;
#endif
}

bool try_accelerate_covariance_cholesky(const Prepared &first, Output &output,
                                        Workspace &work, Audit &audit) {
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL)
    if (__builtin_available(macOS 15.0, *)) {
    if (first.spec->op != Op::covariance || first.count != 1 ||
        output.order != MemoryOrder::c || output.kind != Kind::number)
        return false;
    const auto &input = first.args[0];
    const auto n = input.shape.dim[0], d = input.shape.dim[1];
    const auto needed = covariance_cholesky_accelerate_scratch_doubles(d);
    if (!needed || n < d || n > static_cast<std::size_t>(INT_MAX) ||
        n > std::numeric_limits<std::size_t>::max() / d / sizeof(double))
        return false;
    const bool row_major = input.stride[1] == 1 &&
        input.stride[0] == static_cast<std::ptrdiff_t>(d);
    const bool column_major = input.stride[0] == 1 &&
        input.stride[1] == static_cast<std::ptrdiff_t>(n);
    if (!row_major && !column_major) return false;
    const auto *source = static_cast<const double *>(input.data);
    if (!source || !output.data) return false;

    // Keep special values and extreme scaling on the existing ordered path.
    // The Welford mean follows the current per-column observation order.
    std::vector<double> means(d);
    double maximum = 0.0;
    for (std::size_t j = 0; j < d; ++j) {
        double mean = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const auto value = row_major ? source[i * d + j] : source[j * n + i];
            if (!std::isfinite(value)) return false;
            maximum = std::max(maximum, std::abs(value));
            mean += (value - mean) / static_cast<double>(i + 1);
        }
        means[j] = mean;
    }
    if (maximum > 1e4 || maximum < 1e-8) return false;

    // The temporary means vector and the retained panel together fit the
    // Planner's 513*d bound, including a fresh allocation after prior runs.
    work.ensure(needed - d, 0);
    auto *panel = work.doubles.data();

    struct ThreadLimit {
        BLAS_THREADING previous;
        bool active;
        ThreadLimit() : previous(BLASGetThreading()),
            active(BLASSetThreading(BLAS_THREADING_SINGLE_THREADED) == 0) {}
        ~ThreadLimit() { if (active) BLASSetThreading(previous); }
    } threads;
    if (!threads.active) return false;

    auto *result = static_cast<double *>(output.data);
    constexpr std::size_t block = 512;
    for (std::size_t first_row = 0; first_row < n; first_row += block) {
        const auto count = std::min(block, n - first_row);
        if (row_major) {
            for (std::size_t i = 0; i < count; ++i)
                for (std::size_t j = 0; j < d; ++j)
                    panel[i * d + j] = source[(first_row + i) * d + j] - means[j];
            cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        static_cast<int>(d), static_cast<int>(count), 1.0,
                        panel, static_cast<int>(d), first_row == 0 ? 0.0 : 1.0,
                        result, static_cast<int>(d));
        } else {
            for (std::size_t j = 0; j < d; ++j)
                for (std::size_t i = 0; i < count; ++i)
                    panel[j * count + i] = source[j * n + first_row + i] - means[j];
            // Column-major upper occupies the lower half of row-major result.
            cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                        static_cast<int>(d), static_cast<int>(count), 1.0,
                        panel, static_cast<int>(count), first_row == 0 ? 0.0 : 1.0,
                        result, static_cast<int>(d));
        }
    }
    const auto denominator = static_cast<double>(n - 1);
    for (std::size_t j = 0; j < d; ++j) {
        result[j * d + j] /= denominator;
        if (!std::isfinite(result[j * d + j]) || result[j * d + j] <= 0.0)
            return false;
    }
    const double guard = 16.0 * static_cast<double>(n) *
                         std::numeric_limits<double>::epsilon();
    for (std::size_t j = 0; j < d; ++j)
        for (std::size_t k = j + 1; k < d; ++k) {
            const auto value = (row_major ? result[j * d + k] : result[k * d + j]) /
                               denominator;
            if (!std::isfinite(value) ||
                std::abs(value) <= guard *
                    std::sqrt(result[j * d + j] * result[k * d + k]))
                return false;
            result[j * d + k] = value;
            result[k * d + j] = value;
        }

    char triangle = 'U';
    const auto size = static_cast<int>(d);
    int info = 0;
    // Col-major upper is row-major lower, so the factor is already C-order L.
    dpotrf_(&triangle, &size, result, &size, &info);
    if (info != 0) return false;
    double minimum = std::numeric_limits<double>::max(), largest = 0.0;
    for (std::size_t i = 0; i < d; ++i) {
        const auto diagonal = result[i * d + i];
        minimum = std::min(minimum, diagonal);
        largest = std::max(largest, diagonal);
    }
    if (!(minimum / largest >= 1e-3)) return false;
    for (std::size_t i = 0; i < d; ++i)
        std::fill(result + i * d + i + 1, result + (i + 1) * d, 0.0);
    audit = {};
    audit.isa = "accelerate-dsyrk-dpotrf";
    audit.accelerate_covchol_calls = 1;
    audit.scratch_bytes = needed * sizeof(double);
    audit.algorithm_copy_bytes = n * d * sizeof(double);
    return true;
    }
    return false;
#else
    (void)first; (void)output; (void)work; (void)audit;
    return false;
#endif
}

} // namespace calmetrics_engine::ops
