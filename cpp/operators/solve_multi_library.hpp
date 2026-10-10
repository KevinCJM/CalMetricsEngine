#pragma once
#include "calmetrics_engine/operators.hpp"
#include <algorithm>
#include <cmath>

#ifdef CALMETRICS_HAVE_EIGEN_ROTATION
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_MPL2_ONLY
#define EIGEN_FAST_MATH 0
#include <Eigen/Core>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define CME_SOLVE_MULTI_INLINE inline __attribute__((always_inline))
#else
#define CME_SOLVE_MULTI_INLINE inline
#endif

namespace calmetrics_engine::ops {
struct SolveRhsSpan { double *data; std::size_t columns; };

CME_SOLVE_MULTI_INLINE bool library_eliminate_available(std::size_t n) {
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    if (__builtin_available(macOS 15.0, *)) return n >= 128;
#else
    (void)n;
#endif
    return false;
}
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
namespace solve_multi_detail {
// All maps borrow the caller's LU workspace. Two lazy subtractions retain each
// coefficient's pivot order without an owning array or an intermediate store.
CME_SOLVE_MULTI_INLINE void update_pending(double *matrix, std::size_t n,
    std::size_t row, std::size_t first, std::size_t last, std::size_t column) {
    constexpr std::size_t width = 16;
    const auto count = n - column;
    if (!count || first == last) return;
    using Tile = Eigen::Array<double, width, 1>;
    using Read = Eigen::Map<const Tile, Eigen::Unaligned>;
    std::size_t j = 0;
    for (; count - j >= width; j += width) {
        Eigen::Map<Tile, Eigen::Unaligned> target(matrix + row * n + column + j);
        auto p = first;
        for (; last - p >= 2; p += 2) {
            const Read row0(matrix + p * n + column + j);
            const Read row1(matrix + (p + 1) * n + column + j);
            const double factor0 = matrix[row * n + p];
            const double factor1 = matrix[row * n + p + 1];
            target = (target - factor0 * row0) - factor1 * row1;
        }
        for (; p < last; ++p) {
            const Read source(matrix + p * n + column + j);
            const double factor = matrix[row * n + p];
            target -= factor * source;
        }
    }
    // Keep every subtraction, including zero and nonfinite factors, in tails.
    for (; j < count; ++j) {
        double value = matrix[row * n + column + j];
        for (auto p = first; p < last; ++p)
            value -= matrix[row * n + p] * matrix[p * n + column + j];
        matrix[row * n + column + j] = value;
    }
}

CME_SOLVE_MULTI_INLINE void subtract_row(double *__restrict target,
    const double *__restrict source, std::size_t count, double factor) {
    for (std::size_t j = 0; j < count; ++j) target[j] -= factor * source[j];
}

template <bool RejectNaNPivot, std::size_t SpanCount>
CME_SOLVE_MULTI_INLINE void eliminate_panel(double *matrix,
    const std::array<SolveRhsSpan, SpanCount> &spans,
    std::size_t n, std::size_t first, std::size_t last) {
    for (auto k = first; k < last; ++k) {
        auto pivot = k;
        for (auto i = k + 1; i < n; ++i)
            if (std::abs(matrix[i * n + k]) > std::abs(matrix[pivot * n + k])) pivot = i;
        const auto magnitude = std::abs(matrix[pivot * n + k]);
        // The two public solvers retain their different historical NaN rules.
        require(RejectNaNPivot ? magnitude > 1e-14 : !(magnitude <= 1e-14), "SINGULAR_MATRIX");
        if (pivot != k) {
            // Saved multipliers from this panel must follow the row permutation.
            // Lower cells belonging to earlier panels are no longer consumed.
            for (auto j = first; j < n; ++j)
                std::swap(matrix[k * n + j], matrix[pivot * n + j]);
            for (const auto &rhs : spans)
                for (std::size_t j = 0; j < rhs.columns; ++j)
                    std::swap(rhs.data[k * rhs.columns + j], rhs.data[pivot * rhs.columns + j]);
        }
        update_pending(matrix, n, k, first, k, last);
        const auto diagonal = matrix[k * n + k];
        for (auto i = k + 1; i < n; ++i) {
            const auto factor = matrix[i * n + k] / diagonal;
            matrix[i * n + k] = factor;
            subtract_row(matrix + i * n + k + 1, matrix + k * n + k + 1,
                         last - k - 1, factor);
            // RHS updates remain immediate and keep their original order.
            for (const auto &rhs : spans)
                if (rhs.columns) subtract_row(rhs.data + i * rhs.columns,
                    rhs.data + k * rhs.columns, rhs.columns, factor);
        }
    }
}

// These rows are independent only after their pivot panel is complete. Keep
// each coefficient's ascending-pivot order in fixed Eigen arrays, sharing the
// source row reads and writing each destination tile once. No heap/packing.
CME_SOLVE_MULTI_INLINE void update_group(double *matrix, std::size_t n,
    std::size_t row, std::size_t first, std::size_t last, std::size_t column) {
    constexpr std::size_t width = 8, rows = 4;
    const auto count = n - column;
    using Tile = Eigen::Array<double, width, 1>;
    using Read = Eigen::Map<const Tile, Eigen::Unaligned>;
    std::size_t j = 0;
    for (; count - j >= width; j += width) {
        Tile values[rows];
        for (std::size_t r = 0; r < rows; ++r)
            values[r] = Read(matrix + (row + r) * n + column + j);
        for (auto p = first; p < last; ++p) {
            const Read source(matrix + p * n + column + j);
            for (std::size_t r = 0; r < rows; ++r)
                values[r] -= matrix[(row + r) * n + p] * source;
        }
        for (std::size_t r = 0; r < rows; ++r)
            Eigen::Map<Tile, Eigen::Unaligned>(matrix + (row + r) * n + column + j) = values[r];
    }
    for (; j < count; ++j) for (std::size_t r = 0; r < rows; ++r) {
        double value = matrix[(row + r) * n + column + j];
        for (auto p = first; p < last; ++p)
            value -= matrix[(row + r) * n + p] * matrix[p * n + column + j];
        matrix[(row + r) * n + column + j] = value;
    }
}

template <bool RejectNaNPivot, std::size_t SpanCount>
CME_SOLVE_MULTI_INLINE void eliminate(double *matrix,
    const std::array<SolveRhsSpan, SpanCount> &spans, std::size_t n) {
    static_assert(SpanCount == 1 || SpanCount == 2);
    constexpr std::size_t panel_width = 32;
    for (std::size_t first = 0; first < n;) {
        const auto last = first + std::min(panel_width, n - first);
        eliminate_panel<RejectNaNPivot>(matrix, spans, n, first, last);
        auto i = last;
        if (n >= 1024)
            for (; n - i >= 4; i += 4) update_group(matrix, n, i, first, last, last);
        for (; i < n; ++i)
            update_pending(matrix, n, i, first, last, last);
        first = last;
    }
}
} // namespace solve_multi_detail
#endif

// Private physical implementation. Input copies, back substitution and finite
// checks stay in the canonical caller; explicit scalar selection bypasses it.
template <bool RejectNaNPivot = true, std::size_t SpanCount>
CME_SOLVE_MULTI_INLINE bool library_eliminate_spans(double *matrix,
    const std::array<SolveRhsSpan, SpanCount> &spans, std::size_t n, Audit &audit) {
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    if (library_eliminate_available(n)) {
        solve_multi_detail::eliminate<RejectNaNPivot>(matrix, spans, n);
        audit.isa = "eigen-array-blocked-lu";
        return true;
    }
#else
    (void)matrix; (void)spans; (void)n; (void)audit;
#endif
    return false;
}

template <bool RejectNaNPivot = true>
CME_SOLVE_MULTI_INLINE bool library_eliminate_multi(double *matrix, double *rhs,
    std::size_t n, std::size_t columns, Audit &audit) {
    return library_eliminate_spans<RejectNaNPivot>(matrix,
        std::array<SolveRhsSpan, 1>{{{rhs, columns}}}, n, audit);
}
} // namespace calmetrics_engine::ops
#undef CME_SOLVE_MULTI_INLINE
