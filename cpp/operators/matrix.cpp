#include "calmetrics_engine/operators.hpp"
#include "affine_contract.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include "solve_multi_library.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace calmetrics_engine::ops {
double dot_value(const Value &x, const Value &y) {
    // Vector geometry is fixed for this whole reduction. Keep the canonical
    // left-fold order without repeating generic rank/offset decoding per value.
    if (x.shape.rank == 1 && y.shape.rank == 1) {
        if (x.size() == 0) return 0.0;
        const auto *left = static_cast<const double *>(x.data);
        const auto *right = static_cast<const double *>(y.data);
        if (x.stride[0] == 1 && y.stride[0] == 1)
            return std::inner_product(left, left + x.size(), right, 0.0);
        double total = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i)
            total += left[static_cast<std::ptrdiff_t>(i) * x.stride[0]] *
                     right[static_cast<std::ptrdiff_t>(i) * y.stride[0]];
        return total;
    }
    double result = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        result += x.f(i) * y.f(i);
    return result;
}
void matvec_value(const Value &a, const Value &x, double *out) {
    for (std::size_t row = 0; row < a.shape.dim[0]; ++row)
        out[row] = dot_value(a.row(row), x);
}
void finish_covariance_matrix(Output &out, bool correlation) {
    const auto cols = out.shape.dim[0];
    if (cols == 0) return;
    auto *result = static_cast<double *>(out.data);
    auto *means = result + (cols - 1) * cols;
    if (correlation) {
        const double last = result[cols * cols - 1];
        for (std::size_t j = 0; j < cols; ++j) {
            means[j] = std::sqrt(result[j * cols + j]);
            require(means[j] != 0.0, "NON_FINITE_RESULT");
        }
        for (std::size_t j = 0; j < cols; ++j)
            for (std::size_t k = j; k < cols; ++k)
                result[j * cols + k] = (j == cols - 1 ? last : result[j * cols + k]) /
                                        (means[j] * means[k]);
    }
    for (std::size_t j = 0; j < cols; ++j)
        for (std::size_t k = j + 1; k < cols; ++k)
            result[k * cols + j] = result[j * cols + k];
}
namespace {
double covariance_pair(const Value &x, const Value &y, Workspace &work) {
    require(x.size() >= 2, "INSUFFICIENT_SAMPLE");
    const double mx = reduce_value(Op::mean, x, nullptr, 1, 0.5, work);
    const double my = reduce_value(Op::mean, y, nullptr, 1, 0.5, work);
    double total = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        total += (x.f(i) - mx) * (y.f(i) - my);
    return total / static_cast<double>(x.size() - 1);
}
void covariance_matrix(const Prepared &p, Output &out, Workspace &work, Isa isa, Audit &audit) {
    const auto &x = p.args[0];
    const auto rows = x.shape.dim[0], cols = x.shape.dim[1];
    require(rows >= 2, "INSUFFICIENT_SAMPLE");
    if (cols == 0) return;
    auto *result = static_cast<double *>(out.data);
    // No lower-triangle writes until every mean has been consumed.
    auto *means = result + (cols - 1) * cols;
    if (isa == Isa::scalar || !library_covariance_upper(x, result, work, audit)) {
        for (std::size_t j = 0; j < cols; ++j)
            means[j] = reduce_value(Op::mean, x.column(j), nullptr, 1, 0.5, work);
        for (std::size_t j = 0; j < cols; ++j)
            for (std::size_t k = j; k < cols; ++k) {
                double sum = 0.0;
                for (std::size_t i = 0; i < rows; ++i)
                    sum += (x.at(i, j) - means[j]) * (x.at(i, k) - means[k]);
                const double value = sum / static_cast<double>(rows - 1);
                result[j * cols + k] = value;
            }
    }
    finish_covariance_matrix(out, p.spec->op == Op::correlation);
}
void solve(const Prepared &p, Output &out, Workspace &work, Isa isa, Audit &audit) {
    const auto &matrix = p.args[0];
    const auto &y = p.args[1];
    const auto n = y.size();
    if (n == 0)
        return;
    auto *coefficients = work.doubles.data();
    auto *rhs = coefficients + n * n;
    auto *result = static_cast<double *>(out.data);
    for (std::size_t i = 0; i < n; ++i) {
        rhs[i] = y.f(i);
        for (std::size_t j = 0; j < n; ++j)
            coefficients[i * n + j] = matrix.at(i, j);
    }
    audit.algorithm_copy_bytes = (n * n + n) * sizeof(double);
    if (isa == Isa::scalar || !library_eliminate_multi<false>(coefficients, rhs, n, 1, audit))
    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot_row = col;
        double magnitude = std::abs(coefficients[col * n + col]);
        for (std::size_t candidate = col + 1; candidate < n; ++candidate) {
            const double test = std::abs(coefficients[candidate * n + col]);
            if (test > magnitude) {
                magnitude = test;
                pivot_row = candidate;
            }
        }
        require(!(magnitude <= 1e-14), "SINGULAR_MATRIX");
        if (pivot_row != col) {
            for (std::size_t j = 0; j < n; ++j)
                std::swap(coefficients[col * n + j], coefficients[pivot_row * n + j]);
            std::swap(rhs[col], rhs[pivot_row]);
        }
        const double pivot = coefficients[col * n + col];
        for (std::size_t row = col + 1; row < n; ++row) {
            const double factor = coefficients[row * n + col] / pivot;
            coefficients[row * n + col] = 0.0;
            for (std::size_t j = col + 1; j < n; ++j)
                coefficients[row * n + j] -= factor * coefficients[col * n + j];
            rhs[row] -= factor * rhs[col];
        }
    }
    for (std::size_t reverse = 0; reverse < n; ++reverse) {
        const auto row = n - 1 - reverse;
        double total = rhs[row];
        for (std::size_t j = row + 1; j < n; ++j)
            total -= coefficients[row * n + j] * result[j];
        const double pivot = coefficients[row * n + row];
        require(!(std::abs(pivot) <= 1e-14), "SINGULAR_MATRIX");
        result[row] = total / pivot;
    }
}
} // namespace

void matrix(const Prepared &p, Output &out, Workspace &work, Isa requested, Audit &audit) {
    if (p.spec->op == Op::affine_contract) {
        affine_contract_execute(p, out, requested, audit);
        return;
    }
    const auto op = p.spec->op;
    const auto &a = p.args[0];
    const auto &b = p.args[1];
    if (op == Op::dot) {
        out.set(0, dot_value(a, b));
        return;
    }
    if (op == Op::trace) {
        double sum = 0;
        for (std::size_t i = 0; i < std::min(a.shape.dim[0], a.shape.dim[1]); ++i)
            sum += a.at(i, i);
        out.set(0, sum);
        return;
    }
    if (op == Op::covariance || op == Op::correlation) {
        if (p.count == 1) {
            covariance_matrix(p, out, work, requested, audit);
            return;
        }
        double cov = covariance_pair(a, b, work);
        if (op == Op::correlation) {
            const double sx = reduce_value(Op::std, a, nullptr, 1, 0.5, work);
            const double sy = reduce_value(Op::std, b, nullptr, 1, 0.5, work);
            require(sx != 0 && sy != 0, "NON_FINITE_RESULT");
            cov /= sx * sy;
        }
        out.set(0, cov);
        return;
    }
    if (op == Op::solve) {
        solve(p, out, work, requested, audit);
        return;
    }
    auto *result = static_cast<double *>(out.data);
    if (op == Op::matvec) {
        if (requested != Isa::scalar && library_matvec(a, b, result, audit)) return;
        matvec_value(a, b, result);
        return;
    }
    if (op == Op::outer) {
        for (std::size_t i = 0; i < a.size(); ++i)
            for (std::size_t j = 0; j < b.size(); ++j)
                result[i * b.size() + j] = a.f(i) * b.f(j);
        return;
    }
    if (op == Op::matmul && out.order == MemoryOrder::f) {
        if (a.shape.dim[1] == 0) {
            if (out.shape.size()) std::fill_n(result, out.shape.size(), 0.0);
            return;
        }
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
        if (requested == Isa::automatic && a.shape.dim[0] >= 4 &&
            a.shape.dim[1] >= 32 && b.shape.dim[1] >= 4) {
            if (__builtin_available(macOS 15.0, *)) {
                // Reuse the admitted Eigen panels for rectangular and large F
                // roots. Its mapper writes the final F buffer without a C stage.
                if (library_matmul_packed(a, b, result, work, audit, true)) return;
                const bool c_right = b.stride[1] == 1 &&
                    b.stride[0] >= static_cast<std::ptrdiff_t>(b.shape.dim[1]);
                const bool f_right = b.stride[0] == 1 &&
                    b.stride[1] >= static_cast<std::ptrdiff_t>(a.shape.dim[1]);
                constexpr std::size_t target_working_set_bytes = 64 * 1024;
                constexpr std::size_t bytes_per_inner = (4 + 8) * sizeof(double);
                if ((c_right || f_right) && a.shape.dim[0] >= 32 && b.shape.dim[1] >= 32 &&
                    a.shape.dim[1] > target_working_set_bytes / bytes_per_inner)
                    return library_matmul_f_blocked(a, b, result, audit);
                return library_matmul_f_root(a, b, result, audit);
            }
        }
#endif
        // Canonical ordered dot is also the portable/exceptional F writer.
        // Preserve A*B operand order; do not swap them to emulate a transpose.
        for (std::size_t col = 0; col < b.shape.dim[1]; ++col)
            for (std::size_t row = 0; row < a.shape.dim[0]; ++row)
                result[col * a.shape.dim[0] + row] = dot_value(a.row(row), b.column(col));
        return;
    }
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    // Match shape-only scratch admission before the unbuffered rectangular
    // path. The library rechecks actual dense strides and borrows its panels.
    if (op == Op::matmul && requested == Isa::automatic &&
        a.shape.dim[0] >= 32 && a.shape.dim[0] <= 128 &&
        a.shape.dim[1] >= 1 && a.shape.dim[1] <= 128 && b.shape.dim[1] >= 128) {
        if (__builtin_available(macOS 15.0, *)) {
            if (library_matmul_packed(a, b, result, work, audit)) return;
        }
    }
    // Reuse the existing ordered Eigen packets for rectangular trajectories.
    // Independent output lanes keep the ascending-k binary64 fold, including
    // IEEE exceptional classes and comparison consumers; no GEMM reordering.
    if (op == Op::matmul && requested == Isa::automatic &&
        a.shape.dim[0] >= 1 && a.shape.dim[1] >= 1 && a.shape.dim[1] <= 128 &&
        b.shape.dim[1] >= 128) {
        if (__builtin_available(macOS 15.0, *)) {
            // The existing C-right path is faster for tiny rows; only use
            // the new row tile for the measured F-right scalar bottleneck.
            if (a.shape.dim[0] >= 4 && b.stride[1] == 1 &&
                b.stride[0] >= std::ptrdiff_t(b.shape.dim[1]))
                return library_matmul_c(a, b, result, audit);
            if (b.stride[0] == 1 && b.stride[1] >= std::ptrdiff_t(a.shape.dim[1]))
                return library_matmul_f(a, b, result, audit);
        }
    }
    if (op == Op::matmul && requested == Isa::automatic &&
        a.shape.dim[0] >= 32 && a.shape.dim[1] >= 32 && b.shape.dim[1] >= 32) {
        const bool c_right = b.stride[1] == 1 &&
            b.stride[0] >= static_cast<std::ptrdiff_t>(b.shape.dim[1]);
        const bool f_right = b.stride[0] == 1 &&
            b.stride[1] >= static_cast<std::ptrdiff_t>(a.shape.dim[1]);
        if (__builtin_available(macOS 15.0, *)) {
            if ((c_right || f_right) && a.stride[0] == 0)
                return library_matmul_repeated_rows(a, b, result, audit);
            if (library_matmul_packed(a, b, result, work, audit)) return;
            // A fixed working-set heuristic, not a hardware cache-size query.
            // Divide the bound first so large inner dimensions cannot overflow.
            constexpr std::size_t target_working_set_bytes = 64 * 1024;
            constexpr std::size_t bytes_per_inner = (4 + 8) * sizeof(double);
            if (c_right && a.shape.dim[1] > target_working_set_bytes / bytes_per_inner)
                return library_matmul_c_blocked(a, b, result, audit);
            // Below the blocked gate, retain the previously admitted unblocked
            // square tiles; other layouts continue through the existing paths.
            if (c_right && b.stride[0] == static_cast<std::ptrdiff_t>(b.shape.dim[1]) &&
                a.shape.dim[0] == a.shape.dim[1] && a.shape.dim[1] == b.shape.dim[1] &&
                b.shape.dim[1] % 8 == 0)
                return library_matmul_c(a, b, result, audit);
            if (f_right) return library_matmul_f(a, b, result, audit);
        }
    }
#endif
    if (op == Op::matmul && simd_matmul(a, b, out, requested, audit))
        return;
    if (out.shape.size())
        std::fill_n(result, out.shape.size(), 0.0);
    if (op == Op::diag) {
        for (std::size_t i = 0; i < a.size(); ++i)
            result[i * a.size() + i] = a.f(i);
        return;
    }
    require(op == Op::matmul, "INVALID_MATRIX_OPERATOR");
    const auto rows = a.shape.dim[0], inner = a.shape.dim[1], cols = b.shape.dim[1];
    // Time order of each output reduction is unchanged; j is independent and cache-local.
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t k = 0; k < inner; ++k) {
            const double scale = a.at(i, k);
            for (std::size_t j = 0; j < cols; ++j)
                result[i * cols + j] += scale * b.at(k, j);
        }
}
} // namespace calmetrics_engine::ops
