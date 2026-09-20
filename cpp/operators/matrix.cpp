#include "calmetrics_engine/operators.hpp"
#include <algorithm>
#include <cmath>

namespace calmetrics_engine::ops {
double dot_value(const Value &x, const Value &y) {
    double result = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        result += x.f(i) * y.f(i);
    return result;
}
void matvec_value(const Value &a, const Value &x, double *out) {
    for (std::size_t row = 0; row < a.shape.dim[0]; ++row)
        out[row] = dot_value(a.row(row), x);
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
void covariance_matrix(const Prepared &p, Output &out, Workspace &work) {
    const auto &x = p.args[0];
    const auto rows = x.shape.dim[0], cols = x.shape.dim[1];
    require(rows >= 2, "INSUFFICIENT_SAMPLE");
    auto *means = work.doubles.data();
    auto *result = static_cast<double *>(out.data);
    for (std::size_t j = 0; j < cols; ++j)
        means[j] = reduce_value(Op::mean, x.column(j), nullptr, 1, 0.5, work);
    for (std::size_t j = 0; j < cols; ++j)
        for (std::size_t k = j; k < cols; ++k) {
            double sum = 0.0;
            for (std::size_t i = 0; i < rows; ++i)
                sum += (x.at(i, j) - means[j]) * (x.at(i, k) - means[k]);
            const double value = sum / static_cast<double>(rows - 1);
            result[j * cols + k] = value;
            result[k * cols + j] = value;
        }
    if (p.spec->op != Op::correlation)
        return;
    for (std::size_t j = 0; j < cols; ++j) {
        means[cols + j] = std::sqrt(result[j * cols + j]);
        require(means[cols + j] != 0.0, "NON_FINITE_RESULT");
    }
    for (std::size_t j = 0; j < cols; ++j)
        for (std::size_t k = 0; k < cols; ++k)
            result[j * cols + k] /= means[cols + j] * means[cols + k];
}
void solve(const Prepared &p, Output &out, Workspace &work, Audit &audit) {
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
            covariance_matrix(p, out, work);
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
        solve(p, out, work, audit);
        return;
    }
    auto *result = static_cast<double *>(out.data);
    if (op == Op::matvec) {
        matvec_value(a, b, result);
        return;
    }
    if (op == Op::outer) {
        for (std::size_t i = 0; i < a.size(); ++i)
            for (std::size_t j = 0; j < b.size(); ++j)
                result[i * b.size() + j] = a.f(i) * b.f(j);
        return;
    }
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
