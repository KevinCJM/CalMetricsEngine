#pragma once
#include "calmetrics_engine/operators.hpp"

namespace calmetrics_engine::ops::detail {
// Target-independent loop semantics; the traits supply one AOT ISA implementation.
// Operation dispatch occurs once per block, not once per element.
template <Op op, class V>
std::size_t vector_loop(const Value &x, const Value &y, Output &out, std::size_t n) {
    const auto *lhs = x.shape.rank == 0 ? &x.scalar : static_cast<const double *>(x.data);
    const auto *rhs = y.shape.rank == 0 ? &y.scalar : static_cast<const double *>(y.data);
    const auto stop = n - n % V::width;
    for (std::size_t i = 0; i < stop; i += V::width) {
        const auto a = x.shape.rank == 0 ? V::broadcast(*lhs) : V::load(lhs + i);
        const auto b = y.shape.rank == 0 ? V::broadcast(*rhs) : V::load(rhs + i);
        if constexpr (op >= Op::equal && op <= Op::greater_equal) {
            auto m = V::eq(a, b);
            if constexpr (op == Op::not_equal)
                m = V::invert(m);
            if constexpr (op == Op::less_than)
                m = V::lt(a, b);
            if constexpr (op == Op::less_equal)
                m = V::le(a, b);
            if constexpr (op == Op::greater_than)
                m = V::lt(b, a);
            if constexpr (op == Op::greater_equal)
                m = V::le(b, a);
            V::store_mask(static_cast<std::uint8_t *>(out.data) + i, m);
        } else if constexpr (op == Op::finite_mask) {
            const auto m =
                V::lt(V::absolute(a), V::broadcast(std::numeric_limits<double>::infinity()));
            V::store_mask(static_cast<std::uint8_t *>(out.data) + i, m);
        } else {
            auto result = a;
            if constexpr (op == Op::add)
                result = V::add(a, b);
            if constexpr (op == Op::subtract)
                result = V::sub(a, b);
            if constexpr (op == Op::multiply)
                result = V::mul(a, b);
            if constexpr (op == Op::divide)
                result = V::div(a, b);
            // Preserve first-operand tie/NaN semantics; ISA min/max instructions differ.
            if constexpr (op == Op::minimum)
                result = V::select(V::lt(b, a), b, a);
            if constexpr (op == Op::maximum)
                result = V::select(V::lt(a, b), b, a);
            if constexpr (op == Op::negate)
                result = V::negate(a);
            if constexpr (op == Op::absolute)
                result = V::absolute(a);
            if constexpr (op == Op::sqrt)
                result = V::sqrt(a);
            if constexpr (op == Op::reciprocal)
                result = V::div(V::broadcast(1.0), a);
            V::store(static_cast<double *>(out.data) + i, result);
        }
    }
    return stop;
}
// Register-blocked 4-by-(2 SIMD vectors) GEMM. No packed copies of either input.
// Each output keeps the original k-order and uses separate multiply/add, not FMA.
template <class V> std::size_t matrix_loop(const Value &a, const Value &b, Output &out) {
    const auto rows = a.shape.dim[0], inner = a.shape.dim[1], cols = b.shape.dim[1];
    constexpr auto block_columns = 2 * V::width;
    if (inner == 0 || rows < 4 || cols < block_columns || b.stride[1] != 1)
        return 0;
    const auto full_rows = rows - rows % 4, full_cols = cols - cols % block_columns;
    auto *result = static_cast<double *>(out.data);
    const auto *source = static_cast<const double *>(b.data);
    for (std::size_t i = 0; i < full_rows; i += 4) {
        for (std::size_t j = 0; j < full_cols; j += block_columns) {
            auto c00 = V::broadcast(0.0), c01 = V::broadcast(0.0);
            auto c10 = V::broadcast(0.0), c11 = V::broadcast(0.0);
            auto c20 = V::broadcast(0.0), c21 = V::broadcast(0.0);
            auto c30 = V::broadcast(0.0), c31 = V::broadcast(0.0);
            for (std::size_t k = 0; k < inner; ++k) {
                const auto *row = source + static_cast<std::ptrdiff_t>(k) * b.stride[0] + j;
                const auto b0 = V::load(row), b1 = V::load(row + V::width);
                const auto a0 = V::broadcast(a.at(i, k)), a1 = V::broadcast(a.at(i + 1, k));
                const auto a2 = V::broadcast(a.at(i + 2, k)), a3 = V::broadcast(a.at(i + 3, k));
                c00 = V::add(c00, V::mul(a0, b0));
                c01 = V::add(c01, V::mul(a0, b1));
                c10 = V::add(c10, V::mul(a1, b0));
                c11 = V::add(c11, V::mul(a1, b1));
                c20 = V::add(c20, V::mul(a2, b0));
                c21 = V::add(c21, V::mul(a2, b1));
                c30 = V::add(c30, V::mul(a3, b0));
                c31 = V::add(c31, V::mul(a3, b1));
            }
            V::store(result + i * cols + j, c00);
            V::store(result + i * cols + j + V::width, c01);
            V::store(result + (i + 1) * cols + j, c10);
            V::store(result + (i + 1) * cols + j + V::width, c11);
            V::store(result + (i + 2) * cols + j, c20);
            V::store(result + (i + 2) * cols + j + V::width, c21);
            V::store(result + (i + 3) * cols + j, c30);
            V::store(result + (i + 3) * cols + j + V::width, c31);
        }
        for (std::size_t row = i; row < i + 4; ++row)
            for (std::size_t col = full_cols; col < cols; ++col) {
                double value = 0.0;
                for (std::size_t k = 0; k < inner; ++k)
                    value += a.at(row, k) * b.at(k, col);
                result[row * cols + col] = value;
            }
    }
    for (std::size_t row = full_rows; row < rows; ++row)
        for (std::size_t col = 0; col < cols; ++col) {
            double value = 0.0;
            for (std::size_t k = 0; k < inner; ++k)
                value += a.at(row, k) * b.at(k, col);
            result[row * cols + col] = value;
        }
    return full_rows * full_cols;
}

template <class V>
std::size_t vector_dispatch(Op op, const Value &x, const Value &y, Output &out, std::size_t n) {
    switch (op) {
#define CASE(name)                                                                                 \
    case Op::name:                                                                                 \
        return vector_loop<Op::name, V>(x, y, out, n);
        CASE(add)
        CASE(subtract) CASE(multiply) CASE(divide) CASE(minimum) CASE(maximum) CASE(negate)
            CASE(absolute) CASE(sqrt) CASE(reciprocal) CASE(equal) CASE(not_equal) CASE(less_than)
                CASE(less_equal) CASE(greater_than) CASE(greater_equal) CASE(finite_mask)
#undef CASE
                    default : return 0;
    }
}
} // namespace calmetrics_engine::ops::detail
