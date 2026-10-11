#include "calmetrics_engine/matrix_backend.hpp"
#include "calmetrics_engine/operators.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <climits>
#ifdef CALMETRICS_HAVE_EIGEN_ROTATION
// Header-only, with no private pool, BLAS dispatch or fast-math.
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_MPL2_ONLY
#define EIGEN_FAST_MATH 0
#include <Eigen/Jacobi>
#endif

namespace calmetrics_engine::ops {
MatmulBoundsStatus library_matmul_input_bounds(const Value &input, MatmulInputBounds &result) {
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    if (__builtin_available(macOS 15.0, *)) {
        if (input.kind != Kind::number || input.shape.rank != 2 || !input.data)
            return MatmulBoundsStatus::unavailable;
        const auto rows = input.shape.dim[0], columns = input.shape.dim[1];
        if (!rows || !columns || rows > 8192 || columns > 8192)
            return MatmulBoundsStatus::unavailable;
        const bool c = input.stride[1] == 1 && input.stride[0] >= std::ptrdiff_t(columns) &&
                       input.stride[0] <= INT_MAX;
        const bool f = input.stride[0] == 1 && input.stride[1] >= std::ptrdiff_t(rows) &&
                       input.stride[1] <= INT_MAX;
        if (!c && !f) return MatmulBoundsStatus::unavailable;
        const auto inner = c ? columns : rows, outer = c ? rows : columns;
        const auto leading = c ? input.stride[0] : input.stride[1];
        const auto *data = static_cast<const double *>(input.data);
        MatmulInputBounds bounds;
        constexpr Eigen::Index width = 32;
        using Tile = Eigen::Array<double, width, 1>;
        using Read = Eigen::Map<const Tile, Eigen::Unaligned>;
        for (std::size_t j = 0; j < outer; ++j) {
            const auto *line = data + std::ptrdiff_t(j) * leading;
            std::size_t i = 0;
            for (; i + width <= inner; i += width) {
                const Read values(line + i);
                // Public Eigen reductions borrow a bounded cache-sized span.
                // No arithmetic sum, packing, pool, retained pointer or heap.
                if (!values.isFinite().all()) return MatmulBoundsStatus::rejected;
                const auto absolute = values.abs();
                const double maximum = absolute.maxCoeff(), minimum = absolute.minCoeff();
                if (minimum == 0.0) return MatmulBoundsStatus::scalar_required;
                if (minimum < 0x1p-128 || maximum > 0x1p128)
                    return MatmulBoundsStatus::rejected;
                bounds.maximum = std::max(bounds.maximum, maximum);
                bounds.negative = bounds.negative || values.minCoeff() < 0.0;
            }
            for (; i < inner; ++i) {
                const double value = line[i];
                if (!std::isfinite(value)) return MatmulBoundsStatus::rejected;
                if (value == 0.0) return MatmulBoundsStatus::scalar_required;
                const double absolute = std::abs(value);
                if (absolute < 0x1p-128 || absolute > 0x1p128)
                    return MatmulBoundsStatus::rejected;
                bounds.maximum = std::max(bounds.maximum, absolute);
                bounds.negative = bounds.negative || std::signbit(value);
            }
        }
        result = bounds;
        return MatmulBoundsStatus::valid;
    }
#else
    (void)input; (void)result;
#endif
    return MatmulBoundsStatus::unavailable;
}
bool library_numeric_zero_separated(const double *data,std::size_t count,double error) {
#ifdef CALMETRICS_HAVE_EIGEN_ROTATION
    if (!data || !count || count>65536 || !(error>=0.0)) return false;
    using Span=Eigen::Array<double,Eigen::Dynamic,1>;
    const Eigen::Map<const Span,Eigen::Unaligned> values(data,count);
    return values.isFinite().all() && values.abs().minCoeff()>error;
#else
    (void)data;(void)count;(void)error;
    return false;
#endif
}
bool try_f_rectangle_copy(const double *source, std::size_t source_stride,
                         double *target, std::size_t target_stride,
                         std::size_t rows, std::size_t columns) {
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    if (__builtin_available(macOS 15.0, *)) {
        if (rows < 32 || columns < 32 || source_stride < rows || target_stride < columns)
            return false;
        using F = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
        using C = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        // Bounded maps borrow input and final output. Eigen moves values without
        // arithmetic, an intermediate matrix, a pool or a dynamic allocation.
        constexpr std::size_t tile = 32;
        for (std::size_t row = 0; row < rows; row += tile)
            for (std::size_t column = 0; column < columns; column += tile) {
                const auto height = std::min(tile, rows - row);
                const auto width = std::min(tile, columns - column);
                Eigen::Map<const F, Eigen::Unaligned, Eigen::OuterStride<>> in(
                    source + row + column * source_stride, height, width,
                    Eigen::OuterStride<>(source_stride));
                Eigen::Map<C, Eigen::Unaligned, Eigen::OuterStride<>> out(
                    target + row * target_stride + column, height, width,
                    Eigen::OuterStride<>(target_stride));
                out = in;
            }
        return true;
    }
#else
    (void)source; (void)source_stride; (void)target; (void)target_stride;
    (void)rows; (void)columns;
#endif
    return false;
}
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
namespace {
// Register-sized RHS panels retain each element's ascending coefficient fold.
// These Maps borrow the independent C result, never pack or allocate a matrix.
template<int Width>
EIGEN_ALWAYS_INLINE void triangular_rhs_tile(const Value &a, double *y,
    std::size_t row, std::size_t columns, std::size_t first, std::size_t last,
    std::size_t column) {
    using Tile = Eigen::Array<double, Width, 1>;
    using Read = Eigen::Map<const Tile, Eigen::Unaligned>;
    Tile values = Read(y + row * columns + column);
    for (auto k = first; k < last; ++k)
        values -= a.at(row,k) * Read(y + k * columns + column);
    Eigen::Map<Tile, Eigen::Unaligned>(y + row * columns + column) = values;
}
// Fixed 128-observation panels retain each cell's original ascending fold.
// Last output row remains mean storage until the very last diagonal is written.
// Fixed diagonal segments prevent dynamic stores from escaping the accumulator.
template<int Width, int Start>
EIGEN_ALWAYS_INLINE auto covariance_read_tail(const double* source) {
    using Row = Eigen::Array<double, 1, Width, Eigen::RowMajor>;
    using Tail = Eigen::Array<double, 1, Width - Start, Eigen::RowMajor>;
    Row result = Row::Zero();
    result.template tail<Width - Start>() = Eigen::Map<const Tail>(source + Start);
    return result;
}
template<int Width, int Start>
EIGEN_ALWAYS_INLINE void covariance_write_tail(double* target,
    const Eigen::Array<double, 1, Width, Eigen::RowMajor>& value) {
    using Tail = Eigen::Array<double, 1, Width - Start, Eigen::RowMajor>;
    Eigen::Map<Tail>(target + Start) = value.template tail<Width - Start>();
}
template<int Width, int Offset, int ColumnStride = 1>
EIGEN_ALWAYS_INLINE void covariance_accumulate_tile(const double* x, Eigen::Index count,
    Eigen::Index d, Eigen::Index j, Eigen::Index k, std::ptrdiff_t rs,
    const double* means, double* out, std::ptrdiff_t cs = 1) {
    using Row = Eigen::Array<double, 1, Width, Eigen::RowMajor>;
    using Read = Eigen::Map<const Row, Eigen::Unaligned, Eigen::InnerStride<ColumnStride>>;
    constexpr int s0 = Offset < 0 ? 0 : Offset;
    constexpr int s1 = Offset < 0 ? 0 : Offset + 1;
    constexpr int s2 = Offset < 0 ? 0 : Offset + 2;
    constexpr int s3 = Offset < 0 ? 0 : Offset + 3;
    Row a0 = covariance_read_tail<Width, s0>(out + j * d + k);
    Row a1 = covariance_read_tail<Width, s1>(out + (j + 1) * d + k);
    Row a2 = covariance_read_tail<Width, s2>(out + (j + 2) * d + k);
    Row a3 = covariance_read_tail<Width, s3>(out + (j + 3) * d + k);
    const Row center = Eigen::Map<const Row>(means + k);
    const double m0 = means[j], m1 = means[j + 1], m2 = means[j + 2], m3 = means[j + 3];
    for (Eigen::Index i = 0; i < count; ++i) {
        const auto* row = x + i * rs;
        const Row right = Read(row + k * cs, Eigen::InnerStride<ColumnStride>(cs)) - center;
        a0 += (row[j * cs] - m0) * right;
        a1 += (row[(j + 1) * cs] - m1) * right;
        a2 += (row[(j + 2) * cs] - m2) * right;
        a3 += (row[(j + 3) * cs] - m3) * right;
    }
    covariance_write_tail<Width, s0>(out + j * d + k, a0);
    covariance_write_tail<Width, s1>(out + (j + 1) * d + k, a1);
    covariance_write_tail<Width, s2>(out + (j + 2) * d + k, a2);
    covariance_write_tail<Width, s3>(out + (j + 3) * d + k, a3);
}
EIGEN_ALWAYS_INLINE double covariance_ordered_sum(const double* x, Eigen::Index n,
    Eigen::Index j, Eigen::Index k, std::ptrdiff_t rs, const double* means,
    std::ptrdiff_t cs = 1) {
    const double left = means[j], right = means[k];
    double total = 0.0;
    for (Eigen::Index i = 0; i < n; ++i) {
        const auto* row = x + i * rs;
        total += (row[j * cs] - left) * (row[k * cs] - right);
    }
    return total;
}
template<int ColumnStride = 1>
EIGEN_ALWAYS_INLINE void covariance_c_means_only(const Value &input, double *target) {
    const auto *x = static_cast<const double *>(input.data);
    const auto n = static_cast<Eigen::Index>(input.shape.dim[0]);
    const auto d = static_cast<Eigen::Index>(input.shape.dim[1]);
    const auto rs = input.stride[0];
    using Row = Eigen::Array<double, 1, Eigen::Dynamic>;
    using Read = Eigen::Map<const Row, Eigen::Unaligned, Eigen::InnerStride<ColumnStride>>;
    // Descending F columns share a small working set; each feature still folds
    // logical observations in the original order. Unit-stride C keeps full rows.
    const Eigen::Index width = ColumnStride == 1 ? d : 8;
    for (Eigen::Index first = 0; first < d; first += width) {
        const auto columns = std::min(width, d - first);
        Eigen::Map<Row> means(target + first, columns);
        means.setZero();
        for (Eigen::Index i = 0; i < n; ++i) {
            Read row(x + i * rs + first * input.stride[1], columns,
                     Eigen::InnerStride<ColumnStride>(input.stride[1]));
            means += (row - means) / static_cast<double>(i + 1);
        }
    }
}
template<int ColumnStride = 1>
EIGEN_DONT_INLINE void covariance_c_means(const Value &input, double *out,
    Workspace &, Audit &) {
    const auto d = static_cast<Eigen::Index>(input.shape.dim[1]);
    using Row = Eigen::Array<double, 1, Eigen::Dynamic>;
    covariance_c_means_only<ColumnStride>(input, out + (d - 1) * d);
    for (Eigen::Index j = 0; j < d - 1; ++j)
        Eigen::Map<Row>(out + j * d + j, d - j).setZero();
}
// The bottom four features are one final diagonal tile. Computing it
// after all other pairs permits the final diagonal to overwrite its mean.
template<int ColumnStride = 1>
EIGEN_DONT_INLINE bool covariance_c_bottom(const Value &input, double *out,
    Workspace &, Audit &) {
    const auto *x = static_cast<const double *>(input.data);
    const auto n = static_cast<Eigen::Index>(input.shape.dim[0]);
    const auto d = static_cast<Eigen::Index>(input.shape.dim[1]);
    const auto rs = input.stride[0];
    const auto cs = ColumnStride == 1 ? std::ptrdiff_t{1} : input.stride[1];
    using Row = Eigen::Array<double, 1, 4, Eigen::RowMajor>;
    using Read = Eigen::Map<const Row, Eigen::Unaligned, Eigen::InnerStride<ColumnStride>>;
    const auto j = d - 4;
    const Row center = Eigen::Map<const Row>(out + (d - 1) * d + j);
    Row a0 = Row::Zero(), a1 = Row::Zero(), a2 = Row::Zero(), a3 = Row::Zero();
    for (Eigen::Index i = 0; i < n; ++i) {
        const Row delta = Read(x + i * rs + j * cs, Eigen::InnerStride<ColumnStride>(cs)) - center;
        a0 += delta[0] * delta;
        a1 += delta[1] * delta;
        a2 += delta[2] * delta;
        a3 += delta[3] * delta;
    }
    const double denominator = static_cast<double>(n - 1);
    a0 /= denominator; a1 /= denominator; a2 /= denominator; a3 /= denominator;
    covariance_write_tail<4, 0>(out + j * d + j, a0);
    covariance_write_tail<4, 1>(out + (j + 1) * d + j, a1);
    covariance_write_tail<4, 2>(out + (j + 2) * d + j, a2);
    covariance_write_tail<4, 3>(out + (j + 3) * d + j, a3);
    return true;
}
template<int ColumnStride = 1>
EIGEN_DONT_INLINE bool covariance_c_upper(const Value &input, double *out,
    Workspace &work, Audit &audit) {
    constexpr int Width = 4;
    const auto *x = static_cast<const double *>(input.data);
    const auto n = static_cast<Eigen::Index>(input.shape.dim[0]);
    const auto d = static_cast<Eigen::Index>(input.shape.dim[1]);
    const auto rs = input.stride[0];
    const auto cs = ColumnStride == 1 ? std::ptrdiff_t{1} : input.stride[1];
    using Row = Eigen::Array<double, 1, Eigen::Dynamic>;
    const auto* means = out + (d - 1) * d;
    const auto full_rows = (d - 1) - (d - 1) % 4;
    const auto full_columns = d - d % Width;
    constexpr Eigen::Index block = 128;
    for (Eigen::Index start = 0; start < n; start += block) {
        const auto count = std::min(block, n - start);
        const auto* panel = x + start * rs;
        for (Eigen::Index j = 0; j < full_rows; j += 4) {
            for (Eigen::Index k = (j / Width) * Width; k < full_columns; k += Width) {
                if (k >= j + 4) covariance_accumulate_tile<Width, -1, ColumnStride>(panel, count, d, j, k, rs, means, out, cs);
                else covariance_accumulate_tile<Width, 0, ColumnStride>(panel, count, d, j, k, rs, means, out, cs);
            }
        }
    }
    for (Eigen::Index j = 0; j < full_rows; j += 4)
        for (Eigen::Index row = j; row < j + 4; ++row)
            for (Eigen::Index col = std::max(row, full_columns); col < d; ++col)
                out[row * d + col] = covariance_ordered_sum(x, n, row, col, rs, means, cs);
    for (Eigen::Index row = d % 4 == 0 ? d - 1 : full_rows; row < d - 1; ++row)
        for (Eigen::Index col = row; col < d; ++col)
            out[row * d + col] = covariance_ordered_sum(x, n, row, col, rs, means, cs);
    for (Eigen::Index j = 0; j < d - 1; ++j)
        Eigen::Map<Row>(out + j * d + j, d - j) /= static_cast<double>(n - 1);
    if (d % 4 == 0) return covariance_c_bottom<ColumnStride>(input, out, work, audit);
    out[d * d - 1] = covariance_ordered_sum(x, n, d - 1, d - 1, rs, means, cs) / static_cast<double>(n - 1);
    return true;
}

// The identical four-reference signature permits tail calls after mean setup,
// avoiding a driver frame around the numerical tile or final diagonal tile.
template<int ColumnStride = 1>
EIGEN_DONT_INLINE bool covariance_c(const Value &input, double *out,
    Workspace &work, Audit &audit) {
    covariance_c_means<ColumnStride>(input, out, work, audit);
    ++audit.eigen_covariance_calls;
    return covariance_c_upper<ColumnStride>(input, out, work, audit);
}

// Fuse independent F column pairs through public Eigen expressions. The
// two-packet observation reduction tree matches the pinned Eigen 3.4 sum.
namespace covariance_f_detail {
using Row = Eigen::Array<double, 1, Eigen::Dynamic>;
using Read = Eigen::Map<const Row, Eigen::Unaligned>;
using Four = Eigen::Array<double, 4, 1>;
using Two = Eigen::Array<double, 2, 1>;
template<int Width>
EIGEN_DONT_INLINE void gather_means(const double *input, Eigen::Index n,
    Eigen::Index d, Eigen::Index stride, double *means) {
    using Tile = Eigen::Array<double, Width, 1>;
    const auto full = d - d % Width;
    for (Eigen::Index j = 0; j < full; j += Width) {
        Tile mean = Tile::Zero();
        for (Eigen::Index i = 0; i < n; ++i) {
            Tile row;
            // Public fixed Array gathers; SIMD evaluates independent columns.
            for (int lane = 0; lane < Width; ++lane)
                row[lane] = input[(j + lane) * stride + i];
            mean += (row - mean) / static_cast<double>(i + 1);
        }
        Eigen::Map<Tile>(means + j) = mean;
    }
    for (Eigen::Index j = full; j < d; ++j) {
        double mean = 0.0;
        for (Eigen::Index i = 0; i < n; ++i)
            mean += (input[j * stride + i] - mean) / static_cast<double>(i + 1);
        means[j] = mean;
    }
}

EIGEN_ALWAYS_INLINE double original_pair(const double *left, const double *right,
                                         Eigen::Index n, double ml, double mr) {
    return ((Read(left, n) - ml) * (Read(right, n) - mr)).sum();
}

template<bool Centered, class Tile>
EIGEN_ALWAYS_INLINE Tile centered_values(const double *data, double mean) {
    const Eigen::Map<const Tile, Eigen::Unaligned> values(data);
    if constexpr (Centered) return values;
    else return values - mean;
}

template<bool Centered>
EIGEN_ALWAYS_INLINE double finish_pair(const Four &lanes,
    const double *left, const double *right, Eigen::Index n, double ml, double mr) {
    // Public Array spelling of the pinned arm64 two-packet reduction tree:
    // combine the two 2-lane chains, then optional packet, horizontal sum, odd.
    Two merged = lanes.template head<2>() + lanes.template tail<2>();
    const auto full = n - n % 4;
    if (n - full >= 2)
        merged += centered_values<Centered, Two>(left + full, ml) *
                  centered_values<Centered, Two>(right + full, mr);
    double result = merged.sum();
    if (n % 2) {
        if constexpr (Centered) result += left[n - 1] * right[n - 1];
        else result += (left[n - 1] - ml) * (right[n - 1] - mr);
    }
    return result;
}

template<bool Centered>
EIGEN_ALWAYS_INLINE void fused_pair_tile(const double *x, Eigen::Index n,
    Eigen::Index d, Eigen::Index cs, Eigen::Index j, Eigen::Index k,
    double *out, const double *means) {
    const auto *left0 = x + j * cs, *left1 = left0 + cs;
    const auto *right0 = x + k * cs, *right1 = right0 + cs;
    const double ml0 = Centered ? 0.0 : means[j], ml1 = Centered ? 0.0 : means[j + 1];
    const double mr0 = Centered ? 0.0 : means[k], mr1 = Centered ? 0.0 : means[k + 1];
    Four a00, a01, a10, a11;
    {
        const Four l0 = centered_values<Centered, Four>(left0, ml0), l1 = centered_values<Centered, Four>(left1, ml1);
        const Four r0 = centered_values<Centered, Four>(right0, mr0), r1 = centered_values<Centered, Four>(right1, mr1);
        // Seed with the first products, not +0; preserve signed-zero behavior.
        a00 = l0 * r0; a01 = l0 * r1; a10 = l1 * r0; a11 = l1 * r1;
    }
    const auto full = n - n % 4;
    for (Eigen::Index i = 4; i < full; i += 4) {
        const Four l0 = centered_values<Centered, Four>(left0 + i, ml0), l1 = centered_values<Centered, Four>(left1 + i, ml1);
        const Four r0 = centered_values<Centered, Four>(right0 + i, mr0), r1 = centered_values<Centered, Four>(right1 + i, mr1);
        a00 += l0 * r0; a01 += l0 * r1; a10 += l1 * r0; a11 += l1 * r1;
    }
    const double denominator = static_cast<double>(n - 1);
    out[j * d + k] = finish_pair<Centered>(a00, left0, right0, n, ml0, mr0) / denominator;
    out[j * d + k + 1] = finish_pair<Centered>(a01, left0, right1, n, ml0, mr1) / denominator;
    if (k != j)
        out[(j + 1) * d + k] = finish_pair<Centered>(a10, left1, right0, n, ml1, mr0) / denominator;
    out[(j + 1) * d + k + 1] = finish_pair<Centered>(a11, left1, right1, n, ml1, mr1) / denominator;
}

template<bool Centered = false>
EIGEN_DONT_INLINE bool upper(const Value &input, double *out, Workspace &, Audit &) {
    const auto *x = static_cast<const double *>(input.data);
    const auto n = static_cast<Eigen::Index>(input.shape.dim[0]);
    const auto d = static_cast<Eigen::Index>(input.shape.dim[1]);
    const auto cs = static_cast<Eigen::Index>(input.stride[1]);
    const auto *means = Centered ? nullptr : out + (d - 1) * d;
    const auto paired = d - d % 2;
    const double denominator = static_cast<double>(n - 1);
    for (Eigen::Index j = 0; j < paired; j += 2) {
        for (Eigen::Index k = j; k < paired; k += 2)
            fused_pair_tile<Centered>(x, n, d, cs, j, k, out, means);
        if (!Centered && paired != d) {
            out[j * d + paired] = original_pair(x + j * cs, x + paired * cs, n,
                                                means[j], means[paired]) / denominator;
            out[(j + 1) * d + paired] = original_pair(x + (j + 1) * cs, x + paired * cs, n,
                                                      means[j + 1], means[paired]) / denominator;
        }
    }
    // With even D, only the final 2x2 tile writes any part of the final row:
    // its final diagonal, after copying all four means to scalar values.
    // With odd D, the one remaining diagonal is the final mean consumer.
    if (!Centered && paired != d)
        out[d * d - 1] = original_pair(x + paired * cs, x + paired * cs, n,
                                      means[paired], means[paired]) / denominator;
    return true;
}

EIGEN_DONT_INLINE bool fast(const Value &input, double *out, Workspace &work, Audit &audit) {
    const auto n = static_cast<Eigen::Index>(input.shape.dim[0]);
    const auto d = static_cast<Eigen::Index>(input.shape.dim[1]);
    gather_means<8>(static_cast<const double *>(input.data), n, d,
                    input.stride[1], out + (d - 1) * d);
    ++audit.eigen_covariance_calls;
#if defined(__clang__)
    [[clang::musttail]]
#endif
    return upper(input, out, work, audit);
}
} // namespace covariance_f_detail

EIGEN_DONT_INLINE void covariance_f_fallback_means(const Value &x, double *output,
    Workspace &work, Audit &) {
    const auto d = static_cast<Eigen::Index>(x.shape.dim[1]);
    auto *means = output + (d - 1) * d;
    for (Eigen::Index j = 0; j < d; ++j)
        means[j] = reduce_value(Op::mean, x.column(j), nullptr, 1, 0.5, work);
}
EIGEN_DONT_INLINE bool covariance_f_fallback_products(const Value &x, double *output,
    Workspace &, Audit &audit) {
    const auto n = static_cast<Eigen::Index>(x.shape.dim[0]);
    const auto d = static_cast<Eigen::Index>(x.shape.dim[1]);
    const auto *means = output + (d - 1) * d;
    const auto *input = static_cast<const double *>(x.data);
    using Row = Eigen::Array<double, 1, Eigen::Dynamic>;
    for (Eigen::Index j = 0; j < d; ++j) {
        Eigen::Map<const Row> left(input + j * x.stride[1], n);
        for (Eigen::Index k = j; k < d; ++k) {
            Eigen::Map<const Row> right(input + k * x.stride[1], n);
            output[j * d + k] = ((left - means[j]) * (right - means[k])).sum()
                                 / static_cast<double>(n - 1);
        }
    }
    ++audit.eigen_covariance_calls;
    return true;
}
EIGEN_DONT_INLINE bool covariance_f_fallback(const Value &x, double *output,
    Workspace &work, Audit &audit) {
    covariance_f_fallback_means(x, output, work, audit);
#if defined(__clang__)
    [[clang::musttail]]
#endif
    return covariance_f_fallback_products(x, output, work, audit);
}
EIGEN_DONT_INLINE bool covariance_f(const Value &x, double *output, Workspace &work,
                                   Audit &audit) {
    if (x.stride[0] != 1) return false;
    const auto n = static_cast<Eigen::Index>(x.shape.dim[0]);
    const auto d = static_cast<Eigen::Index>(x.shape.dim[1]);
    const auto *input = static_cast<const double *>(x.data);
    using Row = Eigen::Array<double, 1, Eigen::Dynamic>;
    // Preserve original overflow and nonfinite admission before any write.
    const double limit = std::sqrt(std::numeric_limits<double>::max() /
                                  (8.0 * static_cast<double>(n)));
    bool fused = d >= 8 && x.stride[1] >= n;
    for (Eigen::Index j = 0; j < d; ++j) {
        Eigen::Map<const Row> column(input + j * x.stride[1], n);
        if (column.abs().maxCoeff() > limit) return false;
        if (fused && !column.isFinite().all()) fused = false;
    }
    if (fused) {
#if defined(__clang__)
        [[clang::musttail]]
#endif
        return covariance_f_detail::fast(x, output, work, audit);
    }
    // Small, nonfinite and special-stride columns retain the full-N route.
    // Separate mean/product frames prevent the fallback growing with fast code.
#if defined(__clang__)
    [[clang::musttail]]
#endif
    return covariance_f_fallback(x, output, work, audit);
}

template<int Rows, int Cols, int BColumnStride, bool FOutput = false>
EIGEN_ALWAYS_INLINE void column_contiguous_matmul(const double *a, std::ptrdiff_t ar, std::ptrdiff_t ac,
    const double *b, std::ptrdiff_t br, std::ptrdiff_t bc, double *__restrict out,
    std::size_t m, std::size_t inner, std::size_t n) {
    using Tile = Eigen::Array<double, Rows, Cols, Eigen::RowMajor>;
    using Row = Eigen::Array<double, 1, Cols, Eigen::RowMajor>;
    using Column = Eigen::Array<double, Rows, 1>;
    using ReadRow = Eigen::Map<const Row, Eigen::Unaligned,
                               Eigen::InnerStride<BColumnStride>>;
    const auto full_rows = m - m % Rows, full_cols = n - n % Cols;
    for (std::size_t i = 0; i < full_rows; i += Rows) {
        for (std::size_t j = 0; j < full_cols; j += Cols) {
            // Four independent rows/columns retain ascending k order. The fixed
            // accumulator is register-sized; compiled spills remain auditable.
            Tile accumulator = Tile::Zero();
            for (std::size_t k = 0; k < inner; ++k) {
                const Eigen::Map<const Column, Eigen::Unaligned,
                                 Eigen::InnerStride<Eigen::Dynamic>> left(
                    a + static_cast<std::ptrdiff_t>(i) * ar + static_cast<std::ptrdiff_t>(k) * ac,
                    Rows, Eigen::InnerStride<Eigen::Dynamic>(ar));
                const ReadRow right(b + static_cast<std::ptrdiff_t>(k) * br +
                                      static_cast<std::ptrdiff_t>(j) * bc,
                                    Cols, Eigen::InnerStride<BColumnStride>(bc));
                accumulator += left.template replicate<1, Cols>() *
                               right.template replicate<Rows, 1>();
            }
            if constexpr (FOutput) {
                // RowMajor logical tile, borrowed column-major destination.
                Eigen::Map<Tile, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>(
                    out + j * m + i,
                    Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, static_cast<Eigen::Index>(m))) = accumulator;
            } else {
                Eigen::Map<Tile, Eigen::Unaligned, Eigen::OuterStride<>>(
                    out + i * n + j, Eigen::OuterStride<>(static_cast<Eigen::Index>(n))) = accumulator;
            }
        }
        for (auto row = i; row < i + Rows; ++row)
            for (auto col = full_cols; col < n; ++col) {
                double total = 0.0;
                for (std::size_t k = 0; k < inner; ++k)
                    total += a[static_cast<std::ptrdiff_t>(row) * ar +
                               static_cast<std::ptrdiff_t>(k) * ac] *
                             b[static_cast<std::ptrdiff_t>(k) * br +
                               static_cast<std::ptrdiff_t>(col) * bc];
                out[FOutput ? col * m + row : row * n + col] = total;
            }
    }
    for (auto row = full_rows; row < m; ++row)
        for (std::size_t col = 0; col < n; ++col) {
            double total = 0.0;
            for (std::size_t k = 0; k < inner; ++k)
                total += a[static_cast<std::ptrdiff_t>(row) * ar +
                           static_cast<std::ptrdiff_t>(k) * ac] *
                         b[static_cast<std::ptrdiff_t>(k) * br +
                           static_cast<std::ptrdiff_t>(col) * bc];
            out[FOutput ? col * m + row : row * n + col] = total;
        }
}

template<int Cols, int BColumnStride>
EIGEN_ALWAYS_INLINE void row_contiguous_matmul(const double *a, std::ptrdiff_t ar, std::ptrdiff_t ac,
    const double *b, std::ptrdiff_t br, std::ptrdiff_t bc, double *__restrict out,
    std::size_t m, std::size_t inner, std::size_t n) {
    using Row = Eigen::Array<double, 1, Cols, Eigen::RowMajor>;
    using ReadRow = Eigen::Map<const Row, Eigen::Unaligned, Eigen::InnerStride<BColumnStride>>;
    const auto full_rows = m - m % 4, full_cols = n - n % Cols;
    for (std::size_t i = 0; i < full_rows; i += 4) {
        for (std::size_t j = 0; j < full_cols; j += Cols) {
            // Separate lazy Array expressions keep the wider tile in registers.
            // SIMD lanes are independent columns; k remains an ascending fold.
            Row r0 = Row::Zero();
            Row r1 = Row::Zero();
            Row r2 = Row::Zero();
            Row r3 = Row::Zero();
            for (std::size_t k = 0; k < inner; ++k) {
                const ReadRow right(b + static_cast<std::ptrdiff_t>(k) * br +
                                      static_cast<std::ptrdiff_t>(j) * bc,
                                    Cols, Eigen::InnerStride<BColumnStride>(bc));
                const auto *left = a + static_cast<std::ptrdiff_t>(i) * ar +
                                      static_cast<std::ptrdiff_t>(k) * ac;
                r0 += left[0 * ar] * right;
                r1 += left[1 * ar] * right;
                r2 += left[2 * ar] * right;
                r3 += left[3 * ar] * right;
            }
            Eigen::Map<Row, Eigen::Unaligned>(out + (i + 0) * n + j) = r0;
            Eigen::Map<Row, Eigen::Unaligned>(out + (i + 1) * n + j) = r1;
            Eigen::Map<Row, Eigen::Unaligned>(out + (i + 2) * n + j) = r2;
            Eigen::Map<Row, Eigen::Unaligned>(out + (i + 3) * n + j) = r3;
        }
        for (auto row = i; row < i + 4; ++row)
            for (auto col = full_cols; col < n; ++col) {
                double total = 0.0;
                for (std::size_t k = 0; k < inner; ++k)
                    total += a[static_cast<std::ptrdiff_t>(row) * ar +
                               static_cast<std::ptrdiff_t>(k) * ac] *
                             b[static_cast<std::ptrdiff_t>(k) * br +
                               static_cast<std::ptrdiff_t>(col) * bc];
                out[row * n + col] = total;
            }
    }
    for (auto row = full_rows; row < m; ++row)
        for (std::size_t col = 0; col < n; ++col) {
            double total = 0.0;
            for (std::size_t k = 0; k < inner; ++k)
                total += a[static_cast<std::ptrdiff_t>(row) * ar +
                           static_cast<std::ptrdiff_t>(k) * ac] *
                         b[static_cast<std::ptrdiff_t>(k) * br +
                           static_cast<std::ptrdiff_t>(col) * bc];
            out[row * n + col] = total;
        }
}


// Each output retains one ascending-k binary64 fold across panels. Only the
// independent output stores panel state; there is no pack or numeric workspace.
template<bool FOutput, bool BContiguous>
EIGEN_ALWAYS_INLINE void matmul_c_blocked_tile(const double *a, std::ptrdiff_t ar,
    std::ptrdiff_t ac, const double *b, std::ptrdiff_t br, double *output,
    std::size_t leading, std::ptrdiff_t bc, std::size_t count, bool first) {
    using Row = Eigen::Array<double, 1, 8, Eigen::RowMajor>;
    using Read = Eigen::Map<const Row, Eigen::Unaligned>;
    using Write = Eigen::Map<Row, Eigen::Unaligned>;
    using StridedRead = Eigen::Map<const Row, Eigen::Unaligned,
                                  Eigen::InnerStride<Eigen::Dynamic>>;
    using StridedWrite = Eigen::Map<Row, Eigen::Unaligned,
                                   Eigen::InnerStride<Eigen::Dynamic>>;
    const auto row_stride = FOutput ? std::size_t(1) : leading;
    Row r0, r1, r2, r3;
    if (first) {
        r0.setZero(); r1.setZero(); r2.setZero(); r3.setZero();
    } else {
        if constexpr (FOutput) {
            const Eigen::InnerStride<Eigen::Dynamic> stride{std::ptrdiff_t(leading)};
            r0 = StridedRead(output, 1, 8, stride);
            r1 = StridedRead(output + row_stride, 1, 8, stride);
            r2 = StridedRead(output + 2 * row_stride, 1, 8, stride);
            r3 = StridedRead(output + 3 * row_stride, 1, 8, stride);
        } else {
            r0 = Read(output);
            r1 = Read(output + leading);
            r2 = Read(output + 2 * leading);
            r3 = Read(output + 3 * leading);
        }
    }
    for (std::size_t k = 0; k < count; ++k) {
        const auto right = [&] {
            const auto *ptr = b + static_cast<std::ptrdiff_t>(k) * br;
            if constexpr (BContiguous) return Read(ptr);
            else return Row(StridedRead(ptr, 1, 8, Eigen::InnerStride<Eigen::Dynamic>{bc}));
        }();
        const auto *left = a + static_cast<std::ptrdiff_t>(k) * ac;
        r0 += left[0 * ar] * right;
        r1 += left[1 * ar] * right;
        r2 += left[2 * ar] * right;
        r3 += left[3 * ar] * right;
    }
    if constexpr (FOutput) {
        const Eigen::InnerStride<Eigen::Dynamic> stride{std::ptrdiff_t(leading)};
        StridedWrite{output, 1, 8, stride} = r0;
        StridedWrite{output + row_stride, 1, 8, stride} = r1;
        StridedWrite{output + 2 * row_stride, 1, 8, stride} = r2;
        StridedWrite{output + 3 * row_stride, 1, 8, stride} = r3;
    } else {
        Write{output} = r0;
        Write{output + leading} = r1;
        Write{output + 2 * leading} = r2;
        Write{output + 3 * leading} = r3;
    }
}

EIGEN_ALWAYS_INLINE double matmul_c_blocked_cell(const double *a,
    std::ptrdiff_t ar, std::ptrdiff_t ac, const double *b, std::ptrdiff_t br,
    std::size_t inner, std::size_t row, std::size_t col) {
    double total = 0.0;
    for (std::size_t k = 0; k < inner; ++k)
        total += a[static_cast<std::ptrdiff_t>(row) * ar +
                   static_cast<std::ptrdiff_t>(k) * ac] *
                 b[static_cast<std::ptrdiff_t>(k) * br + static_cast<std::ptrdiff_t>(col)];
    return total;
}

// Keep the same four-reference signature as the bulk/public entry, so the
// completed edge call cannot keep a driver frame alive around the bulk call.
template<bool FOutput, bool BContiguous>
EIGEN_DONT_INLINE void matmul_c_blocked_edges(const Value &a, const Value &b,
                                             double *__restrict output, Audit &audit) {
    (void)audit;
    const auto rows = a.shape.dim[0], inner = a.shape.dim[1], cols = b.shape.dim[1];
    const auto full_rows = rows - rows % 4, full_cols = cols - cols % 8;
    const auto *left = static_cast<const double *>(a.data);
    const auto *right = static_cast<const double *>(b.data);
    const auto store = [&](std::size_t row, std::size_t col) {
        double value;
        if constexpr (BContiguous)
            value = matmul_c_blocked_cell(left, a.stride[0], a.stride[1],
                                         right, b.stride[0], inner, row, col);
        else value = dot_value(a.row(row), b.column(col));
        output[FOutput ? col * rows + row : row * cols + col] = value;
    };
    for (std::size_t row = 0; row < full_rows; ++row)
        for (auto col = full_cols; col < cols; ++col) store(row, col);
    for (auto row = full_rows; row < rows; ++row)
        for (std::size_t col = 0; col < cols; ++col) store(row, col);
}

template<bool FOutput, bool BContiguous>
EIGEN_DONT_INLINE void matmul_c_blocked_bulk(const Value &a, const Value &b,
                                            double *__restrict output, Audit &audit) {
    constexpr std::size_t block_rows = 32, block_cols = 64, block_inner = 64;
    const auto rows = a.shape.dim[0], inner = a.shape.dim[1], cols = b.shape.dim[1];
    const auto full_rows = rows - rows % 4, full_cols = cols - cols % 8;
    const auto *left = static_cast<const double *>(a.data);
    const auto *right = static_cast<const double *>(b.data);
    const auto ar = a.stride[0], ac = a.stride[1], br = b.stride[0];
    const auto bc = BContiguous ? std::ptrdiff_t(1) : b.stride[1];
    for (std::size_t ib = 0; ib < full_rows; ib += block_rows) {
        const auto row_count = std::min(block_rows, full_rows - ib);
        for (std::size_t jb = 0; jb < full_cols; jb += block_cols) {
            const auto col_count = std::min(block_cols, full_cols - jb);
            for (std::size_t kb = 0; kb < inner; kb += block_inner) {
                const auto count = std::min(block_inner, inner - kb);
                for (std::size_t i = 0; i < row_count; i += 4)
                    for (std::size_t j = 0; j < col_count; j += 8)
                        matmul_c_blocked_tile<FOutput,BContiguous>(
                            left + static_cast<std::ptrdiff_t>(ib + i) * ar +
                                   static_cast<std::ptrdiff_t>(kb) * ac,
                            ar, ac,
                            right + static_cast<std::ptrdiff_t>(kb) * br +
                                    static_cast<std::ptrdiff_t>(jb + j) * bc,
                            br, output + (FOutput ? (jb + j) * rows + ib + i
                                                  : (ib + i) * cols + jb + j),
                            FOutput ? rows : cols, bc, count, kb == 0);
            }
        }
    }
    audit.isa = FOutput ? "eigen-block-matmul-f-root" : "eigen-block-matmul";
    if constexpr (FOutput) audit.vector_elements = rows * cols;
}

template <int Rows>
std::size_t row_contiguous_matvec(const double *__restrict matrix, std::ptrdiff_t row_stride,
                                 const double *__restrict vector, std::ptrdiff_t vector_stride,
                                 double *__restrict output, std::size_t rows, std::size_t cols) {
    using Tile = Eigen::Array<double, Rows, 1>;
    using Column = Eigen::Map<const Tile, Eigen::Unaligned,
                              Eigen::InnerStride<Eigen::Dynamic>>;
    constexpr auto tile_rows = static_cast<std::size_t>(Rows);
    std::size_t row = 0;
    for (; rows - row >= tile_rows; row += tile_rows) {
        const auto *base = matrix + static_cast<std::ptrdiff_t>(row) * row_stride;
        // Both maps borrow existing storage. Independent rows expose parallel
        // arithmetic while each result keeps its original column-wise fold.
        Eigen::Map<Tile, Eigen::Unaligned> target(output + row);
        target.setZero();
        for (std::size_t j = 0; j < cols; ++j) {
            const Column column(base + j, Rows, Eigen::InnerStride<Eigen::Dynamic>(row_stride));
            const double coefficient = vector[static_cast<std::ptrdiff_t>(j) * vector_stride];
            target += column * coefficient;
        }
    }
    return row;
}
} // namespace
#endif

#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
void covariance_library_means(const Value &input, double *means) {
    if (input.stride[1] == 1)
        covariance_c_means_only(input, means);
    else if (input.stride[0] != 1)
        covariance_c_means_only<Eigen::Dynamic>(input, means);
    else
        covariance_f_detail::gather_means<8>(static_cast<const double *>(input.data),
            input.shape.dim[0], input.shape.dim[1], input.stride[1], means);
}
void covariance_library_centered(const Value &input, Output &output, Workspace &work, Audit &audit) {
    // Caller admits a complete centered F panel, N >= 32 and even D. Reuse
    // the canonical fixed reduction tree without reading unpublished means.
    auto panel = input;
    panel.data = work.doubles.data();
    panel.stride = {1, static_cast<std::ptrdiff_t>(input.shape.dim[0]), 1};
    covariance_f_detail::upper<true>(panel, static_cast<double *>(output.data), work, audit);
    finish_covariance_matrix(output, false);
    audit.eigen_covariance_calls = 1;
    audit.isa = "eigen-centered-covariance";
}
double covariance_library_pair(const Value &input, std::size_t j, std::size_t k,
                               const double *means) {
    const auto *x = static_cast<const double *>(input.data);
    const auto n = input.shape.dim[0];
    if (input.stride[1] == 1)
        return covariance_ordered_sum(x, n, j, k, input.stride[0], means);
    if (input.stride[0] != 1)
        return covariance_ordered_sum(x, n, j, k, input.stride[0], means, input.stride[1]);
    return covariance_f_detail::original_pair(x + j * input.stride[1],
        x + k * input.stride[1], n, means[j], means[k]);
}
void library_matmul_f(const Value &a, const Value &b, double *output, Audit &audit) {
    // Tiny row blocks still have long independent output lanes. Reuse the
    // same ordered Eigen expression, keeping A*B and ascending-k arithmetic.
    if (a.shape.dim[0] < 4) {
        column_contiguous_matmul<1, 8, Eigen::Dynamic>(
            static_cast<const double *>(a.data), a.stride[0], a.stride[1],
            static_cast<const double *>(b.data), b.stride[0], b.stride[1], output,
            a.shape.dim[0], a.shape.dim[1], b.shape.dim[1]);
        audit.isa = "eigen-array-matmul";
        return;
    }
    // Caller has admitted the F-compatible right operand and the target platform.
    // Keeping the whole kernel in this void entry permits a tail call from matrix.
    column_contiguous_matmul<4, 4, Eigen::Dynamic>(
        static_cast<const double *>(a.data), a.stride[0], a.stride[1],
        static_cast<const double *>(b.data), b.stride[0], b.stride[1], output,
        a.shape.dim[0], a.shape.dim[1], b.shape.dim[1]);
    audit.isa = "eigen-array-matmul";
}
void library_matmul_f_root(const Value &a, const Value &b, double *output, Audit &audit) {
    column_contiguous_matmul<4, 4, Eigen::Dynamic, true>(
        static_cast<const double *>(a.data), a.stride[0], a.stride[1],
        static_cast<const double *>(b.data), b.stride[0], b.stride[1], output,
        a.shape.dim[0], a.shape.dim[1], b.shape.dim[1]);
    audit.isa = "eigen-array-matmul-f-root";
    audit.vector_elements = a.shape.dim[0] * b.shape.dim[1];
}
void library_matmul_c(const Value &a, const Value &b, double *output, Audit &audit) {
    row_contiguous_matmul<8, 1>(
        static_cast<const double *>(a.data), a.stride[0], a.stride[1],
        static_cast<const double *>(b.data), b.stride[0], b.stride[1], output,
        a.shape.dim[0], a.shape.dim[1], b.shape.dim[1]);
    audit.isa = "eigen-array-matmul";
}

EIGEN_DONT_INLINE void library_matmul_c_blocked(const Value &a, const Value &b,
                                                double *output, Audit &audit) {
    if (a.shape.dim[0] % 4 || b.shape.dim[1] % 8)
        matmul_c_blocked_edges<false,true>(a, b, output, audit);
#if defined(__clang__)
    [[clang::musttail]]
#endif
    return matmul_c_blocked_bulk<false,true>(a, b, output, audit);
}

EIGEN_DONT_INLINE void library_matmul_f_blocked(const Value &a, const Value &b,
                                                double *output, Audit &audit) {
    // Existing ordered 4x8 Eigen tile; only borrowed input/output strides vary.
    if (b.stride[1] == 1) {
        if (a.shape.dim[0] % 4 || b.shape.dim[1] % 8)
            matmul_c_blocked_edges<true,true>(a, b, output, audit);
#if defined(__clang__)
        [[clang::musttail]]
#endif
        return matmul_c_blocked_bulk<true,true>(a, b, output, audit);
    }
    if (a.shape.dim[0] % 4 || b.shape.dim[1] % 8)
        matmul_c_blocked_edges<true,false>(a, b, output, audit);
#if defined(__clang__)
    [[clang::musttail]]
#endif
    return matmul_c_blocked_bulk<true,false>(a, b, output, audit);
}

void library_matmul_repeated_rows(const Value &a, const Value &b, double *output, Audit &audit) {
    // Zero row stride is an exact dependency identity, not approximate equality.
    // Compute from current inputs on every call; retain independent result rows.
    const auto rows = a.shape.dim[0], inner = a.shape.dim[1], cols = b.shape.dim[1];
    const auto *left = static_cast<const double *>(a.data);
    const auto *right = static_cast<const double *>(b.data);
    for (std::size_t col = 0; col < cols; ++col) {
        double total = 0.0;
        for (std::size_t k = 0; k < inner; ++k)
            total += left[static_cast<std::ptrdiff_t>(k) * a.stride[1]] *
                     right[static_cast<std::ptrdiff_t>(k) * b.stride[0] +
                           static_cast<std::ptrdiff_t>(col) * b.stride[1]];
        output[col] = total;
    }
    for (std::size_t row = 1; row < rows; ++row)
        std::copy_n(output, cols, output + row * cols);
    audit.algorithm_copy_bytes += (rows - 1) * cols * sizeof(double);
    audit.isa = "matmul-repeated-rows";
}
#endif

namespace {
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
template<bool Square>
bool masked_sum_quad(const Shape &shape, const Value &mask, const Value &left,
    const Value &right, double otherwise, std::size_t row, double *totals,
    void *context, void (*checkpoint)(void *)) {
    if (__builtin_available(macOS 15.0, *)) {
        const auto compatible = [&](const Value &v, Kind kind) {
            return v.kind == kind && (!v.shape.rank || (v.shape == shape && v.data));
        };
        if (shape.rank != 2 || row > shape.dim[0] || shape.dim[0] - row < 4 ||
            shape.dim[1] < 128 || !totals || !compatible(mask, Kind::mask) ||
            !compatible(left, Kind::number) || !compatible(right, Kind::number))
            return false;
        using namespace Eigen::internal;
        using Packet = Packet2d;
        const auto zero = pset1<Packet>(0.0), one = pset1<Packet>(1.0);
        const auto fallback = pset1<Packet>(otherwise);
        Packet total[2]{zero, zero};
        const auto count = shape.dim[1];
        const auto *a = left.shape.rank
            ? static_cast<const double *>(left.data) + left.offset(row * count) : &left.scalar;
        const auto *b = right.shape.rank
            ? static_cast<const double *>(right.data) + right.offset(row * count) : &right.scalar;
        const auto scalar_mask = std::uint8_t(mask.scalar);
        const auto *condition = mask.shape.rank
            ? static_cast<const std::uint8_t *>(mask.data) + mask.offset(row * count) : &scalar_mask;
        const auto ar = left.shape.rank ? left.stride[0] : 0;
        const auto ac = left.shape.rank ? left.stride[1] : 0;
        const auto br = right.shape.rank ? right.stride[0] : 0;
        const auto bc = right.shape.rank ? right.stride[1] : 0;
        const auto mr = mask.shape.rank ? mask.stride[0] : 0;
        const auto mc = mask.shape.rank ? mask.stride[1] : 0;
        for (std::size_t j = 0; j < count; ++j) {
            if ((j & 1023) == 0 && checkpoint) checkpoint(context);
            for (int half = 0; half < 2; ++half) {
                const auto i = std::ptrdiff_t(2 * half);
                const auto *m = condition + i * mr + std::ptrdiff_t(j) * mc;
                const double flags[2]{double(m[0] != 0), double(m[mr] != 0)};
                const auto selected = pcmp_eq(ploadu<Packet>(flags), one);
                const auto av = pgather<double, Packet>(a + i * ar + std::ptrdiff_t(j) * ac, ar);
                const auto bv = pgather<double, Packet>(b + i * br + std::ptrdiff_t(j) * bc, br);
                Packet value;
                if constexpr (Square) {
                    const auto delta = psub(av, bv);
                    value = pselect(selected, pmul(delta, delta), fallback);
                } else {
                    value = pselect(selected, av, bv);
                }
                // No horizontal reduction: every lane keeps ascending-j order.
                total[half] = padd(total[half], value);
            }
        }
        pstoreu<double>(totals + row, total[0]);
        pstoreu<double>(totals + row + 2, total[1]);
        return true;
    }
    return false;
}
#endif
}

bool library_where_sum_quad(const Prepared &where, std::size_t row, double *totals,
    void *context, void (*checkpoint)(void *)) {
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    if (where.count == 3)
        return masked_sum_quad<false>(where.output_shape, where.args[0], where.args[1],
            where.args[2], 0.0, row, totals, context, checkpoint);
#else
    (void)where; (void)row; (void)totals; (void)context; (void)checkpoint;
#endif
    return false;
}

bool library_square_sum_quad(const Prepared &subtraction, const Value &mask, double otherwise,
    std::size_t row, double *totals, void *context, void (*checkpoint)(void *)) {
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    if (subtraction.count == 2)
        return masked_sum_quad<true>(subtraction.output_shape, mask, subtraction.args[0],
            subtraction.args[1], otherwise, row, totals, context, checkpoint);
#else
    (void)subtraction; (void)mask; (void)otherwise; (void)row; (void)totals;
    (void)context; (void)checkpoint;
#endif
    return false;
}

bool library_scan_reduce_quad(const Value &input, std::size_t row, double initial,
    double peak_initial, double subtraction, double *totals, unsigned &errors,
    void *context, void (*checkpoint)(void *)) {
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    if (__builtin_available(macOS 15.0, *)) {
        if (input.shape.rank != 2 || row > input.shape.dim[0] ||
            input.shape.dim[0] - row < 4 || input.shape.dim[1] < 128)
            return false;
        using namespace Eigen::internal;
        using Packet = Packet2d;
        const auto zero = pset1<Packet>(0.0), one = pset1<Packet>(1.0);
        const auto exponent = pset1<Packet>(std::numeric_limits<double>::infinity());
        const auto nan = pset1<Packet>(std::numeric_limits<double>::quiet_NaN());
        const auto constant = pset1<Packet>(subtraction);
        Packet state[2]{pset1<Packet>(initial), pset1<Packet>(initial)};
        Packet peak[2]{pset1<Packet>(peak_initial), pset1<Packet>(peak_initial)};
        Packet maximum[2]{pset1<Packet>(-std::numeric_limits<double>::infinity()),
                          pset1<Packet>(-std::numeric_limits<double>::infinity())};
        Packet zero_peak[2]{zero, zero}, nonfinite[2]{zero, zero};
        const auto *source = static_cast<const double *>(input.data) +
                            input.offset(row * input.shape.dim[1]);
        for (std::size_t j = 0; j < input.shape.dim[1]; ++j) {
            if ((j & 1023) == 0) checkpoint(context);
            for (int half = 0; half < 2; ++half) {
                const auto current = pgather<double, Packet>(source +
                    std::ptrdiff_t(2 * half) * input.stride[0] +
                    std::ptrdiff_t(j) * input.stride[1], input.stride[0]);
                state[half] = pmul(state[half], current);
                peak[half] = pselect(pcmp_lt(peak[half], state[half]), state[half], peak[half]);
                const auto missing = pcmp_eq(peak[half], zero);
                zero_peak[half] = por(zero_peak[half], missing);
                // The canonical zero-denominator branch does no division.
                // Mask both operands before packet division, then select NaN.
                auto numerator = pselect(missing, zero, state[half]);
                auto denominator = pselect(missing, one, peak[half]);
                // Register-only form of Eigen's optimization barrier for
                // Packet2d. Generic +g,w selects stack operands in this loop.
                // This emits no arithmetic or memory instruction; pdiv remains
                // the library operation and cannot discard the safe operands.
                __asm__("" : "+w"(numerator), "+w"(denominator));
                const auto ratio = pselect(missing, nan, pdiv(numerator, denominator));
                const auto value = psub(constant, ratio);
                // Bitwise exponent classification never compares a NaN to a
                // finite limit or performs arithmetic on an all-ones mask.
                nonfinite[half] = por(nonfinite[half], pcmp_eq(pand(value, exponent), exponent));
                maximum[half] = pselect(pcmp_lt(maximum[half], value), value, maximum[half]);
            }
        }
        for (int half = 0; half < 2; ++half) {
            pstoreu<double>(totals + row + 2 * half, maximum[half]);
            if (predux_any(pselect(zero_peak[half], one, zero))) errors |= 1;
            if (predux_any(pselect(nonfinite[half], one, zero))) errors |= 2;
        }
        return true;
    }
#else
    (void)input; (void)row; (void)initial; (void)peak_initial; (void)subtraction;
    (void)totals; (void)errors; (void)context; (void)checkpoint;
#endif
    return false;
}

bool library_matvec(const Value &a, const Value &b, double *output, Audit &audit) {
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    if (__builtin_available(macOS 15.0, *)) {
        const auto rows = a.shape.dim[0], cols = a.shape.dim[1];
        if (rows < 128 || cols < 8) return false;
        const auto *matrix = static_cast<const double *>(a.data);
        const auto *vector = static_cast<const double *>(b.data);
        if (a.stride[1] == 1 && a.stride[0] >= static_cast<std::ptrdiff_t>(cols)) {
            const auto completed = row_contiguous_matvec<8>(
                matrix, a.stride[0], vector, b.stride[0], output, rows, cols);
            if (completed < rows)
                row_contiguous_matvec<1>(
                    matrix + static_cast<std::ptrdiff_t>(completed) * a.stride[0],
                    a.stride[0], vector, b.stride[0], output + completed,
                    rows - completed, cols);
        } else {
            if (a.stride[0] != 1 || a.stride[1] < static_cast<std::ptrdiff_t>(rows)) return false;
            using Array = Eigen::Array<double, Eigen::Dynamic, 1>;
            // SIMD lanes are independent output rows. Each row keeps the original
            // +0, multiply, add sequence over columns, including zero coefficients.
            // Cache-sized output blocks need no scratch, packing or BLAS thread state.
            constexpr std::size_t row_block = 4096;
            for (std::size_t base = 0; base < rows; base += row_block) {
                const auto count = static_cast<Eigen::Index>(std::min(row_block, rows - base));
                Eigen::Map<Array> target(output + base, count);
                target.setZero();
                for (std::size_t j = 0; j < cols; ++j) {
                    Eigen::Map<const Array> column(
                        matrix + base + static_cast<std::ptrdiff_t>(j) * a.stride[1], count);
                    const double coefficient = vector[static_cast<std::ptrdiff_t>(j) * b.stride[0]];
                    target += column * coefficient;
                }
            }
        }
        ++audit.eigen_matvec_calls;
        return true;
    }
#else
    (void)a; (void)b; (void)output; (void)audit;
#endif
    return false;
}

bool library_covariance_upper(const Value &x, double *output, Workspace &work,
                              Audit &audit) {
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    if (__builtin_available(macOS 15.0, *)) {
        const auto n = x.shape.dim[0], d = x.shape.dim[1];
        if (n < 32 || d == 0) return false;
        // Admission and finished setup frames must not surround numeric kernels.
        if (x.stride[1] == 1 && d >= 8) {
#if defined(__clang__)
            [[clang::musttail]]
#endif
            return covariance_c(x, output, work, audit);
        }
        // Descending F has the same scalar observation fold as C. Borrow its
        // strided columns in the existing ordered tiles; no matrix packing.
        if (x.stride[0] == -1 && x.stride[1] >= static_cast<std::ptrdiff_t>(n) && d >= 8) {
#if defined(__clang__)
            [[clang::musttail]]
#endif
            return covariance_c<Eigen::Dynamic>(x, output, work, audit);
        }
#if defined(__clang__)
        [[clang::musttail]]
#endif
        return covariance_f(x, output, work, audit);
    }
    return false;
#else
    (void)x; (void)output; (void)work; (void)audit;
    return false;
#endif
}

void triangular_solve_finish(const Prepared &p, Output &out, Workspace &work,
                             Isa isa, Audit &audit) {
    (void)work;
    const auto &a=p.args[0], &b=p.args[1];
    const auto n=a.shape.dim[0], columns=b.shape.rank==1 ? 1 : b.shape.dim[1];
    auto *y=static_cast<double *>(out.data);
    const bool lower=p.args[2].scalar==1, unit=p.args[3].scalar==1;
    bool library=false;
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    if (__builtin_available(macOS 15.0, *))
        library=isa!=Isa::scalar && n>=32 && columns>=8 &&
            (a.stride[1]==1 || columns<=32);
#else
    (void)isa; (void)audit;
#endif
    for (std::size_t q=0;q<n;++q) {
        const auto i=lower ? q : n-1-q;
        const auto diagonal=unit ? 1.0 : a.at(i,i);
        require(std::isfinite(diagonal),"NON_FINITE_RESULT");
        require(std::abs(diagonal)>1e-14,"SINGULAR_MATRIX");
        const auto first=lower ? 0 : i+1, last=lower ? i : n;
        bool updated=false;
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
        if (library) {
            bool valid=true;
            for (auto k=first;k<last;++k) valid &= std::isfinite(a.at(i,k));
            if (valid) {
                std::size_t j=0;
                for (;columns-j>=16;j+=16) triangular_rhs_tile<16>(a,y,i,columns,first,last,j);
                if (columns-j>=8) { triangular_rhs_tile<8>(a,y,i,columns,first,last,j); j+=8; }
                if (columns-j>=4) { triangular_rhs_tile<4>(a,y,i,columns,first,last,j); j+=4; }
                if (columns-j>=2) { triangular_rhs_tile<2>(a,y,i,columns,first,last,j); j+=2; }
                if (j<columns) triangular_rhs_tile<1>(a,y,i,columns,first,last,j);
                updated=true;
            }
        }
#endif
        // Invalid coefficients use the canonical order and expose the same
        // partial explicit-out writes before throwing. Never consume the other
        // triangle or a unit diagonal, and never skip multiplication by zero.
        if (!updated) for (auto k=first;k<last;++k) {
            const auto coefficient=a.at(i,k);
            require(std::isfinite(coefficient),"NON_FINITE_RESULT");
            for (std::size_t j=0;j<columns;++j) y[i*columns+j]-=coefficient*y[k*columns+j];
        }
        // Division/checks stay interleaved in column order, including failure.
        for (std::size_t j=0;j<columns;++j) {
            y[i*columns+j]/=diagonal;
            require(std::isfinite(y[i*columns+j]),"NON_FINITE_RESULT");
        }
    }
    if (library) audit.isa="eigen-array-triangular";
}

void MatrixRotation::apply(double *x, double *y, std::size_t count,
                           double c, double s, Audit &audit) const {
#ifdef CALMETRICS_HAVE_EIGEN_ROTATION
    if (active_ && count) {
        // Both disjoint row spans belong to the same validated allocation.
        // Map borrows the spans: it neither packs nor allocates a matrix.
        using Rows = Eigen::Matrix<double,2,Eigen::Dynamic,Eigen::RowMajor>;
        Eigen::Map<Rows,Eigen::Unaligned,Eigen::OuterStride<>> rows(
            x,2,static_cast<Eigen::Index>(count),Eigen::OuterStride<>(y-x));
        rows.applyOnTheLeft(0,1,Eigen::JacobiRotation<double>(c,-s));
        ++audit.eigen_rotation_calls;
        return;
    }
#else
    (void)audit;
#endif
    for (std::size_t k = 0; k < count; ++k) {
        const auto u = x[k], v = y[k];
        x[k] = c*u-s*v;
        y[k] = s*u+c*v;
    }
}
} // namespace calmetrics_engine::ops
