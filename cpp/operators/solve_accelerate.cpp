#include "calmetrics_engine/operators.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>
#include <vector>

#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

namespace calmetrics_engine::ops {
std::size_t solve_accelerate_scratch_doubles(std::size_t n, std::size_t k) {
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL)
    if (__builtin_available(macOS 15.0, *)) {
        if (n >= 1024 && n <= 4096 && k >= 1 && k <= 64)
            return n * k + 5 * n;
    }
#endif
    (void)n; (void)k;
    return 0;
}

std::size_t solve_accelerate_copy_reserve_doubles(std::size_t n, std::size_t k) {
    const auto scratch = solve_accelerate_scratch_doubles(n, k);
    // Explicit N² coefficients and scratch; vendor/TLS and ABI vectors are
    // conservative estimates, not additional eagerly allocated buffers.
    return scratch ? n * n + scratch + 514 * n : 0;
}

namespace {
int dense_order(const Value &v) {
    if (v.kind != Kind::number || v.shape.rank != 2 || v.shape.dim[0] != v.shape.dim[1]) return -1;
    const auto n = v.shape.dim[0];
    return v.stride[1] == 1 && v.stride[0] == std::ptrdiff_t(n) ? 0 :
        v.stride[0] == 1 && v.stride[1] == std::ptrdiff_t(n) ? 1 : -1;
}

int producer_order(const Prepared &p) {
    if (p.output_shape.rank != 2 || p.output_shape.dim[0] != p.output_shape.dim[1] ||
        p.count != 2 || (p.spec->op != Op::add && p.spec->op != Op::subtract &&
                         p.spec->op != Op::multiply)) return -1;
    int order = -1;
    for (std::size_t i = 0; i < p.count; ++i) {
        const auto &v = p.args[i];
        if (v.kind != Kind::number) return -1;
        if (!v.shape.rank) continue;
        if (!(v.shape == p.output_shape)) return -1;
        const auto n = v.shape.dim[0];
        const int current = v.stride[1] == 1 && v.stride[0] == std::ptrdiff_t(n) ? 0 :
            v.stride[0] == 1 && v.stride[1] == std::ptrdiff_t(n) ? 1 : -1;
        if (current < 0 || (order >= 0 && current != order)) return -1;
        order = current;
    }
    return order;
}

// Flatten only an already validated, full dense elementwise producer. Its logical
// axes and shape stay intact; each element still uses the canonical operation.
void produce_span(const Prepared &p, std::size_t first, std::size_t count,
                  double *target, Workspace &work, Audit &audit) {
    Arguments<Value> args(p.count);
    for (std::size_t i = 0; i < p.count; ++i) {
        args[i] = p.args[i];
        if (args[i].shape.rank) {
            args[i].data = static_cast<const double *>(args[i].data) + first;
            args[i].shape = vector_shape(count);
            args[i].set_contiguous_strides();
        }
    }
    const auto prepared = prepare(*p.spec, args.data(), p.count);
    Output output; output.kind = Kind::number;
    output.shape = vector_shape(count); output.data = target;
    Audit child;
    execute(prepared, output, work, Isa::automatic, child);
    audit.vector_elements += child.vector_elements;
}

#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL)
struct ThreadLimit {
        BLAS_THREADING prior;
        bool active = false;
        ThreadLimit() : prior(BLASGetThreading()) {
            if (prior == BLAS_THREADING_MULTI_THREADED || prior == BLAS_THREADING_SINGLE_THREADED) {
                active = BLASSetThreading(BLAS_THREADING_SINGLE_THREADED) == 0 &&
                         BLASGetThreading() == BLAS_THREADING_SINGLE_THREADED;
                if (!active) BLASSetThreading(prior);
            }
        }
        ~ThreadLimit() { if (active) BLASSetThreading(prior); }
};

// Bounded-panel orchestration of the same Netlib DGETRF sequence. All
// factor, pivot, triangular and rank-k mathematics stay in mature libraries.
void factor_bounded_panels(__LAPACK_int n, double *a, __LAPACK_int *pivots,
                  __LAPACK_int block, __LAPACK_int &info) {
    info = 0;
    const __LAPACK_int increment = 1;
    for (__LAPACK_int first = 0; first < n; first += block) {
        const __LAPACK_int width = std::min(block, n - first);
        const __LAPACK_int height = n - first;
        __LAPACK_int local_info = 0;
        dgetrf2_(&height, &width, a + first * n + first, &n, pivots + first, &local_info);
        if (!info && local_info > 0) info = local_info + first;
        if (local_info < 0) { info = local_info; return; }
        for (__LAPACK_int i = first; i < first + width; ++i) pivots[i] += first;
        const __LAPACK_int start = first + 1, end = first + width;
        if (first) dlaswp_(&first, a, &n, &start, &end, pivots, &increment);
        const __LAPACK_int trailing = n - first - width;
        if (!trailing) continue;
        auto *upper = a + (first + width) * n + first;
        dlaswp_(&trailing, a + (first + width) * n, &n, &start, &end, pivots, &increment);
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                    int(width), int(trailing), 1., a + first * n + first, int(n), upper, int(n));
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    int(trailing), int(trailing), int(width), -1., a + first * n + first + width,
                    int(n), upper, int(n), 1., a + (first + width) * n + first + width, int(n));
    }
}

void transpose_square(double *m, std::size_t n) {
    constexpr std::size_t tile = 32;
    for (std::size_t first = 0; first < n; first += tile) {
        const auto last = std::min(n, first + tile);
        for (auto i = first; i < last; ++i)
            for (auto j = i + 1; j < last; ++j)
                std::swap(m[i * n + j], m[j * n + i]);
        for (auto next = last; next < n; next += tile) {
            const auto end = std::min(n, next + tile);
            for (auto i = first; i < last; ++i)
                for (auto j = next; j < end; ++j)
                    std::swap(m[i * n + j], m[j * n + i]);
        }
    }
}

double gamma(std::size_t n) {
    const double nu = double(n) * (std::numeric_limits<double>::epsilon() * .5);
    return nu / (1. - nu);
}

// A posteriori certificate for the original ordered partial-pivot path. Only
// identity row permutations enter this domain. Absolute triangular recurrences
// bound inverses of every leading L/U block in O(N²), in F traversal order.
bool canonical_lu_certificate(const double *factor, const __LAPACK_int *pivots,
                              std::size_t n, double maximum_a, double a_norm,
                              double *buffer) {
    auto *lower_inverse = buffer;
    auto *upper_inverse = lower_inverse + n;
    auto *lower_rows = upper_inverse + n;
    auto *upper_rows = lower_rows + n;
    auto *backward_rows = upper_rows + n;
    std::fill_n(lower_inverse, n, 1.);
    std::fill_n(upper_inverse, n, 1.);
    std::fill_n(lower_rows, n, 0.);
    std::fill_n(upper_rows, n, 0.);
    const auto inflate = 1. / (1. - gamma(4 * n));
    const auto upward = [&](double value) { return std::nextafter(value * inflate, INFINITY); };
    double pivot_gap = INFINITY, minimum_diagonal = INFINITY, maximum_upper = 0.;
    for (std::size_t j = 0; j < n; ++j) {
        if (pivots[j] != static_cast<__LAPACK_int>(j + 1)) return false;
        const auto diagonal = std::abs(factor[j * n + j]);
        minimum_diagonal = std::min(minimum_diagonal, diagonal);
        lower_inverse[j] = upward(lower_inverse[j]);
        double column_multiplier = 0.;
        for (std::size_t i = 0; i <= j; ++i) {
            const auto value = std::abs(factor[j * n + i]);
            upper_rows[i] += value;
            maximum_upper = std::max(maximum_upper, value);
        }
        for (std::size_t i = j + 1; i < n; ++i) {
            const auto value = std::abs(factor[j * n + i]);
            column_multiplier = std::max(column_multiplier, value);
            lower_inverse[i] += value * lower_inverse[j];
            lower_rows[i] += value;
        }
        // Exact factors encode the active column as L_ik*U_kk.
        pivot_gap = std::min(pivot_gap, diagonal * (1. - column_multiplier));
    }
    for (std::size_t reverse = 0; reverse < n; ++reverse) {
        const auto j = n - 1 - reverse;
        upper_inverse[j] = upward(upper_inverse[j] / std::abs(factor[j * n + j]));
        for (std::size_t i = 0; i < j; ++i)
            upper_inverse[i] += std::abs(factor[j * n + i]) * upper_inverse[j];
    }
    double l_inverse = 0., u_inverse = 0., l_rows = 0.;
    for (std::size_t i = 0; i < n; ++i) {
        l_inverse = std::max(l_inverse, lower_inverse[i]);
        u_inverse = std::max(u_inverse, upper_inverse[i]);
        l_rows = std::max(l_rows, upward(lower_rows[i]));
        upper_rows[i] = upward(upper_rows[i]);
        backward_rows[i] = upper_rows[i];
    }
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = j + 1; i < n; ++i)
            backward_rows[i] += std::abs(factor[j * n + i]) * upper_rows[j];
    double library_backward = 0.;
    for (std::size_t i = 0; i < n; ++i)
        library_backward = std::max(library_backward, upward(backward_rows[i]) * gamma(4 * n));
    // At a first crossing of 8*M, one canonical update is bounded by 16*M.
    // 32*M covers |L|<=1+u and partial-factor backward rounding. The strict
    // Schur perturbation checks close the induction before a crossing/pivot.
    const auto canonical_backward = 32. * gamma(4 * n) * double(n) * double(n) * maximum_a + 1e-270;
    const auto perturbation = upward(library_backward + canonical_backward);
    const auto inverse = upward(l_inverse * u_inverse);
    if (!std::isfinite(inverse) || !(inverse * perturbation < .25)) return false;
    const auto amplification = upward(1. + inverse * upward(a_norm + library_backward));
    const auto delta = upward(perturbation * amplification * amplification / (1. - inverse * perturbation));
    const auto schur_maximum = upward(maximum_upper * upward(1. + l_rows));
    if (!(std::isfinite(delta) && 2. * delta < pivot_gap &&
          minimum_diagonal - delta > 1e-10 && schur_maximum + delta < 8. * maximum_a)) return false;
    // Close the RHS induction separately. At a first 1e50 crossing, one update
    // is bounded by 2e50. The partial augmented factor has RHS error at most
    // 4*gamma(4N)*N*1e50, and the Schur RHS amplification is bounded above.
    const auto canonical_inverse = upward(inverse / (1. - inverse * perturbation));
    const auto rhs_amplification = upward(1. + upward(a_norm + canonical_backward) * canonical_inverse);
    if (!(rhs_amplification * 4. * gamma(4 * n) * double(n) < .25 &&
          rhs_amplification * 1e6 < .25e50)) return false;
    // U_C^-1=A_C^-1*L_C. |L_C|<=1+u, so every trailing triangular block is
    // covered by this bound. Guard backward-substitution roundoff before the
    // first 1e100 crossing; products/sums/divisions remain far from overflow.
    const auto canonical_upper_inverse = upward(2. * double(n) * canonical_inverse);
    const auto solve_perturbation = upward(gamma(4 * n) * 8. * double(n) * maximum_a + 1e-270);
    const auto upper_change = upward(canonical_upper_inverse * solve_perturbation);
    return upper_change < .25 &&
        canonical_upper_inverse * 1e50 / (1. - upper_change) < 1e90;
}
// Conventional FP64 partitioned GEPP has a componentwise backward envelope
// bounded by gamma(8N)|L||U| (including regrouped updates and reciprocal leaves).
// The local induction below still certifies the original ordered error path;
// final original-equation residuals and canonical regeneration remain mandatory.
bool local_lu_certificate(const double *a, const __LAPACK_int *pivots,
        std::size_t n, double *buffer) {
    constexpr std::size_t width = 256;
    // A coarse column bound can mix ordinary pivots with a much larger
    // later Schur column. Refine at most32 scale transitions or near-tie columns. This
    // only tightens an envelope: every column retains a conservative componentwise bound,
    // and exhausting the refinement budget keeps the conservative wide tile.
    std::vector<std::size_t> column_tile(n);
    std::size_t tiles = 0, refinements = 0;
    double smallest = 0., largest = 0.;
    bool previous_near_tie = false;
    for (std::size_t j = 0; j < n; ++j) {
        double scale = 0.;
        for (std::size_t i = 0; i <= j; ++i)
            scale = std::max(scale, std::abs(a[j * n + i]));
        double maximum_multiplier = 0.;
        for (std::size_t i = j + 1; i < n; ++i)
            maximum_multiplier = std::max(maximum_multiplier, std::abs(a[j * n + i]));
        const bool near_tie = maximum_multiplier > 1. - 1e-4;
        const auto low = std::min(smallest, scale), high = std::max(largest, scale);
        if (j % width == 0 || (refinements < 32 &&
                (high > 64. * low || near_tie || previous_near_tie))) {
            if (j % width != 0) ++refinements;
            ++tiles; smallest = largest = scale;
        } else {
            smallest = low; largest = high;
        }
        column_tile[j] = tiles - 1;
        previous_near_tie = near_tie;
    }
    const auto cells = n * tiles;
    auto *full = buffer;
    auto *tail = full + cells;
    auto *growth = tail + cells;
    auto *error = growth + cells;
    auto *panel = error + cells;
    auto *rhs = panel + 64 * n;
    auto *diagonal = rhs + n;
    auto *solution = diagonal + n;
    auto *left = solution + n;
    constexpr std::size_t batch_limit = 32;
    auto *right = left + batch_limit * tiles;
    std::fill_n(full, cells, 0.);
    std::fill_n(tail, cells, 0.);
    std::fill_n(error, cells, 0.);
    std::fill_n(rhs, n, 1e6);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i <= j; ++i) {
            const auto value = std::abs(a[j * n + i]);
            const auto t = column_tile[j];
            full[i * tiles + t] = std::max(full[i * tiles + t], value);
            if (i != j) tail[i * tiles + t] = std::max(tail[i * tiles + t], value);
        }
    for (std::size_t first = 0; first < n; first += 64) {
        const auto count = std::min<std::size_t>(64, n - first);
        // Preserve |L| * U envelopes, but read the F factor contiguously
        // and fill its bounded rectangle in the same column order. The
        // existing 64N workspace is unchanged; only its physical strides vary.
        for (std::size_t k = 0; k < n; ++k)
            for (std::size_t i = 0; i < count; ++i)
                panel[k * count + i] = k < first + i ? std::abs(a[k * n + first + i]) : k == first + i ? 1. : 0.;
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, int(count), int(tiles), int(n),
                    1., panel, int(count), full, int(tiles), 0., growth + first, int(n));
    }
    const auto inflate = 1. / (1. - gamma(32));
    const auto u = std::numeric_limits<double>::epsilon() * .5;
    for (std::size_t i = 0; i < cells; ++i) {
        growth[i] = std::nextafter(growth[i] / (1. - gamma(4 * n)) + 1e-270, INFINITY);
        if (!std::isfinite(growth[i]) || growth[i] > 1e100) return false;
    }
    for (std::size_t i = 0; i < n; ++i)
        if (pivots[i] < static_cast<__LAPACK_int>(i + 1) ||
            pivots[i] > static_cast<__LAPACK_int>(n)) return false;
    const auto backward = gamma(8 * n);
    for (std::size_t i = 0; i < cells; ++i) {
        error[i] = std::nextafter((backward * growth[i] + 1e-270) * inflate, INFINITY);
        if (!std::isfinite(error[i]) || error[i] > 1e100) return false;
    }
    // Local induction in final permutation coordinates. Strict separation
    // rejects ties and any uncertainty around the original 1e-14 threshold.
    const auto carry_scale = (1. + u) * inflate;
    for (std::size_t begin = 0; begin < n;) {
        const auto tile = column_tile[begin];
        auto end = begin + 1;
        while (end < n && end - begin < batch_limit && column_tile[end] == tile) ++end;
        const auto count = end - begin;
        double weights[batch_limit];
        weights[count - 1] = inflate;
        double scale = 1., sum_weights = 0.;
        for (std::size_t q = count - 1; q > 0; --q)
            weights[q - 1] = weights[q] * carry_scale;
        for (std::size_t q = 0; q < count; ++q) {
            scale *= carry_scale; sum_weights += weights[q];
        }
        auto *alpha = panel;
        auto *beta = panel + batch_limit * n;
        for (std::size_t p = begin; p < end; ++p) {
            const auto d = std::abs(a[p * n + p]), ep = error[tile * n + p];
            const auto lower = std::nextafter(d - ep, -INFINITY);
            diagonal[p] = lower;
            if (!std::isfinite(ep) || !(lower > 1e-10)) return false;
            for (std::size_t i = p + 1; i < n; ++i)
                if (!(lower > (std::abs(a[p * n + i]) * d + error[tile * n + i]) * inflate)) return false;
            const auto q = p - begin;
            for (std::size_t i = p + 1; i < n; ++i) {
                const auto multiplier = std::abs(a[p * n + i]), ei = error[tile * n + i];
                const auto quotient = (multiplier * d + ei) / lower;
                const auto delta = ((ei + multiplier * ep) / lower + u * quotient + 1e-270) * inflate;
                const auto absolute = (multiplier + delta) * inflate;
                if (!(absolute < 2.)) return false;
                rhs[i] = ((1. + u) * rhs[i] + (1. + u) * (1. + u) * absolute * rhs[p] + 1e-270) * inflate;
                if (!std::isfinite(rhs[i]) || rhs[i] > 1e100) return false;
                const auto changed = delta + u * absolute * (2. + u);
                const auto carried = (1. + u) * (1. + u) * absolute;
                alpha[q * n + i] = carried * weights[q];
                beta[q * n + i] = changed * weights[q];
                // All pivots in this batch use the same active column group.
                // Keep its induction, and future pivot rows, in original order.
                const auto after = i < end ? tiles : tile + 1;
                for (std::size_t t = tile; t < after; ++t) {
                    const auto index = t * n + i;
                    const auto next = ((1. + u) * error[index] +
                        changed * tail[p * tiles + t] + carried * error[t * n + p] +
                        u * growth[index] + 1e-270) * inflate;
                    if (!std::isfinite(next) || next > 1e100) return false;
                    error[index] = next;
                }
            }
        }
        const auto future = tiles - tile - 1, rows = n - end;
        if (future && rows) {
            // For independent rows outside the batch, unroll the positive
            // affine recurrence: c^count E + sum_q c^(count-1-q)(alpha E_p
            // + beta U_p + u G + tiny). Pivot-row states are already fixed.
            // The existing64N panel is dead after growth and holds
            // both coefficients; two bounded32×group tiles hold the right sides.
            for (std::size_t t = 0; t < future; ++t)
                for (std::size_t q = 0; q < count; ++q) {
                    left[t * count + q] = error[(tile + 1 + t) * n + begin + q];
                    right[t * count + q] = tail[(begin + q) * tiles + tile + 1 + t];
                }
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        int(rows), int(future), int(count), 1., alpha + end, int(n),
                        left, int(count), scale, error + (tile + 1) * n + end, int(n));
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        int(rows), int(future), int(count), 1., beta + end, int(n),
                        right, int(count), 1., error + (tile + 1) * n + end, int(n));
            // Nonnegative operands give a componentwise positive-product
            // envelope. Cover weights, scale/sum, both conventional FP64
            // products and the sequential recurrence with an outward margin.
            const auto batch_inflate = 1. / (1. - gamma(16 * count + 128));
            for (std::size_t t = tile + 1; t < tiles; ++t)
                for (std::size_t i = end; i < n; ++i) {
                    const auto index = t * n + i;
                    const auto next = std::nextafter((error[index] +
                        (u * growth[index] + 1e-270) * sum_weights + 1e-270) * batch_inflate, INFINITY);
                    if (!std::isfinite(next) || next > 1e100) return false;
                    error[index] = next;
                }
        }
        begin = end;
    }
    for (std::size_t reverse = 0; reverse < n; ++reverse) {
        const auto i = n - 1 - reverse;
        double bound = rhs[i];
        for (std::size_t j = i + 1; j < n; ++j)
            bound = (bound + (std::abs(a[j * n + i]) + error[column_tile[j] * n + i]) * solution[j] + 1e-270) * inflate;
        solution[i] = (bound / diagonal[i] + 1e-270) * inflate;
        if (!std::isfinite(solution[i]) || solution[i] > 1e100) return false;
    }
    return true;
}

#endif
} // namespace

bool try_dense_elementwise(const Prepared &p, Output &out, Workspace &work, Audit &audit,
                           bool private_arena) {
    const auto order = producer_order(p);
    if (out.kind != Kind::number || order < 0 || p.output_shape.dim[0] < 1024 ||
        p.output_shape.dim[0] > 4096 || !solve_accelerate_scratch_doubles(p.output_shape.dim[0], 1))
        return false;
    const auto target_order = private_arena && order == 1 ? MemoryOrder::f : out.order;
    if (order == 0 && target_order == MemoryOrder::c) return false;
    if (order == 1 && target_order == MemoryOrder::f) {
        produce_span(p, 0, p.output_shape.size(), static_cast<double *>(out.data), work, audit);
    } else {
        // Canonical contiguous row spans followed by a bounded physical
        // transpose tile; either direction needs no full intermediate result.
        constexpr std::size_t width = 32;
        double tile[width * width];
        const auto n = p.output_shape.dim[0];
        auto *target = static_cast<double *>(out.data);
        for (std::size_t first = 0; first < n; first += width) {
            const auto rows = std::min(width, n - first);
            for (std::size_t column = 0; column < n; column += width) {
                const auto cols = std::min(width, n - column);
                for (std::size_t i = 0; i < rows; ++i)
                    produce_span(p, (first + i) * n + column, cols, tile + i * width, work, audit);
                for (std::size_t j = 0; j < cols; ++j)
                    for (std::size_t i = 0; i < rows; ++i)
                        target[(column + j) * n + first + i] = tile[i * width + j];
            }
        }
        audit.algorithm_copy_bytes += n * n * sizeof(double);
    }
    out.order = target_order;
    return true;
}

// The graph proves a terminal floating result and exact private arena ownership.
// false can leave factor/output bytes changed: the caller must reconstruct the
// producer from its persistent parents and replay the canonical solve.
static bool solve_accelerate_core(const Prepared &p, const Prepared *producer,
                                  const Value *original, Output &out, Workspace &work,
                                  Audit &audit, double *coefficients, std::size_t offset) {
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL)
    if (__builtin_available(macOS 15.0, *)) {
    const auto n = p.args[0].shape.dim[0], k = p.args[1].shape.dim[1];
    const auto needed = solve_accelerate_scratch_doubles(n, k);
    const int order = producer ? producer_order(*producer) : dense_order(*original);
    if (!needed || order < 0 ||
        std::fegetround() != FE_TONEAREST || (!original && p.args[0].data != coefficients) ||
        p.args[0].shape.rank != 2 || p.args[0].shape.dim[1] != n ||
        p.args[1].shape.rank != 2 || out.kind != Kind::number) return false;
    const bool coefficient_f = p.args[0].stride[0] == 1 && p.args[0].stride[1] == std::ptrdiff_t(n);
    if (!coefficient_f && !p.args[0].contiguous()) return false;
    const auto *matrix = original ? static_cast<const double *>(original->data) : coefficients;
    if (original && work.doubles.capacity()) {
        const auto source = reinterpret_cast<std::uintptr_t>(matrix);
        const auto workspace = reinterpret_cast<std::uintptr_t>(work.doubles.data());
        if (source <= workspace ? workspace - source < n * n * sizeof(double) :
            source - workspace < work.doubles.capacity() * sizeof(double)) return false;
    }
    // Preserve B-before-A error evaluation. Admission scans never throw a new
    // numerical error: canonical replay retains the original exact exception.
    const auto &b = p.args[1];
    double maximum_b = 0.;
    for (std::size_t i = 0; i < b.size(); ++i) {
        const auto value = b.f(i);
        if (!std::isfinite(value)) return false;
        maximum_b = std::max(maximum_b, std::abs(value));
    }
    if (!(maximum_b > 0.) || maximum_b > 1e6) return false;

    std::vector<double> row_sum(n, 0.);
    double maximum_a = 0.;
    if (coefficient_f) {
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t i = 0; i < n; ++i) {
                const auto value = matrix[j * n + i];
                if (!std::isfinite(value)) return false;
                maximum_a = std::max(maximum_a, std::abs(value));
                if (i != j) row_sum[i] += std::abs(value);
            }
    } else {
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                const auto value = matrix[i * n + j];
                if (!std::isfinite(value)) return false;
                maximum_a = std::max(maximum_a, std::abs(value));
                if (i != j) row_sum[i] += std::abs(value);
            }
    }
    if (!(maximum_a >= 1e-8 && maximum_a <= 1e6)) return false;
    double margin = std::numeric_limits<double>::infinity(), a_norm = 0.;
    for (std::size_t i = 0; i < n; ++i) {
        const auto off_upper = std::nextafter(row_sum[i] / (1. - gamma(n)), INFINITY);
        a_norm = std::max(a_norm, std::nextafter(std::abs(matrix[i * n + i]) + off_upper, INFINITY));
        margin = std::min(margin, std::nextafter(std::abs(matrix[i * n + i]) - off_upper, -INFINITY));
    }
    // DD provides an independent forward-error bound. It does not itself
    // certify the canonical pivot choices; that requires the LU check below.
    const auto roundoff_budget = 8. * gamma(n) * double(n) * double(n) * maximum_a;
    if (!(margin > 4. * roundoff_budget && margin / double(n) > 1e-10)) return false;

    ThreadLimit threads;
    if (!threads.active) return false;
    ++audit.accelerate_lu_attempts;
    audit.native_library_workspace_reserve_bytes = 512 * n * sizeof(double);
    work.ensure(offset + needed, 0);
    auto *buffer = work.doubles.data() + offset;
    if (original) {
        coefficients = work.doubles.data();
        std::copy_n(matrix, n * n, coefficients);
        audit.algorithm_copy_bytes += n * n * sizeof(double);
    }
    auto *rhs = buffer;
    auto *certificate = rhs + n * k;
    for (std::size_t j = 0; j < k; ++j)
        for (std::size_t i = 0; i < n; ++i) rhs[j * n + i] = b.at(i, j);
    audit.algorithm_copy_bytes += n * k * sizeof(double);
    if (!coefficient_f) {
        transpose_square(coefficients, n);
        audit.algorithm_copy_bytes += n * n * sizeof(double);
    }
    std::vector<__LAPACK_int> pivots(n);
    __LAPACK_int size = static_cast<__LAPACK_int>(n), columns = static_cast<__LAPACK_int>(k), info = 0;
    factor_bounded_panels(size, coefficients, pivots.data(), 512, info);
    bool safe = info == 0;
    double factor_maximum = 0.;
    for (std::size_t i = 0; i < n * n; ++i) {
        safe = safe && std::isfinite(coefficients[i]);
        factor_maximum = std::max(factor_maximum, std::abs(coefficients[i]));
    }
    for (std::size_t i = 0; i < n; ++i)
        safe = safe && std::abs(coefficients[i * n + i]) > margin / (2. * double(n));
    safe = safe && factor_maximum <= 4. * std::max(1., maximum_a);
    safe = safe && canonical_lu_certificate(coefficients, pivots.data(), n, maximum_a, a_norm, certificate);
    if (!safe) { ++audit.accelerate_lu_fallbacks; return false; }
    const char trans = 'N';
    dgetrs_(&trans, &size, &columns, coefficients, &size, pivots.data(), rhs, &size, &info);
    if (info != 0) { ++audit.accelerate_lu_fallbacks; return false; }
    // Factor lifetime ends after certification and DGETRS. The exclusive N²
    // arena now holds bounded residual panels; the solved RHS stays separate.
    auto *residual_buffer = coefficients;
    auto *product = residual_buffer + 64 * n;
    auto *absolute_x = product + n * k;
    auto *absolute_product = absolute_x + n * k;
    for (std::size_t i = 0; i < n * k; ++i) {
        if (!std::isfinite(rhs[i]) || std::abs(rhs[i]) < 1e-280 || std::abs(rhs[i]) > 1e100) {
            ++audit.accelerate_lu_fallbacks; return false;
        }
        absolute_x[i] = std::abs(rhs[i]);
    }
    std::fill_n(product, n * k, 0.);
    std::fill_n(absolute_product, n * k, 0.);
    double residual_bound = 0.;
    // Canonical producer regenerated in bounded physical panels, never a
    // second full coefficient matrix. Both layouts retain their fast direction.
    for (std::size_t first = 0; first < n; first += 64) {
        const auto count = std::min<std::size_t>(64, n - first);
        // produce_span has zero scratch for add/subtract/multiply; it does not
        // invalidate the containing residual workspace.
        const double *panel = residual_buffer;
        if (producer) {
            Workspace arithmetic;
            produce_span(*producer, first * n, count * n, residual_buffer, arithmetic, audit);
        } else {
            // Signed DGEMM borrows the still-immutable original coefficient
            // panel. Only abs(A) is materialized into bounded scratch below.
            panel = matrix + first * n;
        }
        audit.solve_residual_materialization_bytes += count * n * sizeof(double);
        if (order == 0) {
            // A row panel times a column-major X; expose X as K*N transpose.
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(count), int(k), int(n),
                        1., panel, int(n), rhs, int(n), 0., product + first * k, int(k));
            for (std::size_t i = 0; i < count * n; ++i) residual_buffer[i] = std::abs(panel[i]);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(count), int(k), int(n),
                        1., residual_buffer, int(n), absolute_x, int(n), 0., absolute_product, int(k));
            for (std::size_t j = 0; j < k; ++j) {
                for (std::size_t i = 0; i < count; ++i) {
                    const auto r = std::abs(product[(first + i) * k + j] - b.at(first + i, j));
                    residual_bound = std::max(residual_bound, r + gamma(4 * n) *
                        (absolute_product[i * k + j] + std::abs(b.at(first + i, j))));
                }
            }
        } else {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, int(n), int(k), int(count),
                        1., panel, int(n), rhs + first, int(n), first ? 1. : 0., product, int(n));
            for (std::size_t i = 0; i < count * n; ++i) residual_buffer[i] = std::abs(panel[i]);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, int(n), int(k), int(count),
                        1., residual_buffer, int(n), absolute_x + first, int(n), first ? 1. : 0., absolute_product, int(n));
        }
    }
    if (order == 1) {
        for (std::size_t j = 0; j < k; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                const auto r = std::abs(product[j * n + i] - b.at(i, j));
                residual_bound = std::max(residual_bound, r + gamma(4 * n) *
                    (absolute_product[j * n + i] + std::abs(b.at(i, j))));
            }
        }
    }
    if (!std::isfinite(residual_bound) || !(residual_bound / margin < 4.9e-5)) {
        ++audit.accelerate_lu_fallbacks; return false;
    }
    auto *destination = static_cast<double *>(out.data);
    if (out.order == MemoryOrder::f) {
        std::copy_n(rhs, n * k, destination);
    } else {
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < k; ++j) destination[i * k + j] = rhs[j * n + i];
    }
    audit.algorithm_copy_bytes += n * k * sizeof(double);
    audit.native_library_workspace_reserve_bytes = 512 * n * sizeof(double);
    audit.scratch_bytes = (offset + needed) * sizeof(double) + n * sizeof(__LAPACK_int) + row_sum.capacity() * sizeof(double);
    audit.isa = original ? "accelerate-panel-lu-dgetrs-guarded-copy" : "accelerate-panel-lu-dgetrs-guarded-private";
    ++audit.accelerate_lu_calls;
    return true;
    }
#endif
    (void)p; (void)producer; (void)original; (void)out; (void)work; (void)audit; (void)coefficients; (void)offset;
    return false;
}
bool try_solve_accelerate_private(const Prepared &p, const Prepared &producer,
                                 Output &out, Workspace &work, Audit &audit,
                                 double *coefficients) {
    return solve_accelerate_core(p, &producer, nullptr, out, work, audit, coefficients, 0);
}

bool try_solve_accelerate_private_pair(const Prepared &first, const Prepared &second,
                                      const Prepared &producer, Output &out1, Output &out2,
                                      Workspace &work, Audit &audit, double *coefficients) {
    const auto n = first.args[0].shape.dim[0];
    const auto k1 = first.args[1].shape.dim[1], k2 = second.args[1].shape.dim[1];
    if (first.args[0].shape.rank != 2 || first.args[1].shape.rank != 2 ||
        second.args[1].shape.rank != 2 || !k1 || !k2 || k1 > 64 || k2 > 64 - k1 ||
        !solve_accelerate_scratch_doubles(n, k1 + k2)) return false;
    const auto k = k1 + k2, packed = n * k;
    // The prefix preserves original B until the common residual check finishes.
    // Core ensure() uses exactly this capacity, so the borrowed prefix stays live.
    work.ensure(packed + solve_accelerate_scratch_doubles(n, k), 0);
    auto *rhs = work.doubles.data();
    for (std::size_t j = 0; j < k; ++j)
        for (std::size_t i = 0; i < n; ++i)
            rhs[j * n + i] = j < k1 ? first.args[1].at(i, j) : second.args[1].at(i, j - k1);
    audit.algorithm_copy_bytes += packed * sizeof(double);
    auto combined = first;
    combined.args[1].shape = matrix_shape(n, k);
    combined.args[1].data = rhs;
    combined.args[1].set_contiguous_strides(MemoryOrder::f);
    combined.output_shape = matrix_shape(n, k);
    Output destination;
    destination.kind = Kind::number; destination.shape = combined.output_shape;
    destination.order = MemoryOrder::f; destination.data = rhs;
    if (!solve_accelerate_core(combined, &producer, nullptr, destination, work, audit,
                               coefficients, packed)) return false;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < k1; ++j) out1.set(i * k1 + j, rhs[j * n + i]);
        for (std::size_t j = 0; j < k2; ++j) out2.set(i * k2 + j, rhs[(k1 + j) * n + i]);
    }
    audit.algorithm_copy_bytes += packed * sizeof(double);
    return true;
}

bool try_solve_accelerate_copy(const Prepared &p, Output &out, Workspace &work, Audit &audit) {
    const auto &a = p.args[0];
    if (p.spec->op != Op::solve_multi || p.count != 2 || dense_order(a) < 0 ||
        p.args[1].shape.rank != 2 ||
        !solve_accelerate_scratch_doubles(a.shape.dim[0], p.args[1].shape.dim[1])) return false;
    if (solve_accelerate_core(p, nullptr, &a, out, work, audit, nullptr, a.size())) return true;
    // A remains intact even after failed factorization/certification. Execute
    // the same canonical body, retaining all actual attempted-library costs.
    Audit replay;
    execute(p, out, work, Isa::automatic, replay);
    replay.accelerate_lu_attempts = audit.accelerate_lu_attempts;
    replay.accelerate_lu_fallbacks = audit.accelerate_lu_fallbacks;
    replay.solve_residual_materialization_bytes = audit.solve_residual_materialization_bytes;
    replay.algorithm_copy_bytes += audit.algorithm_copy_bytes;
    replay.native_library_workspace_reserve_bytes = std::max(
        replay.native_library_workspace_reserve_bytes, audit.native_library_workspace_reserve_bytes);
    replay.scratch_bytes = std::max(replay.scratch_bytes, audit.scratch_bytes);
    audit = replay;
    return true;
}
std::size_t solve_concat_reserve_doubles(std::size_t n, std::size_t k) {
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL)
    if (__builtin_available(macOS 15.0, *))
        if (n >= 1024 && n <= 8192 && k >= 1 && k <= 64)
            return n * k + 4 * n * (((n + 255) / 256) + 32) + 67 * n + 64 * (((n + 255) / 256) + 32) + 516 * n;
#endif
    (void)n; (void)k;
    return 0;
}

bool try_solve_accelerate_concat(const Prepared &p, const SolveCoefficientBlocks &source,
        Output &out, Workspace &work, Audit &audit, double *coefficients) {
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL)
    if (__builtin_available(macOS 15.0, *)) {
        const auto n = p.args[0].shape.dim[0], k = p.args[1].shape.dim[1];
        const auto reserve = solve_concat_reserve_doubles(n, k);
        if (!reserve || std::fegetround() != FE_TONEAREST || !source.copy ||
            p.spec->op != Op::solve_multi || p.count != 2 || p.args[0].data != coefficients ||
            !p.args[0].dense_contiguous() || p.args[0].shape.rank != 2 || p.args[0].shape.dim[1] != n ||
            p.args[1].shape.rank != 2 || out.kind != Kind::number) return false;
        const auto &b = p.args[1];
        double maximum_b = 0.;
        for (std::size_t i = 0; i < b.size(); ++i) {
            const auto value = b.f(i), absolute = std::abs(value);
            if (!std::isfinite(value) || (absolute > 0. && absolute < 1e-280)) return false;
            maximum_b = std::max(maximum_b, absolute);
        }
        if (!(maximum_b > 0.) || maximum_b > 1e6) return false;
        const bool column_major = p.args[0].contiguous(MemoryOrder::f);
        double maximum_a = 0., a_norm = 0.;
        if (column_major) {
            // Same ascending-column row sums, with a contiguous F input scan.
            // The 8N-byte vector dies before the certificate workspace is used.
            std::vector<double> sums(n, 0.);
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t i = 0; i < n; ++i) {
                    const auto value = coefficients[j * n + i], absolute = std::abs(value);
                    if (!std::isfinite(value) || (absolute > 0. && absolute < 1e-280)) return false;
                    maximum_a = std::max(maximum_a, absolute); sums[i] += absolute;
                }
            for (const auto row : sums) a_norm = std::max(a_norm, row / (1. - gamma(n)));
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                double row = 0.;
                for (std::size_t j = 0; j < n; ++j) {
                    const auto value = coefficients[i * n + j], absolute = std::abs(value);
                    if (!std::isfinite(value) || (absolute > 0. && absolute < 1e-280)) return false;
                    maximum_a = std::max(maximum_a, absolute); row += absolute;
                }
                a_norm = std::max(a_norm, row / (1. - gamma(n)));
            }
        }
        if (!(maximum_a >= 1e-8 && maximum_a <= 1e6)) return false;
        ThreadLimit threads;
        if (!threads.active) return false;
        const auto needed = reserve - 516 * n;
        work.ensure(needed, 0);
        auto *rhs = work.doubles.data();
        auto *guard = rhs + n * k;
        audit.scratch_bytes = needed * sizeof(double) + n * (sizeof(__LAPACK_int) + sizeof(std::size_t));
        audit.native_library_workspace_reserve_bytes = 512 * n * sizeof(double);
        for (std::size_t j = 0; j < k; ++j)
            for (std::size_t i = 0; i < n; ++i) rhs[j * n + i] = b.at(i, j);
        audit.algorithm_copy_bytes += n * k * sizeof(double);
        ++audit.accelerate_lu_attempts;
        if (!column_major) {
            transpose_square(coefficients, n);
            audit.algorithm_copy_bytes += n * n * sizeof(double);
        }
        std::vector<__LAPACK_int> pivots(n);
        __LAPACK_int size = static_cast<__LAPACK_int>(n), columns = static_cast<__LAPACK_int>(k), info = 0;
        factor_bounded_panels(size, coefficients, pivots.data(), 512, info);
        bool safe = info == 0;
        for (std::size_t i = 0; i < n * n; ++i) safe = safe && std::isfinite(coefficients[i]);
        if (!safe || !local_lu_certificate(coefficients, pivots.data(), n, guard)) {
            ++audit.accelerate_lu_fallbacks; return false;
        }
        const char trans = 'N';
        dgetrs_(&trans, &size, &columns, coefficients, &size, pivots.data(), rhs, &size, &info);
        if (info != 0) { ++audit.accelerate_lu_fallbacks; return false; }
        // Factor is dead. Its private arena holds the original-equation residual
        // panels; only the solved RHS must remain live in separate workspace.
        auto *panel = coefficients, *product = panel + 64 * n;
        auto *absolute_x = product + 64 * k, *absolute_product = absolute_x + n * k;
        double maximum_x = 0.;
        for (std::size_t i = 0; i < n * k; ++i) {
            if (!std::isfinite(rhs[i]) || std::abs(rhs[i]) > 1e100) {
                ++audit.accelerate_lu_fallbacks; return false;
            }
            absolute_x[i] = std::abs(rhs[i]); maximum_x = std::max(maximum_x, absolute_x[i]);
        }
        double residual = 0.;
        for (std::size_t first = 0; first < n; first += 64) {
            const auto count = std::min<std::size_t>(64, n - first);
            source.copy(source.context, first, count, 0, n, panel);
            audit.solve_residual_materialization_bytes += count * n * sizeof(double);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(count), int(k), int(n),
                        1., panel, int(n), rhs, int(n), 0., product, int(k));
            for (std::size_t i = 0; i < count * n; ++i) panel[i] = std::abs(panel[i]);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(count), int(k), int(n),
                        1., panel, int(n), absolute_x, int(n), 0., absolute_product, int(k));
            for (std::size_t i = 0; i < count; ++i)
                for (std::size_t j = 0; j < k; ++j) {
                    const auto original_b = b.at(first + i, j);
                    residual = std::max(residual, std::abs(product[i * k + j] - original_b) +
                        gamma(4 * n) * (absolute_product[i * k + j] + std::abs(original_b)));
                }
        }
        const auto scale = a_norm * maximum_x + maximum_b;
        if (!std::isfinite(residual) || !std::isfinite(scale) || !(residual <= 1e-10 * scale)) {
            ++audit.accelerate_lu_fallbacks; return false;
        }
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < k; ++j) out.set(i * k + j, rhs[j * n + i]);
        audit.algorithm_copy_bytes += n * k * sizeof(double);
        audit.isa = "accelerate-panel-lu-dgetrs-measured-private";
        ++audit.accelerate_lu_calls;
        return true;
    }
#endif
    (void)p; (void)source; (void)out; (void)work; (void)audit; (void)coefficients;
    return false;
}

} // namespace calmetrics_engine::ops
