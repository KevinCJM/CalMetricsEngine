#include "calmetrics_engine/operators.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include <algorithm>
#include <cfenv>
#include <climits>
#include <cmath>
#include <limits>

#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_MPL2_ONLY
#define EIGEN_FAST_MATH 0
#include <Eigen/Core>
#endif

namespace calmetrics_engine::ops {
namespace {
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
constexpr double eps = std::numeric_limits<double>::epsilon();
constexpr double padding = 1.0 + 32.0 * eps;
double gamma(double count) {
    const double q = count * (eps * 0.5);
    return q / (1.0 - q) * padding;
}
std::size_t panel_rows(const Shape &shape) {
    return shape.dim[0] <= 2048 && shape.dim[1] <= 2048 ? shape.dim[0] : 512;
}
struct API_AVAILABLE(macos(15.0)) ThreadLimit {
    BLAS_THREADING previous = BLASGetThreading();
    bool active = false;
    ThreadLimit() {
        if (previous != BLAS_THREADING_MULTI_THREADED && previous != BLAS_THREADING_SINGLE_THREADED)
            return;
        active = BLASSetThreading(BLAS_THREADING_SINGLE_THREADED) == 0;
        if (!active || BLASGetThreading() != BLAS_THREADING_SINGLE_THREADED) {
            active = true;
            restore();
        }
    }
    void restore() {
        if (!active) return;
        const auto status = BLASSetThreading(previous);
        active = false;
        require(status == 0 && BLASGetThreading() == previous,
                "NATIVE_LIBRARY_THREAD_RESTORE_FAILED");
    }
    ~ThreadLimit() { if (active) BLASSetThreading(previous); }
};
bool guarded(const Value &input, int layout, int block, double *result, Workspace &work,
             Audit &audit, bool &centered_ready, bool correlation, bool correlation_consumer,
             bool short_panel = false) {
    const auto n = input.shape.dim[0], d = input.shape.dim[1];
    const auto *x = static_cast<const double *>(input.data);
    if (n < 32 || d < 256 || n > 1000000 || d > 4096) return false;
    // Bounded 32 MiB panel for medium squares; large inputs retain 512 rows.
    if (n <= 2048 && d >= 256 && d <= 2048) block = static_cast<int>(n);
    // Paired ordered-fold roots already retain two matrices; halve the medium panel.
    // Positive F keeps its complete panel for the canonical centered fallback.
    if (correlation_consumer && !layout) block = std::min(block, 1024);
    if (short_panel) block = 128;
    const auto leading = input.stride[layout ? 1 : 0];
    // Lower triangle carries transient error bounds; only a bounded panel and
    // three D-sized vectors are separately allocated.
    const auto scratch = (static_cast<std::size_t>(block) + 3) * d;
    if (work.doubles.capacity() != scratch) std::vector<double>().swap(work.doubles);
    std::vector<std::size_t>().swap(work.indices);
    work.ensure(scratch, 0);
    auto *panel = work.doubles.data(), *means = panel + block * d;
    auto *norm = means + d, *total = norm + d;
    std::fill_n(total, d, 0.0);
    audit.scratch_bytes = work.capacity_bytes();
    covariance_library_means(input, means);
    const bool global_bound = n <= 4096 && !short_panel;
    // With a global bound, the lower triangle need not carry partial errors.
    // Use the faster vendor triangle/layout already screened for F inputs.
    const bool lower_result = layout && global_bound;
    using Row = Eigen::Array<double, 1, Eigen::Dynamic>;
    using Column = Eigen::Array<double, Eigen::Dynamic, 1>;
    for (std::size_t first = 0; first < n; first += block) {
        const auto count = std::min(static_cast<std::size_t>(block), n - first);
        const double eta = gamma(static_cast<double>(count) + 8.0);
        std::fill_n(norm, d, 0.0);
        if (!layout) {
            const std::size_t width = input.stride[1] == 1 ? d : 8;
            for (std::size_t j = 0; j < d; j += width) {
                const auto columns = std::min(width, d - j);
                const Eigen::Map<const Row> mu(means + j, columns);
                for (std::size_t i = 0; i < count; ++i) {
                    Eigen::Map<Row> target(panel + i * d + j, columns);
                    const auto *row = x + static_cast<std::ptrdiff_t>(first + i) * leading +
                                        static_cast<std::ptrdiff_t>(j) * input.stride[1];
                    if (input.stride[1] == 1)
                        target = Eigen::Map<const Row>(row, columns) - mu;
                    else
                        target = Eigen::Map<const Row, Eigen::Unaligned, Eigen::InnerStride<>>(
                            row, columns, Eigen::InnerStride<>(input.stride[1])) - mu;
                    Eigen::Map<Row>(norm + j, columns) += target.square();
                    // Each diagonal keeps its full ascending observation fold.
                    Eigen::Map<Row>(total + j, columns) += target.square();
                }
            }
        } else {
            for (std::size_t j = 0; j < d; ++j) {
                Eigen::Map<Column> target(panel + j * count, count);
                target = Eigen::Map<const Column>(x + j * leading + first, count) - means[j];
                norm[j] = target.square().sum();
                total[j] += norm[j];
            }
        }
        audit.covariance_panel_materialization_bytes += count * d * sizeof(double);
        // Controlled exponents exclude subnormal products. The source cap
        // already bounds all centered squares and norm folds away from overflow.
        double peak = 0.0;
        for (std::size_t i = 0; i < count * d; ++i) {
            const double value = std::abs(panel[i]);
            peak = std::max(peak, value);
            if (value != 0.0 && value < 0x1p-400) return false;
        }
        if (!correlation && peak > 8.0) {
            // Retry only an actual amplitude rejection, once. Ordinary inputs
            // keep their faster global bound; larger amplitudes retain replay.
            if (!short_panel && !layout && n >= 512 && n <= 4096 && peak <= 32.0)
                return guarded(input, layout, 128, result, work, audit, centered_ready,
                               correlation, correlation_consumer, true);
            if (!short_panel || peak > 32.0) {
                centered_ready = layout && first == 0 && count == n && d % 2 == 0;
                return false;
            }
        }
        for (std::size_t j = 0; j < d; ++j)
            norm[j] = std::sqrt(norm[j] * (1.0 + 4.0 * eta)) * padding;
        if (short_panel && !layout) {
            // One conservative bound per 32-column group avoids a full
            // triangle update; group maxima preserve the original recurrence.
            constexpr std::size_t group_width = 32;
            for (std::size_t j = 0; j < d; ++j)
                for (std::size_t k = j + 1; k < d;) {
                    const auto last = std::min(d, (k / group_width + 1) * group_width);
                    const auto width = last - k;
                    const double prior = first == 0 ? 0.0 :
                        Eigen::Map<const Row>(result + j * d + k, width).abs().maxCoeff();
                    const double bound = first == 0 ? 0.0 : result[k * d + j];
                    const double max_norm = Eigen::Map<const Row>(norm + k, width).maxCoeff();
                    const double products = norm[j] * max_norm * padding;
                    result[k * d + j] = ((1.0 + eta) * bound +
                        2.0 * eta * (prior + products)) * padding;
                    if (!std::isfinite(result[k * d + j])) return false;
                    k = last;
                }
        } else if (!global_bound) {
            // An 8 KiB transpose tile permits contiguous Eigen packet updates
            // of independent bounds without a second full matrix.
            constexpr std::size_t tile = 32;
            using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            using Tile = Eigen::Matrix<double, 32, 32, Eigen::RowMajor>;
            Eigen::Map<Matrix> matrix(result, d, d);
            Tile prior_storage;
            for (std::size_t jb = 0; jb < d; jb += tile) {
                const auto columns = std::min(tile, d - jb);
                for (std::size_t j = jb; j < jb + columns; ++j)
                    for (std::size_t k = j + 1; k < jb + columns; ++k) {
                        const double prior = first == 0 ? 0.0 : result[j * d + k];
                        const double bound = first == 0 ? 0.0 : result[k * d + j];
                        const double products = norm[j] * norm[k] * padding;
                        result[k * d + j] = layout ?
                            (bound + eta * (std::abs(prior) + products)) * padding :
                            ((1.0 + eta) * bound + 2.0 * eta * (std::abs(prior) + products)) * padding;
                    }
                const Eigen::Map<const Row> norms(norm + jb, columns);
                for (std::size_t kb = jb + tile; kb < d; kb += tile) {
                    const auto rows = std::min(tile, d - kb);
                    auto prior = prior_storage.topLeftCorner(rows, columns);
                    if (first == 0) prior.setZero();
                    else prior = matrix.block(jb, kb, columns, rows).transpose().cwiseAbs();
                    for (std::size_t k = 0; k < rows; ++k) {
                        Eigen::Map<Row> bound(result + (kb + k) * d + jb, columns);
                        if (first == 0) bound.setZero();
                        if (layout)
                            bound = (bound + eta * (prior.row(k).array() + norms * norm[kb+k] * padding)) * padding;
                        else
                            bound = ((1.0 + eta) * bound +
                                2.0 * eta * (prior.row(k).array() + norms * norm[kb+k] * padding)) * padding;
                    }
                }
            }
        }
        cblas_dsyrk(lower_result ? CblasColMajor : CblasRowMajor,
            CblasUpper, lower_result ? CblasTrans : layout ? CblasNoTrans : CblasTrans,
            static_cast<int>(d), static_cast<int>(count), 1.0, panel,
            static_cast<int>(layout ? count : d), first == 0 ? 0.0 : 1.0,
            result, static_cast<int>(d));
        ++audit.covariance_syrk_calls;
    }
    const double denominator = static_cast<double>(n - 1);
    const double norm_guard = gamma(layout ? std::ceil(static_cast<double>(n) / block) + block + 8.0
                                           : static_cast<double>(n) + 8.0);
    for (std::size_t j = 0; j < d; ++j) {
        result[j * d + j] = layout ?
            covariance_library_pair(input, j, j, means) / denominator :
            total[j] / denominator;
        if (layout) ++audit.covariance_replayed_cells;
        if (correlation || correlation_consumer) {
            // The canonical diagonal fixes the exact normalization denominator.
            // Block norms are dead now; no additional D-sized buffer is needed.
            norm[j] = std::sqrt(result[j * d + j]);
            if (!(norm[j] > 0.0) || !std::isfinite(norm[j])) return false;
        }
        // Positive norm folds have a deterministic relative error bound.
        total[j] = std::sqrt(total[j] / (1.0 - norm_guard)) * padding;
    }
    const double old_depth = layout ? std::ceil(static_cast<double>(n) / 4.0) + 8.0 : n + 8.0;
    const double old_eta = gamma(old_depth);
    std::size_t uncertain = 0;
    for (std::size_t j = 0; j < d; ++j) {
        double group_bound = 0.0;
        for (std::size_t k = j + 1; k < d; ++k) {
            // Read the representative before replacing this group's lower
            // triangle with per-cell replay markers.
            if (short_panel && !layout && (k == j + 1 || k % 32 == 0))
                group_bound = result[k * d + j];
            const double value = result[(lower_result ? k : j) * d + (lower_result ? j : k)] / denominator;
            double bound;
            const double products = total[j] * total[k] * padding;
            if (global_bound)
                bound = (old_eta + gamma(static_cast<double>(n) + 8.0)) * products;
            else
                bound = (short_panel && !layout ? group_bound : result[k * d + j]) +
                        (layout ? old_eta * products : 0.0);
            const double error = (bound / denominator + eps * std::abs(value)) * padding;
            double checked_value = value, checked_error = error;
            if (correlation || correlation_consumer) {
                const double scale = norm[j] * norm[k];
                if (!(scale > 0.0) || !std::isfinite(scale)) return false;
                checked_value = value / scale;
                checked_error = (error / scale + eps * std::abs(checked_value)) * padding;
            }
            const double tolerance = (1e-12 + 1e-12 * std::max(0.0, std::abs(checked_value) - checked_error)) /
                                     (1.0 + 64.0 * eps);
            const bool replay = checked_error > tolerance || std::abs(value) <= error ||
                (correlation_consumer && error >
                 (1e-12 + 1e-12 * std::max(0.0, std::abs(value) - error)) / (1.0 + 64.0 * eps));
            uncertain += replay;
            result[j * d + k] = value;
            result[k * d + j] = replay ? -1.0 : 0.0;
        }
    }
    // Dense uncertainty would spend more than a canonical replay. The caller
    // overwrites the unpublished output with the sole canonical native kernel.
    if (uncertain > d * (d - 1) / 64) return false;
    audit.covariance_replayed_cells += uncertain;
    for (std::size_t j = 0; j < d; ++j)
        for (std::size_t k = j + 1; k < d; ++k) {
            double value = result[j * d + k];
            if (result[k * d + j] < 0.0)
                value = covariance_library_pair(input, j, k, means) / denominator;
            result[j * d + k] = result[k * d + j] = value;
        }
    return true;
}
#endif
} // namespace

std::size_t covariance_accelerate_reserve_doubles(const Shape &shape) {
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    if (__builtin_available(macOS 15.0, *)) {
        if (shape.rank == 2 && shape.dim[0] >= 32 && shape.dim[0] <= 1000000 &&
            shape.dim[1] >= 256 && shape.dim[1] <= 4096) {
            // A shorter interval can choose a larger medium panel than the
            // longest interval. Bound both regimes, plus a measured estimate
            // for the system library; this is not a vendor hard memory bound.
            const auto rows = shape.dim[1] <= 2048 ? std::min<std::size_t>(shape.dim[0], 2048) : 512;
            return (rows + 3) * shape.dim[1] + 4 * 1024 * 1024;
        }
    }
#else
    (void)shape;
#endif
    return 0;
}

bool try_accelerate_covariance(const Prepared &p, Output &output, Workspace &work, Audit &audit,
                              bool correlation_consumer) {
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    if (__builtin_available(macOS 15.0, *)) {
        const auto discard = [&] { std::vector<double>().swap(work.doubles); };
        const bool correlation = p.spec->op == Op::correlation;
        if ((!correlation && p.spec->op != Op::covariance) || p.count != 1 || output.kind != Kind::number ||
            output.order != MemoryOrder::c || !output.data) return false;
        const auto &input = p.args[0];
        const auto reserve = covariance_accelerate_reserve_doubles(input.shape);
        if (!reserve || input.kind != Kind::number || !input.data || std::fegetround() != FE_TONEAREST) {
            discard(); return false;
        }
        const auto n = input.shape.dim[0], d = input.shape.dim[1];
        // BLAS reads only the centered panel; source rows may descend while
        // means, canonical diagonals and replay retain logical observation order.
        const auto columns = static_cast<std::ptrdiff_t>(d);
        const bool c = input.stride[1] == 1 &&
            (input.stride[0] >= columns || input.stride[0] <= -columns);
        const bool f = input.stride[0] == 1 && input.stride[1] >= static_cast<std::ptrdiff_t>(n);
        const bool reverse_f = input.stride[0] == -1 && input.stride[1] >= static_cast<std::ptrdiff_t>(n);
        if (!c && !f && !reverse_f) { discard(); return false; }
        // Negative F uses the canonical scalar observation fold, like C.
        const int layout = f ? 1 : 0;
        const bool column_scan = f || reverse_f;
        const auto leading = input.stride[column_scan ? 1 : 0];
        const auto inner_stride = input.stride[column_scan ? 0 : 1];
        if (leading > INT_MAX || leading < -INT_MAX) { discard(); return false; }
        const auto *data = static_cast<const double *>(input.data);
        // Correlation normalizes its error bound. With n <= 1e6 this
        // exponent cap keeps centered squares, norm folds and products finite.
        const double source_limit = correlation ? 0x1p400 : 1e4;
        bool all_zero = true, positive_zero = true;
        for (std::size_t outer = 0; outer < (column_scan ? d : n); ++outer)
            for (std::size_t inner = 0; inner < (column_scan ? n : d); ++inner) {
                const double value = data[static_cast<std::ptrdiff_t>(outer) * leading +
                                          static_cast<std::ptrdiff_t>(inner) * inner_stride];
                if (!std::isfinite(value) || std::abs(value) > source_limit) { discard(); return false; }
                all_zero = all_zero && value == 0.0;
                positive_zero = positive_zero && value == 0.0 && !std::signbit(value);
            }
        if (all_zero) {
            discard();
            std::vector<std::size_t>().swap(work.indices);
            // Under nearest rounding, an all-+0 input has exact +0 Welford
            // means, centered products, sums and sample division in both
            // canonical layouts. Signed-zero inputs retain that layout's
            // original reduction, without a futile DSYRK attempt.
            if (!positive_zero || correlation) return false;
            std::fill_n(static_cast<double *>(output.data), d * d, 0.0);
            audit = {};
            audit.covariance_zero_calls = 1;
            audit.vector_elements = d * d;
            audit.isa = "canonical-positive-zero-covariance";
            return true;
        }
        ThreadLimit threads;
        if (!threads.active) { discard(); return false; }
        audit = {};
        audit.accelerate_covariance_attempts = 1;
        audit.native_library_workspace_reserve_bytes = reserve * sizeof(double);
        bool centered_ready = false;
        const bool accepted = guarded(input, layout, static_cast<int>(panel_rows(input.shape)),
                                      static_cast<double *>(output.data), work, audit, centered_ready,
                                      correlation, correlation_consumer);
        threads.restore();
        if (accepted) {
            if (correlation) finish_covariance_matrix(output, true);
            audit.accelerate_covariance_calls = 1;
            audit.vector_elements = output.shape.size();
            audit.isa = correlation ? "accelerate-guarded-correlation" : "accelerate-guarded-covariance";
        } else {
            const auto attempted = audit;
            if (centered_ready) {
                covariance_library_centered(input, output, work, audit);
                discard();
                if (correlation) finish_covariance_matrix(output, true);
            } else {
                discard();
                execute(p, output, work, Isa::automatic, audit);
            }
            audit.accelerate_covariance_attempts = attempted.accelerate_covariance_attempts;
            audit.covariance_syrk_calls = attempted.covariance_syrk_calls;
            audit.covariance_replayed_cells = attempted.covariance_replayed_cells + d * (d + 1) / 2;
            audit.covariance_fallbacks = 1;
            audit.covariance_panel_materialization_bytes = attempted.covariance_panel_materialization_bytes;
            audit.native_library_workspace_reserve_bytes = attempted.native_library_workspace_reserve_bytes;
        }
        return true;
    }
#else
    (void)p; (void)output; (void)work; (void)audit; (void)correlation_consumer;
#endif
    return false;
}
} // namespace calmetrics_engine::ops
