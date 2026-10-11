#include "calmetrics_engine/operators.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include <algorithm>
#include <cfenv>
#include <climits>
#include <cmath>
#include <limits>

#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(__APPLE__) && defined(__aarch64__)
#include <Accelerate/Accelerate.h>
#endif

namespace calmetrics_engine::ops {
std::size_t transcendental_accelerate_scratch_doubles(const Shape &shape) {
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(__APPLE__) && defined(__aarch64__)
    if (__builtin_available(macOS 15.0, *)) {
        if (shape.rank < 2 || shape.rank > 3) return 0;
        const auto n = shape.size();
        const auto rows = shape.dim[0];
        if (n < 65536 || n > INT_MAX || rows < 32) return 0;
        const auto columns = n / rows;
        if (columns < 8) return 0;
        return 2 * std::max(std::min(rows, std::size_t{128}) * std::min(columns, std::size_t{32}),
                            std::min(rows, std::size_t{64}) * std::min(columns, std::size_t{64}));
    }
#else
    (void)shape;
#endif
    return 0;
}

bool try_accelerate_transcendental(const Prepared &p, Output &out,
                                  Workspace &work, Audit &audit, bool independent_root) {
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(__APPLE__) && defined(__aarch64__)
    if (__builtin_available(macOS 15.0, *)) {
        if (!p.spec || p.count != 1 || p.borrowed || p.structure_only ||
            (p.spec->op != Op::exp && p.spec->op != Op::log) ||
            p.output_kind != Kind::number || out.kind != Kind::number ||
            !(out.shape == p.output_shape) || out.shape.rank < 1 || out.shape.rank > 3 ||
            (out.order != MemoryOrder::c && out.order != MemoryOrder::f) ||
            !out.data || std::fegetround() != FE_TONEAREST) return false;
        const auto &input = p.args[0];
        const auto n = out.shape.size();
        if (input.kind != Kind::number || !input.data || !(input.shape == out.shape) ||
            n < 1024 || n > INT_MAX || p.scratch_doubles || p.scratch_indices)
            return false;
        const bool contiguous = input.contiguous(out.order);
        const auto rows = input.shape.dim[0];
        const auto d = input.shape.rank > 1 ? input.shape.dim[1] : 1;
        const auto k = input.shape.rank == 3 ? input.shape.dim[2] : 1;
        const auto columns = n / rows;
        // Geometry, not payload, decides the usable physical traversal.
        const bool column_geometry = input.shape.rank >= 2 && input.stride[0] == 1 &&
            input.stride[1] >= static_cast<std::ptrdiff_t>(rows) &&
            (input.shape.rank == 2 ||
             (d <= static_cast<std::size_t>(PTRDIFF_MAX) /
                   static_cast<std::size_t>(input.stride[1]) &&
              input.stride[2] >= input.stride[1] * static_cast<std::ptrdiff_t>(d)));
        const bool row_geometry = input.shape.rank >= 2 &&
            input.stride[0] >= static_cast<std::ptrdiff_t>(columns) &&
            (input.shape.rank == 2 ? input.stride[1] == 1 :
             input.stride[2] == 1 && input.stride[1] == static_cast<std::ptrdiff_t>(k));
        const bool column_runs = out.order == MemoryOrder::f && rows >= 1024 && column_geometry;
        const bool row_runs = out.order == MemoryOrder::c && columns >= 1024 && row_geometry;
        const bool tiled = !contiguous && !column_runs && !row_runs &&
            transcendental_accelerate_scratch_doubles(input.shape) &&
            ((out.order == MemoryOrder::c && column_geometry) ||
             (out.order == MemoryOrder::f && row_geometry));
        if (!contiguous && !column_runs && !row_runs && !tiled) return false;
        const auto *x = static_cast<const double *>(input.data);
        auto *y = static_cast<double *>(out.data);
        const auto valid = [&](const double *values, std::size_t length) {
            for (std::size_t i = 0; i < length; ++i) {
                const auto value = values[i];
                if (!std::isfinite(value) ||
                    (p.spec->op == Op::exp ? std::abs(value) > 700.0 :
                     value < std::numeric_limits<double>::min())) return false;
            }
            return true;
        };
        // Direct API failures expose the canonical written prefix. They still
        // reject the entire library attempt before any write. Independent graph
        // roots are unpublished on failure, so only a bad block needs replay.
        // Neither traversal reads padding or window-external values.
        if (!independent_root) {
            if (contiguous) {
                if (!valid(x, n)) return false;
            } else if (column_geometry) {
                for (std::size_t plane = 0; plane < k; ++plane)
                    for (std::size_t col = 0; col < d; ++col)
                        if (!valid(x + col * input.stride[1] + plane * input.stride[2], rows))
                            return false;
            } else {
                for (std::size_t row = 0; row < rows; ++row)
                    if (!valid(x + row * input.stride[0], columns)) return false;
            }
        }
        audit = {};
        const auto library_apply = [&](double *dst, const double *src, std::size_t count) {
            const int length = static_cast<int>(count);
            if (p.spec->op == Op::exp) vvexp(dst, src, &length);
            else vvlog(dst, src, &length);
            ++audit.accelerate_vforce_calls;
            audit.vector_elements += count;
        };
        const auto apply = [&](double *dst, const double *src, std::size_t count) {
            if (!independent_root) {
                library_apply(dst, src, count);
                return;
            }
            // Bound the part replayed by one outlier without a full-size
            // validity bitmap or transaction buffer. Validation stays close to
            // the library's read, and every scalar uses the existing contract.
            constexpr std::size_t block = 65536;
            for (std::size_t first = 0; first < count; first += block) {
                const auto size = std::min(block, count - first);
                if (valid(src + first, size)) library_apply(dst + first, src + first, size);
                else for (std::size_t i = 0; i < size; ++i)
                    dst[first + i] = scalar_math(p.spec->op, src[first + i]);
            }
        };
        if (tiled) {
            const bool input_f = column_geometry;
            const auto block_rows = std::min(rows, input_f ? std::size_t{128} : std::size_t{64});
            const auto block_cols = std::min(columns, input_f ? std::size_t{32} : std::size_t{64});
            const auto capacity = block_rows * block_cols;
            work.ensure(2 * capacity, 0);
            auto *a = work.doubles.data(), *b = a + capacity;
            audit.scratch_bytes = 2 * capacity * sizeof(double);
            audit.native_library_workspace_reserve_bytes =
                transcendental_accelerate_scratch_doubles(input.shape) * sizeof(double);
            for (std::size_t first = 0; first < rows; first += block_rows) {
                const auto nr = std::min(block_rows, rows - first);
                for (std::size_t col = 0; col < columns; col += block_cols) {
                    const auto nc = std::min(block_cols, columns - col);
                    if (input_f) {
                        for (std::size_t j = 0; j < nc; ++j) {
                            const auto index = col + j;
                            apply(a + j * nr, x + first + (index / k) * input.stride[1] +
                                  (index % k) * input.stride[2], nr);
                        }
                        vDSP_mtransD(a, 1, b, 1, nr, nc);
                        vDSP_mmovD(b, y + first * columns + col, nc, nr, nc, columns);
                    } else {
                        for (std::size_t i = 0; i < nr; ++i)
                            apply(a + i * nc, x + (first + i) * input.stride[0] + col, nc);
                        vDSP_mtransD(a, 1, b, 1, nc, nr);
                        for (std::size_t j = 0; j < nc; ++j) {
                            const auto index = col + j;
                            std::copy_n(b + j * nr, nr,
                                y + first + ((index / k) + d * (index % k)) * rows);
                        }
                    }
                    // Transpose and final move are copies; vForce's mathematical
                    // result write is not an input pack/copy.
                    audit.algorithm_copy_bytes += 2 * nr * nc * sizeof(double);
                }
            }
        } else {
            work.ensure(0, 0);
            if (contiguous) apply(y, x, n);
            else if (row_runs) {
                for (std::size_t row = 0; row < rows; ++row)
                    apply(y + row * columns, x + row * input.stride[0], columns);
            } else {
                for (std::size_t run = 0; run < columns; ++run)
                    apply(y + run * rows,
                          x + (run % d) * input.stride[1] + (run / d) * input.stride[2], rows);
            }
        }
        audit.isa = !audit.accelerate_vforce_calls ? "scalar" :
            tiled ? (p.spec->op == Op::exp ? "accelerate-vforce-exp-tiled" : "accelerate-vforce-log-tiled") :
            (p.spec->op == Op::exp ? "accelerate-vforce-exp" : "accelerate-vforce-log");
        return true;
    }
#else
    (void)p; (void)out; (void)work; (void)audit; (void)independent_root;
#endif
    return false;
}
} // namespace calmetrics_engine::ops
