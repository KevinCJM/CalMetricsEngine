#include "calmetrics_engine/operators.hpp"
#include "axis_primitives.hpp"
#include "numeric_line_sort.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace calmetrics_engine::ops {
namespace {

void numeric_arg(const Prepared &p, std::size_t index, int low, int high) {
    require(p.args[index].kind == Kind::number, "DTYPE_MISMATCH");
    if (p.geometry(index)) {
        const auto &shape = p.args[index].shape;
        require(shape.rank >= low && shape.rank <= high, "RANK_MISMATCH");
        require(shape.size() <= static_cast<std::size_t>(PTRDIFF_MAX) / sizeof(double),
                "SHAPE_OVERFLOW");
    }
}

bool axis_arg(Prepared &p, std::size_t index, int rank) {
    numeric_arg(p, index, 0, 0);
    if (!p.payload(index)) return false;
    const double value = p.args[index].scalar;
    const bool valid = std::isfinite(value) && value >= 0 &&
        value <= (rank < 0 ? 2 : rank - 1) && value == std::floor(value);
    // Preserve the shared structural-only classification of an invalid
    // numerical axis, while still checking every independent operand below.
    if (!valid && p.structure_only) {
        p.payload_available_mask &= ~(1u << index);
        return false;
    }
    require(valid, "INVALID_PARAMETER");
    return true;
}

std::size_t checked_product(std::size_t left, std::size_t right) {
    require(!right || left <= static_cast<std::size_t>(PTRDIFF_MAX) / right,
            "SHAPE_OVERFLOW");
    return left * right;
}

// Decode each non-axis line once. No division/offset is formed for empty
// outputs, and signed/zero strides preserve valid borrowed input layouts.
std::ptrdiff_t line_offset(const Value &values, std::size_t axis,
                           std::size_t outer, std::size_t inner) {
    std::ptrdiff_t offset = 0;
    for (int dimension = static_cast<int>(axis) - 1; dimension >= 0; --dimension) {
        offset += static_cast<std::ptrdiff_t>(outer % values.shape.dim[dimension]) *
            values.stride[dimension];
        outer /= values.shape.dim[dimension];
    }
    for (int dimension = values.shape.rank - 1; dimension > static_cast<int>(axis); --dimension) {
        offset += static_cast<std::ptrdiff_t>(inner % values.shape.dim[dimension]) *
            values.stride[dimension];
        inner /= values.shape.dim[dimension];
    }
    return offset;
}

} // namespace

void axis_path_prepare(Prepared &p) {
    const auto op = p.spec->op;
    const bool quantiles = op == Op::quantiles_axis;
    require(quantiles || op == Op::cumulative_product_axis || op == Op::cumulative_max_axis,
            "UNKNOWN_OPERATOR");
    numeric_arg(p, 0, 1, 3);
    const auto axis_index = quantiles ? 2u : 1u;
    // Independent shape checks do not depend on values being readable.
    numeric_arg(p, quantiles ? 1u : 2u, quantiles ? 1 : 0, quantiles ? 1 : 0);
    const int rank = p.geometry(0) ? p.args[0].shape.rank : -1;
    const bool axis_known = axis_arg(p, axis_index, rank);
    if (quantiles && p.payload(1)) {
        const auto &probabilities = p.args[1];
        for (std::size_t q = 0; q < probabilities.size(); ++q) {
            const double probability = probabilities.f(q);
            const bool valid = std::isfinite(probability) && probability >= 0.0 && probability <= 1.0;
            if (!valid && p.structure_only) {
                p.payload_available_mask &= ~(1u << 1);
                continue;
            }
            require(valid, "INVALID_PARAMETER");
        }
    }
    p.output_kind = Kind::number;
    p.output_geometry_known = p.geometry(0) && axis_known && (!quantiles || p.geometry(1));
    p.scratch_doubles = 0;
    p.scratch_indices = 0;
    if (!p.output_geometry_known) return;
    p.output_shape = p.args[0].shape;
    if (quantiles) {
        const auto axis = static_cast<std::size_t>(p.args[axis_index].scalar);
        p.output_shape.dim[axis] = p.args[1].size();
        // Empty Q or any empty non-axis extent needs no line or sort scratch.
        if (p.output_shape.size() != 0)
            p.scratch_doubles = p.args[0].shape.dim[axis];
    }
}

void axis_path_execute(const Prepared &p, Output &out, Workspace &work, Isa isa, Audit &audit) {
    const auto size = out.shape.size();
    if (size == 0) return;
    const auto op = p.spec->op;
    const bool quantiles = op == Op::quantiles_axis;
    const auto &values = p.args[0];
    const auto axis = static_cast<std::size_t>(p.args[quantiles ? 2 : 1].scalar);
    const auto count = values.shape.dim[axis];
    if (quantiles) require(count != 0, "INSUFFICIENT_SAMPLE");
    std::size_t outer = 1, inner = 1;
    for (std::size_t dimension = 0; dimension < axis; ++dimension)
        outer = checked_product(outer, values.shape.dim[dimension]);
    for (int dimension = static_cast<int>(axis) + 1; dimension < values.shape.rank; ++dimension)
        inner = checked_product(inner, values.shape.dim[dimension]);
    const auto *source = static_cast<const double *>(values.data);
    auto *sorted = quantiles ? work.doubles.data() : nullptr;
    const auto outputs_per_line = quantiles ? p.args[1].size() : count;
    for (std::size_t before = 0; before < outer; ++before) {
        for (std::size_t after = 0; after < inner; ++after) {
            const auto base = line_offset(values, axis, before, after);
            if (quantiles) {
                for (std::size_t index = 0; index < count; ++index)
                    sorted[index] = source[base + static_cast<std::ptrdiff_t>(index) * values.stride[axis]];
                if (outputs_per_line != 1 ||
                    !axis_path_detail::partition_quantile_line(sorted, count, p.args[1].f(0), isa, &audit))
                    axis_path_detail::sort_numeric_line(sorted, count, isa, &audit);
                audit.algorithm_copy_bytes += count * sizeof(double);
                for (std::size_t q = 0; q < outputs_per_line; ++q)
                    out.set((before * outputs_per_line + q) * inner + after,
                            axis_path_detail::sorted_quantile(sorted, count, p.args[1].f(q)));
            } else {
                double state = p.args[2].scalar;
                for (std::size_t index = 0; index < count; ++index) {
                    const double current = source[base + static_cast<std::ptrdiff_t>(index) * values.stride[axis]];
                    state = op == Op::cumulative_product_axis
                        ? axis_path_detail::product_step(state, current)
                        : axis_path_detail::strict_max_step(state, current);
                    const auto position = (before * count + index) * inner + after;
                    if (out.order == MemoryOrder::c)
                        static_cast<double *>(out.data)[position] = state;
                    else out.set(position, state);
                }
            }
        }
    }
}

} // namespace calmetrics_engine::ops
