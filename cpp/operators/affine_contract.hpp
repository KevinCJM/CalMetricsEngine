#pragma once

#include "calmetrics_engine/operators.hpp"
#include <cstddef>

namespace calmetrics_engine::ops {
namespace affine_contract_detail {

// Reusable by independent scalar or fixed-width native lanes. Each lane starts
// at its explicit initial value and keeps ascending-factor multiply/add order.
// Translation units must retain the engine's no-contraction compile policy.
template <class Accumulator, class CoefficientAt, class ValueAt>
inline Accumulator ordered_fold(std::size_t factors, Accumulator accumulator,
                                const CoefficientAt &coefficient_at,
                                const ValueAt &value_at) {
    for (std::size_t factor = 0; factor < factors; ++factor) {
        const Accumulator product = coefficient_at(factor) * value_at(factor);
        accumulator += product;
    }
    return accumulator;
}

} // namespace affine_contract_detail

// prepare() owns rank/type/shape validation; the boundary owns stride/owner and
// alias validation. Payload IEEE values propagate without an extra finite scan.
inline void affine_contract_execute(const Prepared &prepared, Output &out,
                                    Isa requested, Audit &audit) {
    (void)requested;
    (void)audit; // Portable baseline: no library calls, copies or scratch.
    const auto size = out.shape.size();
    if (size == 0) return;

    const auto &values = prepared.args[0];
    const auto &coefficients = prepared.args[1];
    const auto &initial = prepared.args[2];
    const auto assets = coefficients.shape.dim[1];
    const auto factor_axis = values.shape.rank - 1;
    const auto factors = values.shape.dim[factor_axis];
    const auto rows = size / assets;
    const auto *input = static_cast<const double *>(values.data);

    for (std::size_t row = 0; row < rows; ++row) {
        // An empty factor axis may have null data and arbitrary empty-array
        // strides. Do not perform any input offset arithmetic in that case.
        std::ptrdiff_t base = 0;
        if (factors != 0) {
            if (values.shape.rank == 2)
                base = static_cast<std::ptrdiff_t>(row) * values.stride[0];
            else {
                const auto paths = values.shape.dim[1];
                base = static_cast<std::ptrdiff_t>(row / paths) * values.stride[0] +
                       static_cast<std::ptrdiff_t>(row % paths) * values.stride[1];
            }
        }
        for (std::size_t asset = 0; asset < assets; ++asset) {
            const auto coefficient_at = [&](std::size_t factor) noexcept {
                return coefficients.at(factor, asset);
            };
            const auto value_at = [&](std::size_t factor) noexcept {
                return input[base + static_cast<std::ptrdiff_t>(factor) *
                                        values.stride[factor_axis]];
            };
            const auto result = affine_contract_detail::ordered_fold(
                factors, initial.f(asset), coefficient_at, value_at);
            // Logical C-flat coordinates also address contiguous F outputs.
            out.set(row * assets + asset, result);
        }
    }
}

} // namespace calmetrics_engine::ops
