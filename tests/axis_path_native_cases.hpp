#pragma once

#include "calmetrics_engine/operators.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace axis_path_tests {
namespace o = calmetrics_engine::ops;

inline void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
inline bool ieee(double left, double right) {
    if (std::isnan(left) && std::isnan(right)) return true;
    return std::memcmp(&left, &right, sizeof(double)) == 0;
}
template<class Function>
inline void error_is(Function &&function, const char *code) {
    bool caught = false;
    try { function(); }
    catch (const o::Error &error) { caught = std::string(error.what()) == code; }
    check(caught, code);
}
constexpr double sentinel = -989.125;

struct Backing {
    std::vector<double> storage;
    o::Value value;
    Backing(o::Shape shape, const std::vector<double> &logical,
            o::MemoryOrder order = o::MemoryOrder::c, bool negative = false)
        : storage(shape.size() * 2 + 2, sentinel) {
        check(shape.size() == logical.size(), "axis fixture shape");
        value.shape = shape; value.set_contiguous_strides(order);
        for (auto &stride : value.stride) stride *= 2;
        value.data = shape.size() ? storage.data() + 1 : nullptr;
        if (negative && shape.size()) {
            const auto axis = shape.rank - 1;
            value.data = static_cast<const double *>(value.data) +
                static_cast<std::ptrdiff_t>(shape.dim[axis] - 1) * value.stride[axis];
            value.stride[axis] = -value.stride[axis];
        }
        for (std::size_t index = 0; index < logical.size(); ++index)
            const_cast<double *>(static_cast<const double *>(value.data))[value.offset(index)] = logical[index];
    }
};
struct Computed {
    o::Shape shape;
    o::MemoryOrder order;
    std::vector<double> storage;
    o::Audit audit;
    std::size_t capacity = 0;
    double logical(std::size_t index) const {
        o::Value view; view.shape = shape; view.data = storage.data() + 1;
        view.set_contiguous_strides(order);
        return view.f(index);
    }
};
inline Computed run(const char *name, const std::array<o::Value, 3> &args,
                    o::MemoryOrder order = o::MemoryOrder::c, o::Isa isa = o::Isa::scalar) {
    const auto prepared = o::prepare(o::lookup(name), args.data(), args.size());
    Computed result{prepared.output_shape, order,
                    std::vector<double>(prepared.output_shape.size() + 2, sentinel), {}, 0};
    o::Output out; out.shape = result.shape; out.order = order; out.data = result.storage.data() + 1;
    o::Workspace workspace;
    o::execute(prepared, out, workspace, isa, result.audit);
    result.capacity = workspace.capacity_bytes();
    check(result.storage.front() == sentinel && result.storage.back() == sentinel, "axis output canary");
    return result;
}
inline void same(const Computed &left, const Computed &right) {
    check(left.shape == right.shape, "axis output shape");
    for (std::size_t index = 0; index < left.shape.size(); ++index)
        check(ieee(left.logical(index), right.logical(index)), "axis output IEEE values");
}

inline void prefix_views_and_seeds() {
    for (const auto shape : {o::vector_shape(3), o::matrix_shape(2, 3), o::tensor_shape(2, 3, 2)}) {
        std::vector<double> logical(shape.size());
        for (std::size_t i = 0; i < logical.size(); ++i) logical[i] = static_cast<double>(i % 5) - 2.0;
        for (const auto order : {o::MemoryOrder::c, o::MemoryOrder::f})
            for (const bool negative : {false, true}) {
                Backing values(shape, logical, order, negative);
                const auto original = values.storage;
                for (int axis = 0; axis < shape.rank; ++axis)
                    for (const char *name : {"cumulative_product_axis", "cumulative_max_axis"}) {
                        const std::array<o::Value, 3> args{values.value, o::Value::number(axis), o::Value::number(-0.0)};
                        auto actual = run(name, args, order), later = run(name, args, order, o::Isa::automatic);
                        same(actual, later);
                        check(actual.audit.scratch_bytes == 0 && actual.capacity == 0 &&
                              actual.audit.algorithm_copy_bytes == 0, "prefix allocated scratch");
                        std::size_t inner = 1, outer = 1;
                        for (int d = axis + 1; d < shape.rank; ++d) inner *= shape.dim[d];
                        for (int d = 0; d < axis; ++d) outer *= shape.dim[d];
                        for (std::size_t before = 0; before < outer; ++before)
                            for (std::size_t after = 0; after < inner; ++after) {
                                double state = -0.0;
                                for (std::size_t n = 0; n < shape.dim[axis]; ++n) {
                                    const auto index = (before * shape.dim[axis] + n) * inner + after;
                                    const double value = values.value.f(index);
                                    if (std::string(name) == "cumulative_product_axis") state = state * value;
                                    else if (value > state) state = value;
                                    check(ieee(actual.logical(index), state), "prefix changed ordered recurrence");
                                }
                            }
                    }
                check(values.storage == original, "prefix wrote borrowed input or padding");
            }
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Backing values(o::vector_shape(5), {0.0, -0.0, nan, 1.0, -1.0});
    for (const double seed : {-0.0, 0.0, nan}) {
        auto result = run("cumulative_max_axis", {values.value, o::Value::number(0), o::Value::number(seed)});
        check(ieee(result.logical(0), seed) && ieee(result.logical(1), seed) &&
              ieee(result.logical(2), seed), "strict max seed/tie/NaN contract");
    }
    Backing multipliers(o::vector_shape(4), {-2.0, 0.5, -0.0, std::numeric_limits<double>::infinity()});
    auto product = run("cumulative_product_axis",
                       {multipliers.value, o::Value::number(0), o::Value::number(3.0)});
    check(ieee(product.logical(0), -6.0) && ieee(product.logical(1), -3.0) &&
          ieee(product.logical(2), 0.0) && std::isnan(product.logical(3)), "product explicit seed/IEEE");
    Backing repeated(o::matrix_shape(2, 2), {2, 3, 4, 5});
    repeated.value.shape = o::tensor_shape(2, 3, 2);
    repeated.value.stride = {4, 0, 2};
    auto repeated_product = run("cumulative_product_axis",
        {repeated.value, o::Value::number(1), o::Value::number(1.0)}, o::MemoryOrder::f);
    check(ieee(repeated_product.logical(4), 8.0) && ieee(repeated_product.logical(11), 125.0),
          "zero-stride independent lines");
}

inline void quantiles_views_and_ieee() {
    Backing probabilities(o::vector_shape(5), {1.0, 0.25, 0.0, 0.5, 0.5}, o::MemoryOrder::c, true);
    const auto saved_q = probabilities.storage;
    for (const auto shape : {o::vector_shape(3), o::matrix_shape(2, 3), o::tensor_shape(2, 3, 2)}) {
        std::vector<double> logical(shape.size());
        for (std::size_t i = 0; i < logical.size(); ++i) logical[i] = static_cast<double>((i * 7) % 13) - 6.0;
        for (const auto order : {o::MemoryOrder::c, o::MemoryOrder::f})
            for (const bool negative : {false, true}) {
                Backing values(shape, logical, order, negative);
                const auto original = values.storage;
                for (int axis = 0; axis < shape.rank; ++axis) {
                    const std::array<o::Value, 3> args{values.value, probabilities.value, o::Value::number(axis)};
                    auto actual = run("quantiles_axis", args, order), later = run("quantiles_axis", args, order);
                    same(actual, later);
                    check(actual.shape.dim[axis] == 5, "quantile Q axis replacement");
                    check(actual.audit.scratch_bytes == shape.dim[axis] * sizeof(double) &&
                          actual.capacity == shape.dim[axis] * sizeof(double), "quantiles copied Q full lines");
                    check(actual.audit.algorithm_copy_bytes == shape.size() * sizeof(double), "quantile line copy audit");
                    std::size_t inner = 1, outer = 1;
                    for (int d = axis + 1; d < shape.rank; ++d) inner *= shape.dim[d];
                    for (int d = 0; d < axis; ++d) outer *= shape.dim[d];
                    for (std::size_t before = 0; before < outer; ++before)
                        for (std::size_t after = 0; after < inner; ++after) {
                            std::vector<double> ordered(shape.dim[axis]);
                            for (std::size_t n = 0; n < ordered.size(); ++n)
                                ordered[n] = values.value.f((before * ordered.size() + n) * inner + after);
                            std::sort(ordered.begin(), ordered.end());
                            for (std::size_t q = 0; q < probabilities.value.size(); ++q) {
                                const double position = (ordered.size() - 1) * probabilities.value.f(q);
                                const auto lo = static_cast<std::size_t>(std::floor(position));
                                const auto hi = static_cast<std::size_t>(std::ceil(position));
                                const double f = position - static_cast<double>(lo);
                                const double expected = lo == hi ? ordered[lo] :
                                    ordered[lo] * (1.0 - f) + ordered[hi] * f;
                                check(ieee(actual.logical((before * 5 + q) * inner + after), expected),
                                      "quantile interpolation tree");
                            }
                        }
                }
                check(values.storage == original && probabilities.storage == saved_q, "sort mutated inputs or padding");
            }
    }
    const double inf = std::numeric_limits<double>::infinity(), nan = std::numeric_limits<double>::quiet_NaN();
    Backing values(o::vector_shape(3), {inf, 0.0, inf});
    Backing q(o::vector_shape(3), {0.0, 0.5, 1.0});
    auto ranks = run("quantiles_axis", {values.value, q.value, o::Value::number(0)});
    check(ieee(ranks.logical(0), 0.0) && ranks.logical(1) == inf && ranks.logical(2) == inf,
          "integer quantile rank multiplied infinity by zero");
    Backing mixed(o::vector_shape(3), {nan, -inf, inf});
    auto sorted = run("quantiles_axis", {mixed.value, q.value, o::Value::number(0)});
    check(sorted.logical(0) == -inf && sorted.logical(1) == inf && std::isnan(sorted.logical(2)),
          "quantile NaN-last ordering");
    Backing singleton(o::vector_shape(1), {inf}), arbitrary_q(o::vector_shape(2), {0.125, 0.9});
    auto single = run("quantiles_axis", {singleton.value, arbitrary_q.value, o::Value::number(0)});
    check(single.logical(0) == inf && single.logical(1) == inf, "singleton quantile did interpolation");
}

inline void empty_and_partial_constraints() {
    Backing no_values(o::vector_shape(0), {}), no_q(o::vector_shape(0), {});
    for (const char *name : {"cumulative_product_axis", "cumulative_max_axis"}) {
        auto empty = run(name, {no_values.value, o::Value::number(0), o::Value::number(1)});
        check(empty.shape.size() == 0 && empty.capacity == 0, "empty prefix scratch");
    }
    Backing values(o::matrix_shape(2, 0), {}), q(o::vector_shape(1), {0.5});
    auto empty_q = run("quantiles_axis", {values.value, no_q.value, o::Value::number(1)});
    check(empty_q.shape.size() == 0 && empty_q.capacity == 0, "Q0 quantile scratch");
    error_is([&] { (void)run("quantiles_axis", {values.value, q.value, o::Value::number(1)}); }, "INSUFFICIENT_SAMPLE");
    Backing no_outer(o::matrix_shape(0, 3), {}), invalid_q(o::vector_shape(2), {0.5, 1.1});
    error_is([&] { (void)run("quantiles_axis", {no_outer.value, invalid_q.value, o::Value::number(1)}); }, "INVALID_PARAMETER");
    auto empty_outer = run("quantiles_axis", {no_outer.value, q.value, o::Value::number(1)});
    check(empty_outer.shape.size() == 0 && empty_outer.capacity == 0, "empty outer quantile scratch");
    o::Value unknown = o::Value::number(0); unknown.shape.rank = -1;
    std::array<o::Value, 3> partial{unknown, invalid_q.value, o::Value::number(1)};
    const auto structure = o::validate_structure(o::lookup("quantiles_axis"), partial.data(), 3, 6, 6);
    check(!structure.geometry_known, "unknown input geometry became known");
    o::Prepared prepared(3); prepared.spec = &o::lookup("quantiles_axis"); prepared.count = 3;
    prepared.geometry_known_mask = 6; prepared.payload_available_mask = 6; prepared.structure_only = true;
    for (std::size_t i = 0; i < partial.size(); ++i) prepared.args[i] = partial[i];
    o::axis_path_prepare(prepared);
    check(!prepared.payload(1), "invalid structural-only probability remained readable");
    partial[1].shape = o::matrix_shape(1, 2);
    error_is([&] { (void)o::validate_structure(o::lookup("quantiles_axis"), partial.data(), 3, 6, 6); }, "RANK_MISMATCH");
    partial = {unknown, o::Value::number(1), q.value};
    error_is([&] { (void)o::validate_structure(o::lookup("cumulative_product_axis"), partial.data(), 3, 6, 6); }, "RANK_MISMATCH");
    partial[2] = o::Value::number(1); partial[2].kind = o::Kind::integer;
    error_is([&] { (void)o::validate_structure(o::lookup("cumulative_max_axis"), partial.data(), 3, 6, 6); }, "DTYPE_MISMATCH");
}
} // namespace axis_path_tests

inline void test_axis_path_contracts() {
    axis_path_tests::prefix_views_and_seeds();
    axis_path_tests::quantiles_views_and_ieee();
    axis_path_tests::empty_and_partial_constraints();
}
