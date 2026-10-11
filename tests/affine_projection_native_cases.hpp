#pragma once

#include "calmetrics_engine/affine_projection.hpp"
#include "calmetrics_engine/compiler.hpp"
#include "calmetrics_engine/native_runtime.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace affine_projection_tests {
namespace c = calmetrics_engine::compiler;
namespace g = calmetrics_engine::graph;
namespace o = calmetrics_engine::ops;
namespace t = calmetrics_engine::typed;

inline void check(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
inline bool equal_ieee(double actual, double expected) {
  if (std::isnan(actual) && std::isnan(expected)) return true;
  std::uint64_t a = 0, b = 0;
  std::memcpy(&a, &actual, sizeof(a));
  std::memcpy(&b, &expected, sizeof(b));
  return a == b;
}
template<class Function>
inline void error_is(Function &&function, const char *expected) {
  bool caught = false;
  try { function(); }
  catch (const o::Error &error) { caught = std::string(error.what()) == expected; }
  check(caught, expected);
}

struct Array {
  std::vector<double> storage;
  o::Value value;
  Array(o::Shape shape, const std::vector<double> &logical,
        o::MemoryOrder order = o::MemoryOrder::c, bool reverse_last = false)
      : storage(shape.size()) {
    check(logical.size() == storage.size(), "fixture shape");
    value.shape = shape;
    value.set_contiguous_strides(order);
    value.data = storage.empty() ? nullptr : storage.data();
    if (reverse_last && shape.rank && shape.dim[shape.rank - 1] && !storage.empty()) {
      const auto axis = shape.rank - 1;
      value.data = storage.data() +
          static_cast<std::ptrdiff_t>(shape.dim[axis] - 1) * value.stride[axis];
      value.stride[axis] = -value.stride[axis];
    }
    for (std::size_t i = 0; i < logical.size(); ++i)
      static_cast<double *>(const_cast<void *>(value.data))[value.offset(i)] = logical[i];
  }
};

inline std::vector<t::Variable> variables(bool tensor = true) {
  auto asset = t::ValueType::vector("A");
  return {{"x", tensor ? t::ValueType::tensor({"time", "path", "factor"}, {"T", "P", "F"})
                        : t::ValueType::matrix({"time", "factor"}, {"T", "F"})},
          {"k", t::ValueType::matrix({"factor", "asset"}, {"F", "A"})},
          {"i", asset}, {"w", asset}, {"d", t::ValueType::scalar()}};
}
inline const std::string &returns_expression() {
  static const std::string expression = "expm1(affine_contract(x,k,i))";
  return expression;
}
inline std::vector<std::string> roots(bool contributions = true) {
  const auto &r = returns_expression();
  std::vector<std::string> result{
      "swap_axes(sum_axis(multiply(broadcast_like(w," + r + ")," + r + "),2),0,1)",
      "sum_axis(multiply(broadcast_like(w," + r + ")," + r + "),2)"};
  if (contributions)
    result.push_back("sum(multiply(first(slice_axis(w,0,0,1,1)),slice_axis(" + r + ",2,0,1,1)))/d");
  return result;
}
inline std::shared_ptr<c::CompiledGraph> compile(const std::vector<std::string> &expressions,
    o::MemoryOrder order = o::MemoryOrder::c, bool isolate = false, bool tensor = true) {
  auto declarations = variables(tensor);
  if (std::none_of(expressions.begin(), expressions.end(), [](const std::string &expression) {
        return expression.find("/d") != std::string::npos;
      }))
    declarations.erase(std::remove_if(declarations.begin(), declarations.end(),
        [](const t::Variable &variable) { return variable.name == "d"; }), declarations.end());
  return c::compile(expressions, declarations, isolate, {}, {}, 0, 100000000,
                    true, {}, order == o::MemoryOrder::f ? g::OutputOrder::f : g::OutputOrder::c);
}
inline std::vector<o::Value> inputs(const c::CompiledGraph &compiled,
    const Array &x, const Array &k, const Array &initial, const Array &weights) {
  std::vector<o::Value> result;
  for (const auto &name : compiled.input_names) {
    if (name == "x") result.push_back(x.value);
    else if (name == "k") result.push_back(k.value);
    else if (name == "i") result.push_back(initial.value);
    else if (name == "w") result.push_back(weights.value);
    else throw std::runtime_error("unexpected input fixture");
  }
  return result;
}
inline std::vector<double> parameters(const c::CompiledGraph &compiled, double divisor) {
  std::vector<double> result;
  for (const auto &name : compiled.parameter_names) {
    check(name == "d", "unexpected parameter fixture");
    result.push_back(divisor);
  }
  return result;
}

constexpr double sentinel = -961.125;
struct Result {
  g::ResultLayout layout;
  std::vector<double> storage;
  g::Audit audit;
  explicit Result(g::ResultLayout given)
      : layout(std::move(given)), storage((layout.bytes() + 7) / 8 + 2, sentinel) {}
  void *data() { return storage.data() + 1; }
  const void *data() const { return storage.data() + 1; }
  void canaries() const {
    check(storage.front() == sentinel && storage.back() == sentinel, "root writer canaries");
  }
  o::Value root(std::size_t row, std::size_t column) const {
    o::Value result;
    result.shape = audit.result_shapes.at(row * layout.columns + column);
    const auto &slot = layout.slots.at(row * layout.columns + column);
    const auto *pointer = static_cast<const std::uint8_t *>(data()) + slot.byte_offset;
    if (result.shape.rank) {
      result.data = pointer;
      result.set_contiguous_strides(layout.order(column));
    } else std::memcpy(&result.scalar, pointer, sizeof(result.scalar));
    return result;
  }
};
inline void compare(const Result &actual, const Result &expected) {
  actual.canaries(); expected.canaries();
  check(actual.audit.result_shapes == expected.audit.result_shapes, "private root geometry");
  check(actual.audit.root_statuses == expected.audit.root_statuses, "private root statuses");
  for (std::size_t row = 0; row < actual.audit.rows; ++row)
    for (std::size_t column = 0; column < actual.layout.columns; ++column) {
      const auto a = actual.root(row, column), b = expected.root(row, column);
      for (std::size_t i = 0; i < a.size(); ++i)
        check(equal_ieee(a.f(i), b.f(i)), "private/canonical logical bits");
    }
}
inline Result canonical(const g::Program &program, const std::vector<o::Value> &in,
    const std::vector<double> &params, const std::vector<std::int64_t> &starts,
    const std::vector<std::int64_t> &ends) {
  Result result(g::result_layout(program, in, starts.data(), ends.data(), starts.size()));
  const g::ValueObserver observer = [](std::optional<o::Op>, const o::Value &, bool) {};
  result.audit = g::execute_observed(program, in, params.data(), params.size(), starts.data(),
      ends.data(), starts.size(), result.data(), program.roots.size(), &result.layout, observer);
  return result;
}
inline Result private_run(const g::Program &program, const std::vector<o::Value> &in,
    const std::vector<double> &params, const std::vector<std::int64_t> &starts,
    const std::vector<std::int64_t> &ends, o::Workspace &workspace) {
  Result result(g::result_layout(program, in, starts.data(), ends.data(), starts.size()));
  check(program.execution_metadata && program.execution_metadata->affine_projection,
        "complete consumers were not selected");
  check(g::execute_affine_projection(program, *program.execution_metadata->affine_projection,
      in, params.data(), params.size(), starts.data(), ends.data(), starts.size(), result.data(),
      &result.layout, 0, workspace, result.audit, std::chrono::steady_clock::time_point::max(), nullptr),
      "supported geometry declined private execution");
  return result;
}

inline void direct_writers() {
  for (const auto order : {o::MemoryOrder::c, o::MemoryOrder::f})
    for (const bool tensor : {false, true}) {
      const auto shape = tensor ? o::tensor_shape(2, 2, 2) : o::matrix_shape(4, 2);
      Array x(shape, {1, 2, 3, 4, -1, -2, 5, 6}, order, true);
      Array coefficients(o::matrix_shape(2, 3), {2, 0, 1, 3, -1, 4}, order, true);
      Array initial(o::vector_shape(3), {5, -0.0, -2});
      const auto saved_x = x.storage, saved_k = coefficients.storage;
      const std::array<o::Value, 3> args{x.value, coefficients.value, initial.value};
      const auto prepared = o::prepare(o::lookup("affine_contract"), args.data(), args.size());
      std::vector<double> target(prepared.output_shape.size() + 2, sentinel);
      o::Output output;
      output.shape = prepared.output_shape; output.order = order; output.data = target.data() + 1;
      o::Workspace workspace; o::Audit audit;
      o::execute(prepared, output, workspace, o::Isa::scalar, audit);
      check(target.front() == sentinel && target.back() == sentinel, "direct affine canaries");
      o::Value view;
      view.shape = output.shape; view.data = output.data; view.set_contiguous_strides(order);
      const double expected[]{13, -2, 7, 23, -4, 17, -3, 2, -11, 33, -6, 27};
      for (std::size_t i = 0; i < view.size(); ++i)
        check(equal_ieee(view.f(i), expected[i]), "direct affine independent exact oracle");
      check(x.storage == saved_x && coefficients.storage == saved_k, "direct inputs changed");
      check(workspace.capacity_bytes() == 0, "direct affine allocated scratch");

      const std::array<o::Value, 1> unary{view};
      const auto nonlinear = o::prepare(o::lookup("expm1"), unary.data(), unary.size());
      std::vector<double> independent(nonlinear.output_shape.size() + 2, sentinel);
      output.data = independent.data() + 1;
      o::execute(nonlinear, output, workspace, o::Isa::scalar, audit);
      check(independent.front() == sentinel && independent.back() == sentinel, "direct expm1 canaries");
      view.data = independent.data() + 1;
      for (std::size_t i = 0; i < view.size(); ++i)
        check(equal_ieee(view.f(i), std::expm1(expected[i])), "direct expm1 logical order");
      check(target.front() == sentinel && target.back() == sentinel, "prior result changed");
    }
  Array empty_x(o::matrix_shape(3, 0), {}), empty_k(o::matrix_shape(0, 2), {});
  const std::array<o::Value, 3> empty_args{empty_x.value, empty_k.value, o::Value::number(-0.0)};
  const auto empty = o::prepare(o::lookup("affine_contract"), empty_args.data(), empty_args.size());
  std::vector<double> target(empty.output_shape.size() + 2, sentinel);
  o::Output output; output.shape = empty.output_shape; output.data = target.data() + 1;
  o::Workspace workspace; o::Audit audit;
  o::execute(empty, output, workspace, o::Isa::scalar, audit);
  for (std::size_t i = 1; i + 1 < target.size(); ++i)
    check(equal_ieee(target[i], -0.0), "empty factor signed initial");
  check(target.front() == sentinel && target.back() == sentinel, "empty factor canaries");
  const std::array<o::Value, 1> zero{o::Value::number(-0.0)};
  const auto scalar = o::prepare(o::lookup("expm1"), zero.data(), zero.size());
  o::Output scalar_out; o::execute(scalar, scalar_out, workspace, o::Isa::scalar, audit);
  check(equal_ieee(scalar_out.scalar, -0.0), "expm1 scalar negative zero");
}

inline void complete_chain() {
  for (const auto order : {o::MemoryOrder::c, o::MemoryOrder::f}) {
    auto compiled = compile(roots(), order);
    Array x(o::tensor_shape(5, 2, 2), {0, .125, .25, .375, .5, .625, .75, .875,
        -.5, -.375, -.25, -.125, 1, 1.125, 1.25, 1.375, -.75, -.625, -.5, -.375}, order, true);
    Array k(o::matrix_shape(2, 3), {.25, -.125, .5, -.5, .25, -.375}, order, true);
    Array initial(o::vector_shape(3), {0, .125, -.25});
    Array w(o::vector_shape(3), {.25, -.5, 1});
    const auto in = inputs(*compiled, x, k, initial, w);
    const auto saved_x = x.storage, saved_k = k.storage;
    const auto params = parameters(*compiled, 2);
    const std::vector<std::int64_t> starts{1, 3}, ends{3, 5};
    o::Workspace workspace;
    auto actual = private_run(compiled->program, in, params, starts, ends, workspace);
    compare(actual, canonical(compiled->program, in, params, starts, ends));
    check(x.storage == saved_x && k.storage == saved_k, "projection changed input backing");
    check(actual.audit.numeric_arena_bytes == 0 && actual.audit.root_copy_bytes == 0,
          "projection materialized full intermediates");
    check(actual.layout.order(0) == order && actual.layout.order(1) == order, "explicit root order");
    const auto saved_result = actual.storage;
    auto *changed = const_cast<double *>(static_cast<const double *>(x.value.data)) + x.value.offset(4);
    *changed = .75;
    const auto mutated_x = x.storage;
    auto later = private_run(compiled->program, in, params, starts, ends, workspace);
    compare(later, canonical(compiled->program, in, params, starts, ends));
    check(actual.storage == saved_result, "later execution changed independent result");
    check(x.storage == mutated_x, "later execution changed source backing");
    check(k.storage == saved_k, "private execution changed coefficients");
    std::copy(saved_x.begin(), saved_x.end(), x.storage.begin());
    check(x.storage == saved_x, "private execution changed source");
    const auto encoded = c::encode_program(compiled->program, &actual.layout.orders);
    const auto decoded = c::decode_program(encoded);
    check(bool(decoded.execution_metadata->affine_projection), "worker decode lost proof");
    compare(private_run(decoded, in, params, starts, ends, workspace), actual);
  }

  // Cancellation occurs across the 4096-row tile boundary. A single left fold
  // differs from combining tile subtotals; do not impose another libm's tail.
  auto compiled = compile(roots());
  std::vector<double> tape(4098, 1e-16);
  tape[0] = 2.5e-17;
  tape[4096] = std::log(2.0);
  tape[4097] = -std::numeric_limits<double>::infinity();
  Array x(o::tensor_shape(4098, 1, 1), tape), k(o::matrix_shape(1, 1), {1});
  Array initial(o::vector_shape(1), {0}), w(o::vector_shape(1), {1e16});
  auto in = inputs(*compiled, x, k, initial, w);
  auto params = parameters(*compiled, 1);
  o::Workspace workspace;
  auto actual = private_run(compiled->program, in, params, {0}, {4098}, workspace);
  compare(actual, canonical(compiled->program, in, params, {0}, {4098}));
  check(actual.audit.pointwise_tiles == 2, "cross-tile fixture did not cross tile");
  double ordered = 0, first_tile = 0, second_tile = 0;
  for (std::size_t i = 0; i < tape.size(); ++i) {
    const double product = 1e16 * std::expm1(tape[i]);
    ordered += product;
    if (i < 4096) first_tile += product;
    else second_tile += product;
  }
  check(!equal_ieee(ordered, first_tile + second_tile), "cancellation fixture does not distinguish regrouping");
  check(equal_ieee(actual.root(0, 2).scalar, ordered), "contribution regrouped across tiles");
  const auto bound = g::affine_projection_workspace_bytes(compiled->program,
      *compiled->program.execution_metadata->affine_projection, in, 4098);
  check(bound && *bound <= 1048576 + 24 +
      compiled->program.nodes.size() * (sizeof(o::Value) + 1), "projection workspace unbounded");
}

inline void empty_and_nonfinite() {
  for (const auto order : {o::MemoryOrder::c, o::MemoryOrder::f}) {
    auto compiled = compile(roots(), order);
    Array x(o::tensor_shape(2, 3, 0), {}), k(o::matrix_shape(0, 2), {});
    Array initial(o::vector_shape(2), {-.0, .125}), w(o::vector_shape(2), {1, -.5});
    const auto in = inputs(*compiled, x, k, initial, w);
    const auto params = parameters(*compiled, 3);
    o::Workspace workspace;
    compare(private_run(compiled->program, in, params, {0}, {2}, workspace),
            canonical(compiled->program, in, params, {0}, {2}));

    auto period_only = compile(roots(false), order);
    Array asset_x(o::tensor_shape(2, 3, 1), {1, 2, 3, 4, 5, 6});
    Array asset_k(o::matrix_shape(1, 0), {}), no_initial(o::vector_shape(0), {}), no_w(o::vector_shape(0), {});
    const auto asset_in = inputs(*period_only, asset_x, asset_k, no_initial, no_w);
    auto no_assets = private_run(period_only->program, asset_in, {}, {0}, {2}, workspace);
    compare(no_assets, canonical(period_only->program, asset_in, {}, {0}, {2}));
    for (std::size_t column = 0; column < 2; ++column)
      for (std::size_t i = 0; i < no_assets.root(0, column).size(); ++i)
        check(equal_ieee(no_assets.root(0, column).f(i), 0.0), "empty asset sum identity");
    check(workspace.capacity_bytes() == 0, "empty asset retained prior tile");

    Array no_paths(o::tensor_shape(2, 0, 1), {}), one_k(o::matrix_shape(1, 1), {1});
    Array one_initial(o::vector_shape(1), {0}), one_w(o::vector_shape(1), {1});
    const auto path_in = inputs(*period_only, no_paths, one_k, one_initial, one_w);
    compare(private_run(period_only->program, path_in, {}, {0}, {2}, workspace),
            canonical(period_only->program, path_in, {}, {0}, {2}));
    const auto contribution_in = inputs(*compiled, no_paths, one_k, one_initial, one_w);
    error_is([&] { (void)private_run(compiled->program, contribution_in,
        parameters(*compiled, 0), {0}, {2}, workspace); }, "INSUFFICIENT_SAMPLE");

    Array overflow(o::tensor_shape(1, 1, 1), {710}), zero_w(o::vector_shape(1), {0});
    const auto overflow_in = inputs(*compiled, overflow, one_k, one_initial, zero_w);
    auto result = private_run(compiled->program, overflow_in, parameters(*compiled, 1), {0}, {1}, workspace);
    compare(result, canonical(compiled->program, overflow_in, parameters(*compiled, 1), {0}, {1}));
    check(std::isnan(result.root(0, 0).f(0)) && std::isnan(result.root(0, 2).scalar),
          "zero weight skipped non-finite input");
    error_is([&] { (void)private_run(compiled->program, overflow_in,
        parameters(*compiled, 0), {0}, {1}, workspace); }, "DIVIDE_BY_ZERO");
  }
}

// A=17 exercises two complete eight-asset packets and a scalar tail;
// transpose(k) requires a real pack. The test-only expectation makes ON/OFF
// native and sanitizer runs prove their selected branch without a new API.
inline void eigen_packed_intervals() {
  struct RoundingGuard {
    int original = std::fegetround();
    RoundingGuard() { check(std::fesetround(FE_TONEAREST) == 0, "set nearest rounding"); }
    ~RoundingGuard() { (void)std::fesetround(original); }
  } rounding;
  constexpr std::size_t times = 4102, paths = 2, factors = 4, assets = 17;
  constexpr std::size_t pack_bytes = factors * assets * sizeof(double);
  static_assert(pack_bytes == 544, "A17/F4 pack fixture");
  const char *expectation = std::getenv("CALMETRICS_TEST_EXPECT_EIGEN");
  const std::string expected_branch = expectation ? expectation : "";
  check(expected_branch.empty() || expected_branch == "0" || expected_branch == "1",
        "CALMETRICS_TEST_EXPECT_EIGEN must be 0 or 1");
  const auto branch_check = [&](const g::Audit &audit, std::size_t selected_rows) {
    const auto expected_vector = 16 * selected_rows;
    check(audit.vector_elements == 0 || audit.vector_elements == expected_vector,
          "A17 packet/tail counter");
    if (expected_branch == "1")
      check(audit.vector_elements == expected_vector, "expected Eigen packets were not executed");
    if (expected_branch == "0")
      check(audit.vector_elements == 0, "Eigen OFF run executed packets");
    check(audit.algorithm_copy_bytes == (audit.vector_elements ? pack_bytes : 0),
          "transpose pack counter or per-execution lifetime");
  };
  const std::string r = "expm1(affine_contract(x,transpose(k),i))";
  const std::vector<std::string> expressions{
      "swap_axes(sum_axis(multiply(broadcast_like(w," + r + ")," + r + "),2),0,1)",
      "sum_axis(multiply(broadcast_like(w," + r + ")," + r + "),2)",
      "sum(multiply(first(slice_axis(w,0,0,1,1)),slice_axis(" + r + ",2,0,1,1)))/d",
      "sum(multiply(first(slice_axis(w,0,16,17,1)),slice_axis(" + r + ",2,16,17,1)))/d"};
  auto declarations = variables();
  declarations[1] = {"k", t::ValueType::matrix({"asset", "factor"}, {"A", "F"})};
  auto compiled = c::compile(expressions, declarations, false, {}, {}, 0, 100000000,
                             true, {}, g::OutputOrder::f);
  check(compiled->program.execution_metadata &&
        compiled->program.execution_metadata->affine_projection, "transpose proof missing");

  std::vector<double> tape(times * paths * factors, 0.0);
  for (std::size_t row = 0; row < times * paths; ++row) {
    tape[row * factors] = 1e-16;
    for (std::size_t factor = 1; factor < factors; ++factor)
      tape[row * factors + factor] =
          static_cast<double>(static_cast<int>((row + factor) % 7) - 3) / 128.0;
  }
  // The long interval has 8198 rows and crosses two 4096-row boundaries.
  tape[3 * paths * factors] = 2.5e-17;
  tape[((times - 1) * paths) * factors] = std::log(2.0);
  tape[((times - 1) * paths + 1) * factors] = -710.0;
  std::vector<double> beta(assets * factors), seed(assets), weights(assets, 1.0 / 32.0);
  for (std::size_t asset = 0; asset < assets; ++asset) {
    seed[asset] = static_cast<double>(static_cast<int>(asset % 5) - 2) / 64.0;
    for (std::size_t factor = 0; factor < factors; ++factor)
      beta[asset * factors + factor] =
          static_cast<double>(static_cast<int>((asset + factor) % 7) - 3) / 64.0;
  }
  seed[0] = 0.0; weights[0] = 1e16;
  for (std::size_t factor = 0; factor < factors; ++factor)
    beta[factor] = factor == 0 ? 1.0 : 0.0;
  Array x(o::tensor_shape(times, paths, factors), tape, o::MemoryOrder::f, true);
  Array k(o::matrix_shape(assets, factors), beta, o::MemoryOrder::c, true);
  Array initial(o::vector_shape(assets), seed), w(o::vector_shape(assets), weights);
  const auto saved_x = x.storage, saved_k = k.storage, saved_initial = initial.storage;
  const auto in = inputs(*compiled, x, k, initial, w);
  const auto params = parameters(*compiled, static_cast<double>(paths));
  const std::vector<std::int64_t> starts{1, 3}, ends{3, times};
  constexpr std::size_t selected_rows = (2 + times - 3) * paths;
  constexpr std::size_t scratch_doubles = 4096 * assets + 3 * 2 + factors * assets;
  o::Workspace workspace;
  auto audit_check = [&](const Result &result) {
    branch_check(result.audit, selected_rows);
    check(result.audit.pointwise_tiles == 4, "unequal intervals did not cross tiles");
    check(result.audit.numeric_arena_bytes == 0 && result.audit.root_copy_bytes == 0,
          "packed projection materialized full intermediates");
    check(workspace.capacity_bytes() == scratch_doubles * sizeof(double), "packed scratch reserve");
    const auto budget = g::affine_projection_workspace_bytes(compiled->program,
        *compiled->program.execution_metadata->affine_projection, in, times - 3);
    check(budget && result.audit.operator_workspace_capacity_bytes <= *budget,
          "packed execution escaped Planner bound");
    check(result.layout.order(0) == o::MemoryOrder::f && result.layout.order(1) == o::MemoryOrder::f,
          "packed F root order");
  };
  auto actual = private_run(compiled->program, in, params, starts, ends, workspace);
  compare(actual, canonical(compiled->program, in, params, starts, ends));
  audit_check(actual);
  check(x.storage == saved_x && k.storage == saved_k && initial.storage == saved_initial,
        "packing changed borrowed inputs");
  const auto retained = actual.storage;

  double ordered = 0.0, grouped = 0.0, tile_total = 0.0;
  for (std::size_t row = 0; row < (times - 3) * paths; ++row) {
    const double product = 1e16 * std::expm1(tape[(3 * paths + row) * factors]);
    ordered += product; tile_total += product;
    if ((row + 1) % 4096 == 0 || row + 1 == (times - 3) * paths) {
      grouped += tile_total; tile_total = 0.0;
    }
  }
  check(!equal_ieee(ordered, grouped), "packed cancellation fixture does not distinguish regrouping");
  check(equal_ieee(actual.root(1, 2).scalar, ordered / static_cast<double>(paths)),
        "packed contribution regrouped across tiles");

  // Maximum workspace belongs to the second interval above and the first below.
  // A pack at maximum-reserve must not overlap either interval's tile/states.
  auto reversed = private_run(compiled->program, in, params, {3, 1}, {times, 3}, workspace);
  compare(reversed, canonical(compiled->program, in, params, {3, 1}, {times, 3}));
  audit_check(reversed);
  check(actual.storage == retained, "packed interval reorder changed retained result");

  Result guarded(g::result_layout(compiled->program, in, starts.data(), ends.data(), starts.size()));
  std::atomic<bool> cancelled{true};
  g::Audit cancelled_audit;
  bool caught = false;
  try {
    (void)g::execute_affine_projection(compiled->program,
        *compiled->program.execution_metadata->affine_projection, in, params.data(), params.size(),
        starts.data(), ends.data(), starts.size(), guarded.data(), &guarded.layout, 0,
        workspace, cancelled_audit, std::chrono::steady_clock::time_point::max(), &cancelled);
  } catch (const calmetrics_engine::native::Timeout &error) {
    caught = std::string(error.what()) == "native affine projection cancelled";
  }
  check(caught, "packed projection ignored cancellation");
  check(std::all_of(guarded.storage.begin(), guarded.storage.end(),
      [](double value) { return value == sentinel; }), "cancelled execution wrote root storage");
  guarded.canaries();
  cancelled.store(false);

  // A post-pack arithmetic failure must not make a later call reuse old K.
  auto zero_divisor = parameters(*compiled, 0.0);
  error_is([&] {
    (void)g::execute_affine_projection(compiled->program,
        *compiled->program.execution_metadata->affine_projection, in,
        zero_divisor.data(), zero_divisor.size(), starts.data(), ends.data(), starts.size(),
        guarded.data(), &guarded.layout, 0, workspace, cancelled_audit,
        std::chrono::steady_clock::time_point::max(), &cancelled);
  }, "DIVIDE_BY_ZERO");
  guarded.canaries();
  check(cancelled_audit.algorithm_copy_bytes == actual.audit.algorithm_copy_bytes,
        "failure did not reach the admitted producer");
  auto *changed = const_cast<double *>(static_cast<const double *>(k.value.data)) +
      k.value.offset(16 * factors + 2);
  *changed = 0.25;
  const auto mutated_k = k.storage;
  auto recovered = private_run(compiled->program, in, params, starts, ends, workspace);
  compare(recovered, canonical(compiled->program, in, params, starts, ends));
  audit_check(recovered);
  check(actual.storage == retained && x.storage == saved_x && k.storage == mutated_k,
        "packed recovery violated ownership");

  // Scalar-tail and packet lanes both preserve the ordered initial-first fold;
  // zero weight still multiplies an overflowing/non-finite return.
  const double tiny = std::numeric_limits<double>::denorm_min();
  Array extreme_x(o::tensor_shape(5, 1, factors),
      {1e16, -1e16, 0, 0, -0.0, -0.0, -0.0, -0.0,
       tiny, 0, 0, 0, 710, 0, 0, 0, std::numeric_limits<double>::infinity(), 0, 0, 0},
      o::MemoryOrder::c, true);
  std::vector<double> extreme_beta(assets * factors, 0.0), extreme_seed(assets, -0.0);
  std::vector<double> extreme_weights(assets, 0.0);
  for (const auto asset : {std::size_t{0}, std::size_t{16}}) {
    extreme_beta[asset * factors] = 1.0;
    extreme_beta[asset * factors + 1] = 1.0;
    extreme_seed[asset] = 1.0;
  }
  extreme_weights[0] = 1.0;
  Array extreme_k(o::matrix_shape(assets, factors), extreme_beta, o::MemoryOrder::c, true);
  Array extreme_initial(o::vector_shape(assets), extreme_seed), extreme_w(o::vector_shape(assets), extreme_weights);
  const auto extreme_in = inputs(*compiled, extreme_x, extreme_k, extreme_initial, extreme_w);
  const auto extreme_params = parameters(*compiled, 1.0);
  const std::vector<std::int64_t> extreme_starts{0, 1, 2, 3, 4}, extreme_ends{1, 2, 3, 4, 5};
  auto extreme = private_run(compiled->program, extreme_in, extreme_params,
                             extreme_starts, extreme_ends, workspace);
  compare(extreme, canonical(compiled->program, extreme_in, extreme_params, extreme_starts, extreme_ends));
  branch_check(extreme.audit, 5);
  check(equal_ieee(extreme.root(0, 2).scalar, 0.0), "packet lane changed initial-first cancellation");
  check(equal_ieee(extreme.root(0, 3).scalar, 0.0), "tail changed initial-first cancellation");
  check(std::isnan(extreme.root(3, 3).scalar) && std::isnan(extreme.root(4, 3).scalar),
        "tail zero weight skipped overflow or infinity");
  check(actual.storage == retained, "IEEE execution changed independent retained result");
}

inline void rejection_and_partial_geometry() {
  const auto &r = returns_expression();
  for (const auto &expressions : std::vector<std::vector<std::string>>{
      {"sum_axis(multiply(" + r + ",broadcast_like(w," + r + ")),2)"},
      {roots()[0], r},
      {"sum_axis(multiply(broadcast_like(w," + r + ")," + r + "),0)"},
      {roots()[0] + ">0"}}) {
    const auto compiled = compile(expressions);
    check(!compiled->program.execution_metadata->affine_projection, "unproven consumers selected");
  }
  check(!compile(roots(), o::MemoryOrder::c, true)->program.execution_metadata->affine_projection,
        "isolated graph selected private strict path");

  auto compiled = compile(roots(false));
  Array x(o::tensor_shape(2, 1, 1), {.125, .25}), k(o::matrix_shape(1, 1), {1});
  Array initial(o::vector_shape(1), {0}), w(o::vector_shape(1), {1});
  const auto in = inputs(*compiled, x, k, initial, w);
  const std::int64_t starts[]{0, 0}, ends[]{1, 3};
  const std::int64_t valid_ends[]{1, 2};
  Result guarded(g::result_layout(compiled->program, in, starts, valid_ends, 2));
  o::Workspace workspace; g::Audit audit;
  error_is([&] { g::execute_affine_projection(compiled->program,
      *compiled->program.execution_metadata->affine_projection, in, nullptr, 0,
      starts, ends, 2, guarded.data(), &guarded.layout, 0, workspace, audit,
      std::chrono::steady_clock::time_point::max(), nullptr); }, "GRAPH_INTERVAL_BOUNDS");
  check(std::all_of(guarded.storage.begin(), guarded.storage.end(),
      [](double v) { return v == sentinel; }), "late geometry error wrote earlier row");
  auto malformed = guarded.layout;
  malformed.slots.back().capacity += 1;
  error_is([&] { g::execute_affine_projection(compiled->program,
      *compiled->program.execution_metadata->affine_projection, in, nullptr, 0,
      starts, valid_ends, 2, guarded.data(), &malformed, 0, workspace, audit,
      std::chrono::steady_clock::time_point::max(), nullptr); }, "RESULT_LAYOUT_BOUNDS");
  check(std::all_of(guarded.storage.begin(), guarded.storage.end(),
      [](double v) { return v == sentinel; }), "malformed layout wrote output");

  Array wide_x(o::tensor_shape(1, 1, 0), {}), wide_k(o::matrix_shape(0, 131073), {});
  Array wide_initial(o::vector_shape(131073), std::vector<double>(131073, 0));
  Array wide_w(o::vector_shape(131073), std::vector<double>(131073, 1));
  const auto wide = inputs(*compiled, wide_x, wide_k, wide_initial, wide_w);
  const std::int64_t start = 0, end = 1;
  Result unsupported(g::result_layout(compiled->program, wide, &start, &end, 1));
  check(!g::affine_projection_workspace_bytes(compiled->program,
      *compiled->program.execution_metadata->affine_projection, wide, 1), "wide shape has tile budget");
  check(!g::execute_affine_projection(compiled->program,
      *compiled->program.execution_metadata->affine_projection, wide, nullptr, 0,
      &start, &end, 1, unsupported.data(), &unsupported.layout, 0, workspace, audit,
      std::chrono::steady_clock::time_point::max(), nullptr), "wide shape selected private path");
  check(std::all_of(unsupported.storage.begin(), unsupported.storage.end(),
      [](double v) { return v == sentinel; }), "unsupported geometry wrote output");

  auto partial = compile({"stack(sum_axis(affine_contract(x,k,i),1),sum_axis(affine_contract(x,k,i),1),1)"},
                         o::MemoryOrder::c, false, false);
  Array matrix_x(o::matrix_shape(2, 2), {1, 2, 3, 4});
  Array matrix_k(o::matrix_shape(2, 3), {1, 2, 3, 4, 5, 6});
  Array three_initial(o::vector_shape(3), {0, 0, 0}), three_w(o::vector_shape(3), {1, 1, 1});
  auto unknown = inputs(*partial, matrix_x, matrix_k, three_initial, three_w);
  for (std::size_t i = 0; i < partial->input_names.size(); ++i)
    if (partial->input_names[i] == "x") { unknown[i].shape.rank = -1; unknown[i].data = nullptr; }
  std::vector<o::Shape> shapes;
  (void)g::required_array_capacity(partial->program, unknown, 2, &shapes);
  check(shapes.at(partial->program.roots[0]).size() == 0, "unknown downstream geometry became known");
  for (std::size_t i = 0; i < partial->input_names.size(); ++i)
    if (partial->input_names[i] == "i") unknown[i].shape = o::vector_shape(2);
  error_is([&] { (void)g::required_array_capacity(partial->program, unknown, 2); }, "SHAPE_MISMATCH");
}
} // namespace affine_projection_tests

inline void test_affine_projection_contracts() {
  affine_projection_tests::direct_writers();
  affine_projection_tests::complete_chain();
  affine_projection_tests::empty_and_nonfinite();
  affine_projection_tests::eigen_packed_intervals();
  affine_projection_tests::rejection_and_partial_geometry();
}
