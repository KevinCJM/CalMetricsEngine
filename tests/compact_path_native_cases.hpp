#pragma once

#include "affine_projection_native_cases.hpp"
#include "calmetrics_engine/compact_path.hpp"
#include "calmetrics_engine/planner.hpp"

#include <map>

namespace compact_path_tests {
namespace c = calmetrics_engine::compiler;
namespace g = calmetrics_engine::graph;
namespace o = calmetrics_engine::ops;
namespace t = calmetrics_engine::typed;
namespace p = calmetrics_engine::planner;
using affine_projection_tests::Array;
using affine_projection_tests::Result;
using affine_projection_tests::check;
using affine_projection_tests::compare;

struct Spec {
  std::size_t steps = 6, paths = 7, factors = 2, assets = 3, groups = 2;
  bool projected = false, custom_post = false;
  std::size_t duplicates = 0;
};

inline t::ValueType probabilities(const char *length) {
  auto type = t::ValueType::vector(length); type.axes = {"component"}; return type;
}

// Only mathematical template expressions. There is no native test copy of
// projection, prefix, quantile or the fourteen statistical kernel bodies.
inline std::shared_ptr<c::CompiledGraph> compile(const Spec &s,
    o::MemoryOrder order = o::MemoryOrder::c, bool isolate = false) {
  std::vector<t::Variable> variables{{"q5", probabilities("5")},
      {"qreturn", probabilities("3")}, {"qdd", probabilities("1")},
      {"n0", t::ValueType::scalar()}, {"target", t::ValueType::scalar()}};
  if (s.projected) {
    variables.push_back({"x", t::ValueType::tensor({"time", "path", "factor"}, {"T", "P", "F"})});
    variables.push_back({"beta", t::ValueType::matrix({"asset", "factor"}, {"A", "F"})});
    variables.push_back({"initial", t::ValueType::vector("A")});
  }
  if (s.custom_post) {
    auto type = t::ValueType::vector("P"); type.axes = {"path"};
    variables.push_back({"v", std::move(type)});
  }
  std::vector<std::string> roots;
  std::vector<std::map<std::string, std::string>> bindings;
  const auto T = std::to_string(s.steps), last = std::to_string(s.steps - 1),
             next = std::to_string(s.steps + 1), P = std::to_string(s.paths),
             S = std::to_string(std::min<std::size_t>(s.paths, 12));
  for (std::size_t group = 0; group < s.groups; ++group) {
    const auto suffix = std::to_string(group), weight = "w" + suffix;
    std::map<std::string, std::string> f;
    if (s.projected) {
      variables.push_back({weight, t::ValueType::vector("A")});
      f["simple"] = "expm1(affine_contract(x,transpose(beta),initial))";
      f["period"] = "sum_axis(multiply(broadcast_like(" + weight + ",simple),simple),2)";
    } else {
      const auto name = "period" + suffix;
      variables.push_back({name, t::ValueType::matrix({"time", "path"}, {"T", "P"})});
      f["period"] = name;
    }
    f["nav"] = "cumulative_product_axis(add(1,period),0,n0)";
    f["peak"] = "cumulative_max_axis(nav,0,n0)";
    f["drawdown"] = "subtract(1,divide(nav,peak))";
    f["max_dd"] = "max_axis(drawdown,0)";
    f["first_time"] = "slice_axis(nav,0,0,1,1)";
    f["seed"] = "where(finite_mask(first_time),n0,n0)";
    f["full"] = "concatenate(seed,nav,0)";
    f["fan"] = "swap_axes(quantiles_axis(full,q5,1),0,1)";
    f["samples"] = "swap_axes(slice_axis(full,1,0," + S + ",1),0,1)";
    f["terminal_nav"] = "sum_axis(slice_axis(nav,0," + last + "," + T + ",1),0)";
    f["terminal_return"] = "subtract(divide(terminal_nav,n0),1)";
    f["return_quantiles"] = "quantiles_axis(terminal_return,qreturn,0)";
    f["dd_quantiles"] = "quantiles_axis(max_dd,qdd,0)";
    f["terminal_fan"] = "sum_axis(slice_axis(fan,1," + T + "," + next + ",1),1)";
    f["q05"] = "first(slice_axis(return_quantiles,0,0,1,1))";
    f["tail_mask"] = "less_equal(terminal_return,q05)";
    f["tail_head"] = "slice_axis(terminal_return,0,0,1,1)";
    f["tail_zero"] = "where(finite_mask(tail_head),0,0)";
    f["tail_true"] = "equal(tail_zero,tail_zero)";
    f["tail_values"] = "concatenate(terminal_return,tail_zero,0)";
    f["tail_selected"] = "concatenate(tail_mask,tail_true,0)";
    f["tail_total"] = "sum_where(tail_values,tail_selected)";
    f["tail_count"] = "count_true(tail_mask)";
    f["has_tail"] = "greater_than(tail_count,0)";
    f["safe_tail_count"] = "where(has_tail,tail_count,1)";
    f["var"] = "maximum(0,negate(q05))";
    f["es"] = "where(has_tail,maximum(0,negate(divide(tail_total,safe_tail_count))),var)";
    f["loss"] = "divide(count_true(less_than(terminal_return,0)),length(terminal_return))";
    f["hit"] = "divide(count_true(greater_equal(terminal_return,target)),length(terminal_return))";
    f["average_dd"] = "divide(sum(max_dd),length(terminal_return))";
    f["dd95"] = "first(dd_quantiles)";
    std::vector<std::string> names{"fan", "samples"};
    for (std::size_t i = 0; i < 5; ++i) {
      const auto name = "metric" + std::to_string(i);
      f[name] = "first(slice_axis(terminal_fan,0," + std::to_string(i) + "," + std::to_string(i + 1) + ",1))";
      names.push_back(name);
    }
    for (std::size_t i = 0; i < 3; ++i) {
      const auto name = "metric" + std::to_string(i + 5);
      f[name] = "first(slice_axis(return_quantiles,0," + std::to_string(i) + "," + std::to_string(i + 1) + ",1))";
      names.push_back(name);
    }
    for (const auto *name : {"var", "es", "loss", "hit", "average_dd", "dd95"}) names.emplace_back(name);
    if (s.projected) for (std::size_t a = 0; a < s.assets; ++a) {
      const auto name = "contribution" + std::to_string(a), begin = std::to_string(a), end = std::to_string(a + 1);
      f[name] = "divide(sum(multiply(first(slice_axis(" + weight + ",0," + begin + "," + end +
          ",1)),slice_axis(simple,2," + begin + "," + end + ",1)))," + P + ")";
      names.push_back(name);
    }
    if (s.custom_post) {
      // After captures become inputs this pair is eligible for private fusion;
      // deferred replay disables fusion, so child budgeting must do so too.
      f["custom"] = "sum(add(multiply(max_dd,v),n0))"; names.push_back("custom");
    }
    if (s.duplicates) {
      names.push_back("fan"); names.push_back("samples");
      for (std::size_t i = 0; i < s.duplicates; ++i) names.push_back("es");
      if (s.projected && s.assets) names.push_back("contribution0");
    }
    for (const auto &name : names) { roots.push_back(name); bindings.push_back(f); }
  }
  return c::compile(roots, variables, isolate, bindings, {}, 0, 100000000,
      true, {}, order == o::MemoryOrder::f ? g::OutputOrder::f : g::OutputOrder::c);
}

inline std::vector<double> fixture_values(std::size_t n, double scale, int shift = 0) {
  std::vector<double> values(n);
  for (std::size_t i = 0; i < n; ++i) values[i] = scale * static_cast<double>(static_cast<int>(i % 7) - 3 + shift);
  return values;
}
struct Fixture {
  Array q5, qreturn, qdd, x, beta, initial, v;
  std::vector<Array> periods, weights;
  double n0 = 1.7, target = .025;
  explicit Fixture(const Spec &s, o::MemoryOrder input_order = o::MemoryOrder::c,
      bool reverse = false, std::size_t physical_steps = 0)
      : q5(o::vector_shape(5), {.05, .25, .5, .75, .95}),
        qreturn(o::vector_shape(3), {.05, .5, .95}), qdd(o::vector_shape(1), {.95}),
        x(o::tensor_shape(physical_steps ? physical_steps : s.steps, s.paths, s.factors),
          fixture_values((physical_steps ? physical_steps : s.steps) * s.paths * s.factors, .005), input_order, reverse),
        beta(o::matrix_shape(s.assets, s.factors), fixture_values(s.assets * s.factors, .125), input_order, reverse),
        initial(o::vector_shape(s.assets), fixture_values(s.assets, .002), input_order, reverse),
        v(o::vector_shape(s.paths), fixture_values(s.paths, .1)) {
    periods.reserve(s.groups); weights.reserve(s.groups);
    for (std::size_t group = 0; group < s.groups; ++group) {
      periods.emplace_back(o::matrix_shape(physical_steps ? physical_steps : s.steps, s.paths),
          fixture_values((physical_steps ? physical_steps : s.steps) * s.paths, .005, static_cast<int>(group)), input_order, reverse);
      weights.emplace_back(o::vector_shape(s.assets), fixture_values(s.assets, .125, static_cast<int>(group) + 1), input_order, reverse);
    }
  }
  std::vector<o::Value> inputs(const c::CompiledGraph &compiled) const {
    std::vector<o::Value> inputs;
    for (const auto &name : compiled.input_names) {
      if (name == "q5") inputs.push_back(q5.value);
      else if (name == "qreturn") inputs.push_back(qreturn.value);
      else if (name == "qdd") inputs.push_back(qdd.value);
      else if (name == "x") inputs.push_back(x.value);
      else if (name == "beta") inputs.push_back(beta.value);
      else if (name == "initial") inputs.push_back(initial.value);
      else if (name == "v") inputs.push_back(v.value);
      else if (name.rfind("period", 0) == 0) inputs.push_back(periods.at(std::stoul(name.substr(6))).value);
      else if (name.rfind("w", 0) == 0) inputs.push_back(weights.at(std::stoul(name.substr(1))).value);
      else throw std::runtime_error("unexpected compact input");
    }
    return inputs;
  }
  std::vector<double> parameters(const c::CompiledGraph &compiled) const {
    std::vector<double> parameters;
    for (const auto &name : compiled.parameter_names) {
      if (name == "n0") parameters.push_back(n0);
      else if (name == "target") parameters.push_back(target);
      else throw std::runtime_error("unexpected compact parameter");
    }
    return parameters;
  }
  std::vector<std::vector<double>> snapshot() const {
    std::vector<std::vector<double>> saved{q5.storage, qreturn.storage, qdd.storage,
        x.storage, beta.storage, initial.storage, v.storage};
    for (const auto &array : periods) saved.push_back(array.storage);
    for (const auto &array : weights) saved.push_back(array.storage);
    return saved;
  }
};
inline void unchanged(const Fixture &fixture, const std::vector<std::vector<double>> &saved) {
  const auto now = fixture.snapshot(); check(now.size() == saved.size(), "fixture count changed");
  for (std::size_t i = 0; i < now.size(); ++i)
    check(now[i].size() == saved[i].size() && (now[i].empty() ||
        std::memcmp(now[i].data(), saved[i].data(), now[i].size() * sizeof(double)) == 0), "compact mutated input");
}
inline Result run(const c::CompiledGraph &compiled, const Fixture &fixture, bool observed,
    const std::vector<std::int64_t> &starts, const std::vector<std::int64_t> &ends) {
  const auto inputs = fixture.inputs(compiled); const auto parameters = fixture.parameters(compiled);
  Result result(g::result_layout(compiled.program, inputs, starts.data(), ends.data(), starts.size()));
  if (observed) result.audit = g::execute_observed(compiled.program, inputs, parameters.data(), parameters.size(),
      starts.data(), ends.data(), starts.size(), result.data(), compiled.program.roots.size(), &result.layout,
      [](std::optional<o::Op>, const o::Value &, bool) {});
  else result.audit = g::execute(compiled.program, inputs, parameters.data(), parameters.size(), starts.data(),
      ends.data(), starts.size(), result.data(), compiled.program.roots.size(), &result.layout);
  result.canaries(); return result;
}
inline void compare_complete(const Spec &s, o::MemoryOrder output_order,
    o::MemoryOrder input_order, bool reverse) {
  auto compiled = compile(s, output_order); Fixture fixture(s, input_order, reverse);
  check(compiled->program.execution_metadata->compact_path != nullptr, "complete compact DAG not proved");
  const auto saved_inputs = fixture.snapshot();
  const std::vector<std::int64_t> starts{0}, ends{static_cast<std::int64_t>(s.steps)};
  const auto expected = run(*compiled, fixture, true, starts, ends);
  const auto actual = run(*compiled, fixture, false, starts, ends);
  compare(actual, expected);
  check(actual.audit.compact_path_groups == s.groups && actual.audit.compact_path_batches >= 1 &&
      actual.audit.compact_path_steps == s.steps * actual.audit.compact_path_batches,
      "native test did not execute compact scan body");
  const auto saved_result = actual.storage;
  const auto again = run(*compiled, fixture, false, starts, ends);
  compare(again, expected);
  check(actual.storage == saved_result && actual.data() != again.data(), "second compact result aliases prior result");
  unchanged(fixture, saved_inputs);
}
inline std::string failure(const c::CompiledGraph &compiled, const Fixture &fixture, bool observed) {
  const auto inputs = fixture.inputs(compiled);
  const auto parameters = fixture.parameters(compiled);
  const std::int64_t start = 0, end = static_cast<std::int64_t>(fixture.periods.front().value.shape.dim[0]);
  Result result(g::result_layout(compiled.program, inputs, &start, &end, 1));
  std::string code = "NO_ERROR";
  try {
    if (observed) (void)g::execute_observed(compiled.program, inputs, parameters.data(), parameters.size(),
        &start, &end, 1, result.data(), compiled.program.roots.size(), &result.layout,
        [](std::optional<o::Op>, const o::Value &, bool) {});
    else (void)g::execute(compiled.program, inputs, parameters.data(), parameters.size(),
        &start, &end, 1, result.data(), compiled.program.roots.size(), &result.layout);
  } catch (const o::Error &error) { code = error.what(); }
  result.canaries(); return code;
}
inline void error_order() {
  Spec s; s.groups = 2; auto compiled = compile(s); Fixture fixture(s);
  fixture.n0 = 0.0;
  fixture.q5.storage[0] = 1.5;
  check(failure(*compiled, fixture, true) == "INVALID_PARAMETER" &&
      failure(*compiled, fixture, false) == "INVALID_PARAMETER", "fan checkpoint moved behind divide0");
  fixture.q5.storage[0] = .05;
  check(failure(*compiled, fixture, true) == "DIVIDE_BY_ZERO" &&
      failure(*compiled, fixture, false) == "DIVIDE_BY_ZERO", "post divide0 was suppressed");
  fixture.n0 = std::numeric_limits<double>::infinity();
  fixture.qreturn.storage[0] = 1.5;
  check(failure(*compiled, fixture, true) == "INVALID_PARAMETER" &&
      failure(*compiled, fixture, false) == "INVALID_PARAMETER", "max_axis error moved before earlier post quantile");
  fixture.qreturn.storage[0] = .05;
  check(failure(*compiled, fixture, true) == "NON_FINITE_RESULT" &&
      failure(*compiled, fixture, false) == "NON_FINITE_RESULT", "max_axis finite guard was lost");
}
inline void geometry_and_budget() {
  Spec s; s.custom_post = true; s.paths = 257;
  auto compiled = compile(s); Fixture fixture(s, o::MemoryOrder::f, true, s.steps * 2);
  auto inputs = fixture.inputs(*compiled);
  const auto proof = compiled->program.execution_metadata->compact_path;
  check(proof != nullptr, "custom canonical post was not proved");
  const auto budget = g::compact_path_budget(compiled->program, *proof, inputs, {s.steps, s.steps}, p::canonical_child_scratch_bytes);
  check(budget && budget->batch_size == 2 && budget->workspace_doubles > 0 &&
      budget->child_scratch_bytes > 0, "complete budget missing child/state");
  check(!g::compact_path_budget(compiled->program, *proof, inputs, {s.steps, s.steps - 1},
      p::canonical_child_scratch_bytes), "maximum-only compact admission ignored shorter interval");
  auto unknown = inputs;
  for (std::size_t i = 0; i < unknown.size(); ++i)
    if (compiled->program.input_axes[i] != 1) { unknown[i].shape.rank = -1; break; }
  check(!g::compact_path_budget(compiled->program, *proof, unknown, {s.steps},
      p::canonical_child_scratch_bytes), "unknown geometry was admitted");
  const std::vector<std::int64_t> starts{0, static_cast<std::int64_t>(s.steps)},
      ends{static_cast<std::int64_t>(s.steps), static_cast<std::int64_t>(s.steps * 2)};
  const auto expected = run(*compiled, fixture, true, starts, ends), actual = run(*compiled, fixture, false, starts, ends);
  compare(actual, expected);
  check(actual.audit.compact_path_groups == s.groups * starts.size() &&
      actual.audit.compact_path_steps == s.steps * actual.audit.compact_path_batches,
      "multi-interval compact counters");
  check(actual.audit.operator_workspace_capacity_bytes + actual.audit.numeric_arena_bytes + actual.audit.mask_arena_bytes <=
      budget->total_bytes, "actual compact capacity exceeds admitted budget");
  std::vector<std::size_t> sizes;
  for (const auto &input : inputs) sizes.push_back(input.size());
  const auto plan = p::make_plan(compiled, p::Config{}, sizes, starts.data(), ends.data(), starts.size(),
      nullptr, 1, {}, false, false, false, &inputs);
  check(plan->estimated_worker_scratch_bytes >= budget->total_bytes, "Planner omitted compact child budget");
  bool rejected = false;
  try { (void)p::make_plan(compiled, p::Config{}, sizes, starts.data(), ends.data(), starts.size(),
      nullptr, 1, plan->estimated_total_memory_bytes - 1, false, false, false, &inputs); }
  catch (const std::bad_alloc &) { rejected = true; }
  check(rejected, "memory budget below complete plan was admitted");
}
inline void owner_lifetime() {
  Spec s; s.projected = true;
  auto first = compile(s, o::MemoryOrder::f); Fixture a(s, o::MemoryOrder::c, true);
  const auto output = run(*first, a, false, {0}, {static_cast<std::int64_t>(s.steps)});
  const auto saved = output.storage;
  Spec smaller; smaller.steps = 1; smaller.paths = 1; smaller.groups = 1;
  auto second = compile(smaller); Fixture b(smaller);
  compare(run(*second, b, false, {0}, {1}), run(*second, b, true, {0}, {1}));
  g::release_graph_workspace();
  compare(run(*first, a, false, {0}, {static_cast<std::int64_t>(s.steps)}),
      run(*first, a, true, {0}, {static_cast<std::int64_t>(s.steps)}));
  check(output.storage == saved, "owner change/release invalidated returned result");
  output.canaries();
  check(!compile(s, o::MemoryOrder::c, true)->program.execution_metadata->compact_path,
      "isolate was admitted to strict deferred compact path");
  auto selected = *first;
  selected.program.roots.erase(selected.program.roots.begin());
  selected.program.root_outputs.erase(selected.program.root_outputs.begin());
  selected.program.finalize();
  check(!selected.program.execution_metadata->compact_path, "selected-only roots admitted without full fan");
}
} // namespace compact_path_tests

inline void test_compact_path_contracts() {
  using namespace compact_path_tests;
  for (const auto order : {o::MemoryOrder::c, o::MemoryOrder::f}) {
    Spec period; compare_complete(period, order, o::MemoryOrder::f, true);
    Spec tape; tape.projected = true; tape.duplicates = 2;
    compare_complete(tape, order, o::MemoryOrder::c, true);
    Spec one; one.steps = 1; one.paths = 1; one.groups = 1;
    compare_complete(one, order, o::MemoryOrder::c, false);
    for (const auto geometry : {std::pair<std::size_t, std::size_t>{0, 8}, {2, 0}, {0, 0}}) {
      Spec empty; empty.projected = true; empty.factors = geometry.first; empty.assets = geometry.second;
      compare_complete(empty, order, o::MemoryOrder::f, true);
    }
  }
  error_order(); geometry_and_budget(); owner_lifetime();
  g::release_graph_workspace();
}
