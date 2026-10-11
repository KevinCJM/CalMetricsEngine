#pragma once

#include "compact_path_native_cases.hpp"

namespace raw_prefix_tests {
namespace c = calmetrics_engine::compiler;
namespace g = calmetrics_engine::graph;
namespace o = calmetrics_engine::ops;
namespace t = calmetrics_engine::typed;
namespace p = calmetrics_engine::planner;
using affine_projection_tests::Array;
using affine_projection_tests::Result;
using affine_projection_tests::check;
using affine_projection_tests::compare;
using affine_projection_tests::equal_ieee;

struct Spec {
  std::size_t steps = 9, paths = 7, innovations = 4, factors = 9, assets = 9, groups = 2;
  bool student = false, public_factors = false;
};
template<class Function>
inline auto context(const std::string &stage, Function &&function) -> decltype(function()) {
  try { return function(); }
  catch (const std::exception &error) { throw std::runtime_error(stage + ": " + error.what()); }
}
inline std::shared_ptr<c::CompiledGraph> compile(const Spec &s, o::MemoryOrder order) {
  auto means = t::ValueType::vector("F"); means.axes = {"factor"};
  std::vector<t::Variable> variables{
      {"raw", t::ValueType::tensor({"time", "path", "component"}, {"T", "P", "I"})},
      {"root", t::ValueType::matrix({"factor", "component"}, {"F", "I"})},
      {"means", means},
      {"beta", t::ValueType::matrix({"asset", "factor"}, {"A", "F"})},
      {"initial", t::ValueType::vector("A")},
      {"q", compact_path_tests::probabilities("3")}, {"n0", t::ValueType::scalar()}};
  if (s.student) {
    variables.push_back({"df", t::ValueType::scalar()});
    variables.push_back({"chi3", t::ValueType::tensor({"time", "path", "factor"}, {"T", "P", "1"})});
  }
  std::vector<std::string> roots;
  std::vector<std::map<std::string, std::string>> bindings;
  const auto T = std::to_string(s.steps), last = std::to_string(s.steps - 1),
      S = std::to_string(std::min<std::size_t>(s.paths, 12));
  for (std::size_t group = 0; group < s.groups; ++group) {
    const auto weight = "w" + std::to_string(group);
    variables.push_back({weight, t::ValueType::vector("A")});
    std::map<std::string, std::string> f;
    f["rot"] = "affine_contract(raw,transpose(root),0)";
    if (s.student) {
      f["scale"] = "divide(sqrt(divide(subtract(df,2),df)),sqrt(divide(chi3,df)))";
      f["scaled"] = "multiply(rot,broadcast_like(scale,rot))";
    } else f["scaled"] = "multiply(rot,1)";
    f["x"] = "add(broadcast_like(means,rot),scaled)";
    f["simple"] = "expm1(affine_contract(x,transpose(beta),initial))";
    f["period"] = "sum_axis(multiply(broadcast_like(" + weight + ",simple),simple),2)";
    f["nav"] = "cumulative_product_axis(add(1,period),0,n0)";
    f["peak"] = "cumulative_max_axis(nav,0,n0)";
    f["dd"] = "subtract(1,divide(nav,peak))";
    f["max_dd"] = "max_axis(dd,0)";
    f["first_time"] = "slice_axis(nav,0,0,1,1)";
    f["seed"] = "where(finite_mask(first_time),n0,n0)";
    f["full"] = "concatenate(seed,nav,0)";
    f["fan"] = "swap_axes(quantiles_axis(full,q,1),0,1)";
    f["samples"] = "swap_axes(slice_axis(full,1,0," + S + ",1),0,1)";
    f["terminal"] = "sum_axis(slice_axis(nav,0," + last + "," + T + ",1),0)";
    f["dd_total"] = "sum(max_dd)";
    f["nav_total"] = "sum(terminal)";
    for (const auto *name : {"fan", "samples", "dd_total", "nav_total"}) {
      roots.emplace_back(name); bindings.push_back(f);
    }
    if (s.public_factors && !group) { roots.emplace_back("x"); bindings.push_back(f); }
  }
  return context("compile", [&] {
    return c::compile(roots, variables, false, bindings, {}, 0, 100000000,
        true, {}, order == o::MemoryOrder::f ? g::OutputOrder::f : g::OutputOrder::c);
  });
}
struct Fixture {
  Array raw, root, means, beta, initial, q, chi;
  std::vector<Array> weights;
  double df = 5.0, n0 = 1.7;
  explicit Fixture(const Spec &s, o::MemoryOrder order, bool reverse, std::size_t times = 1)
      : raw(o::tensor_shape(s.steps * times, s.paths, s.innovations),
            compact_path_tests::fixture_values(s.steps * times * s.paths * s.innovations, .005), order, reverse),
        root(o::matrix_shape(s.factors, s.innovations),
             compact_path_tests::fixture_values(s.factors * s.innovations, .125), order, reverse),
        means(o::vector_shape(s.factors), compact_path_tests::fixture_values(s.factors, .002), order, reverse),
        beta(o::matrix_shape(s.assets, s.factors),
             compact_path_tests::fixture_values(s.assets * s.factors, .125), order, reverse),
        initial(o::vector_shape(s.assets), compact_path_tests::fixture_values(s.assets, .001), order, reverse),
        q(o::vector_shape(3), {.05, .5, .95}),
        chi(o::tensor_shape(s.steps * times, s.paths, 1),
            std::vector<double>(s.steps * times * s.paths, 3.0), order, reverse) {
    weights.reserve(s.groups);
    for (std::size_t group = 0; group < s.groups; ++group)
      weights.emplace_back(o::vector_shape(s.assets),
          compact_path_tests::fixture_values(s.assets, .125, static_cast<int>(group) + 1), order, reverse);
  }
  std::vector<o::Value> inputs(const c::CompiledGraph &compiled) const {
    std::vector<o::Value> values;
    for (const auto &name : compiled.input_names) {
      if (name == "raw") values.push_back(raw.value);
      else if (name == "root") values.push_back(root.value);
      else if (name == "means") values.push_back(means.value);
      else if (name == "beta") values.push_back(beta.value);
      else if (name == "initial") values.push_back(initial.value);
      else if (name == "q") values.push_back(q.value);
      else if (name == "chi3") values.push_back(chi.value);
      else if (name.rfind("w", 0) == 0) values.push_back(weights.at(std::stoul(name.substr(1))).value);
      else throw std::runtime_error("unexpected raw prefix input");
    }
    return values;
  }
  std::vector<double> parameters(const c::CompiledGraph &compiled) const {
    std::vector<double> values;
    for (const auto &name : compiled.parameter_names) {
      if (name == "df") values.push_back(df);
      else if (name == "n0") values.push_back(n0);
      else throw std::runtime_error("unexpected raw prefix parameter");
    }
    return values;
  }
  std::vector<std::vector<double>> snapshot() const {
    std::vector<std::vector<double>> saved{raw.storage, root.storage, means.storage,
        beta.storage, initial.storage, q.storage, chi.storage};
    for (const auto &weight : weights) saved.push_back(weight.storage);
    return saved;
  }
};
inline Result run(const c::CompiledGraph &compiled, const Fixture &fixture, bool observed,
    const std::vector<std::int64_t> &starts, const std::vector<std::int64_t> &ends) {
  const auto inputs = fixture.inputs(compiled);
  const auto parameters = fixture.parameters(compiled);
  const std::string lane = observed ? "canonical" : "private";
  Result result(context(lane + ".layout", [&] {
    return g::result_layout(compiled.program, inputs, starts.data(), ends.data(), starts.size());
  }));
  result.audit = context(lane + ".execute", [&] {
    if (observed) return g::execute_observed(compiled.program, inputs, parameters.data(), parameters.size(),
        starts.data(), ends.data(), starts.size(), result.data(), compiled.program.roots.size(), &result.layout,
        [](std::optional<o::Op>, const o::Value &, bool) {});
    return g::execute(compiled.program, inputs, parameters.data(), parameters.size(),
        starts.data(), ends.data(), starts.size(), result.data(), compiled.program.roots.size(), &result.layout);
  });
  result.canaries(); return result;
}
inline void complete(const Spec &s, o::MemoryOrder order) {
  const auto compiled = compile(s, order); Fixture fixture(s, order, true, 2);
  check(compiled->program.execution_metadata->compact_path != nullptr, "raw prefix not proved");
  const auto inputs = fixture.inputs(*compiled);
  const auto saved = fixture.snapshot();
  const auto budget = context("budget", [&] {
    return g::compact_path_budget(compiled->program, *compiled->program.execution_metadata->compact_path,
        inputs, {s.steps, s.steps}, p::canonical_child_scratch_bytes);
  });
  check(budget && budget->batch_size == s.groups, "raw prefix budget must have one shared batch");
  const std::vector<std::int64_t> starts{0, static_cast<std::int64_t>(s.steps)},
      ends{static_cast<std::int64_t>(s.steps), static_cast<std::int64_t>(2 * s.steps)};
  const auto expected = run(*compiled, fixture, true, starts, ends),
      actual = run(*compiled, fixture, false, starts, ends);
  compare(actual, expected);
  check(actual.audit.compact_path_raw_rows == 2 * s.steps * s.paths &&
      actual.audit.compact_path_raw_scale_elements == (s.student ? 2 * s.steps * s.paths : 0),
      "raw transform was repeated per consumer or scale audit missing");
  check(actual.audit.compact_path_batches == 2 && actual.audit.compact_path_groups == 2 * s.groups,
      "raw shared batch audit");
  check(actual.audit.operator_workspace_capacity_bytes + actual.audit.numeric_arena_bytes +
      actual.audit.mask_arena_bytes <= budget->total_bytes, "raw prefix exceeded complete workspace budget");
  const auto again = run(*compiled, fixture, false, starts, ends);
  compare(again, expected);
  check(actual.data() != again.data(), "raw results must own distinct storage");
  g::release_graph_workspace(); compare(actual, expected);
  check(fixture.snapshot() == saved, "raw prefix changed borrowed input storage");
}
inline void refusals() {
  Spec s; s.steps = 3; s.groups = 32; s.factors = 1;
  auto compiled = compile(s, o::MemoryOrder::c); Fixture fixture(s, o::MemoryOrder::c, false);
  check(compiled->program.execution_metadata->compact_path != nullptr, "short prefix grammar proof missing");
  check(!g::compact_path_budget(compiled->program, *compiled->program.execution_metadata->compact_path,
      fixture.inputs(*compiled), {s.steps}, p::canonical_child_scratch_bytes), "multiple batches would repeat rotation");
  const auto actual = run(*compiled, fixture, false, {0}, {3}), expected = run(*compiled, fixture, true, {0}, {3});
  compare(actual, expected); check(!actual.audit.compact_path_raw_rows, "short request must use canonical path");
  s.steps = 9; s.public_factors = true;
  check(!compile(s, o::MemoryOrder::c)->program.execution_metadata->compact_path,
      "public transformed tensor was admitted to private row storage");
}
inline void state_batch_budget() {
  Spec s; s.steps = 5; s.groups = 32; s.factors = 32;
  const auto compiled = compile(s, o::MemoryOrder::c); Fixture fixture(s, o::MemoryOrder::c, false);
  auto inputs = fixture.inputs(*compiled);
  // Budgeting is geometry-only: these larger extents deliberately have no
  // payload. Just below/above 16 MiB of full-group states, with A=9 included.
  const auto raw = std::find(compiled->input_names.begin(), compiled->input_names.end(), "raw");
  check(raw != compiled->input_names.end(), "raw input missing");
  auto &value = inputs[static_cast<std::size_t>(raw - compiled->input_names.begin())];
  value.data = nullptr;
  value.shape.dim[1] = 21842; value.set_contiguous_strides();
  const auto small = g::compact_path_budget(compiled->program,
      *compiled->program.execution_metadata->compact_path, inputs, {s.steps}, p::canonical_child_scratch_bytes);
  check(small && small->batch_size == s.groups, "subthreshold states should share the producer once");
  value.shape.dim[1] = 21843; value.set_contiguous_strides();
  const auto large = g::compact_path_budget(compiled->program,
      *compiled->program.execution_metadata->compact_path, inputs, {s.steps}, p::canonical_child_scratch_bytes);
  check(large && large->batch_size == 4, "large short states should be bounded");
  check(large->workspace_doubles * sizeof(double) < 4 * 1024 * 1024,
      "large state batch failed to reduce actual planned workspace");
  check(large->physical_work > small->physical_work,
      "planner hid the repeated raw and asset producer work");
}
inline void linear_fold() {
  Array source(o::matrix_shape(1, 3), {1e16, 1.0, -1e16});
  Array coefficients(o::matrix_shape(3, 9), std::vector<double>(27, 1.0));
  const std::array<o::Value, 3> args{source.value, coefficients.value, o::Value::number(0.0)};
  const auto prepared = context("prepare", [&] {
    return o::prepare(o::lookup("affine_contract"), args.data(), args.size());
  });
  std::array<double, 9> expected{}, actual{};
  o::Output out; out.shape = prepared.output_shape; out.data = expected.data();
  o::Workspace work; o::Audit canonical_audit;
  context("canonical.execute", [&] { o::execute(prepared, out, work, o::Isa::scalar, canonical_audit); });
  for (const bool packets : {false, g::affine_producer_packets_enabled(
      g::affine_producer_coefficient_reserve(1, 3, 9))}) {
    g::Audit audit;
    context(packets ? "packet.execute" : "scalar.execute", [&] {
      g::affine_producer_linear_row(3, 9, coefficients.value, args[2],
          static_cast<const double *>(source.value.data), 0, 1, actual.data(), packets, audit);
    });
    for (std::size_t i = 0; i < actual.size(); ++i)
      check(equal_ieee(actual[i], expected[i]), "linear producer changed ascending fold");
  }
}
} // namespace raw_prefix_tests

inline void test_raw_prefix_contracts() {
  using namespace raw_prefix_tests;
  context("linear", [&] { linear_fold(); });
  context("refusals", [&] { refusals(); });
  context("state_batch_budget", [&] { state_batch_budget(); });
  for (const auto order : {o::MemoryOrder::c, o::MemoryOrder::f})
    for (const bool student : {false, true}) {
      const std::string phase = std::string(order == o::MemoryOrder::c ? "C" : "F") +
          (student ? ".Student" : ".normal");
      Spec s; s.student = student;
      context(phase + ".full", [&] { complete(s, order); });
      auto empty_i = s; empty_i.innovations = 0;
      context(phase + ".emptyI", [&] { complete(empty_i, order); });
      auto empty_f = s; empty_f.factors = 0;
      context(phase + ".emptyF", [&] { complete(empty_f, order); });
      auto empty_a = s; empty_a.assets = 0;
      context(phase + ".emptyA", [&] { complete(empty_a, order); });
      auto empty_if = empty_i; empty_if.factors = 0;
      context(phase + ".emptyIF", [&] { complete(empty_if, order); });
      auto empty_ifa = empty_if; empty_ifa.assets = 0;
      context(phase + ".emptyIFA", [&] { complete(empty_ifa, order); });
    }
  g::release_graph_workspace();
}
