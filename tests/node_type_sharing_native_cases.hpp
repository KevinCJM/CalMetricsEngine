#pragma once

#include "calmetrics_engine/compiler.hpp"
#include <stdexcept>
#include <unordered_set>

namespace node_type_sharing {
namespace c = calmetrics_engine::compiler;
namespace t = calmetrics_engine::typed;
namespace g = calmetrics_engine::graph;
namespace o = calmetrics_engine::ops;

inline void check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

inline const c::NodeInfo &root(const c::CompiledGraph &graph, std::size_t index) {
  return graph.nodes.at(graph.program.roots.at(index));
}

inline void contracts() {
  c::NodeInfo empty;
  check(!empty.inferred_type && t::same_type(empty.type(), t::ValueType::scalar()),
        "default NodeInfo scalar value changed");

  // These are same_type field-level tests, not claims that every altered
  // nominal record can be admitted as a standalone public compiler input.
  const auto original = t::ValueType::record("fit", {"slope", "intercept"});
  for (unsigned field = 0; field < 8; ++field) {
    auto changed = original;
    switch (field) {
    case 0: changed.kind = t::ValueKind::matrix; break;
    case 1: changed.dtype = t::DType::int64; break;
    case 2: changed.axes = {"asset"}; break;
    case 3: changed.shape = {"N"}; break;
    case 4: changed.semantic_dimension = "count"; break;
    case 5: changed.price_basis = "test_basis"; break;
    case 6: changed.record_tag = "interval"; break;
    case 7: changed.fields = {"intercept", "slope"}; break;
    }
    check(!t::same_type(original, changed), "full type identity omitted a field");
  }

  // Actual admitted roots isolate kind, dtype, axes, shape, semantics and basis.
  std::vector<t::Variable> variables;
  std::vector<std::string> expressions;
  auto add = [&](std::string name, t::ValueType type) {
    expressions.push_back(name);
    variables.push_back({std::move(name), std::move(type)});
  };
  add("a", t::ValueType::vector());
  add("b", t::ValueType::vector());
  add("scalar", t::ValueType::scalar());
  auto exact_scalar = t::ValueType::scalar();
  exact_scalar.kind = t::ValueKind::value;
  add("exact_scalar", exact_scalar);
  auto integers = t::ValueType::vector();
  integers.dtype = t::DType::int64;
  add("integers", integers);
  auto factor = t::ValueType::vector();
  factor.axes = {"factor"};
  add("factor", factor);
  add("other_shape", t::ValueType::vector("M"));
  add("counts", t::ValueType::vector("N", "count"));
  add("basis_a", t::ValueType::vector("N", "reported_nav", "test_basis_a"));
  add("basis_b", t::ValueType::vector("N", "reported_nav", "test_basis_b"));
  auto graph = c::compile(expressions, variables, false, {}, {}, 0, 100000000, true);
  for (const auto &node : graph->nodes)
    check(static_cast<bool>(node.inferred_type), "published node has no immutable type owner");
  check(root(*graph, 0).inferred_type == root(*graph, 1).inferred_type,
        "identical admitted root types did not share");
  check(root(*graph, 2).inferred_type != root(*graph, 3).inferred_type,
        "scalar and exact-dtype value kinds were merged");
  for (std::size_t index = 4; index <= 7; ++index)
    check(root(*graph, 0).inferred_type != root(*graph, index).inferred_type,
          "distinct admitted root types were merged");
  check(root(*graph, 8).inferred_type != root(*graph, 9).inferred_type,
        "distinct price bases were merged");
  for (std::size_t index = 0; index < variables.size(); ++index)
    check(t::same_type(root(*graph, index).type(), variables[index].type),
          "shared root type differs from admitted declaration");

  auto second = c::compile(expressions, variables, false, {}, {}, 0, 100000000, true);
  check(root(*graph, 0).inferred_type != root(*second, 0).inferred_type,
        "independent compiles share a global type cache");
  const auto wire = c::encode_program(graph->program);
  const auto fingerprint = graph->fingerprint;
  c::CompiledGraph copied = *graph;
  const auto snapshot = root(*graph, 0);
  check(root(copied, 0).inferred_type == snapshot.inferred_type,
        "graph copy did not retain the immutable type owner");
  graph.reset();
  check(t::same_type(snapshot.type(), variables[0].type) &&
        copied.fingerprint == fingerprint && c::encode_program(copied.program) == wire,
        "snapshot or graph copy lost metadata after original destruction");
  copied.nodes[copied.program.roots[0]].inferred_type =
      std::make_shared<const t::ValueType>(t::ValueType::scalar());
  check(t::same_type(snapshot.type(), variables[0].type) &&
        t::same_type(root(copied, 1).type(), variables[1].type),
        "replacing one type handle changed another owner");

  auto defaults = c::compile({"1", "2"}, {{"x", "series"}});
  check(root(*defaults, 0).inferred_type &&
        root(*defaults, 0).inferred_type == root(*defaults, 1).inferred_type,
        "default scalar publication bypassed compilation-local sharing");
  auto other_defaults = c::compile({"1", "2"}, {{"x", "series"}});
  check(root(*defaults, 0).inferred_type != root(*other_defaults, 0).inferred_type,
        "default published scalars share a global owner");

  // Real coupled records exercise nominal tags/fields without inventing an
  // unsupported record input. Here those two fields differ together.
  auto records = c::compile({"fit_slope(linear_fit(x))", "fit_slope(linear_fit(y))",
      "interval_start(last_drawdown_interval(x))"}, {{"x", "series"}, {"y", "series"}});
  const c::NodeInfo *fit = nullptr, *another_fit = nullptr, *interval = nullptr;
  for (const auto &node : records->nodes) {
    if (node.node.kind != g::NodeKind::operation) continue;
    if (node.node.opcode == static_cast<std::uint16_t>(o::Op::linear_fit)) {
      if (!fit) fit = &node;
      else another_fit = &node;
    }
    if (node.node.opcode == static_cast<std::uint16_t>(o::Op::last_drawdown_interval))
      interval = &node;
  }
  check(fit && another_fit && interval && fit->inferred_type == another_fit->inferred_type &&
        fit->inferred_type != interval->inferred_type &&
        t::same_structure(fit->type(), interval->type()) &&
        fit->type().record_tag != interval->type().record_tag &&
        fit->type().fields != interval->type().fields,
        "real nominal records lost tag/field identity or sharing");

  auto key = t::ValueType::series("T", "category");
  key.dtype = t::DType::int64;
  const auto values = t::ValueType::series("T", "reported_nav", "test_basis");
  auto grouped = c::compile({"key", "group_apply(mean(x),key)", "x"},
      std::vector<t::Variable>{{"x", values}, {"key", key}}, false, {}, {}, 0, 100000000, true);
  check(t::same_type(root(*grouped, 0).type(), key) &&
        t::same_type(root(*grouped, 1).type(), values) &&
        t::same_type(root(*grouped, 2).type(), values),
        "group output type construction modified a captured/input type");

  auto iteration = c::compile({"1", "iteration_count(iterate(iterate_x+1,1,0,2))",
      "iteration_residual(iterate(iterate_x+1,1,0,2))"},
      std::vector<t::Variable>{{"x", t::ValueType::series()}},
      false, {}, {}, 0, 100000000, true);
  check(root(*iteration, 0).type().dtype == t::DType::float64 &&
        root(*iteration, 1).type().dtype == t::DType::int64 &&
        root(*iteration, 2).type().dtype == t::DType::float64 &&
        root(*iteration, 0).inferred_type == root(*iteration, 2).inferred_type &&
        root(*iteration, 0).inferred_type != root(*iteration, 1).inferred_type,
        "iteration projection type construction modified a shared scalar type");

  // Many unique admitted schemas exercise the hash-indexed lookup path; this
  // asserts correct identities, rather than using a brittle timing threshold.
  variables.clear(); expressions.clear();
  for (unsigned index = 0; index < 512; ++index)
    add("v" + std::to_string(index), t::ValueType::vector("N" + std::to_string(index)));
  auto unique = c::compile(expressions, variables, false, {}, {}, 0, 100000000, true);
  std::unordered_set<const t::ValueType *> owners;
  for (std::size_t index = 0; index < variables.size(); ++index) {
    const auto &node = root(*unique, index);
    check(node.inferred_type && t::same_type(node.type(), variables[index].type),
          "unique admitted schema changed during type sharing");
    owners.insert(node.inferred_type.get());
  }
  check(owners.size() == variables.size(), "unique admitted schemas were merged");
}
} // namespace node_type_sharing
