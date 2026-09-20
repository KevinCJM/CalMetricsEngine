#include "calmetrics_engine/compiler.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace calmetrics_engine::compiler {
namespace {
using V = ValueClass;
using O = ops::Op;
constexpr std::size_t max_nodes = 65536;
constexpr std::size_t max_source_bytes = 1 << 20;
constexpr std::size_t max_depth = 128;

void check(bool ok, const std::string &message) {
  if (!ok)
    throw CompileError(message);
}
bool numeric(V v) { return v == V::scalar || v == V::series; }
bool mask(V v) { return v == V::mask_scalar || v == V::mask_series; }
bool array(V v) { return v == V::series || v == V::mask_series; }
bool ident_start(unsigned char c) {
  return std::isalpha(c) || c == '_' || c >= 128;
}
bool ident_rest(unsigned char c) { return ident_start(c) || std::isdigit(c); }

V infer(const ops::Spec &spec, const std::vector<V> &v) {
  const auto op = spec.op;
  auto fail = [&] {
    throw CompileError(std::string(spec.name) +
                       ": invalid value classes for interval DAG");
  };
  const bool any_array = std::any_of(v.begin(), v.end(), array);
  const bool all_numeric = std::all_of(v.begin(), v.end(), numeric);
  const bool all_series =
      std::all_of(v.begin(), v.end(), [](V x) { return x == V::series; });
  const auto numeric_result = any_array ? V::series : V::scalar;
  const auto mask_result = any_array ? V::mask_series : V::mask_scalar;
  switch (spec.family) {
  case ops::Family::elementwise:
    if (op == O::logical_and || op == O::logical_or) {
      if (!mask(v[0]) || v[0] != v[1])
        fail();
      return mask_result;
    }
    if (op == O::logical_not) {
      if (!mask(v[0]))
        fail();
      return mask_result;
    }
    if (op == O::finite_mask) {
      if (v[0] != V::series)
        fail();
      return V::mask_series;
    }
    if (op == O::where) {
      if (!mask(v[0]) || !numeric(v[1]) || !numeric(v[2]))
        fail();
      return numeric_result;
    }
    if (op == O::clip) {
      if (!numeric(v[0]) || v[1] != V::scalar || v[2] != V::scalar)
        fail();
      return v[0];
    }
    if (op == O::divide_or_default) {
      if (v != std::vector<V>{V::series, V::series, V::scalar})
        fail();
      return V::series;
    }
    if (!all_numeric)
      fail();
    if (op == O::equal || op == O::not_equal || op == O::less_than ||
        op == O::less_equal || op == O::greater_than || op == O::greater_equal)
      return mask_result;
    return numeric_result;
  case ops::Family::reduction:
    if (op == O::sum_time || op == O::mean_time || op == O::product_time ||
        op == O::variance_time || op == O::std_time || op == O::min_time ||
        op == O::max_time || op == O::sum_asset || op == O::mean_asset ||
        op == O::product_asset || op == O::variance_asset ||
        op == O::std_asset || op == O::min_asset || op == O::max_asset)
      throw CompileError(std::string(spec.name) +
                         ": matrix input is not supported in interval DAG");
    if (op == O::count_true || op == O::max_consecutive_true) {
      if (v[0] != V::mask_series)
        fail();
      return V::scalar;
    }
    if (std::string(spec.name).find("_where") != std::string::npos) {
      if (v[0] != V::series || v[1] != V::mask_series ||
          (v.size() == 3 && v[2] != V::scalar))
        fail();
      return V::scalar;
    }
    if (v[0] != V::series || (v.size() == 2 && v[1] != V::scalar))
      fail();
    return V::scalar;
  case ops::Family::sequence:
    if (v[0] != V::series || (v.size() == 2 && v[1] != V::scalar))
      fail();
    if (op == O::new_high_mask)
      return V::mask_series;
    if (op == O::first || op == O::last || op == O::length)
      return V::scalar;
    return V::series;
  case ops::Family::rolling:
    if (v[0] != V::series || !std::all_of(v.begin() + 1, v.end(),
                                          [](V x) { return x == V::scalar; }))
      fail();
    return V::series;
  case ops::Family::matrix:
    if ((op == O::dot || op == O::covariance || op == O::correlation) &&
        v.size() == 2 && all_series)
      return V::scalar;
    throw CompileError(
        std::string(spec.name) +
        ": use the direct matrix operator; interval DAG accepts series only");
  case ops::Family::regression:
    if (!all_series)
      fail();
    return op == O::linear_fit ? V::fit : V::scalar;
  case ops::Family::state:
    if (op == O::last_drawdown_interval) {
      if (v[0] != V::series)
        fail();
      return V::interval;
    }
    if (op == O::interval_start || op == O::interval_trough ||
        op == O::interval_recovery) {
      if (v[0] != V::interval)
        fail();
    } else if (op == O::fit_slope || op == O::fit_intercept ||
               op == O::fit_residual_sum_squares ||
               op == O::fit_total_sum_squares ||
               op == O::fit_observation_count) {
      if (v[0] != V::fit)
        fail();
    } else if (op == O::value_at) {
      if (v != std::vector<V>{V::series, V::scalar})
        fail();
    } else if (!std::all_of(v.begin(), v.end(),
                            [](V x) { return x == V::scalar; }))
      fail();
    return V::scalar;
  case ops::Family::composite:
    if (op == O::active_returns) {
      if (!all_numeric)
        fail();
      return numeric_result;
    }
    if (op == O::cumulative_return || op == O::total_return) {
      if (v[0] != V::series)
        fail();
      return op == O::cumulative_return ? V::series : V::scalar;
    }
    if (op == O::annualized_return) {
      if (v != std::vector<V>{V::series, V::scalar})
        fail();
      return V::scalar;
    }
    fail();
  }
  throw CompileError("unsupported operator family");
}

std::string cost(const ops::Spec &spec, V type) {
  if (spec.op == O::median || spec.op == O::quantile ||
      spec.op == O::median_where || spec.op == O::quantile_where)
    return "sort";
  if ((spec.family == ops::Family::state &&
       spec.op != O::last_drawdown_interval) ||
      (spec.family == ops::Family::elementwise && !array(type)))
    return "constant";
  return ops::family_name(spec.family);
}

double linear_factor(const NodeInfo &node) {
  if (node.cost_model == "elementwise")
    return node.simd_eligible ? 0.65 : 1.0;
  if (node.cost_model == "sequence")
    return 1.2;
  if (node.cost_model == "reduction" || node.cost_model == "composite")
    return 1.5;
  if (node.cost_model == "rolling" || node.cost_model == "matrix")
    return 2.0;
  if (node.cost_model == "regression")
    return 3.5;
  return 1.0;
}

PhysicalCost physical_cost(const std::vector<NodeInfo> &nodes,
                           const std::vector<std::uint8_t> &active) {
  PhysicalCost result;
  std::vector<std::uint16_t> summary(nodes.size(), 0), ordered(nodes.size(), 0);
  std::vector<std::uint8_t> has_mad(nodes.size(), 0);
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (!active[i] || nodes[i].node.kind != graph::NodeKind::operation ||
        !nodes[i].node.parent_count)
      continue;
    const auto op = ops::lookup(nodes[i].node.opcode).op;
    const auto source = nodes[i].node.parents[0];
    if (graph::summary_fusion_eligible(op)) {
      ++summary[source];
      has_mad[source] |= op == O::mean_absolute_deviation;
    }
    if (graph::order_fusion_eligible(op))
      ++ordered[source];
  }

  std::vector<std::uint8_t> summary_charged(nodes.size(), 0),
      ordered_charged(nodes.size(), 0);
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (!active[i] || nodes[i].node.kind != graph::NodeKind::operation)
      continue;
    const auto &node = nodes[i];
    const auto op = ops::lookup(node.node.opcode).op;
    if (node.node.parent_count) {
      const auto source = node.node.parents[0];
      if (graph::summary_fusion_eligible(op) && summary[source] > 1) {
        if (!summary_charged[source]) {
          result.linear_per_observation += 1.5;
          if (has_mad[source])
            result.linear_per_observation += 1.0;
          summary_charged[source] = 1;
        }
        continue;
      }
      if (graph::order_fusion_eligible(op) && ordered[source] > 1) {
        if (!ordered_charged[source]) {
          result.sort_nlogn += 2.0;
          ordered_charged[source] = 1;
        }
        continue;
      }
    }

    if (node.cost_model == "constant")
      result.constant_per_row += 1.0;
    else if (node.cost_model == "sort")
      result.sort_nlogn += 2.0;
    else {
      result.linear_per_observation += linear_factor(node);
      if (op == O::mean_absolute_deviation)
        result.linear_per_observation += 1.0;
    }
  }
  return result;
}

struct Dsu {
  std::vector<std::uint32_t> parent;
  explicit Dsu(std::size_t count) : parent(count) {
    std::iota(parent.begin(), parent.end(), 0);
  }
  std::uint32_t find(std::uint32_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }
  void unite(std::uint32_t a, std::uint32_t b) {
    a = find(a);
    b = find(b);
    if (a == b)
      return;
    if (a > b)
      std::swap(a, b);
    parent[b] = a;
  }
};

void build_physical_metadata(CompiledGraph &graph) {
  const auto node_count = graph.nodes.size();
  const auto root_count = graph.program.roots.size();
  std::vector<std::int32_t> owner(node_count, -1);
  std::vector<std::uint32_t> visited(node_count, 0);
  std::uint32_t generation = 0;
  Dsu dsu(root_count);

  for (std::uint32_t root_index = 0; root_index < root_count; ++root_index) {
    ++generation;
    std::vector<std::uint32_t> stack{graph.program.roots[root_index]};
    while (!stack.empty()) {
      const auto node_id = stack.back();
      stack.pop_back();
      if (visited[node_id] == generation)
        continue;
      visited[node_id] = generation;
      const auto &node = graph.program.nodes[node_id];
      if (node.kind == graph::NodeKind::operation) {
        if (owner[node_id] >= 0) {
          // This operation and all of its ancestors were already traversed
          // from another root. Unioning the roots is sufficient; walking the
          // shared prefix again adds no connectivity information.
          dsu.unite(root_index, static_cast<std::uint32_t>(owner[node_id]));
          continue;
        }
        owner[node_id] = static_cast<std::int32_t>(root_index);
      }
      for (std::size_t i = 0; i < node.parent_count; ++i)
        stack.push_back(node.parents[i]);
    }
  }

  std::unordered_map<std::uint32_t, std::uint32_t> summary_owner, order_owner;
  for (std::size_t i = 0; i < node_count; ++i) {
    const auto &node = graph.program.nodes[i];
    if (node.kind != graph::NodeKind::operation || !node.parent_count ||
        owner[i] < 0)
      continue;
    const auto op = ops::lookup(node.opcode).op;
    auto connect = [&](auto &owners) {
      const auto source = node.parents[0];
      const auto root = static_cast<std::uint32_t>(owner[i]);
      const auto [it, inserted] = owners.emplace(source, root);
      if (!inserted)
        dsu.unite(root, it->second);
    };
    if (graph::summary_fusion_eligible(op))
      connect(summary_owner);
    if (graph::order_fusion_eligible(op))
      connect(order_owner);
  }

  std::vector<std::vector<std::uint32_t>> components(root_count);
  for (std::uint32_t root = 0; root < root_count; ++root)
    components[dsu.find(root)].push_back(root);

  graph.branches.clear();
  graph.physical_cost = {};
  for (const auto &roots : components) {
    if (roots.empty())
      continue;
    std::vector<std::uint8_t> active(node_count, 0);
    std::vector<std::uint32_t> stack;
    for (auto root_index : roots)
      stack.push_back(graph.program.roots[root_index]);
    while (!stack.empty()) {
      const auto node_id = stack.back();
      stack.pop_back();
      if (active[node_id])
        continue;
      active[node_id] = 1;
      const auto &node = graph.program.nodes[node_id];
      for (std::size_t i = 0; i < node.parent_count; ++i)
        stack.push_back(node.parents[i]);
    }

    BranchInfo branch;
    branch.root_indices = roots;
    branch.cost = physical_cost(graph.nodes, active);
    std::vector<std::uint32_t> remap(node_count,
                                     std::numeric_limits<std::uint32_t>::max());
    std::vector<std::uint32_t> numeric_slots(
        graph.program.numeric_slots, std::numeric_limits<std::uint32_t>::max());
    std::vector<std::uint32_t> mask_slots(
        graph.program.mask_slots, std::numeric_limits<std::uint32_t>::max());
    std::uint32_t next_numeric_slot = 0, next_mask_slot = 0;
    for (std::size_t old_id = 0; old_id < node_count; ++old_id) {
      if (!active[old_id])
        continue;
      branch.source_nodes.push_back(static_cast<std::uint32_t>(old_id));
      remap[old_id] = static_cast<std::uint32_t>(branch.program.nodes.size());
      auto node = graph.program.nodes[old_id];
      for (std::size_t i = 0; i < node.parent_count; ++i) {
        check(remap[node.parents[i]] !=
                  std::numeric_limits<std::uint32_t>::max(),
              "branch dependency remap failed");
        node.parents[i] = remap[node.parents[i]];
      }
      if (node.storage == graph::StorageKind::numeric) {
        auto &slot = numeric_slots[node.slot];
        if (slot == std::numeric_limits<std::uint32_t>::max())
          slot = next_numeric_slot++;
        node.slot = slot;
      } else if (node.storage == graph::StorageKind::mask) {
        auto &slot = mask_slots[node.slot];
        if (slot == std::numeric_limits<std::uint32_t>::max())
          slot = next_mask_slot++;
        node.slot = slot;
      }
      branch.program.nodes.push_back(node);
    }
    branch.program.input_count = graph.program.input_count;
    branch.program.parameter_count = graph.program.parameter_count;
    branch.program.numeric_slots = next_numeric_slot;
    branch.program.mask_slots = next_mask_slot;
    for (auto root_index : roots)
      branch.program.roots.push_back(remap[graph.program.roots[root_index]]);
    branch.program.finalize();

    graph.physical_cost.constant_per_row += branch.cost.constant_per_row;
    graph.physical_cost.linear_per_observation +=
        branch.cost.linear_per_observation;
    graph.physical_cost.sort_nlogn += branch.cost.sort_nlogn;
    graph.branches.push_back(std::move(branch));
  }
}

template <class T> void key_append(std::string &key, T value) {
  key.append(reinterpret_cast<const char *>(&value), sizeof(value));
}

class Builder {
public:
  std::shared_ptr<CompiledGraph> result = std::make_shared<CompiledGraph>();
  std::unordered_map<std::string, std::pair<bool, std::uint16_t>> bindings;
  std::unordered_map<std::string, std::uint32_t> seen;

  std::uint32_t intern(NodeInfo info) {
    ++result->raw_node_count;
    std::string key;
    key_append(key, static_cast<std::uint8_t>(info.node.kind));
    key_append(key, info.node.opcode);
    key_append(key, info.node.input_index);
    key_append(key, info.node.constant);
    key_append(key, info.node.parent_count);
    for (std::size_t i = 0; i < info.node.parent_count; ++i)
      key_append(key, info.node.parents[i]);
    if (const auto found = seen.find(key); found != seen.end())
      return found->second;
    check(result->nodes.size() < max_nodes, "graph node limit exceeded");
    info.node_id = static_cast<std::uint32_t>(result->nodes.size());
    info.last_use = info.node_id;
    seen.emplace(std::move(key), info.node_id);
    result->nodes.push_back(info);
    return info.node_id;
  }

  std::uint32_t constant(double value) {
    check(std::isfinite(value), "non-finite constant");
    NodeInfo info;
    info.node.kind = graph::NodeKind::constant;
    info.node.constant = value;
    return intern(info);
  }

  std::uint32_t variable(const std::string &name) {
    const auto entry = bindings.find(name);
    check(entry != bindings.end(), "unknown variable: " + name);
    NodeInfo info;
    info.node.kind = entry->second.first ? graph::NodeKind::input
                                         : graph::NodeKind::parameter;
    info.value_class = entry->second.first ? V::series : V::scalar;
    info.node.input_index = entry->second.second;
    return intern(info);
  }

  std::uint32_t call(O op, const std::vector<std::uint32_t> &parents) {
    const auto &spec = ops::lookup(static_cast<std::uint16_t>(op));
    check(parents.size() >= spec.min_args && parents.size() <= spec.max_args,
          std::string(spec.name) + ": incorrect argument count");
    std::vector<V> types;
    for (auto parent : parents)
      types.push_back(result->nodes[parent].value_class);
    const auto type = infer(spec, types);
    if (op == O::annualized_return) {
      const auto one = constant(1);
      const auto total = call(O::total_return, {parents[0]});
      const auto growth = call(O::add, {total, one});
      const auto nonnegative = call(O::require_nonnegative, {growth});
      const auto periods = call(O::require_positive, {parents[1]});
      const auto count = call(O::length, {parents[0]});
      const auto exponent = call(O::divide, {periods, count});
      const auto annual = call(O::power, {nonnegative, exponent});
      return call(O::subtract, {annual, one});
    }
    if (op == O::linear_slope || op == O::linear_intercept ||
        op == O::linear_r_squared || op == O::regression_standard_error) {
      const auto fit = call(O::linear_fit, parents);
      if (op == O::linear_slope)
        return call(O::fit_slope, {fit});
      if (op == O::linear_intercept)
        return call(O::fit_intercept, {fit});
      const auto residual = call(O::fit_residual_sum_squares, {fit});
      if (op == O::linear_r_squared) {
        const auto total = call(O::fit_total_sum_squares, {fit});
        const auto guard = call(O::require_positive, {total});
        const auto ratio = call(O::divide, {residual, guard});
        return call(O::subtract, {constant(1), ratio});
      }
      const auto count = call(O::fit_observation_count, {fit});
      const auto df = call(O::subtract, {count, constant(2)});
      const auto guard = call(O::require_positive, {df});
      const auto variance = call(O::divide, {residual, guard});
      return call(O::sqrt, {variance});
    }
    NodeInfo info;
    info.node.kind = graph::NodeKind::operation;
    info.node.opcode = static_cast<std::uint16_t>(op);
    info.node.parent_count = static_cast<std::uint8_t>(parents.size());
    std::copy(parents.begin(), parents.end(), info.node.parents.begin());
    info.value_class = type;
    info.cost_model = cost(spec, type);
    info.simd_eligible = ops::simd_eligible(op);
    return intern(info);
  }

  std::uint32_t call(const std::string &name,
                     const std::vector<std::uint32_t> &parents) {
    try {
      return call(ops::lookup(name).op, parents);
    } catch (const ops::Error &) {
      throw CompileError("unknown canonical operator: " + name);
    }
  }

  void finish() {
    auto &nodes = result->nodes;
    for (const auto &info : nodes)
      for (std::size_t i = 0; i < info.node.parent_count; ++i)
        nodes[info.node.parents[i]].last_use =
            std::max(nodes[info.node.parents[i]].last_use, info.node_id);
    for (auto root : result->program.roots)
      nodes[root].last_use = static_cast<std::uint32_t>(nodes.size());
    // A borrowed view extends the lifetime of its backing slot, transitively.
    for (std::size_t i = nodes.size(); i-- > 0;) {
      const auto &info = nodes[i];
      if (info.node.kind == graph::NodeKind::operation &&
          info.node.opcode == static_cast<std::uint16_t>(O::lag)) {
        auto &source = nodes[info.node.parents[0]];
        source.last_use = std::max(source.last_use, info.last_use);
      }
    }
    auto allocate = [&](V type, graph::StorageKind storage) {
      std::vector<std::uint32_t> deaths;
      for (auto &info : nodes) {
        if (info.node.kind != graph::NodeKind::operation ||
            info.value_class != type ||
            info.node.opcode == static_cast<std::uint16_t>(O::lag))
          continue;
        std::size_t slot = 0;
        while (slot < deaths.size() && deaths[slot] >= info.node_id)
          ++slot;
        if (slot == deaths.size())
          deaths.push_back(info.last_use);
        else
          deaths[slot] = info.last_use;
        info.node.storage = storage;
        info.node.slot = static_cast<std::uint32_t>(slot);
      }
      return deaths.size();
    };
    result->program.numeric_slots =
        allocate(V::series, graph::StorageKind::numeric);
    result->program.mask_slots =
        allocate(V::mask_series, graph::StorageKind::mask);
    result->program.input_count = result->input_names.size();
    result->program.parameter_count = result->parameter_names.size();
    for (const auto &info : nodes)
      result->program.nodes.push_back(info.node);
    result->program.finalize();
    build_physical_metadata(*result);
    // Content fingerprint is a cache label, not an authentication hash. Native
    // plans use owner identity.
    auto encoded = encode_program(result->program);
    std::uint64_t a = 14695981039346656037ull, b = 7809847782465536322ull;
    auto update = [&](std::uint8_t x) {
      a = (a ^ x) * 1099511628211ull;
      b = (b ^ x) * 14029467366897019727ull;
    };
    for (auto x : encoded)
      update(x);
    for (const auto &v : result->variables) {
      for (unsigned char x : v.first)
        update(x);
      update(0);
      for (unsigned char x : v.second)
        update(x);
      update(0);
    }
    std::ostringstream out;
    out << "native-1-" << std::hex << std::setfill('0') << std::setw(16) << a
        << std::setw(16) << b;
    result->fingerprint = out.str();
  }
};

class Parser {
  Builder &builder;
  const std::string &source;
  std::size_t position = 0;
  std::size_t depth = 0;
  struct Depth {
    std::size_t &d;
    explicit Depth(std::size_t &value) : d(value) {
      check(++d <= max_depth, "expression nesting limit exceeded");
    }
    ~Depth() { --d; }
  };
  void space() {
    while (position < source.size()) {
      if (std::isspace(static_cast<unsigned char>(source[position])))
        ++position;
      else if (source[position] == '#') {
        while (position < source.size() && source[position] != '\n')
          ++position;
      } else
        break;
    }
  }
  bool take(const char *token) {
    space();
    const auto length = std::strlen(token);
    if (source.compare(position, length, token) != 0)
      return false;
    position += length;
    return true;
  }
  std::uint32_t atom() {
    space();
    check(position < source.size(), "expected an expression");
    if (take("(")) {
      auto value = expression();
      check(take(")"), "expected closing parenthesis");
      return value;
    }
    const auto first = static_cast<unsigned char>(source[position]);
    if (std::isdigit(first) || first == '.') {
      const auto start = position;
      bool digits = false;
      while (position < source.size() &&
             std::isdigit(static_cast<unsigned char>(source[position]))) {
        ++position;
        digits = true;
      }
      if (position < source.size() && source[position] == '.') {
        ++position;
        while (position < source.size() &&
               std::isdigit(static_cast<unsigned char>(source[position]))) {
          ++position;
          digits = true;
        }
      }
      check(digits, "invalid numeric literal");
      if (position < source.size() &&
          (source[position] == 'e' || source[position] == 'E')) {
        ++position;
        if (position < source.size() &&
            (source[position] == '+' || source[position] == '-'))
          ++position;
        const auto exponent = position;
        while (position < source.size() &&
               std::isdigit(static_cast<unsigned char>(source[position])))
          ++position;
        check(position > exponent, "invalid exponent");
      }
      std::istringstream number(source.substr(start, position - start));
      number.imbue(std::locale::classic());
      double value = 0;
      number >> value;
      check(!number.fail() && std::isfinite(value),
            "invalid or non-finite numeric literal");
      return builder.constant(value);
    }
    check(ident_start(first), "unsupported expression token");
    const auto start = position++;
    while (position < source.size() &&
           ident_rest(static_cast<unsigned char>(source[position])))
      ++position;
    auto name = source.substr(start, position - start);
    check(name != "True" && name != "False" && name != "None",
          "only numeric constants are allowed");
    if (!take("("))
      return builder.variable(name);
    std::vector<std::uint32_t> args;
    if (!take(")")) {
      do {
        check(args.size() < 4, "too many operator arguments");
        args.push_back(expression());
      } while (take(","));
      check(take(")"), "expected ')' after canonical call");
    }
    return builder.call(name, args);
  }
  std::uint32_t unary() {
    Depth guard(depth);
    if (take("+"))
      return unary();
    if (take("-")) {
      auto x = unary();
      return builder.call(O::negate, {x});
    }
    auto x = atom();
    if (take("**")) {
      auto y = unary();
      x = builder.call(O::power, {x, y});
    }
    return x;
  }
  std::uint32_t product() {
    auto x = unary();
    while (true) {
      if (take("*")) {
        auto y = unary();
        x = builder.call(O::multiply, {x, y});
      } else if (take("/")) {
        auto y = unary();
        x = builder.call(O::divide, {x, y});
      } else
        return x;
    }
  }
  std::uint32_t sum() {
    auto x = product();
    while (true) {
      if (take("+")) {
        auto y = product();
        x = builder.call(O::add, {x, y});
      } else if (take("-")) {
        auto y = product();
        x = builder.call(O::subtract, {x, y});
      } else
        return x;
    }
  }
  std::uint32_t expression() {
    Depth guard(depth);
    auto x = sum();
    const std::pair<const char *, O> comparisons[] = {
        {"==", O::equal},         {"!=", O::not_equal}, {"<=", O::less_equal},
        {">=", O::greater_equal}, {"<", O::less_than},  {">", O::greater_than}};
    for (const auto &pair : comparisons) {
      if (take(pair.first)) {
        auto y = sum();
        return builder.call(pair.second, {x, y});
      }
    }
    return x;
  }

public:
  Parser(Builder &b, const std::string &s) : builder(b), source(s) {}
  std::uint32_t parse() {
    const auto root = expression();
    space();
    check(position == source.size(),
          "unsupported trailing syntax or chained comparison");
    return root;
  }
};

void put(std::vector<std::uint8_t> &out, std::uint64_t value,
         std::size_t bytes) {
  for (std::size_t i = 0; i < bytes; ++i)
    out.push_back(static_cast<std::uint8_t>(value >> (i * 8)));
}
struct Reader {
  const std::vector<std::uint8_t> &bytes;
  std::size_t position = 0;
  std::uint64_t get(std::size_t n) {
    check(n <= bytes.size() - position, "truncated native plan");
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < n; ++i)
      value |= static_cast<std::uint64_t>(bytes[position++]) << (8 * i);
    return value;
  }
};
} // namespace

const char *class_name(V v) noexcept {
  switch (v) {
  case V::scalar:
    return "scalar";
  case V::series:
    return "series";
  case V::mask_scalar:
    return "mask_scalar";
  case V::mask_series:
    return "mask_series";
  case V::fit:
    return "fit";
  case V::interval:
    return "interval";
  }
  return "invalid";
}
const char *kind_name(graph::NodeKind k) noexcept {
  switch (k) {
  case graph::NodeKind::input:
    return "input";
  case graph::NodeKind::parameter:
    return "parameter";
  case graph::NodeKind::constant:
    return "constant";
  case graph::NodeKind::operation:
    return "operation";
  }
  return "invalid";
}
const char *storage_name(graph::StorageKind s) noexcept {
  switch (s) {
  case graph::StorageKind::inline_value:
    return "inline";
  case graph::StorageKind::numeric:
    return "numeric";
  case graph::StorageKind::mask:
    return "mask";
  }
  return "invalid";
}

std::shared_ptr<CompiledGraph>
compile(const std::vector<std::string> &expressions,
        const std::vector<std::pair<std::string, std::string>> &variables) {
  check(!variables.empty() && variables.size() < 65536,
        "invalid variable count");
  check(!expressions.empty() && expressions.size() <= 4096,
        "invalid expression count");
  Builder builder;
  builder.result->variables = variables;
  builder.result->expressions = expressions;
  for (const auto &item : variables) {
    check(!item.first.empty() &&
              ident_start(static_cast<unsigned char>(item.first[0])) &&
              std::all_of(item.first.begin() + 1, item.first.end(),
                          [](unsigned char c) { return ident_rest(c); }),
          "invalid variable name");
    check(item.second == "series" || item.second == "scalar",
          "variable type must be series or scalar");
    auto &names = item.second == "series" ? builder.result->input_names
                                          : builder.result->parameter_names;
    check(builder.bindings
              .emplace(item.first,
                       std::make_pair(item.second == "series",
                                      static_cast<std::uint16_t>(names.size())))
              .second,
          "duplicate variable");
    names.push_back(item.first);
  }
  std::size_t source_bytes = 0;
  for (const auto &source : expressions) {
    check(source.size() <= max_source_bytes - source_bytes,
          "expression byte limit exceeded");
    source_bytes += source.size();
    Parser parser(builder, source);
    auto root = parser.parse();
    auto type = builder.result->nodes[root].value_class;
    check(type == V::scalar || type == V::mask_scalar,
          "interval DAG roots must be scalar");
    builder.result->program.roots.push_back(root);
  }
  builder.finish();
  return builder.result;
}

std::vector<std::uint8_t> encode_program(const graph::Program &p) {
  p.validate();
  check(p.nodes.size() <= max_nodes && p.roots.size() <= 4096,
        "plan size limit exceeded");
  std::vector<std::uint8_t> bytes;
  bytes.reserve(32 + p.nodes.size() * 40);
  put(bytes, 0x434d4547, 4);
  put(bytes, 1, 4);
  put(bytes, p.nodes.size(), 4);
  put(bytes, p.roots.size(), 4);
  put(bytes, p.input_count, 4);
  put(bytes, p.parameter_count, 4);
  put(bytes, p.numeric_slots, 4);
  put(bytes, p.mask_slots, 4);
  for (const auto &n : p.nodes) {
    put(bytes, static_cast<std::uint8_t>(n.kind), 1);
    put(bytes, n.opcode, 2);
    put(bytes, n.input_index, 2);
    std::uint64_t constant_bits = 0;
    std::memcpy(&constant_bits, &n.constant, 8);
    put(bytes, constant_bits, 8);
    put(bytes, n.parent_count, 1);
    put(bytes, static_cast<std::uint8_t>(n.storage), 1);
    put(bytes, n.slot, 4);
    for (auto parent : n.parents)
      put(bytes, parent, 4);
  }
  for (auto root : p.roots)
    put(bytes, root, 4);
  return bytes;
}
graph::Program decode_program(const std::vector<std::uint8_t> &bytes) {
  Reader in{bytes};
  check(in.get(4) == 0x434d4547 && in.get(4) == 1,
        "unsupported native plan version");
  const auto count = in.get(4), roots = in.get(4);
  check(count > 0 && count <= max_nodes && roots > 0 && roots <= 4096,
        "invalid native plan counts");
  graph::Program p;
  p.input_count = in.get(4);
  p.parameter_count = in.get(4);
  p.numeric_slots = in.get(4);
  p.mask_slots = in.get(4);
  check(p.input_count < 65536 && p.parameter_count < 65536 &&
            p.numeric_slots <= count && p.mask_slots <= count,
        "native plan allocation limit");
  p.nodes.resize(static_cast<std::size_t>(count));
  for (auto &n : p.nodes) {
    const auto kind = in.get(1);
    check(kind <= 3, "invalid native node kind");
    n.kind = static_cast<graph::NodeKind>(kind);
    n.opcode = static_cast<std::uint16_t>(in.get(2));
    n.input_index = static_cast<std::uint16_t>(in.get(2));
    const auto bits = in.get(8);
    std::memcpy(&n.constant, &bits, 8);
    n.parent_count = static_cast<std::uint8_t>(in.get(1));
    check(n.parent_count <= 4, "invalid native arity");
    const auto storage = in.get(1);
    check(storage <= 2, "invalid native storage");
    n.storage = static_cast<graph::StorageKind>(storage);
    n.slot = static_cast<std::uint32_t>(in.get(4));
    for (auto &parent : n.parents)
      parent = static_cast<std::uint32_t>(in.get(4));
  }
  for (std::size_t i = 0; i < roots; ++i)
    p.roots.push_back(static_cast<std::uint32_t>(in.get(4)));
  check(in.position == bytes.size(), "trailing native plan bytes");
  p.finalize();
  return p;
}
} // namespace calmetrics_engine::compiler
