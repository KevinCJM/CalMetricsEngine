#include "calmetrics_engine/compiler.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
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
bool array(V v) { return v == V::series || v == V::mask_series; }
V execution_class(const typed::ValueType &value) {
  if (value.kind == typed::ValueKind::record)
    return value.record_tag == "fit" ? V::fit : V::interval;
  if (value.is_mask())
    return value.is_scalar() ? V::mask_scalar : V::mask_series;
  return value.is_scalar() ? V::scalar : V::series;
}
bool ident_start(unsigned char c) {
  return std::isalpha(c) || c == '_' || c >= 128;
}
bool ident_rest(unsigned char c) { return ident_start(c) || std::isdigit(c); }


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

PhysicalCost physical_cost(const CompiledGraph &graph,
                           const std::vector<std::uint8_t> &active) {
  const auto &nodes = graph.nodes;
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
    if (!active[i])
      continue;
    const auto &node = nodes[i];
    if (node.node.kind == graph::NodeKind::rolling_scope) {
      const auto scope_index = node.node.input_index;
      if (scope_index < graph.program.rolling_scopes.size())
        result.linear_per_observation +=
            4.0 * std::max<std::size_t>(
                      1, graph.program.rolling_scopes[scope_index].body_node_count);
      continue;
    }
    if (node.node.kind != graph::NodeKind::operation)
      continue;
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
  if (!graph.program.rolling_scopes.empty()) {
    std::vector<std::uint8_t> active(node_count, 1);
    BranchInfo branch;
    branch.root_indices.resize(graph.program.roots.size());
    std::iota(branch.root_indices.begin(), branch.root_indices.end(), 0);
    branch.source_nodes.resize(node_count);
    std::iota(branch.source_nodes.begin(), branch.source_nodes.end(), 0);
    branch.program = graph.program;
    branch.cost = physical_cost(graph, active);
    graph.physical_cost = branch.cost;
    graph.branches = {std::move(branch)};
    return;
  }
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
    branch.cost = physical_cost(graph, active);
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
  struct LogicalWindow {
    std::uint32_t values = 0;
    std::uint32_t width = 0;
    std::uint32_t minimum = 0;
    bool has_minimum = false;
  };
  std::shared_ptr<CompiledGraph> result = std::make_shared<CompiledGraph>();
  struct Binding {
    bool input = false;
    std::uint16_t index = 0;
    typed::ValueType type = typed::ValueType::scalar();
  };
  std::unordered_map<std::string, Binding> bindings;
  std::unordered_map<std::string, std::uint32_t> seen;
  std::unordered_map<std::uint32_t, LogicalWindow> logical_windows;
  bool rolling_body = false;

  std::uint32_t append(NodeInfo info) {
    ++result->raw_node_count;
    check(result->nodes.size() < max_nodes, "graph node limit exceeded");
    info.node_id = static_cast<std::uint32_t>(result->nodes.size());
    info.last_use = info.node_id;
    result->nodes.push_back(std::move(info));
    return static_cast<std::uint32_t>(result->nodes.size() - 1);
  }

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
    auto entry = bindings.find(name);
    check(entry != bindings.end(), "unknown variable: " + name);
    if (entry->second.index == std::numeric_limits<std::uint16_t>::max()) {
      auto &names = entry->second.input ? result->input_names
                                        : result->parameter_names;
      check(names.size() < std::numeric_limits<std::uint16_t>::max(),
            "too many variable bindings");
      entry->second.index = static_cast<std::uint16_t>(names.size());
      names.push_back(name);
    }
    NodeInfo info;
    info.node.kind = entry->second.input ? graph::NodeKind::input
                                         : graph::NodeKind::parameter;
    info.inferred_type = entry->second.type;
    info.value_class = execution_class(info.inferred_type);
    info.node.input_index = entry->second.index;
    return intern(info);
  }

  std::uint32_t call(O op, const std::vector<std::uint32_t> &parents) {
    const auto &spec = ops::lookup(static_cast<std::uint16_t>(op));
    if (rolling_body && op == O::recursive_smooth)
      throw CompileError("ROLLING_INTERVAL_POLICY_REQUIRED: recursive_smooth requires history outside the window");
    if (rolling_body && spec.family == ops::Family::rolling)
      throw CompileError("ROLLING_NESTED_SCOPE_UNSUPPORTED: rolling body cannot contain rolling operators");
    if (!parents.empty()) {
      const auto found = logical_windows.find(parents[0]);
      if (found != logical_windows.end()) {
        const auto &window = found->second;
        std::string rolling_name;
        if (op == O::mean)
          rolling_name = "rolling_mean";
        else if (op == O::min_value)
          rolling_name = "rolling_min";
        else if (op == O::max_value)
          rolling_name = "rolling_max";
        else if (op == O::std || op == O::variance)
          rolling_name = "rolling_std";
        else
          throw CompileError("ROLLING_REDUCER_UNSUPPORTED: rolling_window requires mean/std/variance/min/max");
        std::vector<std::uint32_t> lowered{window.values, window.width};
        if (op == O::std || op == O::variance) {
          lowered.push_back(parents.size() == 2 ? parents[1] : constant(1));
          lowered.push_back(window.has_minimum ? window.minimum : window.width);
        } else
          lowered.push_back(window.has_minimum ? window.minimum : window.width);
        const auto rolling = call(ops::lookup(rolling_name).op, lowered);
        return op == O::variance ? call(O::multiply, {rolling, rolling})
                                 : rolling;
      }
    }
    check(parents.size() >= spec.min_args && parents.size() <= spec.max_args,
          std::string(spec.name) + ": incorrect argument count");
    std::vector<typed::ValueType> types;
    for (auto parent : parents)
      types.push_back(result->nodes[parent].inferred_type);
    typed::ValueType inferred;
    try {
      inferred = typed::infer(spec, types);
    } catch (const typed::Error &error) {
      throw CompileError(error.what());
    }
    const auto type = execution_class(inferred);
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
    info.inferred_type = inferred;
    info.cost_model = cost(spec, type);
    info.simd_eligible = ops::simd_eligible(op);
    return intern(info);
  }

  std::uint32_t rolling_window(const std::vector<std::uint32_t> &parents) {
    if (rolling_body)
      throw CompileError("ROLLING_NESTED_SCOPE_UNSUPPORTED: rolling body cannot contain rolling_window");
    check(parents.size() == 2 || parents.size() == 3,
          "rolling_window: incorrect argument count");
    const auto &values = result->nodes[parents[0]].inferred_type;
    const auto &width = result->nodes[parents[1]].inferred_type;
    if (values.kind != typed::ValueKind::series || !values.is_numeric())
      throw CompileError("TYPE_MISMATCH: rolling_window requires numeric time series");
    if (!width.is_scalar() || !width.is_numeric() ||
        (width.semantic_dimension != "count" &&
         width.semantic_dimension != "dimensionless"))
      throw CompileError(
          "INVALID_PARAMETER: rolling_window width must be count/dimensionless scalar");
    if (parents.size() == 3) {
      const auto &minimum = result->nodes[parents[2]].inferred_type;
      if (!minimum.is_scalar() || !minimum.is_numeric() ||
          (minimum.semantic_dimension != "count" &&
           minimum.semantic_dimension != "dimensionless"))
        throw CompileError(
            "INVALID_PARAMETER: rolling_window min_periods must be count/dimensionless scalar");
    }
    check(logical_windows.size() < 65536, "too many logical rolling windows");
    const auto id = std::numeric_limits<std::uint32_t>::max() -
                    static_cast<std::uint32_t>(logical_windows.size());
    logical_windows.emplace(id, LogicalWindow{parents[0], parents[1],
                                               parents.size() == 3 ? parents[2] : 0,
                                               parents.size() == 3});
    return id;
  }

  std::uint32_t rolling_apply(const std::string &body_source,
                              const std::vector<std::uint32_t> &arguments);

  std::uint32_t call(const std::string &name,
                     const std::vector<std::uint32_t> &parents) {
    if (name == "rolling_window")
      return rolling_window(parents);
    try {
      return call(ops::lookup(typed::canonical_operator_name(name)).op, parents);
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
        if ((info.node.kind != graph::NodeKind::operation &&
             info.node.kind != graph::NodeKind::rolling_scope) ||
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
    result->program.output_kind = result->output_kind;
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
    auto update_text = [&](const std::string &value) {
      for (unsigned char x : value)
        update(x);
      update(0);
    };
    for (const auto &v : result->variables) {
      update_text(v.first);
      update_text(v.second);
    }
    for (const auto &v : result->variable_types) {
      update_text(v.name);
      update(static_cast<std::uint8_t>(v.type.kind));
      update(static_cast<std::uint8_t>(v.type.dtype));
      update_text(v.type.semantic_dimension);
      update_text(v.type.price_basis);
      for (const auto &axis : v.type.axes)
        update_text(axis);
      update(0xff);
      for (const auto &shape : v.type.shape)
        update_text(shape);
      update(0xfe);
    }
    std::ostringstream out;
    out << "native-2-" << std::hex << std::setfill('0') << std::setw(16) << a
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
    if (name == "rolling_apply") {
      check(!builder.rolling_body,
            "ROLLING_NESTED_SCOPE_UNSUPPORTED: nested rolling_apply");
      space();
      const auto body_start = position;
      std::size_t nested = 0, comma = std::string::npos;
      for (std::size_t cursor = position; cursor < source.size(); ++cursor) {
        const char token = source[cursor];
        if (token == '(')
          ++nested;
        else if (token == ')') {
          if (nested == 0)
            break;
          --nested;
        } else if (token == ',' && nested == 0) {
          comma = cursor;
          break;
        }
      }
      check(comma != std::string::npos && comma > body_start,
            "rolling_apply requires a body expression and width");
      auto body_source = source.substr(body_start, comma - body_start);
      position = comma + 1;
      std::vector<std::uint32_t> args;
      do {
        check(args.size() < 4, "rolling_apply has too many scope arguments");
        args.push_back(expression());
      } while (take(","));
      check(take(")"), "expected ')' after rolling_apply");
      return builder.rolling_apply(body_source, args);
    }
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

std::uint32_t Builder::rolling_apply(
    const std::string &body_source,
    const std::vector<std::uint32_t> &arguments) {
  check(arguments.size() >= 1 && arguments.size() <= 4,
        "rolling_apply: incorrect argument count");

  std::vector<std::uint32_t> scope_arguments = arguments;
  const bool has_system_context =
      bindings.find("observation_dates") != bindings.end() &&
      bindings.find("annual_risk_free_rate_decimal") != bindings.end();
  if (has_system_context &&
      (scope_arguments.size() == 1 || scope_arguments.size() == 2)) {
    const auto width = scope_arguments[0];
    const bool has_minimum_argument = scope_arguments.size() == 2;
    const auto minimum = has_minimum_argument ? scope_arguments[1] : 0;
    scope_arguments = {width, variable("observation_dates"),
                       variable("annual_risk_free_rate_decimal")};
    if (has_minimum_argument)
      scope_arguments.push_back(minimum);
  }

  const bool date_context = scope_arguments.size() >= 3;
  const bool has_minimum =
      scope_arguments.size() == 2 || scope_arguments.size() == 4;
  const auto width_node = scope_arguments[0];
  const auto minimum_node = has_minimum ? scope_arguments.back() : 0;
  const auto dates_node = date_context ? scope_arguments[1] : 0;
  const auto annual_node = date_context ? scope_arguments[2] : 0;
  if (date_context && has_system_context) {
    const auto expected_dates = variable("observation_dates");
    const auto expected_annual = variable("annual_risk_free_rate_decimal");
    if (dates_node != expected_dates || annual_node != expected_annual)
      throw CompileError(
          "ROLLING_CONTEXT_BINDING_INVALID: rolling_apply must use system observation_dates and annual_risk_free_rate_decimal context");
  }

  const auto &width_type = result->nodes[width_node].inferred_type;
  if (!width_type.is_scalar() || !width_type.is_numeric() ||
      (width_type.semantic_dimension != "count" &&
       width_type.semantic_dimension != "dimensionless"))
    throw CompileError(
        "INVALID_PARAMETER: rolling_apply width must be count/dimensionless scalar");
  if (has_minimum) {
    const auto &minimum_type = result->nodes[minimum_node].inferred_type;
    if (!minimum_type.is_scalar() || !minimum_type.is_numeric() ||
        (minimum_type.semantic_dimension != "count" &&
         minimum_type.semantic_dimension != "dimensionless"))
      throw CompileError(
          "INVALID_PARAMETER: rolling_apply min_periods must be count/dimensionless scalar");
  }
  if (date_context) {
    const auto &dates_type = result->nodes[dates_node].inferred_type;
    if (dates_type.kind != typed::ValueKind::series ||
        dates_type.semantic_dimension != "date")
      throw CompileError(
          "TYPE_MISMATCH: rolling_apply dates must be series<time><date>");
    const auto &annual_type = result->nodes[annual_node].inferred_type;
    if (!annual_type.is_scalar() || !annual_type.is_numeric())
      throw CompileError(
          "TYPE_MISMATCH: rolling_apply annual rate must be numeric scalar");
  }

  Builder body;
  body.rolling_body = true;
  body.result->variables = result->variables;
  body.result->variable_types = result->variable_types;
  for (const auto &item : bindings) {
    auto binding = item.second;
    binding.index = std::numeric_limits<std::uint16_t>::max();
    body.bindings.emplace(item.first, std::move(binding));
  }
  Parser body_parser(body, body_source);
  const auto body_root = body_parser.parse();
  const auto &body_type = body.result->nodes[body_root].inferred_type;
  if (!body_type.is_scalar() || !body_type.is_numeric() ||
      body_type.semantic_dimension == "date")
    throw CompileError(
        "ROLLING_BODY_NOT_SCALAR: rolling_apply body must be numeric scalar");
  bool has_aggregation = false;
  for (const auto &node : body.result->nodes) {
    if (node.node.kind != graph::NodeKind::operation)
      continue;
    const auto op = static_cast<O>(node.node.opcode);
    if (op == O::first || op == O::last || op == O::length ||
        op == O::argmin || op == O::argmax || op == O::value_at)
      continue;
    const bool consumes_time = std::any_of(
        node.node.parents.begin(),
        node.node.parents.begin() + node.node.parent_count,
        [&](std::uint32_t parent) {
          const auto &type = body.result->nodes[parent].inferred_type;
          return std::find(type.axes.begin(), type.axes.end(), "time") !=
                 type.axes.end();
        });
    if (consumes_time &&
        (node.inferred_type.is_scalar() ||
         node.inferred_type.kind == typed::ValueKind::record)) {
      has_aggregation = true;
      break;
    }
  }
  if (!has_aggregation)
    throw CompileError(
        "ROLLING_AGGREGATION_REQUIRED: rolling_apply body must contain an interval aggregation");
  body.result->program.roots.push_back(body_root);
  body.result->output_kind = graph::OutputKind::scalar;
  body.finish();

  graph::RollingScope scope;
  scope.body = std::make_shared<graph::Program>(body.result->program);
  scope.width_node = width_node;
  scope.min_periods_node = minimum_node;
  scope.dates_node = dates_node;
  scope.annual_rate_node = annual_node;
  scope.has_min_periods = has_minimum;
  scope.has_date_context = date_context;
  scope.body_node_count = body.result->nodes.size();

  std::vector<const typed::ValueType *> body_input_types;
  for (const auto &name : body.result->input_names) {
    const auto outer = variable(name);
    const auto &type = result->nodes[outer].inferred_type;
    if (type.kind != typed::ValueKind::series || !type.is_numeric())
      throw CompileError(
          "ROLLING_BODY_ARRAY_TYPE: rolling_apply body arrays must be numeric time series");
    scope.input_nodes.push_back(outer);
    body_input_types.push_back(&type);
    if (name == "returns") {
      scope.has_returns = true;
      scope.returns_input =
          static_cast<std::int32_t>(scope.input_nodes.size() - 1);
    } else if (name == "log_returns") {
      scope.has_returns = true;
      if (scope.returns_input < 0)
        scope.returns_input =
            static_cast<std::int32_t>(scope.input_nodes.size() - 1);
    }
  }
  scope.array_count = scope.input_nodes.size();
  if (scope.input_nodes.empty())
    throw CompileError(
        "ROLLING_BODY_NO_SERIES: rolling_apply body must depend on a time series");
  for (const auto *type : body_input_types) {
    const bool preceding =
        scope.has_returns && !type->shape.empty() && type->shape[0] == "L";
    scope.input_preceding.push_back(preceding ? 1 : 0);
    scope.needs_preceding_observation =
        scope.needs_preceding_observation || preceding;
  }

  bool needs_dates = false;
  for (const auto &name : body.result->parameter_names) {
    graph::RollingParameterBinding binding;
    if (name == "observation_count")
      binding.kind = graph::RollingParameterKind::observation_count;
    else if (name == "window_elapsed_days") {
      binding.kind = graph::RollingParameterKind::window_elapsed_days;
      needs_dates = true;
    } else if (name == "risk_free_return_window") {
      binding.kind = graph::RollingParameterKind::risk_free_return_window;
      needs_dates = true;
    } else {
      binding.kind = graph::RollingParameterKind::outer_node;
      binding.node = variable(name);
    }
    scope.parameter_bindings.push_back(binding);
  }
  if (needs_dates && !date_context)
    throw CompileError(
        "ROLLING_CONTEXT_REQUIRED: elapsed/risk-free window context requires dates and annual rate");

  std::vector<std::uint32_t> dependencies = scope.input_nodes;
  auto add_dependency = [&](std::uint32_t node) {
    if (std::find(dependencies.begin(), dependencies.end(), node) ==
        dependencies.end())
      dependencies.push_back(node);
  };
  add_dependency(width_node);
  if (has_minimum)
    add_dependency(minimum_node);
  if (date_context) {
    add_dependency(dates_node);
    add_dependency(annual_node);
  }
  for (const auto &binding : scope.parameter_bindings)
    if (binding.kind == graph::RollingParameterKind::outer_node)
      add_dependency(binding.node);
  check(dependencies.size() <= graph::Node{}.parents.size(),
        "rolling_apply has too many captured dependencies");

  NodeInfo info;
  info.node.kind = graph::NodeKind::rolling_scope;
  info.node.input_index =
      static_cast<std::uint16_t>(result->program.rolling_scopes.size());
  info.node.parent_count = static_cast<std::uint8_t>(dependencies.size());
  std::copy(dependencies.begin(), dependencies.end(), info.node.parents.begin());
  info.inferred_type = typed::ValueType::series(
      "T", body_type.semantic_dimension, body_type.price_basis);
  info.value_class = V::series;
  info.cost_model = "rolling_scope";
  result->program.rolling_scopes.push_back(std::move(scope));
  return append(std::move(info));
}

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
  std::vector<std::uint8_t> blob() {
    const auto count = static_cast<std::size_t>(get(8));
    check(count <= bytes.size() - position, "truncated native plan blob");
    std::vector<std::uint8_t> result(
        bytes.begin() + static_cast<std::ptrdiff_t>(position),
        bytes.begin() + static_cast<std::ptrdiff_t>(position + count));
    position += count;
    return result;
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
  case graph::NodeKind::rolling_scope:
    return "rolling_scope";
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
  std::vector<typed::Variable> typed_variables;
  typed_variables.reserve(variables.size());
  for (const auto &item : variables) {
    if (item.second == "series")
      typed_variables.push_back({item.first, typed::ValueType::series()});
    else if (item.second == "scalar")
      typed_variables.push_back({item.first, typed::ValueType::scalar()});
    else
      throw CompileError("variable type must be series or scalar");
  }
  auto result = compile(expressions, typed_variables);
  result->variables = variables;
  return result;
}

std::shared_ptr<CompiledGraph>
compile(const std::vector<std::string> &expressions,
        const std::vector<typed::Variable> &variables) {
  check(!variables.empty() && variables.size() < 65536,
        "invalid variable count");
  check(!expressions.empty() && expressions.size() <= 4096,
        "invalid expression count");
  Builder builder;
  builder.result->expressions = expressions;
  builder.result->variable_types = variables;
  for (const auto &item : variables) {
    check(!item.name.empty() &&
              ident_start(static_cast<unsigned char>(item.name[0])) &&
              std::all_of(item.name.begin() + 1, item.name.end(),
                          [](unsigned char c) { return ident_rest(c); }),
          "invalid variable name");
    try {
      item.type.validate();
    } catch (const typed::Error &error) {
      throw CompileError(error.what());
    }
    if (item.type.is_mask())
      throw CompileError("typed graph input variables must be numeric");
    const bool input = !item.type.is_scalar();
    if (input && item.type.kind != typed::ValueKind::series)
      throw CompileError(
          "typed interval graph currently binds scalar or time-series variables");
    auto &names = input ? builder.result->input_names
                        : builder.result->parameter_names;
    check(names.size() < std::numeric_limits<std::uint16_t>::max(),
          "too many variable bindings");
    const bool rolling_context =
        !input && (item.name == "observation_count" ||
                   item.name == "window_elapsed_days" ||
                   item.name == "risk_free_return_window");
    Builder::Binding binding;
    binding.input = input;
    binding.index = rolling_context
                        ? std::numeric_limits<std::uint16_t>::max()
                        : static_cast<std::uint16_t>(names.size());
    binding.type = item.type;
    check(builder.bindings.emplace(item.name, std::move(binding)).second,
          "duplicate variable");
    if (!rolling_context)
      names.push_back(item.name);
    builder.result->variables.push_back(
        {item.name, input ? "series" : "scalar"});
  }
  std::size_t source_bytes = 0;
  std::optional<graph::OutputKind> output_kind;
  for (const auto &source : expressions) {
    check(source.size() <= max_source_bytes - source_bytes,
          "expression byte limit exceeded");
    source_bytes += source.size();
    Parser parser(builder, source);
    auto root = parser.parse();
    if (builder.logical_windows.find(root) != builder.logical_windows.end())
      throw CompileError(
          "PUBLIC_ROOT_TYPE: rolling_window is a compiler-only intermediate");
    const auto &type = builder.result->nodes[root].inferred_type;
    graph::OutputKind current;
    if (type.is_scalar() &&
        (type.is_numeric() || type.is_mask()))
      current = graph::OutputKind::scalar;
    else if (type.kind == typed::ValueKind::series && type.is_numeric()) {
      check(type.shape.size() == 1 && type.shape[0] == "T",
            "SERIES_ROOT_ALIGNMENT: public series roots must preserve the interval time axis");
      current = graph::OutputKind::series;
    } else
      throw CompileError(
          "PUBLIC_ROOT_TYPE: roots must be numeric scalar/mask or aligned numeric series");
    if (output_kind && *output_kind != current)
      throw CompileError(
          "MIXED_ROOT_TYPES: scalar and series roots cannot share one execution graph");
    output_kind = current;
    builder.result->program.roots.push_back(root);
  }
  builder.result->output_kind = *output_kind;
  builder.finish();
  return builder.result;
}

std::vector<std::uint8_t> encode_program(const graph::Program &p) {
  p.validate();
  check(p.nodes.size() <= max_nodes && p.roots.size() <= 4096 &&
            p.rolling_scopes.size() <= 4096,
        "plan size limit exceeded");
  std::vector<std::uint8_t> bytes;
  bytes.reserve(48 + p.nodes.size() * 48);
  put(bytes, 0x434d4547, 4);
  put(bytes, 2, 4);
  put(bytes, p.nodes.size(), 4);
  put(bytes, p.roots.size(), 4);
  put(bytes, p.input_count, 4);
  put(bytes, p.parameter_count, 4);
  put(bytes, p.numeric_slots, 4);
  put(bytes, p.mask_slots, 4);
  put(bytes, static_cast<std::uint8_t>(p.output_kind), 1);
  put(bytes, p.rolling_scopes.size(), 4);
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
    for (std::size_t i = 0; i < n.parent_count; ++i)
      put(bytes, n.parents[i], 4);
  }
  for (const auto &scope : p.rolling_scopes) {
    const auto body = encode_program(*scope.body);
    put(bytes, body.size(), 8);
    bytes.insert(bytes.end(), body.begin(), body.end());
    put(bytes, scope.input_nodes.size(), 4);
    for (auto node : scope.input_nodes)
      put(bytes, node, 4);
    put(bytes, scope.input_preceding.size(), 4);
    for (auto value : scope.input_preceding)
      put(bytes, value, 1);
    put(bytes, scope.parameter_bindings.size(), 4);
    for (const auto &binding : scope.parameter_bindings) {
      put(bytes, static_cast<std::uint8_t>(binding.kind), 1);
      put(bytes, binding.node, 4);
    }
    put(bytes, scope.width_node, 4);
    put(bytes, scope.min_periods_node, 4);
    put(bytes, scope.dates_node, 4);
    put(bytes, scope.annual_rate_node, 4);
    put(bytes, scope.has_min_periods, 1);
    put(bytes, scope.has_date_context, 1);
    put(bytes, scope.has_returns, 1);
    put(bytes, scope.needs_preceding_observation, 1);
    put(bytes, static_cast<std::uint32_t>(scope.returns_input + 1), 4);
    put(bytes, scope.body_node_count, 8);
    put(bytes, scope.array_count, 8);
  }
  for (auto root : p.roots)
    put(bytes, root, 4);
  return bytes;
}

graph::Program decode_program(const std::vector<std::uint8_t> &bytes) {
  Reader in{bytes};
  check(in.get(4) == 0x434d4547, "unsupported native plan magic");
  const auto version = in.get(4);
  check(version == 1 || version == 2, "unsupported native plan version");
  const auto count = in.get(4), roots = in.get(4);
  check(count > 0 && count <= max_nodes && roots > 0 && roots <= 4096,
        "invalid native plan counts");
  graph::Program p;
  p.input_count = in.get(4);
  p.parameter_count = in.get(4);
  p.numeric_slots = in.get(4);
  p.mask_slots = in.get(4);
  std::size_t scope_count = 0;
  if (version >= 2) {
    const auto output_kind = in.get(1);
    check(output_kind <= 1, "invalid native output kind");
    p.output_kind = static_cast<graph::OutputKind>(output_kind);
    scope_count = static_cast<std::size_t>(in.get(4));
    check(scope_count <= 4096, "invalid rolling scope count");
  }
  check(p.input_count < 65536 && p.parameter_count < 65536 &&
            p.numeric_slots <= count && p.mask_slots <= count,
        "native plan allocation limit");
  p.nodes.resize(static_cast<std::size_t>(count));
  for (auto &n : p.nodes) {
    const auto kind = in.get(1);
    check(kind <= (version == 1 ? 3 : 4), "invalid native node kind");
    n.kind = static_cast<graph::NodeKind>(kind);
    n.opcode = static_cast<std::uint16_t>(in.get(2));
    n.input_index = static_cast<std::uint16_t>(in.get(2));
    const auto bits = in.get(8);
    std::memcpy(&n.constant, &bits, 8);
    n.parent_count = static_cast<std::uint8_t>(in.get(1));
    check(n.parent_count <= n.parents.size(), "invalid native arity");
    const auto storage = in.get(1);
    check(storage <= 2, "invalid native storage");
    n.storage = static_cast<graph::StorageKind>(storage);
    n.slot = static_cast<std::uint32_t>(in.get(4));
    const auto parent_slots = version == 1 ? 4u : n.parent_count;
    for (std::size_t i = 0; i < parent_slots; ++i) {
      const auto parent = static_cast<std::uint32_t>(in.get(4));
      if (i < n.parents.size())
        n.parents[i] = parent;
    }
  }
  for (std::size_t i = 0; i < scope_count; ++i) {
    graph::RollingScope scope;
    scope.body = std::make_shared<graph::Program>(decode_program(in.blob()));
    const auto inputs = static_cast<std::size_t>(in.get(4));
    check(inputs <= 32, "too many rolling scope inputs");
    for (std::size_t j = 0; j < inputs; ++j)
      scope.input_nodes.push_back(static_cast<std::uint32_t>(in.get(4)));
    const auto preceding = static_cast<std::size_t>(in.get(4));
    check(preceding == inputs, "rolling preceding metadata mismatch");
    for (std::size_t j = 0; j < preceding; ++j)
      scope.input_preceding.push_back(static_cast<std::uint8_t>(in.get(1)));
    const auto params = static_cast<std::size_t>(in.get(4));
    check(params <= 128, "too many rolling scope parameters");
    for (std::size_t j = 0; j < params; ++j) {
      const auto kind = in.get(1);
      check(kind <= 3, "invalid rolling parameter binding");
      graph::RollingParameterBinding binding;
      binding.kind = static_cast<graph::RollingParameterKind>(kind);
      binding.node = static_cast<std::uint32_t>(in.get(4));
      scope.parameter_bindings.push_back(binding);
    }
    scope.width_node = static_cast<std::uint32_t>(in.get(4));
    scope.min_periods_node = static_cast<std::uint32_t>(in.get(4));
    scope.dates_node = static_cast<std::uint32_t>(in.get(4));
    scope.annual_rate_node = static_cast<std::uint32_t>(in.get(4));
    scope.has_min_periods = in.get(1) != 0;
    scope.has_date_context = in.get(1) != 0;
    scope.has_returns = in.get(1) != 0;
    scope.needs_preceding_observation = in.get(1) != 0;
    scope.returns_input = static_cast<std::int32_t>(in.get(4)) - 1;
    scope.body_node_count = static_cast<std::size_t>(in.get(8));
    scope.array_count = static_cast<std::size_t>(in.get(8));
    p.rolling_scopes.push_back(std::move(scope));
  }
  for (std::size_t i = 0; i < roots; ++i)
    p.roots.push_back(static_cast<std::uint32_t>(in.get(4)));
  check(in.position == bytes.size(), "trailing native plan bytes");
  p.finalize();
  return p;
}
} // namespace calmetrics_engine::compiler
