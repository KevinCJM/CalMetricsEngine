#include "calmetrics_engine/planner.hpp"
#include "calmetrics_engine/pointwise.hpp"
#include "calmetrics_engine/native_process.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace calmetrics_engine::planner {
namespace {
void require(bool ok, const char *message) {
  if (!ok)
    throw std::invalid_argument(message);
}
void mix(std::uint64_t &hash, std::uint64_t value) {
  hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
}
double row_work(const compiler::PhysicalCost &cost, std::size_t observations) {
  const auto n = static_cast<double>(observations);
  const auto sort =
      n *
      std::log2(static_cast<double>(std::max<std::size_t>(observations, 2)));
  return cost.constant_per_row + cost.linear_per_observation * n +
         cost.sort_nlogn * sort;
}
double geometry_work(const compiler::PhysicalCost &cost, const Geometry &g) {
  return cost.constant_per_row * static_cast<double>(g.rows) +
         cost.linear_per_observation * static_cast<double>(g.observations) +
         cost.sort_nlogn * g.sort_work;
}
// A scope can have a constant body with no array captures. Its selector or
// containing interval still determines the extent of nested materialized work.
std::size_t apply_observation_bound(const graph::ApplyScope &scope,
                                    const std::vector<ops::Shape> &shapes,
                                    std::size_t containing_extent) {
  if (scope.kind == graph::ApplyKind::bisect) return 0;
  if (scope.kind == graph::ApplyKind::block || shapes.empty()) return containing_extent;
  const auto &selector = shapes.at(scope.argument_nodes.at(0));
  return scope.kind == graph::ApplyKind::segment ? selector.dim[0] : selector.size();
}
// Add only work omitted by the original interval-length model. The same
// native shape walk owns both arena capacity and these materialized extents;
// coefficients and scheduler thresholds remain unchanged. This is an operation
// count estimate, not a hardware performance claim.
double typed_array_extra_work(const graph::Program &program,
                              const std::vector<ops::Value> &inputs,
                              std::size_t observations, bool full_work = false) {
  std::vector<ops::Shape> shapes;
  graph::required_array_capacity(program, inputs, observations, &shapes);
  const auto baseline = full_work ? 0.0 : static_cast<double>(observations);
  const auto sorting = [](double n) { return n * std::log2(std::max(2.0, n)); };
  std::unordered_set<std::uint32_t> summaries, orders;
  double extra = 0;
  for (std::size_t index = 0; index < program.nodes.size(); ++index) {
    const auto &node = program.nodes[index];
    if (node.kind == graph::NodeKind::apply_scope || node.kind == graph::NodeKind::rolling_scope) {
      const bool rolling = node.kind == graph::NodeKind::rolling_scope;
      const auto &captures = rolling ? program.rolling_scopes[node.input_index].input_nodes
                                     : program.apply_scopes[node.input_index].input_nodes;
      const auto &body = rolling ? *program.rolling_scopes[node.input_index].body
                                : *program.apply_scopes[node.input_index].body;
      std::size_t length = captures.empty()
          ? (rolling ? observations : apply_observation_bound(
              program.apply_scopes[node.input_index], shapes, observations)) : 0;
      std::vector<ops::Value> captured;
      for (auto source : captures) {
        ops::Value value;
        value.shape = shapes[source];
        length = std::max(length, value.shape.size());
        captured.push_back(value);
      }
      double repeats = 1.0;
      if (rolling) {
        const auto &scope = program.rolling_scopes[node.input_index];
        const auto &width = program.nodes[scope.width_node];
        const auto maximum_width = width.kind == graph::NodeKind::constant &&
            std::isfinite(width.constant) && width.constant >= 1 && width.constant <= 5000
            ? static_cast<std::size_t>(width.constant) : 5000;
        const auto window = std::min(length, maximum_width);
        for (std::size_t i = 0; i < captured.size(); ++i)
          captured[i].shape.dim[0] = window + (i < scope.input_preceding.size() && scope.input_preceding[i] ? 1 : 0);
        repeats = static_cast<double>(length);
        length = window + (scope.needs_preceding_observation ? 1 : 0);
      } else if (program.apply_scopes[node.input_index].kind == graph::ApplyKind::bisect ||
                 program.apply_scopes[node.input_index].kind == graph::ApplyKind::iterate) {
        const auto &scope = program.apply_scopes[node.input_index];
        const auto &iterations = program.nodes[scope.argument_nodes[scope.kind == graph::ApplyKind::iterate ? 2 : 3]];
        repeats = 2.0 + (iterations.kind == graph::NodeKind::constant &&
            std::isfinite(iterations.constant) && iterations.constant >= 1 && iterations.constant <= 10000
            ? iterations.constant : 10000.0);
      }
      // A full-group/full-block body bounds sums of linear, sorting and dense
      // matrix work over disjoint subsets. Per-group scalar overhead is bounded
      // separately by N * body_nodes. This deliberately remains conservative.
      extra += repeats * (typed_array_extra_work(body, captured, length, true) +
          static_cast<double>(std::max<std::size_t>(length, 1)) * body.nodes.size());
      continue;
    }
    if (node.kind != graph::NodeKind::operation || !node.parent_count) continue;
    const auto &spec = ops::lookup(node.opcode);
    const auto op = spec.op;
    const auto &lhs = shapes[node.parents[0]];
    const auto rhs = node.parent_count > 1 ? shapes[node.parents[1]] : ops::Shape{};
    double extent = static_cast<double>(shapes[index].size());
    for (std::size_t i = 0; i < node.parent_count; ++i)
      extent = std::max(extent, static_cast<double>(shapes[node.parents[i]].size()));
    if (spec.family == ops::Family::state && op != ops::Op::last_drawdown_interval &&
        static_cast<std::uint16_t>(op) < 132)
      continue;
    if (op == ops::Op::state_estimate || op == ops::Op::state_variance ||
        op == ops::Op::continuous_state_values || op == ops::Op::continuous_state_evidence ||
        op == ops::Op::continuous_state_pending)
      continue; // Field projections borrow storage without scanning it.
    // Segment projections borrow too, but first scan both boundary columns.
    if (op == ops::Op::transpose || (op == ops::Op::diag && lhs.rank == 2))
      continue; // Borrowed views have no materialized array traversal.
    const bool sort = op == ops::Op::median || op == ops::Op::quantile ||
        op == ops::Op::median_where || op == ops::Op::quantile_where ||
        op == ops::Op::argsort || op == ops::Op::distinct_count;
    if (sort) {
      if (lhs.rank == 1 && graph::order_fusion_eligible(op) &&
          !orders.insert(node.parents[0]).second) continue;
      extra += 2.0 * std::max(0.0, sorting(extent) - sorting(baseline));
      continue;
    }
    if (lhs.rank == 1 && graph::summary_fusion_eligible(op) &&
        program.execution_metadata &&
        program.execution_metadata->summary_consumers[node.parents[0]] > 1 &&
        !summaries.insert(node.parents[0]).second) continue;
    double factor = 1.0;
    if (spec.family == ops::Family::elementwise)
      factor = ops::simd_eligible(op) ? 0.65 : 1.0;
    else if (spec.family == ops::Family::sequence) factor = 1.2;
    else if (spec.family == ops::Family::state && static_cast<std::uint16_t>(op) >= 132)
      factor = 1.2;
    else if (spec.family == ops::Family::reduction || spec.family == ops::Family::composite)
      factor = 1.5;
    else if (spec.family == ops::Family::matrix || spec.family == ops::Family::rolling)
      factor = 2.0;
    else if (spec.family == ops::Family::regression) factor = 3.5;
    if (op == ops::Op::matmul)
      extent = static_cast<double>(lhs.dim[0]) * lhs.dim[1] * rhs.dim[1];
    else if ((op == ops::Op::covariance || op == ops::Op::correlation) && lhs.rank == 2)
      extent = static_cast<double>(lhs.dim[0]) * lhs.dim[1] * lhs.dim[1];
    else if (op == ops::Op::solve)
      extent = static_cast<double>(lhs.dim[0]) * lhs.dim[0] * lhs.dim[0];
    else if (op == ops::Op::trace)
      extent = static_cast<double>(std::min(lhs.dim[0], lhs.dim[1]));
    else if (op == ops::Op::ps_filter)
      extent = static_cast<double>(lhs.dim[0]) * lhs.dim[0];
    else if (op == ops::Op::local_extrema) {
      const auto window = [&](std::size_t parameter) {
        const auto &value = program.nodes[node.parents[parameter]];
        const auto width = value.kind == graph::NodeKind::constant &&
            std::isfinite(value.constant) && value.constant >= 1 && value.constant <= 5000
            ? value.constant : 5000.0;
        return std::min(static_cast<double>(lhs.dim[0]), width);
      };
      extent = static_cast<double>(lhs.dim[0]) * (window(1) + window(2));
    }
    extra += factor * std::max(0.0, extent - baseline);
  }
  if (!std::isfinite(extra)) throw std::overflow_error("typed array work estimate overflow");
  return extra;
}
bool requires_typed_work(const graph::Program &program) {
  if (program.execution_metadata && program.execution_metadata->requires_shape_planning) return true;
  for (const auto &scope : program.rolling_scopes)
    if (scope.body && requires_typed_work(*scope.body)) return true;
  for (const auto &scope : program.apply_scopes)
    if (scope.body && requires_typed_work(*scope.body)) return true;
  return false;
}
std::size_t scratch_estimate(const graph::Program &program,
                             std::size_t window,
                             const std::vector<ops::Value> *inputs = nullptr,
                             std::size_t logical_window = 0,
                             bool inherited_isolation = false) {
  const bool isolate_errors = program.isolate_errors || inherited_isolation;
  if (!inherited_isolation && inputs && program.execution_metadata && program.execution_metadata->pointwise) {
    bool compatible = true, found = false;
    ops::Shape common;
    for (std::size_t i = 0; i < inputs->size(); ++i) {
      const auto &value = inputs->at(i);
      compatible = compatible && value.kind == ops::Kind::number && value.shape.rank >= 0;
      if (value.shape.rank <= 0) continue;
      auto shape = value.shape;
      if (program.input_axes.empty() || program.input_axes[i] != 1) shape.dim[0] = logical_window;
      if (!found) { common = shape; found = true; }
      else compatible = compatible && shape == common;
    }
    if (compatible && found) return graph::pointwise_workspace_bytes(program, common.size());
  }
  const auto numeric_slots = program.execution_metadata && program.execution_metadata->numeric_roots_only
      ? 0 : program.numeric_slots;
  std::size_t estimate =
      checked_mul(window, checked_add(checked_mul(checked_add(numeric_slots, program.integer_slots), 8),
                                      program.mask_slots));
  estimate = checked_add(
      estimate, checked_mul(program.nodes.size(), sizeof(ops::Value) + 160));
  if (program.execution_metadata && program.execution_metadata->pointwise)
    estimate = checked_add(estimate, graph::pointwise_workspace_bytes(program, window));
  if (isolate_errors) {
    estimate = checked_add(estimate, checked_mul(program.nodes.size(), sizeof(std::int16_t)));
    const auto positional = program.execution_metadata
        ? program.execution_metadata->position_status_nodes : 0;
    if (positional) {
      // Failure provenance is lazy at execution time, but admission must cover
      // its worst case before any fault occurs. `window` is the native element
      // capacity, including multicolumn records, not just the logical time axis.
      estimate = checked_add(estimate, checked_mul(
          checked_mul(positional, window), sizeof(std::int16_t)));
      estimate = checked_add(estimate, checked_mul(
          program.nodes.size(), sizeof(std::vector<std::int16_t>)));
    }
  }
  std::unordered_set<std::uint32_t> order_sources;
  bool operator_scratch = false;
  for (const auto &node : program.nodes) {
    if (node.kind != graph::NodeKind::operation)
      continue;
    const auto op = ops::lookup(node.opcode).op;
    if (op == ops::Op::median || op == ops::Op::quantile)
      order_sources.insert(node.parents[0]);
    if (op == ops::Op::median || op == ops::Op::quantile ||
        op == ops::Op::median_where || op == ops::Op::quantile_where ||
        op == ops::Op::rolling_min || op == ops::Op::rolling_max ||
        op == ops::Op::argsort || op == ops::Op::distinct_count || op == ops::Op::ps_filter ||
        op == ops::Op::solve || op == ops::Op::covariance ||
        op == ops::Op::correlation || op == ops::Op::quadratic_form)
      operator_scratch = true;
  }
  estimate = checked_add(
      estimate, checked_mul(checked_mul(order_sources.size(), window), 8));
  if (operator_scratch)
    estimate = checked_add(estimate, checked_mul(window, 24));
  if (!program.rolling_scopes.empty())
    estimate = checked_add(estimate, checked_mul(window + 1, sizeof(std::size_t)));
  std::vector<ops::Shape> shapes;
  if (inputs && (!program.rolling_scopes.empty() || !program.apply_scopes.empty()))
    graph::required_array_capacity(program, *inputs, logical_window, &shapes);
  const auto scope_scratch = [&](const graph::Program &body,
                                 const std::vector<std::uint32_t> &captures,
                                 const std::vector<std::uint8_t> *preceding,
                                 bool selected, std::size_t uncaptured_extent,
                                 bool validates_numerical_stop = false) {
    const bool child_isolation = isolate_errors || validates_numerical_stop;
    std::vector<ops::Value> captured;
    captured.reserve(captures.size());
    std::size_t length = captures.empty() ? uncaptured_extent : 0;
    for (std::size_t i = 0; i < captures.size(); ++i) {
      ops::Value value;
      value.shape = shapes.empty() ? ops::vector_shape(window) : shapes.at(captures[i]);
      if (preceding && i < preceding->size() && preceding->at(i))
        value.shape.dim[0] = checked_add(value.shape.dim[0], 1);
      length = std::max(length, value.shape.size());
      captured.push_back(value);
    }
    // An unresolved selection validates its child without inventing a slice.
    // Its logical interval is only a capacity bound, up to the enclosing frame.
    if (child_isolation) length = std::max(length, logical_window ? logical_window : window);
    const auto capacity = graph::required_array_capacity(body, captured, length);
    // A failed capture enables child isolation even when its compiled body is
    // strict. Admission must include the resulting node/position statuses at
    // every nesting depth before a numerical failure occurs.
    auto bytes = scratch_estimate(body, capacity, &captured, length, child_isolation);
    if (child_isolation) {
      bytes = checked_add(bytes, checked_mul(captures.size(), sizeof(std::int16_t)));
      bytes = checked_add(bytes, checked_mul(captures.size(), sizeof(ops::Value)));
      bytes = checked_add(bytes, checked_mul(body.parameter_count, sizeof(double)));
    }
    if (child_isolation || body.isolate_errors)
      bytes = checked_add(bytes, checked_mul(body.output_kind == graph::OutputKind::typed ?
          std::max<std::size_t>(capacity, 1) : 1, sizeof(std::int16_t)));
    if (selected)
      bytes = checked_add(bytes, checked_mul(length,
          checked_add(checked_mul(captures.size(), 8), 24)));
    return bytes;
  };
  for (const auto &scope : program.rolling_scopes)
    if (scope.body)
      estimate = checked_add(estimate,
          scope_scratch(*scope.body, scope.input_nodes, &scope.input_preceding, false,
                        logical_window ? logical_window : window));
  for (const auto &scope : program.apply_scopes)
    if (scope.body)
      estimate = checked_add(estimate, scope_scratch(*scope.body, scope.input_nodes,
          nullptr, scope.kind != graph::ApplyKind::bisect && scope.kind != graph::ApplyKind::iterate,
          apply_observation_bound(scope, shapes, logical_window ? logical_window : window),
          scope.kind == graph::ApplyKind::iterate));
  for (const auto &scope : program.apply_scopes)
    if (scope.kind == graph::ApplyKind::iterate) {
      const auto count = shapes.empty() ? window : shapes.at(scope.argument_nodes[0]).size();
      estimate = checked_add(estimate, checked_mul(std::max<std::size_t>(count, 1), 16));
      estimate = checked_add(estimate, checked_mul(scope.input_nodes.size(), sizeof(ops::Value)));
      estimate = checked_add(estimate, checked_mul(scope.parameter_nodes.size(), sizeof(double)));
      estimate = checked_add(estimate, sizeof(graph::ResultLayout) + sizeof(graph::ResultSlot) + 128);
    }
  estimate = checked_add(estimate, checked_mul(program.apply_scopes.size(), sizeof(std::array<double, 3>)));
  return estimate;
}
template <bool CollectMetrics, class SizeAt>
Geometry inspect_geometry(std::size_t size_count, SizeAt size_at,
                 const std::int64_t *starts, const std::int64_t *ends,
                 std::size_t rows, const std::int64_t *product_ids,
                 bool build_groups) {
  require(rows == 0 || (starts && ends), "missing interval arrays");
  for (std::size_t i = 0; i < size_count; ++i)
    require(size_at(i) == size_at(0),
            "graph inputs must share an aligned observation axis");
  Geometry g;
  g.rows = rows;
  g.signature = 0x434d454e41544956ull;
  mix(g.signature, rows);
  mix(g.signature, size_count);
  mix(g.signature, product_ids != nullptr);
  for (std::size_t i = 0; i < size_count; ++i)
    mix(g.signature, size_at(i));
  const auto available = size_count ? size_at(0) : 0;
  std::size_t group_start = 0, group_weight = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    require(starts[row] >= 0 && ends[row] >= starts[row] &&
                static_cast<std::size_t>(ends[row]) <= available,
            "intervals must satisfy 0 <= start <= end <= input length");
    if (product_ids)
      require(product_ids[row] >= 0 &&
                  (row == 0 || product_ids[row] >= product_ids[row - 1]),
              "product_ids must be nonnegative and nondecreasing");
    const auto n = static_cast<std::size_t>(ends[row] - starts[row]);
    // Preserve overflow validation even when only the signature is needed.
    g.observations = checked_add(g.observations, n);
    if constexpr (CollectMetrics) {
      g.row_lengths.push_back(n);
      g.max_window = std::max(g.max_window, n);
      g.sort_work += static_cast<double>(n) *
                     std::log2(static_cast<double>(std::max<std::size_t>(n, 2)));
    }
    mix(g.signature, static_cast<std::uint64_t>(starts[row]));
    mix(g.signature, static_cast<std::uint64_t>(ends[row]));
    if (product_ids)
      mix(g.signature, static_cast<std::uint64_t>(product_ids[row]));
    if constexpr (CollectMetrics) {
      const bool new_group =
          row > 0 && (!product_ids || product_ids[row] != product_ids[row - 1]);
      if (new_group) {
        ++g.products;
        g.max_intervals = std::max(g.max_intervals, row - group_start);
        if (build_groups) {
          g.groups.push_back({group_start, row});
          g.weights.push_back(group_weight);
        }
        group_start = row;
        group_weight = 0;
      }
      group_weight = checked_add(group_weight, n);
    }
  }
  if (CollectMetrics && rows) {
    ++g.products;
    g.max_intervals = std::max(g.max_intervals, rows - group_start);
    if (build_groups) {
      g.groups.push_back({group_start, rows});
      g.weights.push_back(group_weight);
    }
  }
  return g;
}
template <bool CollectMetrics = true>
Geometry typed_geometry(const graph::Program &program,
                        const std::vector<std::size_t> &sizes,
                        const std::vector<ops::Value> *inputs,
                        const std::int64_t *starts, const std::int64_t *ends,
                        std::size_t rows, const std::int64_t *products,
                        bool build_groups = true) {
  if (!inputs) return inspect_geometry<CollectMetrics>(sizes.size(),
      [&](std::size_t i) { return sizes[i]; }, starts, ends, rows, products, build_groups);
  require(inputs->size() == program.input_count, "graph input count mismatch");
  std::size_t temporal_count = 0, available = 0;
  for (std::size_t i = 0; i < inputs->size(); ++i) {
    const auto &value = inputs->at(i);
    const auto axis = program.input_axes.empty() ? 0 : program.input_axes.at(i);
    require((axis != 0 || value.shape.rank == 1) &&
            (axis != 2 || value.shape.rank >= 2) &&
            value.shape.rank >= 0 && value.shape.rank <= 3,
            "graph input rank mismatch");
    if (axis != 1) {
      if (temporal_count) require(available == value.shape.dim[0],
          "graph inputs must share an aligned observation axis");
      available = value.shape.dim[0];
      ++temporal_count;
    }
  }
  if (!temporal_count) {
    for (std::size_t i = 0; i < rows; ++i) {
      require(starts && ends && starts[i] >= 0 && ends[i] >= starts[i], "invalid interval bounds");
      available = std::max(available, static_cast<std::size_t>(ends[i]));
    }
    temporal_count = 1;
  }
  auto geometry = inspect_geometry<CollectMetrics>(temporal_count,
      [&](std::size_t) { return available; }, starts, ends, rows, products, build_groups);
  for (const auto &value : *inputs) {
    mix(geometry.signature, static_cast<std::uint64_t>(value.kind));
    mix(geometry.signature, static_cast<std::uint64_t>(value.shape.rank));
    for (int axis = 0; axis < value.shape.rank; ++axis) {
      mix(geometry.signature, value.shape.dim[axis]);
      mix(geometry.signature, static_cast<std::uint64_t>(value.stride[axis]));
    }
  }
  return geometry;
}
std::size_t total_memory(const Plan &p, std::size_t workers) {
  auto bytes = checked_add(p.estimated_input_bytes, p.estimated_output_bytes);
  // Native status chunks + result array + worst-case IPC serialization buffers.
  bytes = checked_add(bytes, checked_mul(p.estimated_status_bytes, p.lane == "process" ? 6 : 2));
  bytes = checked_add(bytes, checked_mul(p.estimated_result_metadata_bytes, p.lane == "process" ? checked_add(workers, 6) : 2));
  bytes = checked_add(bytes, checked_mul(p.row_count, 16));
  bytes = checked_add(bytes,
                      checked_mul(workers, p.estimated_worker_scratch_bytes));
  if (p.lane == "process") {
    bytes = checked_add(bytes, p.estimated_output_bytes);
    if (p.use_shared_memory) {
      if (!p.inputs_already_shared)
        bytes = checked_add(bytes, p.estimated_input_bytes);
      bytes = checked_add(bytes, checked_mul(p.row_count, 16));
    } else {
      bytes = checked_add(
          bytes, checked_mul(p.estimated_input_bytes, checked_mul(workers, 2)));
      // Typed matrices can dwarf inputs: parent output, worker output, worker
      // frame, received frame and decoded payload may coexist during inline IPC.
      if (p.result_layout)
        bytes = checked_add(bytes, checked_mul(p.estimated_output_bytes, 3));
    }
  }
  return bytes;
}
} // namespace

std::size_t checked_add(std::size_t a, std::size_t b) {
  if (a > static_cast<std::size_t>(PTRDIFF_MAX) - b ||
      b > static_cast<std::size_t>(PTRDIFF_MAX))
    throw std::overflow_error("execution size overflow");
  return a + b;
}
std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (b && a > static_cast<std::size_t>(PTRDIFF_MAX) / b)
    throw std::overflow_error("execution size overflow");
  return a * b;
}
void Config::validate() const {
  require(std::isfinite(thread_work_units) && thread_work_units > 0 &&
              std::isfinite(dag_branch_work_units) &&
              dag_branch_work_units > 0 && std::isfinite(process_work_units) &&
              process_work_units > 0,
          "work thresholds must be finite and positive");
  require(min_rows_per_worker > 0 && max_processes > 0 &&
              max_processes <= 1024 && max_async_jobs > 0 &&
              max_async_jobs <= 1024,
          "invalid planner worker limits");
}

Geometry inspect(const std::vector<std::size_t> &sizes,
                 const std::int64_t *starts, const std::int64_t *ends,
                 std::size_t rows, const std::int64_t *product_ids,
                 bool build_groups) {
  return inspect_geometry<true>(sizes.size(), [&](std::size_t i) { return sizes[i]; },
      starts, ends, rows, product_ids, build_groups);
}
std::vector<Chunk> partition(const Geometry &g, std::size_t workers,
                             bool by_product,
                             const compiler::PhysicalCost *physical_cost) {
  if (!g.rows)
    return {};
  const auto units = by_product ? g.groups.size() : g.rows;
  workers = std::max<std::size_t>(1, std::min(workers, units));

  std::vector<long double> weights(units, 0.0L);
  if (by_product) {
    for (std::size_t group = 0; group < g.groups.size(); ++group) {
      const auto chunk = g.groups[group];
      for (std::size_t row = chunk.begin; row < chunk.end; ++row)
        weights[group] +=
            !g.row_work_units.empty() ? static_cast<long double>(g.row_work_units[row])
            : physical_cost ? static_cast<long double>(
                                row_work(*physical_cost, g.row_lengths[row]))
                          : static_cast<long double>(
                                g.row_lengths[row] ? g.row_lengths[row] : 1);
    }
  } else {
    for (std::size_t row = 0; row < g.rows; ++row)
      weights[row] = !g.row_work_units.empty() ? static_cast<long double>(g.row_work_units[row])
                     : physical_cost
                         ? static_cast<long double>(
                               row_work(*physical_cost, g.row_lengths[row]))
                         : static_cast<long double>(
                               g.row_lengths[row] ? g.row_lengths[row] : 1);
  }

  std::vector<long double> cumulative;
  cumulative.reserve(units);
  long double total = 0.0L;
  for (auto weight : weights) {
    total += weight > 0 ? weight : 1.0L;
    cumulative.push_back(total);
  }

  std::vector<Chunk> result;
  std::size_t previous = 0;
  for (std::size_t worker = 1; worker < workers; ++worker) {
    const auto target = total * static_cast<long double>(worker) /
                        static_cast<long double>(workers);
    auto cut =
        static_cast<std::size_t>(
            std::lower_bound(cumulative.begin(), cumulative.end(), target) -
            cumulative.begin()) +
        1;
    cut = std::max(cut, previous + 1);
    cut = std::min(cut, units - (workers - worker));
    if (by_product)
      result.push_back({g.groups[previous].begin, g.groups[cut - 1].end});
    else
      result.push_back({previous, cut});
    previous = cut;
  }
  if (by_product)
    result.push_back({g.groups[previous].begin, g.groups.back().end});
  else
    result.push_back({previous, g.rows});
  return result;
}

std::shared_ptr<Plan>
make_plan(std::shared_ptr<compiler::CompiledGraph> graph, const Config &config,
          const std::vector<std::size_t> &sizes, const std::int64_t *starts,
          const std::int64_t *ends, std::size_t rows,
          const std::int64_t *product_ids, std::size_t cpu,
          std::optional<std::size_t> memory_budget, bool hard_stop,
          bool async_io, bool already_shared,
          const std::vector<ops::Value> *inputs, std::size_t owned_input_bytes) {
  config.validate();
  require(cpu > 0 && cpu <= 1024, "cpu_budget must be in 1..1024");
  require(!memory_budget || *memory_budget > 0,
          "memory budget must be positive");
  require(graph && sizes.size() == graph->program.input_count,
          "graph input count mismatch");
  auto geometry = typed_geometry(graph->program, sizes, inputs, starts, ends, rows, product_ids);
  const auto array_capacity = inputs ? graph::required_array_capacity(graph->program, *inputs, geometry.max_window) : geometry.max_window;
  auto p = std::make_shared<Plan>();
  p->graph = std::move(graph);
  p->cpu_budget = cpu;
  p->memory_budget_bytes = memory_budget;
  p->row_count = rows;
  p->max_window = geometry.max_window;
  p->interval_observations = geometry.observations;
  p->product_count = geometry.products;
  p->max_intervals_per_product = geometry.max_intervals;
  p->geometry_signature = geometry.signature;
  p->input_sizes = sizes;
  p->inputs_already_shared = already_shared;
  p->async_orchestration = async_io;
  p->hard_stop = hard_stop;
  p->parallel_dimension =
      product_ids &&
              geometry.products >= std::min(cpu, std::max<std::size_t>(1, rows))
          ? "product"
          : "interval";
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    const auto item = inputs && inputs->at(i).kind == ops::Kind::mask ? 1u : 8u;
    p->estimated_input_bytes = checked_add(p->estimated_input_bytes, checked_mul(sizes[i], item));
  }
  p->estimated_input_bytes = std::max(p->estimated_input_bytes, owned_input_bytes);
  const auto output_rows =
      p->graph->program.output_kind == graph::OutputKind::series
          ? geometry.observations
          : rows;
  p->estimated_output_bytes = checked_mul(
      checked_mul(output_rows, p->graph->program.roots.size()),
      graph::output_itemsize(p->graph->program.output_dtype));
  p->estimated_status_bytes = p->graph->program.isolate_errors
      ? checked_mul(checked_mul(output_rows, p->graph->program.roots.size()), 2) : 0;
  if (p->graph->program.output_kind == graph::OutputKind::typed) {
    require(inputs != nullptr, "typed results require bound input geometry");
    p->estimated_result_metadata_bytes = checked_add(
        checked_mul(checked_mul(rows, p->graph->program.roots.size()),
                    sizeof(graph::ResultSlot) + sizeof(ops::Shape) + 2),
        checked_mul(checked_add(rows, 1), 2 * sizeof(std::size_t)));
    // Reject an impossible metadata budget before reserving rows * roots slots.
    if (memory_budget && p->estimated_result_metadata_bytes > *memory_budget)
      throw std::bad_alloc();
    p->result_layout = std::make_shared<graph::ResultLayout>(
        graph::result_layout(p->graph->program, *inputs, starts, ends, rows));
    p->estimated_output_bytes = p->result_layout->bytes();
    p->estimated_status_bytes = p->graph->program.isolate_errors
        ? checked_mul(p->result_layout->status_count(), 2) : 0;
  }
  p->estimated_worker_scratch_bytes =
      scratch_estimate(p->graph->program, array_capacity, inputs, geometry.max_window);
  for (const auto &node : p->graph->nodes) {
    if (node.node.kind != graph::NodeKind::operation)
      continue;
    double work = 0;
    if (node.cost_model == "constant")
      work = static_cast<double>(rows);
    else if (node.cost_model == "sort")
      work = 2 * geometry.sort_work;
    else {
      double factor = 1;
      if (node.cost_model == "elementwise")
        factor = node.simd_eligible ? .65 : 1;
      else if (node.cost_model == "sequence")
        factor = 1.2;
      else if (node.cost_model == "reduction" || node.cost_model == "composite")
        factor = 1.5;
      else if (node.cost_model == "rolling" || node.cost_model == "matrix")
        factor = 2;
      else if (node.cost_model == "regression")
        factor = 3.5;
      work = factor * static_cast<double>(geometry.observations);
    }
    p->estimated_logical_work_units += work;
    if (node.simd_eligible && array_capacity >= config.simd_min_elements)
      p->simd_nodes.push_back(node.node_id);
  }
  p->estimated_work_units = geometry_work(p->graph->physical_cost, geometry);
  const bool typed_arrays = inputs && requires_typed_work(p->graph->program);
  std::map<std::size_t, double> extra_by_length;
  if (typed_arrays) {
    geometry.row_work_units.reserve(rows);
    for (auto length : geometry.row_lengths) {
      auto found = extra_by_length.find(length);
      if (found == extra_by_length.end())
        found = extra_by_length.emplace(length,
            typed_array_extra_work(p->graph->program, *inputs, length)).first;
      p->estimated_typed_array_work_units += found->second;
      geometry.row_work_units.push_back(row_work(p->graph->physical_cost, length) + found->second);
    }
    p->estimated_work_units += p->estimated_typed_array_work_units;
    p->estimated_logical_work_units += p->estimated_typed_array_work_units;
    if (p->estimated_typed_array_work_units > 0)
      p->reason_codes.push_back("typed_array_shape_work_accounted");
  }
  if (p->result_layout) {
    if (geometry.row_work_units.empty())
      for (auto length : geometry.row_lengths)
        geometry.row_work_units.push_back(row_work(p->graph->physical_cost, length));
    for (std::size_t row = 0; row < rows; ++row) {
      const auto work = static_cast<double>(p->result_layout->row_statuses[row + 1] - p->result_layout->row_statuses[row]);
      p->estimated_work_units += work;
      geometry.row_work_units[row] += work;
    }
    p->reason_codes.push_back("typed_result_capacity_write_work_accounted");
  } else if (p->graph->program.output_dtype != graph::OutputDType::float64) {
    // New typed series must be written even for an identity graph. Charge one
    // structural work unit per output element; this is not benchmark tuning
    // and leaves the established float64 cost model unchanged.
    const auto roots = static_cast<double>(p->graph->program.roots.size());
    p->estimated_work_units += roots * static_cast<double>(geometry.observations);
    if (geometry.row_work_units.empty()) {
      geometry.row_work_units.reserve(rows);
      for (auto length : geometry.row_lengths)
        geometry.row_work_units.push_back(row_work(p->graph->physical_cost, length));
    }
    for (std::size_t row = 0; row < rows; ++row)
      geometry.row_work_units[row] += roots * static_cast<double>(geometry.row_lengths[row]);
    p->reason_codes.push_back("typed_output_write_work_accounted");
  }
  std::vector<std::map<std::size_t, double>> branch_extra(p->graph->branches.size());
  const auto branch_work = [&](std::size_t branch, std::size_t length) {
    const auto &info = p->graph->branches[branch];
    const double base = row_work(info.cost, length);
    if (!typed_arrays) return base;
    auto &cache = branch_extra[branch];
    auto found = cache.find(length);
    if (found == cache.end()) found = cache.emplace(length,
        typed_array_extra_work(info.program, *inputs, length)).first;
    return base + found->second;
  };
  const bool process =
      hard_stop ||
      (rows >= std::max<std::size_t>(2, config.min_rows_per_worker) &&
       (p->estimated_work_units >= config.process_work_units ||
        p->estimated_input_bytes >= config.process_input_threshold_bytes));

  std::size_t heavy_branches = 0;
  for (std::size_t branch = 0; branch < p->graph->branches.size(); ++branch) {
    double work = geometry_work(p->graph->branches[branch].cost, geometry);
    if (typed_arrays) {
      work = 0;
      for (auto length : geometry.row_lengths) work += branch_work(branch, length);
    }
    heavy_branches += work >= config.thread_work_units;
  }
  const auto branch_task_count = checked_mul(rows, p->graph->branches.size());
  const bool dag_branch =
      p->graph->program.output_kind == graph::OutputKind::scalar && !process &&
      cpu > 1 && rows > 0 && rows < cpu &&
      p->graph->branches.size() > 1 && heavy_branches >= 2 &&
      p->estimated_work_units >= config.dag_branch_work_units &&
      branch_task_count > rows;

  // Only a complete pointwise region may partition one tensor. Cross-element
  // kernels and recurrence states never inherit this capability by shape alone.
  std::size_t tensor_elements = 0;
  if (!process && rows == 1 && cpu > 1 && inputs && p->result_layout &&
      p->graph->program.execution_metadata->pointwise) {
    bool compatible = true, found = false;
    ops::Shape common;
    for (std::size_t i = 0; i < inputs->size(); ++i) {
      const auto &value = inputs->at(i);
      compatible = compatible && value.kind == ops::Kind::number && value.contiguous();
      if (!value.shape.rank) continue;
      auto shape = value.shape;
      const auto &axes = p->graph->program.input_axes;
      if (axes.empty() || axes[i] != 1) shape.dim[0] = geometry.max_window;
      if (!found) { common = shape; found = true; }
      else compatible = compatible && shape == common;
    }
    if (compatible && found) {
      tensor_elements = common.size();
      for (const auto &slot : p->result_layout->slots)
        if (slot.capacity != tensor_elements) tensor_elements = 0;
    }
  }
  const auto tensor_workers = std::min({cpu, tensor_elements / graph::pointwise_tile_elements,
      static_cast<std::size_t>(std::min(static_cast<double>(cpu),
          p->estimated_work_units / config.thread_work_units))});
  if (process) {
    p->lane = "process";
    p->reason_codes.push_back(hard_stop ? "hard_stop_requires_process_isolation"
                              : p->estimated_input_bytes >=
                                      config.process_input_threshold_bytes
                                  ? "input_exceeds_process_isolation_threshold"
                                  : "work_exceeds_process_threshold");
    p->process_count =
        std::min({cpu, config.max_processes,
                  std::max<std::size_t>(1, rows / config.min_rows_per_worker)});
  } else if (dag_branch) {
    p->lane = "thread";
    p->parallel_dimension = "dag_branch";
    p->thread_count = std::min(cpu, branch_task_count);
    p->estimated_worker_scratch_bytes = 0;
    for (const auto &branch : p->graph->branches) {
      const auto branch_output =
          checked_mul(branch.root_indices.size(), sizeof(double));
      p->estimated_worker_scratch_bytes = std::max(
          p->estimated_worker_scratch_bytes,
          checked_add(scratch_estimate(branch.program, array_capacity, inputs, geometry.max_window),
                      branch_output));
    }
    p->reason_codes.push_back("heavy_independent_dag_branches");
  } else if (tensor_workers >= 2) {
    p->lane = "thread";
    p->parallel_dimension = "tensor";
    p->tensor_elements = tensor_elements;
    p->thread_count = tensor_workers;
    p->reason_codes.push_back("independent_pointwise_tensor_tiles");
  } else if (p->estimated_work_units >= config.thread_work_units && rows >= 2 &&
             cpu > 1) {
    p->lane = "thread";
    p->thread_count = std::min(cpu, rows);
    p->reason_codes.push_back("work_exceeds_thread_threshold");
  } else
    p->reason_codes.push_back("parallel_overhead_expected_to_dominate");

  if (p->lane != "single")
    p->reason_codes.push_back("parallelize_by_" + p->parallel_dimension);
  p->use_shared_memory = process && (hard_stop || already_shared ||
                                     p->estimated_input_bytes >=
                                         config.shared_memory_threshold_bytes);
  if (p->use_shared_memory)
    p->reason_codes.push_back(
        hard_stop        ? "hard_stop_uses_shared_process_transport"
        : already_shared ? "inputs_already_shared"
                         : "process_input_exceeds_shared_memory_threshold");

  auto workers = process ? p->process_count : p->thread_count;
  const auto check_transport = [&] {
    if (!process) return;
    p->chunks = partition(geometry, workers, p->parallel_dimension == "product",
                          &p->graph->physical_cost);
    for (const auto chunk : p->chunks) {
      std::size_t output_bytes = 0, status_bytes = 0, slots = 0;
      if (p->result_layout) {
        const auto &layout = *p->result_layout;
        output_bytes = layout.row_bytes[chunk.end] - layout.row_bytes[chunk.begin];
        if (p->graph->program.isolate_errors)
          status_bytes = checked_mul(layout.row_statuses[chunk.end] - layout.row_statuses[chunk.begin], 2);
        slots = checked_mul(chunk.end - chunk.begin, p->graph->program.roots.size());
      } else {
        auto output_rows = chunk.end - chunk.begin;
        if (p->graph->program.output_kind == graph::OutputKind::series) {
          output_rows = 0;
          for (auto row = chunk.begin; row < chunk.end; ++row)
            output_rows = checked_add(output_rows, geometry.row_lengths[row]);
        }
        const auto count = checked_mul(output_rows, p->graph->program.roots.size());
        output_bytes = checked_mul(count, graph::output_itemsize(p->graph->program.output_dtype));
        if (p->graph->program.isolate_errors) status_bytes = checked_mul(count, 2);
      }
      // Statuses and actual shapes still travel in IPC on the shared path.
      require(native::process_response_bytes(0, status_bytes, slots) <= native::max_process_frame_bytes,
              "process result status/shape metadata exceeds IPC frame limit; reduce the interval batch");
      if (!p->use_shared_memory &&
          native::process_response_bytes(output_bytes, status_bytes, slots) > native::max_process_frame_bytes) {
        p->use_shared_memory = true;
        p->reason_codes.push_back("process_result_exceeds_ipc_frame_limit");
      }
    }
  };
  check_transport();
  p->estimated_total_memory_bytes = total_memory(*p, workers);
  bool reduced = false;
  while (memory_budget && p->estimated_total_memory_bytes > *memory_budget &&
         workers > 1) {
    --workers;
    reduced = true;
    // Fewer workers enlarge each response; recheck before memory admission.
    check_transport();
    p->estimated_total_memory_bytes = total_memory(*p, workers);
  }
  if (memory_budget && p->estimated_total_memory_bytes > *memory_budget)
    throw std::bad_alloc();
  if (process)
    p->process_count = workers;
  else
    p->thread_count = workers;
  if (reduced)
    p->reason_codes.push_back("memory_budget_reduced_parallelism");
  if (!p->simd_nodes.empty())
    p->reason_codes.push_back(
        "eligible_nodes_use_native_runtime_simd_dispatch");
  if (async_io)
    p->reason_codes.push_back("coroutine_orchestration_for_async_boundary");

  if (p->parallel_dimension == "dag_branch") {
    struct WeightedTask {
      BranchTask task;
      double work = 0;
    };
    std::vector<WeightedTask> tasks;
    tasks.reserve(branch_task_count);
    for (std::size_t row = 0; row < rows; ++row)
      for (std::uint32_t branch = 0; branch < p->graph->branches.size();
           ++branch)
        tasks.push_back({{row, branch},
                         branch_work(branch, geometry.row_lengths[row])});
    std::stable_sort(tasks.begin(), tasks.end(),
                     [](const WeightedTask &a, const WeightedTask &b) {
                       return a.work > b.work;
                     });
    for (const auto &task : tasks)
      p->branch_tasks.push_back(task.task);
  } else if (p->parallel_dimension == "tensor") {
    const auto tiles = (p->tensor_elements + graph::pointwise_tile_elements - 1) / graph::pointwise_tile_elements;
    for (std::size_t i = 0; i < workers; ++i)
      p->chunks.push_back({(tiles * i / workers) * graph::pointwise_tile_elements,
          std::min(p->tensor_elements, (tiles * (i + 1) / workers) * graph::pointwise_tile_elements)});
  } else if (!process) {
    p->chunks = partition(geometry, workers, p->parallel_dimension == "product",
                          &p->graph->physical_cost);
  }
  return p;
}
void validate_plan(const Plan &p, const std::vector<std::size_t> &sizes,
                   const std::int64_t *starts, const std::int64_t *ends,
                   std::size_t rows, const std::int64_t *product_ids,
                   std::size_t engine_cpu, const std::vector<ops::Value> *inputs) {
  require(p.cpu_budget <= engine_cpu,
          "execution plan exceeds this engine CPU budget");
  require(p.input_sizes == sizes && p.row_count == rows,
          "stale execution plan input shape");
  const auto geometry = typed_geometry<false>(p.graph->program, sizes, inputs, starts, ends, rows, product_ids, false);
  require(geometry.signature == p.geometry_signature,
          "stale execution plan interval/product geometry");
}
} // namespace calmetrics_engine::planner
