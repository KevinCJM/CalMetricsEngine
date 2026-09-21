#include "calmetrics_engine/graph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <cstring>
#include <stdexcept>

namespace calmetrics_engine::graph {
namespace {

struct SummaryCache {
  std::size_t generation = 0;
  std::size_t count = 0;
  double total = 0.0;
  double mean = 0.0;
  double m2 = 0.0;
  double square_total = 0.0;
  double minimum = 0.0;
  double maximum = 0.0;
  double product = 1.0;
  double growth_product = 1.0;
  double mean_absolute_deviation = 0.0;
  bool mad_ready = false;
};

struct OrderedCache {
  std::size_t generation = 0;
  std::vector<double> values;
};

struct Scratch {
  std::vector<double> numeric;
  std::vector<std::uint8_t> masks;
  std::vector<std::int64_t> integers;
  std::vector<std::vector<double>> selected_numeric;
  std::vector<std::vector<std::int64_t>> selected_integer;
  std::vector<std::vector<std::uint8_t>> selected_masks;
  std::vector<std::size_t> selection;
  std::size_t work_budget = 100000000;
  std::size_t *remaining_work = nullptr;
  std::vector<ops::Value> values;
  bool isolate_errors = false;
  bool interval_known = true;
  std::vector<std::int16_t> statuses;
  std::vector<std::vector<std::int16_t>> position_statuses;
  std::shared_ptr<const ExecutionMetadata> position_owner;
  std::size_t position_window = 0;
  std::vector<SummaryCache> summaries;
  std::vector<OrderedCache> ordered;
  ops::Workspace workspace;
  std::size_t generation = 0;
  std::size_t fused_scalar_calls = 0;
  std::size_t summary_source_scans = 0;
  std::size_t order_stat_sorts = 0;
  std::size_t algorithm_copy_bytes = 0;
  std::vector<std::size_t> rolling_invalid;
  std::vector<std::unique_ptr<Scratch>> children;

  void ensure(const Program &program, std::size_t max_window, bool isolate) {
    isolate_errors = isolate;
    if (program.numeric_slots &&
        max_window >
            std::numeric_limits<std::size_t>::max() / program.numeric_slots)
      throw ops::Error("GRAPH_ARENA_OVERFLOW");
    if (program.mask_slots &&
        max_window >
            std::numeric_limits<std::size_t>::max() / program.mask_slots)
      throw ops::Error("GRAPH_ARENA_OVERFLOW");
    ops::require(program.integer_slots == 0 || max_window <=
                 static_cast<std::size_t>(PTRDIFF_MAX) / program.integer_slots, "GRAPH_ARENA_OVERFLOW");
    integers.resize(program.integer_slots * max_window);
    numeric.resize(program.numeric_slots * max_window);
    masks.resize(program.mask_slots * max_window);
    values.resize(program.nodes.size());
    if (isolate_errors)
      statuses.resize(program.nodes.size());
    else if (statuses.capacity())
      std::vector<std::int16_t>().swap(statuses);
    const bool positions = isolate_errors &&
        program.execution_metadata->position_status_nodes != 0;
    const auto *position_identity = positions ? program.execution_metadata.get() : nullptr;
    const auto position_capacity = positions ? max_window : 0;
    if (position_owner.get() != position_identity || position_window != position_capacity) {
      // Retain buffers only for the same graph/geometry. An old, larger graph
      // must not leave hidden provenance allocations outside the current budget.
      std::vector<std::vector<std::int16_t>> fresh(positions ? program.nodes.size() : 0);
      position_statuses.swap(fresh);
      position_owner = positions ? program.execution_metadata : nullptr;
      position_window = positions ? max_window : 0;
    }
    summaries.resize(program.nodes.size());
    ordered.resize(program.nodes.size());
    const auto scope_count = program.rolling_scopes.size() + program.apply_scopes.size();
    if (children.size() < scope_count) children.resize(scope_count);
    for (std::size_t i = 0; i < scope_count; ++i)
      if (!children[i])
        children[i] = std::make_unique<Scratch>();
  }

  void next_generation() {
    ++generation;
    if (generation == 0) {
      for (auto &summary : summaries)
        summary.generation = 0;
      for (auto &cache : ordered)
        cache.generation = 0;
      generation = 1;
    }
  }
};

thread_local Scratch root_scratch;

std::int16_t position_error(const Scratch &scratch, std::size_t node,
                            std::size_t begin, std::size_t end) {
  if (scratch.position_statuses.empty()) return 0;
  const auto &errors = scratch.position_statuses[node];
  if (errors.empty()) return 0;
  ops::require(begin <= end && end <= errors.size(), "GRAPH_STATUS_GEOMETRY");
  return begin == end ? 0 : *std::max_element(errors.begin() + begin, errors.begin() + end);
}

std::int16_t position_error(const Scratch &scratch, std::size_t node) {
  if (scratch.position_statuses.empty()) return 0;
  return position_error(scratch, node, 0, scratch.position_statuses[node].size());
}

void mark_positions(Scratch &scratch, std::size_t node, std::size_t size,
                    std::size_t begin, std::size_t end, std::int16_t status) {
  if (!status || begin == end) return;
  if (scratch.position_statuses.empty() || !scratch.position_owner->position_status_reachable[node]) {
    // An injected whole-input fault in a strict child has no partial-row map.
    // Its output remains wholly failed while the child checks healthy inputs.
    scratch.statuses[node] = std::max(scratch.statuses[node], status);
    return;
  }
  ops::require(!scratch.position_statuses.empty() &&
      scratch.position_owner->position_status_reachable[node] &&
      begin <= end && end <= size && size <= scratch.position_window,
      "GRAPH_STATUS_GEOMETRY");
  auto &errors = scratch.position_statuses[node];
  if (errors.empty()) {
    // Reserve the planned upper bound once; geometric vector growth between
    // differently sized intervals must not exceed the admitted workspace.
    errors.reserve(scratch.position_window);
    errors.assign(size, 0);
  }
  ops::require(errors.size() == size, "GRAPH_STATUS_GEOMETRY");
  for (auto i = begin; i < end; ++i) errors[i] = std::max(errors[i], status);
}

std::int16_t capture_error(const Scratch &scratch,
                           const std::vector<std::uint32_t> &nodes,
                           std::size_t begin, std::size_t end) {
  if (!scratch.isolate_errors) return 0;
  std::int16_t status = 0;
  for (auto node : nodes) {
    status = std::max(status, scratch.statuses[node]);
    if (!scratch.statuses[node]) status = std::max(status, position_error(scratch, node, begin, end));
  }
  return status;
}

ops::Value interval_view(const ops::Value &base, std::size_t start,
                         std::size_t length) {
  ops::require((base.kind == ops::Kind::number || base.kind == ops::Kind::integer ||
                base.kind == ops::Kind::mask) && base.shape.rank >= 1 && base.shape.rank <= 2,
               "GRAPH_INPUT_TYPE");
  ops::require(start <= base.shape.dim[0] && length <= base.shape.dim[0] - start,
               "GRAPH_INTERVAL_BOUNDS");
  ops::Value view = base;
  view.shape.dim[0] = length;
  if (length && base.data) {
    const auto element_bytes = base.kind == ops::Kind::mask ? 1 : 8;
    view.data = static_cast<const char *>(base.data) +
                static_cast<std::ptrdiff_t>(start) * base.stride[0] * element_bytes;
  }
  return view;
}

bool summary_fusion_impl(ops::Op op) {
  switch (op) {
  case ops::Op::sum:
  case ops::Op::product:
  case ops::Op::mean:
  case ops::Op::min_value:
  case ops::Op::max_value:
  case ops::Op::variance:
  case ops::Op::std:
  case ops::Op::mean_absolute_deviation:
  case ops::Op::root_mean_square:
  case ops::Op::total_return:
    return true;
  default:
    return false;
  }
}

SummaryCache &summary_for(Scratch &scratch, std::size_t source_node,
                          const ops::Value &x) {
  auto &cache = scratch.summaries[source_node];
  if (cache.generation == scratch.generation)
    return cache;

  cache.generation = scratch.generation;
  ++scratch.summary_source_scans;
  cache.count = x.size();
  cache.total = 0.0;
  cache.mean = 0.0;
  cache.m2 = 0.0;
  cache.square_total = 0.0;
  cache.product = 1.0;
  cache.growth_product = 1.0;
  cache.mean_absolute_deviation = 0.0;
  cache.mad_ready = false;

  if (!cache.count)
    return cache;

  cache.minimum = x.f(0);
  cache.maximum = x.f(0);
  for (std::size_t i = 0; i < cache.count; ++i) {
    const double value = x.f(i);
    cache.total += value;
    cache.square_total += value * value;
    cache.product *= value;
    cache.growth_product *= value + 1.0;

    if (i) {
      if (value < cache.minimum)
        cache.minimum = value;
      if (value > cache.maximum)
        cache.maximum = value;
    }

    const double delta = value - cache.mean;
    cache.mean += delta / static_cast<double>(i + 1);
    cache.m2 += delta * (value - cache.mean);
  }
  return cache;
}

double summary_result(Scratch &scratch, const Node &node,
                      const ops::Prepared &prepared) {
  const auto op = prepared.spec->op;
  const auto source_node = static_cast<std::size_t>(node.parents[0]);
  const auto &x = prepared.args[0];
  auto &cache = summary_for(scratch, source_node, x);
  ops::require(cache.count > 0, "INSUFFICIENT_SAMPLE");

  switch (op) {
  case ops::Op::sum:
    return cache.total;
  case ops::Op::product:
    return cache.product;
  case ops::Op::mean:
    return cache.mean;
  case ops::Op::min_value:
    return cache.minimum;
  case ops::Op::max_value:
    return cache.maximum;
  case ops::Op::variance:
  case ops::Op::std: {
    const auto ddof = static_cast<std::size_t>(prepared.args[1].scalar);
    ops::require(ddof < cache.count, "INVALID_PARAMETER");
    const double variance = cache.m2 / static_cast<double>(cache.count - ddof);
    return op == ops::Op::std ? std::sqrt(variance) : variance;
  }
  case ops::Op::root_mean_square:
    return std::sqrt(cache.square_total / static_cast<double>(cache.count));
  case ops::Op::mean_absolute_deviation:
    if (!cache.mad_ready) {
      double total = 0.0;
      for (std::size_t i = 0; i < cache.count; ++i)
        total += std::abs(x.f(i) - cache.mean);
      cache.mean_absolute_deviation = total / static_cast<double>(cache.count);
      cache.mad_ready = true;
    }
    return cache.mean_absolute_deviation;
  case ops::Op::total_return:
    return cache.growth_product - 1.0;
  default:
    throw ops::Error("GRAPH_INVALID_SUMMARY_FUSION");
  }
}

const std::vector<double> &ordered_for(Scratch &scratch,
                                       std::size_t source_node,
                                       const ops::Value &x) {
  auto &cache = scratch.ordered[source_node];
  if (cache.generation == scratch.generation)
    return cache.values;

  cache.generation = scratch.generation;
  ++scratch.order_stat_sorts;
  cache.values.resize(x.size());
  scratch.algorithm_copy_bytes += x.size() * sizeof(double);
  for (std::size_t i = 0; i < x.size(); ++i)
    cache.values[i] = x.f(i);

  std::sort(cache.values.begin(), cache.values.end(), [](double a, double b) {
    if (std::isnan(a))
      return false;
    if (std::isnan(b))
      return true;
    return a < b;
  });
  return cache.values;
}

double ordered_result(Scratch &scratch, const Node &node,
                      const ops::Prepared &prepared) {
  const auto &ordered = ordered_for(
      scratch, static_cast<std::size_t>(node.parents[0]), prepared.args[0]);
  ops::require(!ordered.empty(), "INSUFFICIENT_SAMPLE");
  if (prepared.spec->op == ops::Op::median) {
    const auto middle = ordered.size() / 2;
    return ordered.size() % 2 ? ordered[middle]
                              : (ordered[middle - 1] + ordered[middle]) * 0.5;
  }

  ops::require(prepared.spec->op == ops::Op::quantile,
               "GRAPH_INVALID_ORDER_FUSION");
  const double probability = prepared.args[1].scalar;
  const double position = static_cast<double>(ordered.size() - 1) * probability;
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction;
}

ops::Value output_value(Scratch &scratch, const ops::Prepared &prepared,
                        const ops::Output &output, const Node &node,
                        std::size_t max_window) {
  ops::Value value;
  value.kind = prepared.output_kind;
  value.shape = prepared.output_shape;

  if (prepared.output_shape.rank == 0) {
    if (prepared.output_kind == ops::Kind::fit ||
        prepared.output_kind == ops::Kind::interval) {
      value.record = output.record;
    } else {
      value.scalar = output.scalar;
      value.integer = output.integer;
    }
    return value;
  }

  ops::require(prepared.output_shape.rank <= 2, "GRAPH_OUTPUT_RANK");
  ops::require(prepared.output_shape.size() <= max_window,
               "GRAPH_OUTPUT_TOO_LARGE");
  value.stride[0] = value.shape.rank == 2 ? value.shape.dim[1] : 1;
  value.stride[1] = 1;

  if (prepared.output_kind == ops::Kind::mask) {
    ops::require(node.storage == StorageKind::mask, "GRAPH_STORAGE_KIND");
    value.data =
        scratch.masks.data() + static_cast<std::size_t>(node.slot) * max_window;
  } else if (prepared.output_kind == ops::Kind::integer) {
    ops::require(node.storage == StorageKind::integer, "GRAPH_STORAGE_KIND");
    value.data = scratch.integers.data() + static_cast<std::size_t>(node.slot) * max_window;
  } else {
    ops::require(prepared.output_kind == ops::Kind::number,
                 "GRAPH_STORAGE_KIND");
    ops::require(node.storage == StorageKind::numeric, "GRAPH_STORAGE_KIND");
    value.data = scratch.numeric.data() +
                 static_cast<std::size_t>(node.slot) * max_window;
  }
  return value;
}

} // namespace

bool summary_fusion_eligible(ops::Op op) noexcept {
  return summary_fusion_impl(op);
}
bool order_fusion_eligible(ops::Op op) noexcept {
  return op == ops::Op::median || op == ops::Op::quantile;
}

void Program::validate() const {
  ops::require(!nodes.empty(), "GRAPH_EMPTY");
  ops::require(!roots.empty(), "GRAPH_NO_ROOTS");
  ops::require(static_cast<unsigned>(output_kind) <= 1 &&
                   static_cast<unsigned>(output_dtype) <= 2,
               "GRAPH_OUTPUT_TYPE");
  ops::require(output_kind == OutputKind::series || output_dtype == OutputDType::float64,
               "GRAPH_SCALAR_OUTPUT_DTYPE");
  ops::require(scope_work_budget >= 1 && scope_work_budget <= 1000000000000ULL, "GRAPH_SCOPE_WORK_BUDGET");
  ops::require(input_axes.empty() || input_axes.size() == input_count, "GRAPH_INPUT_AXES");
  for (auto axis : input_axes) ops::require(axis <= 2, "GRAPH_INPUT_AXES");

  for (std::size_t index = 0; index < nodes.size(); ++index) {
    const auto &node = nodes[index];
    ops::require(node.parent_count <= node.parents.size(),
                 "GRAPH_PARENT_COUNT");
    ops::require(static_cast<unsigned>(node.storage) <= 3,
                 "GRAPH_STORAGE_KIND");
    switch (node.kind) {
    case NodeKind::input:
      ops::require(node.input_index < input_count, "GRAPH_INPUT_INDEX");
      ops::require(node.parent_count == 0, "GRAPH_PARENT_COUNT");
      break;
    case NodeKind::parameter:
      ops::require(node.input_index < parameter_count, "GRAPH_PARAMETER_INDEX");
      ops::require(node.parent_count == 0, "GRAPH_PARENT_COUNT");
      break;
    case NodeKind::constant:
      ops::require(node.parent_count == 0, "GRAPH_PARENT_COUNT");
      break;
    case NodeKind::operation: {
      const auto &spec = ops::lookup(node.opcode);
      ops::require(node.parent_count >= spec.min_args &&
                       node.parent_count <= spec.max_args,
                   "GRAPH_ARITY");
      for (std::size_t parent = 0; parent < node.parent_count; ++parent)
        ops::require(node.parents[parent] < index, "GRAPH_NOT_TOPOLOGICAL");
      if (node.storage == StorageKind::numeric)
        ops::require(node.slot < numeric_slots, "GRAPH_SLOT");
      if (node.storage == StorageKind::mask)
        ops::require(node.slot < mask_slots, "GRAPH_SLOT");
      if (node.storage == StorageKind::integer)
        ops::require(node.slot < integer_slots, "GRAPH_SLOT");
      break;
    }
    case NodeKind::interval_tail:
      ops::require(node.parent_count == 1 && node.parents[0] < index &&
                   node.storage == StorageKind::inline_value, "GRAPH_INTERVAL_VIEW");
      break;
    case NodeKind::rolling_scope: {
      ops::require(node.input_index < rolling_scopes.size(),
                   "GRAPH_ROLLING_SCOPE_INDEX");
      ops::require(node.storage == StorageKind::numeric &&
                       node.slot < numeric_slots,
                   "GRAPH_SLOT");
      for (std::size_t parent = 0; parent < node.parent_count; ++parent)
        ops::require(node.parents[parent] < index, "GRAPH_NOT_TOPOLOGICAL");
      const auto &scope = rolling_scopes[node.input_index];
      ops::require(scope.body && scope.body->output_kind == OutputKind::scalar &&
                       scope.body->roots.size() == 1,
                   "GRAPH_ROLLING_BODY");
      ops::require(scope.input_nodes.size() == scope.input_preceding.size() &&
                       scope.body->input_count == scope.input_nodes.size() &&
                       scope.body->parameter_count ==
                           scope.parameter_bindings.size(),
                   "GRAPH_ROLLING_BINDINGS");
      for (auto source : scope.input_nodes)
        ops::require(source < index, "GRAPH_NOT_TOPOLOGICAL");
      ops::require(scope.width_node < index &&
                       (!scope.has_min_periods || scope.min_periods_node < index) &&
                       (!scope.has_date_context ||
                        (scope.dates_node < index && scope.annual_rate_node < index)),
                   "GRAPH_NOT_TOPOLOGICAL");
      if (scope.returns_input >= 0)
        ops::require(static_cast<std::size_t>(scope.returns_input) <
                         scope.input_nodes.size(),
                     "GRAPH_ROLLING_BINDINGS");
      scope.body->validate();
      break;
    }
    case NodeKind::apply_scope: {
      ops::require(node.input_index < apply_scopes.size(), "GRAPH_APPLY_SCOPE_INDEX");
      const auto &scope = apply_scopes[node.input_index];
      ops::require(scope.body && scope.body->output_kind == OutputKind::scalar &&
                   scope.body->roots.size() == 1 &&
                   scope.input_nodes.size() == scope.body->input_count &&
                   scope.parameter_nodes.size() == scope.body->parameter_count,
                   "GRAPH_APPLY_BINDINGS");
      const auto argc = scope.argument_nodes.size();
      ops::require((scope.kind == ApplyKind::bisect && argc == 4) ||
                   (scope.kind == ApplyKind::filter && (argc == 1 || argc == 2)) ||
                   ((scope.kind == ApplyKind::block || scope.kind == ApplyKind::group ||
                     scope.kind == ApplyKind::segment) && argc == 1),
                   "GRAPH_APPLY_ARGUMENTS");
      const bool array = scope.kind == ApplyKind::block || scope.kind == ApplyKind::group ||
                         scope.kind == ApplyKind::segment;
      ops::require(array ? node.storage == StorageKind::numeric && node.slot < numeric_slots
                         : node.storage == StorageKind::inline_value, "GRAPH_SLOT");
      auto dependency = [&](std::uint32_t id) {
        ops::require(id < index && std::find(node.parents.begin(),
            node.parents.begin() + node.parent_count, id) != node.parents.begin() + node.parent_count,
            "GRAPH_APPLY_DEPENDENCY");
      };
      for (std::size_t i = 0; i < node.parent_count; ++i) dependency(node.parents[i]);
      for (auto id : scope.input_nodes) dependency(id);
      for (auto id : scope.argument_nodes) dependency(id);
      for (auto id : scope.parameter_nodes) {
        if (id == std::numeric_limits<std::uint32_t>::max())
          ops::require(scope.kind == ApplyKind::bisect, "GRAPH_APPLY_PARAMETER");
        else dependency(id);
      }
      scope.body->validate();
      break;
    }
    default:
      throw ops::Error("GRAPH_NODE_KIND");
    }
  }

  for (auto root : roots)
    ops::require(root < nodes.size(), "GRAPH_ROOT_INDEX");
}

void Program::finalize() {
  validate();
  auto metadata = std::make_shared<ExecutionMetadata>();
  metadata->requires_shape_planning = std::any_of(input_axes.begin(), input_axes.end(),
      [](std::uint8_t axis) { return axis != 0; });
  metadata->specs.resize(nodes.size(), nullptr);
  metadata->summary_consumers.resize(nodes.size(), 0);
  metadata->order_consumers.resize(nodes.size(), 0);
  metadata->position_status_reachable.resize(nodes.size(), 0);
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const auto &node = nodes[i];
    bool positions = false;
    if (node.kind == NodeKind::apply_scope) {
      const auto &scope = apply_scopes[node.input_index];
      if (!scope.body->execution_metadata) scope.body->finalize();
      positions = scope.kind == ApplyKind::segment ||
          scope.body->execution_metadata->position_status_nodes != 0;
    } else if (node.kind == NodeKind::rolling_scope) {
      const auto &scope = rolling_scopes[node.input_index];
      if (!scope.body->execution_metadata) scope.body->finalize();
      positions = scope.body->execution_metadata->position_status_nodes != 0;
    }
    for (std::size_t parent = 0; parent < node.parent_count; ++parent)
      positions = positions || metadata->position_status_reachable[node.parents[parent]];
    metadata->position_status_reachable[i] = positions;
    metadata->position_status_nodes += positions;
    if (node.kind != NodeKind::operation)
      continue;
    const auto &spec = ops::lookup(node.opcode);
    metadata->specs[i] = &spec;
    metadata->requires_shape_planning = metadata->requires_shape_planning || spec.family == ops::Family::matrix ||
        spec.op == ops::Op::scalar_kalman ||
        (static_cast<std::uint16_t>(spec.op) >= 127 && spec.family == ops::Family::state);
    if (!node.parent_count)
      continue;
    const auto source = node.parents[0];
    if (summary_fusion_eligible(spec.op) &&
        metadata->summary_consumers[source] < 255)
      ++metadata->summary_consumers[source];
    if (order_fusion_eligible(spec.op) &&
        metadata->order_consumers[source] < 255)
      ++metadata->order_consumers[source];
  }
  execution_metadata = std::move(metadata);
}

std::size_t required_array_capacity(const Program &program,
                                   const std::vector<ops::Value> &inputs,
                                   std::size_t max_window,
                                   std::vector<ops::Shape> *node_shapes) {
  ops::require(inputs.size() == program.input_count, "GRAPH_INPUT_COUNT");
  if (!node_shapes && program.execution_metadata &&
      !program.execution_metadata->requires_shape_planning &&
      std::all_of(inputs.begin(), inputs.end(), [](const ops::Value &input) { return input.shape.rank == 1; }))
    return std::max<std::size_t>(max_window, 1);
  std::size_t capacity = std::max<std::size_t>(max_window, 1);
  std::vector<ops::Shape> shapes(program.nodes.size());
  for (std::size_t i = 0; i < program.nodes.size(); ++i) {
    const auto &node = program.nodes[i];
    auto &shape = shapes[i];
    if (node.kind == NodeKind::input) {
      shape = inputs[node.input_index].shape;
      // Unavailable geometry cannot produce numerical arrays. This is only a
      // capacity bound; the executor retains the unknown descriptor unchanged.
      if (shape.rank < 0) shape = ops::vector_shape(0);
      const auto axis = program.input_axes.empty() ? 0 : program.input_axes[node.input_index];
      if (axis != 1 && shape.rank) shape.dim[0] = std::min(shape.dim[0], max_window);
    } else if (node.kind == NodeKind::interval_tail) shape = shapes[node.parents[0]];
    else if (node.kind == NodeKind::operation) {
      const auto op = static_cast<ops::Op>(node.opcode);
      if (node.storage != StorageKind::inline_value || op == ops::Op::lag || op == ops::Op::transpose ||
          ops::series_record_projection(op) ||
          (op == ops::Op::diag && shapes[node.parents[0]].rank == 2)) {
        for (std::size_t j = 0; j < node.parent_count; ++j)
          if (shapes[node.parents[j]].rank > shape.rank ||
              shapes[node.parents[j]].size() > shape.size()) shape = shapes[node.parents[j]];
        const auto lhs = shapes[node.parents[0]];
        const auto rhs = node.parent_count > 1 ? shapes[node.parents[1]] : ops::Shape{};
        if (op == ops::Op::scalar_kalman || op == ops::Op::between_events)
          shape = ops::matrix_shape(lhs.dim[0], 2);
        else if (op == ops::Op::state_continuous)
          shape = ops::matrix_shape(lhs.dim[0], 3);
        else if (ops::series_record_projection(op))
          shape = ops::vector_shape(lhs.dim[0]);
        else if (op >= ops::Op::sum_time && op <= ops::Op::max_time && lhs.rank == 2)
          shape = ops::vector_shape(lhs.dim[1]);
        else if (op >= ops::Op::sum_asset && op <= ops::Op::max_asset && lhs.rank == 2)
          shape = ops::vector_shape(lhs.dim[0]);
        else if (op == ops::Op::transpose && lhs.rank == 2)
          shape = ops::matrix_shape(lhs.dim[1], lhs.dim[0]);
        else if (op == ops::Op::outer) shape = ops::matrix_shape(lhs.size(), rhs.size());
        else if (op == ops::Op::matmul) shape = ops::matrix_shape(lhs.dim[0], rhs.dim[1]);
        else if (op == ops::Op::matvec || op == ops::Op::portfolio_returns)
          shape = ops::vector_shape(lhs.dim[0]);
        else if (op == ops::Op::diag)
          shape = lhs.rank == 1 ? ops::matrix_shape(lhs.dim[0], lhs.dim[0])
                               : ops::vector_shape(std::min(lhs.dim[0], lhs.dim[1]));
        else if (op == ops::Op::solve) shape = rhs;
        else if ((op == ops::Op::covariance || op == ops::Op::correlation) && lhs.rank == 2)
          shape = ops::matrix_shape(lhs.dim[1], lhs.dim[1]);
        else if (op == ops::Op::gather) shape = rhs;
        else if (static_cast<std::uint16_t>(op) >= 127) shape = lhs;
      }
    } else if (node.kind == NodeKind::rolling_scope) shape = ops::vector_shape(max_window);
    else if (node.kind == NodeKind::apply_scope && node.storage != StorageKind::inline_value) {
      const auto &scope = program.apply_scopes[node.input_index];
      if (scope.kind == ApplyKind::block && scope.input_nodes.empty()) shape = ops::vector_shape(max_window);
      else if (scope.kind == ApplyKind::segment)
        shape = ops::vector_shape(shapes[scope.argument_nodes[0]].dim[0]);
      else {
        const auto source = scope.kind == ApplyKind::group ? scope.argument_nodes[0] : scope.input_nodes[0];
        shape = ops::vector_shape(shapes[source].size());
      }
    }
    capacity = std::max(capacity, shape.size());
  }
  if (node_shapes) *node_shapes = std::move(shapes);
  return capacity;
}

static Audit execute_impl(const Program &program,
                          const std::vector<ops::Value> &inputs,
                          const double *parameters,
                          std::size_t parameter_count,
                          const std::int64_t *starts,
                          const std::int64_t *ends,
                          std::size_t rows, void *output,
                          std::size_t output_columns, Scratch &scratch,
                          bool prebound_inputs = false,
                          const std::vector<std::int16_t> *input_failures = nullptr,
                          bool interval_known = true);

static std::size_t positive_integer(double value, const char *code,
                                    std::size_t maximum = 5000) {
  ops::require(std::isfinite(value) && value >= 1.0 &&
                   value <= static_cast<double>(maximum) &&
                   std::floor(value) == value,
               code);
  return static_cast<std::size_t>(value);
}

static bool numerical_error(const ops::Error &error) {
  const std::string_view code(error.what());
  return code == "DIVIDE_BY_ZERO" || code == "DOMAIN_ERROR" ||
      code == "INSUFFICIENT_SAMPLE" || code == "INVALID_PARAMETER" ||
      code == "SINGULAR_MATRIX" || code == "NONFINITE_RESULT" ||
      code == "NON_FINITE_RESULT" || code == "ROOT_NOT_BRACKETED" ||
      code == "NON_CONVERGENCE";
}

// Failed values retain their type and, when derivable, their exact geometry.
// rank=-1 is internal unknown geometry, never an executable array or a length
// estimate. Payload availability is carried separately by Scratch statuses.
static bool known_geometry(const ops::Value &value) { return value.shape.rank >= 0; }
static bool available_payload(const Scratch &scratch, std::uint32_t id) {
  return (!scratch.isolate_errors || scratch.statuses[id] == 0) &&
      position_error(scratch, id) == 0;
}

static std::int16_t node_failure(const Scratch &scratch, std::uint32_t id) {
  return scratch.isolate_errors ? std::max(scratch.statuses[id], position_error(scratch, id)) : 0;
}

static ops::Value scope_view(const ops::Value &value, std::size_t start, std::size_t length) {
  return known_geometry(value) ? interval_view(value, start, length) : value;
}

static ops::Value validate_apply_structure(const ApplyScope &scope,
    std::size_t interval_length, const Scratch &scratch, bool isolate) {
  const auto available = [&](std::uint32_t id) {
    return !isolate || available_payload(scratch, id);
  };
  std::size_t count = 0;
  bool count_known = false;
  for (auto id : scope.input_nodes) {
    const auto &value = scratch.values[id];
    if (!known_geometry(value)) continue;
    ops::require(value.shape.rank == 1, "SCOPE_CAPTURE_RANK");
    if (count_known && scope.kind != ApplyKind::bisect)
      ops::require(value.size() == count, "SCOPE_ALIGNMENT_MISMATCH");
    count = std::max(count, value.size());
    count_known = true;
  }
  const auto scalar = [&](std::uint32_t id) {
    const auto &value = scratch.values[id];
    ops::require(value.kind == ops::Kind::number &&
        (!known_geometry(value) || value.shape.rank == 0), "SCOPE_PARAMETER_TYPE");
  };
  for (auto id : scope.parameter_nodes)
    if (id != std::numeric_limits<std::uint32_t>::max()) scalar(id);
  for (std::size_t i = 0; i < scope.argument_nodes.size(); ++i)
    if (scope.kind == ApplyKind::bisect || scope.kind == ApplyKind::block || i != 0)
      scalar(scope.argument_nodes[i]);

  if (scope.kind == ApplyKind::filter || scope.kind == ApplyKind::group ||
      scope.kind == ApplyKind::segment) {
    const auto id = scope.argument_nodes[0];
    const auto &selector = scratch.values[id];
    const bool segment = scope.kind == ApplyKind::segment;
    const auto kind = scope.kind == ApplyKind::filter ? ops::Kind::mask : ops::Kind::integer;
    const auto code = segment ? "SCOPE_SEGMENT_GEOMETRY" : "SCOPE_ALIGNMENT_MISMATCH";
    ops::require(selector.kind == kind, code);
    if (known_geometry(selector)) {
      ops::require(selector.shape.rank == (segment ? 2 : 1), code);
      if (segment) ops::require(selector.shape.dim[1] == 2, code);
      const auto size = selector.shape.dim[0];
      if (count_known) ops::require(size == count, code);
      count = size;
      count_known = true;
      if (available(id)) {
        if (scope.kind == ApplyKind::filter) {
          for (std::size_t i = 0; i < size; ++i)
            ops::require(selector.u(i) <= 1, "INVALID_MASK");
        } else if (segment) {
          for (std::size_t left = 0; left < size;) {
            const auto start = selector.i(2 * left), end = selector.i(2 * left + 1);
            if (start == -1 && end == -1) { ++left; continue; }
            ops::require(start == static_cast<std::int64_t>(left) && end > start &&
                static_cast<std::uint64_t>(end) < size, "SCOPE_SEGMENT_BOUNDS");
            const auto right = static_cast<std::size_t>(end);
            for (auto row = left; row < right; ++row)
              ops::require(selector.i(2 * row) == start && selector.i(2 * row + 1) == end,
                           "SCOPE_SEGMENT_GEOMETRY");
            left = right;
          }
        }
      }
    }
  }
  auto result = ops::Value::number(NAN);
  if (scope.kind == ApplyKind::filter || scope.kind == ApplyKind::bisect) return result;
  if (scope.kind == ApplyKind::block) {
    if (scope.input_nodes.empty()) { count = interval_length; count_known = true; }
    const auto id = scope.argument_nodes[0];
    const auto width = scratch.values[id].scalar;
    if (available(id) && std::isfinite(width) && width >= 1 &&
        width <= 1000000 && std::floor(width) == width) count /= static_cast<std::size_t>(width);
    else count_known = false;
  }
  result.shape = ops::vector_shape(count);
  if (!count_known) result.shape.rank = -1;
  return result;
}

static ops::Value validate_rolling_structure(const Program &program,
    const std::vector<ops::Value> &raw_inputs, const RollingScope &scope,
    std::size_t start, std::size_t length, const Scratch &scratch) {
  ops::require(scope.body && scope.body->output_kind == OutputKind::scalar, "GRAPH_ROLLING_BODY");
  const auto available = [&](std::uint32_t id) {
    return !scratch.isolate_errors || available_payload(scratch, id);
  };
  const auto aligned = [&](std::uint32_t id) {
    const auto &value = scratch.values[id];
    ops::require(value.kind == ops::Kind::number && (!known_geometry(value) ||
        (value.shape.rank == 1 && (!scratch.interval_known || value.size() == length))),
        "ROLLING_ALIGNMENT_MISMATCH");
  };
  const auto scalar = [&](std::uint32_t id) {
    const auto &value = scratch.values[id];
    ops::require(value.kind == ops::Kind::number &&
        (!known_geometry(value) || value.shape.rank == 0), "SCOPE_PARAMETER_TYPE");
  };
  for (auto id : scope.input_nodes) aligned(id);
  scalar(scope.width_node);
  if (scope.has_min_periods) {
    scalar(scope.min_periods_node);
    if (available(scope.min_periods_node)) {
      const auto minimum = positive_integer(scratch.values[scope.min_periods_node].scalar,
                                           "INVALID_MIN_PERIODS");
      const auto width = scratch.values[scope.width_node].scalar;
      if (available(scope.width_node) && std::isfinite(width) && width >= 1 && width <= 5000 &&
          std::floor(width) == width)
        ops::require(minimum <= width, "INVALID_MIN_PERIODS");
    }
  }
  for (const auto &binding : scope.parameter_bindings)
    if (binding.kind == RollingParameterKind::outer_node) scalar(binding.node);
  if (scope.has_date_context) {
    aligned(scope.dates_node);
    scalar(scope.annual_rate_node);
    const auto &dates = scratch.values[scope.dates_node];
    if (scratch.interval_known && available(scope.dates_node)) {
      for (std::size_t i = 0; i < length; ++i)
        ops::require(std::isfinite(dates.f(i)) && (i == 0 || dates.f(i) > dates.f(i - 1)),
                     "ROLLING_DATE_AXIS_INVALID");
      if (scope.needs_preceding_observation) {
        const auto &date_node = program.nodes[scope.dates_node];
        ops::require(date_node.kind == NodeKind::input, "ROLLING_DATE_CONTEXT_MUST_BE_INPUT");
        const auto &raw_dates = raw_inputs[date_node.input_index];
        if (start > 0 && length)
          ops::require(std::isfinite(raw_dates.f(start - 1)) && raw_dates.f(start - 1) < dates.f(0),
                       "ROLLING_DATE_AXIS_INVALID");
      }
    }
  }
  auto result = ops::Value::number(NAN);
  result.shape = ops::vector_shape(length);
  return result;
}

static void validate_unresolved_scope(const Program &program, const Node &node,
    std::size_t length, Scratch &scratch) {
  const bool rolling = node.kind == NodeKind::rolling_scope;
  const auto &captures = rolling ? program.rolling_scopes[node.input_index].input_nodes
                                : program.apply_scopes[node.input_index].input_nodes;
  const auto &body = rolling ? *program.rolling_scopes[node.input_index].body
                            : *program.apply_scopes[node.input_index].body;
  const bool bisect = !rolling && program.apply_scopes[node.input_index].kind == ApplyKind::bisect;
  bool body_interval_known = bisect;
  if (bisect) {
    length = 0;
    for (auto id : captures) {
      if (known_geometry(scratch.values[id])) length = std::max(length, scratch.values[id].size());
      else body_interval_known = false;
    }
  }
  std::vector<ops::Value> inputs;
  std::vector<std::int16_t> failures;
  inputs.reserve(captures.size());
  failures.reserve(captures.size());
  for (auto id : captures) {
    auto value = scratch.values[id];
    auto failure = node_failure(scratch, id);
    if (!bisect) {
      // Unknown membership makes selected payload and shape unavailable. Do
      // not validate arbitrary full-input rows as if they had been selected.
      value.data = nullptr;
      value.shape.rank = -1;
      failure = 4;
    }
    inputs.push_back(value);
    failures.push_back(failure);
  }
  std::vector<double> parameters(body.parameter_count, NAN);
  const auto scalar = [&](std::uint32_t id) {
    return available_payload(scratch, id) ? scratch.values[id].scalar : NAN;
  };
  if (rolling) {
    const auto &scope = program.rolling_scopes[node.input_index];
    for (std::size_t i = 0; i < parameters.size(); ++i)
      if (scope.parameter_bindings[i].kind == RollingParameterKind::outer_node)
        parameters[i] = scalar(scope.parameter_bindings[i].node);
  } else {
    const auto &scope = program.apply_scopes[node.input_index];
    for (std::size_t i = 0; i < parameters.size(); ++i)
      if (scope.parameter_nodes[i] != std::numeric_limits<std::uint32_t>::max())
        parameters[i] = scalar(scope.parameter_nodes[i]);
  }
  auto &child = *scratch.children[(rolling ? 0 : program.rolling_scopes.size()) + node.input_index];
  child.remaining_work = scratch.remaining_work;
  const std::int64_t start = 0, end = static_cast<std::int64_t>(length);
  for (std::size_t endpoint = 0; endpoint < (bisect ? 2u : 1u); ++endpoint) {
    if (bisect) {
      const auto &scope = program.apply_scopes[node.input_index];
      for (std::size_t i = 0; i < parameters.size(); ++i)
        if (scope.parameter_nodes[i] == std::numeric_limits<std::uint32_t>::max())
          parameters[i] = scalar(scope.argument_nodes[endpoint]);
    }
    if (child.remaining_work) {
      const auto rows = std::max<std::size_t>(length, 1);
      const auto nodes = std::max<std::size_t>(body.nodes.size(), 1);
      ops::require(rows <= *child.remaining_work / nodes, "SCOPE_COMPUTE_BUDGET_EXCEEDED");
      *child.remaining_work -= rows * nodes;
    }
    double ignored = NAN;
    const auto audit = execute_impl(body, inputs, parameters.data(), parameters.size(),
        &start, &end, 1, &ignored, 1, child, true, &failures, body_interval_known);
    scratch.algorithm_copy_bytes += audit.algorithm_copy_bytes;
  }
}

static ops::Value validate_failed_node(const Program &program,
    const std::vector<ops::Value> &inputs, const Node &node,
    std::size_t start, std::size_t length, Scratch &scratch) {
  if (node.kind == NodeKind::apply_scope || node.kind == NodeKind::rolling_scope) {
    auto result = node.kind == NodeKind::apply_scope
        ? validate_apply_structure(program.apply_scopes[node.input_index], length, scratch, true)
        : validate_rolling_structure(program, inputs, program.rolling_scopes[node.input_index],
                                     start, length, scratch);
    validate_unresolved_scope(program, node, length, scratch);
    if (!scratch.interval_known && result.shape.rank != 0) result.shape.rank = -1;
    return result;
  }
  if (node.kind == NodeKind::interval_tail) {
    auto value = scratch.values[node.parents[0]];
    if (known_geometry(value)) {
      ops::require(value.shape.rank == 1, "GRAPH_INPUT_TYPE");
      value.shape = ops::vector_shape(value.size() ? value.size() - 1 : 0);
    }
    value.data = nullptr;
    return value;
  }
  ops::require(node.kind == NodeKind::operation, "GRAPH_FAILED_NODE_KIND");
  std::array<ops::Value, 8> args{};
  std::uint8_t geometry = 0, payload = 0;
  for (std::size_t i = 0; i < node.parent_count; ++i) {
    const auto id = node.parents[i];
    args[i] = scratch.values[id];
    if (known_geometry(args[i])) geometry |= static_cast<std::uint8_t>(1u << i);
    if (available_payload(scratch, id)) payload |= static_cast<std::uint8_t>(1u << i);
  }
  const auto structure = ops::validate_structure(ops::lookup(node.opcode), args.data(),
                                                node.parent_count, geometry, payload);
  auto value = ops::Value::number(NAN);
  value.kind = structure.output_kind;
  value.shape = structure.output_shape;
  if (!structure.geometry_known) value.shape.rank = -1;
  return value;
}

static bool pointwise_status(const ops::Spec &spec) {
  return spec.family == ops::Family::elementwise || spec.op == ops::Op::state_select;
}

static bool mapped_status(ops::Op op) {
  return op == ops::Op::lag || op == ops::Op::difference || op == ops::Op::aligned_shift ||
      op == ops::Op::first || op == ops::Op::last || op == ops::Op::length;
}

// Only membership controls decide whether real slices can be visited. A failed
// body parameter or an unavailable capture does not prevent child validation.
static bool scope_maps_status(const Program &program, const Node &node,
                               const Scratch &scratch) {
  const auto failed = [&](auto id) { return !available_payload(scratch, id); };
  if (node.kind == NodeKind::apply_scope) {
    const auto &scope = program.apply_scopes[node.input_index];
    if (scope.kind == ApplyKind::bisect)
      return std::none_of(scope.argument_nodes.begin(), scope.argument_nodes.end(), failed);
    if (failed(scope.argument_nodes[0])) return false;
    if (scope.kind != ApplyKind::block) return known_geometry(scratch.values[scope.argument_nodes[0]]);
    return (scope.input_nodes.empty() && scratch.interval_known) || std::any_of(scope.input_nodes.begin(), scope.input_nodes.end(),
        [&](auto id) { return known_geometry(scratch.values[id]); });
  }
  if (node.kind != NodeKind::rolling_scope) return false;
  const auto &scope = program.rolling_scopes[node.input_index];
  return scratch.interval_known && !failed(scope.width_node) && (!scope.has_min_periods || !failed(scope.min_periods_node)) &&
      (!scope.has_date_context || (!failed(scope.dates_node) && !failed(scope.annual_rate_node)));
}

static void map_position_status(Scratch &scratch, std::size_t node_index,
                                 const Node &node, const ops::Prepared &prepared) {
  const auto op = prepared.spec->op;
  const auto source = node.parents[0];
  const auto count = prepared.output_shape.size();
  if (op == ops::Op::length) return; // Shape, not failed numerical observations.
  if (op == ops::Op::first || op == ops::Op::last) {
    const auto i = op == ops::Op::first ? 0 : prepared.args[0].size() - 1;
    scratch.statuses[node_index] = position_error(scratch, source, i, i + 1);
    return;
  }
  const auto periods = static_cast<std::size_t>(prepared.args[1].scalar);
  for (std::size_t i = 0; i < count; ++i) {
    std::int16_t status = 0;
    if (op == ops::Op::aligned_shift) {
      if (i >= periods) status = position_error(scratch, source, i - periods, i - periods + 1);
    } else {
      status = position_error(scratch, source, i, i + 1);
      if (op == ops::Op::difference)
        status = std::max(status, position_error(scratch, source, i + periods, i + periods + 1));
    }
    if (status) mark_positions(scratch, node_index, count, i, i + 1, status);
  }
}

static void execute_pointwise_with_status(const Node &node, std::size_t node_index,
    const ops::Prepared &prepared, ops::Output &output, Scratch &scratch) {
  const auto size = output.shape.size();
  // Structural mask validation still covers failed rows. Skipping their
  // numerical evaluation must not turn an invalid bool payload into success.
  for (std::size_t parent = 0; parent < prepared.count; ++parent)
    if (prepared.args[parent].kind == ops::Kind::mask)
      for (std::size_t i = 0; i < prepared.args[parent].size(); ++i)
        ops::require(prepared.args[parent].u(i) <= 1, "INVALID_MASK");
  for (std::size_t i = 0; i < size; ++i) {
    std::int16_t status = 0;
    for (std::size_t parent = 0; parent < node.parent_count; ++parent)
      status = std::max(status, position_error(scratch, node.parents[parent], i, i + 1));
    ops::Output cell = output;
    cell.shape = ops::vector_shape(1);
    const auto width = output.kind == ops::Kind::mask ? 1u : 8u;
    cell.data = static_cast<char *>(output.data) + i * width;
    if (!status) {
      // Kernels validate numerical domains. Never feed failed placeholders into
      // them: log/sqrt/etc. could otherwise turn one bad row into a whole-root failure.
      auto args = prepared.args;
      for (std::size_t parent = 0; parent < prepared.count; ++parent)
        if (args[parent].shape.rank) args[parent] = interval_view(args[parent], i, 1);
      try {
        const auto item = ops::prepare(*prepared.spec, args.data(), prepared.count);
        ops::Audit audit;
        ops::execute(item, cell, scratch.workspace, ops::Isa::automatic, audit);
        scratch.algorithm_copy_bytes += audit.algorithm_copy_bytes;
      } catch (const ops::Error &error) {
        if (!numerical_error(error)) throw;
        status = 4;
      }
    }
    if (status) {
      mark_positions(scratch, node_index, size, i, i + 1, status);
      if (output.kind == ops::Kind::mask) cell.set_mask(0, 0);
      else if (output.kind == ops::Kind::integer) cell.set_integer(0, 0);
      else cell.set(0, NAN);
    }
  }
}

static ops::Value execute_rolling_scope(
    const Program &program, const std::vector<ops::Value> &raw_inputs,
    const RollingScope &scope, const Node &node, std::size_t node_index, std::size_t interval_start,
    std::size_t interval_length, std::size_t max_window, Scratch &scratch) {
  validate_rolling_structure(program, raw_inputs, scope, interval_start, interval_length, scratch);
  const auto width =
      positive_integer(scratch.values[scope.width_node].scalar,
                       "INVALID_PARAMETER");
  const auto minimum =
      scope.has_min_periods
          ? positive_integer(scratch.values[scope.min_periods_node].scalar,
                             "INVALID_MIN_PERIODS", width)
          : width;
  ops::require(minimum <= width, "INVALID_MIN_PERIODS");

  const auto work_window = std::min(interval_length, width);
  ops::require(scope.body_node_count == 0 ||
                   interval_length <=
                       100000000ull /
                           std::max<std::size_t>(
                               1, work_window * scope.body_node_count),
               "ROLLING_COMPUTE_BUDGET_EXCEEDED");
  const auto scratch_bytes =
      interval_length * 16ull + 8ull +
      (width + 1ull) * std::max<std::size_t>(1, scope.array_count) * 8ull;
  ops::require(scratch_bytes <= 64ull * 1024ull * 1024ull,
               "ROLLING_MEMORY_BUDGET_EXCEEDED");

  std::vector<ops::Value> captured;
  captured.reserve(scope.input_nodes.size());
  for (auto source_node : scope.input_nodes) {
    const auto &value = scratch.values[source_node];
    captured.push_back(value);
  }

  const ops::Value *dates = nullptr;
  const ops::Value *raw_dates = nullptr;
  double annual = 0.0;
  if (scope.has_date_context) {
    dates = &scratch.values[scope.dates_node];
    annual = scratch.values[scope.annual_rate_node].scalar;
    ops::require(std::isfinite(annual), "INVALID_PARAMETER");
    if (scope.needs_preceding_observation) {
      const auto &date_node = program.nodes[scope.dates_node];
      raw_dates = &raw_inputs[date_node.input_index];
    }
  }

  scratch.rolling_invalid.assign(interval_length + 1, 0);
  for (std::size_t i = 0; i < interval_length; ++i) {
    bool invalid = false;
    for (std::size_t j = 0; j < captured.size(); ++j)
      invalid = invalid || (scratch.isolate_errors && scratch.statuses[scope.input_nodes[j]]) ||
          !std::isfinite(captured[j].f(i));
    scratch.rolling_invalid[i + 1] =
        scratch.rolling_invalid[i] + (invalid ? 1u : 0u);
  }

  ops::Value result;
  result.kind = ops::Kind::number;
  result.shape = ops::vector_shape(interval_length);
  result.stride[0] = 1;
  auto *destination =
      scratch.numeric.data() + static_cast<std::size_t>(node.slot) * max_window;
  result.data = destination;
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  std::fill(destination, destination + interval_length, nan);

  auto &child = *scratch.children[node.input_index];
  child.remaining_work = scratch.remaining_work;
  std::vector<ops::Value> body_inputs(scope.input_nodes.size());
  std::vector<double> body_parameters(scope.parameter_bindings.size());
  std::int16_t parameter_failure = 0;
  for (const auto &binding : scope.parameter_bindings)
    if (binding.kind == RollingParameterKind::outer_node)
      parameter_failure = std::max(parameter_failure, node_failure(scratch, binding.node));

  for (std::size_t right = 0; right < interval_length; ++right) {
    std::size_t left = right + 1 > width ? right + 1 - width : 0;
    if (scope.has_min_periods && scope.needs_preceding_observation &&
        interval_start + left == 0 && right > 0)
      left = 1;
    const auto count = right + 1 - left;
    const auto invalid =
        scratch.rolling_invalid[right + 1] - scratch.rolling_invalid[left];
    if (!scope.has_min_periods && count != width)
      continue;
    const auto inherited = std::max(parameter_failure,
        capture_error(scratch, scope.input_nodes, left, right + 1));
    if (inherited) {
      mark_positions(scratch, node_index, interval_length, right, right + 1, inherited);
    }
    if (!inherited && count - invalid < minimum)
      continue;

    const auto global_left = interval_start + left;
    const auto global_right = interval_start + right;
    if (scope.needs_preceding_observation && global_left == 0)
      continue;

    bool preceding_valid = true;
    for (std::size_t i = 0; i < captured.size(); ++i) {
      if (i < scope.input_preceding.size() && scope.input_preceding[i]) {
        const auto &source_node = program.nodes[scope.input_nodes[i]];
        ops::require(source_node.kind == NodeKind::input,
                     "ROLLING_PRECEDING_SOURCE_MUST_BE_INPUT");
        const auto &raw = raw_inputs[source_node.input_index];
        if (global_left == 0 || !std::isfinite(raw.f(global_left - 1))) {
          preceding_valid = false;
          break;
        }
        body_inputs[i] = interval_view(raw, global_left - 1, count + 1);
      } else {
        body_inputs[i] = scope_view(captured[i], left, count);
      }
    }
    if (!preceding_valid)
      continue;

    double elapsed = static_cast<double>(count > 0 ? count - 1 : 0);
    if (dates) {
      if (scope.needs_preceding_observation) {
        ops::require(raw_dates != nullptr, "ROLLING_DATE_CONTEXT_MUST_BE_INPUT");
        elapsed = raw_dates->f(global_right) - raw_dates->f(global_left - 1);
      } else
        elapsed = dates->f(right) - dates->f(left);
    }

    double observation_count =
        scope.has_returns ? static_cast<double>(count)
                          : static_cast<double>(count > 0 ? count - 1 : 0);
    if (scope.has_returns && scope.has_min_periods &&
        scope.returns_input >= 0) {
      const auto index = static_cast<std::size_t>(scope.returns_input);
      const auto id = scope.input_nodes[index];
      if (scratch.isolate_errors && (scratch.statuses[id] ||
          position_error(scratch, id, left, right + 1))) {
        // This derived parameter depends on failed observations. NaN marks it
        // unavailable at the child's parameter boundary; never invent a count.
        observation_count = NAN;
      } else {
        observation_count = 0.0;
        for (std::size_t i = left; i <= right; ++i)
          if (std::isfinite(captured[index].f(i))) observation_count += 1.0;
      }
    }

    for (std::size_t i = 0; i < scope.parameter_bindings.size(); ++i) {
      const auto &binding = scope.parameter_bindings[i];
      switch (binding.kind) {
      case RollingParameterKind::outer_node:
        body_parameters[i] = available_payload(scratch, binding.node) ? scratch.values[binding.node].scalar : NAN;
        break;
      case RollingParameterKind::observation_count:
        body_parameters[i] = observation_count;
        break;
      case RollingParameterKind::window_elapsed_days:
        body_parameters[i] = elapsed;
        break;
      case RollingParameterKind::risk_free_return_window:
        body_parameters[i] =
            std::pow(std::max(0.0, 1.0 + annual), elapsed / 365.0) - 1.0;
        break;
      }
    }

    const std::int64_t local_start = 0;
    const auto local_end = static_cast<std::int64_t>(count);
    double value = nan;
    std::vector<std::int16_t> failures;
    if (inherited) {
      failures.resize(captured.size());
      for (std::size_t i = 0; i < captured.size(); ++i) {
        const auto id = scope.input_nodes[i];
        failures[i] = scratch.statuses[id];
        if (!failures[i]) failures[i] = position_error(scratch, id, left, right + 1);
      }
    }
    try {
      execute_impl(*scope.body, body_inputs,
                   body_parameters.empty() ? nullptr : body_parameters.data(),
                   body_parameters.size(), &local_start, &local_end, 1, &value,
                   1, child, true, inherited ? &failures : nullptr);
      if (!inherited && std::isfinite(value))
        destination[right] = value;
    } catch (const ops::Error &error) {
      if (!numerical_error(error)) throw;
      if (!scratch.isolate_errors && scope.body->execution_metadata->position_status_nodes) throw;
      destination[right] = nan;
      if (scratch.isolate_errors && program.execution_metadata->position_status_reachable[node_index])
        mark_positions(scratch, node_index, interval_length, right, right + 1, 4);
    }
  }
  return result;
}

static ops::Value execute_apply_scope(const Program &program, const ApplyScope &scope,
    const Node &node, std::size_t node_index, std::size_t interval_length,
    std::size_t max_window, Scratch &scratch) {
  const bool trace = scratch.isolate_errors &&
      program.execution_metadata->position_status_reachable[node_index];
  const auto structure = validate_apply_structure(scope, interval_length, scratch, scratch.isolate_errors);
  auto &child = *scratch.children[program.rolling_scopes.size() + node.input_index];
  child.remaining_work = scratch.remaining_work;
  std::vector<ops::Value> captured;
  captured.reserve(scope.input_nodes.size());
  std::size_t count = 0;
  for (auto id : scope.input_nodes) {
    const auto &value = scratch.values[id];
    if (known_geometry(value)) count = std::max(count, value.size());
    captured.push_back(value);
  }
  if (captured.empty() && scope.kind != ApplyKind::bisect) {
    count = scope.kind == ApplyKind::block ? interval_length
        : scope.kind == ApplyKind::segment
            ? scratch.values[scope.argument_nodes[0]].shape.dim[0]
            : scratch.values[scope.argument_nodes[0]].size();
  }
  if (scope.kind == ApplyKind::filter || scope.kind == ApplyKind::group || scope.kind == ApplyKind::segment)
    count = scratch.values[scope.argument_nodes[0]].shape.dim[0];
  std::vector<double> parameters(scope.parameter_nodes.size());
  std::int16_t parameter_failure = 0;
  for (std::size_t i = 0; i < parameters.size(); ++i)
    if (scope.parameter_nodes[i] != std::numeric_limits<std::uint32_t>::max()) {
      const auto id = scope.parameter_nodes[i];
      parameter_failure = std::max(parameter_failure, node_failure(scratch, id));
      parameters[i] = available_payload(scratch, id) ? scratch.values[id].scalar : NAN;
    }
  auto evaluate = [&](const std::vector<ops::Value> &inputs, std::size_t length,
                      double solve_x = 0.0,
                      const std::vector<std::int16_t> *failures = nullptr) {
    if (child.remaining_work) {
      const auto rows = std::max<std::size_t>(length, 1);
      const auto nodes = std::max<std::size_t>(scope.body_node_count, 1);
      ops::require(rows <= *child.remaining_work / nodes, "SCOPE_COMPUTE_BUDGET_EXCEEDED");
      *child.remaining_work -= rows * nodes;
    }
    for (std::size_t i = 0; i < parameters.size(); ++i)
      if (scope.parameter_nodes[i] == std::numeric_limits<std::uint32_t>::max()) parameters[i] = solve_x;
    const std::int64_t begin = 0, end = static_cast<std::int64_t>(length);
    double value = NAN;
    std::vector<std::int16_t> parameter_only_failures;
    if (parameter_failure && !failures) {
      parameter_only_failures.resize(inputs.size());
      failures = &parameter_only_failures;
    }
    const auto audit = execute_impl(*scope.body, inputs, parameters.data(), parameters.size(),
        &begin, &end, 1, &value, 1, child, true, failures,
        scope.kind != ApplyKind::bisect || std::all_of(inputs.begin(), inputs.end(), known_geometry));
    scratch.algorithm_copy_bytes += audit.algorithm_copy_bytes;
    return value;
  };
  auto scalar_arg = [&](std::size_t i) {
    const auto &value = scratch.values[scope.argument_nodes[i]];
    return value.scalar;
  };
  if (scope.kind == ApplyKind::bisect) {
    double low = scalar_arg(0), high = scalar_arg(1), tolerance = scalar_arg(2);
    const auto iterations = positive_integer(scalar_arg(3), "INVALID_PARAMETER", 10000);
    ops::require(std::isfinite(low) && std::isfinite(high) && low < high &&
                 std::isfinite(tolerance) && tolerance > 0.0, "INVALID_PARAMETER");
    std::int16_t inherited = parameter_failure;
    if (scratch.isolate_errors)
      for (auto id : scope.input_nodes)
        inherited = std::max(inherited, std::max(scratch.statuses[id], position_error(scratch, id)));
    if (inherited) {
      // Bisect consumes each complete capture, which may have a different
      // length. Keep failed payloads unavailable while validating both actual
      // endpoint evaluations; no numerical root iteration can use them.
      std::vector<std::int16_t> failures(captured.size());
      for (std::size_t i = 0; i < captured.size(); ++i) {
        const auto id = scope.input_nodes[i];
        failures[i] = std::max(scratch.statuses[id], position_error(scratch, id));
      }
      evaluate(captured, count, low, &failures);
      evaluate(captured, count, high, &failures);
      scratch.statuses[node_index] = inherited;
      return ops::Value::number(NAN);
    }
    double fl = evaluate(captured, count, low), fh = evaluate(captured, count, high);
    ops::require(std::isfinite(fl) && std::isfinite(fh), "NONFINITE_RESULT");
    if (fl == 0.0) return ops::Value::number(low);
    if (fh == 0.0) return ops::Value::number(high);
    ops::require(std::signbit(fl) != std::signbit(fh), "ROOT_NOT_BRACKETED");
    for (std::size_t i = 0; i < iterations; ++i) {
      const double mid = low * 0.5 + high * 0.5;
      if (high - low <= tolerance || mid == low || mid == high) return ops::Value::number(mid);
      const double fm = evaluate(captured, count, mid);
      ops::require(std::isfinite(fm), "NONFINITE_RESULT");
      if (fm == 0.0) return ops::Value::number(mid);
      if (std::signbit(fm) == std::signbit(fl)) { low = mid; fl = fm; }
      else high = mid;
    }
    if (high - low <= tolerance) return ops::Value::number(low * 0.5 + high * 0.5);
    throw ops::Error("NON_CONVERGENCE");
  }
  auto selected = [&](const std::vector<std::size_t> &indices, std::size_t first, std::size_t last) {
    const auto length = last - first;
    std::vector<ops::Value> inputs;
    inputs.reserve(captured.size());
    const bool contiguous = length == 0 || indices[last - 1] - indices[first] == length - 1;
    child.selected_numeric.resize(captured.size());
    child.selected_integer.resize(captured.size());
    child.selected_masks.resize(captured.size());
    ops::require(captured.size() == 0 || length <= (64ull * 1024ull * 1024ull) /
                 (8ull * captured.size()), "SCOPE_MEMORY_BUDGET_EXCEEDED");
    for (std::size_t i = 0; i < captured.size(); ++i) {
      const auto &source = captured[i];
      if (!known_geometry(source)) {
        inputs.push_back(source);
        continue;
      }
      if (contiguous) {
        inputs.push_back(interval_view(source, length ? indices[first] : 0, length));
        continue;
      }
      ops::Value view = source;
      view.shape = ops::vector_shape(length);
      view.stride[0] = 1;
      if (!available_payload(scratch, scope.input_nodes[i]) &&
          scratch.statuses[scope.input_nodes[i]]) {
        view.data = nullptr;
        inputs.push_back(view);
        continue;
      }
      if (source.kind == ops::Kind::number) {
        auto &buffer = child.selected_numeric[i]; buffer.resize(length);
        for (std::size_t j = 0; j < length; ++j) buffer[j] = source.f(indices[first + j]);
        view.data = buffer.data();
      } else if (source.kind == ops::Kind::integer) {
        auto &buffer = child.selected_integer[i]; buffer.resize(length);
        for (std::size_t j = 0; j < length; ++j) buffer[j] = source.i(indices[first + j]);
        view.data = buffer.data();
      } else {
        ops::require(source.kind == ops::Kind::mask, "SCOPE_CAPTURE_TYPE");
        auto &buffer = child.selected_masks[i]; buffer.resize(length);
        for (std::size_t j = 0; j < length; ++j) buffer[j] = source.u(indices[first + j]);
        view.data = buffer.data();
      }
      scratch.algorithm_copy_bytes += length * (source.kind == ops::Kind::mask ? 1 : 8);
      inputs.push_back(view);
    }
    return inputs;
  };
  const auto selected_error = [&](const std::vector<std::size_t> &indices,
                                   std::size_t first, std::size_t last) {
    std::int16_t status = 0;
    if (!scratch.isolate_errors || std::none_of(scope.input_nodes.begin(), scope.input_nodes.end(),
          [&](auto id) { return !available_payload(scratch, id); })) return status;
    for (auto i = first; i < last; ++i)
      status = std::max(status, capture_error(scratch, scope.input_nodes, indices[i], indices[i] + 1));
    return status;
  };
  const auto slice_failures = [&](std::size_t first, std::size_t last) {
    std::vector<std::int16_t> failures(captured.size());
    for (std::size_t i = 0; i < captured.size(); ++i) {
      const auto id = scope.input_nodes[i];
      failures[i] = scratch.statuses[id];
      if (!failures[i]) failures[i] = position_error(scratch, id, first, last);
    }
    return failures;
  };
  const auto selected_failures = [&](const std::vector<std::size_t> &indices,
                                     std::size_t first, std::size_t last) {
    std::vector<std::int16_t> failures(captured.size());
    for (std::size_t i = 0; i < captured.size(); ++i) {
      const auto id = scope.input_nodes[i];
      failures[i] = scratch.statuses[id];
      if (!failures[i])
        for (auto row = first; row < last; ++row)
          failures[i] = std::max(failures[i], position_error(scratch, id, indices[row], indices[row] + 1));
    }
    return failures;
  };
  if (scope.kind == ApplyKind::filter) {
    const auto &mask = scratch.values[scope.argument_nodes[0]];
    auto &indices = child.selection; indices.clear();
    for (std::size_t i = 0; i < count; ++i) {
      const auto selected = mask.u(i);
      if (selected) indices.push_back(i);
    }
    if (indices.empty()) return ops::Value::number(scope.argument_nodes.size() == 2 ? scalar_arg(1) : NAN);
    if (const auto status = selected_error(indices, 0, indices.size())) {
      const auto failures = selected_failures(indices, 0, indices.size());
      evaluate(selected(indices, 0, indices.size()), indices.size(), 0, &failures);
      scratch.statuses[node_index] = status;
      return ops::Value::number(NAN);
    }
    return ops::Value::number(evaluate(selected(indices, 0, indices.size()), indices.size()));
  }
  ops::Value result;
  result.kind = ops::Kind::number;
  result.stride[0] = 1;
  ops::require(count <= max_window, "SCOPE_OUTPUT_TOO_LARGE");
  auto *destination = scratch.numeric.data() + static_cast<std::size_t>(node.slot) * max_window;
  result.data = destination;
  if (scope.kind == ApplyKind::segment) {
    const auto &segments = scratch.values[scope.argument_nodes[0]];
    result.shape = structure.shape;
    std::fill(destination, destination + count, NAN);
    std::vector<ops::Value> inputs(captured.size());
    // Geometry was validated before any body runs. Endpoints are inclusive in
    // the body, but ownership of output rows is [left,right).
    for (std::size_t left = 0; left < count;) {
      if (segments.i(2 * left) == -1) { ++left; continue; }
      const auto right = static_cast<std::size_t>(segments.i(2 * left + 1));
      const auto length = right - left + 1;
      for (std::size_t i = 0; i < captured.size(); ++i)
        inputs[i] = scope_view(captured[i], left, length);
      double value = NAN;
      auto failure = capture_error(scratch, scope.input_nodes, left, right + 1);
      try {
        if (!failure) value = evaluate(inputs, length);
        else {
          const auto failures = slice_failures(left, right + 1);
          evaluate(inputs, length, 0, &failures);
        }
      } catch (const ops::Error &error) {
        if (!scratch.isolate_errors || !numerical_error(error)) throw;
        failure = 4;
      }
      if (failure) mark_positions(scratch, node_index, count, left, right, failure);
      std::fill(destination + left, destination + right, value);
      left = right;
    }
    return result;
  }
  if (scope.kind == ApplyKind::block) {
    const auto width = positive_integer(scalar_arg(0), "INVALID_PARAMETER", 1000000);
    const auto blocks = count / width;
    result.shape = ops::vector_shape(blocks);
    std::vector<ops::Value> inputs(captured.size());
    for (std::size_t block = 0; block < blocks; ++block) {
      for (std::size_t i = 0; i < captured.size(); ++i)
        inputs[i] = scope_view(captured[i], block * width, width);
      auto failure = capture_error(scratch, scope.input_nodes, block * width, (block + 1) * width);
      destination[block] = NAN;
      try {
        if (!failure) destination[block] = evaluate(inputs, width);
        else {
          const auto failures = slice_failures(block * width, (block + 1) * width);
          evaluate(inputs, width, 0, &failures);
        }
      } catch (const ops::Error &error) {
        if (!trace || !numerical_error(error)) throw;
        failure = 4;
      }
      if (failure) mark_positions(scratch, node_index, blocks, block, block + 1, failure);
    }
    return result;
  }
  const auto &keys = scratch.values[scope.argument_nodes[0]];
  result.shape = ops::vector_shape(count);
  auto &indices = child.selection; indices.resize(count);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
    return keys.i(a) == keys.i(b) ? a < b : keys.i(a) < keys.i(b);
  });
  for (std::size_t first = 0; first < count;) {
    std::size_t last = first + 1;
    while (last < count && keys.i(indices[last]) == keys.i(indices[first])) ++last;
    double value = NAN;
    auto failure = selected_error(indices, first, last);
    try {
      if (!failure) value = evaluate(selected(indices, first, last), last - first);
      else {
        const auto failures = selected_failures(indices, first, last);
        evaluate(selected(indices, first, last), last - first, 0, &failures);
      }
    } catch (const ops::Error &error) {
      if (!scratch.isolate_errors || !numerical_error(error)) throw;
      if (trace) failure = 4;
    }
    for (std::size_t i = first; i < last; ++i) {
      destination[indices[i]] = value;
      if (failure) mark_positions(scratch, node_index, count, indices[i], indices[i] + 1, failure);
    }
    first = last;
  }
  return result;
}

template <bool TrackPositions>
static Audit execute_impl_body(const Program &program,
                          const std::vector<ops::Value> &inputs,
                          const double *parameters,
                          std::size_t parameter_count,
                          const std::int64_t *starts,
                          const std::int64_t *ends,
                          std::size_t rows, void *output,
                          std::size_t output_columns, Scratch &scratch,
                          bool prebound_inputs,
                          const std::vector<std::int16_t> *input_failures,
                          bool interval_known) {
  ops::require(inputs.size() == program.input_count, "GRAPH_INPUT_COUNT");
  ops::require(parameter_count == program.parameter_count,
               "GRAPH_PARAMETER_COUNT");
  ops::require(output_columns == program.roots.size(), "GRAPH_OUTPUT_COLUMNS");
  ops::require(rows == 0 || (starts && ends && output), "GRAPH_NULL_BATCH");
  const auto clear_invalid = [&](std::size_t index) {
    // Integer and bool have no NaN. Their deterministic placeholder is usable
    // only when the matching status is zero; isolate results retain statuses.
    if (program.output_dtype == OutputDType::float64)
      static_cast<double *>(output)[index] = NAN;
    else if (program.output_dtype == OutputDType::int64)
      static_cast<std::int64_t *>(output)[index] = 0;
    else
      static_cast<std::uint8_t *>(output)[index] = 0;
  };

  std::size_t max_window = 0;
  if (prebound_inputs) {
    ops::require(rows == 1, "GRAPH_PREBOUND_ROWS");
    ops::require(starts[0] >= 0 && ends[0] >= starts[0], "GRAPH_INTERVAL_BOUNDS");
    // Constant-body scopes have no captured arrays, but nested operations still
    // materialize arrays on the explicit scope interval (e.g. block_apply(1,1)).
    max_window = static_cast<std::size_t>(ends[0] - starts[0]);
    for (const auto &input : inputs)
      if (known_geometry(input)) max_window = std::max(max_window, input.size());
  } else {
    for (std::size_t row = 0; row < rows; ++row) {
      ops::require(starts[row] >= 0 && ends[row] >= starts[row],
                   "GRAPH_INTERVAL_BOUNDS");
      const auto start = static_cast<std::size_t>(starts[row]);
      const auto end = static_cast<std::size_t>(ends[row]);
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto axis = program.input_axes.empty() ? 0 : program.input_axes[i];
        ops::require(inputs[i].shape.rank >= 1 && inputs[i].shape.rank <= 2, "GRAPH_INPUT_TYPE");
        if (axis != 1) ops::require(end <= inputs[i].shape.dim[0], "GRAPH_INTERVAL_BOUNDS");
      }
      max_window = std::max(max_window, end - start);
    }
  }

  const auto logical_window = max_window;
  max_window = required_array_capacity(program, inputs, max_window);
  scratch.ensure(program, max_window, program.isolate_errors || input_failures != nullptr);
  scratch.interval_known = interval_known;
  if (!prebound_inputs) {
    scratch.work_budget = static_cast<std::size_t>(program.scope_work_budget);
    scratch.remaining_work = &scratch.work_budget;
  }
  scratch.fused_scalar_calls = 0;
  scratch.summary_source_scans = 0;
  scratch.order_stat_sorts = 0;
  scratch.algorithm_copy_bytes = 0;
  const auto &summary_consumers = program.execution_metadata->summary_consumers;
  const auto &order_consumers = program.execution_metadata->order_consumers;
  std::size_t series_base = 0;
  Audit audit;
  if (scratch.isolate_errors) {
    std::size_t output_rows = rows;
    if (program.output_kind == OutputKind::series) {
      output_rows = 0;
      for (std::size_t row = 0; row < rows; ++row) {
        const auto count = static_cast<std::size_t>(ends[row] - starts[row]);
        ops::require(count <= static_cast<std::size_t>(PTRDIFF_MAX) - output_rows,
                     "GRAPH_STATUS_OVERFLOW");
        output_rows += count;
      }
    }
    ops::require(output_columns == 0 || output_rows <=
                     static_cast<std::size_t>(PTRDIFF_MAX) / output_columns,
                 "GRAPH_STATUS_OVERFLOW");
    audit.statuses.resize(output_rows * output_columns);
  }

  for (std::size_t row = 0; row < rows; ++row) {
    scratch.next_generation();
    const auto start = static_cast<std::size_t>(starts[row]);
    const auto end = static_cast<std::size_t>(ends[row]);
    const auto length = end - start;
    if (interval_known && length < program.minimum_observations) {
      ops::require(scratch.isolate_errors, "INSUFFICIENT_OBSERVATIONS");
      const auto begin = program.output_kind == OutputKind::scalar ? row : series_base;
      const auto count = program.output_kind == OutputKind::scalar ? 1 : length;
      for (std::size_t i = 0; i < count * output_columns; ++i) {
        clear_invalid(begin * output_columns + i);
        audit.statuses[begin * output_columns + i] = 1;
      }
      if (program.output_kind == OutputKind::series) series_base += length;
      continue;
    }

    for (std::size_t node_index = 0; node_index < program.nodes.size();
         ++node_index) {
      const auto &node = program.nodes[node_index];
      std::int16_t inherited_positions = 0;
      if (scratch.isolate_errors) {
        if constexpr (TrackPositions) scratch.position_statuses[node_index].clear();
        auto &status = scratch.statuses[node_index];
        status = 0;
        if (!interval_known && (node.kind == NodeKind::apply_scope || node.kind == NodeKind::rolling_scope) &&
            !scope_maps_status(program, node, scratch))
          status = 4;
        for (std::size_t parent = 0; parent < node.parent_count; ++parent) {
          status = std::max(status, scratch.statuses[node.parents[parent]]);
          if constexpr (TrackPositions)
            inherited_positions = std::max(inherited_positions, position_error(scratch, node.parents[parent]));
        }
        if (inherited_positions) {
          const bool mapped = node.kind == NodeKind::interval_tail ||
              (node.kind == NodeKind::operation &&
               (pointwise_status(*program.execution_metadata->specs[node_index]) ||
                mapped_status(static_cast<ops::Op>(node.opcode)))) ||
              scope_maps_status(program, node, scratch);
          // A same-shaped result is not proof of pointwise independence. Shared
          // fits, recurrences, sorting and other nonlocal kernels fail closed.
          if (!mapped) status = std::max(status, inherited_positions);
        }
        if (status && !scope_maps_status(program, node, scratch)) {
          scratch.values[node_index] = validate_failed_node(program, inputs, node, start, length, scratch);
          continue;
        }
      }
      try {
        if (node.kind == NodeKind::input) {
          scratch.values[node_index] =
              (prebound_inputs || (!program.input_axes.empty() && program.input_axes[node.input_index] == 1))
                  ? inputs[node.input_index] : interval_view(inputs[node.input_index], start,
                                              length);
          if (input_failures) scratch.statuses[node_index] = (*input_failures)[node.input_index];
          continue;
        }
        if (node.kind == NodeKind::parameter) {
          scratch.values[node_index] =
              ops::Value::number(parameters[node.input_index]);
          if (scratch.isolate_errors && !std::isfinite(parameters[node.input_index]))
            scratch.statuses[node_index] = 4;
          continue;
        }
        if (node.kind == NodeKind::constant) {
          scratch.values[node_index] = ops::Value::number(node.constant);
          continue;
        }
        if (node.kind == NodeKind::interval_tail) {
          const auto &source = scratch.values[node.parents[0]];
          const auto skip = source.size() ? 1u : 0u;
          scratch.values[node_index] = interval_view(source, skip, source.size() - skip);
          if (inherited_positions)
            for (std::size_t i = skip; i < source.size(); ++i)
              mark_positions(scratch, node_index, source.size() - skip, i - skip, i - skip + 1,
                             position_error(scratch, node.parents[0], i, i + 1));
          continue;
        }
        if (node.kind == NodeKind::rolling_scope) {
          scratch.values[node_index] = execute_rolling_scope(
              program, inputs, program.rolling_scopes[node.input_index], node,
              node_index, start, length, max_window, scratch);
          continue;
        }

        if (node.kind == NodeKind::apply_scope) {
          scratch.values[node_index] = execute_apply_scope(program,
              program.apply_scopes[node.input_index], node, node_index, length, max_window, scratch);
          // Explicit empty/NaN filter results are missing data, not execution
          // failures. Preserve them for ordinary where selection; public roots
          // still report non-finite results as unavailable.
          if (scratch.isolate_errors && program.apply_scopes[node.input_index].kind != ApplyKind::filter &&
              scratch.values[node_index].shape.rank == 0 &&
              !std::isfinite(scratch.values[node_index].scalar)) scratch.statuses[node_index] = 4;
          continue;
        }

        std::array<ops::Value, 8> arguments{};
        for (std::size_t parent = 0; parent < node.parent_count; ++parent)
          arguments[parent] = scratch.values[node.parents[parent]];

        const auto &spec = *program.execution_metadata->specs[node_index];
        const auto prepared =
            ops::prepare(spec, arguments.data(), node.parent_count);
        if (inherited_positions && pointwise_status(spec) && prepared.output_shape.rank != 1) {
          scratch.statuses[node_index] = inherited_positions;
          scratch.values[node_index] = validate_failed_node(program, inputs, node, start, length, scratch);
          continue;
        }
        if (inherited_positions && mapped_status(spec.op))
          map_position_status(scratch, node_index, node, prepared);

        if (prepared.borrowed) {
          scratch.values[node_index] = prepared.view;
          continue;
        }

        ops::Output native_output;
        native_output.kind = prepared.output_kind;
        native_output.shape = prepared.output_shape;

        if (prepared.output_shape.rank >= 1) {
          ops::require(prepared.output_shape.size() <= max_window,
                       "GRAPH_OUTPUT_TOO_LARGE");
          if (prepared.output_kind == ops::Kind::mask) {
            ops::require(node.storage == StorageKind::mask, "GRAPH_STORAGE_KIND");
            native_output.data = scratch.masks.data() +
                                 static_cast<std::size_t>(node.slot) * max_window;
          } else if (prepared.output_kind == ops::Kind::integer) {
            ops::require(node.storage == StorageKind::integer, "GRAPH_STORAGE_KIND");
            native_output.data = scratch.integers.data() + static_cast<std::size_t>(node.slot) * max_window;
          } else {
            ops::require(prepared.output_kind == ops::Kind::number,
                         "GRAPH_STORAGE_KIND");
            ops::require(node.storage == StorageKind::numeric,
                         "GRAPH_STORAGE_KIND");
            native_output.data = scratch.numeric.data() +
                                 static_cast<std::size_t>(node.slot) * max_window;
          }
        } else {
          ops::require(node.storage == StorageKind::inline_value,
                       "GRAPH_STORAGE_KIND");
        }

        bool fused = false;
        if (prepared.output_shape.rank == 0 && node.parent_count &&
            prepared.args[0].kind == ops::Kind::number &&
            prepared.args[0].shape.rank == 1) {
          const auto source = static_cast<std::size_t>(node.parents[0]);
          if (summary_fusion_eligible(spec.op) && summary_consumers[source] > 1) {
            native_output.set(0, summary_result(scratch, node, prepared));
            fused = true;
          } else if (order_fusion_eligible(spec.op) &&
                     order_consumers[source] > 1) {
            native_output.set(0, ordered_result(scratch, node, prepared));
            fused = true;
          }
          if (fused)
            ++scratch.fused_scalar_calls;
        }

        if (inherited_positions && pointwise_status(spec)) {
          execute_pointwise_with_status(node, node_index, prepared, native_output, scratch);
        } else if (!fused) {
          ops::Audit operator_audit;
          ops::execute(prepared, native_output, scratch.workspace,
                       ops::Isa::automatic, operator_audit);
          scratch.algorithm_copy_bytes += operator_audit.algorithm_copy_bytes;
        }
        scratch.values[node_index] =
            output_value(scratch, prepared, native_output, node, max_window);
        if (scratch.isolate_errors) {
          const auto &value = scratch.values[node_index];
          auto &status = scratch.statuses[node_index];
          if (value.kind == ops::Kind::fit)
            status = std::isfinite(value.record[0]) && value.record[4] >= 2 ? 0 : 4;
          else if (value.kind == ops::Kind::interval)
            status = value.record[3] >= 0 ? 0 : 4;
          else if (value.kind != ops::Kind::integer && value.shape.rank == 0 && !std::isfinite(value.scalar)) {
            const auto op = spec.op;
            status = op == ops::Op::value_at || op == ops::Op::days_between ? 7 : 4;
            if (op == ops::Op::interval_start || op == ops::Op::interval_trough ||
                op == ops::Op::interval_recovery)
              status = arguments[0].record[3] == 0 ? 9 : 10;
          }
        }
      } catch (const ops::Error &error) {
        if (!scratch.isolate_errors)
          throw;
        const std::string_view code(error.what());
        // Batch numeric exceptions use the platform's unavailable-result status.
        // Structural shape/bounds, allocation and worker failures still escape.
        std::int16_t status = 0;
        if (code == "DIVIDE_BY_ZERO" || code == "DOMAIN_ERROR") {
          status = 4;
          if (node.kind == NodeKind::operation) {
            const auto op = static_cast<ops::Op>(node.opcode);
            const auto scalar_parent = [&](std::size_t i) {
              return i < node.parent_count &&
                     scratch.values[node.parents[i]].shape.rank == 0;
            };
            // Match the platform's guarded scalar diagnostics. Array and other
            // operator exceptions use its generic unavailable-result status.
            if (code == "DIVIDE_BY_ZERO" &&
                ((op == ops::Op::divide && scalar_parent(1)) ||
                 (op == ops::Op::reciprocal && scalar_parent(0))))
              status = 2;
            if (code == "DOMAIN_ERROR" && scalar_parent(0) &&
                (op == ops::Op::sqrt || op == ops::Op::log ||
                 op == ops::Op::require_positive || op == ops::Op::require_nonnegative))
              status = 3;
          }
        } else if (code == "NONFINITE_RESULT" || code == "NON_FINITE_RESULT") status = 4;
        else if (code == "INSUFFICIENT_SAMPLE" || code == "INVALID_PARAMETER" ||
                 code == "SINGULAR_MATRIX" || code == "ROOT_NOT_BRACKETED" ||
                 code == "NON_CONVERGENCE") status = 4;
        else throw;
        scratch.statuses[node_index] = status;
        scratch.values[node_index] = validate_failed_node(program, inputs, node, start, length, scratch);
      }
    }

    for (std::size_t root_index = 0; root_index < program.roots.size();
         ++root_index) {
      const auto &value = scratch.values[program.roots[root_index]];
      const auto status = scratch.isolate_errors
                              ? scratch.statuses[program.roots[root_index]] : 0;
      if (status) {
        const auto begin = program.output_kind == OutputKind::scalar ? row : series_base;
        const auto count = program.output_kind == OutputKind::scalar ? 1 : length;
        for (std::size_t offset = 0; offset < count; ++offset) {
          const auto index = (begin + offset) * output_columns + root_index;
          clear_invalid(index);
          audit.statuses[index] = static_cast<std::int16_t>(status);
        }
        continue;
      }
      if (program.output_kind == OutputKind::scalar) {
        ops::require(value.shape.rank == 0, "GRAPH_ROOT_NOT_SCALAR");
        ops::require(value.kind == ops::Kind::number ||
                         value.kind == ops::Kind::mask || value.kind == ops::Kind::integer,
                     "GRAPH_ROOT_NOT_SCALAR");
        if (value.kind == ops::Kind::integer)
          ops::require(value.integer >= -9007199254740992LL && value.integer <= 9007199254740992LL,
                       "GRAPH_INTEGER_RESULT_PRECISION");
        auto *numeric_output = static_cast<double *>(output);
        numeric_output[row * output_columns + root_index] = value.kind == ops::Kind::integer ?
            static_cast<double>(value.integer) : value.scalar;
        if (scratch.isolate_errors && !std::isfinite(numeric_output[row * output_columns + root_index]))
          audit.statuses[row * output_columns + root_index] = 4;
      } else {
        const auto expected_kind = program.output_dtype == OutputDType::boolean ? ops::Kind::mask
            : program.output_dtype == OutputDType::int64 ? ops::Kind::integer : ops::Kind::number;
        ops::require(value.kind == expected_kind && value.shape.rank == 1 &&
                         value.size() == length,
                     "GRAPH_ROOT_NOT_ALIGNED_SERIES");
        for (std::size_t offset = 0; offset < length; ++offset) {
          const auto index = (series_base + offset) * output_columns + root_index;
          const auto position = TrackPositions
              ? position_error(scratch, program.roots[root_index], offset, offset + 1) : 0;
          if (position) {
            clear_invalid(index);
            audit.statuses[index] = position;
            continue;
          }
          if (program.output_dtype == OutputDType::int64)
            static_cast<std::int64_t *>(output)[index] = value.i(offset);
          else if (program.output_dtype == OutputDType::boolean) {
            const auto mask = value.u(offset);
            ops::require(mask <= 1, "INVALID_MASK");
            static_cast<std::uint8_t *>(output)[index] = mask;
          } else {
            const auto numeric = value.f(offset);
            static_cast<double *>(output)[index] = numeric;
            if (scratch.isolate_errors && !std::isfinite(numeric))
              audit.statuses[index] = 4;
          }
        }
      }
    }
    if (program.output_kind == OutputKind::series)
      series_base += length;
  }

  audit.rows = rows;
  audit.nodes = program.nodes.size();
  audit.max_window = logical_window;
  audit.numeric_arena_bytes = scratch.numeric.capacity() * sizeof(double) + scratch.integers.capacity() * sizeof(std::int64_t);
  audit.mask_arena_bytes = scratch.masks.capacity() * sizeof(std::uint8_t);
  audit.operator_workspace_capacity_bytes = scratch.workspace.capacity_bytes();
  const auto position_bytes = [](const Scratch &state) {
    auto bytes = state.position_statuses.capacity() * sizeof(std::vector<std::int16_t>);
    for (const auto &errors : state.position_statuses) bytes += errors.capacity() * sizeof(std::int16_t);
    return bytes;
  };
  if constexpr (TrackPositions) audit.operator_workspace_capacity_bytes += position_bytes(scratch);
  const auto scratch_bytes = [&](const auto &self, const Scratch &state) -> std::size_t {
    std::size_t bytes = state.numeric.capacity() * 8 + state.integers.capacity() * 8 + state.masks.capacity();
    bytes += state.statuses.capacity() * sizeof(std::int16_t);
    bytes += position_bytes(state);
    bytes += state.workspace.capacity_bytes() + state.selection.capacity() * sizeof(std::size_t);
    for (const auto &v : state.selected_numeric) bytes += v.capacity() * 8;
    for (const auto &v : state.selected_integer) bytes += v.capacity() * 8;
    for (const auto &v : state.selected_masks) bytes += v.capacity();
    for (const auto &nested : state.children) if (nested) bytes += self(self, *nested);
    return bytes;
  };
  for (const auto &child : scratch.children)
    if (child) audit.operator_workspace_capacity_bytes += scratch_bytes(scratch_bytes, *child);

  audit.input_copy_bytes = 0;
  audit.fused_scalar_calls = scratch.fused_scalar_calls;
  audit.summary_source_scans = scratch.summary_source_scans;
  audit.order_stat_sorts = scratch.order_stat_sorts;
  audit.algorithm_copy_bytes = scratch.algorithm_copy_bytes;
  for (const auto &cache : scratch.ordered)
    audit.order_scratch_capacity_bytes +=
        cache.values.capacity() * sizeof(double);
  return audit;
}

static Audit execute_impl(const Program &program, const std::vector<ops::Value> &inputs,
    const double *parameters, std::size_t parameter_count, const std::int64_t *starts,
    const std::int64_t *ends, std::size_t rows, void *output, std::size_t output_columns,
    Scratch &scratch, bool prebound_inputs,
    const std::vector<std::int16_t> *input_failures, bool interval_known) {
  if (!program.execution_metadata) {
    Program finalized = program;
    finalized.finalize();
    return execute_impl(finalized, inputs, parameters, parameter_count, starts,
                        ends, rows, output, output_columns, scratch, prebound_inputs, input_failures, interval_known);
  }
  // Choose once per native execution. AOT specialization removes optional
  // per-node provenance branches from graphs that cannot produce segment errors.
  ops::require(!input_failures || input_failures->size() == program.input_count, "GRAPH_INPUT_STATUS_COUNT");
  if ((program.isolate_errors || input_failures) && program.execution_metadata->position_status_nodes)
    return execute_impl_body<true>(program, inputs, parameters, parameter_count,
        starts, ends, rows, output, output_columns, scratch, prebound_inputs, input_failures, interval_known);
  return execute_impl_body<false>(program, inputs, parameters, parameter_count,
      starts, ends, rows, output, output_columns, scratch, prebound_inputs, input_failures, interval_known);
}

Audit execute(const Program &program, const std::vector<ops::Value> &inputs,
              const double *parameters, std::size_t parameter_count,
              const std::int64_t *starts, const std::int64_t *ends,
              std::size_t rows, void *output, std::size_t output_columns) {
  return execute_impl(program, inputs, parameters, parameter_count, starts, ends,
                      rows, output, output_columns, root_scratch, false);
}

} // namespace calmetrics_engine::graph
