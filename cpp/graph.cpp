#include "calmetrics_engine/graph.hpp"
#include "calmetrics_engine/affine_projection.hpp"
#include "calmetrics_engine/compact_path.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include "calmetrics_engine/pointwise.hpp"
#include "calmetrics_engine/native_runtime.hpp"
#include "operators/axis_primitives.hpp"
#include "operators/elementwise_math.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <cfenv>
#include <limits>
#include <numeric>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace calmetrics_engine::graph {
namespace {

thread_local const ValueObserver *value_observer = nullptr;

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
  std::shared_ptr<const ExecutionMetadata> arena_owner;
  std::shared_ptr<const ExecutionMetadata> projection_owner;
  std::size_t arena_window = 0;
  native::Deadline deadline = native::Deadline::max();
  const std::atomic<bool> *cancelled = nullptr;
  std::vector<std::array<double, 3>> iteration_info;
  std::vector<double> iteration_state, iteration_next;
  std::vector<ops::Value> iteration_inputs;
  std::vector<double> iteration_parameters;
  ResultLayout iteration_layout;
  std::vector<double> numeric;
  std::vector<double> matmul_statistics_cache;
  std::vector<std::size_t> matmul_statistics_offsets;
  std::vector<std::uint8_t> masks;
  std::vector<std::int64_t> integers;
  ArrayArenaLayout arena;
  std::vector<std::vector<double>> selected_numeric;
  std::vector<std::vector<std::int64_t>> selected_integer;
  std::vector<std::vector<std::uint8_t>> selected_masks;
  std::vector<std::size_t> selection;
  std::size_t work_budget = 100000000;
  std::size_t *remaining_work = nullptr;
  std::vector<ops::Value> values;
  bool isolate_errors = false;
  bool allow_workspace_loan = false;
  bool expanded_matmul_roots = true;
  bool allow_row_subviews = true;
  bool propagate_scope_errors = false;
  bool interval_known = true;
  std::vector<std::int16_t> statuses;
  std::vector<std::vector<std::int16_t>> position_statuses;
  std::shared_ptr<const ExecutionMetadata> position_owner;
  std::size_t position_window = 0;
  std::vector<SummaryCache> summaries;
  std::vector<OrderedCache> ordered;
  ops::Workspace workspace;
  std::size_t matrix_workspace_limit = 0;
  std::size_t generation = 0;
  std::size_t fused_scalar_calls = 0;
  std::size_t summary_source_scans = 0;
  std::size_t order_stat_sorts = 0;
  std::size_t highway_vqsort_calls = 0;
  std::size_t algorithm_copy_bytes = 0;
  std::size_t eigen_rotation_calls = 0;
  std::size_t eigen_matvec_calls = 0;
  std::size_t eigen_covariance_calls = 0;
  std::size_t eigen_density_calls = 0;
  std::size_t gaussian_filter_streams = 0;
  std::size_t accelerate_covchol_calls = 0;
  std::size_t accelerate_matmul_calls = 0, matmul_zero_calls = 0;
  std::size_t accelerate_covariance_attempts = 0;
  std::size_t accelerate_covariance_calls = 0;
  std::size_t covariance_syrk_calls = 0;
  std::size_t covariance_replayed_cells = 0;
  std::size_t covariance_fallbacks = 0;
  std::size_t covariance_zero_calls = 0;
  std::size_t accelerate_vforce_calls = 0;
  std::size_t accelerate_lu_attempts = 0;
  std::size_t accelerate_lu_calls = 0;
  std::size_t accelerate_lu_fallbacks = 0;
  std::size_t solve_residual_materialization_bytes = 0;
  std::size_t covariance_panel_materialization_bytes = 0;
  std::size_t native_library_workspace_reserve_bytes = 0;
  std::size_t state_copy_bytes = 0, child_root_copy_bytes = 0, child_direct_output_bytes = 0;
  std::size_t child_vector_elements = 0, child_pointwise_tiles = 0, child_fused_pointwise_calls = 0;
  std::size_t child_pointwise_capacity = 0;
  std::vector<std::size_t> rolling_invalid;
  std::vector<std::unique_ptr<Scratch>> children;
  std::unique_ptr<Scratch> compact_child;
  std::size_t compact_child_limit = 0;
  std::size_t failed_node = std::numeric_limits<std::size_t>::max();

  std::size_t slot_capacity(StorageKind kind, std::size_t slot) const {
    const auto k=static_cast<std::size_t>(kind)-1;
    return (slot+1<arena.offsets[k].size() ? arena.offsets[k][slot+1] : arena.elements[k])-arena.offsets[k][slot];
  }
  double *numeric_slot(std::size_t slot) {
    return numeric.empty() ? nullptr : numeric.data() + arena.offsets[0][slot];
  }

  void release_execution_buffers() {
    const auto clear = [](auto &v) { std::decay_t<decltype(v)>().swap(v); };
    clear(matmul_statistics_cache); clear(matmul_statistics_offsets);
    clear(numeric); clear(masks); clear(integers); clear(values); clear(statuses);
    clear(position_statuses); clear(summaries); clear(ordered); clear(rolling_invalid);
    arena = {};
    clear(workspace.doubles); clear(workspace.indices); clear(iteration_info);
    clear(iteration_state); clear(iteration_next); clear(selected_numeric);
    clear(iteration_inputs); clear(iteration_parameters); iteration_layout = {};
    clear(selected_integer); clear(selected_masks); clear(selection); children.clear();
    compact_child.reset(); compact_child_limit = 0;
    arena_owner.reset(); projection_owner.reset(); position_owner.reset(); arena_window = position_window = 0;
  }

  void ensure(const Program &program, std::size_t max_window, bool isolate,
              const std::vector<ops::Shape> &shapes, bool fuse, bool summary_fusion,
              bool private_where_sums, bool allow_matmul_chain = true,
              const std::vector<std::uint8_t> *row_subviews = nullptr,
              std::int32_t loaned_state_source = -1) {
    if (arena_owner.get() != program.execution_metadata.get() || max_window < arena_window) {
      // A previous graph or larger batch must not retain workspace outside the
      // current plan's admission estimate. Parent-owned capture/state buffers
      // below remain alive: this execution may currently be reading them.
      std::vector<double>().swap(numeric);
      std::vector<std::uint8_t>().swap(masks);
      std::vector<std::int64_t>().swap(integers);
      std::vector<ops::Value>().swap(values);
      std::vector<SummaryCache>().swap(summaries);
      std::vector<OrderedCache>().swap(ordered);
      std::vector<double>().swap(workspace.doubles);
      std::vector<std::size_t>().swap(workspace.indices);
      children.clear();
      arena = {};
      arena_owner = program.execution_metadata;
    }
    const auto panel_limit = matrix_workspace_doubles(program, shapes, max_window, expanded_matmul_roots);
    if (panel_limit != matrix_workspace_limit) {
      // The largest array can stay unchanged while a matrix dimension (and
      // hence panel memory) changes. Arena capacity cannot bound retained panels.
      std::vector<double>().swap(workspace.doubles);
      matrix_workspace_limit = panel_limit;
    }
    arena_window = max_window;
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
    const auto resize_arena = [](auto &buffer, std::size_t count) {
      if (count != buffer.capacity()) std::decay_t<decltype(buffer)>().swap(buffer);
      buffer.resize(count);
    };
    const auto cache_size = summary_fusion
        ? matmul_summary_cache_doubles(program, shapes, &matmul_statistics_offsets) : 0;
    if (!summary_fusion) std::vector<std::size_t>().swap(matmul_statistics_offsets);
    resize_arena(matmul_statistics_cache, cache_size);
    plan_array_arena(program, max_window, shapes, arena, fuse || summary_fusion,
        false, summary_fusion, private_where_sums, allow_matmul_chain, row_subviews,
        loaned_state_source);
    resize_arena(integers, arena.elements[2]);
    resize_arena(numeric, arena.elements[0]);
    resize_arena(masks, arena.elements[1]);
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
    if (iteration_info.capacity() > program.apply_scopes.size())
      std::vector<std::array<double, 3>>(program.apply_scopes.size()).swap(iteration_info);
    else iteration_info.resize(program.apply_scopes.size());
    children.resize(scope_count);
    for (std::size_t i = 0; i < scope_count; ++i) {
      if (!children[i])
        children[i] = std::make_unique<Scratch>();
      const bool iterate = i >= program.rolling_scopes.size() &&
          program.apply_scopes[i - program.rolling_scopes.size()].kind == ApplyKind::iterate;
      const auto bound = iterate ? std::max<std::size_t>(max_window, 1) : 0;
      for (auto *buffer : {&children[i]->iteration_state, &children[i]->iteration_next})
        if (buffer->capacity() > bound) std::vector<double>().swap(*buffer);
    }
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
                base.kind == ops::Kind::mask) && base.shape.rank >= 1 && base.shape.rank <= 3,
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

ops::Value output_value(const ops::Prepared &prepared,
                        const ops::Output &output,
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

  ops::require(prepared.output_shape.rank <= 3, "GRAPH_OUTPUT_RANK");
  ops::require(prepared.output_shape.size() <= max_window,
               "GRAPH_OUTPUT_TOO_LARGE");
  value.set_contiguous_strides(output.order);

  value.data = output.data;
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
  ops::require(static_cast<unsigned>(output_kind) <= 2 &&
                   static_cast<unsigned>(output_dtype) <= 2,
               "GRAPH_OUTPUT_TYPE");
  ops::require(output_kind != OutputKind::scalar || output_dtype == OutputDType::float64,
               "GRAPH_SCALAR_OUTPUT_DTYPE");
  ops::require(root_outputs.size() == (output_kind == OutputKind::typed ? roots.size() : 0), "RESULT_SCHEMA");
  for (const auto &root : root_outputs)
    ops::require(static_cast<unsigned>(root.dtype) <= 2 && root.rank <= 3 &&
                 static_cast<unsigned>(root.order) <= 2, "RESULT_SCHEMA");
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
      ops::require(scope.body && scope.body->output_kind ==
                   (scope.kind == ApplyKind::iterate ? OutputKind::typed : OutputKind::scalar) &&
                   scope.body->roots.size() == 1 &&
                   scope.input_nodes.size() == scope.body->input_count &&
                   scope.parameter_nodes.size() == scope.body->parameter_count,
                   "GRAPH_APPLY_BINDINGS");
      scope.body->validate();
      const auto argc = scope.argument_nodes.size();
      ops::require((scope.kind == ApplyKind::bisect && argc == 4) ||
                   (scope.kind == ApplyKind::iterate && argc == 3) ||
                   (scope.kind == ApplyKind::filter && (argc == 1 || argc == 2)) ||
                   ((scope.kind == ApplyKind::block || scope.kind == ApplyKind::group ||
                     scope.kind == ApplyKind::segment) && argc == 1),
                   "GRAPH_APPLY_ARGUMENTS");
      ops::require(scope.state_input_index >= -1 && (scope.state_input_index == -1 ||
          (scope.kind == ApplyKind::iterate && static_cast<std::size_t>(scope.state_input_index) < scope.input_nodes.size() &&
           scope.input_nodes[scope.state_input_index] == scope.argument_nodes[0])), "GRAPH_ITERATION_STATE_BINDING");
      const bool array = scope.kind == ApplyKind::block || scope.kind == ApplyKind::group ||
                         scope.kind == ApplyKind::segment ||
                         (scope.kind == ApplyKind::iterate && scope.body->root_outputs[0].rank != 0);
      ops::require(array ? node.storage == StorageKind::numeric && node.slot < numeric_slots
                         : node.storage == StorageKind::inline_value, "GRAPH_SLOT");
      auto dependency = [&](std::uint32_t id) {
        ops::require(id < index && node.parents.contains(id, node.parent_count),
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
    case NodeKind::iteration_projection:
      ops::require(node.parent_count == 1 && node.parents[0] < index && node.opcode <= 2 &&
          node.storage == StorageKind::inline_value && nodes[node.parents[0]].kind == NodeKind::apply_scope &&
          apply_scopes[nodes[node.parents[0]].input_index].kind == ApplyKind::iterate, "GRAPH_ITERATION_PROJECTION");
      break;
    default:
      throw ops::Error("GRAPH_NODE_KIND");
    }
  }

  for (auto root : roots)
    ops::require(root < nodes.size(), "GRAPH_ROOT_INDEX");
}

bool independent_transcendental_root(const Program &program, std::size_t index) {
  if (program.output_kind != OutputKind::typed || program.isolate_errors ||
      program.roots.size() != 1 || program.roots[0] != index ||
      !program.apply_scopes.empty() || !program.rolling_scopes.empty() ||
      program.root_outputs[0].rank < 1 || program.root_outputs[0].rank > 3 ||
      program.root_outputs[0].dtype != OutputDType::float64) return false;
  const auto &node = program.nodes[index];
  if (node.kind != NodeKind::operation || node.parent_count != 1 ||
      (node.opcode != static_cast<std::uint16_t>(ops::Op::exp) &&
       node.opcode != static_cast<std::uint16_t>(ops::Op::log))) return false;
  for (const auto &consumer : program.nodes)
    for (std::size_t j = 0; j < consumer.parent_count; ++j)
      if (consumer.parents[j] == index) return false;
  return true;
}

static bool independent_matmul_root(const Program &program, std::size_t index);
static bool independent_covariance_root(const Program &program, std::size_t index) {
  if (program.output_kind != OutputKind::typed || program.isolate_errors ||
      program.roots.empty() || program.roots.size() > 2 ||
      !program.apply_scopes.empty() || !program.rolling_scopes.empty() ||
      std::find(program.roots.begin(), program.roots.end(), index) == program.roots.end())
    return false;
  for (std::size_t r = 0; r < program.roots.size(); ++r) {
    const auto root = program.roots[r];
    const auto &node = program.nodes[root];
    if (program.root_outputs[r].rank != 2 || program.root_outputs[r].dtype != OutputDType::float64 ||
        node.kind != NodeKind::operation || node.parent_count != 1 ||
        (node.opcode != static_cast<std::uint16_t>(ops::Op::covariance) &&
         node.opcode != static_cast<std::uint16_t>(ops::Op::correlation))) return false;
    for (const auto &consumer : program.nodes)
      for (std::size_t j = 0; j < consumer.parent_count; ++j)
        if (consumer.parents[j] == root) return false;
  }
  if (program.roots.size() == 2) {
    const auto &cov = program.nodes[program.roots[0]], &corr = program.nodes[program.roots[1]];
    if (program.roots[0] >= program.roots[1] ||
        cov.opcode != static_cast<std::uint16_t>(ops::Op::covariance) ||
        corr.opcode != static_cast<std::uint16_t>(ops::Op::correlation) ||
        cov.parents[0] != corr.parents[0]) return false;
  }
  return true;
}

static bool independent_cholesky_f_root(const Program &program, std::size_t index,
                                        const ExecutionMetadata *metadata = nullptr) {
  if (program.output_kind != OutputKind::typed || program.isolate_errors ||
      program.roots.size() != 1 || program.roots[0] != index ||
      !program.apply_scopes.empty() || !program.rolling_scopes.empty() ||
      program.root_outputs[0].rank != 2 || program.root_outputs[0].dtype != OutputDType::float64 ||
      program.root_outputs[0].order != OutputOrder::f) return false;
  const auto &node = program.nodes[index];
  if (node.kind != NodeKind::operation || node.parent_count != 1 ||
      node.opcode != static_cast<std::uint16_t>(ops::Op::cholesky)) return false;
  // Only the already-proven private pair can share this F result allocation.
  // Finalization passes its new metadata, before Program publishes it.
  const auto &parent = program.nodes[node.parents[0]];
  if (parent.kind == NodeKind::operation &&
      parent.opcode == static_cast<std::uint16_t>(ops::Op::covariance)) {
    const auto *active = metadata ? metadata : program.execution_metadata.get();
    if (!active || index >= active->fused_from.size() ||
        active->fused_from[index] != static_cast<std::int32_t>(node.parents[0])) return false;
  }
  for (const auto &consumer : program.nodes)
    for (std::size_t j = 0; j < consumer.parent_count; ++j)
      if (consumer.parents[j] == index) return false;
  return true;
}

static bool cholesky_f_root_requires_arena(const Program &program,
    std::size_t index, bool canonical_route) {
  return canonical_route && program.execution_metadata->root_destinations[index] >= 0 &&
      independent_cholesky_f_root(program, index);
}

static std::shared_ptr<RowBatchPlan> plan_row_batch(const Program &program,
    const std::shared_ptr<ExecutionMetadata> &metadata) {
  if (program.output_kind != OutputKind::typed || program.isolate_errors ||
      program.roots.empty() || !program.apply_scopes.empty() ||
      !program.rolling_scopes.empty() || metadata->pointwise ||
      !metadata->matmul_summaries.empty() || metadata->affine_projection ||
      metadata->compact_path) return nullptr;
  std::uint32_t producer = UINT32_MAX;
  for (std::uint32_t id = 0; id < program.nodes.size(); ++id) {
    const auto &node = program.nodes[id];
    if (node.kind != NodeKind::operation || node.opcode != std::uint16_t(ops::Op::matmul)) continue;
    if (producer != UINT32_MAX || node.parent_count != 2 ||
        metadata->root_destinations[id] >= 0 ||
        std::find(program.roots.begin(), program.roots.end(), id) != program.roots.end())
      return nullptr;
    producer = id;
  }
  if (producer == UINT32_MAX) return nullptr;
  // A row-local left operand can be derived from one static input. Keep
  // its actual logical nodes in every child; do not materialize the full
  // transformed weights or trajectory before partitioning. The axis proof
  // below rejects cross-row reductions, and runtime shape validation proves
  // the same dependency closure for the full and blocked input geometry.
  std::uint32_t source_input = UINT32_MAX;
  std::vector<std::uint8_t> visited(program.nodes.size(), 0);
  std::vector<std::uint32_t> pending{program.nodes[producer].parents[0]};
  while (!pending.empty()) {
    const auto id = pending.back(); pending.pop_back();
    if (visited[id]) continue;
    visited[id] = 1;
    const auto &node = program.nodes[id];
    if (node.kind == NodeKind::input) {
      if (node.input_index >= program.input_axes.size() ||
          program.input_axes[node.input_index] != 1 ||
          (source_input != UINT32_MAX && source_input != node.input_index)) return nullptr;
      source_input = node.input_index;
    } else if (node.kind == NodeKind::operation) {
      for (std::size_t j = 0; j < node.parent_count; ++j)
        pending.push_back(node.parents[j]);
    } else if (node.kind != NodeKind::constant && node.kind != NodeKind::parameter) return nullptr;
  }
  if (source_input == UINT32_MAX) return nullptr;
  auto plan = std::make_shared<RowBatchPlan>();
  plan->input = source_input; plan->producer = producer;
  plan->axes.assign(program.nodes.size(), -1);
  const auto axis_literal = [&](const Node &node, std::size_t argument) {
    if (argument >= node.parent_count) return -1;
    const auto &axis = program.nodes[node.parents[argument]];
    if (axis.kind != NodeKind::constant || !std::isfinite(axis.constant) ||
        axis.constant != std::floor(axis.constant) || axis.constant < 0 || axis.constant > 2) return -1;
    return int(axis.constant);
  };
  for (std::size_t id = 0; id < program.nodes.size(); ++id) {
    const auto &node = program.nodes[id];
    if (node.kind == NodeKind::input) {
      if (node.input_index == plan->input) plan->axes[id] = 0;
      continue;
    }
    if (node.kind == NodeKind::constant || node.kind == NodeKind::parameter) continue;
    if (node.kind != NodeKind::operation) return nullptr;
    const auto op = static_cast<ops::Op>(node.opcode);
    const auto first = node.parent_count ? plan->axes[node.parents[0]] : -1;
    if (id == producer) {
      if (first != 0 || plan->axes[node.parents[1]] >= 0) return nullptr;
      plan->axes[id] = 0;
    } else if (op == ops::Op::transpose) {
      if (node.parent_count != 1 || first > 1) return nullptr;
      plan->axes[id] = first < 0 ? -1 : 1 - first;
    } else if (op == ops::Op::broadcast_like) {
      if (node.parent_count != 2 || plan->axes[node.parents[1]] < 0) return nullptr;
      plan->axes[id] = plan->axes[node.parents[1]];
    } else if (op == ops::Op::sum_axis || op == ops::Op::mean_axis ||
               op == ops::Op::variance_axis || op == ops::Op::max_axis) {
      const auto axis = axis_literal(node, 1);
      if (axis < 0 || first < 0 || axis == first) return nullptr;
      for (std::size_t j = 1; j < node.parent_count; ++j)
        if (plan->axes[node.parents[j]] >= 0) return nullptr;
      plan->axes[id] = first - (axis < first);
    } else if (ops::axis_path_operator(op)) {
      const auto axis = axis_literal(node, op == ops::Op::quantiles_axis ? 2 : 1);
      if (axis < 0 || first < 0 || axis == first) return nullptr;
      for (std::size_t j = 1; j < node.parent_count; ++j)
        if (plan->axes[node.parents[j]] >= 0) return nullptr;
      plan->axes[id] = first;
    } else if (metadata->specs[id]->family == ops::Family::elementwise) {
      int independent = -1;
      for (std::size_t j = 0; j < node.parent_count; ++j) {
        const auto axis = plan->axes[node.parents[j]];
        if (axis < 0) continue;
        if (independent >= 0 && independent != axis) return nullptr;
        independent = axis;
      }
      plan->axes[id] = independent;
    } else return nullptr;
  }
  for (std::size_t root = 0; root < program.roots.size(); ++root)
    if (plan->axes[program.roots[root]] != 0 || program.root_outputs[root].rank < 1 ||
        program.root_outputs[root].rank > 2 ||
        program.root_outputs[root].dtype != OutputDType::float64) return nullptr;
  // Child execution explicitly disables physical reordering and uses the same
  // canonical logical nodes. No parent/child ownership cycle is introduced.
  auto child = std::make_shared<ExecutionMetadata>(*metadata);
  std::vector<std::size_t> uses(program.nodes.size());
  for (const auto &node : program.nodes)
    for (std::size_t j = 0; j < node.parent_count; ++j) ++uses[node.parents[j]];
  for (std::uint32_t id = 0; id < program.nodes.size(); ++id) {
    const auto &node = program.nodes[id];
    if (node.kind != NodeKind::operation || node.parent_count != 2 ||
        node.opcode != std::uint16_t(ops::Op::sum_axis) || axis_literal(node, 1) != 1) continue;
    const auto source = node.parents[0];
    const auto &first = program.nodes[source];
    if (first.kind != NodeKind::operation || first.parent_count != 3 ||
        first.opcode != std::uint16_t(ops::Op::where) || uses[source] != 1 ||
        metadata->root_destinations[source] >= 0 || first.storage != StorageKind::numeric) continue;
    if (child->where_sum_to.empty()) {
      child->where_sum_to.assign(program.nodes.size(), -1);
      child->where_sum_from.assign(program.nodes.size(), -1);
    }
    child->where_sum_to[source] = std::int32_t(id);
    child->where_sum_from[id] = std::int32_t(source);
  }
  const auto operation = [&](std::uint32_t id, ops::Op op, std::size_t arity) {
    const auto &node = program.nodes[id];
    return node.kind == NodeKind::operation && node.opcode == std::uint16_t(op) &&
        node.parent_count == arity;
  };
  const auto scalar = [&](std::uint32_t id) {
    const auto kind = program.nodes[id].kind;
    return kind == NodeKind::constant || kind == NodeKind::parameter;
  };
  const auto find_child = [&](std::uint32_t parent, ops::Op op, std::size_t arity) {
    std::uint32_t found = UINT32_MAX;
    for (std::uint32_t id = parent + 1; id < program.nodes.size(); ++id) {
      const auto &node = program.nodes[id];
      if (!operation(id, op, arity) || !node.parents.contains(parent, arity)) continue;
      if (found != UINT32_MAX) return UINT32_MAX;
      found = id;
    }
    return found;
  };
  for (std::uint32_t id = 0; id < program.nodes.size(); ++id) {
    if (!operation(id, ops::Op::cumulative_product_axis, 3) ||
        axis_literal(program.nodes[id], 1) != 1 || !scalar(program.nodes[id].parents[2]) ||
        uses[id] != 2 || metadata->root_destinations[id] >= 0) continue;
    ScanReductionRegion region; region.nav = id;
    region.peak = find_child(id, ops::Op::cumulative_max_axis, 3);
    if (region.peak == UINT32_MAX || program.nodes[region.peak].parents[0] != id ||
        axis_literal(program.nodes[region.peak], 1) != 1 ||
        !scalar(program.nodes[region.peak].parents[2]) || uses[region.peak] != 1 ||
        metadata->root_destinations[region.peak] >= 0) continue;
    region.division = find_child(id, ops::Op::divide, 2);
    if (region.division == UINT32_MAX || program.nodes[region.division].parents[0] != id ||
        program.nodes[region.division].parents[1] != region.peak ||
        uses[region.division] != 1 || metadata->root_destinations[region.division] >= 0) continue;
    region.subtraction = find_child(region.division, ops::Op::subtract, 2);
    if (region.subtraction == UINT32_MAX || program.nodes[region.subtraction].parents[1] != region.division ||
        !scalar(program.nodes[region.subtraction].parents[0]) ||
        uses[region.subtraction] != 1 || metadata->root_destinations[region.subtraction] >= 0) continue;
    region.reduction = find_child(region.subtraction, ops::Op::max_axis, 2);
    if (region.reduction == UINT32_MAX || program.nodes[region.reduction].parents[0] != region.subtraction ||
        axis_literal(program.nodes[region.reduction], 1) != 1) continue;
    bool closed = true;
    for (auto j = id; j <= region.reduction; ++j) {
      if (j == id || j == region.peak || j == region.division ||
          j == region.subtraction || j == region.reduction) continue;
      closed = closed && scalar(j);
    }
    if (!closed) continue;
    if (child->scan_members.empty()) child->scan_members.assign(program.nodes.size(), -1);
    const auto group = std::int32_t(child->scan_reductions.size());
    for (auto member : {region.nav, region.peak, region.division, region.subtraction, region.reduction})
      child->scan_members[member] = group;
    child->scan_reductions.push_back(region);
  }
  for (std::uint32_t id = 0; id < program.nodes.size(); ++id) {
    if (!operation(id, ops::Op::subtract, 2) || uses[id] != 2 ||
        metadata->root_destinations[id] >= 0) continue;
    SquareReductionRegion region; region.subtraction = id;
    region.square = find_child(id, ops::Op::multiply, 2);
    if (region.square == UINT32_MAX || program.nodes[region.square].parents[0] != id ||
        program.nodes[region.square].parents[1] != id || uses[region.square] != 1 ||
        metadata->root_destinations[region.square] >= 0) continue;
    region.selection = find_child(region.square, ops::Op::where, 3);
    if (region.selection == UINT32_MAX || program.nodes[region.selection].parents[1] != region.square ||
        !scalar(program.nodes[region.selection].parents[2]) || uses[region.selection] != 1 ||
        metadata->root_destinations[region.selection] >= 0) continue;
    const auto mask = program.nodes[region.selection].parents[0];
    const auto &mask_node = program.nodes[mask];
    if (mask >= id || mask_node.kind != NodeKind::operation ||
        mask_node.storage != StorageKind::mask ||
        (mask_node.opcode != std::uint16_t(ops::Op::less_than) &&
         mask_node.opcode != std::uint16_t(ops::Op::less_equal) &&
         mask_node.opcode != std::uint16_t(ops::Op::greater_than) &&
         mask_node.opcode != std::uint16_t(ops::Op::greater_equal) &&
         mask_node.opcode != std::uint16_t(ops::Op::equal) &&
         mask_node.opcode != std::uint16_t(ops::Op::not_equal))) continue;
    region.reduction = find_child(region.selection, ops::Op::sum_axis, 2);
    if (region.reduction == UINT32_MAX || program.nodes[region.reduction].parents[0] != region.selection ||
        axis_literal(program.nodes[region.reduction], 1) != 1) continue;
    bool closed = true;
    for (auto j = id; j <= region.reduction; ++j) {
      if (j == id || j == region.square || j == region.selection || j == region.reduction) continue;
      closed = closed && scalar(j);
    }
    if (!closed) continue;
    if (child->square_members.empty()) child->square_members.assign(program.nodes.size(), -1);
    const auto group = std::int32_t(child->square_reductions.size());
    for (auto member : {region.subtraction, region.square, region.selection, region.reduction})
      child->square_members[member] = group;
    child->square_reductions.push_back(region);
    // This closed region owns its selection cache; do not also execute where/sum fusion.
    if (!child->where_sum_to.empty()) child->where_sum_to[region.selection] = -1;
    if (!child->where_sum_from.empty()) child->where_sum_from[region.reduction] = -1;
  }
  plan->canonical_metadata = std::move(child);
  return plan;
}

static void plan_state_decisions(const Program &p, ExecutionMetadata &m) {
  if (p.output_kind!=OutputKind::typed || p.isolate_errors || p.minimum_observations ||
      !p.apply_scopes.empty() || !p.rolling_scopes.empty()) return;
  std::vector<std::size_t> uses(p.nodes.size());
  std::vector<bool> scalar(p.nodes.size());
  for (const auto &n:p.nodes)
    for (std::size_t j=0;j<n.parent_count;++j) ++uses[n.parents[j]];
  for (std::size_t i=0;i<p.nodes.size();++i) {
    const auto &n=p.nodes[i];
    scalar[i]=n.kind==NodeKind::constant || n.kind==NodeKind::parameter;
    if (n.kind==NodeKind::operation && n.storage==StorageKind::inline_value && n.parent_count) {
      scalar[i]=true;
      for (std::size_t j=0;j<n.parent_count;++j) scalar[i]=scalar[i] && scalar[n.parents[j]];
    }
  }
  const auto select=[&](std::size_t i) {
    const auto &n=p.nodes[i];
    return n.kind==NodeKind::operation && n.parent_count==4 &&
        n.opcode==static_cast<std::uint16_t>(ops::Op::state_select) && n.storage==StorageKind::integer;
  };
  for (auto root:p.roots) {
    if (!select(root) || m.root_destinations[root]<0) continue;
    StateDecisionRegion region;
    std::vector<bool> members(p.nodes.size());
    auto id=root;
    bool closed=true, initialized=false;
    while (closed) {
      const auto &n=p.nodes[id];
      const auto comparison=n.parents[0];
      const auto &q=p.nodes[comparison];
      const auto op=static_cast<ops::Op>(q.opcode);
      if (q.kind!=NodeKind::operation || q.parent_count!=2 || q.storage!=StorageKind::mask ||
          op<ops::Op::equal || op>ops::Op::greater_equal || uses[comparison]!=1 ||
          m.root_destinations[comparison]>=0 || members[comparison] ||
          (id!=root && (uses[id]!=1 || m.root_destinations[id]>=0))) break;
      const auto side=scalar[q.parents[1]] ? 0u : 1u;
      if (!scalar[q.parents[1-side]] || scalar[q.parents[side]]) break;
      const auto source=q.parents[side], valid=n.parents[3];
      if (!initialized) {
        region.source=source; region.valid=valid; initialized=true;
        // Input validity is immutable for one execution. Internal masks need a
        // separate backing/lifetime proof and retain the canonical path here.
        if (p.nodes[valid].kind!=NodeKind::input) break;
        const auto &owner=p.nodes[source];
        if (owner.kind!=NodeKind::input && !(owner.kind==NodeKind::operation &&
            (owner.storage==StorageKind::integer || owner.storage==StorageKind::numeric))) break;
      }
      if (region.source!=source || region.valid!=valid) break;
      const auto previous=select(n.parents[1]) ? 1 : select(n.parents[2]) ? 2 : -1;
      if (previous<0 ? !(scalar[n.parents[1]] && scalar[n.parents[2]])
                     : !scalar[n.parents[3-previous]]) break;
      members[id]=members[comparison]=true;
      region.steps.push_back({comparison,id,static_cast<std::int8_t>(previous)});
      if (previous<0) { closed=false; break; }
      id=n.parents[previous];
    }
    if (closed || region.steps.size()<2 || region.steps.back().previous_argument>=0) continue;
    if (!m.state_decision_members.empty() && std::any_of(region.steps.begin(),region.steps.end(),
        [&](const auto &step) { return m.state_decision_members[step.selection]>=0 ||
                                      m.state_decision_members[step.comparison]>=0; })) continue;
    // Trace borrowed sources to their owned slot. Every delayed source read
    // must precede any real reuse; elided state writes do not overwrite it.
    auto owner=region.source;
    while (p.nodes[owner].kind==NodeKind::operation && p.nodes[owner].parent_count) {
      const auto op=static_cast<ops::Op>(p.nodes[owner].opcode);
      if (op!=ops::Op::lag && op!=ops::Op::transpose && op!=ops::Op::diag &&
          !ops::foundation_view(op) && !ops::latent_view(op) && !ops::series_record_projection(op)) break;
      owner=p.nodes[owner].parents[0];
    }
    const auto &backing=p.nodes[owner];
    bool stable=true;
    if (backing.storage!=StorageKind::inline_value && m.root_destinations[owner]<0)
      for (auto i=owner+1;stable && i<=root;++i)
        if (!members[i] && m.root_destinations[i]<0 &&
            p.nodes[i].storage==backing.storage && p.nodes[i].slot==backing.slot) stable=false;
    if (!stable) continue;
    std::reverse(region.steps.begin(),region.steps.end());
    const auto &source = p.nodes[region.source];
    const auto destination = m.root_destinations[root];
    region.loan_source = source.kind == NodeKind::operation && source.parent_count == 2 &&
        source.opcode == std::uint16_t(ops::Op::argmax_axis) && source.storage == StorageKind::integer &&
        m.root_destinations[region.source] < 0 &&
        p.nodes[source.parents[1]].kind == NodeKind::constant && p.nodes[source.parents[1]].constant == 1 &&
        p.root_outputs[destination].rank == 1 && p.root_outputs[destination].dtype == OutputDType::int64 &&
        std::count(p.roots.begin(), p.roots.end(), root) == 1;
    for (std::size_t consumer = 0; region.loan_source && consumer < p.nodes.size(); ++consumer) {
      if (consumer != root && m.root_destinations[consumer] == destination) region.loan_source = false;
      for (std::size_t j = 0; region.loan_source && j < p.nodes[consumer].parent_count; ++j)
        if (p.nodes[consumer].parents[j] == region.source &&
            std::none_of(region.steps.begin(), region.steps.end(),
                [&](const auto &step) { return step.comparison == consumer; })) region.loan_source = false;
    }
    if (m.state_decision_members.empty()) m.state_decision_members.assign(p.nodes.size(),-1);
    const auto group=static_cast<std::int32_t>(m.state_decisions.size());
    for (const auto &step:region.steps)
      m.state_decision_members[step.comparison]=m.state_decision_members[step.selection]=group;
    m.state_decisions.push_back(std::move(region));
    m.requires_shape_planning=true;
  }
}

static void plan_matmul_scalar_chain(const Program &p, ExecutionMetadata &m) {
  if (p.output_kind!=OutputKind::typed || p.isolate_errors || m.pointwise ||
      p.roots.size()!=1 || p.roots[0]+1!=p.nodes.size() || p.root_outputs[0].rank!=2 ||
      p.root_outputs[0].dtype!=OutputDType::float64 || !p.apply_scopes.empty() ||
      !p.rolling_scopes.empty()) return;
  std::vector<std::size_t> uses(p.nodes.size(),0);
  for (const auto &node:p.nodes)
    for (std::size_t j=0;j<node.parent_count;++j) ++uses[node.parents[j]];
  std::vector<std::uint32_t> chain;
  auto id=p.roots[0];
  for (;;) {
    const auto &node=p.nodes[id];
    if (node.kind!=NodeKind::operation || node.storage!=StorageKind::numeric ||
        node.parent_count!=2 || (id==p.roots[0] ? uses[id]!=0 : uses[id]!=1)) return;
    const auto op=static_cast<ops::Op>(node.opcode);
    chain.push_back(id);
    if (op==ops::Op::matmul) {
      if (chain.size()<2 || p.nodes[node.parents[0]].kind!=NodeKind::input ||
          p.nodes[node.parents[1]].kind!=NodeKind::input) return;
      break;
    }
    // Throwing division must preserve the canonical caller-owned failure prefix.
    if (op!=ops::Op::add && op!=ops::Op::subtract && op!=ops::Op::multiply) return;
    std::int32_t predecessor=-1;
    for (std::size_t j=0;j<2;++j) {
      const auto parent=node.parents[j]; const auto &other=p.nodes[node.parents[1-j]];
      if (p.nodes[parent].kind==NodeKind::operation &&
          (other.kind==NodeKind::constant || other.kind==NodeKind::parameter ||
           other.kind==NodeKind::input)) predecessor=static_cast<std::int32_t>(parent);
    }
    if (predecessor<0) return;
    id=static_cast<std::uint32_t>(predecessor);
  }
  for (std::size_t i=id+1;i<p.nodes.size();++i)
    if (p.nodes[i].kind==NodeKind::operation &&
        std::find(chain.begin(),chain.end(),i)==chain.end()) return;
  std::reverse(chain.begin(),chain.end());
  m.matmul_scalar_chain=std::move(chain);
}

static void plan_matmul_shared_chains(const Program &p, ExecutionMetadata &m) {
  if (p.output_kind!=OutputKind::typed || p.isolate_errors || m.pointwise ||
      p.roots.size()<2 || p.roots.size()>8 || !p.apply_scopes.empty() ||
      !p.rolling_scopes.empty()) return;
  std::vector<std::size_t> uses(p.nodes.size(),0);
  for (const auto &node:p.nodes)
    for (std::size_t j=0;j<node.parent_count;++j) ++uses[node.parents[j]];
  std::vector<std::vector<std::uint32_t>> chains;
  std::vector<std::uint8_t> members(p.nodes.size(),0);
  std::uint32_t producer=UINT32_MAX;
  for (std::size_t root=0;root<p.roots.size();++root) {
    // Planner arena admission precedes runtime slots. Different declared
    // orders cannot borrow one backing product: reserve the canonical arena
    // already in metadata instead of rejecting only during execution.
    if (p.root_outputs[root].rank!=2 || p.root_outputs[root].dtype!=OutputDType::float64 ||
        p.root_outputs[root].order!=p.root_outputs[0].order)
      return;
    std::vector<std::uint32_t> chain;
    auto id=p.roots[root];
    for (;;) {
      const auto &node=p.nodes[id];
      if (node.kind!=NodeKind::operation || node.storage!=StorageKind::numeric ||
          node.parent_count!=2) return;
      const auto op=static_cast<ops::Op>(node.opcode);
      chain.push_back(id);
      if (op==ops::Op::matmul) {
        if (chain.size()<2 || chain.size()>9 || uses[id]!=p.roots.size() ||
            (producer!=UINT32_MAX && producer!=id)) return;
        for (std::size_t j=0;j<2;++j) {
          const auto &input=p.nodes[node.parents[j]];
          if (input.kind!=NodeKind::input || p.input_axes.empty() ||
              p.input_axes[input.input_index]!=1) return;
        }
        producer=id;members[id]=1;break;
      }
      if ((id==p.roots[root] ? uses[id]!=0 : uses[id]!=1) || members[id] ||
          (op!=ops::Op::add && op!=ops::Op::subtract && op!=ops::Op::multiply)) return;
      members[id]=1;
      std::int32_t predecessor=-1;
      for (std::size_t j=0;j<2;++j) {
        const auto parent=node.parents[j];const auto &other=p.nodes[node.parents[1-j]];
        if (p.nodes[parent].kind==NodeKind::operation &&
            (other.kind==NodeKind::constant || other.kind==NodeKind::parameter ||
             other.kind==NodeKind::input)) predecessor=static_cast<std::int32_t>(parent);
      }
      if (predecessor<0) return;
      id=static_cast<std::uint32_t>(predecessor);
    }
    std::reverse(chain.begin(),chain.end());chains.push_back(std::move(chain));
  }
  for (std::size_t i=0;i<p.nodes.size();++i)
    if (p.nodes[i].kind==NodeKind::operation && !members[i]) return;
  const auto last=static_cast<std::size_t>(std::max_element(p.roots.begin(),p.roots.end())-p.roots.begin());
  for (std::size_t root=0;root<p.roots.size();++root)
    if (root!=last && chains[last][1]<=chains[root].back()) return;
  m.matmul_shared_chains=std::move(chains);
  m.matmul_shared_backing_root=static_cast<std::int32_t>(last);
}

static const std::vector<std::uint32_t> &matmul_chain_for_root(const Program &p,std::size_t root) {
  const auto &m=*p.execution_metadata;
  return m.matmul_shared_chains.empty() ? m.matmul_scalar_chain : m.matmul_shared_chains[root];
}

static std::int32_t matmul_chain_destination(const Program &p,std::size_t id) {
  const auto &m=*p.execution_metadata;
  if (m.matmul_shared_chains.empty())
    return std::find(m.matmul_scalar_chain.begin(),m.matmul_scalar_chain.end(),id)==m.matmul_scalar_chain.end() ? -1 : 0;
  if (id==m.matmul_shared_chains[0].front()) return m.matmul_shared_backing_root;
  for (std::size_t root=0;root<m.matmul_shared_chains.size();++root) {
    const auto &chain=m.matmul_shared_chains[root];
    if (std::find(chain.begin()+1,chain.end(),id)!=chain.end()) return static_cast<std::int32_t>(root);
  }
  return -1;
}

static bool matmul_scalar_chain_geometry(const Program &p,
    const std::vector<ops::Shape> &shapes) {
  const auto &m=*p.execution_metadata;
  if (shapes.size()!=p.nodes.size()) return false;
  const auto check=[&](const std::vector<std::uint32_t> &chain) {
    if (chain.empty()) return false;
    const auto &matrix=shapes[chain.front()];
    if (matrix.rank!=2 || matrix.size()<65536) return false;
    for (std::size_t i=1;i<chain.size();++i) {
      const auto &node=p.nodes[chain[i]];
      const auto side=node.parents[node.parents[0]==chain[i-1] ? 1 : 0];
      if (!(shapes[chain[i]]==matrix) || shapes[side].rank!=0) return false;
    }
    return true;
  };
  if (m.matmul_shared_chains.empty()) return check(m.matmul_scalar_chain);
  return m.matmul_shared_chains.size()==p.roots.size() && p.roots.size()<=8 &&
      std::all_of(m.matmul_shared_chains.begin(),m.matmul_shared_chains.end(),check);
}

static bool matmul_shared_layout_eligible(const Program &p,const std::vector<ops::Shape> &shapes,
    const ResultLayout *layout,std::size_t first,std::size_t rows) {
  if (p.execution_metadata->matmul_shared_chains.empty()) return true;
  if (!layout || layout->columns!=p.roots.size() || layout->row_bytes.empty() ||
      (!layout->orders.empty() && layout->orders.size()!=p.roots.size()) ||
      first>=layout->row_bytes.size() || rows>=layout->row_bytes.size()-first) return false;
  if (first+rows>layout->slots.size()/p.roots.size()) return false;
  for (std::size_t row=first;row<first+rows;++row) {
    const auto begin=layout->row_bytes[row],end=layout->row_bytes[row+1];
    if (end<begin) return false;
    std::size_t previous=begin;
    for (std::size_t root=0;root<p.roots.size();++root) {
      const auto &slot=layout->slots[row*p.roots.size()+root];
      const auto order=layout->order(root);
      if ((order!=ops::MemoryOrder::c && order!=ops::MemoryOrder::f) ||
          slot.byte_offset<previous || slot.byte_offset>end ||
          slot.byte_offset%alignof(double) ||
          slot.capacity>(end-slot.byte_offset)/sizeof(double) ||
          shapes[p.roots[root]].size()>slot.capacity) return false;
      previous=slot.byte_offset+slot.capacity*sizeof(double);
    }
    // The backing branch is in-place; all roots must share its actual order.
    // A mixed raw-native layout retains the canonical arenas/error sequence.
    for (std::size_t root=1;root<p.roots.size();++root)
      if (layout->order(root)!=layout->order(0)) return false;
  }
  return true;
}

static bool matmul_shared_dense_row(const Program &p,const std::vector<ops::Shape> &shapes,
    const ResultLayout &layout,std::size_t row) {
  auto next=layout.row_bytes[row];
  for (std::size_t root=0;root<p.roots.size();++root) {
    const auto &slot=layout.slots[row*p.roots.size()+root];
    if (slot.byte_offset!=next || slot.capacity!=shapes[p.roots[root]].size()) return false;
    next+=slot.capacity*sizeof(double);
  }
  return next==layout.row_bytes[row+1];
}

static bool matmul_scalar_chain_member(const Program &p,std::size_t id) {
  return matmul_chain_destination(p,id)>=0;
}

static bool row_subview_same_tree(const Program &p, std::uint32_t selected,
    std::uint32_t full, std::uint32_t take, unsigned depth = 0) {
  if (selected == full) return true;
  if (selected == take && full == p.nodes[take].parents[0]) return true;
  if (depth >= 12) return false;
  const auto &a = p.nodes[selected], &b = p.nodes[full];
  if (a.kind != NodeKind::operation || b.kind != NodeKind::operation ||
      a.opcode != b.opcode || a.parent_count != b.parent_count) return false;
  const auto op = static_cast<ops::Op>(a.opcode);
  if (op != ops::Op::add && op != ops::Op::subtract && op != ops::Op::multiply &&
      op != ops::Op::divide && op != ops::Op::broadcast_like) return false;
  for (std::size_t j = 0; j < a.parent_count; ++j)
    if (!row_subview_same_tree(p, a.parents[j], b.parents[j], take, depth + 1)) return false;
  return true;
}

static void plan_row_subviews(const Program &p, ExecutionMetadata &m) {
  if (p.output_kind != OutputKind::typed || p.isolate_errors ||
      !p.apply_scopes.empty() || !p.rolling_scopes.empty()) return;
  const auto is_root = [&](std::size_t id) {
    return std::find(p.roots.begin(), p.roots.end(), id) != p.roots.end();
  };
  const auto owned_root = [&](std::size_t id) {
    const auto root = m.root_destinations[id];
    if (root < 0 || p.roots[root] != id || p.nodes[id].kind != NodeKind::operation ||
        p.nodes[id].storage != StorageKind::numeric || p.root_outputs[root].rank != 2 ||
        p.root_outputs[root].dtype != OutputDType::float64) return false;
    // This independent root has exactly one physical writer for the whole call.
    for (std::size_t j = 0; j < m.root_destinations.size(); ++j)
      if (j != id && m.root_destinations[j] == root) return false;
    return true;
  };
  for (std::size_t id = 0; id < p.nodes.size(); ++id) {
    const auto &take = p.nodes[id];
    if (take.kind != NodeKind::operation || take.parent_count != 3 || is_root(id) ||
        take.storage != StorageKind::numeric || take.opcode != std::uint16_t(ops::Op::take_axis) ||
        p.nodes[take.parents[2]].kind != NodeKind::constant ||
        p.nodes[take.parents[2]].constant != 0) continue;
    if (p.nodes[take.parents[0]].kind == NodeKind::input || owned_root(take.parents[0]))
      m.row_subview_takes.push_back(static_cast<std::uint32_t>(id));
  }
  for (std::size_t id = 0; id < p.nodes.size(); ++id) {
    const auto &selected = p.nodes[id];
    if (selected.kind != NodeKind::operation || selected.storage != StorageKind::numeric || is_root(id)) continue;
    const auto op = static_cast<ops::Op>(selected.opcode);
    if (op != ops::Op::add && op != ops::Op::subtract && op != ops::Op::multiply && op != ops::Op::divide) continue;
    bool found = false;
    for (const auto take : m.row_subview_takes) {
      if (take >= id || p.nodes[p.nodes[take].parents[0]].kind != NodeKind::input) continue;
      for (const auto root : p.roots) {
        if (root >= id || !owned_root(root)) continue;
        if (row_subview_same_tree(p, static_cast<std::uint32_t>(id), root, take)) {
          m.row_subview_reuses.push_back({static_cast<std::uint32_t>(id), root, take});
          found = true; break;
        }
      }
      if (found) break;
    }
  }
}

void Program::finalize() {
  validate();
  auto metadata = std::make_shared<ExecutionMetadata>();
  metadata->pointwise = pointwise_eligible(*this);
  metadata->root_destinations.assign(nodes.size(), -1);
  for (std::size_t i = 0; i < roots.size(); ++i)
    metadata->root_destinations[roots[i]] = static_cast<std::int32_t>(i);
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
      positions = scope.kind == ApplyKind::segment || scope.kind == ApplyKind::group ||
          scope.body->execution_metadata->position_status_nodes != 0;
      metadata->contains_segment_scope = metadata->contains_segment_scope ||
          scope.kind == ApplyKind::segment || scope.body->execution_metadata->contains_segment_scope;
    } else if (node.kind == NodeKind::rolling_scope) {
      const auto &scope = rolling_scopes[node.input_index];
      if (!scope.body->execution_metadata) scope.body->finalize();
      positions = true; // Every rolling body can originate a local numerical failure.
      metadata->contains_segment_scope = metadata->contains_segment_scope ||
          scope.body->execution_metadata->contains_segment_scope;
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
  // HMM's measured three-history route can borrow otherwise-unused graph
  // capacity. New private elisions must not remove that capacity and force a
  // beta copy each iteration. Keep the previously qualified HMM graph route
  // until physical arena/history cost selection is qualified as one change.
  const bool retain_hmm_history=std::any_of(nodes.begin(),nodes.end(),[](const Node &n) {
    return n.kind==NodeKind::operation && n.opcode==static_cast<std::uint16_t>(ops::Op::gaussian_hmm_em);
  });
  // Mixed graphs may fuse a single-use pair without changing their logical DAG.
  // Most pairs require persistent backing. A proven private C input may survive
  // to a later arithmetic consumer without an intervening slot writer.
  // The adjacent eigenvalue projection
  // copies its entire source before writing, so its dead arena can be reused.
  if (output_kind == OutputKind::typed && !metadata->pointwise && !isolate_errors) {
    std::vector<std::size_t> uses(nodes.size(), 0);
    std::vector<bool> persistent(nodes.size(), false);
    metadata->deferred_to.assign(nodes.size(), -1);
    metadata->fused_from.assign(nodes.size(), -1);
    metadata->deferred_broadcast_argument.assign(nodes.size(), -1);
    metadata->deferred_private_argument.assign(nodes.size(), -1);
    for (const auto &n : nodes)
      for (std::size_t j=0;j<n.parent_count;++j) ++uses[n.parents[j]];
    for (std::size_t i=0;i<nodes.size();++i) {
      const auto &n=nodes[i];
      persistent[i]=n.kind==NodeKind::input || n.kind==NodeKind::constant || n.kind==NodeKind::parameter;
      if (n.kind!=NodeKind::operation) continue;
      const auto op=static_cast<ops::Op>(n.opcode);
      const bool view=op==ops::Op::lag || op==ops::Op::transpose || ops::foundation_view(op) ||
          ops::latent_view(op) || ops::series_record_projection(op);
      if (view) persistent[i]=persistent[n.parents[0]];
      else if (metadata->root_destinations[i]>=0 && n.storage==StorageKind::numeric) persistent[i]=true;
      const bool reduce=op==ops::Op::sum_axis || op==ops::Op::mean_axis;
      const bool arithmetic=op==ops::Op::add || op==ops::Op::subtract || op==ops::Op::multiply || op==ops::Op::divide;
      const bool factor=op==ops::Op::cholesky && n.parent_count==1;
      const bool spectrum=op==ops::Op::eigenvalues && n.parent_count==1;
      const bool scatter_max=op==ops::Op::argmax_axis && n.parent_count==2 &&
          nodes[n.parents[1]].kind==NodeKind::constant && nodes[n.parents[1]].constant==1;
      if ((!reduce && !arithmetic && !factor && !spectrum && !scatter_max) ||
          n.storage!=(scatter_max ? StorageKind::integer : StorageKind::numeric)) continue;
      std::int32_t cached_candidate=-1;
      std::int8_t cached_argument=-1, cached_private=-1;
      for (std::size_t j=0;j<(arithmetic ? 2u : 1u);++j) {
        const auto id=n.parents[j]; const auto &first=nodes[id];
        if (first.kind!=NodeKind::operation || first.storage!=StorageKind::numeric || uses[id]!=1 ||
            metadata->root_destinations[id]>=0 || metadata->fused_from[id]>=0) continue;
        const auto first_op=static_cast<ops::Op>(first.opcode);
        // Keep covariance failure evaluation adjacent to its original position.
        bool eligible=scatter_max ? first_op==ops::Op::scatter_rows && id+1==i :
            spectrum ? first_op==ops::Op::symmetric_eigh && id+1==i :
            factor ? first_op==ops::Op::covariance && first.parent_count==1 && id+1==i :
            reduce ? first_op==ops::Op::take_axis :
            first_op==ops::Op::add || first_op==ops::Op::subtract || first_op==ops::Op::multiply;
        // The integer destination can reuse a dead index slot. Only input-backed
        // index views are stable here; computed indices keep canonical execution.
        if (scatter_max) eligible=eligible && persistent[first.parents[1]];
        if (reduce) eligible=eligible && nodes[first.parents[2]].kind==NodeKind::constant &&
            nodes[n.parents[1]].kind==NodeKind::constant &&
            nodes[first.parents[2]].constant==nodes[n.parents[1]].constant;
        // The spectrum kernel copies the adjacent source before writing its
        // smaller result, so an arena-backed source can safely die here.
        std::int8_t saved_argument=-1, private_argument=-1;
        if (!spectrum && !scatter_max)
          for (std::size_t a=0;a<first.parent_count;++a) {
            if (persistent[first.parents[a]]) continue;
            const auto source=first.parents[a];
            const auto &view=nodes[source];
            const bool short_broadcast=arithmetic && saved_argument<0 &&
                view.kind==NodeKind::operation && view.parent_count==2 &&
                view.opcode==static_cast<std::uint16_t>(ops::Op::broadcast_like);
            if (short_broadcast) { saved_argument=static_cast<std::int8_t>(a); continue; }
            // The compiler's original slots remain unchanged. Delay reading an
            // owned private array only if no intervening operation can reuse it.
            bool retained=!retain_hmm_history && arithmetic && private_argument<0 && apply_scopes.empty() &&
                rolling_scopes.empty() && view.kind==NodeKind::operation &&
                view.storage==StorageKind::numeric && metadata->root_destinations[source]<0 &&
                metadata->deferred_to[source]<0 && view.slot!=first.slot;
            if (retained) {
              const auto source_op=static_cast<ops::Op>(view.opcode);
              retained=source_op!=ops::Op::lag && source_op!=ops::Op::transpose &&
                  source_op!=ops::Op::diag &&
                  !ops::foundation_view(source_op) && !ops::latent_view(source_op) &&
                  !ops::series_record_projection(source_op);
            }
            for (std::size_t between=id+1;retained && between<i;++between)
              if (nodes[between].storage==StorageKind::numeric &&
                  nodes[between].slot==view.slot) retained=false;
            if (retained && metadata->root_destinations[i]<0 && n.slot==view.slot) {
              // In-place C elementwise writes are safe for this exact full
              // input. A second view of the same arena could read future cells;
              // reject it rather than relying on an Eigen aliasing assumption.
              auto other=n.parents[1-j];
              while (nodes[other].kind==NodeKind::operation && nodes[other].parent_count) {
                const auto other_op=static_cast<ops::Op>(nodes[other].opcode);
                if (other_op!=ops::Op::lag && other_op!=ops::Op::transpose &&
                    other_op!=ops::Op::diag &&
                    !ops::foundation_view(other_op) && !ops::latent_view(other_op) &&
                    !ops::series_record_projection(other_op)) break;
                other=nodes[other].parents[0];
              }
              if (nodes[other].storage==StorageKind::numeric && nodes[other].slot==view.slot)
                retained=false;
            }
            if (retained) private_argument=static_cast<std::int8_t>(a);
            else eligible=false;
          }
        if (!eligible) continue;
        // Geometry-dependent caching must not shadow an existing proven pair
        // in the other operand when the broadcast later turns out scalar.
        if (saved_argument>=0 || private_argument>=0) {
          if (cached_candidate<0) { cached_candidate=static_cast<std::int32_t>(id); cached_argument=saved_argument; cached_private=private_argument; }
          continue;
        }
        metadata->deferred_to[id]=static_cast<std::int32_t>(i);
        metadata->fused_from[i]=static_cast<std::int32_t>(id);
        metadata->deferred_broadcast_argument[id]=saved_argument;
        break;
      }
      if (metadata->fused_from[i]<0 && cached_candidate>=0) {
        metadata->deferred_to[cached_candidate]=static_cast<std::int32_t>(i);
        metadata->fused_from[i]=cached_candidate;
        metadata->deferred_broadcast_argument[cached_candidate]=cached_argument;
        metadata->deferred_private_argument[cached_candidate]=cached_private;
      }
    }
    // Adjacent private column gathering may feed an independent scatter root,
    // or the existing scatter/argmax pair. Retain the original logical nodes:
    // the take and scatter validations still execute at their own positions.
    if (!retain_hmm_history && apply_scopes.empty() && rolling_scopes.empty()) {
      auto backing = [&](std::uint32_t id) {
        for (;;) {
          const auto &source=nodes[id];
          bool borrowed=source.kind==NodeKind::interval_tail;
          if (source.kind==NodeKind::operation) {
            const auto op=static_cast<ops::Op>(source.opcode);
            borrowed=op==ops::Op::lag || op==ops::Op::transpose ||
                op==ops::Op::diag || ops::foundation_view(op) ||
                ops::latent_view(op) || ops::series_record_projection(op);
          }
          if (!borrowed || !source.parent_count) return id;
          id=source.parents[0];
        }
      };
      for (std::size_t i=1;i<nodes.size();++i) {
        const auto &scatter=nodes[i];
        if (scatter.kind!=NodeKind::operation || scatter.storage!=StorageKind::numeric ||
            scatter.opcode!=static_cast<std::uint16_t>(ops::Op::scatter_rows) ||
            metadata->fused_from[i]>=0) continue;
        const auto take_id=scatter.parents[0];const auto &take=nodes[take_id];
        bool consecutive=true;
        for (auto between=take_id+1;between<i;++between) {
          const auto kind=nodes[between].kind;
          consecutive=consecutive && (kind==NodeKind::input || kind==NodeKind::constant || kind==NodeKind::parameter);
        }
        if (!consecutive || take.kind!=NodeKind::operation || take.parent_count!=3 ||
            take.storage!=StorageKind::numeric || uses[take_id]!=1 ||
            metadata->root_destinations[take_id]>=0 || metadata->deferred_to[take_id]>=0 ||
            take.opcode!=static_cast<std::uint16_t>(ops::Op::take_axis) ||
            nodes[take.parents[2]].kind!=NodeKind::constant || nodes[take.parents[2]].constant!=1.) continue;
        const auto last=metadata->deferred_to[i]>=0 ?
            static_cast<std::size_t>(metadata->deferred_to[i]) : i;
        if (last==i && metadata->root_destinations[i]<0) continue;
        bool stable=true;
        for (std::size_t argument=0;argument<2;++argument) {
          const auto source_id=backing(take.parents[argument]);const auto &source=nodes[source_id];
          if (source.kind==NodeKind::input || source.kind==NodeKind::constant ||
              source.kind==NodeKind::parameter) continue;
          const auto kind=argument==0 ? StorageKind::numeric : StorageKind::integer;
          if (source.kind!=NodeKind::operation || source.storage!=kind) {stable=false;break;}
          for (auto writer=take_id+1;writer<=last;++writer) {
            // The scatter is geometry-only in the established argmax pair;
            // a public root writes independent result storage, not this arena.
            if ((writer==i && last!=i) || metadata->root_destinations[writer]>=0) continue;
            if (nodes[writer].storage==kind && nodes[writer].slot==source.slot) {stable=false;break;}
          }
          if (!stable) break;
        }
        if (stable) {
          metadata->deferred_to[take_id]=static_cast<std::int32_t>(i);
          metadata->fused_from[i]=static_cast<std::int32_t>(take_id);
        }
      }
    }
    if (apply_scopes.empty() && rolling_scopes.empty()) {
      // Input-backed views stay live across deferred nodes. Public operation
      // roots can lose direct backing when finalization applies F layout, so
      // they are deliberately not treated as persistent leaves here.
      std::vector<bool> input_backed(nodes.size(), false);
      for (std::size_t id = 0; id < nodes.size(); ++id) {
        const auto &node = nodes[id];
        input_backed[id] = node.kind == NodeKind::input || node.kind == NodeKind::constant ||
            node.kind == NodeKind::parameter;
        if (node.kind != NodeKind::operation) continue;
        const auto op = static_cast<ops::Op>(node.opcode);
        if (op == ops::Op::lag || op == ops::Op::transpose || ops::foundation_view(op) ||
            ops::latent_view(op) || ops::series_record_projection(op))
          input_backed[id] = input_backed[node.parents[0]];
      }
      // Private matrix trajectories may be consumed by several row statistics.
      // Parents must remain input-backed; every consumer retains its own logical
      // prepare/error position. Reordering is further restricted to float roots
      // with positive literal scaling and at most one square root.
      std::vector<std::vector<std::uint32_t>> consumers(nodes.size());
      for (std::uint32_t id=0; id<nodes.size(); ++id)
        for (std::size_t j=0; j<nodes[id].parent_count; ++j)
          consumers[nodes[id].parents[j]].push_back(id);
      for (std::uint32_t id=0; id<nodes.size(); ++id) {
        const auto &producer=nodes[id];
        if (producer.kind!=NodeKind::operation || producer.parent_count!=2 ||
            producer.opcode!=static_cast<std::uint16_t>(ops::Op::matmul) ||
            producer.storage!=StorageKind::numeric || metadata->root_destinations[id]>=0 ||
            !input_backed[producer.parents[0]] || !input_backed[producer.parents[1]] ||
            consumers[id].empty()) continue;
        MatmulSummaryRegion region; region.producer=id; region.moments=true;
        bool valid=true, tail_valid=true;
        for (const auto child:consumers[id]) {
          const auto &n=nodes[child];
          const auto op=static_cast<ops::Op>(n.opcode);
          if (n.kind!=NodeKind::operation || n.parents[0]!=id || n.parent_count<2 ||
              n.storage!=StorageKind::numeric ||
              (op!=ops::Op::sum_axis && op!=ops::Op::mean_axis &&
               op!=ops::Op::variance_axis && op!=ops::Op::quantiles_axis)) {
            valid=false; break;
          }
          const bool quantiles=op==ops::Op::quantiles_axis;
          const auto axis=n.parents[quantiles ? 2 : 1];
          if (nodes[axis].kind!=NodeKind::constant || nodes[axis].constant!=1. ||
              (quantiles && (n.parent_count!=3 || !input_backed[n.parents[1]] ||
               (n.parents[1]>=id && (nodes[n.parents[1]].kind!=NodeKind::input ||
                input_axes.empty() || input_axes[nodes[n.parents[1]].input_index]!=1))))) {
            valid=false; break;
          }
          region.consumers.push_back(child);
          if (quantiles) { region.quantiles.push_back(child); region.moments=false; }
          const auto kind=quantiles ? 3u : op==ops::Op::sum_axis ? 0u : op==ops::Op::mean_axis ? 1u : 2u;
          if (kind==2) {
            if (n.parent_count!=3 || nodes[n.parents[2]].kind!=NodeKind::constant ||
                !std::isfinite(nodes[n.parents[2]].constant) || nodes[n.parents[2]].constant<0 ||
                nodes[n.parents[2]].constant!=std::floor(nodes[n.parents[2]].constant) ||
                nodes[n.parents[2]].constant>1000000.) {
              region.moments=false; region.trajectory_library=false;
            }
            else region.maximum_ddof=std::max(region.maximum_ddof,
                static_cast<std::size_t>(nodes[n.parents[2]].constant));
          }
          struct Tail { std::uint32_t id; double scale; bool square_root; };
          std::vector<Tail> pending{{child,1.,false}};
          std::size_t visited=0;
          while (!pending.empty() && tail_valid) {
            const auto tail=pending.back(); pending.pop_back();
            if (++visited>4096 || !std::isfinite(tail.scale)) { tail_valid=false; break; }
            // Every intermediate can overflow before a later scale shrinks the
            // root. Subnormal/zero cumulative scales also lose a reliable error
            // coefficient: retain bounded canonical rows in that rare domain.
            if (tail.scale<std::numeric_limits<double>::min()) {
              region.moments=false; region.trajectory_library=false;
            }
            const auto root=metadata->root_destinations[tail.id];
            if (root>=0) {
              if (root_outputs[root].dtype!=OutputDType::float64 ||
                  root_outputs[root].rank!=(quantiles ? 2 : 1)) tail_valid=false;
            }
            auto &scale=quantiles ? region.quantile_scale :
                tail.square_root ? region.sqrt_scale[kind] : region.linear_scale[kind];
            scale=std::max(scale,tail.scale);
            if (consumers[tail.id].empty() && root<0) { tail_valid=false; break; }
            for (const auto next:consumers[tail.id]) {
              const auto &n=nodes[next];
              if (n.kind!=NodeKind::operation) { tail_valid=false; break; }
              const auto op=static_cast<ops::Op>(n.opcode);
              if (quantiles && op==ops::Op::negate && n.parent_count==1)
                pending.push_back({next,tail.scale,false});
              else if (!quantiles && op==ops::Op::sqrt && n.parent_count==1 && !tail.square_root)
                pending.push_back({next,std::sqrt(tail.scale),true});
              else if (op==ops::Op::multiply && n.parent_count==2) {
                const auto other=n.parents[n.parents[0]==tail.id ? 1 : 0];
                const auto &literal=nodes[other];
                if (literal.kind!=NodeKind::constant || !std::isfinite(literal.constant) ||
                    literal.constant<=0.) { tail_valid=false; break; }
                pending.push_back({next,tail.scale*literal.constant,tail.square_root});
              } else { tail_valid=false; break; }
            }
          }
        }
        if (!valid || !tail_valid) continue;
        if (metadata->matmul_summary_producer.empty()) {
          metadata->matmul_summary_producer.assign(nodes.size(),-1);
          metadata->matmul_summary_consumer.assign(nodes.size(),-1);
        }
        const auto group=static_cast<std::int32_t>(metadata->matmul_summaries.size());
        metadata->matmul_summary_producer[id]=group;
        for (const auto child:region.consumers) metadata->matmul_summary_consumer[child]=group;
        metadata->matmul_summaries.push_back(std::move(region));
      }
      // A private concatenation tree only moves bytes. Validate every logical
      // node in order, but write persistent leaves once into the final rectangle.
      // Descending order selects maximal, non-overlapping trees.
      for (std::size_t reverse = 0; reverse < nodes.size(); ++reverse) {
        const auto target = nodes.size() - 1 - reverse;
        const auto &root = nodes[target];
        if (root.kind != NodeKind::operation || root.storage != StorageKind::numeric ||
            root.opcode != static_cast<std::uint16_t>(ops::Op::concatenate) ||
            metadata->deferred_to[target] >= 0 || metadata->fused_from[target] >= 0) continue;
        std::vector<std::uint32_t> pending{static_cast<std::uint32_t>(target)}, children;
        bool valid = true;
        while (valid && !pending.empty()) {
          const auto id = pending.back(); pending.pop_back();
          const auto &node = nodes[id];
          const bool expand = id == target || (node.kind == NodeKind::operation &&
              node.opcode == static_cast<std::uint16_t>(ops::Op::concatenate) &&
              node.storage == StorageKind::numeric && uses[id] == 1 &&
              metadata->root_destinations[id] < 0 && metadata->deferred_to[id] < 0 &&
              metadata->fused_from[id] < 0);
          if (!expand) { valid = input_backed[id]; continue; }
          if (node.parent_count != 3) { valid = false; break; }
          const auto &axis = nodes[node.parents[2]];
          if (axis.kind != NodeKind::constant || !std::isfinite(axis.constant) ||
              axis.constant != std::floor(axis.constant) || axis.constant < 0 || axis.constant > 2) {
            valid = false; break;
          }
          if (id != target) children.push_back(id);
          pending.push_back(node.parents[1]); pending.push_back(node.parents[0]);
        }
        if (!valid || children.empty()) continue;
        if (metadata->concat_tree_roots.empty()) metadata->concat_tree_roots.assign(nodes.size(), 0);
        metadata->concat_tree_roots[target] = 1;
        for (auto child : children) metadata->deferred_to[child] = static_cast<std::int32_t>(target);
      }
      for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto &solve = nodes[i];
        if (solve.kind != NodeKind::operation || solve.parent_count != 2 ||
            solve.opcode != static_cast<std::uint16_t>(ops::Op::solve_multi)) continue;
        const auto root = metadata->root_destinations[i];
        const auto source = solve.parents[0];
        const auto &producer = nodes[source];
        const bool terminal = root >= 0 && uses[i] == 0 &&
            root_outputs[root].dtype == OutputDType::float64 && root_outputs[root].rank == 2;
        // Borrow existing inputs/public/shared coefficients. The private arena
        // path stays first and must not regain a full coefficient copy.
        const bool retained = producer.kind != NodeKind::operation || uses[source] != 1 ||
            metadata->root_destinations[source] >= 0;
        if (terminal && (retained || root_outputs[root].order == OutputOrder::f)) {
          if (metadata->solve_copy_roots.empty()) metadata->solve_copy_roots.assign(nodes.size(), 0);
          metadata->solve_copy_roots[i] = 1;
        }
        if (producer.kind != NodeKind::operation || producer.storage != StorageKind::numeric ||
            uses[source] != 1 || metadata->root_destinations[source] >= 0) continue;
        const auto op = static_cast<ops::Op>(producer.opcode);
        const bool arithmetic = op == ops::Op::add || op == ops::Op::subtract || op == ops::Op::multiply;
        if (!arithmetic && op != ops::Op::concatenate) continue;
        if (metadata->solve_private_source.empty())
          metadata->solve_private_source.assign(nodes.size(), -1);
        metadata->solve_private_source[i] = static_cast<std::int32_t>(source);
        bool library = arithmetic && root >= 0 && uses[i] == 0 &&
            root_outputs[root].dtype == OutputDType::float64 && root_outputs[root].rank == 2 &&
            metadata->fused_from[source] < 0 && metadata->deferred_to[source] < 0;
        for (std::size_t j = 0; j < producer.parent_count; ++j)
          library = library && persistent[producer.parents[j]] &&
              nodes[producer.parents[j]].kind != NodeKind::operation;
        if (library) {
          if (metadata->solve_library_source.empty()) {
            metadata->solve_library_source.assign(nodes.size(), -1);
            metadata->solve_library_consumer.assign(nodes.size(), -1);
          }
          metadata->solve_library_source[i] = static_cast<std::int32_t>(source);
          metadata->solve_library_consumer[source] = static_cast<std::int32_t>(i);
        }
      }
      for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto &slice = nodes[i];
        if (slice.kind != NodeKind::operation || slice.parent_count != 5 ||
            slice.opcode != static_cast<std::uint16_t>(ops::Op::slice_axis)) continue;
        const auto source = slice.parents[0];
        const auto &filter = nodes[source];
        if (filter.kind != NodeKind::operation || filter.storage != StorageKind::numeric ||
            filter.opcode != static_cast<std::uint16_t>(ops::Op::hmm_filter) ||
            uses[source] != 1 || metadata->root_destinations[source] >= 0) continue;
        bool constants = true;
        for (std::size_t j = 1; j < 5; ++j)
          constants = constants && nodes[slice.parents[j]].kind == NodeKind::constant;
        if (!constants || nodes[slice.parents[1]].constant != 0 ||
            nodes[slice.parents[4]].constant != 1) continue;
        if (metadata->filter_tail_source.empty()) {
          metadata->filter_tail_source.assign(nodes.size(), -1);
          metadata->filter_tail_slice.assign(nodes.size(), -1);
        }
        metadata->filter_tail_source[i] = static_cast<std::int32_t>(source);
        metadata->filter_tail_slice[source] = static_cast<std::int32_t>(i);
        // Copying a descriptor cannot extend a dead arena value. Prove that
        // each Gaussian argument's backing survives until the filter replay.
        const auto gaussian_id = filter.parents[0];
        const auto &gaussian = nodes[gaussian_id];
        if (gaussian.kind != NodeKind::operation || gaussian.parent_count != 3 ||
            gaussian.storage != StorageKind::numeric ||
            gaussian.opcode != static_cast<std::uint16_t>(ops::Op::gaussian_log_density) ||
            uses[gaussian_id] != 1 || metadata->root_destinations[gaussian_id] >= 0) continue;
        bool stable = true;
        for (std::size_t argument = 0; argument < gaussian.parent_count && stable; ++argument) {
          auto backing_id = gaussian.parents[argument];
          for (;;) {
            const auto &backing = nodes[backing_id];
            const auto op = static_cast<ops::Op>(backing.opcode);
            const bool view = backing.kind == NodeKind::interval_tail ||
                (backing.kind == NodeKind::operation && (op == ops::Op::lag ||
                 op == ops::Op::transpose || ops::foundation_view(op) ||
                 ops::latent_view(op) || ops::series_record_projection(op)));
            if (!view) break;
            backing_id = backing.parents[0];
          }
          const auto &backing = nodes[backing_id];
          if (backing.storage == StorageKind::inline_value) {
            // Unknown inline operations could be unsupported borrowed views.
            if (backing.kind == NodeKind::operation) stable = false;
            continue;
          }
          if (backing.storage != StorageKind::numeric) { stable = false; break; }
          for (std::size_t writer = gaussian_id + 1; writer < source; ++writer)
            if (nodes[writer].storage == StorageKind::numeric &&
                nodes[writer].slot == backing.slot) { stable = false; break; }
        }
        if (stable) {
          if (metadata->gaussian_filter_source.empty()) {
            metadata->gaussian_filter_source.assign(nodes.size(), -1);
            metadata->gaussian_filter_consumer.assign(nodes.size(), -1);
          }
          metadata->gaussian_filter_source[source] = static_cast<std::int32_t>(gaussian_id);
          metadata->gaussian_filter_consumer[gaussian_id] = static_cast<std::int32_t>(source);
        }
      }
    }
    // Only two adjacent projections may observe a private Jacobi record. Keep
    // the logical producer, but allow the kernel to write their own buffers.
    if (roots.size()==2 && apply_scopes.empty() && rolling_scopes.empty()) {
      const auto first=std::min(roots[0],roots[1]),second=std::max(roots[0],roots[1]);
      const auto static_input=[&](std::uint32_t id) {
        const auto &n=nodes[id];
        return n.kind==NodeKind::input && !input_axes.empty() && input_axes[n.input_index]==1;
      };
      const bool pair_roots=first!=second && nodes[first].parent_count==2 && nodes[second].parent_count==2 &&
          nodes[first].kind==NodeKind::operation && nodes[second].kind==NodeKind::operation &&
          nodes[first].opcode==static_cast<std::uint16_t>(ops::Op::solve_multi) &&
          nodes[second].opcode==static_cast<std::uint16_t>(ops::Op::solve_multi) &&
          nodes[first].parents[0]==nodes[second].parents[0] && !uses[first] && !uses[second];
      bool solve_pair=pair_roots, private_pair=pair_roots;
      for (auto id:roots) {
        const auto &schema=root_outputs[metadata->root_destinations[id]];
        private_pair=private_pair && schema.rank==2 && schema.dtype==OutputDType::float64 &&
            static_input(nodes[id].parents[1]);
        solve_pair=solve_pair && schema.rank==2 && schema.dtype==OutputDType::float64 &&
            schema.order!=OutputOrder::f;
        for (std::size_t j=0;j<nodes[id].parent_count;++j) solve_pair=solve_pair && static_input(nodes[id].parents[j]);
      }
      for (auto id=first+1;id<second;++id) {
        solve_pair=solve_pair && static_input(id);
        private_pair=private_pair && static_input(id);
      }
      if (private_pair) {
        const auto source=nodes[first].parents[0];
        const auto &producer=nodes[source];
        const auto op=static_cast<ops::Op>(producer.opcode);
        private_pair=producer.kind==NodeKind::operation && producer.storage==StorageKind::numeric &&
            uses[source]==2 && metadata->root_destinations[source]<0 &&
            (op==ops::Op::add || op==ops::Op::subtract || op==ops::Op::multiply) &&
            metadata->fused_from[source]<0 && metadata->deferred_to[source]<0;
        for (std::size_t j=0;j<producer.parent_count;++j)
          private_pair=private_pair && persistent[producer.parents[j]] &&
              nodes[producer.parents[j]].kind!=NodeKind::operation;
        if (private_pair) {
          metadata->solve_pair_private_source=static_cast<std::int32_t>(source);
          if (metadata->solve_library_consumer.empty()) {
            metadata->solve_library_source.assign(nodes.size(),-1);
            metadata->solve_library_consumer.assign(nodes.size(),-1);
          }
          metadata->solve_library_consumer[source]=static_cast<std::int32_t>(first);
        }
      }
      if (solve_pair || private_pair) {
        metadata->solve_pair_first=static_cast<std::int32_t>(first);
        metadata->solve_pair_second=static_cast<std::int32_t>(second);
      }
      std::int32_t values=-1,vectors=-1;
      for (auto id:roots) {
        const auto &n=nodes[id];
        if (n.kind!=NodeKind::operation || n.parent_count!=1 || uses[id]) continue;
        if (n.opcode==static_cast<std::uint16_t>(ops::Op::eigenvalues)) values=static_cast<std::int32_t>(id);
        if (n.opcode==static_cast<std::uint16_t>(ops::Op::eigenvectors)) vectors=static_cast<std::int32_t>(id);
      }
      if (values>=0 && vectors>=0 && nodes[values].parents[0]==nodes[vectors].parents[0]) {
        const auto source=nodes[values].parents[0];
        const auto &producer=nodes[source];
        const auto values_root=metadata->root_destinations[values];
        const auto vectors_root=metadata->root_destinations[vectors];
        if (producer.kind==NodeKind::operation && producer.storage==StorageKind::numeric &&
            producer.opcode==static_cast<std::uint16_t>(ops::Op::symmetric_eigh) && uses[source]==2 &&
            metadata->root_destinations[source]<0 && std::min(values,vectors)==static_cast<std::int32_t>(source+1) &&
            std::max(values,vectors)==static_cast<std::int32_t>(source+2) &&
            values_root>=0 && vectors_root>=0 &&
            root_outputs[values_root].rank==1 && root_outputs[values_root].dtype==OutputDType::float64 &&
            root_outputs[vectors_root].rank==2 && root_outputs[vectors_root].dtype==OutputDType::float64 &&
            root_outputs[vectors_root].order!=OutputOrder::c) {
          metadata->split_eigh_source=static_cast<std::int32_t>(source);
          metadata->split_eigh_values=values;
          metadata->split_eigh_vectors=vectors;
        }
      }
    }
    if (metadata->split_eigh_source<0 && metadata->filter_tail_source.empty() &&
        std::none_of(metadata->deferred_to.begin(), metadata->deferred_to.end(), [](auto x){return x>=0;})) {
      metadata->deferred_to.clear(); metadata->fused_from.clear();
      metadata->deferred_private_argument.clear();
    }
  }
  // A transpose of a private MatMul is an F-contiguous view of the MatMul's
  // C-contiguous bytes. Let the existing kernel write those bytes directly
  // into the independent F root result instead of retaining a full matrix in
  // the arena and copying it after the view is created.
  if (output_kind == OutputKind::typed && !isolate_errors && roots.size() == 1 &&
      root_outputs[0].rank == 2 && root_outputs[0].dtype == OutputDType::float64 &&
      root_outputs[0].order != OutputOrder::c && apply_scopes.empty() &&
      rolling_scopes.empty()) {
    const auto root_id = roots[0];
    const auto &root = nodes[root_id];
    if (root.kind == NodeKind::operation && root.parent_count == 1 &&
        root.opcode == static_cast<std::uint16_t>(ops::Op::transpose)) {
      const auto child_id = root.parents[0];
      const auto &child = nodes[child_id];
      if (child.kind == NodeKind::operation && child.parent_count == 2 &&
          child.storage == StorageKind::numeric &&
          child.opcode == static_cast<std::uint16_t>(ops::Op::matmul)) {
        std::size_t uses = 0;
        for (const auto &node : nodes)
          for (std::size_t j = 0; j < node.parent_count; ++j)
            uses += node.parents[j] == child_id;
        if (uses == 1 && metadata->root_destinations[child_id] < 0 &&
            (metadata->deferred_to.empty() || metadata->deferred_to[child_id] < 0)) {
          metadata->root_destinations[root_id] = -1;
          metadata->root_destinations[child_id] = 0;
        }
      }
    }
  }
  // F output is direct only for proven stride-aware writers. Other kernels
  // compute in their budgeted C arena, then materialize the independent root.
  for (std::size_t r=0;r<root_outputs.size();++r) {
    if (root_outputs[r].rank < 2 || root_outputs[r].order != OutputOrder::f) continue;
    const auto id=roots[r];
    // Independent covariance/correlation is bitwise symmetric after its canonical
    // upper-triangle finish. Its C bytes already represent the same square F
    // matrix; no arena, transpose or alternative numerical kernel is needed.
    // The admitted covariance/Cholesky pair instead transposes its own result
    // in place after successful C execution, with that movement audited.
    if (static_cast<std::int32_t>(id)!=metadata->split_eigh_vectors && !independent_covariance_root(*this, id) &&
        !independent_cholesky_f_root(*this, id, metadata.get()) &&
        !independent_matmul_root(*this, id) &&
        ((nodes[id].kind == NodeKind::operation && metadata->specs[id]->family != ops::Family::elementwise &&
          metadata->specs[id]->op != ops::Op::affine_contract &&
          !ops::axis_path_operator(metadata->specs[id]->op)) ||
         nodes[id].kind == NodeKind::apply_scope))
      metadata->root_destinations[id] = -1;
  }
  plan_state_decisions(*this, *metadata);
  plan_matmul_scalar_chain(*this, *metadata);
  plan_matmul_shared_chains(*this, *metadata);
  if (metadata->pointwise) plan_pointwise_regions(*this, *metadata);
  metadata->affine_projection = plan_affine_projection(*this);
  metadata->compact_path = plan_compact_path(*this);
  metadata->row_batch = plan_row_batch(*this, metadata);
  plan_row_subviews(*this, *metadata);
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
  std::vector<std::uint8_t> geometry_known(program.nodes.size(), 1);
  for (std::size_t i = 0; i < program.nodes.size(); ++i) {
    const auto &node = program.nodes[i];
    auto &shape = shapes[i];
    if (node.kind == NodeKind::input) {
      shape = inputs[node.input_index].shape;
      geometry_known[i] = shape.rank >= 0;
      // Unavailable geometry cannot produce numerical arrays. This is only a
      // capacity bound; the executor retains the unknown descriptor unchanged.
      if (shape.rank < 0) shape = ops::vector_shape(0);
      const auto axis = program.input_axes.empty() ? 0 : program.input_axes[node.input_index];
      if (axis != 1 && shape.rank) shape.dim[0] = std::min(shape.dim[0], max_window);
    } else if (node.kind == NodeKind::interval_tail) {
      shape = shapes[node.parents[0]];
      geometry_known[i] = geometry_known[node.parents[0]];
    }
    else if (node.kind == NodeKind::operation) {
      const auto op = static_cast<ops::Op>(node.opcode);
      if (node.storage != StorageKind::inline_value)
        for (std::size_t j = 0; j < node.parent_count; ++j)
          geometry_known[i] &= geometry_known[node.parents[j]];
      if (node.storage != StorageKind::inline_value || op == ops::Op::lag || op == ops::Op::transpose ||
          ops::series_record_projection(op) ||
          (op == ops::Op::diag && shapes[node.parents[0]].rank == 2)) {
        for (std::size_t j = 0; j < node.parent_count; ++j)
          if (shapes[node.parents[j]].rank > shape.rank ||
              (shapes[node.parents[j]].rank == shape.rank &&
               shapes[node.parents[j]].size() > shape.size())) shape = shapes[node.parents[j]];
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
        else if (op == ops::Op::affine_contract) {
          if (geometry_known[node.parents[0]] && geometry_known[node.parents[1]]) {
            ops::require(lhs.rank == 2 || lhs.rank == 3, "RANK_MISMATCH");
            shape = lhs;
            shape.dim[shape.rank - 1] = rhs.dim[1];
          } else shape = ops::vector_shape(0);
        }
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
      if (ops::matrix_foundation(op) || ops::latent_operator(op) || ops::axis_path_operator(op) ||
          op == ops::Op::affine_contract) {
        ops::Arguments<ops::Value> args(node.parent_count);
        std::uint32_t payload=0, geometries=0;
        for (std::size_t j=0;j<node.parent_count;++j) {
          const auto parent=node.parents[j];
          args[j].shape=shapes[parent];
          if (geometry_known[parent]) geometries |= 1u << j;
          if (program.nodes[parent].kind==NodeKind::constant) {
            args[j].scalar=program.nodes[parent].constant;
            payload|=1u<<j;
          }
        }
        if (op==ops::Op::take_axis || op==ops::Op::partition_mean || op==ops::Op::scatter_rows) args[1].kind=ops::Kind::integer;
        if (op==ops::Op::transition_counts || op==ops::Op::one_hot) args[0].kind=ops::Kind::integer;
        const auto result=ops::validate_structure(ops::lookup(node.opcode),args.data(),
            node.parent_count,geometries,payload);
        if (op == ops::Op::affine_contract || ops::axis_path_operator(op) ||
            geometries != (1u << node.parent_count) - 1) {
          geometry_known[i] = result.geometry_known;
          shape = result.geometry_known ? result.output_shape : ops::vector_shape(0);
        } else {
          ops::require(result.geometry_known,"CONSTANT_LAYOUT_REQUIRED");
          shape=result.output_shape;
        }
      }
    } else if (node.kind == NodeKind::rolling_scope) shape = ops::vector_shape(max_window);
    else if (node.kind == NodeKind::apply_scope && node.storage != StorageKind::inline_value) {
      const auto &scope = program.apply_scopes[node.input_index];
      if (scope.kind == ApplyKind::iterate) {
        shape = shapes[scope.argument_nodes[0]];
        geometry_known[i] = geometry_known[scope.argument_nodes[0]];
      }
      else if (scope.kind == ApplyKind::block && scope.input_nodes.empty()) shape = ops::vector_shape(max_window);
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

bool row_batch_inputs(const Program &program, const std::vector<ops::Value> &inputs,
    std::size_t logical_window, std::vector<ops::Value> &blocked, std::size_t &batch,
    std::size_t &count) {
  if (!program.execution_metadata || !program.execution_metadata->row_batch ||
      inputs.size() != program.input_count) return false;
  const auto &plan = *program.execution_metadata->row_batch;
  const auto &weights = inputs[plan.input];
  if (weights.kind != ops::Kind::number || weights.shape.rank != 2 ||
      weights.shape.dim[0] < 128 || weights.shape.dim[0] > 1000000 ||
      weights.shape.dim[1] > 128) return false;
  count = weights.shape.dim[0];
  std::vector<ops::Shape> full, local;
  blocked = inputs;
  try {
    required_array_capacity(program, inputs, logical_window, &full);
    const auto trajectory = full[plan.producer];
    if (trajectory.rank != 2 || trajectory.dim[0] != count || trajectory.dim[1] > 1000000) return false;
    // A large quantile list or broadcast may be wider than the trajectory.
    // Bound the whole varying dependency closure, not just its producer.
    std::size_t maximum_width = 1;
    for (std::size_t id = 0; id < full.size(); ++id) {
      const auto axis = plan.axes[id];
      if (axis < 0) continue;
      if (axis >= full[id].rank || full[id].dim[axis] != count) return false;
      maximum_width = std::max(maximum_width, full[id].size() / count);
    }
    batch = std::min<std::size_t>(64, std::max<std::size_t>(1,
        163840 / maximum_width));
    blocked[plan.input].shape.dim[0] = batch;
    required_array_capacity(program, blocked, logical_window, &local);
  } catch (const ops::Error &) {
    return false; // Preserve the canonical structural exception and its order.
  }
  for (std::size_t id = 0; id < full.size(); ++id) {
    const auto axis = plan.axes[id];
    auto expected = full[id];
    if (axis >= 0) {
      if (axis >= expected.rank || expected.dim[axis] != count) return false;
      expected.dim[axis] = batch;
    }
    if (!(local[id] == expected)) return false;
    const auto &node = program.nodes[id];
    if (node.kind == NodeKind::operation && node.opcode == std::uint16_t(ops::Op::broadcast_like)) {
      const auto parent = node.parents[0];
      if (plan.axes[parent] >= 0 &&
          plan.axes[parent] + full[id].rank - full[parent].rank != axis) return false;
    }
  }
  return true;
}

// The only allowed consumer is an optional transpose that changes coordinates.
// No unused/internal comparison or state node may observe the reordered sum.
static bool independent_matmul_root(const Program &program, std::size_t index) {
  if (program.output_kind != OutputKind::typed || program.isolate_errors ||
      program.roots.size() != 1 || !program.apply_scopes.empty() ||
      !program.rolling_scopes.empty() || program.root_outputs[0].rank != 2 ||
      program.root_outputs[0].dtype != OutputDType::float64) return false;
  const auto &node = program.nodes[index];
  if (node.kind != NodeKind::operation || node.parent_count != 2 ||
      node.opcode != static_cast<std::uint16_t>(ops::Op::matmul)) return false;
  const auto root = program.roots[0];
  if (root != index) {
    const auto &view = program.nodes[root];
    if (view.kind != NodeKind::operation || view.parent_count != 1 ||
        view.parents[0] != index ||
        view.opcode != static_cast<std::uint16_t>(ops::Op::transpose)) return false;
  }
  for (std::size_t i = 0; i < program.nodes.size(); ++i)
    for (std::size_t j = 0; j < program.nodes[i].parent_count; ++j)
      if (program.nodes[i].parents[j] == index && i != root) return false;
  return true;
}

// Qualification changes only this private terminal region's numerical route.
// Later leaves are read for admission without preparing or validating them out
// of logical order. Rejection retains the original operation/error sequence.
static bool matmul_chain_scalar(const Program &p, std::uint32_t id,
    const std::vector<ops::Value> &inputs, const double *parameters,
    double &value) {
  const auto &node=p.nodes[id];
  if (node.kind==NodeKind::constant) value=node.constant;
  else if (node.kind==NodeKind::parameter) value=parameters[node.input_index];
  else if (node.kind==NodeKind::input) {
    const auto &input=inputs[node.input_index];
    if (input.kind!=ops::Kind::number || input.shape.rank!=0) return false;
    value=input.data ? *static_cast<const double *>(input.data) : input.scalar;
  } else return false;
  return std::isfinite(value);
}

static void matmul_chain_checkpoint(Scratch &state) {
  native::check_deadline(state.deadline);
  if (state.cancelled && state.cancelled->load())
    throw native::Timeout("native matmul cancelled");
}

static bool matmul_chain_input_bound(const ops::Value &input, ops::MatmulInputBounds &combined,
                                    Scratch &scratch) {
  if (input.kind!=ops::Kind::number || input.shape.rank!=2 || !input.data)
    return false;
  const auto rows=input.shape.dim[0], columns=input.shape.dim[1];
  if (!rows || !columns) return false;
  // Bounded Eigen Maps borrow the original pitches; no packing or owned data.
  constexpr std::size_t tile=256;
  combined={};
  for (std::size_t row=0;row<rows;row+=tile)
    for (std::size_t column=0;column<columns;column+=tile) {
      matmul_chain_checkpoint(scratch);
      auto part=input;
      part.shape=ops::matrix_shape(std::min(tile,rows-row),std::min(tile,columns-column));
      part.data=static_cast<const double *>(input.data)+
          std::ptrdiff_t(row)*input.stride[0]+std::ptrdiff_t(column)*input.stride[1];
      ops::MatmulInputBounds bounds;
      if (ops::library_matmul_input_bounds(part,bounds)!=ops::MatmulBoundsStatus::valid)
        return false;
      combined.maximum=std::max(combined.maximum,bounds.maximum);
      combined.negative=combined.negative||bounds.negative;
    }
  return true;
}

static bool matmul_chain_certificate(const Program &p, const ops::Prepared &prepared,
    const std::vector<ops::Value> &inputs, const double *parameters,
    Scratch &scratch, double &final_error, ops::MatmulTerminalProof &proof,
    const std::vector<std::uint32_t> &chain, const ops::MatmulTerminalProof *cached=nullptr,
    bool single_precision=false) {
  if (chain.size()<2 || chain.size()>9 || std::fegetround()!=FE_TONEAREST ||
      !ops::matmul_accelerate_reserve_doubles(prepared.args[0].shape,
          prepared.args[1].shape,scratch.expanded_matmul_roots)) return false;
  std::array<double,8> scalars{};
  for (std::size_t i=1;i<chain.size();++i) {
    const auto &node=p.nodes[chain[i]];
    const auto side=node.parents[node.parents[0]==chain[i-1] ? 1 : 0];
    if (!matmul_chain_scalar(p,side,inputs,parameters,scalars[i-1])) return false;
    const auto absolute=std::abs(scalars[i-1]);
    const auto op=static_cast<ops::Op>(node.opcode);
    if (op==ops::Op::multiply) {
      // With <=8 multiplications, the lowest exact binary grid stays normal,
      // including fused library products. Zero/sign ambiguity is replayed below.
      if (absolute<0x1p-24 || absolute>1.0) return false;
    } else if (op==ops::Op::add || op==ops::Op::subtract) {
      if (absolute>16.0 || (absolute!=0.0 && absolute<0x1p-128)) return false;
    } else return false;
  }
  ops::MatmulInputBounds left,right;
  if (cached) { left=cached->left;right=cached->right; }
  else if (!matmul_chain_input_bound(prepared.args[0],left,scratch) ||
      !matmul_chain_input_bound(prepared.args[1],right,scratch)) return false;
  const double ma=left.maximum,mb=right.maximum;
  const auto up=[](double x) { return std::nextafter(x,std::numeric_limits<double>::infinity()); };
  constexpr double unit=0x1p-53, tolerance=0.000049;
  const double k=double(prepared.args[0].shape.dim[1]);
  const double ku=2.0*k*unit;
  if (ku>=1.0) return false;
  const double gamma=up(ku/std::nextafter(1.0-ku,0.0));
  // Match the mature root helper's full dot bound exactly, so it cannot enter
  // its less restrictive partitioned budget inside this arithmetic chain.
  double error=up(up(up(up(2.0*gamma)*k)*ma)*mb);
  if (single_precision) {
    // Input conversion plus a conservative binary32 dot bound, and the
    // original binary64 fold. Range/grid admission is checked before packing.
    if (ma>1.0 || mb>1.0) return false;
    constexpr double low_unit=0x1p-24;
    const double low_ku=up((2.0*k+8.0)*low_unit);
    if (low_ku>=1.0) return false;
    const double low_gamma=up(low_ku/std::nextafter(1.0-low_ku,0.0));
    const double conversion=up(up(2.0*low_unit)+up(low_unit*low_unit));
    const double factor=up(up(low_gamma+up(conversion*up(1.0+low_gamma)))+gamma);
    error=up(up(up(factor*k)*ma)*mb);
  }
  if (!std::isfinite(error) || error>tolerance) return false;
  const double rounding=up(unit/std::nextafter(1.0-unit,0.0));
  double magnitude=up(up(up(k*ma)*mb)*up(1.0+gamma));
  double lower=-up(magnitude+error),upper=up(magnitude+error);
  for (std::size_t i=1;i<chain.size();++i) {
    const auto absolute=std::abs(scalars[i-1]);
    if (single_precision) {
      const auto &node=p.nodes[chain[i]];
      const double scalar=scalars[i-1];
      double lo,hi;
      if (node.opcode==std::uint16_t(ops::Op::multiply)) {
        lo=lower*scalar;hi=upper*scalar;if (scalar<0.0) std::swap(lo,hi);
      } else if (node.opcode==std::uint16_t(ops::Op::add)) {lo=lower+scalar;hi=upper+scalar;}
      else if (node.parents[0]==chain[i-1]) {lo=lower-scalar;hi=upper-scalar;}
      else {lo=scalar-upper;hi=scalar-lower;}
      lower=std::nextafter(lo,-std::numeric_limits<double>::infinity());upper=up(hi);
      if (!std::isfinite(lower) || !std::isfinite(upper)) return false;
    }
    if (p.nodes[chain[i]].opcode==std::uint16_t(ops::Op::multiply)) {
      error=up(up(absolute*error)+up(up(up(2.0*rounding)*absolute)*up(magnitude+error)));
      magnitude=up(up(1.0+rounding)*up(absolute*magnitude));
    } else {
      error=up(error+up(up(2.0*rounding)*up(up(magnitude+absolute)+error)));
      magnitude=up(up(1.0+rounding)*up(magnitude+absolute));
    }
    if (!std::isfinite(magnitude) || !std::isfinite(error) || error>tolerance)
      return false;
  }
  if (single_precision && !(lower>error || upper<-error)) return false;
  final_error=error;
  proof.source=&prepared; proof.left=left; proof.right=right;
  return true;
}

static bool matmul_region_certificate(const Program &p,const ops::Prepared &prepared,
    const std::vector<ops::Value> &inputs,const double *parameters,Scratch &scratch,
    double &error,ops::MatmulTerminalProof &proof, bool single_precision=false,
    const ops::MatmulTerminalProof *cached=nullptr) {
  if (p.execution_metadata->matmul_shared_chains.empty())
    return matmul_chain_certificate(p,prepared,inputs,parameters,scratch,error,proof,
                                   p.execution_metadata->matmul_scalar_chain,cached,single_precision);
  error=0.0;ops::MatmulTerminalProof bounds;bool first=cached==nullptr;
  if (cached) bounds=*cached;
  for (const auto &chain:p.execution_metadata->matmul_shared_chains) {
    double branch_error=0.0;
    if (!matmul_chain_certificate(p,prepared,inputs,parameters,scratch,branch_error,proof,
                                 chain,first ? nullptr : &bounds,single_precision)) return false;
    if (first) { bounds=proof;first=false; }
    error=std::max(error,branch_error);
  }
  return true;
}

static void replay_matmul_chain_ambiguity(const Program &p, ops::Output &out,
                                         double error, Scratch &scratch, const std::vector<std::uint32_t> &chain) {
  const auto &matrix=p.nodes[chain.front()];
  const auto &a=scratch.values[matrix.parents[0]], &b=scratch.values[matrix.parents[1]];
  auto *data=static_cast<double *>(out.data);
  const auto rows=out.shape.dim[0], columns=out.shape.dim[1];
  for (std::size_t index=0;index<out.shape.size();++index) {
    if (index%65536==0) {
      matmul_chain_checkpoint(scratch);
      const auto count=std::min<std::size_t>(65536,out.shape.size()-index);
      if (ops::library_numeric_zero_separated(data+index,count,error)) {
        index+=count-1;
        continue;
      }
    }
    if (std::isfinite(data[index]) && std::abs(data[index])>error) continue;
    const auto row=out.order==ops::MemoryOrder::f ? index%rows : index/columns;
    const auto column=out.order==ops::MemoryOrder::f ? index/rows : index%columns;
    double value=ops::dot_value(a.row(row),b.column(column));
    for (std::size_t i=1;i<chain.size();++i) {
      const auto id=chain[i]; const auto &node=p.nodes[id];
      const auto argument=node.parents[0]==chain[i-1] ? 0 : 1;
      std::array<ops::Value,2> args;
      args[argument]=ops::Value::number(value);
      args[1-argument]=scratch.values[node.parents[1-argument]];
      const auto prepared=ops::prepare(*p.execution_metadata->specs[id],args.data(),2);
      ops::Output scalar;
      scalar.kind=prepared.output_kind; scalar.shape=prepared.output_shape;
      ops::Audit audit;
      ops::execute(prepared,scalar,scratch.workspace,ops::Isa::scalar,audit);
      value=scalar.scalar;
    }
    data[index]=value;
  }
}

static bool filter_tail_geometry(const Program &, std::size_t,
                                 const std::vector<ops::Shape> &);

std::size_t matrix_workspace_doubles(const Program &program,
                                     const std::vector<ops::Shape> &shapes,
                                     std::size_t capacity, bool expanded_matmul_roots) {
  std::size_t limit = 0;
  for (std::size_t index = 0; index < program.nodes.size(); ++index) {
    const auto &node = program.nodes[index];
    if (node.kind != NodeKind::operation) continue;
    const auto op = static_cast<ops::Op>(node.opcode);
    if (op == ops::Op::quantiles_axis) {
      if (shapes.empty()) limit = std::max(limit, capacity);
      else if (shapes.at(index).size()) {
        const auto &axis = program.nodes[node.parents[2]];
        const auto &source = shapes.at(node.parents[0]);
        if (axis.kind == NodeKind::constant && axis.constant >= 0 &&
            axis.constant < source.rank && axis.constant == std::floor(axis.constant))
          limit = std::max(limit, source.dim[static_cast<std::size_t>(axis.constant)]);
        else limit = std::max(limit, source.size());
      }
    } else if (gaussian_filter_stream_geometry(program, index, shapes) ||
               (op == ops::Op::gaussian_log_density &&
                !program.execution_metadata->gaussian_filter_consumer.empty() &&
                program.execution_metadata->gaussian_filter_consumer[index] >= 0 &&
                gaussian_filter_stream_geometry(program,
                    static_cast<std::size_t>(program.execution_metadata->gaussian_filter_consumer[index]), shapes))) {
      const auto k = shapes[index].dim[1];
      limit = std::max(limit, 131 * k);
    } else if (op == ops::Op::hmm_filter && filter_tail_geometry(program, index, shapes)) {
      // The private history projection holds two K-row states, in addition to
      // its K-row arena result. Admission and execution use the same proof.
      limit = std::max(limit, 2 * shapes[index].dim[1]);
    } else if (op == ops::Op::matmul) {
      const auto a = shapes.empty() ? ops::matrix_shape(1024, capacity) : shapes.at(node.parents[0]);
      const auto b = shapes.empty() ? ops::matrix_shape(capacity, 1024) : shapes.at(node.parents[1]);
      limit = std::max(limit, ops::matmul_scratch_doubles(a, b));
      if (independent_matmul_root(program, index) ||
          (program.execution_metadata &&
           matmul_chain_destination(program,index)>=0 &&
           program.nodes[index].opcode==std::uint16_t(ops::Op::matmul) &&
           (shapes.empty() || matmul_scalar_chain_geometry(program,shapes))))
        limit = std::max(limit, ops::matmul_accelerate_reserve_doubles(a, b, expanded_matmul_roots));
    } else if (independent_covariance_root(program, index)) {
      const auto shape = shapes.empty() ? ops::matrix_shape(capacity, capacity) : shapes.at(node.parents[0]);
      limit = std::max(limit, ops::covariance_accelerate_reserve_doubles(shape));
    } else if ((op == ops::Op::exp || op == ops::Op::log) &&
               independent_transcendental_root(program, index)) {
      // Input/output order is execution geometry; cover either tiled direction.
      // Actual allocation remains lazy, and same-order execution uses no tile.
      limit = std::max(limit, ops::transcendental_accelerate_scratch_doubles(
          shapes.empty() ? ops::matrix_shape(128, 512) : shapes.at(node.parents[0])));
    } else if (op == ops::Op::solve_multi && program.execution_metadata &&
               !program.execution_metadata->solve_copy_roots.empty() &&
               program.execution_metadata->solve_copy_roots[index]) {
      const auto n = shapes.empty() ? std::min<std::size_t>(capacity, 4096) : shapes.at(node.parents[0]).dim[0];
      const auto k = shapes.empty() ? 64 : shapes.at(node.parents[1]).dim[1];
      limit = std::max(limit, ops::solve_accelerate_copy_reserve_doubles(n, k));
      if (!program.execution_metadata->concat_tree_roots.empty() &&
          program.execution_metadata->concat_tree_roots[node.parents[0]])
        limit = std::max(limit, ops::solve_concat_reserve_doubles(n, k));
    } else if (op == ops::Op::solve_multi && !shapes.empty() &&
               !program.execution_metadata->solve_private_source.empty() &&
               program.execution_metadata->solve_private_source[index] >= 0 &&
               !program.execution_metadata->concat_tree_roots.empty() &&
               program.execution_metadata->concat_tree_roots[node.parents[0]]) {
      limit = std::max(limit, ops::solve_concat_reserve_doubles(
          shapes[node.parents[0]].dim[0], shapes[node.parents[1]].dim[1]));
    } else if (op == ops::Op::cholesky) {
      const auto n = shapes.empty() ? capacity : shapes.at(node.parents[0]).dim[0];
      limit = std::max(limit, ops::cholesky_scratch_doubles(n));
      if (program.roots.size() == 1 && program.roots[0] == index &&
          program.execution_metadata &&
          index < program.execution_metadata->fused_from.size() &&
          program.execution_metadata->fused_from[index] >= 0)
        limit = std::max(limit, ops::covariance_cholesky_accelerate_scratch_doubles(n));
    } else if (program.execution_metadata &&
               static_cast<std::int32_t>(index)==program.execution_metadata->split_eigh_source) {
      limit=std::max(limit,shapes.empty() ? capacity :
          ops::symmetric_eigh_split_scratch_doubles(shapes.at(node.parents[0]).dim[0]));
    }
  }
  return limit;
}

// The rank-dependent pair must use the same proof in admission and execution.
// Unknown geometry keeps its canonical matrix arena rather than underbudgeting.
static std::size_t deferred_broadcast_extent(const Program &program, std::size_t producer,
                                             const std::vector<ops::Shape> &shapes) {
  const auto &metadata=*program.execution_metadata;
  if (metadata.deferred_broadcast_argument.empty() ||
      metadata.deferred_broadcast_argument[producer]<0 || shapes.size()!=program.nodes.size()) return 0;
  const auto argument=std::size_t(metadata.deferred_broadcast_argument[producer]);
  const auto view=program.nodes[producer].parents[argument];
  const auto source=program.nodes[view].parents[0];
  const auto &output=shapes[producer];
  return output.rank==2 && output.dim[0]>=2 && output.dim[1]>0 && shapes[view]==output &&
      shapes[source].rank==1 && shapes[source].dim[0]==output.dim[1] ? output.dim[1] : 0;
}

static bool deferred_pair_geometry(const Program &program, std::size_t producer,
                                   const std::vector<ops::Shape> &shapes) {
  const auto &metadata=*program.execution_metadata;
  if (metadata.deferred_to.empty() || metadata.deferred_to[producer]<0) return false;
  const auto consumer=static_cast<std::size_t>(metadata.deferred_to[producer]);
  if (!metadata.deferred_private_argument.empty() && metadata.deferred_private_argument[producer]>=0) {
    if (shapes.size()!=program.nodes.size()) return false;
    const auto source=program.nodes[producer].parents[metadata.deferred_private_argument[producer]];
    // Private owned writers have canonical C geometry. If the destination
    // reuses that dead slot, every cell is read before the same cell is written.
    if (shapes[source].rank<1 || !(shapes[source]==shapes[producer]) ||
        !(shapes[producer]==shapes[consumer])) return false;
  }
  if (!metadata.deferred_broadcast_argument.empty() && metadata.deferred_broadcast_argument[producer]>=0)
    return deferred_broadcast_extent(program,producer,shapes)!=0;
  if (program.nodes[consumer].opcode==static_cast<std::uint16_t>(ops::Op::scatter_rows)) {
    const auto &take=program.nodes[producer];
    // Root-layout finalization/fallback can clear direct backing after pair
    // admission while leaving the program typed. A nondeferred scatter then
    // writes its arena and cannot rely on the original independent-root proof.
    if (metadata.deferred_to[consumer]<0 && metadata.root_destinations[consumer]<0) return false;
    if (metadata.deferred_to[consumer]>=0) {
      const auto reduce=std::size_t(metadata.deferred_to[consumer]);
      if (metadata.root_destinations[reduce]<0) {
        auto index=take.parents[1];
        for (;;) {
          const auto &n=program.nodes[index];
          bool borrowed=n.kind==NodeKind::interval_tail;
          if (n.kind==NodeKind::operation) {
            const auto op=static_cast<ops::Op>(n.opcode);
            borrowed=op==ops::Op::lag || op==ops::Op::transpose || op==ops::Op::diag ||
                ops::foundation_view(op) || ops::latent_view(op) || ops::series_record_projection(op);
          }
          if (!borrowed || !n.parent_count) break;
          index=n.parents[0];
        }
        const auto &indices=program.nodes[index];
        if (indices.storage==StorageKind::integer && program.nodes[reduce].slot==indices.slot) return false;
      }
    }
    return program.output_kind==OutputKind::typed &&
        shapes.size()==program.nodes.size() && shapes[take.parents[0]].rank==2 &&
        shapes[take.parents[1]].rank==1 && shapes[producer].rank==2 && shapes[consumer].rank==2 &&
        shapes[producer].dim[0]==shapes[take.parents[0]].dim[0] &&
        shapes[producer].dim[1]==shapes[take.parents[1]].dim[0];
  }
  if (program.nodes[consumer].opcode==static_cast<std::uint16_t>(ops::Op::argmax_axis))
    return !shapes.empty() && shapes[producer].rank==2;
  return true;
}

// The same exact geometry proof controls the cache extent and its only reader.
static bool where_sum_geometry(const Program &program, std::size_t producer,
                               const std::vector<ops::Shape> &shapes) {
  const auto &metadata = *program.execution_metadata;
  if (metadata.where_sum_to.empty() || metadata.where_sum_to[producer] < 0 ||
      shapes.size() != program.nodes.size()) return false;
  const auto consumer = std::size_t(metadata.where_sum_to[producer]);
  return shapes[producer].rank == 2 && shapes[consumer].rank == 1 &&
      shapes[producer].dim[0] == shapes[consumer].dim[0];
}

static bool square_reduce_geometry(const Program &program, std::size_t member,
                                   const std::vector<ops::Shape> &shapes) {
  const auto &metadata = *program.execution_metadata;
  if (metadata.square_members.empty() || metadata.square_members[member] < 0 ||
      shapes.size() != program.nodes.size()) return false;
  const auto &region = metadata.square_reductions[metadata.square_members[member]];
  const auto &path = shapes[region.subtraction];
  const auto &selection = program.nodes[region.selection];
  return path.rank == 2 && shapes[region.square] == path && shapes[region.selection] == path &&
      shapes[selection.parents[0]] == path && shapes[region.reduction].rank == 1 &&
      shapes[region.reduction].dim[0] == path.dim[0];
}

static void execute_square_reduce_rows(const ops::Prepared &subtraction,
    const ops::Value &mask, double otherwise, double *totals, Scratch &scratch) {
  const auto rows = subtraction.output_shape.dim[0], count = subtraction.output_shape.dim[1];
  const auto checkpoint = [](void *context) {
    auto &state = *static_cast<Scratch *>(context);
    native::check_deadline(state.deadline);
    if (state.cancelled && state.cancelled->load()) throw native::Timeout("native square reduction cancelled");
  };
  for (std::size_t row = 0; row < rows;) {
    if (ops::library_square_sum_quad(subtraction, mask, otherwise, row, totals, &scratch, checkpoint)) {
      row += 4;
      continue;
    }
    checkpoint(&scratch);
    double total = 0;
    if (count) {
      const auto &a = subtraction.args[0], &b = subtraction.args[1];
      const auto *left = a.shape.rank ? static_cast<const double *>(a.data) + a.offset(row * count) : &a.scalar;
      const auto *right = b.shape.rank ? static_cast<const double *>(b.data) + b.offset(row * count) : &b.scalar;
      const auto *condition = static_cast<const std::uint8_t *>(mask.data) + mask.offset(row * count);
      const auto ls = a.shape.rank ? a.stride[1] : 0, rs = b.shape.rank ? b.stride[1] : 0;
      for (std::size_t j = 0; j < count; ++j) {
        const auto delta = ops::elementwise_detail::subtract_step(
            left[std::ptrdiff_t(j) * ls], right[std::ptrdiff_t(j) * rs]);
        const auto square = ops::elementwise_detail::multiply_step(delta, delta);
        const auto value = condition[std::ptrdiff_t(j) * mask.stride[1]] ? square : otherwise;
        total = ops::axis_path_detail::ordered_sum_step(total, value);
      }
    }
    totals[row] = total;
    ++row;
  }
}

static bool scan_reduce_geometry(const Program &program, std::size_t member,
                                 const std::vector<ops::Shape> &shapes) {
  const auto &metadata = *program.execution_metadata;
  if (metadata.scan_members.empty() || metadata.scan_members[member] < 0 ||
      shapes.size() != program.nodes.size()) return false;
  const auto &region = metadata.scan_reductions[metadata.scan_members[member]];
  const auto &path = shapes[region.nav];
  return path.rank == 2 && path.dim[1] > 0 && shapes[region.peak] == path &&
      shapes[region.division] == path && shapes[region.subtraction] == path &&
      shapes[region.reduction].rank == 1 && shapes[region.reduction].dim[0] == path.dim[0];
}

// Cache the terminal value, not the private time paths. Numerical faults are
// recorded here and replayed at the original divide/max node after its prepare.
static unsigned execute_scan_reduce_rows(const ops::Prepared &nav,
    double peak_initial, double subtraction, double *totals, Scratch &scratch) {
  const auto &values = nav.args[0];
  const auto rows = values.shape.dim[0], count = values.shape.dim[1];
  unsigned errors = 0;
  const auto checkpoint = [](void *context) {
    auto &state = *static_cast<Scratch *>(context);
    native::check_deadline(state.deadline);
    if (state.cancelled && state.cancelled->load())
      throw native::Timeout("native scan cancelled");
  };
  for (std::size_t row = 0; row < rows;) {
    if (ops::library_scan_reduce_quad(values, row, nav.args[2].scalar,
        peak_initial, subtraction, totals, errors, &scratch, checkpoint)) {
      row += 4;
      continue;
    }
    checkpoint(&scratch);
    const auto *source = static_cast<const double *>(values.data) + values.offset(row * count);
    double state = nav.args[2].scalar, peak = peak_initial;
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < count; ++j) {
      state = ops::axis_path_detail::product_step(state, source[std::ptrdiff_t(j) * values.stride[1]]);
      peak = ops::axis_path_detail::strict_max_step(peak, state);
      double ratio;
      if (peak == 0.0) {
        errors |= 1; ratio = std::numeric_limits<double>::quiet_NaN();
      } else ratio = ops::elementwise_detail::nonzero_divide_step(state, peak);
      const auto value = ops::elementwise_detail::subtract_step(subtraction, ratio);
      if (!std::isfinite(value)) errors |= 2;
      maximum = ops::axis_path_detail::strict_max_step(maximum, value);
    }
    totals[row] = maximum;
    ++row;
  }
  return errors;
}

// Selection and left-fold sum use the original arithmetic and original order.
// All operands are already checked by the producer's canonical prepare.
static void execute_where_sum_rows(const ops::Prepared &where, double *totals, Scratch &scratch) {
  ops::validate_payload_structure(where);
  const auto rows = where.output_shape.dim[0], count = where.output_shape.dim[1];
  const auto checkpoint = [](void *context) {
    auto &state = *static_cast<Scratch *>(context);
    native::check_deadline(state.deadline);
    if (state.cancelled && state.cancelled->load()) throw native::Timeout("native selection reduction cancelled");
  };
  const auto scalar_bits = [](double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  };
  const auto &yes = where.args[1], &no = where.args[2];
  // Positive binary values have an exact count as their canonical left sum.
  // Exclude negative zero and lengths beyond exact binary64 integers.
  const bool count_true = !yes.shape.rank && !no.shape.rank &&
      scalar_bits(yes.scalar) == UINT64_C(0x3ff0000000000000) && scalar_bits(no.scalar) == 0;
  const bool count_false = !yes.shape.rank && !no.shape.rank &&
      scalar_bits(yes.scalar) == 0 && scalar_bits(no.scalar) == UINT64_C(0x3ff0000000000000);
  if ((count_true || count_false) && count <= (UINT64_C(1) << 53)) {
    const auto &mask = where.args[0];
    for (std::size_t row = 0; row < rows; ++row) {
      checkpoint(&scratch);
      std::size_t selected = 0;
      if (count) {
        const auto *condition = static_cast<const std::uint8_t *>(mask.data) + mask.offset(row * count);
        const auto wanted = std::uint8_t(count_true ? 1 : 0);
        if (mask.stride[1] == 1)
          selected = std::size_t(std::count(condition, condition + count, wanted));
        else
          for (std::size_t j = 0; j < count; ++j)
            selected += condition[std::ptrdiff_t(j) * mask.stride[1]] == wanted;
      }
      totals[row] = static_cast<double>(selected);
    }
    return;
  }
  for (std::size_t row = 0; row < rows;) {
    if (ops::library_where_sum_quad(where, row, totals, &scratch, checkpoint)) {
      row += 4;
      continue;
    }
    checkpoint(&scratch);
    double total = 0;
    if (count) {
      const auto &mask = where.args[0];
      const auto scalar_mask = std::uint8_t(mask.scalar);
      const auto *condition = mask.shape.rank
          ? static_cast<const std::uint8_t *>(mask.data) + mask.offset(row * count) : &scalar_mask;
      const auto *left = where.args[1].shape.rank
          ? static_cast<const double *>(where.args[1].data) + where.args[1].offset(row * count)
          : &where.args[1].scalar;
      const auto *right = where.args[2].shape.rank
          ? static_cast<const double *>(where.args[2].data) + where.args[2].offset(row * count)
          : &where.args[2].scalar;
      const auto ms = mask.shape.rank ? mask.stride[1] : 0;
      const auto ls = where.args[1].shape.rank ? where.args[1].stride[1] : 0;
      const auto rs = where.args[2].shape.rank ? where.args[2].stride[1] : 0;
      for (std::size_t j = 0; j < count; ++j) {
        const auto value = condition[std::ptrdiff_t(j) * ms]
            ? left[std::ptrdiff_t(j) * ls] : right[std::ptrdiff_t(j) * rs];
        total = ops::axis_path_detail::ordered_sum_step(total, value);
      }
    }
    totals[row] = total;
    ++row;
  }
}

static bool matmul_quantile_root(const Program &program, std::uint32_t child) {
  const auto root=program.execution_metadata->root_destinations[child];
  return root>=0 && program.roots[root]==child;
}

bool matmul_summary_geometry(const Program &program, std::size_t producer,
                             const std::vector<ops::Shape> &shapes) {
  const auto &metadata=*program.execution_metadata;
  if (shapes.size()!=program.nodes.size() || metadata.matmul_summary_producer.empty() ||
      metadata.matmul_summary_producer[producer]<0) return false;
  const auto &node=program.nodes[producer];
  const auto &a=shapes[node.parents[0]], &b=shapes[node.parents[1]];
  if (!(a.rank==2 && b.rank==2 && a.dim[1]==b.dim[0] &&
      a.dim[0]>=32 && a.dim[0]<=1000000 && b.dim[1]>=32 && b.dim[1]<=1000000 && a.dim[1]<=128)) return false;
  const auto &region=metadata.matmul_summaries[metadata.matmul_summary_producer[producer]];
  std::size_t cache_columns=2;
  for (const auto child:region.quantiles) {
    const auto &q=shapes[program.nodes[child].parents[1]];
    if (q.rank!=1 || q.dim[0]>128 || shapes[child].rank!=2 ||
        shapes[child].dim[0]!=a.dim[0] || shapes[child].dim[1]!=q.dim[0]) return false;
    // Do not replace a narrow trajectory with an even larger summary cache.
    if (!matmul_quantile_root(program,child)) {
      if (q.dim[0]>=b.dim[1] || cache_columns>=b.dim[1]-q.dim[0]) return false;
      cache_columns+=q.dim[0];
    }
  }
  return true;
}
static std::size_t matmul_quantile_offset(const Program &program,
    const MatmulSummaryRegion &region, std::uint32_t child,
    const std::vector<ops::Shape> &shapes) {
  const auto rows=shapes[region.producer].dim[0];
  std::size_t offset=2*rows;
  for (const auto id:region.quantiles) {
    if (id==child) return offset;
    if (!matmul_quantile_root(program,id)) offset+=rows*shapes[id].dim[1];
  }
  throw ops::Error("INVALID_PHYSICAL_PAIR");
}
std::size_t matmul_summary_cache_doubles(const Program &program,
    const std::vector<ops::Shape> &shapes, std::vector<std::size_t> *offsets) {
  const auto &groups=program.execution_metadata->matmul_summaries;
  if (offsets) {
    const auto count=groups.empty() ? 0 : groups.size()+1;
    if (offsets->capacity()!=count) std::vector<std::size_t>().swap(*offsets);
    offsets->assign(count,0);
  }
  std::size_t total=0, auxiliary=0;
  for (std::size_t i=0; i<groups.size(); ++i) {
    if (offsets) (*offsets)[i]=total;
    const auto producer=groups[i].producer;
    if (!matmul_summary_geometry(program,producer,shapes)) continue;
    const auto &node=program.nodes[producer];
    auto increment=2*shapes[producer].dim[0];
    for (const auto child:groups[i].quantiles) {
      if (matmul_quantile_root(program,child)) continue;
      const auto extra=shapes[producer].dim[0]*shapes[child].dim[1];
      ops::require(extra<=static_cast<std::size_t>(PTRDIFF_MAX)/sizeof(double)-increment,
                   "GRAPH_ARENA_OVERFLOW");
      increment+=extra;
    }
    ops::require(increment<=static_cast<std::size_t>(PTRDIFF_MAX)/sizeof(double)-total,
                 "GRAPH_ARENA_OVERFLOW");
    total+=increment;
    auxiliary=std::max(auxiliary,groups[i].quantiles.empty() ?
        ops::matmul_statistics_scratch_doubles(shapes[node.parents[0]],shapes[node.parents[1]]) :
        ops::matmul_quantiles_scratch_doubles(shapes[node.parents[0]],shapes[node.parents[1]]));
  }
  if (offsets && !groups.empty()) offsets->back()=total;
  ops::require(total<=static_cast<std::size_t>(PTRDIFF_MAX)/sizeof(double)-auxiliary,
               "GRAPH_ARENA_OVERFLOW");
  return total+auxiliary;
}

// No leaf lifetime is extended unless its backing is persistent. Intermediate
// descriptors are geometry only and are never read by this byte-copy lowering.
struct ConcatRectangle { std::size_t row, rows, column, columns; };
static void execute_concat_tree(const Program &program, std::size_t target,
                                ops::Output &out, const Scratch &scratch,
                                const ConcatRectangle *rectangle = nullptr) {
  struct Piece { std::uint32_t id; std::array<std::size_t, 3> origin{}; };
  std::vector<Piece> pending{{static_cast<std::uint32_t>(target), {}}};
  const auto &metadata = *program.execution_metadata;
  ops::require(out.kind == ops::Kind::number &&
      (out.order == ops::MemoryOrder::c || (out.order == ops::MemoryOrder::f && out.shape.rank == 2)),
               "CONCAT_TREE_CONTRACT");
  auto *destination = static_cast<double *>(out.data);
  auto geometry = out.shape;
  if (rectangle) {
    ops::require(out.shape.rank == 2 && rectangle->row <= out.shape.dim[0] &&
        rectangle->rows <= out.shape.dim[0] - rectangle->row &&
        rectangle->column <= out.shape.dim[1] &&
        rectangle->columns <= out.shape.dim[1] - rectangle->column, "CONCAT_TREE_CONTRACT");
    geometry = ops::matrix_shape(rectangle->rows, rectangle->columns);
  }
  while (!pending.empty()) {
    const auto piece = pending.back(); pending.pop_back();
    const auto &node = program.nodes[piece.id];
    if (piece.id == target || metadata.deferred_to[piece.id] == static_cast<std::int32_t>(target)) {
      const auto axis = static_cast<std::size_t>(program.nodes[node.parents[2]].constant);
      auto right = piece.origin;
      right[axis] += scratch.values[node.parents[0]].shape.dim[axis];
      pending.push_back({node.parents[1], right});
      pending.push_back({node.parents[0], piece.origin});
      continue;
    }
    auto leaf = scratch.values[piece.id];
    auto origin = piece.origin;
    ops::require(leaf.kind == ops::Kind::number && leaf.shape.rank == out.shape.rank,
                 "CONCAT_TREE_CONTRACT");
    for (int axis = 0; axis < out.shape.rank; ++axis)
      ops::require(piece.origin[axis] <= out.shape.dim[axis] &&
          leaf.shape.dim[axis] <= out.shape.dim[axis] - piece.origin[axis], "CONCAT_TREE_CONTRACT");
    if (rectangle) {
      const std::array<std::size_t,2> first{rectangle->row, rectangle->column};
      const std::array<std::size_t,2> count{rectangle->rows, rectangle->columns};
      std::ptrdiff_t offset = 0;
      bool empty = false;
      for (int axis = 0; axis < 2; ++axis) {
        const auto lower = std::max(origin[axis], first[axis]);
        const auto upper = std::min(origin[axis] + leaf.shape.dim[axis], first[axis] + count[axis]);
        if (lower >= upper) { empty = true; break; }
        offset += static_cast<std::ptrdiff_t>(lower - origin[axis]) * leaf.stride[axis];
        leaf.shape.dim[axis] = upper - lower;
        origin[axis] = lower - first[axis];
        ops::require(origin[axis] <= count[axis] && leaf.shape.dim[axis] <= count[axis] - origin[axis],
                     "CONCAT_TREE_CONTRACT");
      }
      if (empty) continue;
      leaf.data = static_cast<const double *>(leaf.data) + offset;
    }
    if (!leaf.size()) continue;
    const auto last = leaf.shape.rank - 1;
    const auto width = leaf.shape.dim[last];
    const auto *source = static_cast<const double *>(leaf.data);
    if (out.order == ops::MemoryOrder::f) {
      auto *target = destination + origin[0] + origin[1] * geometry.dim[0];
      const auto rows = leaf.shape.dim[0], columns = leaf.shape.dim[1];
      if (leaf.stride[0] == 1) {
        for (std::size_t column = 0; column < columns; ++column)
          std::memcpy(target + column * geometry.dim[0],
              source + static_cast<std::ptrdiff_t>(column) * leaf.stride[1], rows * sizeof(double));
        continue;
      }
      // View both rectangles transposed to reuse the mature F-to-C copier.
      if (leaf.stride[1] == 1 && leaf.stride[0] >= static_cast<std::ptrdiff_t>(columns) &&
          rows >= 32 && columns >= 32 && ops::try_f_rectangle_copy(source,
              static_cast<std::size_t>(leaf.stride[0]), target, geometry.dim[0], columns, rows)) continue;
      constexpr std::size_t tile = 32;
      for (std::size_t column = 0; column < columns; column += tile)
        for (std::size_t row = 0; row < rows; row += tile)
          for (std::size_t j = column; j < std::min(columns, column + tile); ++j)
            for (std::size_t i = row; i < std::min(rows, row + tile); ++i)
              target[i + j * geometry.dim[0]] = source[static_cast<std::ptrdiff_t>(i) * leaf.stride[0] +
                  static_cast<std::ptrdiff_t>(j) * leaf.stride[1]];
      continue;
    }
    if (leaf.shape.rank == 2 && leaf.stride[0] == 1 &&
        leaf.stride[1] >= static_cast<std::ptrdiff_t>(leaf.shape.dim[0]) &&
        leaf.shape.dim[0] >= 32 && width >= 32) {
      if (ops::try_f_rectangle_copy(source, static_cast<std::size_t>(leaf.stride[1]),
          destination + origin[0] * geometry.dim[1] + origin[1], geometry.dim[1],
          leaf.shape.dim[0], width)) continue;
      // F-backed rectangles need bounded tiles. Walking an entire C row of
      // a large F input otherwise touches a new page for nearly every value.
      constexpr std::size_t tile = 32;
      for (std::size_t first_row = 0; first_row < leaf.shape.dim[0]; first_row += tile)
        for (std::size_t first_col = 0; first_col < width; first_col += tile) {
          const auto row_end = std::min(leaf.shape.dim[0], first_row + tile);
          const auto col_end = std::min(width, first_col + tile);
          for (auto col = first_col; col < col_end; ++col)
            for (auto row = first_row; row < row_end; ++row)
              destination[(origin[0] + row) * geometry.dim[1] + origin[1] + col] =
                  source[static_cast<std::ptrdiff_t>(row) + static_cast<std::ptrdiff_t>(col) * leaf.stride[1]];
        }
      continue;
    }
    for (std::size_t line = 0; line < leaf.size() / width; ++line) {
      auto coordinates = origin;
      auto remaining = line;
      std::ptrdiff_t source_offset = 0;
      for (int axis = last - 1; axis >= 0; --axis) {
        const auto coordinate = remaining % leaf.shape.dim[axis];
        remaining /= leaf.shape.dim[axis];
        coordinates[axis] += coordinate;
        source_offset += static_cast<std::ptrdiff_t>(coordinate) * leaf.stride[axis];
      }
      std::size_t base = 0;
      for (int axis = 0; axis <= last; ++axis) base = base * geometry.dim[axis] + coordinates[axis];
      if (leaf.stride[last] == 1)
        std::memcpy(destination + base, source + source_offset, width * sizeof(double));
      else
        for (std::size_t j = 0; j < width; ++j)
          destination[base + j] = source[source_offset + static_cast<std::ptrdiff_t>(j) * leaf.stride[last]];
    }
  }
}

// This proof controls both the physical arena size and the private execution.
static bool filter_last_row_geometry(const Program &program, std::size_t source,
                                 const std::vector<ops::Shape> &shapes) {
  const auto &metadata = *program.execution_metadata;
  if (metadata.filter_tail_slice.empty() || metadata.filter_tail_slice[source] < 0 ||
      shapes.empty()) return false;
  const auto slice_id = static_cast<std::size_t>(metadata.filter_tail_slice[source]);
  const auto &full = shapes[source], &tail = shapes[slice_id];
  if (full.rank != 2 || full.dim[0] < 3 || !full.dim[1] ||
      tail.rank != 2 ||
      tail.dim[0] != 1 || tail.dim[1] != full.dim[1]) return false;
  const auto &slice = program.nodes[slice_id];
  return program.nodes[slice.parents[2]].constant == double(full.dim[0] - 1) &&
         program.nodes[slice.parents[3]].constant == double(full.dim[0]);
}

static bool gaussian_filter_stream_candidate(const Program &program, std::size_t filter,
                                     const std::vector<ops::Shape> &shapes) {
  if (!program.execution_metadata) return false;
  const auto &m = *program.execution_metadata;
  if (filter >= program.nodes.size() || shapes.size() != program.nodes.size() ||
      m.gaussian_filter_source.empty() || m.gaussian_filter_source[filter] < 0 ||
      !filter_last_row_geometry(program, filter, shapes)) return false;
  const auto source = static_cast<std::size_t>(m.gaussian_filter_source[filter]);
  const auto &g = program.nodes[source];
  const auto &full = shapes[filter], &x = shapes[g.parents[0]];
  const auto &means = shapes[g.parents[1]], &vars = shapes[g.parents[2]];
  return full.size() >= 131072 && full.dim[1] <= ops::latent_gaussian_stream_max_components &&
      x.rank == 2 && x.dim[0] == full.dim[0] && x.dim[1] > 0 && x.dim[1] <= ops::latent_gaussian_stream_max_features &&
      means.rank == 2 && means.dim[0] == full.dim[1] && means.dim[1] == x.dim[1] &&
      vars == means && shapes[source] == full;
}
// Same unpublished-result proof for admission and the actual fit loan.
static bool latent_future_root(const Program &p, std::size_t fit, std::size_t root,
                               std::size_t cells, std::size_t capacity) {
  if (root>=p.roots.size() || capacity<cells) return false;
  const auto id=p.roots[root]; const auto &node=p.nodes[id];
  if (id<=fit || node.kind!=NodeKind::operation || node.storage!=StorageKind::numeric ||
      node.opcode!=std::uint16_t(ops::Op::scatter_rows) ||
      p.execution_metadata->root_destinations[id]!=std::int32_t(root) ||
      p.root_outputs[root].dtype!=OutputDType::float64 || p.root_outputs[root].rank!=2) return false;
  for (std::size_t writer=0;writer<=fit;++writer)
    if (p.execution_metadata->root_destinations[writer]==std::int32_t(root)) return false;
  return true;
}
static bool gaussian_stream_retains_hmm_history(const Program &,
                                               const std::vector<ops::Shape> &);
bool gaussian_filter_stream_geometry(const Program &program, std::size_t filter,
                                     const std::vector<ops::Shape> &shapes) {
  if (!gaussian_filter_stream_candidate(program,filter,shapes)) return false;
  const auto source=std::size_t(program.execution_metadata->gaussian_filter_source[filter]);
  // Preserve the already-qualified small-state route. New shapes also keep
  // the existing three-history fit loan, instead of creating per-EM replay.
  if (shapes[filter].dim[1]<=4 && shapes[program.nodes[source].parents[0]].dim[1]<=4) return true;
  return gaussian_stream_retains_hmm_history(program,shapes);
}
static bool gaussian_stream_producer_geometry(const Program &program, std::size_t source,
                                              const std::vector<ops::Shape> &shapes) {
  const auto &m = *program.execution_metadata;
  return !m.gaussian_filter_consumer.empty() && m.gaussian_filter_consumer[source] >= 0 &&
      gaussian_filter_stream_geometry(program,
          static_cast<std::size_t>(m.gaussian_filter_consumer[source]), shapes);
}
static bool filter_tail_geometry(const Program &program, std::size_t source,
                                 const std::vector<ops::Shape> &shapes) {
  return filter_last_row_geometry(program, source, shapes) && shapes[source].size() >= 1024 * 1024;
}

static bool state_decision_geometry(const Program &p, std::size_t id,
                                    const std::vector<ops::Shape> &shapes) {
  const auto &m=*p.execution_metadata;
  if (m.state_decision_members.empty() || m.state_decision_members[id]<0 || shapes.empty()) return false;
  const auto &region=m.state_decisions[m.state_decision_members[id]];
  const auto &shape=shapes[region.source];
  if (shape.rank!=1 || !(shapes[region.valid]==shape) ||
      m.root_destinations[region.steps.back().selection]<0) return false;
  for (const auto &step:region.steps) {
    if (!(shapes[step.comparison]==shape) || !(shapes[step.selection]==shape)) return false;
    const auto &comparison=p.nodes[step.comparison], &selection=p.nodes[step.selection];
    for (std::size_t j=0;j<2;++j)
      if (comparison.parents[j]!=region.source && shapes[comparison.parents[j]].rank!=0) return false;
    for (std::size_t j=1;j<=2;++j)
      if (static_cast<std::int8_t>(j)!=step.previous_argument && shapes[selection.parents[j]].rank!=0) return false;
  }
  return true;
}

// A per-call descriptor; old result generations and caller inputs are never donors.
struct StateRootLoan {
  std::int32_t source = -1;
  std::uint32_t terminal = 0;
  std::int64_t *data = nullptr;
  std::size_t count = 0, completed = 0;
  bool active = false;
  ~StateRootLoan() {
    if (active && data && completed < count)
      std::memset(data + completed, 0, (count - completed) * sizeof(std::int64_t));
  }
};

static StateRootLoan prove_state_root_loan(const Program &p,
    const std::vector<ops::Value> &inputs, const std::vector<ops::Shape> &shapes,
    const ResultLayout &layout, std::size_t result_row, void *output) {
  StateRootLoan proof;
  if (shapes.size() != p.nodes.size() || layout.columns != p.roots.size() ||
      layout.row_bytes.size() < 2 || result_row >= layout.row_bytes.size() - 1 ||
      layout.columns > layout.slots.size() / (result_row + 1)) return proof;
  const auto address = reinterpret_cast<std::uintptr_t>(output);
  if (!address || layout.bytes() > UINTPTR_MAX - address) return proof;
  const auto &m = *p.execution_metadata;
  for (const auto &region : m.state_decisions) {
    if (!region.loan_source || !state_decision_geometry(p, region.steps.back().selection, shapes)) continue;
    const auto source = region.source, terminal = region.steps.back().selection;
    const auto &shape = shapes[source], &matrix = shapes[p.nodes[source].parents[0]];
    if (shape.rank != 1 || shape.dim[0] < state_root_loan_min_elements || matrix.rank != 2 ||
        matrix.dim[0] != shape.dim[0] || !matrix.dim[1] || !(shapes[terminal] == shape)) continue;
    // Arena slots may also hold an earlier or later unrelated integer writer.
    // Prove a real capacity reduction, not just omission of this one node.
    std::size_t other_extent = 0;
    bool known = true;
    for (std::size_t id = 0; id < p.nodes.size(); ++id) {
      const auto &node = p.nodes[id];
      if (id == source || node.storage != StorageKind::integer || node.slot != p.nodes[source].slot ||
          state_decision_geometry(p, id, shapes) ||
          (m.root_destinations[id] >= 0 && node.kind == NodeKind::operation)) continue;
      if (shapes[id].rank < 0) { known = false; break; }
      other_extent = std::max(other_extent, shapes[id].size());
    }
    if (!known || other_extent > shape.size() ||
        shape.size() - other_extent < state_root_loan_min_elements) continue;
    const auto root = static_cast<std::size_t>(m.root_destinations[terminal]);
    const auto &slot = layout.slots[result_row * layout.columns + root];
    const auto row_first = layout.row_bytes[result_row], row_last = layout.row_bytes[result_row + 1];
    if (slot.byte_offset < row_first || slot.byte_offset > row_last ||
        slot.capacity < shape.size() || slot.capacity > (row_last - slot.byte_offset) / 8 ||
        slot.byte_offset % alignof(std::int64_t)) continue;
    const auto first = address + slot.byte_offset, last = first + shape.size() * 8;
    bool safe = true;
    for (std::size_t other = 0; safe && other < layout.columns; ++other) {
      if (other == root) continue;
      const auto &o = layout.slots[result_row * layout.columns + other];
      const auto width = output_itemsize(p.root_outputs[other].dtype);
      if (o.byte_offset < row_first || o.byte_offset > row_last ||
          o.capacity > (row_last - o.byte_offset) / width) { safe = false; break; }
      const auto lo = address + o.byte_offset, hi = lo + o.capacity * width;
      if (o.capacity && first < hi && lo < last) safe = false;
    }
    // Bounding spans are conservative for reverse, zero and irregular strides.
    for (const auto &input : inputs) {
      if (!safe || !input.data || !input.size()) continue;
      if (input.kind != ops::Kind::number && input.kind != ops::Kind::integer &&
          input.kind != ops::Kind::mask) { safe = false; break; }
      const auto width = input.kind == ops::Kind::mask ? 1u : 8u;
      std::size_t lower = 0, upper = 0;
      for (int axis = 0; safe && axis < input.shape.rank; ++axis) {
        const auto stride = input.stride[axis];
        if (stride == PTRDIFF_MIN) { safe = false; break; }
        const auto magnitude = static_cast<std::size_t>(stride < 0 ? -stride : stride);
        const auto extent = input.shape.dim[axis] - 1;
        if (extent && magnitude > SIZE_MAX / extent) { safe = false; break; }
        const auto amount = magnitude * extent;
        auto &bound = stride < 0 ? lower : upper;
        if (amount > SIZE_MAX - bound) { safe = false; break; }
        bound += amount;
      }
      if (!safe) break;
      const auto start = reinterpret_cast<std::uintptr_t>(input.data);
      if (lower > start / width || upper >= (UINTPTR_MAX - start) / width) { safe = false; break; }
      const auto lo = start - lower * width, hi = start + (upper + 1) * width;
      if (first < hi && lo < last) safe = false;
    }
    if (!safe) continue;
    proof.source = static_cast<std::int32_t>(source); proof.terminal = terminal;
    proof.data = reinterpret_cast<std::int64_t *>(first); proof.count = shape.size();
    break;
  }
  return proof;
}

template <bool LoanSource>
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
static void execute_state_decision(const Program &p, const StateDecisionRegion &region,
    const std::vector<ops::Value> &values, ops::Output &out, ops::Workspace &workspace,
    Audit &audit, std::chrono::steady_clock::time_point deadline, const std::atomic<bool> *cancelled,
    StateRootLoan *loan, std::size_t &copy_bytes) {
  std::array<std::uint8_t,state_decision_tile_elements> mask;
  std::array<std::int64_t, LoanSource ? state_decision_tile_elements : 0> original;
  const auto slice=[](ops::Value v,std::size_t first,std::size_t count) {
    if (v.shape.rank) {
      const auto offset=static_cast<std::ptrdiff_t>(first)*v.stride[0];
      if (offset) {
        if (v.kind==ops::Kind::integer) v.data=static_cast<const std::int64_t *>(v.data)+offset;
        else if (v.kind==ops::Kind::mask) v.data=static_cast<const std::uint8_t *>(v.data)+offset;
        else v.data=static_cast<const double *>(v.data)+offset;
      }
      v.shape=ops::vector_shape(count);
    }
    return v;
  };
  for (std::size_t first=0;first<out.shape.size();first+=state_decision_tile_elements) {
    native::check_deadline(deadline);
    if (cancelled && cancelled->load()) throw native::Timeout("native state decision cancelled");
    const auto count=std::min(state_decision_tile_elements,out.shape.size()-first);
    if constexpr (LoanSource) {
      ops::require(loan && loan->active && loan->count == out.shape.size() &&
          loan->data == out.data && values[region.source].data == out.data, "STATE_ROOT_LOAN_CONTRACT");
      std::copy_n(loan->data + first, count, original.data());
      copy_bytes += count * sizeof(std::int64_t);
    }
    ops::Output target;
    target.kind=ops::Kind::integer; target.shape=ops::vector_shape(count);
    target.data=static_cast<std::int64_t *>(out.data)+first;
    for (const auto &step:region.steps) {
      const auto &comparison=p.nodes[step.comparison], &selection=p.nodes[step.selection];
      std::array<ops::Value,2> operands;
      for (std::size_t j=0;j<2;++j) {
        operands[j]=slice(values[comparison.parents[j]],first,count);
        if constexpr (LoanSource)
          if (comparison.parents[j] == region.source) {
            operands[j].data = original.data(); operands[j].set_contiguous_strides();
          }
      }
      const auto predicate=ops::prepare(*p.execution_metadata->specs[step.comparison],operands.data(),2);
      ops::Output predicate_out; predicate_out.kind=ops::Kind::mask;
      predicate_out.shape=ops::vector_shape(count); predicate_out.data=mask.data();
      ops::Audit operator_audit;
      ops::elementwise(predicate,predicate_out,workspace,ops::Isa::automatic,operator_audit);
      std::array<ops::Value,4> branches;
      branches[0].kind=ops::Kind::mask; branches[0].shape=target.shape;
      branches[0].data=mask.data(); branches[0].set_contiguous_strides();
      for (std::size_t j=1;j<=2;++j) {
        if (static_cast<std::int8_t>(j)==step.previous_argument) {
          branches[j].kind=ops::Kind::integer; branches[j].shape=target.shape;
          branches[j].data=target.data; branches[j].set_contiguous_strides();
        } else branches[j]=values[selection.parents[j]];
      }
      branches[3]=slice(values[region.valid],first,count);
      // Original full preparation validated states and validity at each logical
      // node. Tile descriptors need geometry only; canonical state writes read
      // the previous state at this same coordinate before overwriting it.
      auto prepared=ops::prepare_geometry(*p.execution_metadata->specs[step.selection],branches.data(),4);
      ops::state(prepared,target,workspace,operator_audit);
      if constexpr (LoanSource) loan->completed = first + count;
      audit.vector_elements+=operator_audit.vector_elements;
    }
    ++audit.pointwise_tiles;
  }
  ++audit.fused_pointwise_calls;
  audit.pointwise_workspace_capacity_bytes=std::max(audit.pointwise_workspace_capacity_bytes,mask.size() + (LoanSource ? original.size() * sizeof(std::int64_t) : 0));
}

static void plan_array_arena_mode(const Program &program, std::size_t capacity,
                     const std::vector<ops::Shape> &shapes, ArrayArenaLayout &layout,
                     bool fuse, bool prebound_inputs, bool summary_fusion,
                     bool private_where_sums, bool allow_matmul_chain,
                     const std::vector<std::uint8_t> *row_subviews, std::int32_t loaned_state_source,
                     bool unrestricted_gaussian) {
  ops::require(!row_subviews || row_subviews->size() == program.nodes.size(), "ROW_SUBVIEW_CONTRACT");
  const std::array<std::size_t,3> slots{program.numeric_slots,program.mask_slots,program.integer_slots};
  for (std::size_t kind=0;kind<3;++kind) layout.offsets[kind].assign(slots[kind],0);
  const auto &metadata=*program.execution_metadata;
  const auto gaussian_filter=[&](std::size_t i) {
    return unrestricted_gaussian ? gaussian_filter_stream_candidate(program,i,shapes)
                                 : gaussian_filter_stream_geometry(program,i,shapes);
  };
  const auto gaussian_producer=[&](std::size_t i) {
    return !metadata.gaussian_filter_consumer.empty() && metadata.gaussian_filter_consumer[i]>=0 &&
        gaussian_filter(std::size_t(metadata.gaussian_filter_consumer[i]));
  };
  const bool split_eigh=fuse && metadata.split_eigh_source>=0 && !shapes.empty() &&
      ops::symmetric_eigh_split_scratch_doubles(shapes[metadata.split_eigh_source].dim[0])!=0;
  for (std::size_t i=0;i<program.nodes.size();++i) {
    const auto &node=program.nodes[i];
    if (static_cast<std::int32_t>(i) == loaned_state_source ||
        (row_subviews && (*row_subviews)[i])) continue;
    if (node.storage==StorageKind::inline_value) continue;
    if (fuse && state_decision_geometry(program,i,shapes)) continue;
    if (fuse && allow_matmul_chain &&
        (!prebound_inputs || metadata.matmul_shared_chains.empty()) &&
        matmul_scalar_chain_geometry(program,shapes) &&
        matmul_scalar_chain_member(program,i)) continue;
    if (node.kind==NodeKind::operation &&
        node.opcode==static_cast<std::uint16_t>(ops::Op::eigenvalues) &&
        !(fuse && !program.execution_metadata->fused_from.empty() &&
          program.execution_metadata->fused_from[i]>=0)) continue;
    const bool direct=program.output_kind==OutputKind::typed &&
        program.execution_metadata->root_destinations[i]>=0 &&
        !cholesky_f_root_requires_arena(program, i, prebound_inputs) &&
        (node.kind==NodeKind::operation || (node.kind==NodeKind::apply_scope &&
          program.apply_scopes[node.input_index].kind==ApplyKind::iterate));
    const bool transposed_direct_root = program.roots.size() == 1 &&
        program.roots[0] == i && node.kind == NodeKind::operation &&
        node.parent_count == 1 &&
        node.opcode == static_cast<std::uint16_t>(ops::Op::transpose) &&
        program.execution_metadata->root_destinations[node.parents[0]] == 0;
    if ((fuse && summary_fusion && !prebound_inputs && matmul_summary_geometry(program,i,shapes)) ||
        direct || transposed_direct_root ||
        (split_eigh && static_cast<std::int32_t>(i)==metadata.split_eigh_source)) continue;
    auto &count=layout.offsets[static_cast<std::size_t>(node.storage)-1][node.slot];
    if (fuse && deferred_pair_geometry(program,i,shapes)) {
      count=std::max(count,deferred_broadcast_extent(program,i,shapes));
      continue;
    }
    if (private_where_sums && !prebound_inputs && square_reduce_geometry(program, i, shapes)) {
      const auto &region = metadata.square_reductions[metadata.square_members[i]];
      if (i == region.subtraction) count = std::max(count, shapes[region.reduction].size());
      if (i != region.reduction) continue;
    }
    if (private_where_sums && !prebound_inputs && scan_reduce_geometry(program, i, shapes)) {
      const auto &region = metadata.scan_reductions[metadata.scan_members[i]];
      if (i == region.nav) count = std::max(count, shapes[region.reduction].size());
      // No other writer can occupy the cache's slot in the continuous closure.
      if (i != region.reduction) continue;
    }
    if (private_where_sums && !prebound_inputs && where_sum_geometry(program, i, shapes)) {
      count = std::max(count, shapes[i].dim[0]);
      continue;
    }
    if (fuse && gaussian_producer(i)) continue;
    if (fuse && (filter_tail_geometry(program, i, shapes) || gaussian_filter(i))) {
      count = std::max(count, shapes[i].dim[1]);
      continue;
    }
    count=std::max(count,shapes.empty() ? capacity : shapes[i].size());
  }
  for (std::size_t kind=0;kind<3;++kind) {
    std::size_t total=0;
    for (auto &count:layout.offsets[kind]) {
      const auto size=count;
      const auto limit=static_cast<std::size_t>(PTRDIFF_MAX)/(kind==1 ? 1 : 8);
      ops::require(size<=limit-total,"GRAPH_ARENA_OVERFLOW");
      count=total; total+=size;
    }
    layout.elements[kind]=total;
  }
}

void plan_array_arena(const Program &program, std::size_t capacity,
                     const std::vector<ops::Shape> &shapes, ArrayArenaLayout &layout,
                     bool fuse, bool prebound_inputs, bool summary_fusion,
                     bool private_where_sums, bool allow_matmul_chain,
                     const std::vector<std::uint8_t> *row_subviews, std::int32_t loaned_state_source) {
  plan_array_arena_mode(program,capacity,shapes,layout,fuse,prebound_inputs,summary_fusion,
                       private_where_sums,allow_matmul_chain,row_subviews,loaned_state_source,false);
}

static bool gaussian_stream_retains_hmm_history(const Program &p,
                                               const std::vector<ops::Shape> &shapes) {
  // These calls cannot loan a full history in the first place.
  if (p.output_kind!=OutputKind::typed || p.isolate_errors ||
      !p.apply_scopes.empty() || !p.rolling_scopes.empty()) return true;
  std::vector<std::size_t> fits;
  for (std::size_t i=0;i<p.nodes.size();++i) {
    const auto &node=p.nodes[i];
    if (node.kind==NodeKind::operation && node.opcode==std::uint16_t(ops::Op::gaussian_hmm_em) &&
        ops::matrix_shape(shapes[node.parents[0]].dim[0],shapes[node.parents[1]].size()).size()>=131072) fits.push_back(i);
  }
  if (fits.empty()) return true;
  // A lower bound: admit every possible take-axis view, and every existing
  // elision plus all widened Gaussian streams. Declining an elision can only
  // increase numeric slot maxima. Use the one arena planner, not a second model.
  std::vector<std::uint8_t> possible_views(p.nodes.size(),0);
  for (const auto id:p.execution_metadata->row_subview_takes) possible_views[id]=1;
  for (const auto &reuse:p.execution_metadata->row_subview_reuses) possible_views[reuse.selected]=1;
  ArrayArenaLayout compact;
  plan_array_arena_mode(p,0,shapes,compact,true,false,true,true,true,
                       &possible_views,-1,true);
  for (const auto i:fits) {
    const auto &node=p.nodes[i];
    const auto cells=ops::matrix_shape(shapes[node.parents[0]].dim[0],shapes[node.parents[1]].size()).size();
    bool future=false;
    for (std::size_t root=0;root<p.roots.size();++root)
      future=future || latent_future_root(p,i,root,cells,shapes[p.roots[root]].size());
    if (compact.elements[0]/cells < (future ? 2u : 3u)) return false;
  }
  return true;
}

static void accumulate_child(Scratch &scratch, const Audit &audit) {
  scratch.highway_vqsort_calls += audit.highway_vqsort_calls;
  scratch.eigen_rotation_calls += audit.eigen_rotation_calls;
  scratch.eigen_matvec_calls += audit.eigen_matvec_calls;
  scratch.eigen_covariance_calls += audit.eigen_covariance_calls;
  scratch.eigen_density_calls += audit.eigen_density_calls;
  scratch.gaussian_filter_streams += audit.gaussian_filter_streams;
  scratch.accelerate_covchol_calls += audit.accelerate_covchol_calls;
  scratch.accelerate_matmul_calls += audit.accelerate_matmul_calls;
  scratch.matmul_zero_calls += audit.matmul_zero_calls;
  scratch.accelerate_covariance_attempts += audit.accelerate_covariance_attempts;
  scratch.accelerate_covariance_calls += audit.accelerate_covariance_calls;
  scratch.covariance_syrk_calls += audit.covariance_syrk_calls;
  scratch.covariance_replayed_cells += audit.covariance_replayed_cells;
  scratch.covariance_fallbacks += audit.covariance_fallbacks;
  scratch.covariance_zero_calls += audit.covariance_zero_calls;
  scratch.accelerate_vforce_calls += audit.accelerate_vforce_calls;
  scratch.accelerate_lu_attempts += audit.accelerate_lu_attempts;
  scratch.accelerate_lu_calls += audit.accelerate_lu_calls;
  scratch.accelerate_lu_fallbacks += audit.accelerate_lu_fallbacks;
  scratch.solve_residual_materialization_bytes += audit.solve_residual_materialization_bytes;
  scratch.covariance_panel_materialization_bytes += audit.covariance_panel_materialization_bytes;
  scratch.native_library_workspace_reserve_bytes = std::max(
      scratch.native_library_workspace_reserve_bytes, audit.native_library_workspace_reserve_bytes);
  scratch.algorithm_copy_bytes += audit.algorithm_copy_bytes;
  scratch.state_copy_bytes += audit.state_copy_bytes;
  scratch.child_root_copy_bytes += audit.root_copy_bytes;
  scratch.child_direct_output_bytes += audit.direct_output_bytes;
  scratch.child_vector_elements += audit.vector_elements;
  scratch.child_pointwise_tiles += audit.pointwise_tiles;
  scratch.child_fused_pointwise_calls += audit.fused_pointwise_calls;
  scratch.child_pointwise_capacity = std::max(scratch.child_pointwise_capacity,
      audit.pointwise_workspace_capacity_bytes);
}
static void accumulate_child(Scratch &scratch, const ops::Audit &audit) {
  scratch.highway_vqsort_calls += audit.highway_vqsort_calls;
  scratch.eigen_rotation_calls += audit.eigen_rotation_calls;
  scratch.eigen_matvec_calls += audit.eigen_matvec_calls;
  scratch.eigen_covariance_calls += audit.eigen_covariance_calls;
  scratch.eigen_density_calls += audit.eigen_density_calls;
  scratch.accelerate_covchol_calls += audit.accelerate_covchol_calls;
  scratch.accelerate_matmul_calls += audit.accelerate_matmul_calls;
  scratch.matmul_zero_calls += audit.matmul_zero_calls;
  scratch.accelerate_covariance_attempts += audit.accelerate_covariance_attempts;
  scratch.accelerate_covariance_calls += audit.accelerate_covariance_calls;
  scratch.covariance_syrk_calls += audit.covariance_syrk_calls;
  scratch.covariance_replayed_cells += audit.covariance_replayed_cells;
  scratch.covariance_fallbacks += audit.covariance_fallbacks;
  scratch.covariance_zero_calls += audit.covariance_zero_calls;
  scratch.accelerate_vforce_calls += audit.accelerate_vforce_calls;
  scratch.accelerate_lu_attempts += audit.accelerate_lu_attempts;
  scratch.accelerate_lu_calls += audit.accelerate_lu_calls;
  scratch.accelerate_lu_fallbacks += audit.accelerate_lu_fallbacks;
  scratch.solve_residual_materialization_bytes += audit.solve_residual_materialization_bytes;
  scratch.covariance_panel_materialization_bytes += audit.covariance_panel_materialization_bytes;
  scratch.native_library_workspace_reserve_bytes = std::max(
      scratch.native_library_workspace_reserve_bytes, audit.native_library_workspace_reserve_bytes);
  scratch.algorithm_copy_bytes += audit.algorithm_copy_bytes;
  scratch.child_vector_elements += audit.vector_elements;
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
                          bool interval_known = true,
                          bool propagate_scope_errors = false,
                          const ResultLayout *layout = nullptr, std::size_t result_row = 0,
                          const std::vector<const char *> *deferred_errors = nullptr);

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

// A strict child of an isolating scope may fail before visiting an independent
// structural constraint. Audit the same real slice with failure tracking before
// propagating that numerical exception; never invent membership or publish the
// audit's result. Healthy and standalone strict executions keep their path.
static Audit execute_validated_scope_body(const Program &body,
    const std::vector<ops::Value> &inputs, const double *parameters,
    std::size_t parameter_count, const std::int64_t *starts,
    const std::int64_t *ends, void *output, Scratch &child,
    const std::vector<std::int16_t> *input_failures, bool interval_known,
    bool propagate_scope_errors, std::size_t body_node_count) {
  try {
    return execute_impl(body, inputs, parameters, parameter_count, starts, ends,
        1, output, 1, child, true, input_failures, interval_known,
        propagate_scope_errors);
  } catch (const ops::Error &error) {
    if (!propagate_scope_errors || input_failures || body.isolate_errors ||
        !numerical_error(error)) throw;
    native::check_deadline(child.deadline);
    if (child.cancelled && child.cancelled->load())
      throw native::Timeout("native scope validation cancelled");
    const auto rows = std::max<std::size_t>(ends[0] - starts[0], 1);
    const auto nodes = std::max<std::size_t>(body_node_count, 1);
    if (child.remaining_work) {
      ops::require(rows <= *child.remaining_work / nodes,
                   "SCOPE_COMPUTE_BUDGET_EXCEEDED");
      *child.remaining_work -= rows * nodes;
    }
    std::vector<std::int16_t> failures(inputs.size(), 0);
    execute_impl(body, inputs, parameters, parameter_count, starts, ends,
        1, output, 1, child, true, &failures, interval_known, true);
    throw;
  }
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
    ops::require(scope.kind == ApplyKind::iterate || value.shape.rank == 1, "SCOPE_CAPTURE_RANK");
    if (count_known && scope.kind != ApplyKind::bisect && scope.kind != ApplyKind::iterate)
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

  if (scope.kind == ApplyKind::iterate) {
    const auto &initial = scratch.values[scope.argument_nodes[0]];
    ops::require(initial.kind == ops::Kind::number, "ITERATION_STATE_TYPE");
    const auto tol = scope.argument_nodes[1], limit = scope.argument_nodes[2];
    if (available(tol)) ops::require(std::isfinite(scratch.values[tol].scalar) &&
        scratch.values[tol].scalar >= 0, "INVALID_PARAMETER");
    if (available(limit)) positive_integer(scratch.values[limit].scalar, "INVALID_PARAMETER", 10000);
    auto result = initial; result.data = nullptr; result.scalar = NAN;
    return result;
  }

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
  const bool iterate = !rolling && program.apply_scopes[node.input_index].kind == ApplyKind::iterate;
  bool body_interval_known = bisect || iterate;
  if (bisect || iterate) {
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
    if (!bisect && !iterate) {
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
  child.deadline = scratch.deadline; child.cancelled = scratch.cancelled;
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
    std::vector<double> buffer;
    ResultLayout layout;
    if (iterate) {
      std::vector<ops::Shape> shapes;
      required_array_capacity(body, inputs, length, &shapes);
      const auto capacity = shapes.at(body.roots[0]).size();
      buffer.resize(std::max<std::size_t>(capacity, 1));
      layout.columns = 1; layout.slots.push_back({0, 0, capacity});
      layout.row_bytes = {0, capacity * sizeof(double)};
      layout.row_statuses = {0, capacity};
    }
    const auto audit = execute_impl(body, inputs, parameters.data(), parameters.size(),
        &start, &end, 1, iterate ? buffer.data() : &ignored, 1, child, true, &failures, body_interval_known,
        scratch.isolate_errors || scratch.propagate_scope_errors, iterate ? &layout : nullptr);
    accumulate_child(scratch, audit);
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
  if (node.kind == NodeKind::iteration_projection) {
    auto value = ops::Value::number(NAN);
    if (node.opcode != 2) value.kind = ops::Kind::integer;
    return value;
  }
  ops::require(node.kind == NodeKind::operation, "GRAPH_FAILED_NODE_KIND");
  ops::Arguments<ops::Value> args(node.parent_count);
  std::uint32_t geometry = 0, payload = 0;
  for (std::size_t i = 0; i < node.parent_count; ++i) {
    const auto id = node.parents[i];
    args[i] = scratch.values[id];
    if (known_geometry(args[i])) geometry |= static_cast<std::uint32_t>(1u << i);
    if (available_payload(scratch, id)) payload |= static_cast<std::uint32_t>(1u << i);
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
    if (scope.kind == ApplyKind::iterate) return false;
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
        accumulate_child(scratch, audit);
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
    std::size_t interval_length, Scratch &scratch) {
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
      scratch.numeric_slot(node.slot);
  result.data = destination;
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  std::fill(destination, destination + interval_length, nan);

  auto &child = *scratch.children[node.input_index];
  child.remaining_work = scratch.remaining_work;
  child.deadline = scratch.deadline; child.cancelled = scratch.cancelled;
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
      execute_validated_scope_body(*scope.body, body_inputs,
                   body_parameters.empty() ? nullptr : body_parameters.data(),
                   body_parameters.size(), &local_start, &local_end, &value,
                   child, inherited ? &failures : nullptr, true,
                   scratch.isolate_errors || scratch.propagate_scope_errors,
                   scope.body_node_count);
      if (!inherited && std::isfinite(value))
        destination[right] = value;
    } catch (const ops::Error &error) {
      if (!numerical_error(error)) throw;
      // Preserve the existing standalone strict rolling NaN policy, but never
      // swallow a nested failure before an isolating outer scope can record it.
      if (!scratch.isolate_errors && (scratch.propagate_scope_errors ||
          scope.body->execution_metadata->contains_segment_scope)) throw;
      destination[right] = nan;
      if (scratch.isolate_errors)
        mark_positions(scratch, node_index, interval_length, right, right + 1, 4);
    }
  }
  return result;
}

static ops::Value execute_iteration(const Program &program, const ApplyScope &scope,
    const Node &node, Scratch &scratch, void *root_destination) {
  auto &child = *scratch.children[program.rolling_scopes.size() + node.input_index];
  child.remaining_work = scratch.remaining_work;
  child.deadline = scratch.deadline; child.cancelled = scratch.cancelled;
  const auto &initial = scratch.values[scope.argument_nodes[0]];
  const auto count = initial.size();
  const auto storage = std::max<std::size_t>(count, 1);
  auto &alternate = child.iteration_next;
  double scalar_destination = 0;
  auto *destination = initial.shape.rank ? (root_destination ? static_cast<double *>(root_destination)
      : scratch.numeric_slot(node.slot)) : &scalar_destination;
  bool finite = true;
  auto state = initial;
  state.set_contiguous_strides();
  auto &inputs = child.iteration_inputs;
  inputs.clear();
  inputs.reserve(scope.input_nodes.size());
  std::size_t extent = std::max<std::size_t>(count, 1);
  for (auto id : scope.input_nodes) {
    inputs.push_back(scratch.values[id]);
    extent = std::max(extent, inputs.back().size());
  }
  auto &parameters = child.iteration_parameters;
  parameters.clear();
  parameters.reserve(scope.parameter_nodes.size());
  for (auto id : scope.parameter_nodes) parameters.push_back(scratch.values[id].scalar);
  if (scope.state_input_index >= 0) inputs[scope.state_input_index] = initial;
  const bool seed_inplace = !value_observer && pointwise_iteration_inplace_safe(*scope.body,inputs,
      parameters.data(),static_cast<std::size_t>(scope.state_input_index));
  for (std::size_t i = 0; i < count; ++i) {
    const auto value = initial.f(i);
    finite = finite && std::isfinite(value);
    if (seed_inplace) destination[i] = value;
  }
  if (seed_inplace) scratch.state_copy_bytes += count*sizeof(double);
  const bool inplace = seed_inplace && finite;
  const double *current = static_cast<const double *>(initial.data);
  if (inplace) {
    // Finite input and the checked coefficient domain prove every update finite.
    // Copy only into our independent result, then scan each old value before
    // overwriting it in the register kernel. Generic solvers retain two states.
    current = destination;
    if (alternate.capacity()) std::vector<double>().swap(alternate);
    if (child.iteration_state.capacity()) std::vector<double>().swap(child.iteration_state);
  } else {
    alternate.resize(storage);
    const bool copy_initial = !initial.shape.rank || !initial.contiguous() || current == destination;
    if (copy_initial) {
      child.iteration_state.resize(storage);
      for (std::size_t i = 0; i < count; ++i) child.iteration_state[i] = initial.f(i);
      current = child.iteration_state.data();
      scratch.state_copy_bytes += count*sizeof(double);
    } else if (child.iteration_state.capacity()) std::vector<double>().swap(child.iteration_state);
  }
  auto *next = destination;
  const auto limit = positive_integer(scratch.values[scope.argument_nodes[2]].scalar, "INVALID_PARAMETER", 10000);
  const auto tolerance = scratch.values[scope.argument_nodes[1]].scalar;
  auto &diagnostics = scratch.iteration_info[node.input_index];
  diagnostics = {static_cast<double>(finite ? SolverStatus::iteration_limit : SolverStatus::numerical_failure), 0, INFINITY};
  auto &layout = child.iteration_layout;
  layout.columns = 1; layout.slots.resize(1); layout.slots[0] = {0, 0, count};
  layout.row_bytes = {0, count * sizeof(double)}; layout.row_statuses = {0, count};
  const std::int64_t begin = 0, end = static_cast<std::int64_t>(extent);
  const auto charge_work = [&] {
    native::check_deadline(scratch.deadline);
    if (scratch.cancelled && scratch.cancelled->load()) throw native::Timeout("native iteration cancelled");
    const auto nodes = std::max<std::size_t>(scope.body_node_count, 1);
    if (child.remaining_work) {
      ops::require(extent <= *child.remaining_work / nodes, "SCOPE_COMPUTE_BUDGET_EXCEEDED");
      *child.remaining_work -= extent * nodes;
    }
  };
  const auto validate_after_failure = [&](bool failed_initial) {
    // A numerical stop must not hide an independent structural error in a
    // later body node. Supplying the failure vector enables validation across
    // numerical failures without publishing or accepting its candidate state.
    charge_work();
    std::vector<std::int16_t> failures(inputs.size(), 0);
    if (failed_initial && scope.state_input_index >= 0)
      failures[scope.state_input_index] = 4;
    const auto audit = execute_impl(*scope.body, inputs, parameters.data(), parameters.size(),
        &begin, &end, 1, next, 1, child, true, &failures, true, true, &layout);
    accumulate_child(scratch, audit);
  };
  state.data = current;
  if (!state.shape.rank) state.scalar = current[0];
  if (scope.state_input_index >= 0) inputs[scope.state_input_index] = state;
  if (!finite) validate_after_failure(true);
  InplacePointwiseChain repeated_chain;
  Audit iteration_audit;
  for (std::size_t iteration = 0; finite && iteration < limit; ++iteration) {
    charge_work();
    state.data = current;
    if (!state.shape.rank) state.scalar = current[0];
    if (scope.state_input_index >= 0) inputs[scope.state_input_index] = state;
    diagnostics[1] = static_cast<double>(iteration + 1);
    double residual = 0;
    bool scanned = false;
    try {
      if (repeated_chain.kernel) {
        execute_inplace_pointwise_chain(repeated_chain,iteration_audit,residual,finite,scratch.deadline,scratch.cancelled);
        scanned = true;
      } else if (!value_observer) scanned = execute_pointwise(*scope.body, inputs, parameters.data(), parameters.size(),
          &begin, &end, 1, next, &layout, 0, iteration_audit, scratch.deadline, scratch.cancelled,
          true, 0, SIZE_MAX, current, &residual, &finite,inplace ? &repeated_chain : nullptr);
      if (!scanned) iteration_audit = execute_impl(*scope.body, inputs, parameters.data(), parameters.size(),
          &begin, &end, 1, next, 1, child, true, nullptr, true, true, &layout);
      accumulate_child(scratch, iteration_audit);
      ops::require(iteration_audit.result_shapes[0] == initial.shape, "ITERATION_STATE_SHAPE");
      finite = finite && iteration_audit.root_statuses[0] == 0;
    } catch (const ops::Error &error) {
      if (!numerical_error(error)) throw;
      validate_after_failure(false);
      finite = false;
    }
    if (!scanned && finite)
      scratch.child_vector_elements += ops::iteration_residual(current, next, count, residual, finite);
    if (!finite) {
      diagnostics[0] = static_cast<double>(SolverStatus::numerical_failure);
      diagnostics[2] = INFINITY;
      break; // Keep the last finite state; the invalid candidate is not published.
    }
    current = next;
    if (!inplace) next = next == destination ? alternate.data() : destination;
    diagnostics[2] = residual;
    if (residual <= tolerance) {
      diagnostics[0] = static_cast<double>(SolverStatus::converged);
      break;
    }
  }
  auto result = initial;
  result.set_contiguous_strides();
  if (!result.shape.rank) {
    result.scalar = current[0];
    // Computed scalars live inline. Retaining the initial array pointer would
    // refresh a downstream captured input from the old value instead.
    result.data = nullptr;
  }
  else {
    if (current != destination && count) {
      std::copy_n(current, count, destination);
      scratch.state_copy_bytes += count * sizeof(double);
    }
    result.data = destination;
  }
  return result;
}

static ops::Value execute_apply_scope(const Program &program, const ApplyScope &scope,
    const Node &node, std::size_t node_index, std::size_t interval_length,
    std::size_t max_window, Scratch &scratch, void *root_destination) {
  const bool trace = scratch.isolate_errors &&
      program.execution_metadata->position_status_reachable[node_index];
  const auto structure = validate_apply_structure(scope, interval_length, scratch, scratch.isolate_errors);
  if (scope.kind == ApplyKind::iterate) return execute_iteration(program, scope, node, scratch, root_destination);
  auto &child = *scratch.children[program.rolling_scopes.size() + node.input_index];
  child.remaining_work = scratch.remaining_work;
  child.deadline = scratch.deadline; child.cancelled = scratch.cancelled;
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
    const auto audit = execute_validated_scope_body(*scope.body, inputs, parameters.data(), parameters.size(),
        &begin, &end, &value, child, failures,
        scope.kind != ApplyKind::bisect || std::all_of(inputs.begin(), inputs.end(), known_geometry),
        scratch.isolate_errors || scratch.propagate_scope_errors, scope.body_node_count);
    accumulate_child(scratch, audit);
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
  auto *destination = scratch.numeric_slot(node.slot);
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
      failure = 4;
    }
    for (std::size_t i = first; i < last; ++i) {
      destination[indices[i]] = value;
      if (failure) mark_positions(scratch, node_index, count, indices[i], indices[i] + 1, failure);
    }
    first = last;
  }
  return result;
}

static bool reuse_covariance_root(const Program &program, std::size_t node_index,
    std::size_t result_row, void *output, const ResultLayout &layout,
    ops::Output &native_output, Scratch &scratch) {
  const auto &node = program.nodes[node_index];
  const auto &destinations = program.execution_metadata->root_destinations;
  const auto elements = native_output.shape.size();
  if (elements > static_cast<std::size_t>(PTRDIFF_MAX) / sizeof(double)) return false;
  const auto bytes = elements * sizeof(double);
  const auto slot_base = result_row * program.roots.size();
  const auto &destination = layout.slots[slot_base + destinations[node_index]];
  for (const auto candidate : program.roots) {
    // A lower node id has already completed in this interval. Direct roots
    // retain their own result storage even when intermediate arena slots die.
    if (candidate >= node_index || destinations[candidate] < 0) continue;
    const auto &prior = program.nodes[candidate];
    if (prior.kind != NodeKind::operation || prior.opcode != static_cast<std::uint16_t>(ops::Op::covariance) ||
        prior.parent_count != 1 || prior.parents[0] != node.parents[0]) continue;
    if (scratch.isolate_errors && (scratch.statuses[candidate] || position_error(scratch, candidate))) continue;
    const auto root = static_cast<std::size_t>(destinations[candidate]);
    const auto &schema = program.root_outputs[root];
    const auto &slot = layout.slots[slot_base + root];
    const auto &value = scratch.values[candidate];
    if (schema.dtype != OutputDType::float64 || schema.rank != 2 ||
        slot.capacity < elements || value.kind != ops::Kind::number ||
        !(value.shape == native_output.shape) || !value.dense_contiguous() || !value.data ||
        value.data != static_cast<std::uint8_t *>(output) + slot.byte_offset) continue;
    const auto distance = slot.byte_offset < destination.byte_offset
        ? destination.byte_offset - slot.byte_offset : slot.byte_offset - destination.byte_offset;
    if (distance < bytes) continue;
    const auto *covariance = static_cast<const double *>(value.data);
    if (!std::all_of(covariance, covariance + elements, [](double v) { return std::isfinite(v); })) continue;
    // Completed covariance is bitwise symmetric: C/F bytes represent the same
    // matrix. Preserve the zero-scratch lifecycle. Count the independent copy
    // before normalization, which can fail for a zero variance column.
    scratch.workspace.ensure(0, 0);
    std::memcpy(native_output.data, covariance, bytes);
    scratch.algorithm_copy_bytes += bytes;
    ops::finish_covariance_matrix(native_output, true);
    return true;
  }
  return false;
}

static bool reuse_private_solve_pair(const Program &program, const ops::Prepared &first,
    const ops::Prepared &second, ops::Output &out1, ops::Output &out2, Scratch &scratch) {
  const auto &metadata=*program.execution_metadata;
  const auto source=metadata.solve_pair_private_source;
  const auto &a=first.args[0];
  const auto n=a.shape.dim[0], k1=first.args[1].shape.dim[1], k2=second.args[1].shape.dim[1];
  if (a.shape.rank!=2 || a.shape.dim[1]!=n || !k1 || !k2 || k1>64 || k2>64-k1 ||
      !a.dense_contiguous() || !ops::solve_accelerate_scratch_doubles(n,k1+k2)) return false;
  const auto &producer=program.nodes[source];
  auto *coefficients=scratch.numeric_slot(producer.slot);
  if (!coefficients || a.data!=coefficients || a.size()>scratch.slot_capacity(StorageKind::numeric,producer.slot))
    return false;
  const auto begin=reinterpret_cast<std::uintptr_t>(coefficients), bytes=a.size()*sizeof(double);
  const auto overlaps=[&](std::uintptr_t address,std::size_t count) {
    return count && (address<=begin ? begin-address<count : address-begin<bytes);
  };
  if (overlaps(reinterpret_cast<std::uintptr_t>(out1.data),out1.shape.size()*sizeof(double)) ||
      overlaps(reinterpret_cast<std::uintptr_t>(out2.data),out2.shape.size()*sizeof(double))) return false;
  for (const auto *rhs : {&first.args[1],&second.args[1]}) {
    std::size_t lower=0,upper=0;
    const auto limit=static_cast<std::size_t>(PTRDIFF_MAX)/sizeof(double);
    for (int axis=0;axis<rhs->shape.rank;++axis) {
      const auto stride=rhs->stride[axis];
      if (stride==PTRDIFF_MIN) return false;
      const auto magnitude=static_cast<std::size_t>(stride<0 ? -stride : stride);
      const auto steps=rhs->shape.dim[axis]-1;
      if (magnitude && steps>limit/magnitude) return false;
      const auto distance=steps*magnitude;
      auto &bound=stride<0 ? lower : upper;
      if (distance>limit-bound) return false;
      bound+=distance;
    }
    const auto address=reinterpret_cast<std::uintptr_t>(rhs->data);
    if (lower>address/sizeof(double) || upper>=limit-lower ||
        overlaps(address-lower*sizeof(double),(lower+upper+1)*sizeof(double))) return false;
  }
  ops::Arguments<ops::Value> arguments(producer.parent_count);
  for (std::size_t j=0;j<producer.parent_count;++j) arguments[j]=scratch.values[producer.parents[j]];
  const auto producer_plan=ops::prepare(*metadata.specs[source],arguments.data(),producer.parent_count);
  ops::Audit kernel;
  const bool completed=ops::try_solve_accelerate_private_pair(first,second,producer_plan,
      out1,out2,scratch.workspace,kernel,coefficients);
  accumulate_child(scratch,kernel); // Retain attempted-library costs on fallback too.
  if (!completed && kernel.accelerate_lu_attempts) {
    // Keep the prepared arguments' original physical strides on replay.
    ops::Output restored;
    restored.kind=producer_plan.output_kind; restored.shape=producer_plan.output_shape;
    restored.data=coefficients; restored.order=a.contiguous(ops::MemoryOrder::f)
        ? ops::MemoryOrder::f : ops::MemoryOrder::c;
    ops::Audit rebuild;
    if (!ops::try_dense_elementwise(producer_plan,restored,scratch.workspace,rebuild,false))
      ops::execute(producer_plan,restored,scratch.workspace,ops::Isa::automatic,rebuild);
    accumulate_child(scratch,rebuild);
  }
  return completed;
}

static bool reuse_solve_pair(const Program &program,const std::vector<ops::Value> &inputs,
    std::size_t row,void *output,const ResultLayout &layout,const ops::Prepared &first,
    ops::Output &out1,Scratch &scratch,std::size_t capacity,Audit &audit) {
  const auto second_id=program.execution_metadata->solve_pair_second;
  const auto &node=program.nodes[second_id];
  const bool private_pair=program.execution_metadata->solve_pair_private_source>=0;
  const auto root=private_pair ? static_cast<std::int32_t>(std::find(program.roots.begin(),program.roots.end(),
      static_cast<std::uint32_t>(second_id))-program.roots.begin())
      : program.execution_metadata->root_destinations[second_id];
  const auto &schema=program.root_outputs[root];
  if ((!private_pair && (out1.order!=ops::MemoryOrder::c || layout.order(root)!=ops::MemoryOrder::c)) ||
      first.args[0].shape.rank!=2 || first.args[1].shape.rank!=2) return false;
  const std::array<ops::Value,2> values{first.args[0],inputs[program.nodes[node.parents[1]].input_index]};
  if (values[1].shape.rank!=2) return false;
  ops::Prepared second;
  try {
    second=ops::prepare(*program.execution_metadata->specs[second_id],values.data(),values.size());
  } catch (const ops::Error &) {
    return false; // Do not move a later node's preparation error ahead of this solve.
  }
  const auto &slot=layout.slots[row*layout.columns+root];
  if (second.output_shape.rank!=schema.rank || second.output_shape.size()>slot.capacity ||
      second.output_kind!=ops::Kind::number || schema.dtype!=OutputDType::float64) return false;
  ops::Output out2;
  out2.kind=second.output_kind; out2.shape=second.output_shape;
  out2.order=private_pair ? layout.order(root) : ops::MemoryOrder::c;
  out2.data=static_cast<std::uint8_t *>(output)+slot.byte_offset;
  auto destination1=out1;
  if (private_pair) {
    const auto first_root=std::find(program.roots.begin(),program.roots.end(),
        static_cast<std::uint32_t>(program.execution_metadata->solve_pair_first))-program.roots.begin();
    const auto &first_slot=layout.slots[row*layout.columns+first_root];
    if (first.output_shape.size()>first_slot.capacity) return false;
    destination1.data=static_cast<std::uint8_t *>(output)+first_slot.byte_offset;
    destination1.order=layout.order(first_root);
  }
  const auto begin1=reinterpret_cast<std::uintptr_t>(destination1.data),begin2=reinterpret_cast<std::uintptr_t>(out2.data);
  const auto bytes1=out1.shape.size()*sizeof(double),bytes2=out2.shape.size()*sizeof(double);
  if (bytes1 && bytes2 && (begin1<=begin2 ? begin2-begin1<bytes1 : begin1-begin2<bytes2)) return false;
  ops::Audit kernel_audit;
  if (private_pair) {
    if (!reuse_private_solve_pair(program,first,second,destination1,out2,scratch)) return false;
    if (program.execution_metadata->root_destinations[program.execution_metadata->solve_pair_first]<0)
      audit.direct_output_bytes+=bytes1;
    out1=destination1;
  } else {
    if (!ops::solve_multi_pair(first,second,out1,out2,scratch.workspace,kernel_audit)) return false;
    accumulate_child(scratch,kernel_audit);
  }
  scratch.values[second_id]=output_value(second,out2,capacity);
  audit.direct_output_bytes+=bytes2;
  return true;
}

struct ConcatCoefficientContext {
  const Program *program;
  const Scratch *scratch;
  std::size_t source;
  ops::Shape shape;
};
static void copy_concat_coefficients(const void *opaque, std::size_t row, std::size_t rows,
    std::size_t column, std::size_t columns, double *target) {
  const auto &context = *static_cast<const ConcatCoefficientContext *>(opaque);
  ops::Output out; out.kind = ops::Kind::number; out.shape = context.shape; out.data = target;
  const ConcatRectangle rectangle{row, rows, column, columns};
  execute_concat_tree(*context.program, context.source, out, *context.scratch, &rectangle);
}
static bool independent_concat_solve(const Program &program, std::size_t index) {
  const auto &metadata = *program.execution_metadata;
  if (program.output_kind != OutputKind::typed || program.isolate_errors ||
      !program.apply_scopes.empty() || !program.rolling_scopes.empty() ||
      metadata.concat_tree_roots.empty() ||
      !metadata.concat_tree_roots[program.nodes[index].parents[0]]) return false;
  // F results may still use a small C root arena before final materialization.
  // Terminal identity belongs to logical roots, independently of that layout.
  const auto found = std::find(program.roots.begin(), program.roots.end(), index);
  if (found == program.roots.end()) return false;
  const auto &root = program.root_outputs[static_cast<std::size_t>(found - program.roots.begin())];
  if (root.dtype != OutputDType::float64 || root.rank != 2) return false;
  for (const auto &node : program.nodes)
    for (std::size_t j = 0; j < node.parent_count; ++j)
      if (node.parents[j] == index) return false;
  return true;
}

// Select only an unobserved, exclusive coefficient arena of an admitted solve.
// Result capacity is resolved before execution; the solve still validates its
// actual RHS and geometry, and any rejection restores the canonical C arena.
static bool concat_tree_prefers_column_order(const Program &program, std::size_t source,
    const ops::Shape &shape, const Scratch &scratch) {
  const auto &metadata = *program.execution_metadata;
  std::vector<std::uint32_t> pending{static_cast<std::uint32_t>(source)};
  std::size_t remaining = shape.size(), columns = 0, rows = 0;
  while (!pending.empty()) {
    const auto id = pending.back(); pending.pop_back();
    if (id == source || metadata.deferred_to[id] == static_cast<std::int32_t>(source)) {
      const auto &node = program.nodes[id];
      pending.push_back(node.parents[1]); pending.push_back(node.parents[0]);
      continue;
    }
    const auto &leaf = scratch.values[id];
    if (leaf.kind != ops::Kind::number || leaf.shape.rank != 2) return false;
    const auto cells = leaf.size();
    if (cells > remaining) return false;
    remaining -= cells;
    if (!cells) continue;
    // Count actual leaf bytes. Unknown/reverse/broadcast layouts retain the
    // established C construction instead of guessing their copying cost.
    if (leaf.stride[0] == 1 && leaf.stride[1] >= static_cast<std::ptrdiff_t>(leaf.shape.dim[0]))
      columns += cells;
    else if (leaf.stride[1] == 1 && leaf.stride[0] >= static_cast<std::ptrdiff_t>(leaf.shape.dim[1]))
      rows += cells;
    else return false;
  }
  return !remaining && columns > rows;
}

static bool concat_solve_column_order(const Program &program, std::size_t source,
    const ops::Output &out, const ResultLayout &layout, std::size_t row, const Scratch &scratch) {
  const auto &metadata = *program.execution_metadata;
  if (out.kind != ops::Kind::number || out.shape.rank != 2 ||
      out.shape.dim[0] != out.shape.dim[1] || metadata.solve_private_source.empty() ||
      metadata.root_destinations[source] >= 0 || !layout.columns ||
      row >= layout.slots.size() / layout.columns || std::fegetround() != FE_TONEAREST) return false;
  const auto n = out.shape.dim[0];
  if (!n) return false;
  for (std::size_t root = 0; root < program.roots.size(); ++root) {
    const auto solve = program.roots[root];
    if (root >= layout.columns || metadata.solve_private_source[solve] != static_cast<std::int32_t>(source) ||
        !independent_concat_solve(program, solve)) continue;
    const auto capacity = layout.slots[row * layout.columns + root].capacity;
    if (capacity % n == 0 && ops::solve_concat_reserve_doubles(n, capacity / n))
      return concat_tree_prefers_column_order(program, source, out.shape, scratch);
  }
  return false;
}

static bool reuse_private_solve(const Program &program, std::size_t index,
    const ops::Prepared &prepared, ops::Output &destination, Scratch &scratch,
    ops::Audit &audit, bool allow_concat_library) {
  const auto &sources = program.execution_metadata->solve_private_source;
  const auto &metadata = *program.execution_metadata;
  const bool concat_library = allow_concat_library && independent_concat_solve(program, index) &&
      ops::solve_concat_reserve_doubles(prepared.args[0].shape.dim[0], prepared.args[1].shape.dim[1]);
  const bool private_f = concat_library || (destination.order == ops::MemoryOrder::f &&
      !metadata.solve_library_source.empty() && metadata.solve_library_source[index] >= 0 &&
      ops::solve_accelerate_scratch_doubles(prepared.args[0].shape.dim[0], prepared.args[1].shape.dim[1]));
  if (sources.empty() || sources[index] < 0 ||
      (destination.order != ops::MemoryOrder::c && !private_f))
    return false;
  const auto &a = prepared.args[0];
  if (a.shape.rank != 2 || a.shape.dim[0] != a.shape.dim[1] ||
      (!a.contiguous() && !(a.contiguous(ops::MemoryOrder::f) && (concat_library ||
        (!metadata.solve_library_source.empty() && metadata.solve_library_source[index] >= 0))))) return false;
  const auto cells = a.size();
  if (cells < 1024 * 1024 || cells > static_cast<std::size_t>(PTRDIFF_MAX) / sizeof(double))
    return false;
  const auto &producer = program.nodes[sources[index]];
  auto *coefficients = scratch.numeric_slot(producer.slot);
  if (!coefficients || a.data != coefficients ||
      cells > scratch.slot_capacity(StorageKind::numeric, producer.slot)) return false;
  const auto begin = reinterpret_cast<std::uintptr_t>(coefficients);
  const auto bytes = cells * sizeof(double);
  const auto overlaps = [&](std::uintptr_t address, std::size_t count) {
    return count && (address <= begin ? begin - address < count : address - begin < bytes);
  };
  if (destination.shape.size() > static_cast<std::size_t>(PTRDIFF_MAX) / sizeof(double) ||
      overlaps(reinterpret_cast<std::uintptr_t>(destination.data),
               destination.shape.size() * sizeof(double))) return false;
  // Conservatively reject overlapping RHS bounding spans, including reverse/zero strides.
  const auto &b = prepared.args[1];
  if (b.size()) {
    std::size_t lower = 0, upper = 0;
    const auto limit = static_cast<std::size_t>(PTRDIFF_MAX) / sizeof(double);
    for (int axis = 0; axis < b.shape.rank; ++axis) {
      const auto stride = b.stride[axis];
      if (stride == PTRDIFF_MIN) return false;
      const auto magnitude = static_cast<std::size_t>(stride < 0 ? -stride : stride);
      const auto steps = b.shape.dim[axis] - 1;
      if (magnitude && steps > limit / magnitude) return false;
      const auto distance = steps * magnitude;
      auto &bound = stride < 0 ? lower : upper;
      if (distance > limit - bound) return false;
      bound += distance;
    }
    const auto address = reinterpret_cast<std::uintptr_t>(b.data);
    if (lower > address / sizeof(double) || upper >= limit - lower) return false;
    if (overlaps(address - lower * sizeof(double), (lower + upper + 1) * sizeof(double)))
      return false;
  }
  if (concat_library) {
    const ConcatCoefficientContext context{&program, &scratch, static_cast<std::size_t>(sources[index]), a.shape};
    const ops::SolveCoefficientBlocks blocks{&context, copy_concat_coefficients};
    if (ops::try_solve_accelerate_concat(prepared, blocks, destination,
                                       scratch.workspace, audit, coefficients)) return true;
    auto canonical = prepared;
    if (audit.accelerate_lu_attempts || a.contiguous(ops::MemoryOrder::f)) {
      blocks.copy(blocks.context, 0, a.shape.dim[0], 0, a.shape.dim[1], coefficients);
      canonical.args[0].set_contiguous_strides();
      scratch.values[static_cast<std::size_t>(sources[index])].set_contiguous_strides();
      audit.algorithm_copy_bytes += bytes;
    }
    ops::Audit replay;
    if (destination.order == ops::MemoryOrder::c)
      ops::solve_multi_private(canonical, destination, scratch.workspace, replay, coefficients);
    else
      ops::execute(canonical, destination, scratch.workspace, ops::Isa::automatic, replay);
    replay.accelerate_lu_attempts = audit.accelerate_lu_attempts;
    replay.accelerate_lu_fallbacks = audit.accelerate_lu_fallbacks;
    replay.solve_residual_materialization_bytes = audit.solve_residual_materialization_bytes;
    replay.algorithm_copy_bytes += audit.algorithm_copy_bytes;
    replay.native_library_workspace_reserve_bytes = std::max(
        replay.native_library_workspace_reserve_bytes, audit.native_library_workspace_reserve_bytes);
    replay.scratch_bytes = std::max(replay.scratch_bytes, audit.scratch_bytes);
    audit = replay;
    return true;
  }
  if (!metadata.solve_library_source.empty() && metadata.solve_library_source[index] >= 0 &&
      (!a.contiguous() || ops::solve_accelerate_scratch_doubles(a.shape.dim[0], b.shape.dim[1]))) {
    ops::Arguments<ops::Value> arguments(producer.parent_count);
    for (std::size_t j = 0; j < producer.parent_count; ++j)
      arguments[j] = scratch.values[producer.parents[j]];
    const auto producer_plan = ops::prepare(*metadata.specs[sources[index]], arguments.data(), producer.parent_count);
    if (ops::try_solve_accelerate_private(prepared, producer_plan, destination,
                                         scratch.workspace, audit, coefficients)) return true;
    // Failed admission or library validation can leave the private coefficient
    // in F/factor form. Reconstruct canonical C from still-live original leaves.
    if (!a.contiguous() || audit.accelerate_lu_attempts) {
      ops::Output restored; restored.kind = producer_plan.output_kind;
      restored.shape = producer_plan.output_shape; restored.data = coefficients;
      ops::Audit rebuild;
      ops::execute(producer_plan, restored, scratch.workspace, ops::Isa::automatic, rebuild);
    }
    auto canonical = prepared;
    canonical.args[0].set_contiguous_strides();
    ops::Audit replay;
    if (destination.order == ops::MemoryOrder::c)
      ops::solve_multi_private(canonical, destination, scratch.workspace, replay, coefficients);
    else
      ops::execute(canonical, destination, scratch.workspace, ops::Isa::automatic, replay);
    replay.accelerate_lu_attempts = audit.accelerate_lu_attempts;
    replay.accelerate_lu_fallbacks = audit.accelerate_lu_fallbacks;
    replay.solve_residual_materialization_bytes = audit.solve_residual_materialization_bytes;
    replay.algorithm_copy_bytes += audit.algorithm_copy_bytes;
    replay.native_library_workspace_reserve_bytes = std::max(
        replay.native_library_workspace_reserve_bytes, audit.native_library_workspace_reserve_bytes);
    replay.scratch_bytes = std::max(replay.scratch_bytes, audit.scratch_bytes);
    audit = replay;
    return true;
  }
  ops::solve_multi_private(prepared, destination, scratch.workspace, audit, coefficients);
  return true;
}

// A borrowed future result is unpublished and was initialized to zero. Restore
// that state before downstream execution or exception propagation can expose it.
struct UnwrittenLatentResult {
  double *data = nullptr;
  std::size_t count = 0;
  UnwrittenLatentResult(double *span, std::size_t elements) : data(span), count(elements) {}
  UnwrittenLatentResult(const UnwrittenLatentResult &) = delete;
  UnwrittenLatentResult &operator=(const UnwrittenLatentResult &) = delete;
  ~UnwrittenLatentResult() { if (data) std::fill_n(data, count, 0.0); }
};

// Loan only exclusive physical buffers; the logical graph and one fit body stay intact.
static bool loan_latent_workspace(const Program &program, std::size_t index,
    std::size_t row, void *output, const ResultLayout &layout,
    const ops::Prepared &prepared, ops::Output &destination, Scratch &scratch,
    ops::Audit &audit) {
  const auto op = prepared.spec->op;
  if ((op != ops::Op::gaussian_mixture_em && op != ops::Op::gaussian_hmm_em) ||
      program.output_kind != OutputKind::typed || program.isolate_errors ||
      !program.apply_scopes.empty() || !program.rolling_scopes.empty() ||
      destination.order != ops::MemoryOrder::c) return false;
  const auto cells = prepared.args[0].shape.dim[0] * prepared.args[1].size();
  const auto future_span = [&]() -> double * {
    double *future = nullptr;
    for (std::size_t root = 0; root < program.roots.size(); ++root) {
      if (layout.columns == 0 || root >= layout.columns ||
          root >= layout.slots.size() ||
          row > (layout.slots.size() - 1 - root) / layout.columns) continue;
      const auto &slot = layout.slots[row * layout.columns + root];
      if (!latent_future_root(program,index,root,cells,slot.capacity)) continue;
      if (slot.capacity < cells || slot.byte_offset > layout.bytes() ||
          cells > (layout.bytes() - slot.byte_offset) / sizeof(double) ||
          slot.byte_offset % alignof(double)) continue;
      future = reinterpret_cast<double *>(static_cast<std::uint8_t *>(output) + slot.byte_offset);
      const auto future_address = reinterpret_cast<std::uintptr_t>(future);
      const auto destination_address = reinterpret_cast<std::uintptr_t>(destination.data);
      const auto destination_bytes = destination.shape.size() * sizeof(double);
      if (destination_bytes && (future_address <= destination_address ?
          destination_address-future_address < cells*8 :
          future_address-destination_address < destination_bytes)) { future=nullptr; continue; }
      return future; // current-call independent future storage only
    }
    return nullptr;
  };
  // A full-posterior GMM needs one NK matrix. Use a still-unwritten current
  // result before touching the arena, so no live input requires save/restore.
  // Prepared/observers/isolation are rejected by the caller and guards above.
  if (op == ops::Op::gaussian_mixture_em && cells >= 1024 * 1024 &&
      prepared.scratch_doubles >= cells) {
    if (auto *future = future_span()) {
      UnwrittenLatentResult restore_future(future, cells);
      const auto small = prepared.scratch_doubles - cells;
      if (scratch.workspace.doubles.capacity() > small)
        std::vector<double>().swap(scratch.workspace.doubles);
      scratch.workspace.ensure(small, prepared.scratch_indices);
      ops::LatentFitScratch fit;
      fit.em = fit.alpha = fit.beta = future;
      fit.small = scratch.workspace.doubles.data();
      fit.cells = cells; fit.small_count = small;
      ops::execute(prepared, destination, scratch.workspace, ops::Isa::automatic, audit, &fit);
      return true;
    }
  }
  const auto minimum_cells = op == ops::Op::gaussian_hmm_em ? std::size_t{131072}
                                                           : std::size_t{2 * 1024 * 1024};
  if (cells < minimum_cells || scratch.numeric.empty()) return false;
  if (op == ops::Op::gaussian_mixture_em && prepared.scratch_doubles < cells) return false;
  const auto planned_matrices = ops::latent_fit_matrix_count(op,
      prepared.args[0].shape.dim[0], prepared.args[1].size());
  // A third existing matrix avoids density replay without allocating one.
  // The preferred route may use only arena/future-result spans already covered
  // by the graph plan; otherwise return to the two-matrix reservation.
  auto matrices = planned_matrices == 2 ? std::size_t{3} : planned_matrices;
  auto arena_matrices = std::min(matrices, scratch.numeric.size() / cells);
  if (!arena_matrices) return false;
  const auto begin = reinterpret_cast<std::uintptr_t>(scratch.numeric.data());
  const auto bytes = scratch.numeric.size() * sizeof(double);
  const auto in_arena = [&](const void *data) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return address >= begin && address - begin < bytes;
  };
  // A future edge to a borrowed view keeps its materialized backing alive.
  std::vector<std::uint8_t> live(program.nodes.size(), 0);
  const auto keep = [&](auto &&self, std::size_t id) -> void {
    if (id >= index || live[id]) return;
    live[id] = 1;
    const auto &node = program.nodes[id];
    if (node.kind == NodeKind::interval_tail || (node.kind == NodeKind::operation &&
        (node.opcode == static_cast<std::uint16_t>(ops::Op::lag) ||
         node.opcode == static_cast<std::uint16_t>(ops::Op::transpose) ||
         ops::foundation_view(static_cast<ops::Op>(node.opcode)) ||
         ops::latent_view(static_cast<ops::Op>(node.opcode)) ||
         ops::series_record_projection(static_cast<ops::Op>(node.opcode)))))
      self(self, node.parents[0]);
  };
  for (std::size_t i = index; i < program.nodes.size(); ++i)
    for (std::size_t j = 0; j < program.nodes[i].parent_count; ++j)
      keep(keep, program.nodes[i].parents[j]);
  for (const auto root : program.roots) keep(keep, root);
  struct Save { double *source; std::size_t count, offset; };
  std::vector<Save> saves;
  std::size_t save_count = 0;
  for (std::size_t i = 0; i < index; ++i) {
    if (!live[i] || program.nodes[i].storage != StorageKind::numeric) continue;
    const auto &value = scratch.values[i];
    if (!in_arena(value.data)) continue; // external inputs and own past roots
    if (value.kind != ops::Kind::number || !value.c_contiguous() ||
        value.data != scratch.numeric_slot(program.nodes[i].slot)) return false;
    const auto count = value.size();
    const auto offset = (reinterpret_cast<std::uintptr_t>(value.data) - begin) / sizeof(double);
    if (offset > scratch.numeric.size() || count > scratch.numeric.size() - offset ||
        count > static_cast<std::size_t>(PTRDIFF_MAX) / sizeof(double) - save_count) return false;
    for (const auto &save : saves) {
      const auto a = reinterpret_cast<std::uintptr_t>(save.source);
      const auto b = reinterpret_cast<std::uintptr_t>(value.data);
      if (count && save.count && (a <= b ? b-a < save.count*8 : a-b < count*8)) return false;
    }
    saves.push_back({const_cast<double *>(static_cast<const double *>(value.data)), count, save_count});
    save_count += count;
  }
  // Validate every arena-backed fit argument before saving or writing any buffer.
  for (std::size_t j = 0; j < prepared.count; ++j) {
    const auto &arg = prepared.args[j];
    if (!in_arena(arg.data)) continue;
    if (arg.kind != ops::Kind::number || arg.shape.rank < 0 || arg.shape.rank > 3)
      return false;
    // A transpose, broadcast or reverse view must fit completely in its saved
    // owner, not merely have its first logical element inside that range.
    std::size_t lower = 0, upper = 0;
    const auto stride_limit = static_cast<std::size_t>(PTRDIFF_MAX);
    for (int axis = 0; axis < arg.shape.rank; ++axis) {
      const auto stride = arg.stride[axis];
      if (stride == PTRDIFF_MIN) return false;
      const auto magnitude = static_cast<std::size_t>(stride < 0 ? -stride : stride);
      const auto steps = arg.shape.dim[axis] ? arg.shape.dim[axis] - 1 : 0;
      if (magnitude && steps > stride_limit / magnitude) return false;
      const auto distance = steps * magnitude;
      auto &bound = stride < 0 ? lower : upper;
      if (distance > stride_limit - bound) return false;
      bound += distance;
    }
    bool found = false;
    const auto address = reinterpret_cast<std::uintptr_t>(arg.data);
    for (const auto &save : saves) {
      const auto first = reinterpret_cast<std::uintptr_t>(save.source);
      if (address < first || (address-first) % sizeof(double)) continue;
      const auto offset = (address - first) / sizeof(double);
      found = found || (offset < save.count && lower <= offset &&
          upper < save.count - offset);
    }
    if (!found) return false;
  }
  double *future = arena_matrices < matrices ? future_span() : nullptr;
  if (planned_matrices == 2 && arena_matrices + (future ? 1 : 0) < 3) {
    matrices = 2;
    arena_matrices = std::min(matrices, scratch.numeric.size() / cells);
    if (arena_matrices == matrices) future = nullptr;
  }
  const auto missing = matrices - arena_matrices - (future ? 1 : 0);
  const auto output_count = in_arena(destination.data) ? destination.shape.size() : 0;
  const auto full_scratch = op == ops::Op::gaussian_hmm_em ?
      ops::latent_hmm_full_scratch(prepared.args[0].shape.dim[0],prepared.args[1].size()) :
      prepared.scratch_doubles;
  const auto small_count = full_scratch - planned_matrices * cells;
  const auto limit = static_cast<std::size_t>(PTRDIFF_MAX) / sizeof(double);
  if (output_count > limit - save_count || missing * cells > limit - save_count - output_count ||
      small_count > limit - save_count - output_count - missing * cells) return false;
  const auto reserve = save_count + output_count + missing * cells + small_count;
  if (reserve >= prepared.scratch_doubles) return false;
  UnwrittenLatentResult restore_future(future, cells);
  // No scratch borrow can survive this ordinary call. Remove an obsolete larger
  // fit reservation instead of retaining it behind a smaller logical size.
  if (scratch.workspace.doubles.capacity() > reserve)
    std::vector<double>().swap(scratch.workspace.doubles);
  scratch.workspace.ensure(reserve, prepared.scratch_indices);
  auto *packed = scratch.workspace.doubles.data();
  for (const auto &save : saves)
    std::copy_n(save.source, save.count, packed + save.offset);
  auto rebound = prepared;
  for (std::size_t j = 0; j < rebound.count; ++j) {
    auto &arg = rebound.args[j];
    if (!in_arena(arg.data)) continue;
    const auto address = reinterpret_cast<std::uintptr_t>(arg.data);
    for (const auto &save : saves) {
      const auto first = reinterpret_cast<std::uintptr_t>(save.source);
      if (address >= first && address - first < save.count * 8) {
        arg.data = packed + save.offset + (address - first) / 8;
        break;
      }
    }
  }
  auto fit_output = destination;
  if (output_count) fit_output.data = packed + save_count;
  std::array<double *, 3> spans{};
  std::size_t used = 0;
  for (; used < arena_matrices; ++used) spans[used] = scratch.numeric.data() + used * cells;
  if (future) spans[used++] = future;
  for (std::size_t j = 0; used < matrices; ++used, ++j)
    spans[used] = packed + save_count + output_count + j * cells;
  ops::LatentFitScratch fit;
  fit.em = spans[0]; fit.alpha = matrices >= 2 ? spans[1] : spans[0];
  fit.beta = matrices == 3 ? spans[2] : spans[0];
  fit.small = packed + save_count + output_count + missing * cells;
  fit.cells = cells; fit.small_count = small_count;
  const auto restore = [&] {
    for (const auto &save : saves)
      std::copy_n(packed + save.offset, save.count, save.source);
    if (output_count) std::copy_n(static_cast<double *>(fit_output.data), output_count,
                                static_cast<double *>(destination.data));
  };
  try { ops::execute(rebound, fit_output, scratch.workspace, ops::Isa::automatic, audit, &fit); }
  catch (...) { restore(); throw; }
  restore();
  audit.algorithm_copy_bytes += (2 * save_count + output_count) * sizeof(double);
  return true;
}

struct RowSubviewPlan {
  std::vector<std::uint8_t> omit;
  std::vector<std::size_t> first;
  std::vector<std::uint32_t> source;
};

// A bounded descriptor probe only. Any invalid slice/index keeps the original
// node's validation and error order; no numerical kernel runs during planning.
static bool row_index_view(const Program &p, const std::vector<ops::Value> &inputs,
    const double *parameters, std::uint32_t id, std::size_t start, std::size_t length,
    ops::Value &value, unsigned depth = 0) {
  if (depth >= 8) return false;
  const auto &node = p.nodes[id];
  if (node.kind == NodeKind::constant) { value = ops::Value::number(node.constant); return true; }
  if (node.kind == NodeKind::parameter) {
    if (!parameters) return false;
    value = ops::Value::number(parameters[node.input_index]); return true;
  }
  if (node.kind == NodeKind::input) {
    value = (!p.input_axes.empty() && p.input_axes[node.input_index] == 1)
        ? inputs[node.input_index] : interval_view(inputs[node.input_index], start, length);
    return value.shape.rank != 0;
  }
  if (node.kind != NodeKind::operation || node.opcode != std::uint16_t(ops::Op::slice_axis) ||
      node.parent_count != 5) return false;
  ops::Arguments<ops::Value> args(5);
  for (std::size_t j = 0; j < 5; ++j)
    if (!row_index_view(p, inputs, parameters, node.parents[j], start, length, args[j], depth + 1)) return false;
  const auto slice = ops::prepare(*p.execution_metadata->specs[id], args.data(), 5);
  if (!slice.borrowed) return false;
  value = slice.view; return true;
}

// Only positive, fully contiguous external backing is eligible. A logical
// row slice keeps original strides of external backing and independent roots.
static bool row_subview_backing_contiguous(const ops::Value &value) {
  if (value.kind != ops::Kind::number || value.shape.rank != 2) return false;
  return (value.stride[1] == 1 &&
          value.stride[0] == static_cast<std::ptrdiff_t>(value.shape.dim[1])) ||
         (value.stride[0] == 1 &&
          value.stride[1] == static_cast<std::ptrdiff_t>(value.shape.dim[0]));
}

static RowSubviewPlan prove_row_subviews(const Program &p,
    const std::vector<ops::Value> &inputs, const double *parameters,
    const std::vector<ops::Shape> &shapes, std::size_t start, std::size_t length,
    const ResultLayout &layout, std::size_t result_row, void *output) {
  RowSubviewPlan plan;
  const auto &m = *p.execution_metadata;
  if (m.row_subview_takes.empty() || shapes.size() != p.nodes.size() ||
      layout.columns != p.roots.size() || layout.row_bytes.size() < 2 ||
      result_row >= layout.row_bytes.size() - 1 ||
      (!layout.orders.empty() && layout.orders.size() != layout.columns) ||
      layout.columns > layout.slots.size() / (result_row + 1)) return plan;
  const auto root_contiguous = [&](std::uint32_t id) {
    const auto root = m.root_destinations[id];
    if (root < 0 || p.roots[root] != id ||
        (layout.order(root) != ops::MemoryOrder::c && layout.order(root) != ops::MemoryOrder::f)) return false;
    const auto &slot = layout.slots[result_row * layout.columns + root];
    return p.root_outputs[root].rank == 2 && p.root_outputs[root].dtype == OutputDType::float64 &&
        slot.capacity >= shapes[id].size();
  };
  for (const auto id : m.row_subview_takes) {
    const auto &take = p.nodes[id];
    const auto source_id = take.parents[0];
    const auto &shape = shapes[id], &full = shapes[source_id];
    if (shape.rank != 2 || full.rank != 2 || !shape.dim[0] || !shape.dim[1] ||
        shape.dim[1] != full.dim[1] || shape.size() < row_subview_min_elements) continue;
    if (p.nodes[source_id].kind == NodeKind::input) {
      auto source = inputs[p.nodes[source_id].input_index];
      if (p.input_axes.empty() || p.input_axes[p.nodes[source_id].input_index] != 1)
        source = interval_view(source, start, length);
      if (!row_subview_backing_contiguous(source) || !(source.shape == full) ||
          !source.data) continue;
      // Native callers may supply their own result span. Decline if it could
      // overwrite external backing that canonical take would have copied.
      const auto lo = reinterpret_cast<std::uintptr_t>(source.data);
      const auto out = reinterpret_cast<std::uintptr_t>(output);
      if (full.size() > (UINTPTR_MAX - lo) / 8 || layout.bytes() > UINTPTR_MAX - out) continue;
      if (lo < out + layout.bytes() && out < lo + full.size() * 8) continue;
    } else if (!root_contiguous(source_id)) continue;
    std::size_t first = 0;
    bool contiguous = false;
    try {
      ops::Value indices;
      if (!row_index_view(p, inputs, parameters, take.parents[1], start, length, indices) ||
          indices.kind != ops::Kind::integer || indices.shape.rank != 1 ||
          indices.size() != shape.dim[0] || !indices.data || indices.i(0) < 0) continue;
      first = static_cast<std::size_t>(indices.i(0));
      if (first > full.dim[0] || indices.size() > full.dim[0] - first) continue;
      contiguous = true;
      for (std::size_t j = 1; j < indices.size(); ++j)
        if (indices.i(j) != static_cast<std::int64_t>(first + j)) { contiguous = false; break; }
    } catch (const ops::Error &) { continue; }
    if (!contiguous) continue;
    if (plan.omit.empty()) {
      plan.omit.resize(p.nodes.size()); plan.first.resize(p.nodes.size()); plan.source.resize(p.nodes.size());
    }
    plan.omit[id] = 1; plan.first[id] = first; plan.source[id] = source_id;
  }
  if (plan.omit.empty()) return plan;
  for (const auto &region : m.row_subview_reuses) {
    if (!plan.omit[region.take] || !root_contiguous(region.full_root)) continue;
    const auto &selected = shapes[region.selected], &full = shapes[region.full_root];
    if (selected.rank != 2 || full.rank != 2 || !(selected == shapes[region.take]) ||
        selected.size() < row_subview_min_elements || full.dim[1] != selected.dim[1] ||
        plan.first[region.take] > full.dim[0] ||
        selected.dim[0] > full.dim[0] - plan.first[region.take]) continue;
    plan.omit[region.selected] = 2;
    plan.first[region.selected] = plan.first[region.take]; plan.source[region.selected] = region.full_root;
  }
  return plan;
}

template <bool TrackPositions, bool Observe = false, bool DeferErrors = false>
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
                          bool interval_known, const ResultLayout *layout, std::size_t result_row,
                          const std::vector<const char *> *deferred_errors = nullptr) {
  scratch.failed_node = std::numeric_limits<std::size_t>::max();
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
        ops::require(inputs[i].shape.rank >= 0 && inputs[i].shape.rank <= 3, "GRAPH_INPUT_TYPE");
        if (axis != 1) ops::require(end <= inputs[i].shape.dim[0], "GRAPH_INTERVAL_BOUNDS");
      }
      max_window = std::max(max_window, end - start);
    }
  }

  const auto logical_window = max_window;
  std::vector<ops::Shape> arena_shapes;
  max_window = required_array_capacity(program, inputs, max_window,
      program.execution_metadata->requires_shape_planning ? &arena_shapes : nullptr);
  const bool fusion_context = !Observe && !DeferErrors && !program.isolate_errors && input_failures == nullptr;
  const bool fuse_pairs = fusion_context && !program.execution_metadata->deferred_to.empty();
  const bool fuse_decisions = fusion_context && !program.execution_metadata->state_decisions.empty();
  const bool private_matmul_chain = fusion_context && matmul_scalar_chain_geometry(program,arena_shapes) &&
      (program.execution_metadata->matmul_shared_chains.empty() ||
       (!prebound_inputs && matmul_shared_layout_eligible(program,arena_shapes,layout,result_row,rows)));
  const bool summary_fusion=fusion_context && !prebound_inputs &&
      !program.execution_metadata->matmul_summaries.empty();
  const bool private_where_sums = !Observe && !prebound_inputs &&
      !program.isolate_errors && input_failures == nullptr &&
      (!program.execution_metadata->where_sum_to.empty() ||
       !program.execution_metadata->scan_reductions.empty() ||
       !program.execution_metadata->square_reductions.empty()) &&
      (!deferred_errors || std::all_of(deferred_errors->begin(), deferred_errors->end(),
          [](const char *error) { return error == nullptr; }));
  RowSubviewPlan row_subviews;
  if (fusion_context && scratch.allow_row_subviews && !prebound_inputs && rows == 1 && layout &&
      program.output_kind == OutputKind::typed && scratch.allow_workspace_loan &&
      program.apply_scopes.empty() && program.rolling_scopes.empty())
    row_subviews = prove_row_subviews(program, inputs, parameters, arena_shapes,
        static_cast<std::size_t>(starts[0]), static_cast<std::size_t>(ends[0] - starts[0]),
        *layout, result_row, output);
  StateRootLoan state_loan;
  if (fusion_context && fuse_decisions && !prebound_inputs && rows == 1 && layout &&
      program.output_kind == OutputKind::typed && scratch.allow_workspace_loan)
    state_loan = prove_state_root_loan(program, inputs, arena_shapes, *layout, result_row, output);
  scratch.ensure(program, max_window, program.isolate_errors || input_failures != nullptr,
                 arena_shapes, fuse_pairs || fuse_decisions || private_matmul_chain, summary_fusion, private_where_sums,
                 program.execution_metadata->matmul_shared_chains.empty() || private_matmul_chain,
                 row_subviews.omit.empty() ? nullptr : &row_subviews.omit, state_loan.source);
  scratch.interval_known = interval_known;
  if (!prebound_inputs) {
    scratch.work_budget = static_cast<std::size_t>(program.scope_work_budget);
    scratch.remaining_work = &scratch.work_budget;
  }
  scratch.fused_scalar_calls = 0;
  scratch.summary_source_scans = 0;
  scratch.order_stat_sorts = 0;
  scratch.highway_vqsort_calls = 0;
  scratch.algorithm_copy_bytes = 0;
  scratch.eigen_rotation_calls = 0;
  scratch.eigen_matvec_calls = 0;
  scratch.eigen_covariance_calls = 0;
  scratch.eigen_density_calls = 0;
  scratch.gaussian_filter_streams = 0;
  scratch.accelerate_covchol_calls = 0;
  scratch.accelerate_matmul_calls = scratch.matmul_zero_calls = 0;
  scratch.accelerate_covariance_attempts = 0;
  scratch.accelerate_covariance_calls = 0;
  scratch.covariance_syrk_calls = 0;
  scratch.covariance_replayed_cells = 0;
  scratch.covariance_fallbacks = 0;
  scratch.covariance_zero_calls = 0;
  scratch.accelerate_vforce_calls = 0;
  scratch.accelerate_lu_attempts = 0;
  scratch.accelerate_lu_calls = 0;
  scratch.accelerate_lu_fallbacks = 0;
  scratch.solve_residual_materialization_bytes = 0;
  scratch.covariance_panel_materialization_bytes = 0;
  scratch.native_library_workspace_reserve_bytes = 0;
  scratch.state_copy_bytes = scratch.child_root_copy_bytes = scratch.child_direct_output_bytes = 0;
  scratch.child_vector_elements = scratch.child_pointwise_tiles = scratch.child_fused_pointwise_calls = 0;
  scratch.child_pointwise_capacity = 0;
  const auto &summary_consumers = program.execution_metadata->summary_consumers;
  const auto &order_consumers = program.execution_metadata->order_consumers;
  std::size_t series_base = 0;
  Audit audit;
  const bool typed_output = program.output_kind == OutputKind::typed;
  if (typed_output) {
    ops::require(layout && !layout->row_bytes.empty() &&
        layout->row_statuses.size() == layout->row_bytes.size() &&
        layout->columns == output_columns &&
        result_row <= layout->row_bytes.size() - 1 && rows <= layout->row_bytes.size() - 1 - result_row,
        "RESULT_LAYOUT_BOUNDS");
    ops::require(reinterpret_cast<std::uintptr_t>(output) % alignof(double) == 0,
                 "RESULT_OUTPUT_ALIGNMENT");
    audit.result_shapes.resize(rows * output_columns);
    audit.root_statuses.resize(rows * output_columns);
    const auto bytes = layout->row_bytes[result_row + rows] - layout->row_bytes[result_row];
    // A strict compact post graph writes a subset of an already live result
    // row. Clearing that row would erase its independently produced roots.
    // A dense private row is completely written on success. The per-root
    // guards below restore every unpublished root on any exception, including
    // failed geometry/parameter nodes. Padding and multi-row prefixes retain
    // canonical eager clearing rather than relying on this storage proof.
    const bool complete_private_row=private_matmul_chain && rows==1 &&
        !program.execution_metadata->matmul_shared_chains.empty() &&
        matmul_shared_dense_row(program,arena_shapes,*layout,result_row);
    if (bytes && !DeferErrors && !complete_private_row)
      std::memset(static_cast<std::uint8_t *>(output) + layout->row_bytes[result_row], 0, bytes);
    if (scratch.isolate_errors)
      audit.statuses.resize(layout->row_statuses[result_row + rows] - layout->row_statuses[result_row]);
  } else if (scratch.isolate_errors) {
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
    struct PrivateMatrixRootGuard {
      struct Buffer { void *data=nullptr;std::size_t bytes=0;bool published=false; };
      std::array<Buffer,8> buffers{};
      int exceptions=std::uncaught_exceptions();
      void clear_unpublished() {
        for (const auto &buffer:buffers)
          if (buffer.data && !buffer.published) std::memset(buffer.data,0,buffer.bytes);
      }
      ~PrivateMatrixRootGuard() {
        if (std::uncaught_exceptions()>exceptions) clear_unpublished();
      }
    } matrix_root_guard;
    if (private_matmul_chain) {
      for (std::size_t root=0;root<program.roots.size();++root) {
        const auto &slot=layout->slots[(result_row+row)*output_columns+root];
        matrix_root_guard.buffers[root].data=static_cast<std::uint8_t *>(output)+slot.byte_offset;
        matrix_root_guard.buffers[root].bytes=slot.capacity*sizeof(double);
      }
    }
    scratch.next_generation();
    bool split_eigh_completed=false;
    bool solve_pair_completed=false;
    bool matrix_chain_accelerated=false;
    double matrix_chain_error=0.0;
    const auto start = static_cast<std::size_t>(starts[row]);
    const auto end = static_cast<std::size_t>(ends[row]);
    const auto length = end - start;
    if (interval_known && length < program.minimum_observations) {
      ops::require(scratch.isolate_errors, "INSUFFICIENT_OBSERVATIONS");
      if (typed_output) {
        for (std::size_t root = 0; root < output_columns; ++root) {
          const auto index = row * output_columns + root;
          audit.result_shapes[index].rank = -1;
          audit.root_statuses[index] = 1;
          const auto &slot = layout->slots[(result_row + row) * output_columns + root];
          const auto first = slot.status_offset - layout->row_statuses[result_row];
          if (slot.capacity) std::fill_n(audit.statuses.data() + first, slot.capacity, 1);
        }
        continue;
      }
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
      if (solve_pair_completed && static_cast<std::int32_t>(node_index)==
          program.execution_metadata->solve_pair_second) continue;
      if (split_eigh_completed &&
          (static_cast<std::int32_t>(node_index)==program.execution_metadata->split_eigh_values ||
           static_cast<std::int32_t>(node_index)==program.execution_metadata->split_eigh_vectors)) continue;
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
        if constexpr (DeferErrors)
          if (const auto *failure = (*deferred_errors)[node_index])
            throw ops::Error(failure);
        if (node.kind == NodeKind::input) {
          scratch.values[node_index] =
              (prebound_inputs || (!program.input_axes.empty() && program.input_axes[node.input_index] == 1))
                  ? inputs[node.input_index] : interval_view(inputs[node.input_index], start,
                                              length);
          auto &input = scratch.values[node_index];
          // Rank-zero arrays remain live views across prepared executions.
          // Refresh the local descriptor, never mutate a shared bound batch.
          if (input.shape.rank == 0 && input.data) {
            if (input.kind == ops::Kind::integer)
              input.integer = *static_cast<const std::int64_t *>(input.data);
            else if (input.kind == ops::Kind::mask)
              input.scalar = *static_cast<const std::uint8_t *>(input.data);
            else input.scalar = *static_cast<const double *>(input.data);
          }
          if (input_failures) scratch.statuses[node_index] = (*input_failures)[node.input_index];
          if constexpr (Observe)
            (*value_observer)(std::nullopt, scratch.values[node_index], false);
          continue;
        }
        if (node.kind == NodeKind::parameter) {
          scratch.values[node_index] =
              ops::Value::number(parameters[node.input_index]);
          if (scratch.isolate_errors && !std::isfinite(parameters[node.input_index]))
            scratch.statuses[node_index] = 4;
          if constexpr (Observe)
            (*value_observer)(std::nullopt, scratch.values[node_index], false);
          continue;
        }
        if (node.kind == NodeKind::constant) {
          scratch.values[node_index] = ops::Value::number(node.constant);
          if constexpr (Observe)
            (*value_observer)(std::nullopt, scratch.values[node_index], false);
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
          if constexpr (Observe)
            (*value_observer)(std::nullopt, scratch.values[node_index], true);
          continue;
        }
        if (node.kind == NodeKind::rolling_scope) {
          scratch.values[node_index] = execute_rolling_scope(
              program, inputs, program.rolling_scopes[node.input_index], node,
              node_index, start, length, scratch);
          if constexpr (Observe)
            (*value_observer)(std::nullopt, scratch.values[node_index], true);
          continue;
        }

        if (node.kind == NodeKind::apply_scope) {
          void *root_destination = nullptr;
          const auto root = program.execution_metadata->root_destinations[node_index];
          if (typed_output && root >= 0 && program.apply_scopes[node.input_index].kind == ApplyKind::iterate) {
            const auto &slot = layout->slots[(result_row + row) * output_columns + root];
            const auto &initial = scratch.values[program.apply_scopes[node.input_index].argument_nodes[0]];
            ops::require(program.root_outputs[root].dtype == OutputDType::float64,
                "RESULT_DTYPE_MISMATCH");
            ops::require(initial.shape.rank == program.root_outputs[root].rank,
                "RESULT_CAPACITY_MISMATCH");
            ops::require(initial.size() <= slot.capacity, "RESULT_CAPACITY_MISMATCH");
            root_destination = static_cast<std::uint8_t *>(output) + slot.byte_offset;
          }
          scratch.values[node_index] = execute_apply_scope(program,
              program.apply_scopes[node.input_index], node, node_index, length, max_window, scratch, root_destination);
          // Explicit empty/NaN filter results are missing data, not execution
          // failures. Preserve them for ordinary where selection; public roots
          // still report non-finite results as unavailable.
          if (scratch.isolate_errors && program.apply_scopes[node.input_index].kind != ApplyKind::filter &&
              scratch.values[node_index].shape.rank == 0 &&
              !std::isfinite(scratch.values[node_index].scalar)) scratch.statuses[node_index] = 4;
          if constexpr (Observe)
            (*value_observer)(std::nullopt, scratch.values[node_index], true);
          continue;
        }

        if (node.kind == NodeKind::iteration_projection) {
          const auto scope_index = program.nodes[node.parents[0]].input_index;
          const auto diagnostic = scratch.iteration_info[scope_index][node.opcode];
          auto result = ops::Value::number(diagnostic);
          if (node.opcode != 2) { result.kind = ops::Kind::integer; result.integer = static_cast<std::int64_t>(diagnostic); }
          scratch.values[node_index] = result;
          if constexpr (Observe)
            (*value_observer)(std::nullopt, scratch.values[node_index], true);
          continue;
        }

        ops::Arguments<ops::Value> arguments(node.parent_count);
        for (std::size_t parent = 0; parent < node.parent_count; ++parent)
          arguments[parent] = scratch.values[node.parents[parent]];

        const auto &spec = *program.execution_metadata->specs[node_index];
        if constexpr (Observe)
          for (std::size_t i = 0; i < node.parent_count; ++i)
            (*value_observer)(spec.op, arguments[i], false);
        if (fuse_pairs && !program.execution_metadata->filter_tail_source.empty() &&
            program.execution_metadata->filter_tail_source[node_index] >= 0) {
          const auto source = static_cast<std::size_t>(
              program.execution_metadata->filter_tail_source[node_index]);
          if (filter_tail_geometry(program, source, arena_shapes) ||
              gaussian_filter_stream_geometry(program, source, arena_shapes)) {
            // The full descriptor is geometry only. Do not form the canonical
            // (N-1)*K pointer into a private buffer containing just one K row.
            const auto tail = ops::prepare_geometry(spec, arguments.data(), node.parent_count);
            ops::require(tail.output_geometry_known && tail.output_shape == arena_shapes[node_index],
                         "FILTER_TAIL_CONTRACT");
            ops::Value value;
            value.kind = tail.output_kind;
            value.shape = tail.output_shape;
            value.set_contiguous_strides();
            value.stride[0] = 0;
            value.data = scratch.numeric_slot(program.nodes[source].slot);
            scratch.values[node_index] = value;
            continue;
          }
        }
        const bool private_scatter_consumer=fuse_pairs && spec.op==ops::Op::argmax_axis &&
            program.execution_metadata->fused_from[node_index]>=0 &&
            deferred_pair_geometry(program,program.execution_metadata->fused_from[node_index],arena_shapes);
        // This consumer only needs the deferred scatter's geometry. An empty
        // source can have a null data pointer while its filled result is nonempty.
        const bool private_decision=fuse_decisions && state_decision_geometry(program,node_index,arena_shapes);
        if (private_decision && spec.op==ops::Op::state_select) {
          const auto &region=program.execution_metadata->state_decisions[
              program.execution_metadata->state_decision_members[node_index]];
          const auto step=std::find_if(region.steps.begin(),region.steps.end(),
              [&](const auto &entry) { return entry.selection==node_index; });
          ops::require(step!=region.steps.end(),"STATE_DECISION_CONTRACT");
          // Only generated predicate/previous-state descriptors lack payload.
          // Real input validity and scalar branches keep canonical NULL_INPUT
          // checks before domain validation, including simultaneous faults.
          for (std::size_t j=0;j<node.parent_count;++j)
            if (j!=0 && static_cast<std::int8_t>(j)!=step->previous_argument)
              ops::require(arguments[j].size()==0 || arguments[j].shape.rank==0 ||
                  arguments[j].data!=nullptr,"NULL_INPUT");
        }
        const bool private_stream_filter = fuse_pairs &&
            gaussian_filter_stream_geometry(program, node_index, arena_shapes);
        if (private_stream_filter)
          for (std::size_t j = 1; j < node.parent_count; ++j)
            ops::require(arguments[j].size() == 0 || arguments[j].shape.rank == 0 ||
                         arguments[j].data != nullptr, "NULL_INPUT");
        auto prepared = (private_scatter_consumer || private_stream_filter ||
            (private_decision && spec.op==ops::Op::state_select))
            ? ops::prepare_geometry(spec,arguments.data(),node.parent_count)
            : ops::prepare(spec,arguments.data(),node.parent_count);
        if (fuse_pairs && gaussian_stream_producer_geometry(program, node_index, arena_shapes)) {
          ops::validate_payload_structure(prepared);
          ops::Audit validation_audit;
          ops::gaussian_density_stream_validate(prepared, scratch.workspace,
                                                ops::Isa::automatic, validation_audit);
          accumulate_child(scratch, validation_audit);
          ops::Value geometry;
          geometry.kind = prepared.output_kind; geometry.shape = prepared.output_shape;
          geometry.set_contiguous_strides();
          scratch.values[node_index] = geometry;
          continue;
        }
        if (private_stream_filter) {
          ops::validate_payload_structure(prepared);
          const auto source = static_cast<std::size_t>(program.execution_metadata->gaussian_filter_source[node_index]);
          const auto &producer = program.nodes[source];
          ops::Arguments<ops::Value> gaussian_args(producer.parent_count);
          for (std::size_t j = 0; j < producer.parent_count; ++j)
            gaussian_args[j] = scratch.values[producer.parents[j]];
          const auto gaussian = ops::prepare(*program.execution_metadata->specs[source],
                                             gaussian_args.data(), producer.parent_count);
          ops::Output tail;
          tail.kind = ops::Kind::number; tail.shape = ops::matrix_shape(1, prepared.output_shape.dim[1]);
          ops::require(node.storage == StorageKind::numeric &&
              tail.shape.size() <= scratch.slot_capacity(node.storage, node.slot), "GRAPH_OUTPUT_TOO_LARGE");
          tail.data = scratch.numeric_slot(node.slot);
          ops::Audit tail_audit;
          ops::gaussian_filter_tail(gaussian, prepared, tail, scratch.workspace,
                                    ops::Isa::automatic, tail_audit);
          accumulate_child(scratch, tail_audit);
          ++scratch.gaussian_filter_streams;
          ops::Value geometry;
          geometry.kind = prepared.output_kind; geometry.shape = prepared.output_shape;
          geometry.set_contiguous_strides(); scratch.values[node_index] = geometry;
          continue;
        }
        if (!row_subviews.omit.empty() && row_subviews.omit[node_index]) {
          // Keep every logical prepare above. Only the materialized bytes are
          // replaced; external and completed independent roots outlive this view.
          const auto &source = scratch.values[row_subviews.source[node_index]];
          const auto first = row_subviews.first[node_index];
          ops::require(prepared.output_geometry_known && prepared.output_kind == ops::Kind::number &&
              node.storage == StorageKind::numeric && prepared.output_shape == arena_shapes[node_index] &&
              prepared.output_shape.size() <= max_window && source.kind == ops::Kind::number &&
              row_subview_backing_contiguous(source) &&
              source.data && first <= source.shape.dim[0] &&
              prepared.output_shape.dim[0] <= source.shape.dim[0] - first &&
              source.shape.dim[1] == prepared.output_shape.dim[1], "ROW_SUBVIEW_CONTRACT");
          auto view = source;
          view.shape = prepared.output_shape;
          view.data = static_cast<const double *>(source.data) + first * static_cast<std::size_t>(source.stride[0]);
          scratch.values[node_index] = view;
          continue;
        }
        if (private_decision) {
          const auto &region=program.execution_metadata->state_decisions[
              program.execution_metadata->state_decision_members[node_index]];
          // No placeholder payload is read. Preserve original scalar/domain,
          // mask, rank, geometry, storage and size failures at this exact node.
          ops::validate_payload_structure(prepared);
          ops::require(prepared.output_shape.size()<=max_window,"GRAPH_OUTPUT_TOO_LARGE");
          ops::require(node.storage==(prepared.output_kind==ops::Kind::mask ? StorageKind::mask :
              prepared.output_kind==ops::Kind::integer ? StorageKind::integer : StorageKind::numeric),
              "GRAPH_STORAGE_KIND");
          if (node_index!=region.steps.back().selection) {
            ops::Value value; value.kind=prepared.output_kind; value.shape=prepared.output_shape;
            value.set_contiguous_strides(); scratch.values[node_index]=value;
            continue;
          }
        }
        if (fuse_pairs && filter_tail_geometry(program, node_index, arena_shapes)) {
          ops::Output tail;
          tail.kind = ops::Kind::number;
          tail.shape = ops::matrix_shape(1, prepared.output_shape.dim[1]);
          ops::require(node.storage == StorageKind::numeric &&
              tail.shape.size() <= scratch.slot_capacity(node.storage, node.slot),
              "GRAPH_OUTPUT_TOO_LARGE");
          tail.data = scratch.numeric_slot(node.slot);
          ops::Audit tail_audit;
          ops::hmm_filter_tail(prepared, tail, scratch.workspace, tail_audit);
          accumulate_child(scratch, tail_audit);
          ops::Value geometry;
          geometry.kind = prepared.output_kind;
          geometry.shape = prepared.output_shape;
          geometry.set_contiguous_strides();
          scratch.values[node_index] = geometry;
          continue;
        }
        if (fuse_pairs && typed_output &&
            static_cast<std::int32_t>(node_index)==program.execution_metadata->split_eigh_source &&
            ops::symmetric_eigh_split_scratch_doubles(prepared.args[0].shape.dim[0])) {
          std::array<ops::Output,2> destinations;
          std::array<ops::Prepared,2> projections;
          const std::array<std::int32_t,2> ids{program.execution_metadata->split_eigh_values,
                                             program.execution_metadata->split_eigh_vectors};
          // Projection preparation is geometry-only: no fictional record is
          // exposed, read, or kept alive after this validation.
          ops::Value geometry;
          geometry.kind=prepared.output_kind; geometry.shape=prepared.output_shape;
          geometry.set_contiguous_strides();
          for (std::size_t j=0;j<ids.size();++j) {
            const auto id=ids[j];
            const auto structure=ops::validate_structure(*program.execution_metadata->specs[id],
                &geometry,1,UINT32_MAX,0);
            ops::require(structure.geometry_known,"RESULT_CAPACITY_MISMATCH");
            const auto root=program.execution_metadata->root_destinations[id];
            ops::require(root>=0,"RESULT_CAPACITY_MISMATCH");
            const auto &schema=program.root_outputs[root];
            const auto &slot=layout->slots[(result_row+row)*output_columns+root];
            ops::require(schema.dtype==OutputDType::float64 && schema.rank==structure.output_shape.rank &&
                structure.output_shape.size()<=slot.capacity,"RESULT_CAPACITY_MISMATCH");
            auto &destination=destinations[j];
            destination.kind=ops::Kind::number; destination.shape=structure.output_shape;
            destination.order=layout->order(root);
            destination.data=static_cast<std::uint8_t *>(output)+slot.byte_offset;
            projections[j].output_shape=structure.output_shape;
            projections[j].output_kind=ops::Kind::number;
          }
          ops::Audit kernel_audit;
          ops::symmetric_eigh_split_pair(prepared,destinations[0],destinations[1],scratch.workspace,kernel_audit);
          accumulate_child(scratch,kernel_audit);
          for (std::size_t j=0;j<ids.size();++j) {
            scratch.values[ids[j]]=output_value(projections[j],destinations[j],max_window);
            audit.direct_output_bytes+=destinations[j].shape.size()*sizeof(double);
          }
          split_eigh_completed=true;
          continue;
        }
        if (inherited_positions && pointwise_status(spec) && prepared.output_shape.rank != 1) {
          scratch.statuses[node_index] = inherited_positions;
          scratch.values[node_index] = validate_failed_node(program, inputs, node, start, length, scratch);
          continue;
        }
        if (inherited_positions && mapped_status(spec.op))
          map_position_status(scratch, node_index, node, prepared);

        if (fuse_pairs && spec.op==ops::Op::eigenvalues &&
            program.execution_metadata->fused_from[node_index]>=0)
          prepared.borrowed=false;
        if (prepared.borrowed) {
          scratch.values[node_index] = prepared.view;
          continue;
        }

        if (private_where_sums && square_reduce_geometry(program, node_index, arena_shapes)) {
          const auto &region = program.execution_metadata->square_reductions[
              program.execution_metadata->square_members[node_index]];
          if (node_index != region.reduction) {
            if (node_index == region.subtraction) {
              const auto &selection = program.nodes[region.selection];
              const auto &otherwise = program.nodes[selection.parents[2]];
              const auto value = otherwise.kind == NodeKind::constant ? otherwise.constant : parameters[otherwise.input_index];
              ops::require(prepared.output_shape.dim[0] <= scratch.slot_capacity(node.storage, node.slot),
                           "GRAPH_OUTPUT_TOO_LARGE");
              auto *cache = scratch.numeric_slot(node.slot);
              execute_square_reduce_rows(prepared, scratch.values[selection.parents[0]], value, cache, scratch);
              ops::Value geometry;
              geometry.kind = prepared.output_kind; geometry.shape = prepared.output_shape;
              geometry.set_contiguous_strides(); geometry.data = cache;
              scratch.values[node_index] = geometry;
              ++audit.fused_pointwise_calls;
            } else {
              if (node_index == region.selection) ops::validate_payload_structure(prepared);
              scratch.values[node_index] = scratch.values[region.subtraction];
            }
            continue;
          }
        }

        if (private_where_sums && scan_reduce_geometry(program, node_index, arena_shapes)) {
          const auto &region = program.execution_metadata->scan_reductions[
              program.execution_metadata->scan_members[node_index]];
          if (node_index != region.reduction) {
            if (node_index == region.nav) {
              const auto bound_scalar = [&](std::uint32_t id) {
                const auto &source = program.nodes[id];
                return source.kind == NodeKind::constant ? source.constant : parameters[source.input_index];
              };
              ops::require(prepared.output_shape.dim[0] <= scratch.slot_capacity(node.storage, node.slot),
                           "GRAPH_OUTPUT_TOO_LARGE");
              auto *cache = scratch.numeric_slot(node.slot);
              const auto errors = execute_scan_reduce_rows(prepared,
                  bound_scalar(program.nodes[region.peak].parents[2]),
                  bound_scalar(program.nodes[region.subtraction].parents[0]), cache, scratch);
              ops::Value geometry;
              geometry.kind = prepared.output_kind; geometry.shape = prepared.output_shape;
              geometry.set_contiguous_strides(); geometry.data = cache; geometry.scalar = double(errors);
              scratch.values[node_index] = geometry;
              ++audit.fused_pointwise_calls;
            } else {
              const auto &cache = scratch.values[region.nav];
              if (node_index == region.division && (unsigned(cache.scalar) & 1))
                throw ops::Error("DIVIDE_BY_ZERO");
              scratch.values[node_index] = cache;
            }
            continue;
          }
        }

        if (private_where_sums && where_sum_geometry(program, node_index, arena_shapes)) {
          ops::require(prepared.output_shape.dim[0] <= scratch.slot_capacity(node.storage, node.slot),
                       "GRAPH_OUTPUT_TOO_LARGE");
          auto *cache = scratch.numeric_slot(node.slot);
          execute_where_sum_rows(prepared, cache, scratch);
          ops::Value geometry;
          geometry.kind = prepared.output_kind; geometry.shape = prepared.output_shape;
          geometry.set_contiguous_strides(); geometry.data = cache;
          scratch.values[node_index] = geometry;
          ++audit.fused_pointwise_calls;
          continue;
        }

        if (summary_fusion && matmul_summary_geometry(program,node_index,arena_shapes)) {
          const auto group=program.execution_metadata->matmul_summary_producer[node_index];
          const auto &region=program.execution_metadata->matmul_summaries[group];
          ops::MatmulStatisticsAccuracy accuracy;
          accuracy.allowed=region.quantiles.empty() ? region.moments : region.trajectory_library;
          accuracy.maximum_ddof=region.maximum_ddof;
          std::copy(region.linear_scale.begin(),region.linear_scale.end(),accuracy.linear_scale);
          std::copy(region.sqrt_scale.begin(),region.sqrt_scale.end(),accuracy.sqrt_scale);
          auto *total=scratch.matmul_statistics_cache.data()+scratch.matmul_statistics_offsets[group];
          auto *squared=total+prepared.output_shape.dim[0];
          auto *aux=scratch.matmul_statistics_cache.data()+scratch.matmul_statistics_offsets.back();
          ops::Audit statistics_audit;
          const auto checkpoint=[](void *context) {
            auto &state=*static_cast<Scratch *>(context);
            native::check_deadline(state.deadline);
            if (state.cancelled && state.cancelled->load()) throw native::Timeout("native statistics cancelled");
          };
          if (region.quantiles.empty()) {
            scratch.summary_source_scans+=ops::matmul_statistics(prepared,total,squared,aux,
                accuracy,statistics_audit,&scratch,checkpoint);
            if (statistics_audit.covariance_syrk_calls) scratch.summary_source_scans+=2;
          } else {
            std::vector<ops::Value> probabilities; probabilities.reserve(region.quantiles.size());
            std::vector<ops::Output> roots; roots.reserve(region.quantiles.size());
            std::vector<ops::MatmulQuantileOutput> destinations;
            for (const auto child:region.quantiles) {
              const auto qid=program.nodes[child].parents[1];
              const auto &qnode=program.nodes[qid];
              // A later static input has no logical prepare or interval slicing
              // to execute early. Earlier input-backed views are already checked.
              probabilities.push_back(qid>=node_index ? inputs[qnode.input_index] : scratch.values[qid]);
              if (matmul_quantile_root(program,child)) {
                const auto root=program.execution_metadata->root_destinations[child];
                const auto &slot=layout->slots[(result_row+row)*output_columns+root];
                ops::Output destination; destination.kind=ops::Kind::number;
                destination.shape=ops::matrix_shape(prepared.output_shape.dim[0],probabilities.back().size());
                // A malformed raw-native layout must still fail at the original
                // logical root. Do not write beyond its capacity or throw early.
                if (destination.shape.size()<=slot.capacity) {
                  destination.data=static_cast<std::uint8_t *>(output)+slot.byte_offset;
                  destination.order=layout->order(root);
                }
                roots.push_back(destination);
                destinations.push_back({&probabilities.back(),nullptr,&roots.back()});
              } else destinations.push_back({&probabilities.back(),
                  total+matmul_quantile_offset(program,region,child,arena_shapes),nullptr});
            }
            scratch.summary_source_scans+=ops::matmul_quantiles(prepared,total,squared,aux,
                accuracy,region.quantile_scale,destinations.data(),destinations.size(),
                statistics_audit,&scratch,checkpoint);
          }
          accumulate_child(scratch,statistics_audit);
          ++audit.fused_pointwise_calls;
          auto placeholder=prepared.args[0];
          placeholder.data=&scratch.values[node_index].scalar;
          placeholder.kind=prepared.output_kind; placeholder.shape=prepared.output_shape;
          placeholder.set_contiguous_strides();
          scratch.values[node_index]=placeholder;
          continue;
        }
        if (fuse_pairs && deferred_pair_geometry(program,node_index,arena_shapes)) {
          const auto pair_consumer=program.execution_metadata->deferred_to[node_index];
          if (spec.op==ops::Op::take_axis && program.nodes[pair_consumer].opcode==static_cast<std::uint16_t>(ops::Op::scatter_rows)) {
            ops::require(prepared.output_shape.size()<=max_window,"GRAPH_OUTPUT_TOO_LARGE");
            // Raw C++ callers can supply the wrong payload dtype. Preserve the
            // producer's storage failure before downstream scatter validation.
            ops::require(prepared.output_kind==ops::Kind::number,"GRAPH_STORAGE_KIND");
          }
          if (const auto extent=deferred_broadcast_extent(program,node_index,arena_shapes)) {
            const auto argument=std::size_t(program.execution_metadata->deferred_broadcast_argument[node_index]);
            ops::require(prepared.output_shape.size()<=max_window,"GRAPH_OUTPUT_TOO_LARGE");
            ops::require(node.storage==StorageKind::numeric && prepared.output_kind==ops::Kind::number,
                         "GRAPH_STORAGE_KIND");
            ops::require(extent<=scratch.slot_capacity(node.storage,node.slot),"GRAPH_OUTPUT_TOO_LARGE");
            auto *saved=scratch.numeric_slot(node.slot);
            if (prepared.output_shape.size()) {
              for (std::size_t column=0;column<extent;++column) saved[column]=prepared.args[argument].f(column);
              scratch.algorithm_copy_bytes+=extent*sizeof(double);
            }
          }
          if (spec.op==ops::Op::scatter_rows) {
            ops::Audit validation_audit;
            ops::validate_scatter_indices(prepared,scratch.workspace,validation_audit);
            accumulate_child(scratch,validation_audit);
          }
          // A geometry-only placeholder has exactly one consumer, handled below.
          // prepare still validates the producer (including take-axis indices).
          auto placeholder = prepared.args[0];
          // Scalar-first arithmetic has no array payload in argument zero.
          // Select an actual array backing for the geometry-only descriptor;
          // the sole fused consumer never reads this placeholder's values.
          for (std::size_t j=0;j<prepared.count;++j)
            if (prepared.args[j].shape.rank>0) { placeholder.data=prepared.args[j].data; break; }
          // Concatenation preparation never consumes numeric array elements.
          // Keep a non-null descriptor even when its first leaf is empty.
          if (spec.op == ops::Op::concatenate) {
            // Raw native callers still receive the original storage error here,
            // before an independent later node can fail. Python dtype binding
            // alone cannot prove this native execution contract.
            ops::require(prepared.output_shape.size() <= max_window, "GRAPH_OUTPUT_TOO_LARGE");
            ops::require(prepared.output_kind == ops::Kind::number, "GRAPH_STORAGE_KIND");
            placeholder.data = &scratch.values[node_index].scalar;
          }
          placeholder.kind = prepared.output_kind;
          placeholder.shape = prepared.output_shape;
          placeholder.set_contiguous_strides();
          scratch.values[node_index] = placeholder;
          continue;
        }

        const bool matrix_chain_step=private_matmul_chain && matmul_scalar_chain_member(program,node_index);
        const bool loaned_argmax = static_cast<std::int32_t>(node_index) == state_loan.source;
        const auto destination_root = loaned_argmax
            ? program.execution_metadata->root_destinations[state_loan.terminal]
            : matrix_chain_step ? matmul_chain_destination(program,node_index)
            : program.execution_metadata->root_destinations[node_index];
        bool matrix_chain_separated=false;
        ops::Output native_output;
        native_output.kind = prepared.output_kind;
        native_output.shape = prepared.output_shape;

        if (prepared.output_shape.rank >= 1) {
          ops::require(prepared.output_shape.size() <= max_window,
                       "GRAPH_OUTPUT_TOO_LARGE");
          if (!(typed_output && destination_root>=0)) {
            ops::require(node.storage!=StorageKind::inline_value,"GRAPH_STORAGE_KIND");
            ops::require(prepared.output_shape.size()<=scratch.slot_capacity(node.storage,node.slot),
                         "GRAPH_OUTPUT_TOO_LARGE");
          }
          if (prepared.output_kind == ops::Kind::mask) {
            ops::require(node.storage == StorageKind::mask, "GRAPH_STORAGE_KIND");
            native_output.data = scratch.masks.empty() ? nullptr : scratch.masks.data() + scratch.arena.offsets[1][node.slot];
          } else if (prepared.output_kind == ops::Kind::integer) {
            ops::require(node.storage == StorageKind::integer, "GRAPH_STORAGE_KIND");
            native_output.data = scratch.integers.empty() ? nullptr : scratch.integers.data() + scratch.arena.offsets[2][node.slot];
          } else {
            ops::require(prepared.output_kind == ops::Kind::number,
                         "GRAPH_STORAGE_KIND");
            ops::require(node.storage == StorageKind::numeric,
                         "GRAPH_STORAGE_KIND");
            native_output.data = scratch.numeric.empty() ? nullptr : scratch.numeric_slot(node.slot);
          }
        } else {
          ops::require(node.storage == StorageKind::inline_value,
                       "GRAPH_STORAGE_KIND");
        }

        // Root storage belongs to this call; downstream consumers borrow it.
        bool symmetric_f_root = false;
        if (typed_output && prepared.output_shape.rank > 0 && destination_root >= 0) {
          const auto &schema = program.root_outputs[destination_root];
          const auto &slot = layout->slots[(result_row + row) * output_columns + destination_root];
          ops::require(prepared.output_shape.rank == schema.rank &&
              prepared.output_shape.size() <= slot.capacity, "RESULT_CAPACITY_MISMATCH");
          const auto expected = schema.dtype == OutputDType::boolean ? ops::Kind::mask
              : schema.dtype == OutputDType::int64 ? ops::Kind::integer : ops::Kind::number;
          ops::require(prepared.output_kind == expected, "RESULT_DTYPE_MISMATCH");
          native_output.data = static_cast<std::uint8_t *>(output) + slot.byte_offset;
          native_output.order = (matrix_chain_step || program.roots[destination_root] == node_index)
              ? layout->order(destination_root) : ops::MemoryOrder::c;
          symmetric_f_root = native_output.order == ops::MemoryOrder::f &&
              (spec.op == ops::Op::covariance || spec.op == ops::Op::correlation);
          if (symmetric_f_root) {
            ops::require(prepared.output_shape.rank == 2 &&
                prepared.output_shape.dim[0] == prepared.output_shape.dim[1], "RESULT_CAPACITY_MISMATCH");
            // Only independent single-argument covariance/correlation survived F-root
            // admission above. Keep the existing C writer and publish F strides
            // after it has completed its bit-identical symmetric mirror.
            native_output.order = ops::MemoryOrder::c;
          }
          audit.direct_output_bytes += prepared.output_shape.size() * output_itemsize(schema.dtype);
        }
        if (loaned_argmax) {
          ops::require(native_output.kind == ops::Kind::integer && native_output.shape.rank == 1 &&
              native_output.shape.size() == state_loan.count && native_output.data == state_loan.data,
              "STATE_ROOT_LOAN_CONTRACT");
          state_loan.active = true;
        }
        bool fused = false;
        if (private_decision) {
          const auto &region=program.execution_metadata->state_decisions[
              program.execution_metadata->state_decision_members[node_index]];
          if (state_loan.source == static_cast<std::int32_t>(region.source))
            execute_state_decision<true>(program,region,scratch.values,native_output,scratch.workspace,
                audit,scratch.deadline,scratch.cancelled,&state_loan,scratch.algorithm_copy_bytes);
          else
            execute_state_decision<false>(program,region,scratch.values,native_output,scratch.workspace,
                audit,scratch.deadline,scratch.cancelled,nullptr,scratch.algorithm_copy_bytes);
          fused=true;
        }
        if (private_where_sums && square_reduce_geometry(program, node_index, arena_shapes)) {
          const auto &region = program.execution_metadata->square_reductions[
              program.execution_metadata->square_members[node_index]];
          const auto *cache = static_cast<const double *>(scratch.values[region.subtraction].data);
          for (std::size_t j = 0; j < prepared.output_shape.size(); ++j) native_output.set(j, cache[j]);
          if (native_output.data != cache)
            scratch.algorithm_copy_bytes += prepared.output_shape.size() * sizeof(double);
          fused = true;
        }
        if (private_where_sums && scan_reduce_geometry(program, node_index, arena_shapes)) {
          const auto &region = program.execution_metadata->scan_reductions[
              program.execution_metadata->scan_members[node_index]];
          const auto &cache = scratch.values[region.nav];
          if (unsigned(cache.scalar) & 2) throw ops::Error("NON_FINITE_RESULT");
          const auto *totals = static_cast<const double *>(cache.data);
          for (std::size_t j = 0; j < prepared.output_shape.size(); ++j) native_output.set(j, totals[j]);
          if (native_output.data != cache.data)
            scratch.algorithm_copy_bytes += prepared.output_shape.size() * sizeof(double);
          fused = true;
        }
        if (private_where_sums && !program.execution_metadata->where_sum_from.empty() &&
            program.execution_metadata->where_sum_from[node_index] >= 0 &&
            where_sum_geometry(program,
                std::size_t(program.execution_metadata->where_sum_from[node_index]), arena_shapes)) {
          const auto source = std::size_t(program.execution_metadata->where_sum_from[node_index]);
          const auto *cache = static_cast<const double *>(scratch.values[source].data);
          for (std::size_t j = 0; j < prepared.output_shape.size(); ++j)
            native_output.set(j, cache[j]);
          scratch.algorithm_copy_bytes += prepared.output_shape.size() * sizeof(double);
          fused = true;
        }
        if (summary_fusion && !program.execution_metadata->matmul_summary_consumer.empty()) {
          const auto group=program.execution_metadata->matmul_summary_consumer[node_index];
          if (group>=0 && matmul_summary_geometry(program,
              program.execution_metadata->matmul_summaries[group].producer,arena_shapes)) {
            const auto count=prepared.args[0].shape.dim[1];
            const auto rows=prepared.args[0].shape.dim[0];
            const auto *total=scratch.matmul_statistics_cache.data()+scratch.matmul_statistics_offsets[group];
            const auto *squared=total+rows;
            if (spec.op==ops::Op::quantiles_axis) {
              if (prepared.output_shape.size()) {
                ops::require(count>0,"INSUFFICIENT_SAMPLE");
                if (!matmul_quantile_root(program,node_index)) {
                  const auto &region=program.execution_metadata->matmul_summaries[group];
                  const auto *quantiles=total+matmul_quantile_offset(program,region,node_index,arena_shapes);
                  for (std::size_t i=0;i<prepared.output_shape.size();++i) native_output.set(i,quantiles[i]);
                }
              }
            } else {
              const auto ddof=spec.op==ops::Op::variance_axis ? std::size_t(prepared.args[2].scalar) : 0;
              if (spec.op!=ops::Op::sum_axis) ops::require(count>ddof,"INSUFFICIENT_SAMPLE");
              for (std::size_t i=0; i<rows; ++i) {
                const auto value=spec.op==ops::Op::sum_axis ? total[i] :
                    spec.op==ops::Op::mean_axis ? total[i]/double(count) : squared[i]/double(count-ddof);
                if (spec.op==ops::Op::variance_axis) ops::require(std::isfinite(value),"NON_FINITE_RESULT");
                native_output.set(i,value);
              }
            }
            fused=true;
          }
        }
        if (fuse_pairs && program.execution_metadata->fused_from[node_index] >= 0 &&
            deferred_pair_geometry(program,program.execution_metadata->fused_from[node_index],arena_shapes)) {
          const auto first_id=program.execution_metadata->fused_from[node_index];
          const auto &first=program.nodes[first_id];
          ops::Arguments<ops::Value> values(first.parent_count);
          for (std::size_t j=0;j<first.parent_count;++j) values[j]=scratch.values[first.parents[j]];
          if (deferred_broadcast_extent(program,first_id,arena_shapes)) {
            const auto argument=std::size_t(program.execution_metadata->deferred_broadcast_argument[first_id]);
            auto &saved=values[argument];
            saved.kind=ops::Kind::number;
            saved.data=scratch.numeric_slot(first.slot);
            saved.stride={0,1,0};
          }
          const auto first_plan=ops::prepare(*program.execution_metadata->specs[first_id],values.data(),first.parent_count);
          if (first_plan.spec->op==ops::Op::symmetric_eigh) {
            ops::Audit pair_audit;
            ops::symmetric_eigh_values_pair(first_plan,native_output,scratch.workspace,pair_audit);
            accumulate_child(scratch,pair_audit);
          } else if (first_plan.spec->op==ops::Op::covariance) {
            ops::Audit pair_audit;
            const bool independent_root=program.roots.size()==1 &&
                program.roots[0]==node_index;
            ops::covariance_cholesky_pair(first_plan,prepared,native_output,
                                           scratch.workspace,pair_audit,independent_root);
            accumulate_child(scratch,pair_audit);
          } else {
            if (first_plan.spec->op==ops::Op::scatter_rows) {
              const auto take_id=program.execution_metadata->fused_from[first_id];
              if (take_id>=0 && deferred_pair_geometry(program,take_id,arena_shapes)) {
                const auto &take=program.nodes[take_id];
                ops::Arguments<ops::Value> take_values(take.parent_count);
                for (std::size_t j=0;j<take.parent_count;++j) take_values[j]=scratch.values[take.parents[j]];
                const auto take_plan=ops::prepare(*program.execution_metadata->specs[take_id],take_values.data(),take.parent_count);
                ops::scatter_argmax_pair(first_plan,prepared,native_output,&take_plan);
              } else ops::scatter_argmax_pair(first_plan,prepared,native_output);
            } else if (first_plan.spec->op==ops::Op::take_axis && spec.op==ops::Op::scatter_rows) {
              ops::Audit pair_audit;
              ops::scatter_take_pair(first_plan,prepared,native_output,scratch.workspace,pair_audit);
              accumulate_child(scratch,pair_audit);
            } else if (first_plan.spec->op==ops::Op::take_axis) ops::take_reduce(first_plan,prepared,native_output);
            else ops::elementwise_pair(first_plan,prepared,native_output,
                node.parents[0]==std::uint32_t(first_id),
                matrix_chain_accelerated && matrix_chain_step &&
                    program.execution_metadata->root_destinations[node_index]>=0 ? &matrix_chain_separated : nullptr,
                matrix_chain_error);
            ++audit.fused_pointwise_calls;
          }
          fused=true;
        }
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

        if constexpr (!Observe && !DeferErrors) {
          if (!fused && !prebound_inputs && !scratch.isolate_errors && !input_failures &&
              typed_output && (destination_root>=0 || program.execution_metadata->solve_pair_private_source>=0) &&
              static_cast<std::int32_t>(node_index)==program.execution_metadata->solve_pair_first) {
            fused=reuse_solve_pair(program,inputs,result_row+row,output,*layout,
                                  prepared,native_output,scratch,max_window,audit);
            solve_pair_completed=fused;
          }
        }
        if constexpr (!Observe) {
          if (!fused && !prebound_inputs && typed_output && destination_root >= 0 &&
              spec.op == ops::Op::correlation && node.parent_count == 1 &&
              prepared.args[0].shape.rank == 2 && prepared.args[0].shape.dim[0] >= 2 &&
              prepared.output_shape.rank == 2 && prepared.output_shape.dim[0] > 0) {
            fused = reuse_covariance_root(program, node_index, result_row + row,
                                          output, *layout, native_output, scratch);
          }
        }

        if (inherited_positions && pointwise_status(spec)) {
          execute_pointwise_with_status(node, node_index, prepared, native_output, scratch);
        } else if (!fused) {
          ops::Audit operator_audit;
          bool accelerated = false;
          if (fuse_pairs && !program.execution_metadata->concat_tree_roots.empty() &&
              program.execution_metadata->concat_tree_roots[node_index]) {
            if constexpr (!Observe && !DeferErrors) {
              if (!prebound_inputs && !scratch.isolate_errors && !input_failures && layout &&
                  concat_solve_column_order(program, node_index, native_output, *layout, result_row + row, scratch))
                native_output.order = ops::MemoryOrder::f;
            }
            execute_concat_tree(program, node_index, native_output, scratch);
            accelerated = true;
          }
          if constexpr (!Observe) {
            if (!prebound_inputs && !scratch.isolate_errors && !input_failures &&
                native_output.order == ops::MemoryOrder::f &&
                independent_cholesky_f_root(program, node_index)) {
              ops::cholesky_f_root(prepared, native_output, scratch.workspace,
                                   ops::Isa::automatic, operator_audit);
              accelerated = true;
            }
            if (!prebound_inputs && !scratch.isolate_errors && !input_failures &&
                spec.op == ops::Op::matmul && independent_matmul_root(program, node_index))
              accelerated = ops::try_accelerate_matmul_root(
                  prepared, native_output, scratch.workspace, operator_audit, &scratch,
                  [](void *context) {
                    auto &state=*static_cast<Scratch *>(context);
                    native::check_deadline(state.deadline);
                    if(state.cancelled&&state.cancelled->load())throw native::Timeout("native matmul cancelled");
                  }, scratch.expanded_matmul_roots);
            ops::MatmulTerminalProof terminal_proof;
            if (matrix_chain_step && spec.op==ops::Op::matmul &&
                matmul_region_certificate(program,prepared,inputs,parameters,
                                         scratch,matrix_chain_error,terminal_proof)) {
              double reduced_error=0.0;
              ops::MatmulTerminalProof reduced_proof;
              if (matmul_region_certificate(program,prepared,inputs,parameters,scratch,
                    reduced_error,reduced_proof,true,&terminal_proof)) {
                accelerated=ops::try_accelerate_matmul_terminal_float(
                    prepared,native_output,scratch.workspace,operator_audit,&scratch,
                    [](void *context) { matmul_chain_checkpoint(*static_cast<Scratch *>(context)); },
                    scratch.expanded_matmul_roots,&reduced_proof);
                if (accelerated) matrix_chain_error=reduced_error;
              }
              if (!accelerated)
                accelerated=ops::try_accelerate_matmul_root(
                    prepared,native_output,scratch.workspace,operator_audit,&scratch,
                    [](void *context) { matmul_chain_checkpoint(*static_cast<Scratch *>(context)); },
                    scratch.expanded_matmul_roots,&terminal_proof);
              matrix_chain_accelerated=accelerated;
            }
            if (!prebound_inputs && !scratch.isolate_errors && !input_failures &&
                independent_covariance_root(program, node_index))
              accelerated = ops::try_accelerate_covariance(
                  prepared, native_output, scratch.workspace, operator_audit,
                  program.roots.size() == 2 && spec.op == ops::Op::covariance);
            if (!prebound_inputs && !scratch.isolate_errors && !input_failures &&
                independent_transcendental_root(program, node_index))
              accelerated = ops::try_accelerate_transcendental(
                  prepared, native_output, scratch.workspace, operator_audit, true);
          }
          if constexpr (!Observe && !DeferErrors) {
            if (!accelerated && !scratch.isolate_errors && !input_failures && typed_output &&
                spec.op == ops::Op::solve_multi)
              accelerated = reuse_private_solve(program, node_index, prepared, native_output,
                                                 scratch, operator_audit, !prebound_inputs);
            if (!accelerated && !scratch.isolate_errors && !input_failures && typed_output &&
                !program.execution_metadata->solve_copy_roots.empty() &&
                program.execution_metadata->solve_copy_roots[node_index])
              accelerated = ops::try_solve_accelerate_copy(prepared, native_output,
                                                          scratch.workspace, operator_audit);
            if (!accelerated && !prebound_inputs && !scratch.isolate_errors &&
                !input_failures && typed_output && scratch.allow_workspace_loan)
              accelerated = loan_latent_workspace(program, node_index, result_row + row,
                  output, *layout, prepared, native_output, scratch, operator_audit);
          }
          if constexpr (!Observe && !DeferErrors) {
            const bool private_arena = !program.execution_metadata->solve_library_consumer.empty() &&
                program.execution_metadata->solve_library_consumer[node_index] >= 0;
            if (!accelerated && !prebound_inputs && !scratch.isolate_errors && !input_failures &&
                (private_arena || (typed_output && destination_root >= 0)))
              accelerated = ops::try_dense_elementwise(prepared, native_output,
                  scratch.workspace, operator_audit, private_arena);
          }
          if (!accelerated)
            ops::execute(prepared, native_output, scratch.workspace,
                         ops::Isa::automatic, operator_audit);
          scratch.eigen_rotation_calls += operator_audit.eigen_rotation_calls;
          scratch.highway_vqsort_calls += operator_audit.highway_vqsort_calls;
          scratch.eigen_matvec_calls += operator_audit.eigen_matvec_calls;
          scratch.eigen_covariance_calls += operator_audit.eigen_covariance_calls;
          scratch.eigen_density_calls += operator_audit.eigen_density_calls;
          scratch.accelerate_covchol_calls += operator_audit.accelerate_covchol_calls;
          scratch.accelerate_matmul_calls += operator_audit.accelerate_matmul_calls;
          scratch.matmul_zero_calls += operator_audit.matmul_zero_calls;
          scratch.accelerate_covariance_attempts += operator_audit.accelerate_covariance_attempts;
          scratch.accelerate_covariance_calls += operator_audit.accelerate_covariance_calls;
          scratch.covariance_syrk_calls += operator_audit.covariance_syrk_calls;
          scratch.covariance_replayed_cells += operator_audit.covariance_replayed_cells;
          scratch.covariance_fallbacks += operator_audit.covariance_fallbacks;
          scratch.covariance_zero_calls += operator_audit.covariance_zero_calls;
          scratch.accelerate_vforce_calls += operator_audit.accelerate_vforce_calls;
          scratch.accelerate_lu_attempts += operator_audit.accelerate_lu_attempts;
          scratch.accelerate_lu_calls += operator_audit.accelerate_lu_calls;
          scratch.accelerate_lu_fallbacks += operator_audit.accelerate_lu_fallbacks;
          scratch.solve_residual_materialization_bytes += operator_audit.solve_residual_materialization_bytes;
          scratch.covariance_panel_materialization_bytes += operator_audit.covariance_panel_materialization_bytes;
          scratch.native_library_workspace_reserve_bytes = std::max(
              scratch.native_library_workspace_reserve_bytes, operator_audit.native_library_workspace_reserve_bytes);
          scratch.algorithm_copy_bytes += operator_audit.algorithm_copy_bytes;
          audit.vector_elements += operator_audit.vector_elements;
        }
        if (matrix_chain_step && program.execution_metadata->root_destinations[node_index]>=0) {
          const auto root=static_cast<std::size_t>(program.execution_metadata->root_destinations[node_index]);
          if (matrix_chain_accelerated && !matrix_chain_separated)
            replay_matmul_chain_ambiguity(program,native_output,matrix_chain_error,scratch,
                                         matmul_chain_for_root(program,root));
          matrix_root_guard.buffers[root].published=true;
        }
        if (node_index == state_loan.terminal && state_loan.active) state_loan.active = false;
        if (symmetric_f_root) native_output.order = ops::MemoryOrder::f;
        scratch.values[node_index] =
            output_value(prepared, native_output, max_window);
        if constexpr (Observe)
          (*value_observer)(spec.op, scratch.values[node_index], true);
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
        if (!scratch.isolate_errors) {
          // This add/subtract/multiply-only region has no numeric throwing
          // prefix. A structural/library rejection cannot publish its private
          // intermediate matrix through a caller-owned failed result buffer.
          if (private_matmul_chain) matrix_root_guard.clear_unpublished();
          scratch.failed_node = node_index;
          throw;
        }
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
      if (typed_output) {
        const auto root = row * output_columns + root_index;
        const auto &schema = program.root_outputs[root_index];
        const auto &slot = layout->slots[(result_row + row) * output_columns + root_index];
        auto &shape = audit.result_shapes[root];
        shape = value.shape;
        audit.root_statuses[root] = static_cast<std::int16_t>(status);
        if (shape.rank < 0) {
          ops::require(status != 0, "RESULT_UNKNOWN_SUCCESS_SHAPE");
          continue;
        }
        ops::require(shape.rank == schema.rank && shape.size() <= slot.capacity, "RESULT_CAPACITY_MISMATCH");
        const auto expected = schema.dtype == OutputDType::boolean ? ops::Kind::mask
            : schema.dtype == OutputDType::int64 ? ops::Kind::integer : ops::Kind::number;
        ops::require(status || value.kind == expected, "RESULT_DTYPE_MISMATCH");
        auto *destination = static_cast<std::uint8_t *>(output) + slot.byte_offset;
        const auto status_base = slot.status_offset - layout->row_statuses[result_row];
        const auto elements = shape.size();
        ops::Output target;
        target.shape = shape; target.order = layout->order(root_index);
        if (!scratch.isolate_errors && shape.rank > 0 && value.contiguous(target.order)) {
          if (value.kind == ops::Kind::mask) {
            // Inspect raw uint8 inputs too: a contiguous buffer is not proof
            // that every byte satisfies the public boolean contract.
            std::uint8_t bits = 0;
            const auto *source = static_cast<const std::uint8_t *>(value.data);
            for (std::size_t i = 0; i < elements; ++i) bits |= source[i];
            ops::require(bits <= 1, "INVALID_MASK");
          }
          if (elements && destination != value.data) {
            const auto bytes = elements * output_itemsize(schema.dtype);
            std::memcpy(destination, value.data, bytes);
            audit.root_copy_bytes += bytes;
          }
          continue;
        }
        // A temporal window of an F input has gaps between columns, but each
        // first-axis run is contiguous. Pack those runs directly into F output;
        // do not scan both source and destination in logical row-major order.
        if (!scratch.isolate_errors && target.order == ops::MemoryOrder::f &&
            shape.rank > 1 && value.stride[0] == 1 && value.kind != ops::Kind::mask && elements) {
          const auto run = shape.dim[0], width = output_itemsize(schema.dtype);
          for (std::size_t column=0;column<elements/run;++column) {
            const auto logical = shape.rank == 2 ? column :
                (column % shape.dim[1]) * shape.dim[2] + column / shape.dim[1];
            const auto *source = static_cast<const std::uint8_t *>(value.data) + value.offset(logical)*static_cast<std::ptrdiff_t>(width);
            std::memcpy(destination + column*run*width,source,run*width);
          }
          audit.root_copy_bytes += elements*width;
          continue;
        }
        for (std::size_t offset = 0; offset < elements; ++offset) {
          const auto physical = target.offset(offset);
          auto failure = static_cast<std::int16_t>(status);
          if (!failure && TrackPositions)
            failure = position_error(scratch, program.roots[root_index], offset, offset + 1);
          if (!failure && (!value.shape.rank || value.data != destination))
            audit.root_copy_bytes += output_itemsize(schema.dtype);
          if (schema.dtype == OutputDType::float64) {
            const auto v = failure ? NAN : value.f(offset);
            std::memcpy(destination + physical * sizeof(v), &v, sizeof(v));
            if (!failure && scratch.isolate_errors && !std::isfinite(v)) failure = 4;
          } else if (schema.dtype == OutputDType::int64) {
            const std::int64_t v = failure ? 0 : value.i(offset);
            std::memcpy(destination + physical * sizeof(v), &v, sizeof(v));
          } else {
            const auto v = failure ? 0 : value.u(offset);
            ops::require(v <= 1, "INVALID_MASK");
            destination[physical] = static_cast<std::uint8_t>(v);
          }
          if (scratch.isolate_errors) audit.statuses[status_base + offset] = failure;
          audit.root_statuses[root] = std::max(audit.root_statuses[root], failure);
        }
        continue;
      }
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
  audit.operator_workspace_capacity_bytes = scratch.workspace.capacity_bytes() +
      scratch.matmul_statistics_cache.capacity()*sizeof(double) +
      scratch.matmul_statistics_offsets.capacity()*sizeof(std::size_t) +
      scratch.iteration_info.capacity() * sizeof(std::array<double, 3>);
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
    bytes += (state.iteration_state.capacity() + state.iteration_next.capacity()) * sizeof(double);
    bytes += state.iteration_inputs.capacity() * sizeof(ops::Value) + state.iteration_parameters.capacity() * sizeof(double);
    bytes += state.iteration_layout.slots.capacity() * sizeof(ResultSlot);
    bytes += (state.iteration_layout.row_bytes.capacity() + state.iteration_layout.row_statuses.capacity()) * sizeof(std::size_t);
    bytes += state.iteration_info.capacity() * sizeof(std::array<double, 3>);
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
  audit.highway_vqsort_calls = scratch.highway_vqsort_calls;
  audit.eigen_rotation_calls = scratch.eigen_rotation_calls;
  audit.eigen_matvec_calls = scratch.eigen_matvec_calls;
  audit.eigen_covariance_calls = scratch.eigen_covariance_calls;
  audit.eigen_density_calls = scratch.eigen_density_calls;
  audit.gaussian_filter_streams = scratch.gaussian_filter_streams;
  audit.accelerate_covchol_calls = scratch.accelerate_covchol_calls;
  audit.accelerate_matmul_calls = scratch.accelerate_matmul_calls;
  audit.matmul_zero_calls = scratch.matmul_zero_calls;
  audit.accelerate_covariance_attempts = scratch.accelerate_covariance_attempts;
  audit.accelerate_covariance_calls = scratch.accelerate_covariance_calls;
  audit.covariance_syrk_calls = scratch.covariance_syrk_calls;
  audit.covariance_replayed_cells = scratch.covariance_replayed_cells;
  audit.covariance_fallbacks = scratch.covariance_fallbacks;
  audit.covariance_zero_calls = scratch.covariance_zero_calls;
  audit.accelerate_vforce_calls = scratch.accelerate_vforce_calls;
  audit.accelerate_lu_attempts = scratch.accelerate_lu_attempts;
  audit.accelerate_lu_calls = scratch.accelerate_lu_calls;
  audit.accelerate_lu_fallbacks = scratch.accelerate_lu_fallbacks;
  audit.solve_residual_materialization_bytes = scratch.solve_residual_materialization_bytes;
  audit.covariance_panel_materialization_bytes = scratch.covariance_panel_materialization_bytes;
  audit.native_library_workspace_reserve_bytes = scratch.native_library_workspace_reserve_bytes;
  audit.algorithm_copy_bytes = scratch.algorithm_copy_bytes;
  audit.state_copy_bytes += scratch.state_copy_bytes;
  audit.operator_workspace_capacity_bytes += scratch.child_pointwise_capacity;
  audit.pointwise_workspace_capacity_bytes = std::max(audit.pointwise_workspace_capacity_bytes,
      scratch.child_pointwise_capacity);
  audit.root_copy_bytes += scratch.child_root_copy_bytes;
  audit.direct_output_bytes += scratch.child_direct_output_bytes;
  audit.vector_elements += scratch.child_vector_elements;
  audit.pointwise_tiles += scratch.child_pointwise_tiles;
  audit.fused_pointwise_calls += scratch.child_fused_pointwise_calls;
  for (const auto &cache : scratch.ordered)
    audit.order_scratch_capacity_bytes +=
        cache.values.capacity() * sizeof(double);
  return audit;
}

// The full root geometry belongs to the request, including a matrix's F
// leading dimension. A worker range must never replace it with its local N.
static Audit execute_row_batches(const Program &program,
    const std::vector<ops::Value> &inputs, const double *parameters,
    std::size_t parameter_count, const std::int64_t *starts,
    const std::int64_t *ends, std::size_t rows, void *output,
    Scratch &scratch, const ResultLayout &layout, std::size_t result_row,
    std::size_t range_first = 0,
    std::size_t range_last = std::numeric_limits<std::size_t>::max(),
    bool clear_output = true) {
  const auto &plan = *program.execution_metadata->row_batch;
  const auto columns = program.roots.size();
  if (scratch.arena_owner || (scratch.projection_owner &&
      scratch.projection_owner.get() != program.execution_metadata.get()))
    scratch.release_execution_buffers();
  scratch.projection_owner = program.execution_metadata;
  if (!scratch.compact_child) scratch.compact_child = std::make_unique<Scratch>();
  release_pointwise_workspace();
  auto &child_scratch = *scratch.compact_child;
  child_scratch.deadline = scratch.deadline; child_scratch.cancelled = scratch.cancelled;
  Program canonical = program; canonical.execution_metadata = plan.canonical_metadata;
  const std::vector<const char *> errors(program.nodes.size(), nullptr);
  Audit merged; merged.rows = rows; merged.nodes = program.nodes.size();
  merged.result_shapes.resize(rows * columns);
  merged.root_statuses.resize(rows * columns);
  if (clear_output) {
    const auto bytes = layout.row_bytes[result_row + rows] - layout.row_bytes[result_row];
    if (bytes) std::memset(static_cast<std::uint8_t *>(output) +
        layout.row_bytes[result_row], 0, bytes);
  }
  std::vector<ops::Value> blocked;
  std::vector<ops::Shape> full;
  for (std::size_t row = 0; row < rows; ++row) {
    const auto length = std::size_t(ends[row] - starts[row]);
    ops::require(length >= program.minimum_observations, "INSUFFICIENT_OBSERVATIONS");
    std::size_t width = 0, total_count = 0;
    ops::require(row_batch_inputs(program, inputs, length, blocked, width, total_count),
                 "RESULT_CAPACITY_MISMATCH");
    required_array_capacity(program, inputs, length, &full);
    const auto last = range_last == std::numeric_limits<std::size_t>::max()
        ? total_count : range_last;
    ops::require(range_first <= last && last <= total_count, "ROW_BATCH_RANGE_GEOMETRY");
    bool stage = false;
    std::size_t block_elements = 0;
    for (std::size_t root = 0; root < columns; ++root) {
      const auto shape = full[program.roots[root]];
      const auto elements = shape.size();
      const auto &slot = layout.slots[(result_row + row) * columns + root];
      ops::require(shape.rank >= 1 && shape.rank <= 2 && shape.dim[0] == total_count &&
          slot.capacity >= elements && slot.byte_offset % alignof(double) == 0 &&
          slot.byte_offset <= layout.bytes() &&
          elements <= (layout.bytes() - slot.byte_offset) / sizeof(double),
          "RESULT_CAPACITY_MISMATCH");
      stage = stage || (shape.rank == 2 && layout.order(root) == ops::MemoryOrder::f);
      const auto local_elements = width * (elements / total_count);
      ops::require(local_elements <= std::size_t(PTRDIFF_MAX) / sizeof(double) - block_elements,
          "RESULT_SIZE_OVERFLOW");
      block_elements += local_elements;
      merged.result_shapes[row * columns + root] = shape;
    }
    // Zero extent is valid. This request-local block never retains a larger
    // prior request's allocation and is not part of the returned result owner.
    std::vector<double> staged(stage ? block_elements : 0);
    double empty_output = 0.;
    void *staged_output = staged.empty() ? static_cast<void *>(&empty_output) : staged.data();
    auto earliest = std::numeric_limits<std::size_t>::max();
    std::string failure;
    for (std::size_t first = range_first; first < last; first += width) {
      native::check_deadline(scratch.deadline);
      if (scratch.cancelled && scratch.cancelled->load()) throw native::Timeout("native row batch cancelled");
      blocked = inputs;
      auto &weights = blocked[plan.input];
      if (weights.data && weights.shape.dim[1])
        weights.data = static_cast<const double *>(weights.data) +
            std::ptrdiff_t(first) * weights.stride[0];
      const auto count = std::min(width, last - first);
      weights.shape.dim[0] = count;
      ResultLayout selected;
      selected.columns = columns; selected.row_bytes = {0, 0};
      selected.row_statuses = {0, 0}; selected.orders.assign(columns, ops::MemoryOrder::c);
      std::size_t offset = 0;
      for (std::size_t root = 0; root < columns; ++root) {
        const auto root_width = full[program.roots[root]].size() / total_count;
        auto slot = layout.slots[(result_row + row) * columns + root];
        slot.capacity = count * root_width;
        slot.byte_offset = stage ? offset * sizeof(double)
            : slot.byte_offset + first * root_width * sizeof(double);
        selected.slots.push_back(slot);
        offset += slot.capacity;
      }
      try {
        const auto child = execute_impl(canonical, blocked, parameters, parameter_count,
            starts + row, ends + row, 1, stage ? staged_output : output, columns,
            child_scratch, false, nullptr, true, false, &selected, 0, &errors);
        for (auto member : {&Audit::root_copy_bytes, &Audit::direct_output_bytes,
            &Audit::state_copy_bytes, &Audit::vector_elements, &Audit::pointwise_tiles,
            &Audit::fused_pointwise_calls, &Audit::algorithm_copy_bytes,
            &Audit::fused_scalar_calls, &Audit::summary_source_scans, &Audit::order_stat_sorts,
            &Audit::highway_vqsort_calls, &Audit::eigen_rotation_calls,
            &Audit::eigen_matvec_calls, &Audit::eigen_covariance_calls,
            &Audit::accelerate_matmul_calls}) merged.*member += child.*member;
        for (auto member : {&Audit::max_window, &Audit::numeric_arena_bytes, &Audit::mask_arena_bytes,
            &Audit::order_scratch_capacity_bytes, &Audit::pointwise_workspace_capacity_bytes,
            &Audit::native_library_workspace_reserve_bytes})
          merged.*member = std::max(merged.*member, child.*member);
        merged.operator_workspace_capacity_bytes = std::max(merged.operator_workspace_capacity_bytes,
            child.operator_workspace_capacity_bytes + staged.capacity() * sizeof(double));
        if (stage) {
          for (std::size_t root = 0; root < columns; ++root) {
            const auto shape = full[program.roots[root]];
            const auto root_width = shape.size() / total_count;
            const auto size = count * root_width;
            if (!size) continue;
            const auto source_offset = selected.slots[root].byte_offset;
            const auto *source = staged.data() + source_offset / sizeof(double);
            auto *target = reinterpret_cast<double *>(static_cast<std::uint8_t *>(output) +
                layout.slots[(result_row + row) * columns + root].byte_offset);
            if (shape.rank == 1 || layout.order(root) == ops::MemoryOrder::c)
              std::memcpy(target + first * root_width, source, size * sizeof(double));
            else
              for (std::size_t j = 0; j < root_width; ++j)
                for (std::size_t i = 0; i < count; ++i)
                  target[j * total_count + first + i] = source[i * root_width + j];
            merged.root_copy_bytes += size * sizeof(double);
          }
        }
      } catch (const ops::Error &error) {
        if (child_scratch.failed_node == std::numeric_limits<std::size_t>::max()) throw;
        if (child_scratch.failed_node < earliest) {
          earliest = child_scratch.failed_node; failure = error.what();
        }
      }
    }
    if (!failure.empty()) {
      scratch.failed_node = earliest;
      throw ops::Error(failure.c_str());
    }
  }
  return merged;
}

static Audit execute_impl(const Program &program, const std::vector<ops::Value> &inputs,
    const double *parameters, std::size_t parameter_count, const std::int64_t *starts,
    const std::int64_t *ends, std::size_t rows, void *output, std::size_t output_columns,
    Scratch &scratch, bool prebound_inputs,
    const std::vector<std::int16_t> *input_failures, bool interval_known,
    bool propagate_scope_errors, const ResultLayout *layout, std::size_t result_row,
    const std::vector<const char *> *deferred_errors) {
  if (deferred_errors)
    ops::require(!value_observer && !program.isolate_errors && !input_failures &&
        !propagate_scope_errors && deferred_errors->size() == program.nodes.size(),
        "GRAPH_DEFERRED_ERROR_CONTEXT");
  if (!program.execution_metadata) {
    Program finalized = program;
    finalized.finalize();
    return execute_impl(finalized, inputs, parameters, parameter_count, starts,
                        ends, rows, output, output_columns, scratch, prebound_inputs,
                        input_failures, interval_known, propagate_scope_errors, layout, result_row,
                        deferred_errors);
  }
  if ((value_observer || prebound_inputs || input_failures) && program.roots.size() == 1) {
    const auto root = program.roots[0];
    if (cholesky_f_root_requires_arena(program, root, true)) {
      // These routes keep the canonical C writer and its original F copy.
      // Restore its arena in the same metadata used by Scratch's shape plan;
      // merely declining the private F entry would leave no C output slot.
      Program fallback = program;
      auto metadata = std::make_shared<ExecutionMetadata>(*program.execution_metadata);
      metadata->root_destinations[root] = -1;
      fallback.execution_metadata = std::move(metadata);
      return execute_impl(fallback, inputs, parameters, parameter_count, starts,
                          ends, rows, output, output_columns, scratch, prebound_inputs,
                          input_failures, interval_known, propagate_scope_errors, layout, result_row,
                          deferred_errors);
    }
  }
  scratch.propagate_scope_errors = propagate_scope_errors;
  ops::require(output_columns == program.roots.size(), "GRAPH_OUTPUT_COLUMNS");
  // Choose once per native execution. AOT specialization removes optional
  // per-node provenance branches from graphs that cannot produce local scope errors.
  ops::require(!input_failures || input_failures->size() == program.input_count, "GRAPH_INPUT_STATUS_COUNT");
  if (!deferred_errors && !value_observer && !prebound_inputs && !input_failures &&
      interval_known && !propagate_scope_errors && !program.isolate_errors &&
      program.execution_metadata->row_batch && layout && rows && starts && ends && output &&
      inputs.size() == program.input_count && parameter_count == program.parameter_count &&
      layout->columns == output_columns && result_row + rows < layout->row_bytes.size() &&
      layout->row_statuses.size() == layout->row_bytes.size() &&
      reinterpret_cast<std::uintptr_t>(output) % alignof(double) == 0) {
    std::vector<ops::Value> blocked;
    bool admitted = layout->slots.size() >= (result_row + rows) * output_columns;
    for (std::size_t row = 0; row < rows && admitted; ++row) {
      admitted = starts[row] >= 0 && ends[row] >= starts[row];
      if (!admitted) break;
      for (std::size_t input = 0; input < inputs.size(); ++input) {
        const auto &value = inputs[input];
        admitted = admitted && value.shape.rank >= 0 && value.shape.rank <= 3;
        if (program.input_axes.empty() || program.input_axes[input] != 1)
          admitted = admitted && value.shape.rank > 0 && std::size_t(ends[row]) <= value.shape.dim[0];
      }
      std::size_t width = 0, count = 0;
      admitted = admitted && row_batch_inputs(program, inputs,
          std::size_t(ends[row] - starts[row]), blocked, width, count);
      std::vector<ops::Shape> full;
      if (admitted)
        required_array_capacity(program, inputs, std::size_t(ends[row] - starts[row]), &full);
      for (std::size_t root = 0; root < output_columns && admitted; ++root) {
        const auto &slot = layout->slots[(result_row + row) * output_columns + root];
        const auto elements = full[program.roots[root]].size();
        admitted = slot.capacity >= elements && slot.byte_offset % alignof(double) == 0 &&
            slot.byte_offset <= layout->bytes() &&
            elements <= (layout->bytes() - slot.byte_offset) / sizeof(double);
      }
    }
    if (admitted)
      return execute_row_batches(program, inputs, parameters, parameter_count, starts,
          ends, rows, output, scratch, *layout, result_row);
  }
  if (!deferred_errors && !value_observer && !prebound_inputs && !input_failures &&
      interval_known && !propagate_scope_errors && !program.isolate_errors &&
      program.execution_metadata->compact_path) {
    if (scratch.arena_owner || (scratch.projection_owner &&
        scratch.projection_owner.get() != program.execution_metadata.get()))
      scratch.release_execution_buffers();
    release_pointwise_workspace();
    if (!scratch.compact_child) scratch.compact_child = std::make_unique<Scratch>();
    // Keep the owner even after a numerical exception, so another execution
    // cannot retain this private workspace outside its own admitted budget.
    scratch.projection_owner = program.execution_metadata;
    Audit fused;
    const CompactPostExecutor post = [&](const Program &child_program,
        const std::vector<ops::Value> &child_inputs, const double *child_parameters,
        std::size_t child_parameter_count, const std::vector<const char *> &errors,
        void *destination, const ResultLayout &original_layout, std::size_t original_row,
        const std::vector<std::size_t> &original_roots, Audit &merged,
        native::Deadline deadline, const std::atomic<bool> *cancelled) {
      ops::require(child_program.roots.size() == original_roots.size() &&
          original_row >= result_row && original_row - result_row < rows &&
          original_row + 1 < original_layout.row_bytes.size(), "RESULT_LAYOUT_BOUNDS");
      ResultLayout selected;
      selected.columns = original_roots.size();
      selected.row_bytes = {original_layout.row_bytes[original_row],
                            original_layout.row_bytes[original_row + 1]};
      selected.row_statuses = {0, 0};
      selected.orders.assign(selected.columns, ops::MemoryOrder::c);
      selected.slots.reserve(selected.columns);
      for (std::size_t root = 0; root < selected.columns; ++root) {
        const auto original = original_roots[root];
        ops::require(original < program.roots.size(), "RESULT_LAYOUT_BOUNDS");
        selected.slots.push_back(original_layout.slots.at(
            original_row * original_layout.columns + original));
        selected.orders.set(root, original_layout.order(original));
      }
      auto &child_scratch = *scratch.compact_child;
      const auto child_limit = planner::canonical_child_scratch_bytes(
          child_program, child_inputs, 0);
      // A large fan can keep max_window unchanged when the path vector gets
      // smaller. In that case the canonical arena resizes but ordered scratch
      // may still retain its old capacity outside this request's admission.
      if (child_limit < scratch.compact_child_limit)
        child_scratch.release_execution_buffers();
      scratch.compact_child_limit = child_limit;
      child_scratch.deadline = deadline; child_scratch.cancelled = cancelled;
      const std::int64_t begin = 0, end = 1;
      const auto child = execute_impl(child_program, child_inputs, child_parameters,
          child_parameter_count, &begin, &end, 1, destination, selected.columns,
          child_scratch, true, nullptr, true, false, &selected, 0, &errors);
      merged.result_shapes.resize(rows * program.roots.size());
      merged.root_statuses.resize(rows * program.roots.size());
      const auto local_row = original_row - result_row;
      for (std::size_t root = 0; root < selected.columns; ++root) {
        const auto index = local_row * program.roots.size() + original_roots[root];
        merged.result_shapes[index] = child.result_shapes[root];
        merged.root_statuses[index] = child.root_statuses[root];
      }
      for (auto member : {&Audit::root_copy_bytes, &Audit::direct_output_bytes,
          &Audit::state_copy_bytes, &Audit::vector_elements, &Audit::pointwise_tiles,
          &Audit::fused_pointwise_calls, &Audit::algorithm_copy_bytes,
          &Audit::fused_scalar_calls, &Audit::summary_source_scans, &Audit::order_stat_sorts,
          &Audit::highway_vqsort_calls,
          &Audit::eigen_rotation_calls, &Audit::eigen_matvec_calls,
          &Audit::eigen_covariance_calls, &Audit::eigen_density_calls, &Audit::gaussian_filter_streams,
          &Audit::accelerate_covchol_calls, &Audit::accelerate_matmul_calls,
          &Audit::matmul_zero_calls, &Audit::accelerate_covariance_attempts,
          &Audit::accelerate_covariance_calls, &Audit::covariance_syrk_calls,
          &Audit::covariance_replayed_cells, &Audit::covariance_fallbacks,
          &Audit::covariance_zero_calls, &Audit::accelerate_vforce_calls,
          &Audit::accelerate_lu_attempts,
          &Audit::accelerate_lu_calls,
          &Audit::accelerate_lu_fallbacks,
          &Audit::solve_residual_materialization_bytes,
          &Audit::covariance_panel_materialization_bytes})
        merged.*member += child.*member;
      for (auto member : {&Audit::numeric_arena_bytes, &Audit::mask_arena_bytes,
          &Audit::order_scratch_capacity_bytes, &Audit::pointwise_workspace_capacity_bytes,
          &Audit::operator_workspace_capacity_bytes,
          &Audit::native_library_workspace_reserve_bytes})
        merged.*member = std::max(merged.*member, child.*member);
    };
    if (execute_compact_path(program, *program.execution_metadata->compact_path,
        inputs, parameters, parameter_count, starts, ends, rows, output, layout,
        result_row, scratch.workspace, fused, planner::canonical_child_scratch_bytes,
        post, scratch.deadline, scratch.cancelled))
      return fused;
  }
  if (!deferred_errors && !value_observer && !prebound_inputs && !input_failures && interval_known &&
      program.execution_metadata->affine_projection) {
    if (scratch.arena_owner || (scratch.projection_owner &&
        scratch.projection_owner.get() != program.execution_metadata.get()))
      scratch.release_execution_buffers();
    release_pointwise_workspace();
    Audit fused;
    if (execute_affine_projection(program, *program.execution_metadata->affine_projection,
        inputs, parameters, parameter_count, starts, ends, rows, output, layout,
        result_row, scratch.workspace, fused, scratch.deadline, scratch.cancelled)) {
      scratch.projection_owner = program.execution_metadata;
      return fused;
    }
  }
  if (!prebound_inputs && scratch.projection_owner) scratch.release_execution_buffers();
  if (!deferred_errors && !value_observer && !input_failures && interval_known && program.execution_metadata->pointwise) {
    if (!prebound_inputs && scratch.arena_owner) scratch.release_execution_buffers();
    Audit fused;
    if (execute_pointwise(program, inputs, parameters, parameter_count, starts, ends,
        rows, output, layout, result_row, fused, scratch.deadline, scratch.cancelled, prebound_inputs))
      return fused;
  }
  if (!prebound_inputs && scratch.arena_owner.get() != program.execution_metadata.get())
    release_pointwise_workspace();
  if (deferred_errors)
    return execute_impl_body<false, false, true>(program, inputs, parameters, parameter_count,
        starts, ends, rows, output, output_columns, scratch, prebound_inputs, input_failures,
        interval_known, layout, result_row, deferred_errors);
  if (value_observer) {
    if ((program.isolate_errors || input_failures) && program.execution_metadata->position_status_nodes)
      return execute_impl_body<true, true>(program, inputs, parameters, parameter_count,
          starts, ends, rows, output, output_columns, scratch, prebound_inputs, input_failures, interval_known, layout, result_row);
    return execute_impl_body<false, true>(program, inputs, parameters, parameter_count,
        starts, ends, rows, output, output_columns, scratch, prebound_inputs, input_failures, interval_known, layout, result_row);
  }
  if ((program.isolate_errors || input_failures) && program.execution_metadata->position_status_nodes)
    return execute_impl_body<true>(program, inputs, parameters, parameter_count,
        starts, ends, rows, output, output_columns, scratch, prebound_inputs, input_failures, interval_known, layout, result_row);
  return execute_impl_body<false>(program, inputs, parameters, parameter_count,
      starts, ends, rows, output, output_columns, scratch, prebound_inputs, input_failures, interval_known, layout, result_row);
}

void release_graph_workspace() {
  if (root_scratch.arena_owner || root_scratch.projection_owner ||
      root_scratch.compact_child || root_scratch.workspace.capacity_bytes())
    root_scratch.release_execution_buffers();
}

Audit execute(const Program &program, const std::vector<ops::Value> &inputs,
              const double *parameters, std::size_t parameter_count,
              const std::int64_t *starts, const std::int64_t *ends,
              std::size_t rows, void *output, std::size_t output_columns,
              const ResultLayout *layout, std::size_t result_row,
              std::chrono::steady_clock::time_point deadline, const std::atomic<bool> *cancelled,
              bool allow_workspace_loan, bool expanded_matmul_roots, bool row_subviews) {
  struct RestoreLoan {
    bool previous, previous_matmul, previous_rows;
    ~RestoreLoan() {
      root_scratch.allow_workspace_loan = previous;
      root_scratch.expanded_matmul_roots = previous_matmul;
      root_scratch.allow_row_subviews = previous_rows;
    }
  } restore{std::exchange(root_scratch.allow_workspace_loan, allow_workspace_loan),
            std::exchange(root_scratch.expanded_matmul_roots, expanded_matmul_roots),
            std::exchange(root_scratch.allow_row_subviews, row_subviews)};
  root_scratch.deadline = deadline; root_scratch.cancelled = cancelled;
  ResultLayout local;
  if (program.output_kind == OutputKind::typed && !layout) {
    local = result_layout(program, inputs, starts, ends, rows);
    layout = &local;
  }
  return execute_impl(program, inputs, parameters, parameter_count, starts, ends,
                      rows, output, output_columns, root_scratch, false, nullptr, true, false, layout, result_row);
}

Audit execute_row_batch_range(const Program &program,
    const std::vector<ops::Value> &inputs, const double *parameters,
    std::size_t parameter_count, const std::int64_t *starts,
    const std::int64_t *ends, void *output, const ResultLayout &layout,
    std::size_t first, std::size_t last, bool publish_metadata,
    std::size_t &failed_node, std::chrono::steady_clock::time_point deadline,
    const std::atomic<bool> *cancelled, bool allow_workspace_loan, bool expanded_matmul_roots) {
  failed_node = std::numeric_limits<std::size_t>::max();
  ops::require(program.execution_metadata && program.execution_metadata->row_batch &&
      !program.isolate_errors && starts && ends && output && starts[0] >= 0 &&
      ends[0] >= starts[0], "ROW_BATCH_RANGE_GEOMETRY");
  std::vector<ops::Value> views;
  std::size_t width = 0, count = 0;
  ops::require(row_batch_inputs(program, inputs, std::size_t(ends[0] - starts[0]),
      views, width, count) && first <= last && last <= count && last - first >= 128,
      "ROW_BATCH_RANGE_GEOMETRY");
  const auto columns = program.roots.size();
  ops::require(layout.columns == columns && layout.slots.size() == columns &&
      layout.row_bytes.size() == 2 && layout.row_statuses.size() == 2,
      "ROW_BATCH_RANGE_LAYOUT");
  struct RestoreLoan {
    bool previous, previous_matmul;
    ~RestoreLoan() {
      root_scratch.allow_workspace_loan = previous;
      root_scratch.expanded_matmul_roots = previous_matmul;
    }
  } restore{std::exchange(root_scratch.allow_workspace_loan, allow_workspace_loan),
            std::exchange(root_scratch.expanded_matmul_roots, expanded_matmul_roots)};
  root_scratch.deadline = deadline; root_scratch.cancelled = cancelled;
  root_scratch.failed_node = std::numeric_limits<std::size_t>::max();
  try {
    auto result = execute_row_batches(program, inputs, parameters, parameter_count,
        starts, ends, 1, output, root_scratch, layout, 0, first, last, false);
    if (!publish_metadata) {
      result.result_shapes.clear(); result.root_statuses.clear();
    }
    return result;
  } catch (const ops::Error &) {
    failed_node = root_scratch.failed_node;
    throw;
  }
}

Audit execute_observed(const Program &program, const std::vector<ops::Value> &inputs,
                       const double *parameters, std::size_t parameter_count,
                       const std::int64_t *starts, const std::int64_t *ends,
                       std::size_t rows, void *output, std::size_t output_columns,
                       const ResultLayout *layout, const ValueObserver &observer) {
  ops::require(bool(observer), "GRAPH_NULL_OBSERVER");
  struct Restore {
    const ValueObserver *previous;
    ~Restore() { value_observer = previous; }
  } restore{value_observer};
  value_observer = &observer;
  return execute(program, inputs, parameters, parameter_count, starts, ends,
                 rows, output, output_columns, layout);
}

} // namespace calmetrics_engine::graph
