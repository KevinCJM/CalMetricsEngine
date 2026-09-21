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
  std::vector<std::int16_t> statuses;
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

  void ensure(const Program &program, std::size_t max_window) {
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
    if (program.isolate_errors)
      statuses.resize(program.nodes.size());
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

ops::Value interval_view(const ops::Value &base, std::size_t start,
                         std::size_t length) {
  ops::require((base.kind == ops::Kind::number || base.kind == ops::Kind::integer ||
                base.kind == ops::Kind::mask) && base.shape.rank >= 1 && base.shape.rank <= 2,
               "GRAPH_INPUT_TYPE");
  ops::require(start <= base.shape.dim[0] && length <= base.shape.dim[0] - start,
               "GRAPH_INTERVAL_BOUNDS");
  ops::Value view = base;
  view.shape.dim[0] = length;
  if (length) {
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
                   ((scope.kind == ApplyKind::block || scope.kind == ApplyKind::group) && argc == 1),
                   "GRAPH_APPLY_ARGUMENTS");
      const bool array = scope.kind == ApplyKind::block || scope.kind == ApplyKind::group;
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
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const auto &node = nodes[i];
    if (node.kind != NodeKind::operation)
      continue;
    const auto &spec = ops::lookup(node.opcode);
    metadata->specs[i] = &spec;
    metadata->requires_shape_planning = metadata->requires_shape_planning || spec.family == ops::Family::matrix;
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
      const auto axis = program.input_axes.empty() ? 0 : program.input_axes[node.input_index];
      if (axis != 1 && shape.rank) shape.dim[0] = std::min(shape.dim[0], max_window);
    } else if (node.kind == NodeKind::interval_tail) shape = shapes[node.parents[0]];
    else if (node.kind == NodeKind::operation) {
      const auto op = static_cast<ops::Op>(node.opcode);
      if (node.storage != StorageKind::inline_value || op == ops::Op::lag || op == ops::Op::transpose ||
          (op == ops::Op::diag && shapes[node.parents[0]].rank == 2)) {
        for (std::size_t j = 0; j < node.parent_count; ++j)
          if (shapes[node.parents[j]].rank > shape.rank ||
              shapes[node.parents[j]].size() > shape.size()) shape = shapes[node.parents[j]];
        const auto lhs = shapes[node.parents[0]];
        const auto rhs = node.parent_count > 1 ? shapes[node.parents[1]] : ops::Shape{};
        if (op >= ops::Op::sum_time && op <= ops::Op::max_time && lhs.rank == 2)
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
      }
    } else if (node.kind == NodeKind::rolling_scope) shape = ops::vector_shape(max_window);
    else if (node.kind == NodeKind::apply_scope && node.storage != StorageKind::inline_value) {
      const auto &scope = program.apply_scopes[node.input_index];
      if (scope.kind == ApplyKind::block && scope.input_nodes.empty()) shape = ops::vector_shape(max_window);
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
                          std::size_t rows, double *output,
                          std::size_t output_columns, Scratch &scratch,
                          bool prebound_inputs = false);

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

static ops::Value execute_rolling_scope(
    const Program &program, const std::vector<ops::Value> &raw_inputs,
    const RollingScope &scope, const Node &node, std::size_t interval_start,
    std::size_t interval_length, std::size_t max_window, Scratch &scratch) {
  ops::require(scope.body && scope.body->output_kind == OutputKind::scalar,
               "GRAPH_ROLLING_BODY");
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
    ops::require(value.kind == ops::Kind::number && value.shape.rank == 1 &&
                     value.size() == interval_length,
                 "ROLLING_ALIGNMENT_MISMATCH");
    captured.push_back(value);
  }

  const ops::Value *dates = nullptr;
  const ops::Value *raw_dates = nullptr;
  double annual = 0.0;
  if (scope.has_date_context) {
    dates = &scratch.values[scope.dates_node];
    ops::require(dates->kind == ops::Kind::number && dates->shape.rank == 1 &&
                     dates->size() == interval_length,
                 "ROLLING_ALIGNMENT_MISMATCH");
    annual = scratch.values[scope.annual_rate_node].scalar;
    ops::require(std::isfinite(annual), "INVALID_PARAMETER");
    for (std::size_t i = 0; i < interval_length; ++i)
      ops::require(std::isfinite(dates->f(i)) &&
                       (i == 0 || dates->f(i) > dates->f(i - 1)),
                   "ROLLING_DATE_AXIS_INVALID");
    if (scope.needs_preceding_observation) {
      const auto &date_node = program.nodes[scope.dates_node];
      ops::require(date_node.kind == NodeKind::input,
                   "ROLLING_DATE_CONTEXT_MUST_BE_INPUT");
      raw_dates = &raw_inputs[date_node.input_index];
      if (interval_start > 0)
        ops::require(std::isfinite(raw_dates->f(interval_start - 1)) &&
                         raw_dates->f(interval_start - 1) < dates->f(0),
                     "ROLLING_DATE_AXIS_INVALID");
    }
  }

  scratch.rolling_invalid.assign(interval_length + 1, 0);
  for (std::size_t i = 0; i < interval_length; ++i) {
    bool invalid = false;
    for (const auto &value : captured)
      invalid = invalid || !std::isfinite(value.f(i));
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

  for (std::size_t right = 0; right < interval_length; ++right) {
    std::size_t left = right + 1 > width ? right + 1 - width : 0;
    if (scope.has_min_periods && scope.needs_preceding_observation &&
        interval_start + left == 0 && right > 0)
      left = 1;
    const auto count = right + 1 - left;
    const auto invalid =
        scratch.rolling_invalid[right + 1] - scratch.rolling_invalid[left];
    if (count - invalid < minimum)
      continue;
    if (!scope.has_min_periods && count != width)
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
        body_inputs[i] = interval_view(captured[i], left, count);
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
      observation_count = 0.0;
      const auto &returns =
          captured[static_cast<std::size_t>(scope.returns_input)];
      for (std::size_t i = left; i <= right; ++i)
        if (std::isfinite(returns.f(i)))
          observation_count += 1.0;
    }

    for (std::size_t i = 0; i < scope.parameter_bindings.size(); ++i) {
      const auto &binding = scope.parameter_bindings[i];
      switch (binding.kind) {
      case RollingParameterKind::outer_node:
        body_parameters[i] = scratch.values[binding.node].scalar;
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
    try {
      execute_impl(*scope.body, body_inputs,
                   body_parameters.empty() ? nullptr : body_parameters.data(),
                   body_parameters.size(), &local_start, &local_end, 1, &value,
                   1, child, true);
      if (std::isfinite(value))
        destination[right] = value;
    } catch (const ops::Error &error) {
      if (!numerical_error(error)) throw;
      destination[right] = nan;
    }
  }
  return result;
}

static ops::Value execute_apply_scope(const Program &program, const ApplyScope &scope,
    const Node &node, std::size_t interval_length, std::size_t max_window, Scratch &scratch) {
  auto &child = *scratch.children[program.rolling_scopes.size() + node.input_index];
  child.remaining_work = scratch.remaining_work;
  std::vector<ops::Value> captured;
  captured.reserve(scope.input_nodes.size());
  std::size_t count = 0;
  for (auto id : scope.input_nodes) {
    const auto &value = scratch.values[id];
    ops::require(value.shape.rank == 1, "SCOPE_CAPTURE_RANK");
    if (!captured.empty() && scope.kind != ApplyKind::bisect)
      ops::require(value.size() == count, "SCOPE_ALIGNMENT_MISMATCH");
    count = std::max(count, value.size());
    captured.push_back(value);
  }
  if (captured.empty() && scope.kind != ApplyKind::bisect) {
    count = scope.kind == ApplyKind::block ? interval_length
        : scratch.values[scope.argument_nodes[0]].size();
  }
  std::vector<double> parameters(scope.parameter_nodes.size());
  for (std::size_t i = 0; i < parameters.size(); ++i)
    if (scope.parameter_nodes[i] != std::numeric_limits<std::uint32_t>::max()) {
      const auto &value = scratch.values[scope.parameter_nodes[i]];
      ops::require(value.shape.rank == 0 && value.kind == ops::Kind::number, "SCOPE_PARAMETER_TYPE");
      parameters[i] = value.scalar;
    }
  auto evaluate = [&](const std::vector<ops::Value> &inputs, std::size_t length,
                      double solve_x = 0.0) {
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
    const auto audit = execute_impl(*scope.body, inputs, parameters.data(), parameters.size(),
                                   &begin, &end, 1, &value, 1, child, true);
    scratch.algorithm_copy_bytes += audit.algorithm_copy_bytes;
    return value;
  };
  auto scalar_arg = [&](std::size_t i) {
    const auto &value = scratch.values[scope.argument_nodes[i]];
    ops::require(value.shape.rank == 0 && value.kind == ops::Kind::number, "SCOPE_PARAMETER_TYPE");
    return value.scalar;
  };
  if (scope.kind == ApplyKind::bisect) {
    double low = scalar_arg(0), high = scalar_arg(1), tolerance = scalar_arg(2);
    const auto iterations = positive_integer(scalar_arg(3), "INVALID_PARAMETER", 10000);
    ops::require(std::isfinite(low) && std::isfinite(high) && low < high &&
                 std::isfinite(tolerance) && tolerance > 0.0, "INVALID_PARAMETER");
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
      if (contiguous) {
        inputs.push_back(interval_view(source, length ? indices[first] : 0, length));
        continue;
      }
      ops::Value view = source;
      view.shape = ops::vector_shape(length);
      view.stride[0] = 1;
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
  if (scope.kind == ApplyKind::filter) {
    const auto &mask = scratch.values[scope.argument_nodes[0]];
    ops::require(mask.kind == ops::Kind::mask && mask.shape.rank == 1 && mask.size() == count,
                 "SCOPE_ALIGNMENT_MISMATCH");
    auto &indices = child.selection; indices.clear();
    for (std::size_t i = 0; i < count; ++i) {
      const auto selected = mask.u(i);
      ops::require(selected <= 1, "INVALID_MASK");
      if (selected) indices.push_back(i);
    }
    if (indices.empty()) return ops::Value::number(scope.argument_nodes.size() == 2 ? scalar_arg(1) : NAN);
    return ops::Value::number(evaluate(selected(indices, 0, indices.size()), indices.size()));
  }
  ops::Value result;
  result.kind = ops::Kind::number;
  result.stride[0] = 1;
  auto *destination = scratch.numeric.data() + static_cast<std::size_t>(node.slot) * max_window;
  result.data = destination;
  if (scope.kind == ApplyKind::block) {
    const auto width = positive_integer(scalar_arg(0), "INVALID_PARAMETER", 1000000);
    const auto blocks = count / width;
    result.shape = ops::vector_shape(blocks);
    std::vector<ops::Value> inputs(captured.size());
    for (std::size_t block = 0; block < blocks; ++block) {
      for (std::size_t i = 0; i < captured.size(); ++i)
        inputs[i] = interval_view(captured[i], block * width, width);
      destination[block] = evaluate(inputs, width);
    }
    return result;
  }
  const auto &keys = scratch.values[scope.argument_nodes[0]];
  ops::require(keys.kind == ops::Kind::integer && keys.shape.rank == 1 && keys.size() == count,
               "SCOPE_ALIGNMENT_MISMATCH");
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
    try {
      value = evaluate(selected(indices, first, last), last - first);
    } catch (const ops::Error &error) {
      if (!program.isolate_errors || !numerical_error(error)) throw;
    }
    for (std::size_t i = first; i < last; ++i) destination[indices[i]] = value;
    first = last;
  }
  return result;
}

static Audit execute_impl(const Program &program,
                          const std::vector<ops::Value> &inputs,
                          const double *parameters,
                          std::size_t parameter_count,
                          const std::int64_t *starts,
                          const std::int64_t *ends,
                          std::size_t rows, double *output,
                          std::size_t output_columns, Scratch &scratch,
                          bool prebound_inputs) {
  if (!program.execution_metadata) {
    Program finalized = program;
    finalized.finalize();
    return execute_impl(finalized, inputs, parameters, parameter_count, starts,
                        ends, rows, output, output_columns, scratch,
                        prebound_inputs);
  }
  ops::require(inputs.size() == program.input_count, "GRAPH_INPUT_COUNT");
  ops::require(parameter_count == program.parameter_count,
               "GRAPH_PARAMETER_COUNT");
  ops::require(output_columns == program.roots.size(), "GRAPH_OUTPUT_COLUMNS");
  ops::require(rows == 0 || (starts && ends && output), "GRAPH_NULL_BATCH");

  std::size_t max_window = 0;
  if (prebound_inputs) {
    ops::require(rows == 1, "GRAPH_PREBOUND_ROWS");
    for (const auto &input : inputs)
      max_window = std::max(max_window, input.size());
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
  scratch.ensure(program, max_window);
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
  if (program.isolate_errors) {
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
    if (length < program.minimum_observations) {
      ops::require(program.isolate_errors, "INSUFFICIENT_OBSERVATIONS");
      const auto begin = program.output_kind == OutputKind::scalar ? row : series_base;
      const auto count = program.output_kind == OutputKind::scalar ? 1 : length;
      for (std::size_t i = 0; i < count * output_columns; ++i) {
        output[begin * output_columns + i] = NAN;
        audit.statuses[begin * output_columns + i] = 1;
      }
      if (program.output_kind == OutputKind::series) series_base += length;
      continue;
    }

    for (std::size_t node_index = 0; node_index < program.nodes.size();
         ++node_index) {
      const auto &node = program.nodes[node_index];
      if (program.isolate_errors) {
        auto &status = scratch.statuses[node_index];
        status = 0;
        for (std::size_t parent = 0; parent < node.parent_count; ++parent)
          status = std::max(status, scratch.statuses[node.parents[parent]]);
        if (status) {
          scratch.values[node_index] = ops::Value::number(NAN);
          continue;
        }
      }
      try {
        if (node.kind == NodeKind::input) {
          scratch.values[node_index] =
              (prebound_inputs || (!program.input_axes.empty() && program.input_axes[node.input_index] == 1))
                  ? inputs[node.input_index] : interval_view(inputs[node.input_index], start,
                                              length);
          continue;
        }
        if (node.kind == NodeKind::parameter) {
          scratch.values[node_index] =
              ops::Value::number(parameters[node.input_index]);
          if (program.isolate_errors && !std::isfinite(parameters[node.input_index]))
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
          continue;
        }
        if (node.kind == NodeKind::rolling_scope) {
          scratch.values[node_index] = execute_rolling_scope(
              program, inputs, program.rolling_scopes[node.input_index], node,
              start, length, max_window, scratch);
          continue;
        }

        if (node.kind == NodeKind::apply_scope) {
          scratch.values[node_index] = execute_apply_scope(program,
              program.apply_scopes[node.input_index], node, length, max_window, scratch);
          // Explicit empty/NaN filter results are missing data, not execution
          // failures. Preserve them for ordinary where selection; public roots
          // still report non-finite results as unavailable.
          if (program.isolate_errors && program.apply_scopes[node.input_index].kind != ApplyKind::filter &&
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

        if (!fused) {
          ops::Audit operator_audit;
          ops::execute(prepared, native_output, scratch.workspace,
                       ops::Isa::automatic, operator_audit);
          scratch.algorithm_copy_bytes += operator_audit.algorithm_copy_bytes;
        }
        scratch.values[node_index] =
            output_value(scratch, prepared, native_output, node, max_window);
        if (program.isolate_errors) {
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
        if (!program.isolate_errors)
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
        scratch.values[node_index] = ops::Value::number(NAN);
      }
    }

    for (std::size_t root_index = 0; root_index < program.roots.size();
         ++root_index) {
      const auto &value = scratch.values[program.roots[root_index]];
      const auto status = program.isolate_errors
                              ? scratch.statuses[program.roots[root_index]] : 0;
      if (status) {
        const auto begin = program.output_kind == OutputKind::scalar ? row : series_base;
        const auto count = program.output_kind == OutputKind::scalar ? 1 : length;
        for (std::size_t offset = 0; offset < count; ++offset) {
          const auto index = (begin + offset) * output_columns + root_index;
          output[index] = NAN;
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
        output[row * output_columns + root_index] = value.kind == ops::Kind::integer ?
            static_cast<double>(value.integer) : value.scalar;
        if (program.isolate_errors && !std::isfinite(output[row * output_columns + root_index]))
          audit.statuses[row * output_columns + root_index] = 4;
      } else {
        ops::require(value.kind == ops::Kind::number && value.shape.rank == 1 &&
                         value.size() == length,
                     "GRAPH_ROOT_NOT_ALIGNED_SERIES");
        for (std::size_t offset = 0; offset < length; ++offset) {
          const auto index = (series_base + offset) * output_columns + root_index;
          output[index] = value.f(offset);
          if (program.isolate_errors && !std::isfinite(output[index]))
            audit.statuses[index] = 4;
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
  const auto scratch_bytes = [&](const auto &self, const Scratch &state) -> std::size_t {
    std::size_t bytes = state.numeric.capacity() * 8 + state.integers.capacity() * 8 + state.masks.capacity();
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

Audit execute(const Program &program, const std::vector<ops::Value> &inputs,
              const double *parameters, std::size_t parameter_count,
              const std::int64_t *starts, const std::int64_t *ends,
              std::size_t rows, double *output, std::size_t output_columns) {
  return execute_impl(program, inputs, parameters, parameter_count, starts, ends,
                      rows, output, output_columns, root_scratch, false);
}

} // namespace calmetrics_engine::graph
