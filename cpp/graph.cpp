#include "calmetrics_engine/graph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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
  std::vector<ops::Value> values;
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
    numeric.resize(program.numeric_slots * max_window);
    masks.resize(program.mask_slots * max_window);
    values.resize(program.nodes.size());
    summaries.resize(program.nodes.size());
    ordered.resize(program.nodes.size());
    if (children.size() < program.rolling_scopes.size())
      children.resize(program.rolling_scopes.size());
    for (std::size_t i = 0; i < program.rolling_scopes.size(); ++i)
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
  ops::require(base.kind == ops::Kind::number && base.shape.rank == 1,
               "GRAPH_INPUT_TYPE");
  ops::require(start <= base.size() && length <= base.size() - start,
               "GRAPH_INTERVAL_BOUNDS");
  ops::Value view = base;
  view.shape = ops::vector_shape(length);
  if (length) {
    view.data = static_cast<const double *>(base.data) +
                static_cast<std::ptrdiff_t>(start) * base.stride[0];
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
    }
    return value;
  }

  ops::require(prepared.output_shape.rank == 1,
               "GRAPH_MATRIX_OUTPUT_UNSUPPORTED");
  ops::require(prepared.output_shape.size() <= max_window,
               "GRAPH_OUTPUT_TOO_LARGE");
  value.stride[0] = 1;

  if (prepared.output_kind == ops::Kind::mask) {
    ops::require(node.storage == StorageKind::mask, "GRAPH_STORAGE_KIND");
    value.data =
        scratch.masks.data() + static_cast<std::size_t>(node.slot) * max_window;
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

  for (std::size_t index = 0; index < nodes.size(); ++index) {
    const auto &node = nodes[index];
    ops::require(node.parent_count <= node.parents.size(),
                 "GRAPH_PARENT_COUNT");
    ops::require(static_cast<unsigned>(node.storage) <= 2,
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
      break;
    }
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
  metadata->specs.resize(nodes.size(), nullptr);
  metadata->summary_consumers.resize(nodes.size(), 0);
  metadata->order_consumers.resize(nodes.size(), 0);
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const auto &node = nodes[i];
    if (node.kind != NodeKind::operation)
      continue;
    const auto &spec = ops::lookup(node.opcode);
    metadata->specs[i] = &spec;
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
    } catch (const std::exception &) {
      destination[right] = nan;
    }
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
      for (const auto &input : inputs)
        ops::require(end <= input.size(), "GRAPH_INTERVAL_BOUNDS");
      max_window = std::max(max_window, end - start);
    }
  }

  scratch.ensure(program, max_window);
  scratch.fused_scalar_calls = 0;
  scratch.summary_source_scans = 0;
  scratch.order_stat_sorts = 0;
  scratch.algorithm_copy_bytes = 0;
  const auto &summary_consumers = program.execution_metadata->summary_consumers;
  const auto &order_consumers = program.execution_metadata->order_consumers;
  std::size_t series_base = 0;

  for (std::size_t row = 0; row < rows; ++row) {
    scratch.next_generation();
    const auto start = static_cast<std::size_t>(starts[row]);
    const auto end = static_cast<std::size_t>(ends[row]);
    const auto length = end - start;

    for (std::size_t node_index = 0; node_index < program.nodes.size();
         ++node_index) {
      const auto &node = program.nodes[node_index];

      if (node.kind == NodeKind::input) {
        scratch.values[node_index] =
            prebound_inputs ? inputs[node.input_index]
                            : interval_view(inputs[node.input_index], start,
                                            length);
        continue;
      }
      if (node.kind == NodeKind::parameter) {
        scratch.values[node_index] =
            ops::Value::number(parameters[node.input_index]);
        continue;
      }
      if (node.kind == NodeKind::constant) {
        scratch.values[node_index] = ops::Value::number(node.constant);
        continue;
      }
      if (node.kind == NodeKind::rolling_scope) {
        scratch.values[node_index] = execute_rolling_scope(
            program, inputs, program.rolling_scopes[node.input_index], node,
            start, length, max_window, scratch);
        continue;
      }

      std::array<ops::Value, 4> arguments{};
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

      if (prepared.output_shape.rank == 1) {
        ops::require(prepared.output_shape.size() <= max_window,
                     "GRAPH_OUTPUT_TOO_LARGE");
        if (prepared.output_kind == ops::Kind::mask) {
          ops::require(node.storage == StorageKind::mask, "GRAPH_STORAGE_KIND");
          native_output.data = scratch.masks.data() +
                               static_cast<std::size_t>(node.slot) * max_window;
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
    }

    for (std::size_t root_index = 0; root_index < program.roots.size();
         ++root_index) {
      const auto &value = scratch.values[program.roots[root_index]];
      if (program.output_kind == OutputKind::scalar) {
        ops::require(value.shape.rank == 0, "GRAPH_ROOT_NOT_SCALAR");
        ops::require(value.kind == ops::Kind::number ||
                         value.kind == ops::Kind::mask,
                     "GRAPH_ROOT_NOT_SCALAR");
        output[row * output_columns + root_index] = value.scalar;
      } else {
        ops::require(value.kind == ops::Kind::number && value.shape.rank == 1 &&
                         value.size() == length,
                     "GRAPH_ROOT_NOT_ALIGNED_SERIES");
        for (std::size_t offset = 0; offset < length; ++offset)
          output[(series_base + offset) * output_columns + root_index] =
              value.f(offset);
      }
    }
    if (program.output_kind == OutputKind::series)
      series_base += length;
  }

  Audit audit;
  audit.rows = rows;
  audit.nodes = program.nodes.size();
  audit.max_window = max_window;
  audit.numeric_arena_bytes = scratch.numeric.capacity() * sizeof(double);
  audit.mask_arena_bytes = scratch.masks.capacity() * sizeof(std::uint8_t);
  audit.operator_workspace_capacity_bytes = scratch.workspace.capacity_bytes();
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
