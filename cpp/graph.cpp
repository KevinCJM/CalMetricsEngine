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

thread_local Scratch scratch;

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

SummaryCache &summary_for(std::size_t source_node, const ops::Value &x) {
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

double summary_result(const Node &node, const ops::Prepared &prepared) {
  const auto op = prepared.spec->op;
  const auto source_node = static_cast<std::size_t>(node.parents[0]);
  const auto &x = prepared.args[0];
  auto &cache = summary_for(source_node, x);
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

const std::vector<double> &ordered_for(std::size_t source_node,
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

double ordered_result(const Node &node, const ops::Prepared &prepared) {
  const auto &ordered =
      ordered_for(static_cast<std::size_t>(node.parents[0]), prepared.args[0]);
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

Audit execute(const Program &program, const std::vector<ops::Value> &inputs,
              const double *parameters, std::size_t parameter_count,
              const std::int64_t *starts, const std::int64_t *ends,
              std::size_t rows, double *output, std::size_t output_columns) {
  if (!program.execution_metadata) {
    Program finalized = program;
    finalized.finalize();
    return execute(finalized, inputs, parameters, parameter_count, starts, ends,
                   rows, output, output_columns);
  }
  ops::require(inputs.size() == program.input_count, "GRAPH_INPUT_COUNT");
  ops::require(parameter_count == program.parameter_count,
               "GRAPH_PARAMETER_COUNT");
  ops::require(output_columns == program.roots.size(), "GRAPH_OUTPUT_COLUMNS");
  ops::require(rows == 0 || (starts && ends && output), "GRAPH_NULL_BATCH");

  std::size_t max_window = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    ops::require(starts[row] >= 0 && ends[row] >= starts[row],
                 "GRAPH_INTERVAL_BOUNDS");
    const auto start = static_cast<std::size_t>(starts[row]);
    const auto end = static_cast<std::size_t>(ends[row]);
    for (const auto &input : inputs)
      ops::require(end <= input.size(), "GRAPH_INTERVAL_BOUNDS");
    max_window = std::max(max_window, end - start);
  }

  scratch.ensure(program, max_window);
  scratch.fused_scalar_calls = 0;
  scratch.summary_source_scans = 0;
  scratch.order_stat_sorts = 0;
  scratch.algorithm_copy_bytes = 0;
  const auto &summary_consumers = program.execution_metadata->summary_consumers;
  const auto &order_consumers = program.execution_metadata->order_consumers;

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
            interval_view(inputs[node.input_index], start, length);
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
          native_output.set(0, summary_result(node, prepared));
          fused = true;
        } else if (order_fusion_eligible(spec.op) &&
                   order_consumers[source] > 1) {
          native_output.set(0, ordered_result(node, prepared));
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
          output_value(prepared, native_output, node, max_window);
    }

    for (std::size_t root_index = 0; root_index < program.roots.size();
         ++root_index) {
      const auto &value = scratch.values[program.roots[root_index]];
      ops::require(value.shape.rank == 0, "GRAPH_ROOT_NOT_SCALAR");
      ops::require(value.kind == ops::Kind::number ||
                       value.kind == ops::Kind::mask,
                   "GRAPH_ROOT_NOT_SCALAR");
      output[row * output_columns + root_index] = value.scalar;
    }
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

} // namespace calmetrics_engine::graph
