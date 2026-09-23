#include "calmetrics_engine/graph.hpp"
#include <algorithm>
#include <map>
#include <limits>

namespace calmetrics_engine::graph {
namespace {
std::size_t add(std::size_t a, std::size_t b) {
  ops::require(a <= static_cast<std::size_t>(PTRDIFF_MAX) &&
      b <= static_cast<std::size_t>(PTRDIFF_MAX) - a, "RESULT_SIZE_OVERFLOW");
  return a + b;
}
std::size_t mul(std::size_t a, std::size_t b) {
  ops::require(!b || a <= static_cast<std::size_t>(PTRDIFF_MAX) / b, "RESULT_SIZE_OVERFLOW");
  return a * b;
}
}
ResultLayout result_layout(const Program &p, const std::vector<ops::Value> &inputs,
                           const std::int64_t *starts, const std::int64_t *ends, std::size_t rows) {
  p.validate();
  ops::require(inputs.size() == p.input_count, "GRAPH_INPUT_COUNT");
  ops::require(p.output_kind == OutputKind::typed && p.root_outputs.size() == p.roots.size(),
               "RESULT_SCHEMA");
  ops::require(rows == 0 || (starts && ends), "GRAPH_NULL_BATCH");
  ResultLayout out;
  out.columns = p.roots.size();
  out.slots.reserve(mul(rows, out.columns));
  out.row_bytes.reserve(add(rows, 1));
  out.row_statuses.reserve(add(rows, 1));
  std::size_t bytes = 0, statuses = 0;
  // Repeated windows share geometry inference; values and parameter payloads are never read.
  std::map<std::size_t, std::vector<std::size_t>> capacities;
  for (std::size_t row = 0; row < rows; ++row) {
    ops::require(starts[row] >= 0 && ends[row] >= starts[row], "GRAPH_INTERVAL_BOUNDS");
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto axis = p.input_axes.empty() ? 0 : p.input_axes[i];
      if (axis != 1)
        ops::require(static_cast<std::size_t>(ends[row]) <= inputs[i].shape.dim[0], "GRAPH_INTERVAL_BOUNDS");
    }
    const auto length = static_cast<std::size_t>(ends[row] - starts[row]);
    auto found = capacities.find(length);
    if (found == capacities.end()) {
      std::vector<ops::Shape> shapes;
      required_array_capacity(p, inputs, length, &shapes);
      std::vector<std::size_t> sizes;
      for (std::size_t root = 0; root < out.columns; ++root)
        sizes.push_back(p.root_outputs[root].rank == 0 ? 1 : shapes[p.roots[root]].size());
      found = capacities.emplace(length, std::move(sizes)).first;
    }
    out.row_bytes.push_back(bytes);
    out.row_statuses.push_back(statuses);
    for (std::size_t root = 0; root < out.columns; ++root) {
      const auto count = found->second[root];
      out.slots.push_back({bytes, statuses, count});
      const auto raw = mul(count, output_itemsize(p.root_outputs[root].dtype));
      bytes = add(bytes, mul(add(raw, 7) / 8, 8));
      statuses = add(statuses, count);
    }
  }
  out.row_bytes.push_back(bytes);
  out.row_statuses.push_back(statuses);
  return out;
}
} // namespace calmetrics_engine::graph
