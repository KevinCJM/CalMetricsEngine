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
ops::MemoryOrder choose_order(const Program &p, const std::vector<ops::Value> &inputs,
                              std::size_t root) {
  using ops::MemoryOrder;
  const auto &schema=p.root_outputs[root];
  if (schema.rank < 2 || schema.order == OutputOrder::c) return MemoryOrder::c;
  if (schema.order == OutputOrder::f) return MemoryOrder::f;
  // Position statuses are stored in logical C order. Measured isolated F root
  // materialization can regress substantially; retain C until both traversals
  // have a jointly optimized path. Explicit F remains a caller constraint.
  if (p.isolate_errors) return MemoryOrder::c;
  const auto id=p.roots[root]; const auto &node=p.nodes[id];
  // One order serves all windows. The paired writer uses F; shapes below its
  // measured admission retain the record and materialize the same F result.
  if (p.execution_metadata && static_cast<std::int32_t>(id)==p.execution_metadata->split_eigh_vectors &&
      ops::symmetric_eigh_split_scratch_doubles(512)) return MemoryOrder::f;
  if (p.execution_metadata && p.execution_metadata->pointwise) {
    // A single physical traversal serves all array roots. Do not select mixed
    // orders for a shared tile or force a conversion for C/strided sources.
    bool array=false;
    for (const auto &v:inputs) if (v.shape.rank) {
      if (v.kind != ops::Kind::number || v.shape.rank < 2 || v.packed_order() != MemoryOrder::f || !v.f_contiguous())
        return MemoryOrder::c;
      array=true;
    }
    return array ? MemoryOrder::f : MemoryOrder::c;
  }
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(__APPLE__) && defined(__aarch64__)
  if (__builtin_available(macOS 15.0, *)) {
    if (independent_transcendental_root(p, id)) {
      const auto &source = p.nodes[node.parents[0]];
      if (source.kind == NodeKind::input) {
        const auto &v = inputs[source.input_index];
        if (v.kind == ops::Kind::number && v.shape.rank >= 2 &&
            v.shape.dim[0] >= 1024 && v.f_contiguous()) return MemoryOrder::f;
      }
    }
  }
#endif
  if (node.kind == NodeKind::input) {
    const auto &v=inputs[node.input_index];
    return v.dense_contiguous() ? v.packed_order() : MemoryOrder::c;
  }
  if (node.kind == NodeKind::operation && static_cast<ops::Op>(node.opcode) == ops::Op::transpose) {
    const auto &source=p.nodes[node.parents[0]];
    if (source.kind == NodeKind::input && inputs[source.input_index].c_contiguous())
      return MemoryOrder::f;
    if (p.roots.size() == 1 && p.execution_metadata &&
        p.execution_metadata->root_destinations[node.parents[0]] == static_cast<std::int32_t>(root))
      return MemoryOrder::f;
  }
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
  // Measured Eigen two-operation column writer. Auto admission is structural;
  // zero-denominator and other payload checks remain in execution.
  if (__builtin_available(macOS 15.0, *)) {
    if (!p.execution_metadata) return MemoryOrder::c;
    const auto &fused=p.execution_metadata->fused_from;
    if (schema.rank == 2 && id < fused.size() && fused[id] >= 0) {
      const auto &first=p.nodes[fused[id]];
      const auto scalar_or_broadcast = [&](std::uint32_t i) {
        const auto &n=p.nodes[i];
        return n.storage == StorageKind::inline_value ||
            (n.kind == NodeKind::operation && static_cast<ops::Op>(n.opcode) == ops::Op::broadcast_like);
      };
      const auto first_op=static_cast<ops::Op>(first.opcode), last_op=static_cast<ops::Op>(node.opcode);
      const bool math=first_op == ops::Op::add || first_op == ops::Op::subtract || first_op == ops::Op::multiply;
      const bool last_math=last_op == ops::Op::add || last_op == ops::Op::subtract || last_op == ops::Op::multiply ||
          (last_op == ops::Op::divide && node.parents[0] == static_cast<std::uint32_t>(fused[id]));
      if (math && last_math && scalar_or_broadcast(node.parents[node.parents[0] == static_cast<std::uint32_t>(fused[id]) ? 1 : 0])) {
        for (std::size_t side=0;side<2;++side) {
          const auto &source=p.nodes[first.parents[side]];
          if (source.kind != NodeKind::input || !scalar_or_broadcast(first.parents[1-side])) continue;
          const auto &v=inputs[source.input_index];
          if (v.shape.rank == 2 && v.shape.dim[0] >= 32 && v.shape.dim[1] >= 8 &&
              (v.f_contiguous() || (v.stride[0] == -1 && v.stride[1] >= std::ptrdiff_t(v.shape.dim[0]))))
            return MemoryOrder::f;
        }
      }
    }
  }
#endif
  return MemoryOrder::c;
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
  for (std::size_t root=0;root<out.columns;++root) {
    const auto order=choose_order(p,inputs,root);
    if (order == ops::MemoryOrder::f) {
      if (out.orders.empty()) out.orders.assign(out.columns,ops::MemoryOrder::c);
      out.orders.set(root,order);
    }
  }
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
