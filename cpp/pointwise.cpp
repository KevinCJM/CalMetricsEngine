#include "calmetrics_engine/pointwise.hpp"
#include "calmetrics_engine/native_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace calmetrics_engine::graph {
namespace {
bool safe_op(ops::Op op) {
  using ops::Op;
  switch (op) {
  case Op::add: case Op::subtract: case Op::multiply:
  case Op::minimum: case Op::maximum: case Op::negate: case Op::absolute:
  case Op::equal: case Op::not_equal: case Op::less_than: case Op::less_equal:
  case Op::greater_than: case Op::greater_equal: case Op::finite_mask: return true;
  default: return false;
  }
}
struct Buffers {
  std::vector<ops::Value> values, block;
  std::vector<double> numeric;
  std::vector<std::uint8_t> masks;
  ops::Workspace workspace;
};
thread_local Buffers buffers;
struct ChainBinding {
  ops::ScalarChain chain;
  ops::ScalarChainKernel kernel = nullptr;
  std::size_t source = SIZE_MAX, first = SIZE_MAX, second = SIZE_MAX, predicate = SIZE_MAX;
};
ChainBinding bind_chain(const Program &p, const std::vector<ops::Value> &values) {
  ChainBinding result;
  if (p.execution_metadata->pointwise_complete_region) {
    const auto &r = p.execution_metadata->pointwise_regions.front();
    result.source = r.source; result.first = r.first;
    result.second = r.second == UINT32_MAX ? SIZE_MAX : r.second;
    result.predicate = r.predicate == UINT32_MAX ? SIZE_MAX : r.predicate;
    for (std::size_t j = 0; j < 2; ++j) {
      if (r.scalars[j] != UINT32_MAX) result.chain.constants[j] = values[r.scalars[j]].scalar;
      result.chain.constant_first[j] = r.scalar_first[j];
    }
    if (r.scalars[2] != UINT32_MAX) result.chain.threshold = values[r.scalars[2]].scalar;
    result.chain.threshold_first = r.scalar_first[2];
    result.kernel = r.kernel;
    return result;
  }
  std::array<ops::Op,2> math{ops::Op(0),ops::Op(0)};
  auto predicate = ops::Op(0);
  std::size_t previous = SIZE_MAX, steps = 0;
  for (std::size_t i = 0; i < p.nodes.size(); ++i) {
    const auto &n = p.nodes[i];
    if (n.kind != NodeKind::operation || !values[i].shape.rank) continue;
    if (result.predicate != SIZE_MAX) return {};
    const auto op = static_cast<ops::Op>(n.opcode);
    const bool arithmetic = op == ops::Op::add || op == ops::Op::subtract || op == ops::Op::multiply;
    const bool comparison = op >= ops::Op::equal && op <= ops::Op::greater_equal;
    if (!arithmetic && !comparison && op != ops::Op::finite_mask) return {};
    const auto side = values[n.parents[0]].shape.rank ? 0u : 1u;
    if (side >= n.parent_count) return {};
    const auto parent = n.parents[side];
    if (!values[parent].shape.rank || (n.parent_count == 2 && values[n.parents[1-side]].shape.rank)) return {};
    if (previous == SIZE_MAX) {
      if (p.nodes[parent].kind != NodeKind::input) return {};
      previous = result.source = parent;
    }
    if (parent != previous) return {};
    if (arithmetic) {
      if (steps == 2) return {};
      math[steps] = op;
      result.chain.constants[steps] = values[n.parents[1-side]].scalar;
      result.chain.constant_first[steps] = side == 1;
      (steps ? result.second : result.first) = i;
      ++steps; previous = i;
    } else {
      result.predicate = i; predicate = op;
      if (n.parent_count == 2) result.chain.threshold = values[n.parents[1-side]].scalar;
      result.chain.threshold_first = side == 1;
    }
  }
  if (result.source != SIZE_MAX && p.execution_metadata->root_destinations[
      result.predicate == SIZE_MAX ? previous : result.predicate] >= 0)
    result.kernel = ops::scalar_chain_kernel(math[0],math[1],predicate);
  return result;
}
}
void plan_pointwise_regions(const Program &p, ExecutionMetadata &m) {
  for (std::size_t r = 0; r < p.roots.size(); ++r)
    if (p.nodes[p.roots[r]].kind != NodeKind::operation || m.root_destinations[p.roots[r]] != static_cast<std::int32_t>(r))
      m.pointwise_root_copies.push_back(r);
  std::vector<std::size_t> consumers(p.nodes.size());
  std::vector<bool> elided(p.nodes.size());
  for (const auto &n : p.nodes)
    for (std::size_t j = 0; j < n.parent_count; ++j) ++consumers[n.parents[j]];
  const auto scalar = [&](std::size_t id) {
    return p.nodes[id].kind == NodeKind::constant || p.nodes[id].kind == NodeKind::parameter;
  };
  const auto arithmetic = [](ops::Op op) {
    return op == ops::Op::add || op == ops::Op::subtract || op == ops::Op::multiply;
  };
  const auto next = [&](std::size_t id) {
    while (++id < p.nodes.size() && scalar(id)) {}
    return id;
  };
  for (std::size_t i = 0; i < p.nodes.size(); ++i) {
    const auto &n = p.nodes[i];
    if (n.kind != NodeKind::operation || n.storage != StorageKind::numeric ||
        n.parent_count != 2 || !arithmetic(static_cast<ops::Op>(n.opcode))) continue;
    const auto side = scalar(n.parents[1]) ? 0u : 1u;
    if (!scalar(n.parents[1-side]) || scalar(n.parents[side])) continue;
    PointwiseRegion r;
    r.first = static_cast<std::uint32_t>(i); r.source = n.parents[side];
    r.scalars[0] = n.parents[1-side]; r.scalar_first[0] = side == 1;
    auto last = i;
    auto second_op = ops::Op(0), predicate = ops::Op(0);
    auto j = next(i);
    if (j < p.nodes.size()) {
      const auto &q = p.nodes[j];
      const auto s = q.parents[0] == i ? 0u : 1u;
      if (q.kind == NodeKind::operation && q.parent_count == 2 &&
          q.parents[s] == i && scalar(q.parents[1-s]) && arithmetic(static_cast<ops::Op>(q.opcode))) {
        r.second = static_cast<std::uint32_t>(j); r.scalars[1] = q.parents[1-s];
        r.scalar_first[1] = s == 1; second_op = static_cast<ops::Op>(q.opcode);
        last = j; j = next(j);
      }
    }
    if (j < p.nodes.size()) {
      const auto &q = p.nodes[j]; const auto op = static_cast<ops::Op>(q.opcode);
      const auto s = q.parents[0] == last ? 0u : 1u;
      if (q.kind == NodeKind::operation &&
          ((op == ops::Op::finite_mask && q.parent_count == 1 && q.parents[0] == last) ||
           (op >= ops::Op::equal && op <= ops::Op::greater_equal && q.parent_count == 2 &&
            q.parents[s] == last && scalar(q.parents[1-s])))) {
        r.predicate = static_cast<std::uint32_t>(j); predicate = op;
        if (q.parent_count == 2) { r.scalars[2] = q.parents[1-s]; r.scalar_first[2] = s == 1; }
      }
    }
    if (r.second == UINT32_MAX && r.predicate == UINT32_MAX) continue;
    r.kernel = ops::scalar_chain_kernel(static_cast<ops::Op>(n.opcode), second_op, predicate);
    if (!r.kernel) continue;
    r.retain_first = consumers[i] != 1 || m.root_destinations[i] >= 0;
    elided[i] = !r.retain_first;
    m.pointwise_regions.push_back(r);
    i = r.end();
  }
  if (m.pointwise_regions.size() == 1) {
    const auto &r = m.pointwise_regions.front();
    m.pointwise_complete_region = p.nodes[r.source].kind == NodeKind::input &&
        m.root_destinations[r.end()] >= 0;
    for (std::size_t i = 0; i < p.nodes.size(); ++i)
      if (p.nodes[i].kind == NodeKind::operation && i != r.first && i != r.second && i != r.predicate)
        m.pointwise_complete_region = false;
  }
  for (std::size_t i = 0; i < p.nodes.size(); ++i) {
    const auto &n = p.nodes[i];
    if (n.kind != NodeKind::operation || m.root_destinations[i] >= 0 || elided[i]) continue;
    if (n.storage == StorageKind::numeric)
      m.pointwise_numeric_slots = std::max(m.pointwise_numeric_slots, std::size_t(n.slot)+1);
    if (n.storage == StorageKind::mask)
      m.pointwise_mask_slots = std::max(m.pointwise_mask_slots, std::size_t(n.slot)+1);
  }
}
void release_pointwise_workspace() {
  auto clear = [](auto &v) { std::decay_t<decltype(v)>().swap(v); };
  clear(buffers.values); clear(buffers.block);
  clear(buffers.numeric); clear(buffers.masks);
}
bool pointwise_iteration_inplace_safe(const Program &p, const std::vector<ops::Value> &inputs,
    const double *parameters, std::size_t state_input) {
  if (!p.execution_metadata->pointwise || p.roots.size() != 1 ||
      p.root_outputs[0].dtype != OutputDType::float64) return false;
  if (state_input >= inputs.size() || !inputs[state_input].shape.rank) return false;
  for (std::size_t i = 0; i < inputs.size(); ++i)
    if (inputs[i].kind != ops::Kind::number || inputs[i].shape.rank < 0 ||
        (i != state_input && inputs[i].shape.rank)) return false;
  // Scalars are resolved without allocation. No scalar operation, array
  // capture, branch, or third transform is admitted by this proof.
  const auto scalar = [&](std::size_t id, double &value) {
    const auto &n = p.nodes[id];
    if (n.kind == NodeKind::constant) value = n.constant;
    else if (n.kind == NodeKind::parameter) value = parameters[n.input_index];
    else if (n.kind == NodeKind::input && n.input_index != state_input &&
        inputs[n.input_index].kind == ops::Kind::number && !inputs[n.input_index].shape.rank) {
      const auto &v = inputs[n.input_index];
      value = v.data ? *static_cast<const double *>(v.data) : v.scalar;
    } else return false;
    return std::isfinite(value);
  };
  std::size_t last = SIZE_MAX, steps = 0;
  for (std::size_t i = 0; i < p.nodes.size(); ++i) {
    const auto &n = p.nodes[i];
    if (n.kind == NodeKind::input && n.input_index == state_input) {
      if (last != SIZE_MAX) return false;
      last = i;
    } else if (n.kind == NodeKind::operation) {
      if (steps == 2 || n.parent_count != 2 || last == SIZE_MAX) return false;
      const auto side = n.parents[0] == last ? 0u : 1u;
      if (n.parents[side] != last) return false;
      double constant;
      if (!scalar(n.parents[1-side],constant)) return false;
      const auto op = static_cast<ops::Op>(n.opcode);
      if (!(op == ops::Op::multiply ? std::abs(constant) <= 1 :
          (op == ops::Op::add || op == ops::Op::subtract) && constant == 0)) return false;
      last = i; ++steps;
    }
  }
  return steps && p.roots[0] == last;
}
std::size_t pointwise_workspace_bytes(const Program &p, std::size_t count) {
  const auto &m = *p.execution_metadata;
  if (m.pointwise_complete_region) return p.nodes.size() * sizeof(ops::Value);
  return p.nodes.size() * (2 * sizeof(ops::Value)) + std::min(count, pointwise_tile_elements) *
      (m.pointwise_numeric_slots * sizeof(double) + m.pointwise_mask_slots);
}
bool pointwise_eligible(const Program &p) {
  if (p.output_kind != OutputKind::typed || p.isolate_errors || p.minimum_observations) return false;
  for (const auto &root : p.root_outputs) if (!root.rank) return false;
  bool operation = false;
  for (const auto &n : p.nodes) {
    if (n.kind == NodeKind::operation) {
      if (!safe_op(static_cast<ops::Op>(n.opcode))) return false;
      operation = true;
    } else if (n.kind != NodeKind::input && n.kind != NodeKind::constant && n.kind != NodeKind::parameter)
      return false;
  }
  return operation;
}

bool execute_pointwise(const Program &p, const std::vector<ops::Value> &inputs,
    const double *parameters, std::size_t parameter_count, const std::int64_t *starts,
    const std::int64_t *ends, std::size_t rows, void *output, const ResultLayout *layout,
    std::size_t result_row, Audit &audit, std::chrono::steady_clock::time_point deadline,
    const std::atomic<bool> *cancelled, bool prebound, std::size_t tile_begin, std::size_t tile_end,
    const double *previous, double *residual, bool *finite, InplacePointwiseChain *capture) {
  if (!p.execution_metadata->pointwise || !layout) return false;
  if (!prebound) release_graph_workspace();
  // Integer/mask inputs retain their exact-type general path.
  for (const auto &v : inputs)
    if (v.shape.rank < 0 || v.kind != ops::Kind::number) return false;
  ops::require(inputs.size() == p.input_count, "GRAPH_INPUT_COUNT");
  ops::require(parameter_count == p.parameter_count, "GRAPH_PARAMETER_COUNT");
  ops::require(!parameter_count || parameters, "GRAPH_NULL_PARAMETERS");
  native::check_deadline(deadline);
  if (cancelled && cancelled->load()) throw native::Timeout("native tensor cancelled");
  for (const auto &v : inputs) ops::require(!v.shape.rank || !v.size() || v.data, "NULL_INPUT");
  ops::require(!rows || (starts && ends && output), "GRAPH_NULL_BATCH");
  ops::require(layout->columns == p.roots.size() && !layout->row_bytes.empty() &&
      result_row < layout->row_bytes.size() && rows <= layout->row_bytes.size() - 1 - result_row,
      "RESULT_LAYOUT_BOUNDS");
  ops::require(reinterpret_cast<std::uintptr_t>(output) % alignof(double) == 0, "RESULT_OUTPUT_ALIGNMENT");
  auto &b = buffers;
  if (b.values.capacity() != p.nodes.size()) release_pointwise_workspace();
  b.values.resize(p.nodes.size());
  const auto &roots = p.execution_metadata->root_destinations;
  audit = {}; audit.rows = rows; audit.nodes = p.nodes.size();
  audit.result_shapes.resize(rows * p.roots.size());
  audit.root_statuses.resize(rows * p.roots.size());
  for (std::size_t row = 0; row < rows; ++row) {
    ops::require(starts[row] >= 0 && ends[row] >= starts[row], "GRAPH_INTERVAL_BOUNDS");
    const auto length = static_cast<std::size_t>(ends[row] - starts[row]);
    audit.max_window = std::max(audit.max_window, length);
    ops::Shape shape; bool has_shape = false;
    // Prepare all logical operators before execution, including independent
    // shape checks. Scalar broadcasts remain scalar throughout the tile loop.
    for (std::size_t i = 0; i < p.nodes.size(); ++i) {
      const auto &n = p.nodes[i]; auto &v = b.values[i];
      if (n.kind == NodeKind::input) {
        v = inputs[n.input_index];
        const auto axis = p.input_axes.empty() ? 0 : p.input_axes[n.input_index];
        if (!prebound && axis != 1 && v.shape.rank) {
          ops::require(static_cast<std::size_t>(ends[row]) <= v.shape.dim[0], "GRAPH_INTERVAL_BOUNDS");
          const auto offset = starts[row] * v.stride[0];
          if (offset) v.data = static_cast<const double *>(v.data) + offset;
          v.shape.dim[0] = length;
        }
        if (!v.shape.rank && v.data) v.scalar = *static_cast<const double *>(v.data);
      } else if (n.kind == NodeKind::constant) v = ops::Value::number(n.constant);
      else if (n.kind == NodeKind::parameter) v = ops::Value::number(parameters[n.input_index]);
      else {
        ops::require(n.parent_count <= 2, "ARITY_MISMATCH");
        ops::Value args[2];
        for (std::size_t j = 0; j < n.parent_count; ++j) args[j] = b.values[n.parents[j]];
        auto q = ops::prepare_geometry(*p.execution_metadata->specs[i], args, n.parent_count);
        v = ops::Value::number(0); v.kind = q.output_kind; v.shape = q.output_shape;
        v.set_contiguous_strides();
        ops::require(n.storage == (v.shape.rank ?
            (v.kind == ops::Kind::mask ? StorageKind::mask : StorageKind::numeric) :
            StorageKind::inline_value), "GRAPH_STORAGE_KIND");
        if (!v.shape.rank) {
          ops::Output out; out.kind = q.output_kind;
          ops::Audit a; ops::execute(q, out, b.workspace, ops::Isa::automatic, a);
          v.scalar = out.scalar;
        }
      }
      if (v.shape.rank) {
        if (!has_shape) { shape = v.shape; has_shape = true; }
        else if (!(shape == v.shape)) return false;
      }
    }
    const auto count = shape.size();
    const auto begin = std::min(tile_begin, count), end = std::min(tile_end, count);
    const auto tile_capacity = std::min(pointwise_tile_elements, count);
    auto chain = bind_chain(p,b.values);
    // Cache-sized intermediates; roots live directly in independent output.
    const auto numeric = chain.kernel ? 0 : p.execution_metadata->pointwise_numeric_slots * tile_capacity;
    const auto masks = chain.kernel ? 0 : p.execution_metadata->pointwise_mask_slots * tile_capacity;
    const auto resize = [](auto &v, std::size_t n) {
      if (v.capacity() != n) std::decay_t<decltype(v)>(n).swap(v); else v.resize(n);
    };
    resize(b.numeric, numeric); resize(b.masks, masks);
    resize(b.block, chain.kernel ? 0 : p.nodes.size());
    for (std::size_t r = 0; r < p.roots.size(); ++r) {
      const auto &v = b.values[p.roots[r]]; const auto &schema = p.root_outputs[r];
      const auto &slot = layout->slots[(result_row + row) * p.roots.size() + r];
      ops::require(v.shape.rank == schema.rank && v.size() <= slot.capacity, "RESULT_CAPACITY_MISMATCH");
      ops::require(schema.dtype == (v.kind == ops::Kind::mask ? OutputDType::boolean : OutputDType::float64),
          "RESULT_DTYPE_MISMATCH");
      audit.result_shapes[row * p.roots.size() + r] = v.shape;
      if (chain.kernel && p.nodes[p.roots[r]].kind == NodeKind::operation &&
          roots[p.roots[r]] == static_cast<std::int32_t>(r))
        audit.direct_output_bytes += (end-begin)*output_itemsize(schema.dtype);
      // Clear only unused capacity/alignment bytes. In a tensor fork, the
      // first task exclusively owns padding while other tasks own values.
      if (tile_begin == 0) {
        const auto used = slot.byte_offset + v.size() * output_itemsize(schema.dtype);
        const auto limit = r + 1 < p.roots.size()
            ? layout->slots[(result_row + row) * p.roots.size() + r + 1].byte_offset
            : layout->row_bytes[result_row + row + 1];
        ops::require(used <= limit, "RESULT_LAYOUT_BOUNDS");
        if (limit > used) std::memset(static_cast<std::uint8_t *>(output) + used, 0, limit - used);
      }
    }
    if (capture && rows == 1 && chain.kernel && chain.predicate == SIZE_MAX &&
        p.roots.size() == 1 && previous == output &&
        b.values[chain.source].data == output &&
        p.roots[0] == (chain.second == SIZE_MAX ? chain.first : chain.second) &&
        layout->slots[result_row].byte_offset == 0) {
      capture->kernel = chain.kernel; capture->chain = chain.chain;
      capture->input = b.values[chain.source]; capture->count = count;
      capture->output = static_cast<double *>(output);
      capture->output_is_first = chain.second == SIZE_MAX;
    }
    // Geometry was validated above. Reuse one small-arity descriptor instead
    // of retaining a full generic Prepared (eight Value arguments) per node.
    ops::Prepared q;
    for (std::size_t first = begin; first < end; first += pointwise_tile_elements) {
      native::check_deadline(deadline);
      if (cancelled && cancelled->load()) throw native::Timeout("native tensor cancelled");
      const auto size = std::min(pointwise_tile_elements, end - first);
      std::size_t region_index = 0;
      if (chain.kernel) {
        const auto destination = [&](std::size_t id) -> void * {
          if (id == SIZE_MAX || roots[id] < 0) return nullptr;
          const auto r = static_cast<std::size_t>(roots[id]);
          return static_cast<std::uint8_t *>(output) + layout->slots[(result_row+row)*p.roots.size()+r].byte_offset
              + first*output_itemsize(p.root_outputs[r].dtype);
        };
        chain.chain.first_output = static_cast<double *>(destination(chain.first));
        chain.chain.value_output = static_cast<double *>(destination(chain.second));
        chain.chain.mask_output = static_cast<std::uint8_t *>(destination(chain.predicate));
        chain.chain.previous = previous ? previous+first : nullptr;
        chain.chain.residual = residual; chain.chain.finite = finite;
        audit.vector_elements += chain.kernel(b.values[chain.source],first,size,chain.chain);
        for (const auto r : p.execution_metadata->pointwise_root_copies) {
          const auto id = p.roots[r];
          auto *dest = static_cast<std::uint8_t *>(output) +
              layout->slots[(result_row+row)*p.roots.size()+r].byte_offset +
              first*output_itemsize(p.root_outputs[r].dtype);
          const auto bytes = size*output_itemsize(p.root_outputs[r].dtype);
          if (p.nodes[id].kind == NodeKind::operation) {
            std::memcpy(dest, destination(id), bytes);
          } else {
            const auto &v = b.values[id];
            if (v.contiguous()) std::memcpy(dest, static_cast<const double *>(v.data)+first, bytes);
            else for (std::size_t j = 0; j < size; ++j)
              static_cast<double *>(static_cast<void *>(dest))[j] = v.f(first+j);
          }
          audit.root_copy_bytes += bytes;
        }
      } else for (std::size_t i = 0; i < p.nodes.size(); ++i) {
        const auto &n = p.nodes[i]; auto &v = b.block[i]; v = b.values[i];
        if (!v.shape.rank) continue;
        if (n.kind == NodeKind::input) {
          if (v.contiguous()) {
            v.data = static_cast<const double *>(v.data) + first;
            v.shape = ops::vector_shape(size); v.stride = {1, 1, 1};
          }
          continue;
        }
        v.shape = ops::vector_shape(size); v.stride = {1, 1, 1};
        const auto prepare_output = [&](std::size_t id) -> void * {
          auto &value = b.block[id]; value = b.values[id];
          value.shape = ops::vector_shape(size); value.stride = {1,1,1};
          void *dest;
          if (roots[id] >= 0) {
            const auto r = static_cast<std::size_t>(roots[id]);
            dest = static_cast<std::uint8_t *>(output) +
                layout->slots[(result_row+row)*p.roots.size()+r].byte_offset +
                first*output_itemsize(p.root_outputs[r].dtype);
          } else if (value.kind == ops::Kind::mask)
            dest = b.masks.data() + p.nodes[id].slot*tile_capacity;
          else dest = b.numeric.data() + p.nodes[id].slot*tile_capacity;
          value.data = dest;
          return dest;
        };
        const auto &regions = p.execution_metadata->pointwise_regions;
        while (region_index < regions.size() && regions[region_index].first < i) ++region_index;
        if (region_index < regions.size() && regions[region_index].first == i) {
          const auto &region = regions[region_index++];
          ops::ScalarChain local;
          for (std::size_t j = 0; j < 2; ++j) {
            if (region.scalars[j] != UINT32_MAX) local.constants[j] = b.values[region.scalars[j]].scalar;
            local.constant_first[j] = region.scalar_first[j];
          }
          if (region.scalars[2] != UINT32_MAX) local.threshold = b.values[region.scalars[2]].scalar;
          local.threshold_first = region.scalar_first[2];
          if (region.retain_first) local.first_output = static_cast<double *>(prepare_output(region.first));
          if (region.second != UINT32_MAX) local.value_output = static_cast<double *>(prepare_output(region.second));
          if (region.predicate != UINT32_MAX) local.mask_output = static_cast<std::uint8_t *>(prepare_output(region.predicate));
          const auto &source = b.block[region.source];
          audit.vector_elements += region.kernel(source, source.contiguous() ? 0 : first, size, local);
          i = region.end();
          continue;
        }
        q.spec = p.execution_metadata->specs[i]; q.count = n.parent_count;
        q.output_kind = v.kind;
        if (n.parent_count == 1) q.args[1] = ops::Value::number(0);
        for (std::size_t j = 0; j < n.parent_count; ++j) q.args[j] = b.values[n.parents[j]].shape.rank ? b.block[n.parents[j]] : b.values[n.parents[j]];
        q.payload_available_mask = UINT32_MAX;
        q.output_shape = v.shape;
        ops::Output out; out.shape = v.shape; out.kind = v.kind;
        out.data = prepare_output(i);
        ops::Audit a;
        if (!q.args[0].contiguous() || !q.args[1].contiguous()) {
          std::array<std::size_t, 2> offsets{q.args[0].contiguous() ? 0 : first,
                                          q.args[1].contiguous() ? 0 : first};
          ops::elementwise_span(q, out, offsets);
        } else ops::execute(q, out, b.workspace, ops::Isa::automatic, a);
        audit.vector_elements += a.vector_elements;
        v.data = out.data;
      }
      if (!chain.kernel) for (std::size_t r = 0; r < p.roots.size(); ++r) {
        const auto id = p.roots[r]; const auto &v = b.block[id];
        auto *dest = static_cast<std::uint8_t *>(output) + layout->slots[(result_row + row) * p.roots.size() + r].byte_offset
            + first * output_itemsize(p.root_outputs[r].dtype);
        if (dest != v.data) {
          const auto bytes = size * output_itemsize(p.root_outputs[r].dtype);
          if (v.contiguous()) std::memcpy(dest, v.data, bytes);
          else for (std::size_t i = 0; i < size; ++i)
            static_cast<double *>(static_cast<void *>(dest))[i] = v.f(first + i);
          audit.root_copy_bytes += bytes;
        } else audit.direct_output_bytes += size * output_itemsize(p.root_outputs[r].dtype);
      }
      if (previous && !chain.kernel) {
        const auto *candidate = static_cast<const double *>(output);
        audit.vector_elements += ops::iteration_residual(previous + first,
            candidate + first, size, *residual, *finite);
      }
      ++audit.pointwise_tiles;
    }
  }
  audit.numeric_arena_bytes = b.numeric.capacity() * sizeof(double);
  audit.mask_arena_bytes = b.masks.capacity();
  audit.operator_workspace_capacity_bytes = b.values.capacity() * sizeof(ops::Value)
      + b.block.capacity() * sizeof(ops::Value);
  audit.fused_pointwise_calls = rows;
  audit.pointwise_workspace_capacity_bytes = audit.numeric_arena_bytes + audit.mask_arena_bytes
      + audit.operator_workspace_capacity_bytes;
  return true;
}

void execute_inplace_pointwise_chain(const InplacePointwiseChain &prepared, Audit &audit,
    double &residual, bool &finite, std::chrono::steady_clock::time_point deadline,
    const std::atomic<bool> *cancelled) {
  // The first execution's shape/status vectors and bounded TLS workspace stay
  // valid throughout this scope. Reset only counters, never allocate per step.
  audit.vector_elements = 0; audit.pointwise_tiles = 0;
  audit.direct_output_bytes = prepared.count*sizeof(double);
  audit.root_copy_bytes = 0;
  for (std::size_t first = 0; first < prepared.count; first += pointwise_tile_elements) {
    native::check_deadline(deadline);
    if (cancelled && cancelled->load()) throw native::Timeout("native iteration cancelled");
    auto chain = prepared.chain;
    (prepared.output_is_first ? chain.first_output : chain.value_output) = prepared.output+first;
    chain.previous = prepared.output+first; chain.residual = &residual; chain.finite = &finite;
    audit.vector_elements += prepared.kernel(prepared.input,first,
        std::min(pointwise_tile_elements,prepared.count-first),chain);
    ++audit.pointwise_tiles;
  }
}
}
