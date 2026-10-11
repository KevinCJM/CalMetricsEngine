#include "calmetrics_engine/affine_projection.hpp"
#include "calmetrics_engine/native_runtime.hpp"
#include "operators/affine_contract.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>

#ifdef CALMETRICS_HAVE_EIGEN_ROTATION
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_MPL2_ONLY
#define EIGEN_FAST_MATH 0
#include <Eigen/Core>
#endif

namespace calmetrics_engine::graph {
namespace {
constexpr std::size_t tile_elements = 131072; // At most one MiB of float64.
constexpr std::size_t tile_rows = 4096;
constexpr std::size_t coefficient_pack_elements = 16384; // bounded 128 KiB.
constexpr std::size_t asset_packet_width = 8;

bool eigen_shape_candidate(std::size_t rows, std::size_t factors, std::size_t assets) {
  return rows && factors && assets >= asset_packet_width &&
      factors <= coefficient_pack_elements / assets;
}

// Both paths instantiate the single canonical factor fold. SIMD is only across
// independent assets; no horizontal reduction or factor subtotal is introduced.
template <class Epilogue>
void produce_asset_row(std::size_t factors, std::size_t assets,
    const ops::Value &coefficients, const ops::Value &initial,
    const double *source, std::ptrdiff_t base, std::ptrdiff_t factor_stride,
    double *destination, bool use_eigen, Audit &audit, const Epilogue &epilogue) {
  std::size_t asset = 0;
#ifdef CALMETRICS_HAVE_EIGEN_ROTATION
  if (use_eigen) {
    using Packet = Eigen::Array<double, asset_packet_width, 1>;
    using Read = Eigen::Map<const Packet, Eigen::Unaligned>;
    const auto full = assets - assets % asset_packet_width;
    const auto *coefficient_data = static_cast<const double *>(coefficients.data);
    for (; asset < full; asset += asset_packet_width) {
      Packet seed;
      for (std::size_t lane = 0; lane < asset_packet_width; ++lane)
        seed[static_cast<Eigen::Index>(lane)] = initial.f(asset + lane);
      const auto coefficient_at = [&](std::size_t factor) {
        return Read(coefficient_data + static_cast<std::ptrdiff_t>(factor) * coefficients.stride[0] +
                    static_cast<std::ptrdiff_t>(asset));
      };
      const auto value_at = [&](std::size_t factor) {
        return source[base + static_cast<std::ptrdiff_t>(factor) * factor_stride];
      };
      const Packet affine = ops::affine_contract_detail::ordered_fold(
          factors, seed, coefficient_at, value_at);
      for (std::size_t lane = 0; lane < asset_packet_width; ++lane)
        destination[asset + lane] = epilogue(affine[static_cast<Eigen::Index>(lane)]);
      // Counts actual Eigen-array producer lanes; it is not an instruction counter.
      audit.vector_elements += asset_packet_width;
    }
  }
#else
  (void)use_eigen; (void)audit;
#endif
  for (; asset < assets; ++asset) {
    const auto coefficient_at = [&](std::size_t factor) { return coefficients.at(factor, asset); };
    const auto value_at = [&](std::size_t factor) {
      return source[base + static_cast<std::ptrdiff_t>(factor) * factor_stride];
    };
    const auto affine = ops::affine_contract_detail::ordered_fold(
        factors, initial.f(asset), coefficient_at, value_at);
    destination[asset] = epilogue(affine);
  }
}

bool operation(const Program &p, std::uint32_t id, ops::Op op, std::size_t arity) {
  if (id >= p.nodes.size()) return false;
  const auto &n = p.nodes[id];
  return n.kind == NodeKind::operation && n.opcode == static_cast<std::uint16_t>(op) &&
      n.parent_count == arity;
}
bool literal(const Program &p, std::uint32_t id, std::size_t &value) {
  if (id >= p.nodes.size()) return false;
  const auto &n = p.nodes[id];
  if (n.kind != NodeKind::constant || !std::isfinite(n.constant) ||
      n.constant < 0 || n.constant > 9007199254740991.0 ||
      n.constant >= static_cast<double>(PTRDIFF_MAX) ||
      std::floor(n.constant) != n.constant) return false;
  value = static_cast<std::size_t>(n.constant);
  return true;
}
bool exact_literal(const Program &p, std::uint32_t id, std::size_t value) {
  std::size_t actual;
  return literal(p, id, actual) && actual == value;
}
bool static_input(const Program &p, std::uint32_t id) {
  if (id >= p.nodes.size()) return false;
  const auto &n = p.nodes[id];
  return n.kind == NodeKind::input && n.input_index < p.input_count &&
      !p.input_axes.empty() && p.input_axes[n.input_index] == 1;
}
std::size_t add(std::size_t a, std::size_t b) {
  ops::require(a <= static_cast<std::size_t>(PTRDIFF_MAX) &&
      b <= static_cast<std::size_t>(PTRDIFF_MAX) - a, "GRAPH_ARENA_OVERFLOW");
  return a + b;
}
std::size_t mul(std::size_t a, std::size_t b) {
  ops::require(!b || a <= static_cast<std::size_t>(PTRDIFF_MAX) / b, "GRAPH_ARENA_OVERFLOW");
  return a * b;
}

struct BoundGeometry {
  std::vector<ops::Value> values;
  std::vector<std::uint8_t> payload;
  std::size_t rows = 0, assets = 0, tile_capacity = 0, doubles = 0, coefficient_reserve = 0;
};

// Geometry-only descriptors for virtual arrays have no borrowed numerical
// backing. The shared validator sees precisely which operands are available.
bool bind_geometry(const Program &p, const AffineProjection &region,
    const std::vector<ops::Value> &inputs, std::size_t start, std::size_t length,
    const double *parameters, bool numerical, BoundGeometry &bound) {
  if (p.output_kind != OutputKind::typed || p.isolate_errors ||
      !p.apply_scopes.empty() || !p.rolling_scopes.empty()) return false;
  ops::require(inputs.size() == p.input_count, "GRAPH_INPUT_COUNT");
  for (const auto &v : inputs)
    if (v.kind != ops::Kind::number || v.shape.rank < 0 || v.shape.rank > 3) return false;
  bound.values.resize(p.nodes.size());
  bound.payload.assign(p.nodes.size(), 0);
  for (std::size_t id = 0; id < p.nodes.size(); ++id) {
    const auto &node = p.nodes[id];
    auto &v = bound.values[id];
    if (node.kind == NodeKind::input) {
      v = inputs[node.input_index];
      const auto axis = p.input_axes.empty() ? 0 : p.input_axes[node.input_index];
      if (axis != 1) {
        ops::require(v.shape.rank >= 1, "GRAPH_INPUT_TYPE");
        ops::require(start <= v.shape.dim[0] && length <= v.shape.dim[0] - start,
                     "GRAPH_INTERVAL_BOUNDS");
        v.shape.dim[0] = length;
        if (numerical && v.size() && v.data && start)
          v.data = static_cast<const double *>(v.data) +
              static_cast<std::ptrdiff_t>(start) * v.stride[0];
      }
      if (numerical) {
        ops::require(!v.shape.rank || !v.size() || v.data, "NULL_INPUT");
        if (!v.shape.rank && v.data) v.scalar = *static_cast<const double *>(v.data);
        bound.payload[id] = 1;
      } else v.data = nullptr;
      continue;
    }
    if (node.kind == NodeKind::constant || node.kind == NodeKind::parameter) {
      const bool constant = node.kind == NodeKind::constant;
      v = ops::Value::number(constant ? node.constant : numerical ? parameters[node.input_index] : 0.0);
      bound.payload[id] = constant || numerical;
      continue;
    }
    ops::Arguments<ops::Value> args(node.parent_count);
    std::uint32_t available = 0;
    for (std::size_t j = 0; j < node.parent_count; ++j) {
      args[j] = bound.values[node.parents[j]];
      if (bound.payload[node.parents[j]]) available |= 1u << j;
    }
    const auto &spec = ops::lookup(node.opcode);
    const auto shape = ops::validate_structure(spec, args.data(), node.parent_count,
                                             UINT32_MAX, available);
    ops::require(shape.geometry_known, "CONSTANT_LAYOUT_REQUIRED");
    v = ops::Value::number(0.0);
    v.kind = shape.output_kind; v.shape = shape.output_shape;
    v.set_contiguous_strides();
    // Only actual input views and weight extraction receive payloads here.
    // All other nodes stay geometry-only; no full affine/returns slot exists.
    if (numerical && available == (1u << node.parent_count) - 1 &&
        (spec.op == ops::Op::transpose || spec.op == ops::Op::slice_axis)) {
      const auto prepared = ops::prepare(spec, args.data(), node.parent_count);
      ops::require(prepared.borrowed, "INVALID_PHYSICAL_PAIR");
      v = prepared.view;
      bound.payload[id] = 1;
    } else if (numerical && spec.op == ops::Op::first && (available & 1u)) {
      ops::require(args[0].size() != 0, "INSUFFICIENT_SAMPLE");
      v.scalar = args[0].f(0); bound.payload[id] = 1;
    }
  }
  const auto &returns = bound.values[region.returns];
  if (returns.shape.rank != 2 && returns.shape.rank != 3) return false;
  const auto axis = returns.shape.rank - 1;
  if (region.asset_axis != static_cast<std::size_t>(axis)) return false;
  bound.assets = returns.shape.dim[axis];
  if (bound.assets > tile_elements) return false;
  bound.rows = returns.shape.dim[0];
  if (returns.shape.rank == 3) bound.rows = mul(bound.rows, returns.shape.dim[1]);
  for (const auto &consumer : region.periods) {
    const auto &w = bound.values[consumer.weights];
    if (w.shape.rank != 1 || w.shape.dim[0] != bound.assets) return false;
    const auto &root = bound.values[p.roots[consumer.root]];
    if (root.shape.rank != returns.shape.rank - 1 || (consumer.swapped && root.shape.rank != 2)) return false;
  }
  for (const auto &consumer : region.contributions) {
    const auto &w = bound.values[consumer.weights];
    if (w.shape.rank != 1 || w.shape.dim[0] != bound.assets || consumer.asset >= bound.assets) return false;
  }
  bound.tile_capacity = bound.assets ? std::min({tile_rows, bound.rows, tile_elements / bound.assets}) : 0;
  bound.doubles = add(mul(bound.tile_capacity, bound.assets), mul(region.contributions.size(), 3));
  const auto &producer_input = bound.values[p.nodes[region.affine].parents[0]];
  const auto factors = producer_input.shape.dim[producer_input.shape.rank - 1];
  // Geometry-only transpose descriptors do not prove physical contiguity.
  // Reserve for every candidate shape, even if this execution borrows K.
  bound.coefficient_reserve = affine_producer_coefficient_reserve(bound.rows, factors, bound.assets);
  bound.doubles = add(bound.doubles, bound.coefficient_reserve);
  return true;
}

std::size_t projection_metadata_bytes(const AffineProjection &region) {
  auto bytes = add(sizeof(AffineProjection), mul(region.periods.capacity(), sizeof(AffinePeriodRoot)));
  bytes = add(bytes, mul(region.contributions.capacity(), sizeof(AffineContributionRoot)));
  bytes = add(bytes, mul(region.contribution_links.capacity(), sizeof(AffineContributionLink)));
  return add(bytes, mul(region.unmatched_contributions.capacity(), sizeof(std::size_t)));
}
std::size_t descriptor_bytes(const Program &p, const AffineProjection &region) {
  return add(mul(p.nodes.size(), sizeof(ops::Value) + sizeof(std::uint8_t)),
             projection_metadata_bytes(region));
}

// One ordered mathematical body for periods with and without contributions.
// Consumers receive the exact weight-first product after the asset fold step;
// no subtotal, reassociation or extra product matrix is introduced.
template <class ProductConsumer>
double ordered_weighted_sum(const double *weights, std::ptrdiff_t weight_stride,
    const double *returns, std::size_t assets, ProductConsumer &&consume) {
  double total = 0.0;
  for (std::size_t asset = 0; asset < assets; ++asset) {
    const double product = weights[static_cast<std::ptrdiff_t>(asset) * weight_stride] * returns[asset];
    total += product;
    consume(asset, product);
  }
  return total;
}
void checkpoint(std::chrono::steady_clock::time_point deadline, const std::atomic<bool> *cancelled) {
  native::check_deadline(deadline);
  if (cancelled && cancelled->load()) throw native::Timeout("native affine projection cancelled");
}
} // namespace

std::size_t affine_producer_coefficient_reserve(std::size_t rows,
    std::size_t factors, std::size_t assets) {
  return eigen_shape_candidate(rows, factors, assets) ? mul(factors, assets) : 0;
}
bool affine_producer_packets_enabled(std::size_t coefficient_reserve) {
#ifdef CALMETRICS_HAVE_EIGEN_ROTATION
  return coefficient_reserve && std::fegetround() == FE_TONEAREST;
#else
  (void)coefficient_reserve;
  return false;
#endif
}
void affine_producer_linear_row(std::size_t factors, std::size_t outputs,
    const ops::Value &coefficients, const ops::Value &initial,
    const double *source, std::ptrdiff_t base, std::ptrdiff_t factor_stride,
    double *destination, bool use_packets, Audit &audit) {
  ops::require(!use_packets || coefficients.stride[1] == 1, "GRAPH_INPUT_TYPE");
  produce_asset_row(factors, outputs, coefficients, initial, source, base,
                    factor_stride, destination, use_packets, audit,
                    [](double value) { return value; });
}
void affine_producer_return_row(std::size_t factors, std::size_t assets,
    const ops::Value &coefficients, const ops::Value &initial,
    const double *source, std::ptrdiff_t base, std::ptrdiff_t factor_stride,
    double *destination, bool use_packets, Audit &audit) {
  ops::require(!use_packets || coefficients.stride[1] == 1, "GRAPH_INPUT_TYPE");
  produce_asset_row(factors, assets, coefficients, initial, source, base,
                    factor_stride, destination, use_packets, audit,
                    [](double value) { return std::expm1(value); });
}

std::shared_ptr<const AffineProjection> plan_affine_projection(const Program &p) {
  if (p.output_kind != OutputKind::typed || p.isolate_errors || p.roots.empty() ||
      !p.apply_scopes.empty() || !p.rolling_scopes.empty() || p.root_outputs.size() != p.roots.size()) return {};
  auto result = std::make_shared<AffineProjection>();
  std::vector<std::uint8_t> marked(p.nodes.size(), 0);
  std::uint32_t shared = UINT32_MAX;
  std::size_t asset_axis = SIZE_MAX;
  const auto accept_axis = [&](std::size_t axis) {
    if (asset_axis != SIZE_MAX && asset_axis != axis) return false;
    asset_axis = axis; return true;
  };
  const auto mark = [&](std::uint32_t id) {
    marked[id] = 1;
    for (std::size_t j = 0; j < p.nodes[id].parent_count; ++j)
      if (p.nodes[p.nodes[id].parents[j]].kind == NodeKind::constant) marked[p.nodes[id].parents[j]] = 1;
  };
  const auto accept_returns = [&](std::uint32_t id) {
    if (!operation(p, id, ops::Op::expm1, 1) || (shared != UINT32_MAX && shared != id)) return false;
    shared = id; mark(id); return true;
  };
  for (std::size_t root = 0; root < p.roots.size(); ++root) {
    if (p.root_outputs[root].dtype != OutputDType::float64) return {};
    auto id = p.roots[root];
    bool swapped = false;
    if (operation(p, id, ops::Op::swap_axes, 3)) {
      const auto &n = p.nodes[id];
      const bool swap = (exact_literal(p, n.parents[1], 0) && exact_literal(p, n.parents[2], 1)) ||
          (exact_literal(p, n.parents[1], 1) && exact_literal(p, n.parents[2], 0));
      if (!swap || p.root_outputs[root].rank != 2) return {};
      mark(id); id = n.parents[0]; swapped = true;
    }
    if (operation(p, id, ops::Op::sum_axis, 2)) {
      const auto &sum = p.nodes[id];
      std::size_t axis;
      if (!literal(p, sum.parents[1], axis) || axis != std::size_t(p.root_outputs[root].rank) ||
          (axis != 1 && axis != 2) || !accept_axis(axis) ||
          !operation(p, sum.parents[0], ops::Op::multiply, 2)) return {};
      const auto multiply = sum.parents[0]; const auto &product = p.nodes[multiply];
      if (!operation(p, product.parents[0], ops::Op::broadcast_like, 2)) return {};
      const auto broadcast = product.parents[0]; const auto &broadcast_node = p.nodes[broadcast];
      const auto weights = broadcast_node.parents[0], returns = product.parents[1];
      if (!static_input(p, weights) || broadcast_node.parents[1] != returns || !accept_returns(returns)) return {};
      mark(id); mark(multiply); mark(broadcast); marked[weights] = 1;
      result->periods.push_back({static_cast<std::uint32_t>(root), weights, swapped});
      continue;
    }
    if (swapped || p.root_outputs[root].rank != 0 || !operation(p, id, ops::Op::divide, 2)) return {};
    const auto &divide = p.nodes[id]; const auto divisor = divide.parents[1];
    if (p.nodes[divisor].kind != NodeKind::constant && p.nodes[divisor].kind != NodeKind::parameter) return {};
    const auto sum = divide.parents[0];
    if (!operation(p, sum, ops::Op::sum, 1) || !operation(p, p.nodes[sum].parents[0], ops::Op::multiply, 2)) return {};
    const auto multiply = p.nodes[sum].parents[0]; const auto &product = p.nodes[multiply];
    if (!operation(p, product.parents[0], ops::Op::first, 1) ||
        !operation(p, product.parents[1], ops::Op::slice_axis, 5)) return {};
    const auto first = product.parents[0], returns_slice = product.parents[1];
    const auto weights_slice = p.nodes[first].parents[0];
    if (!operation(p, weights_slice, ops::Op::slice_axis, 5)) return {};
    const auto &w = p.nodes[weights_slice], &r = p.nodes[returns_slice];
    const auto weights = w.parents[0], returns = r.parents[0];
    std::size_t asset, returns_asset, axis;
    if (!static_input(p, weights) || !exact_literal(p, w.parents[1], 0) ||
        !literal(p, w.parents[2], asset) || !exact_literal(p, w.parents[3], asset + 1) ||
        !exact_literal(p, w.parents[4], 1) || !literal(p, r.parents[1], axis) || (axis != 1 && axis != 2) ||
        !accept_axis(axis) ||
        !literal(p, r.parents[2], returns_asset) || returns_asset != asset ||
        !exact_literal(p, r.parents[3], asset + 1) || !exact_literal(p, r.parents[4], 1) ||
        !accept_returns(returns)) return {};
    for (auto node : {id, divisor, sum, multiply, first, returns_slice, weights_slice, weights}) mark(node);
    result->contributions.push_back({static_cast<std::uint32_t>(root), weights, divisor, sum, asset});
  }
  if (shared == UINT32_MAX) return {};
  const auto affine = p.nodes[shared].parents[0];
  if (!operation(p, affine, ops::Op::affine_contract, 3)) return {};
  const auto &producer = p.nodes[affine];
  const auto source = producer.parents[0], initial = producer.parents[2];
  auto coefficients = producer.parents[1];
  if (p.nodes[source].kind != NodeKind::input) return {};
  if (operation(p, coefficients, ops::Op::transpose, 1)) {
    mark(coefficients); coefficients = p.nodes[coefficients].parents[0];
  }
  if (!static_input(p, coefficients)) return {};
  const auto &init = p.nodes[initial];
  if (init.kind != NodeKind::constant && init.kind != NodeKind::parameter && !static_input(p, initial)) return {};
  for (auto node : {affine, source, coefficients, initial}) mark(node);
  // Every operation must belong to this exact proof. Public intermediate roots,
  // comparisons, hidden consumers and unrelated kernels therefore fail closed.
  for (std::size_t id = 0; id < p.nodes.size(); ++id)
    if ((p.nodes[id].kind == NodeKind::operation && !marked[id]) ||
        (p.nodes[id].kind != NodeKind::operation && p.nodes[id].kind != NodeKind::input &&
         p.nodes[id].kind != NodeKind::constant && p.nodes[id].kind != NodeKind::parameter)) return {};
  result->affine = affine; result->returns = shared; result->asset_axis = asset_axis;
  // Scalar failures follow their original topological order, independently of
  // the caller's presentation order for output roots.
  std::sort(result->contributions.begin(), result->contributions.end(), [&](const auto &a, const auto &b) {
    return p.roots[a.root] < p.roots[b.root];
  });
  // Static identity, rather than pointer aliases or equal numerical weights,
  // proves that both consumers use precisely the same product. Duplicate period
  // roots never become additional carriers for an already linked contribution.
  std::unordered_map<std::uint32_t, std::size_t> carriers;
  carriers.reserve(result->periods.size());
  for (std::size_t period = 0; period < result->periods.size(); ++period)
    carriers.emplace(result->periods[period].weights, period);
  struct PendingLink {
    std::size_t period, asset, contribution;
  };
  std::vector<PendingLink> pending;
  pending.reserve(result->contributions.size());
  for (std::size_t c = 0; c < result->contributions.size(); ++c) {
    const auto &consumer = result->contributions[c];
    const auto found = carriers.find(consumer.weights);
    if (found == carriers.end()) result->unmatched_contributions.push_back(c);
    else pending.push_back({found->second, consumer.asset, c});
  }
  std::sort(pending.begin(), pending.end(), [](const auto &a, const auto &b) {
    if (a.period != b.period) return a.period < b.period;
    if (a.asset != b.asset) return a.asset < b.asset;
    return a.contribution < b.contribution;
  });
  result->contribution_links.reserve(pending.size());
  std::size_t next = 0;
  for (std::size_t period = 0; period < result->periods.size(); ++period) {
    auto &consumer = result->periods[period];
    consumer.contribution_begin = result->contribution_links.size();
    while (next < pending.size() && pending[next].period == period) {
      result->contribution_links.push_back({pending[next].asset, pending[next].contribution});
      ++next;
    }
    consumer.contribution_end = result->contribution_links.size();
    if (consumer.contribution_begin != consumer.contribution_end) {
      const auto first = result->contribution_links[consumer.contribution_begin].contribution;
      bool dense = true;
      for (auto link = consumer.contribution_begin; link < consumer.contribution_end; ++link) {
        const auto offset = link - consumer.contribution_begin;
        const auto &entry = result->contribution_links[link];
        dense &= entry.asset == offset && entry.contribution == first + offset;
      }
      if (dense) consumer.dense_contribution_begin = first;
    }
  }
  return result;
}

std::optional<std::size_t> affine_projection_workspace_bytes(const Program &p,
    const AffineProjection &region, const std::vector<ops::Value> &inputs,
    std::size_t logical_window) {
  BoundGeometry bound;
  if (!bind_geometry(p, region, inputs, 0, logical_window, nullptr, false, bound)) return std::nullopt;
  return add(mul(bound.doubles, sizeof(double)), descriptor_bytes(p, region));
}

bool execute_affine_projection(const Program &p, const AffineProjection &region,
    const std::vector<ops::Value> &inputs, const double *parameters, std::size_t parameter_count,
    const std::int64_t *starts, const std::int64_t *ends, std::size_t rows,
    void *output, const ResultLayout *layout, std::size_t result_row, ops::Workspace &workspace,
    Audit &audit, std::chrono::steady_clock::time_point deadline,
    const std::atomic<bool> *cancelled) {
  if (!layout || p.output_kind != OutputKind::typed || p.isolate_errors ||
      !p.apply_scopes.empty() || !p.rolling_scopes.empty()) return false;
  ops::require(inputs.size() == p.input_count, "GRAPH_INPUT_COUNT");
  ops::require(parameter_count == p.parameter_count, "GRAPH_PARAMETER_COUNT");
  ops::require(!parameter_count || parameters, "GRAPH_NULL_PARAMETERS");
  ops::require(!rows || (starts && ends && output), "GRAPH_NULL_BATCH");
  ops::require(layout->columns == p.roots.size() && !layout->row_bytes.empty() &&
      layout->row_statuses.size() == layout->row_bytes.size() &&
      result_row <= layout->row_bytes.size() - 1 && rows <= layout->row_bytes.size() - 1 - result_row &&
      layout->row_bytes[result_row + rows] <= static_cast<std::size_t>(PTRDIFF_MAX) &&
      layout->slots.size() >= mul(add(result_row, rows), p.roots.size()), "RESULT_LAYOUT_BOUNDS");
  ops::require(reinterpret_cast<std::uintptr_t>(output) % alignof(double) == 0, "RESULT_OUTPUT_ALIGNMENT");
  checkpoint(deadline, cancelled);
  BoundGeometry bound;
  // Admit every interval before writing any result. Geometry-only checks do not
  // inspect numeric data, and cannot mistake an unsupported shape for success.
  std::size_t maximum = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    ops::require(starts[row] >= 0 && ends[row] >= starts[row], "GRAPH_INTERVAL_BOUNDS");
    const auto length = static_cast<std::size_t>(ends[row] - starts[row]);
    if (!bind_geometry(p, region, inputs, static_cast<std::size_t>(starts[row]), length,
                       nullptr, false, bound)) return false;
    maximum = std::max(maximum, bound.doubles);
    for (std::size_t root = 0; root < p.roots.size(); ++root) {
      const auto &shape = bound.values[p.roots[root]].shape;
      const auto &slot = layout->slots[(result_row + row) * p.roots.size() + root];
      ops::require(p.root_outputs[root].dtype == OutputDType::float64 &&
          shape.rank == p.root_outputs[root].rank && shape.size() <= slot.capacity, "RESULT_CAPACITY_MISMATCH");
      const auto limit = root + 1 < p.roots.size()
          ? layout->slots[(result_row + row) * p.roots.size() + root + 1].byte_offset
          : layout->row_bytes[result_row + row + 1];
      ops::require(slot.byte_offset >= layout->row_bytes[result_row + row] &&
          slot.byte_offset <= limit && mul(slot.capacity, sizeof(double)) <= limit - slot.byte_offset,
          "RESULT_LAYOUT_BOUNDS");
    }
  }
  // A retained larger workspace would otherwise escape the new plan's budget.
  if (workspace.doubles.capacity() != maximum) std::vector<double>().swap(workspace.doubles);
  if (workspace.indices.capacity()) std::vector<std::size_t>().swap(workspace.indices);
  workspace.ensure(maximum, 0);
  audit = {}; audit.rows = rows; audit.nodes = p.nodes.size();
  audit.result_shapes.resize(mul(rows, p.roots.size()));
  audit.root_statuses.resize(mul(rows, p.roots.size()));
  ops::Value packed_coefficients;
  bool packed_ready = false;
  for (std::size_t row = 0; row < rows; ++row) {
    checkpoint(deadline, cancelled);
    const auto length = static_cast<std::size_t>(ends[row] - starts[row]);
    ops::require(length >= p.minimum_observations, "INSUFFICIENT_OBSERVATIONS");
    if (!bind_geometry(p, region, inputs, static_cast<std::size_t>(starts[row]), length,
                       parameters, true, bound)) throw ops::Error("INVALID_PHYSICAL_PAIR");
    audit.max_window = std::max(audit.max_window, length);
    const auto &producer = p.nodes[region.affine];
    const auto &x = bound.values[producer.parents[0]], &coefficients = bound.values[producer.parents[1]],
               &initial = bound.values[producer.parents[2]];
    const auto factors = x.shape.dim[x.shape.rank - 1];
    const bool use_eigen = affine_producer_packets_enabled(bound.coefficient_reserve);
    const ops::Value *producer_coefficients = &coefficients;
    if (use_eigen && coefficients.stride[1] != 1) {
      if (!packed_ready) {
        ops::require(bound.coefficient_reserve <= maximum, "GRAPH_ARENA_OVERFLOW");
        auto *storage = workspace.doubles.data() + maximum - bound.coefficient_reserve;
        for (std::size_t factor = 0; factor < factors; ++factor)
          for (std::size_t asset = 0; asset < bound.assets; ++asset)
            storage[factor * bound.assets + asset] = coefficients.at(factor, asset);
        packed_coefficients = coefficients;
        packed_coefficients.data = storage;
        packed_coefficients.set_contiguous_strides();
        audit.algorithm_copy_bytes += mul(bound.coefficient_reserve, sizeof(double));
        packed_ready = true;
      }
      producer_coefficients = &packed_coefficients;
    }
    auto *tile = workspace.doubles.data();
    auto *totals = tile ? tile + bound.tile_capacity * bound.assets : nullptr;
    auto *weights = totals ? totals + region.contributions.size() : nullptr;
    auto *divisors = weights ? weights + region.contributions.size() : nullptr;
    for (std::size_t c = 0; c < region.contributions.size(); ++c) {
      const auto &consumer = region.contributions[c];
      totals[c] = 0.0; weights[c] = bound.values[consumer.weights].f(consumer.asset);
      divisors[c] = bound.values[consumer.divisor].scalar;
    }
    if (!region.contributions.empty()) ops::require(bound.rows != 0, "INSUFFICIENT_SAMPLE");
    for (std::size_t root = 0; root < p.roots.size(); ++root) {
      audit.result_shapes[row * p.roots.size() + root] = bound.values[p.roots[root]].shape;
      const auto &slot = layout->slots[(result_row + row) * p.roots.size() + root];
      const auto used = add(slot.byte_offset, mul(bound.values[p.roots[root]].size(), sizeof(double)));
      const auto limit = root + 1 < p.roots.size()
          ? layout->slots[(result_row + row) * p.roots.size() + root + 1].byte_offset
          : layout->row_bytes[result_row + row + 1];
      if (limit > used) std::memset(static_cast<std::uint8_t *>(output) + used, 0, limit - used);
    }
    const auto *source = static_cast<const double *>(x.data);
    const auto factor_stride = x.stride[x.shape.rank - 1];
    const auto capacity = bound.assets ? bound.tile_capacity : tile_rows;
    for (std::size_t first = 0; first < bound.rows; first += capacity) {
      checkpoint(deadline, cancelled);
      const auto count = std::min(capacity, bound.rows - first);
      for (std::size_t i = 0; bound.assets && i < count; ++i) {
        const auto logical_row = first + i;
        std::ptrdiff_t base = 0;
        // The empty contracted axis must not form an input offset or pointer.
        if (factors) {
          if (x.shape.rank == 2) base = static_cast<std::ptrdiff_t>(logical_row) * x.stride[0];
          else base = static_cast<std::ptrdiff_t>(logical_row / x.shape.dim[1]) * x.stride[0] +
              static_cast<std::ptrdiff_t>(logical_row % x.shape.dim[1]) * x.stride[1];
        }
        affine_producer_return_row(factors, bound.assets, *producer_coefficients, initial,
                          source, base, factor_stride, tile + i * bound.assets,
                          use_eigen, audit);
      }
      for (const auto &consumer : region.periods) {
        ops::Output destination;
        destination.kind = ops::Kind::number;
        destination.shape = bound.values[p.roots[consumer.root]].shape;
        destination.order = layout->order(consumer.root);
        destination.data = static_cast<std::uint8_t *>(output) +
            layout->slots[(result_row + row) * p.roots.size() + consumer.root].byte_offset;
        const auto &w = bound.values[consumer.weights];
        // Geometry admission has already proved a rank-1 float64 weights view.
        const auto *weight_data = static_cast<const double *>(w.data);
        const auto weight_stride = w.stride[0];
        const bool dense = consumer.dense_contribution_begin != SIZE_MAX &&
            consumer.contribution_end - consumer.contribution_begin == bound.assets;
        for (std::size_t i = 0; i < count; ++i) {
          // Zero assets must not form a pointer into the absent tile.
          const auto *returns = bound.assets ? tile + i * bound.assets : nullptr;
          double total;
          if (consumer.contribution_begin == consumer.contribution_end) {
            total = ordered_weighted_sum(weight_data, weight_stride, returns, bound.assets,
                                         [](std::size_t, double) {});
          } else if (dense) {
            auto *states = totals + consumer.dense_contribution_begin;
            total = ordered_weighted_sum(weight_data, weight_stride, returns, bound.assets,
                [&](std::size_t asset, double product) { states[asset] += product; });
          } else {
            auto link = consumer.contribution_begin;
            total = ordered_weighted_sum(weight_data, weight_stride, returns, bound.assets,
                                        [&](std::size_t asset, double product) {
              while (link < consumer.contribution_end && region.contribution_links[link].asset == asset) {
                // Each state sees logical step then path across all tiles.
                totals[region.contribution_links[link].contribution] += product;
                ++link;
              }
            });
          }
          auto logical = first + i;
          if (consumer.swapped) {
            const auto paths = x.shape.dim[1];
            logical = (logical % paths) * x.shape.dim[0] + logical / paths;
          }
          destination.set(logical, total);
        }
      }
      for (const auto c : region.unmatched_contributions) {
        const auto asset = region.contributions[c].asset;
        for (std::size_t i = 0; i < count; ++i) {
          const double product = weights[c] * tile[i * bound.assets + asset];
          // One scalar left fold spans all tiles, logical step then path.
          totals[c] += product;
        }
      }
      ++audit.pointwise_tiles;
    }
    for (std::size_t c = 0; c < region.contributions.size(); ++c) {
      const auto value = ops::scalar_math(ops::Op::divide, totals[c], divisors[c]);
      const auto &slot = layout->slots[(result_row + row) * p.roots.size() + region.contributions[c].root];
      std::memcpy(static_cast<std::uint8_t *>(output) + slot.byte_offset, &value, sizeof(value));
    }
    for (std::size_t root = 0; root < p.roots.size(); ++root)
      audit.direct_output_bytes += mul(bound.values[p.roots[root]].size(), sizeof(double));
  }
  audit.fused_pointwise_calls = rows;
  audit.numeric_arena_bytes = 0;
  audit.operator_workspace_capacity_bytes = add(workspace.capacity_bytes(),
      add(add(mul(bound.values.capacity(), sizeof(ops::Value)), bound.payload.capacity()),
          projection_metadata_bytes(region)));
  audit.pointwise_workspace_capacity_bytes = audit.operator_workspace_capacity_bytes;
  return true;
}

} // namespace calmetrics_engine::graph
