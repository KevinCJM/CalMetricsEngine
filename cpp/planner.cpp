#include "calmetrics_engine/planner.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include "calmetrics_engine/affine_projection.hpp"
#include "calmetrics_engine/compact_path.hpp"
#include "calmetrics_engine/pointwise.hpp"
#include "calmetrics_engine/native_process.hpp"
#include "operators/numeric_line_sort.hpp"
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
bool may_sort_numeric_line(const graph::Program &program) {
  for (const auto &node : program.nodes)
    if (node.kind == graph::NodeKind::operation &&
        node.opcode == static_cast<std::uint16_t>(ops::Op::quantiles_axis))
      return true;
  for (const auto &scope : program.rolling_scopes)
    if (scope.body && may_sort_numeric_line(*scope.body)) return true;
  for (const auto &scope : program.apply_scopes)
    if (scope.body && may_sort_numeric_line(*scope.body)) return true;
  return false;
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
// Direct replay is counted at the top-level call separately and here for
// recursively weighted child bodies, including healthy children of isolation.
double gaussian_stream_extra_work(const graph::Program &,
    const std::vector<ops::Value> &, std::size_t);
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
    if (op == ops::Op::transpose || ops::foundation_view(op) || ops::latent_view(op) || (op == ops::Op::diag && lhs.rank == 2))
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
    if (ops::latent_operator(op)) {
      if(op==ops::Op::gaussian_log_density || op==ops::Op::gaussian_mixture_em || op==ops::Op::gaussian_hmm_em) {
        const bool hmm=op==ops::Op::gaussian_hmm_em, fit=op!=ops::Op::gaussian_log_density;
        const auto &means=shapes[node.parents[fit?(hmm?3:2):1]];
        extent=double(lhs.dim[0])*means.dim[0]*(lhs.dim[1]+(hmm?means.dim[0]:0));
        if(fit){double iterations=100;const auto i=hmm?5u:4u;
          if(node.parent_count>i){const auto &p=program.nodes[node.parents[i]];
            iterations=p.kind==graph::NodeKind::constant&&p.constant>=1&&p.constant<=10000?p.constant:10000;}
          extent*=iterations;
        }
      } else if(op==ops::Op::hmm_filter || op==ops::Op::hmm_smooth)
        extent=double(lhs.dim[0])*lhs.dim[1]*lhs.dim[1];
      else if(op==ops::Op::sample_unique_indices)extent*=extent;
      else if(op==ops::Op::partition_mean || op==ops::Op::scatter_rows)extent+=sorting(double(rhs.size()));
    }
    else if (op == ops::Op::quantiles_axis) {
      const auto &axis_node = program.nodes[node.parents[2]];
      if (axis_node.kind == graph::NodeKind::constant && axis_node.constant >= 0 &&
          axis_node.constant < lhs.rank && axis_node.constant == std::floor(axis_node.constant)) {
        const auto axis = static_cast<std::size_t>(axis_node.constant);
        const auto n = static_cast<double>(lhs.dim[axis]);
        double lines = 1;
        for (int dimension = 0; dimension < lhs.rank; ++dimension)
          if (static_cast<std::size_t>(dimension) != axis) lines *= lhs.dim[dimension];
        extent = shapes[index].size() ? lines * (sorting(n) + rhs.size()) : 0;
      }
    }
    else if (op == ops::Op::affine_contract)
      extent = static_cast<double>(shapes[index].size()) * lhs.dim[lhs.rank - 1];
    else if (op == ops::Op::matmul)
      extent = static_cast<double>(lhs.dim[0]) * lhs.dim[1] * rhs.dim[1];
    else if ((op == ops::Op::covariance || op == ops::Op::correlation) && lhs.rank == 2)
      extent = static_cast<double>(lhs.dim[0]) * lhs.dim[1] * lhs.dim[1];
    else if (op == ops::Op::solve)
      extent = static_cast<double>(lhs.dim[0]) * lhs.dim[0] * lhs.dim[0];
    else if (op == ops::Op::cholesky || op == ops::Op::solve_multi ||
             op == ops::Op::triangular_solve || op == ops::Op::symmetric_eigh) {
      const double n=static_cast<double>(lhs.dim[0]);
      extent=n*n*(op==ops::Op::triangular_solve ? (rhs.rank==2 ? rhs.dim[1] : 1)
           : n+(op==ops::Op::solve_multi ? rhs.dim[1] : 0));
      if (op==ops::Op::symmetric_eigh) {
        double sweeps=32;
        if (node.parent_count>2) {
          const auto &option=program.nodes[node.parents[2]];
          sweeps=option.kind==graph::NodeKind::constant && option.constant>=1 && option.constant<=128
              ? option.constant : 128;
        }
        extent*=sweeps;
      }
    }
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
  if (full_work) extra += gaussian_stream_extra_work(program, inputs, observations);
  if (!std::isfinite(extra)) throw std::overflow_error("typed array work estimate overflow");
  return extra;
}
// The second density pass is physical work only. Use the executor's geometry
// proof, including the same static backing-slot stability metadata.
double gaussian_stream_extra_work(const graph::Program &program,
                                 const std::vector<ops::Value> &inputs,
                                 std::size_t observations) {
  if (program.isolate_errors || !program.execution_metadata ||
      program.execution_metadata->gaussian_filter_source.empty()) return 0;
  std::vector<ops::Shape> shapes;
  graph::required_array_capacity(program, inputs, observations, &shapes);
  double extra = 0;
  for (std::size_t filter = 0; filter < program.nodes.size(); ++filter) {
    if (!graph::gaussian_filter_stream_geometry(program, filter, shapes)) continue;
    const auto source = static_cast<std::size_t>(
        program.execution_metadata->gaussian_filter_source[filter]);
    const auto &x = shapes[program.nodes[source].parents[0]];
    extra += double(x.dim[0]) * shapes[filter].dim[1] * (4.0 * x.dim[1] + 4.0);
  }
  if (!std::isfinite(extra)) throw std::overflow_error("Gaussian streaming work estimate overflow");
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
                             bool inherited_isolation = false,
                             bool prebound_inputs = false,
                             bool allow_physical_fusion = true,
                             bool expanded_matmul_roots = true,
                             bool row_subviews = false) {
  const bool isolate_errors = program.isolate_errors || inherited_isolation;
  if (allow_physical_fusion && !isolate_errors && !prebound_inputs && inputs &&
      program.execution_metadata && program.execution_metadata->row_batch) {
    std::vector<ops::Value> blocked;
    std::size_t batch = 0, count = 0;
    if (graph::row_batch_inputs(program, *inputs, logical_window, blocked, batch, count)) {
      auto canonical = program;
      canonical.execution_metadata = program.execution_metadata->row_batch->canonical_metadata;
      std::vector<ops::Shape> shapes;
      const auto capacity = graph::required_array_capacity(canonical, blocked, logical_window, &shapes);
      const auto child = scratch_estimate(canonical, capacity, &blocked, logical_window,
          false, false, false);
      // Shorter intervals may increase batch width across floor(target/T).
      // Each varying node has at most min(64*w, max(target,w)) elements for
      // any w <= its upper width. These are arena sizing shapes only, never
      // logical shapes or inputs used by an operator.
      constexpr std::size_t target = 163840;
      auto upper = shapes;
      const auto &axes = program.execution_metadata->row_batch->axes;
      for (std::size_t id = 0; id < upper.size(); ++id) {
        const auto axis = axes[id];
        if (axis < 0) continue;
        const auto width = shapes[id].size() / batch;
        if (!width) continue;
        const auto bound = std::min(checked_mul(64, width), std::max(target, width));
        upper[id].dim[axis] = checked_add(bound, width - 1) / width;
      }
      graph::ArrayArenaLayout current_arena, upper_arena;
      graph::plan_array_arena(canonical, capacity, shapes, current_arena, false, false, true, true);
      graph::plan_array_arena(canonical, capacity, upper, upper_arena, false, false, true, true);
      const auto arena_bytes = [&](const graph::ArrayArenaLayout &arena) {
        return checked_add(checked_mul(checked_add(arena.elements[0], arena.elements[2]), 8), arena.elements[1]);
      };
      const auto current_bytes = arena_bytes(current_arena);
      const auto upper_bytes = arena_bytes(upper_arena);
      const auto upper_gap = upper_bytes > current_bytes ? upper_bytes - current_bytes : 0;
      // A shorter history can change a row block from fewer than 32 rows
      // into the packed MatMul geometry. Arena shapes above are not logical
      // operator shapes: bound the actual independent row count separately.
      std::size_t packed_upper = 0;
      for (const auto &node : canonical.nodes) {
        if (node.kind != graph::NodeKind::operation ||
            node.opcode != static_cast<std::uint16_t>(ops::Op::matmul) ||
            node.parent_count != 2 || axes[node.parents[0]] != 0)
          continue;
        auto a = shapes[node.parents[0]];
        const auto &b = shapes[node.parents[1]];
        if (a.rank != 2 || b.rank != 2) continue;
        a.dim[0] = std::min<std::size_t>(64, count);
        packed_upper = std::max(packed_upper, ops::matmul_scratch_doubles(a, b));
      }
      // One reused Workspace holds the maximum, not a panel per interval.
      const auto current_work = graph::matrix_workspace_doubles(canonical, shapes, capacity);
      const auto packed_gap = checked_mul(
          packed_upper > current_work ? packed_upper - current_work : 0, sizeof(double));
      // F matrix roots need a bounded C block before scattering into their
      // full leading dimensions. Reserve it for all orders so geometry-only
      // plans remain safe if a supported input layout selects F automatically.
      std::size_t result_block = 0;
      bool matrix_root = false;
      for (const auto root : program.roots) {
        const auto &shape = shapes[root];
        matrix_root = matrix_root || shape.rank == 2;
        const auto width = shape.size() / batch;
        result_block = checked_add(result_block,
            std::min(checked_mul(64, width), std::max(target, width)));
      }
      const auto result_bytes = matrix_root ? checked_mul(result_block, sizeof(double)) : 0;
      const auto descriptors = checked_add(checked_mul(inputs->size(), sizeof(ops::Value)),
          checked_add(checked_mul(program.nodes.size(),
              sizeof(graph::Node) + sizeof(const char *) + 3 * sizeof(ops::Shape)),
              checked_mul(program.roots.size(), sizeof(graph::ResultSlot) + 3 * sizeof(ops::Shape))));
      return checked_add(child, checked_add(result_bytes, checked_add(packed_gap,
          checked_add(upper_gap, checked_add(descriptors, 4096)))));
    }
  }
  if (allow_physical_fusion && !inherited_isolation && !prebound_inputs && inputs && program.execution_metadata &&
      program.execution_metadata->affine_projection) {
    const auto bytes = graph::affine_projection_workspace_bytes(program,
        *program.execution_metadata->affine_projection, *inputs, logical_window);
    if (bytes) return *bytes;
  }
  if (allow_physical_fusion && !inherited_isolation && inputs && program.execution_metadata && program.execution_metadata->pointwise) {
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
  std::vector<ops::Shape> shapes;
  if (inputs && program.execution_metadata && program.execution_metadata->requires_shape_planning)
    graph::required_array_capacity(program, *inputs, logical_window, &shapes);
  graph::ArrayArenaLayout arena;
  // Child execution also disables direct F Cholesky writing. Reserve its C
  // arena using the same downgrade predicate as the executor.
  graph::plan_array_arena(program,window,shapes,arena,
      allow_physical_fusion && !isolate_errors,prebound_inputs, true,
      !isolate_errors && !prebound_inputs && program.execution_metadata &&
          (!program.execution_metadata->where_sum_to.empty() ||
           !program.execution_metadata->scan_reductions.empty()));
  std::size_t estimate=checked_add(checked_mul(checked_add(arena.elements[0],arena.elements[2]),8),
                                  arena.elements[1]);
  estimate=checked_add(estimate,checked_mul(checked_add(checked_add(program.numeric_slots,
      program.mask_slots),program.integer_slots),sizeof(std::size_t)));
  estimate = checked_add(
      estimate, checked_mul(program.nodes.size(), sizeof(ops::Value) + 160));
  if (row_subviews && allow_physical_fusion && !isolate_errors && !prebound_inputs &&
      program.execution_metadata && shapes.size() == program.nodes.size() &&
      std::any_of(program.execution_metadata->row_subview_takes.begin(),
          program.execution_metadata->row_subview_takes.end(), [&](std::uint32_t id) {
            return shapes[id].size() >= graph::row_subview_min_elements;
          })) {
    // Keep the original materialization budget: unknown/noncontiguous indices
    // retain canonical execution. Add per-call proof tables and bounded stack.
    estimate = checked_add(estimate, checked_add(65536,
        checked_mul(program.nodes.size(), sizeof(std::size_t) + sizeof(std::uint32_t) + 1)));
  }
  if (program.execution_metadata && program.execution_metadata->pointwise)
    estimate = checked_add(estimate, graph::pointwise_workspace_bytes(program, window));
  if (allow_physical_fusion && !isolate_errors && program.execution_metadata &&
      !program.execution_metadata->state_decisions.empty())
    estimate = checked_add(estimate, graph::state_decision_tile_elements);
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
  // Sequential nodes reuse one Workspace; scopes/branches account for their
  // own copies through the existing recursive and per-worker admission.
  estimate = checked_add(estimate, checked_mul(
      graph::matrix_workspace_doubles(program, shapes, window, expanded_matmul_roots), sizeof(double)));
  if (allow_physical_fusion && !isolate_errors && !prebound_inputs) {
    const auto statistics=graph::matmul_summary_cache_doubles(program,shapes);
    estimate=checked_add(estimate,checked_mul(statistics,sizeof(double)));
    if (statistics) {
      estimate=checked_add(estimate,checked_mul(
          program.execution_metadata->matmul_summaries.size()+1,sizeof(std::size_t)));
      // Vendor scratch is reserved even when a numerical guard selects replay.
      estimate=checked_add(estimate,1024*1024);
    }
  }
  std::unordered_set<std::uint32_t> order_sources;
  bool operator_scratch = false, latent_workspace = false;
  for (const auto &node : program.nodes) {
    if (node.kind != graph::NodeKind::operation)
      continue;
    const auto op = ops::lookup(node.opcode).op;
    latent_workspace = latent_workspace || op == ops::Op::gaussian_log_density ||
        op == ops::Op::gaussian_mixture_em || op == ops::Op::gaussian_hmm_em ||
        op == ops::Op::hmm_smooth || op == ops::Op::partition_mean ||
        op == ops::Op::scatter_rows;
    if (op == ops::Op::median || op == ops::Op::quantile)
      order_sources.insert(node.parents[0]);
    if (op == ops::Op::median || op == ops::Op::quantile ||
        op == ops::Op::median_where || op == ops::Op::quantile_where ||
        op == ops::Op::rolling_min || op == ops::Op::rolling_max ||
        op == ops::Op::argsort || op == ops::Op::distinct_count || op == ops::Op::ps_filter ||
        op == ops::Op::solve || op == ops::Op::solve_multi || op == ops::Op::symmetric_eigh || op == ops::Op::covariance ||
        op == ops::Op::correlation || op == ops::Op::quadratic_form)
      operator_scratch = true;
  }
  estimate = checked_add(
      estimate, checked_mul(checked_mul(order_sources.size(), window), 8));
  if (operator_scratch)
    estimate = checked_add(estimate, checked_mul(window, 24));
  if (!program.rolling_scopes.empty())
    estimate = checked_add(estimate, checked_mul(window + 1, sizeof(std::size_t)));
  if (inputs && shapes.empty() && (latent_workspace || !program.rolling_scopes.empty() || !program.apply_scopes.empty()))
    graph::required_array_capacity(program, *inputs, logical_window, &shapes);
  if (latent_workspace) {
    std::size_t doubles = 0, indices = 0;
    for (const auto &node : program.nodes) {
      if (node.kind != graph::NodeKind::operation) continue;
      const auto op = ops::lookup(node.opcode).op;
      if (op == ops::Op::partition_mean || op == ops::Op::scatter_rows) {
        indices = std::max(indices, shapes.empty() ? window : shapes[node.parents[1]].size());
      } else if (op == ops::Op::gaussian_log_density || op == ops::Op::gaussian_mixture_em ||
                 op == ops::Op::gaussian_hmm_em || op == ops::Op::hmm_smooth) {
        // NK is an internal EM extent, not necessarily any DAG node's shape.
        // In particular K>D invalidates a bound based only on max node size.
        if (shapes.empty()) {
          doubles = std::max(doubles, checked_add(checked_mul(checked_mul(window, window), 5),
                                                 checked_mul(window, 4)));
        } else {
          std::array<ops::Value, 5> arguments{};
          for (std::size_t i = 0; i < std::min<std::size_t>(node.parent_count, arguments.size()); ++i)
            arguments[i].shape = shapes[node.parents[i]];
          // Tolerance can be a runtime binding. Reserve the full-history
          // envelope for HMM rather than infer a zero tolerance from a shape.
          const auto scratch_for_shape = [&]() {
            return op == ops::Op::gaussian_hmm_em ? ops::latent_hmm_full_scratch(
                arguments[0].shape.dim[0],arguments[1].shape.size()) :
                ops::latent_scratch(op,arguments.data());
          };
          auto bound = scratch_for_shape();
          if (op == ops::Op::gaussian_mixture_em) {
            const auto k = arguments[1].shape.size();
            if (bound < checked_mul(arguments[0].shape.dim[0], k)) {
              // Maximum shapes may select bounded replay while a shorter row
              // retains NK. Cover both the cutoff and storage>=NK boundaries;
              // derive geometry through the same primitive, never a second
              // allocation formula or an actual unused posterior reservation.
              arguments[0].shape.dim[0] = std::max(
                  (ops::latent_gmm_replay_min_elements - 1) / k, bound / k);
              bound = std::max(bound, scratch_for_shape());
            }
          }
          if (op == ops::Op::gaussian_hmm_em && ops::latent_fit_matrix_count(
                  op, arguments[0].shape.dim[0], arguments[1].shape.size()) == 2) {
            // Shapes are upper extents, not exact lengths for every interval
            // or child scope. The physical two-history cutoff is non-monotone:
            // a shorter row can still require three histories. Reuse the unique
            // primitive geometry for the peak below the cutoff; do not reserve
            // or allocate a third history for the actual two-history execution.
            arguments[0].shape.dim[0] = ops::latent_hmm_replay_min_rows - 1;
            bound = std::max(bound, scratch_for_shape());
          }
          doubles = std::max(doubles, bound);
        }
      }
    }
    // The two reusable buffers can retain different high-water marks.
    estimate = checked_add(estimate, checked_mul(doubles, sizeof(double)));
    estimate = checked_add(estimate, checked_mul(indices, sizeof(std::size_t)));
  }
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
    auto bytes = scratch_estimate(body, capacity, &captured, length, child_isolation, true);
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
std::size_t canonical_child_scratch_bytes(const graph::Program &program,
    const std::vector<ops::Value> &inputs, std::size_t logical_window) {
  const auto capacity = graph::required_array_capacity(program, inputs, logical_window);
  // Deferred original-node failures disable pair/pointwise fusion in the
  // private post executor. Admission must reserve that same canonical arena.
  return scratch_estimate(program, capacity, &inputs, logical_window, false, true, false);
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
  std::optional<graph::CompactPathBudget> compact;
  if (inputs && graph->program.execution_metadata &&
      graph->program.execution_metadata->compact_path)
    compact = graph::compact_path_budget(graph->program,
        *graph->program.execution_metadata->compact_path, *inputs,
        geometry.row_lengths, canonical_child_scratch_bytes);
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
        checked_add(checked_mul(checked_add(rows, 1), 2 * sizeof(std::size_t)),
                    p->graph->program.roots.size() * sizeof(ops::MemoryOrder)));
    // Reject an impossible metadata budget before reserving rows * roots slots.
    if (memory_budget && p->estimated_result_metadata_bytes > *memory_budget)
      throw std::bad_alloc();
    p->result_layout = std::make_shared<graph::ResultLayout>(
        graph::result_layout(p->graph->program, *inputs, starts, ends, rows));
    p->estimated_output_bytes = p->result_layout->bytes();
    p->estimated_status_bytes = p->graph->program.isolate_errors
        ? checked_mul(p->result_layout->status_count(), 2) : 0;
  }
  p->estimated_worker_scratch_bytes = compact ? compact->total_bytes :
      scratch_estimate(p->graph->program, array_capacity, inputs, geometry.max_window,
          false, false, true, p->expanded_matmul_roots, p->row_subviews);
  if (compact) p->reason_codes.push_back("bounded_compact_scan_consumer_closure");
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
  p->estimated_work_units = compact ? compact->physical_work :
      geometry_work(p->graph->physical_cost, geometry);
  const bool typed_arrays = !compact && inputs && requires_typed_work(p->graph->program);
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
  } else if (compact) {
    // Keep the unfused logical comparison visible. The physical estimate
    // includes the actual number of bounded producer batches and post work.
    for (auto length : geometry.row_lengths) {
      auto found = extra_by_length.find(length);
      if (found == extra_by_length.end())
        found = extra_by_length.emplace(length,
            typed_array_extra_work(p->graph->program, *inputs, length)).first;
      p->estimated_logical_work_units += found->second;
      const auto row = graph::compact_path_budget(p->graph->program,
          *p->graph->program.execution_metadata->compact_path, *inputs,
          std::vector<std::size_t>{length}, canonical_child_scratch_bytes);
      require(row.has_value(), "compact geometry admission changed");
      geometry.row_work_units.push_back(row->physical_work);
    }
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
  std::map<std::size_t, double> stream_by_length;
  if (!compact && inputs && !p->graph->program.isolate_errors &&
      !p->graph->program.execution_metadata->gaussian_filter_source.empty()) {
    if (geometry.row_work_units.empty())
      for (auto length : geometry.row_lengths)
        geometry.row_work_units.push_back(row_work(p->graph->physical_cost, length));
    double replay = 0;
    for (std::size_t row = 0; row < rows; ++row) {
      const auto length = geometry.row_lengths[row];
      auto found = stream_by_length.find(length);
      if (found == stream_by_length.end()) found = stream_by_length.emplace(length,
          gaussian_stream_extra_work(p->graph->program, *inputs, length)).first;
      replay += found->second;
      geometry.row_work_units[row] += found->second;
    }
    p->estimated_work_units += replay;
    if (replay > 0) p->reason_codes.push_back("gaussian_filter_stream_replay_work_accounted");
  }
  std::vector<std::map<std::size_t, double>> branch_extra(p->graph->branches.size());
  const auto branch_work = [&](std::size_t branch, std::size_t length) {
    const auto &info = p->graph->branches[branch];
    const double base = row_work(info.cost, length);
    if (!inputs) return base;
    auto &cache = branch_extra[branch];
    auto found = cache.find(length);
    if (found == cache.end()) found = cache.emplace(length,
        (typed_arrays ? typed_array_extra_work(info.program, *inputs, length) : 0) +
        gaussian_stream_extra_work(info.program, *inputs, length)).first;
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
    if (typed_arrays || !stream_by_length.empty()) {
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

  // Reuse the explicit independent-row proof, retaining every shared root and
  // each row's complete time/contracting axes in the same native task.
  std::size_t array_rows = 0, row_range_overhead = 0;
  if (!process && rows == 1 && cpu > 1 && inputs && p->result_layout &&
      p->graph->program.execution_metadata->row_batch) {
    std::vector<ops::Value> blocked;
    std::size_t batch = 0, count = 0;
    if (graph::row_batch_inputs(p->graph->program, *inputs, geometry.max_window,
        blocked, batch, count)) array_rows = count;
  }
  const auto row_workers = std::min({cpu, array_rows / 128,
      static_cast<std::size_t>(std::min(static_cast<double>(cpu),
          p->estimated_work_units / config.thread_work_units))});

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
  // Retry only the explicit budget-rejection branch, not arbitrary allocation
  // errors. Geometry and result layout stay shared with this one Plan; reset
  // all physical lane/transport choices before considering the original domain.
  auto base_scratch = p->estimated_worker_scratch_bytes;
  const auto base_reasons = p->reason_codes.size();
  const auto base_dimension = p->parallel_dimension;
  for (;;) {
    p->estimated_worker_scratch_bytes = base_scratch;
    p->lane = "single"; p->parallel_dimension = base_dimension;
    p->process_count = p->thread_count = p->threads_per_process = 1;
    p->tensor_elements = 0; p->use_shared_memory = false;
    p->chunks.clear(); row_range_overhead = 0;
    p->reason_codes.resize(base_reasons);
    if (!p->expanded_matmul_roots)
      p->reason_codes.push_back("memory_budget_uses_original_matmul_geometry");
    if (!p->row_subviews)
      p->reason_codes.push_back("memory_budget_uses_original_row_geometry");
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
    } else if (row_workers >= 2) {
      p->lane = "thread"; p->parallel_dimension = "array_row";
      p->tensor_elements = array_rows; p->thread_count = row_workers;
      row_range_overhead = checked_add(checked_mul(inputs->size(), sizeof(ops::Value)),
          checked_add(checked_mul(p->graph->program.nodes.size(), 2 * sizeof(ops::Shape)),
              checked_add(checked_mul(p->graph->program.roots.size(),
                  sizeof(graph::ResultSlot) + sizeof(ops::Shape) + sizeof(std::int16_t)), 4096)));
      p->estimated_worker_scratch_bytes = checked_add(p->estimated_worker_scratch_bytes,
          row_range_overhead);
      p->reason_codes.push_back("independent_matrix_row_closure");
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

    // Logical consumers can have tiny scalar/compact roots while sorting a large
    // private source. Check all scope bodies, not root sizes or numerical guards.
    // Charge one fixed estimated library stack/TLS resource per active worker,
    // after compact/branch scratch selection. This never allocates a fake buffer.
    if (rows && ops::axis_path_detail::numeric_line_sort_available() &&
        may_sort_numeric_line(p->graph->program)) {
      p->estimated_worker_scratch_bytes = checked_add(
          p->estimated_worker_scratch_bytes,
          ops::axis_path_detail::native_sort_library_reserve_bytes);
      p->reason_codes.push_back("native_numeric_line_sort_resource_reserve");
    }
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
    // Optional proofs must not reject the previously admissible geometry or
    // reduce its worker budget just to reserve this new optimization's tables.
    if (memory_budget && p->estimated_total_memory_bytes > *memory_budget &&
        p->row_subviews && !compact) {
      const auto original_scratch = scratch_estimate(p->graph->program,
          array_capacity, inputs, geometry.max_window, false, false, true,
          p->expanded_matmul_roots, false);
      if (original_scratch < base_scratch) {
        base_scratch = original_scratch;
        p->row_subviews = false;
        continue;
      }
    }
    bool reduced = false;
    while (memory_budget && p->estimated_total_memory_bytes > *memory_budget &&
           workers > 1) {
      --workers;
      reduced = true;
      // Fewer workers enlarge each response; recheck before memory admission.
      check_transport();
      p->estimated_total_memory_bytes = total_memory(*p, workers);
    }
    if (!process && row_range_overhead && workers == 1) {
      // A constrained request must retain the old admissible single path; range
      // descriptions are not used by that path and must not block its admission.
      p->estimated_worker_scratch_bytes -= row_range_overhead;
      p->lane = "single"; p->parallel_dimension = "interval";
      p->tensor_elements = 0;
      p->reason_codes.erase(std::remove(p->reason_codes.begin(), p->reason_codes.end(),
          "parallelize_by_array_row"), p->reason_codes.end());
      p->reason_codes.push_back("row_parallelism_reduced_to_single");
      p->estimated_total_memory_bytes = total_memory(*p, 1);
    }
    if (memory_budget && p->estimated_total_memory_bytes > *memory_budget) {
      if (p->expanded_matmul_roots && !compact) {
        const auto original_scratch = scratch_estimate(p->graph->program,
            array_capacity, inputs, geometry.max_window, false, false, true, false,
            p->row_subviews);
        if (original_scratch < base_scratch) {
          base_scratch = original_scratch;
          p->expanded_matmul_roots = false;
          continue;
        }
      }
      throw std::bad_alloc();
    }
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
    } else if (p->parallel_dimension == "array_row") {
      for (std::size_t i = 0; i < workers; ++i)
        p->chunks.push_back({p->tensor_elements * i / workers,
                            p->tensor_elements * (i + 1) / workers});
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
  if (p.parallel_dimension == "array_row") {
    require(p.lane == "thread" && p.thread_count >= 2 && p.thread_count <= p.cpu_budget &&
        rows == 1 && inputs && p.result_layout && p.graph->program.execution_metadata->row_batch &&
        p.chunks.size() == p.thread_count, "invalid independent row plan");
    std::vector<ops::Value> blocked;
    std::size_t batch = 0, count = 0;
    require(graph::row_batch_inputs(p.graph->program, *inputs, geometry.max_window,
        blocked, batch, count) && count == p.tensor_elements, "stale independent row geometry");
    std::size_t end = 0;
    for (const auto chunk : p.chunks) {
      require(chunk.begin == end && chunk.end >= chunk.begin && chunk.end <= count &&
          chunk.end - chunk.begin >= 128, "invalid independent row range");
      end = chunk.end;
    }
    require(end == count, "incomplete independent row ranges");
  }
}
} // namespace calmetrics_engine::planner
