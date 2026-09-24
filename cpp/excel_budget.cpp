#include "excel_internal.hpp"
#include <limits>

namespace calmetrics_engine::excel {
namespace {
using N = std::size_t;
N add(N a, N b) {
  if (b > std::numeric_limits<N>::max() - a)
    throw Error("OVER_BUDGET: symbolic integer addition overflow");
  return a + b;
}
N mul(N a, N b) {
  if (b && a > std::numeric_limits<N>::max() / b)
    throw Error("OVER_BUDGET: symbolic integer multiplication overflow");
  return a * b;
}
N product(std::initializer_list<N> xs) {
  N v = 1;
  for (auto x : xs)
    v = mul(v, x);
  return v;
}
N sum_cells(N n) {
  N result = 0;
  while (n > 1) {
    n = n / 32 + (n % 32 != 0);
    result = add(result, n);
  }
  return result;
}
N all_cells(N n) { return add(n, sum_cells(n)); }
N triangular(N n) {
  return n % 2 ? mul(n, add(n, 1) / 2) : mul(n / 2, add(n, 1));
}
struct Cost {
  N cells = 0, formulas = 0, work = 0, guards = 0;
  void steps(N n) {
    cells = add(cells, n);
    formulas = add(formulas, n);
  }
  void text(N n) { cells = add(cells, n); }
  void include(const Cost &b, N times = 1) {
    cells = add(cells, mul(b.cells, times));
    formulas = add(formulas, mul(b.formulas, times));
    work = add(work, mul(b.work, times));
    guards = add(guards, mul(b.guards, times));
  }
};
struct Value {
  ops::Shape shape;
  ops::Kind kind = ops::Kind::number;
  std::optional<double> scalar;
  bool positions = false, unavailable = false;
  N size() const {
    if (shape.rank < -1 || shape.rank > 3)
      throw Error("INVALID_INPUT: symbolic rank");
    if (shape.rank < 0)
      return 1;
    if (kind == ops::Kind::fit)
      return 5;
    if (kind == ops::Kind::interval)
      return 4;
    N n = 1;
    for (int i = 0; i < shape.rank; ++i)
      n = mul(n, shape.dim[i]);
    return n;
  }
};
Value literal(double x) {
  Value v;
  v.scalar = x;
  return v;
}
struct Result {
  std::vector<Value> roots;
  Cost cost;
};
struct Analyzer {
  Limits limits;
  std::function<void()> checkpoint;
  std::chrono::steady_clock::time_point started =
      std::chrono::steady_clock::now();
  std::vector<BudgetNode> largest;
  Analyzer(Limits l, std::function<void()> c)
      : limits(l), checkpoint(std::move(c)) {}
  void check(const Cost &c) const {
    if (checkpoint)
      checkpoint();
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      started)
            .count() > limits.timeout_seconds)
      throw Error("TIMEOUT: symbolic Excel budget");
    const auto chars = mul(c.formulas, 8192);
    if (c.cells > limits.cells || chars > limits.formula_characters ||
        c.work > limits.work_units)
      throw Error(
          "OVER_BUDGET: symbolic cell/character/work bound; cells=" +
          std::to_string(c.cells) +
          ", cell_budget=" + std::to_string(limits.cells) +
          ", formula_characters=" + std::to_string(chars) +
          ", character_budget=" + std::to_string(limits.formula_characters) +
          ", work=" + std::to_string(c.work));
  }
  N bound(const Value &v, Cost &c, const char *name, N maximum = 1000000) {
    if (!v.scalar || !std::isfinite(*v.scalar) || *v.scalar < 0 ||
        *v.scalar > double(maximum) || std::floor(*v.scalar) != *v.scalar)
      throw Error(std::string("UNKNOWN_BOUND: symbolic ") + name);
    c.guards = add(c.guards, 1);
    return N(*v.scalar);
  }
  Value slice(Value v, N n) {
    if (v.shape.rank != 1 || n > v.size())
      throw Error("INVALID_INPUT: symbolic slice geometry");
    v.shape = ops::vector_shape(n);
    v.scalar.reset();
    return v;
  }
  N reduction(Op op, N n, bool masked = false) {
    if (!n)
      return 0;
    if (op >= Op::sum_where && op <= Op::quantile_where) {
      const Op map[]{Op::sum,       Op::mean,      Op::variance, Op::std,
                     Op::min_value, Op::max_value, Op::median,   Op::quantile};
      op = map[unsigned(op) - unsigned(Op::sum_where)];
      masked = true;
    }
    if (op == Op::median || op == Op::quantile)
      return add(add(mul(n, add(n, sum_cells(n))), mul(3, n)), 2);
    if (op == Op::min_value || op == Op::max_value || op == Op::argmin ||
        op == Op::argmax)
      return add(add(mul(3, n), 1), masked ? all_cells(n) : 0);
    if (op == Op::sum || op == Op::product || op == Op::root_mean_square)
      return add(mul(2, n), 1);
    if (op == Op::mean || op == Op::variance || op == Op::std)
      return add(mul(4, n), 1);
    return add(add(mul(5, n), sum_cells(n)), 1);
  }
  Value operation(Op op, const std::vector<Value> &given, Cost &c,
                  bool graph_mode, bool isolate) {
    const auto &spec = ops::lookup(static_cast<std::uint16_t>(op));
    if (given.size() < spec.min_args || given.size() > spec.max_args)
      throw Error("INVALID_INPUT: symbolic operator arity");
    const N n = given[0].size();
    N work = 0;
    if (op == Op::ps_filter)
      work = product({n, n, n, n, 100});
    else if (op == Op::argsort || op == Op::median || op == Op::quantile ||
             op == Op::median_where || op == Op::quantile_where ||
             op == Op::local_extrema || op == Op::between_events ||
             op == Op::distinct_count || op == Op::phase_direction ||
             op == Op::drawdown_cycle_reference)
      work = product({n, n, 24});
    else if (op == Op::solve) {
      auto m = given[1].size();
      work = product({m, m, m, 32});
    } else if (op == Op::matmul)
      work = product({given[0].shape.dim[0], given[0].shape.dim[1],
                      given[1].shape.dim[1], 16});
    else if (op == Op::quadratic_form)
      work = product({n, n, 16});
    else if ((op == Op::covariance || op == Op::correlation) &&
             given[0].shape.rank == 2)
      work = product({given[0].shape.dim[0], given[0].shape.dim[1],
                      given[0].shape.dim[1], 32});
    else if (op == Op::rolling_min || op == Op::rolling_max)
      work = product({n, n, 16});
    else
      work = mul(add(n, 1), 200);
    c.work = add(c.work, work);
    check(c);
    std::vector<ops::Value> native;
    for (auto &a : given) {
      ops::Value v;
      v.kind = a.kind;
      v.shape = a.shape;
      v.scalar = a.scalar.value_or(0);
      native.push_back(v);
    }
    Value out;
    auto a = given;
    bool unavailable = std::any_of(a.begin(), a.end(),
                                   [](const auto &v) { return v.unavailable; });
    if (!unavailable) {
      try {
        auto p = ops::prepare_geometry(spec, native.data(), native.size());
        if (!p.output_geometry_known)
          throw Error("UNKNOWN_BOUND: symbolic operator geometry");
        out.shape = p.output_shape;
        out.kind = p.output_kind;
        for (N i = a.size(); i < spec.max_args; ++i)
          a.push_back(literal(p.args[i].scalar));
      } catch (const ops::Error &e) {
        const std::string code = e.what();
        if (!graph_mode ||
            (code != "INVALID_PARAMETER" && code != "INSUFFICIENT_SAMPLE"))
          throw;
        for (auto &v : a)
          if (!v.shape.rank) {
            if (!v.scalar)
              throw Error(
                  "UNKNOWN_BOUND: symbolic unavailable scalar geometry");
            c.guards = add(c.guards, 1);
          }
        unavailable = true;
      }
    }
    if (unavailable) {
      std::uint32_t geometry = 0;
      for (N i = 0; i < a.size(); ++i)
        if (a[i].shape.rank >= 0)
          geometry |= 1u << i;
      auto p = ops::validate_structure(spec, native.data(), native.size(),
                                       geometry, 0);
      out.shape = p.output_shape;
      out.kind = p.output_kind;
      if (!p.geometry_known)
        out.shape.rank = -1;
      out.unavailable = true;
    }
    const N o = out.size();
    N recipe = 0;
    if (!unavailable) {
      if (spec.family == ops::Family::elementwise || op == Op::active_returns) {
        recipe = mul(o, op == Op::normal_ppf ? 23 : op == Op::clip ? 2 : 1);
        if (!out.shape.rank && given[0].scalar &&
            (given.size() == 1 || given[1].scalar) &&
            out.kind == ops::Kind::number) {
          try {
            out.scalar = ops::scalar_math(
                op, *given[0].scalar, given.size() > 1 ? *given[1].scalar : 0,
                given.size() > 2 && given[2].scalar ? *given[2].scalar : 0);
          } catch (const ops::Error &) {
          }
        }
      } else if (op >= Op::sum_time && op <= Op::max_asset) {
        const Op map[]{Op::sum, Op::mean,      Op::product,  Op::variance,
                       Op::std, Op::min_value, Op::max_value};
        const bool time = op <= Op::max_time;
        recipe =
            mul(o, reduction(map[unsigned(op) -
                                 unsigned(time ? Op::sum_time : Op::sum_asset)],
                             a[0].shape.dim[time ? 0 : 1]));
      } else if (spec.family == ops::Family::reduction) {
        if (op == Op::count_true || op == Op::max_consecutive_true)
          recipe = add(mul(2, n), 1);
        else if (op == Op::distinct_count)
          recipe = add(add(mul(n, all_cells(n)), n), add(sum_cells(n), 1));
        else
          recipe = reduction(op, n);
      } else if (op == Op::first || op == Op::last || op == Op::length) {
        recipe = 1;
        if (op == Op::length)
          out.scalar = double(n);
      } else if (op == Op::lag || op == Op::difference ||
                 op == Op::aligned_shift) {
        bound(a[1], c, "shift periods");
        recipe = o;
      } else if (op == Op::argsort)
        recipe = mul(n, add(mul(2, n), sum_cells(n)));
      else if (op == Op::gather || op == Op::value_at)
        recipe = mul(o, std::max<N>(n, 1));
      else if (op == Op::cumulative_sum || op == Op::cumulative_product ||
               op == Op::cumulative_return || op == Op::cumulative_max ||
               op == Op::cumulative_min || op == Op::drawdown_series ||
               op == Op::new_high_mask)
        recipe = mul(2, n);
      else if (op == Op::total_return || op == Op::annualized_return)
        recipe = add(n, 1);
      else if (op == Op::rolling_mean || op == Op::rolling_std ||
               op == Op::rolling_min || op == Op::rolling_max) {
        auto w = bound(a[1], c, "rolling window");
        if (!w)
          throw Error("INVALID_INPUT: zero window");
        recipe = mul(n, op == Op::rolling_mean  ? 4
                        : op == Op::rolling_std ? 5
                                                : add(4, std::min(n, w)));
      } else if (op == Op::transpose)
        recipe = 0;
      else if (op == Op::diag)
        recipe = a[0].shape.rank == 1 ? o : 0;
      else if (op == Op::trace)
        recipe =
            add(sum_cells(std::min(a[0].shape.dim[0], a[0].shape.dim[1])), 1);
      else if (op == Op::outer)
        recipe = o;
      else if (op == Op::dot)
        recipe = n;
      else if (op == Op::quadratic_form)
        recipe = add(mul(n, n), n);
      else if (op == Op::matmul || op == Op::matvec ||
               op == Op::portfolio_returns)
        recipe = mul(o, a[0].shape.dim[1]);
      else if (op == Op::solve) {
        auto m = a[1].size();
        // Pivot search, all row swaps/selection, elimination, back
        // substitution.
        recipe = add(product({3, m, m, m}), add(mul(8, mul(m, m)), mul(4, m)));
      } else if (op == Op::covariance || op == Op::correlation) {
        if (given.size() == 2)
          recipe = add(mul(op == Op::correlation ? 17 : 9, n),
                       op == Op::correlation ? 6 : 3);
        else {
          auto rows = a[0].shape.dim[0], cols = a[0].shape.dim[1];
          recipe = add(mul(triangular(cols), add(mul(9, rows), 3)), o);
        }
      } else if (op == Op::linear_fit || (op >= Op::linear_slope &&
                                          op <= Op::regression_standard_error))
        recipe = add(mul(6, n), add(mul(2, sum_cells(n)), 5));
      else if ((op >= Op::fit_slope && op <= Op::fit_observation_count) ||
               (op >= Op::interval_start && op <= Op::interval_recovery) ||
               op == Op::state_estimate || op == Op::state_variance ||
               op == Op::continuous_state_values ||
               op == Op::continuous_state_evidence ||
               op == Op::continuous_state_pending || op == Op::segment_starts ||
               op == Op::segment_ends)
        recipe = 0;
      else if (op == Op::days_between || op == Op::require_positive ||
               op == Op::require_nonnegative || op == Op::state_select)
        recipe = o;
      else if (op == Op::last_drawdown_interval)
        recipe = add(mul(7, n), 4);
      else if (op == Op::recursive_smooth)
        recipe = mul(2, n);
      else if (op == Op::recursive_filter)
        recipe = mul(4, n);
      else if (op == Op::recursive_filter_adaptive)
        recipe = mul(8, n);
      else if (op == Op::linear_filter2 || op == Op::state_confirm)
        recipe = mul(11, n);
      else if (op == Op::scalar_kalman)
        recipe = mul(7, n);
      else if (op == Op::state_hysteresis)
        recipe = mul(2, n);
      else if (op == Op::state_continuous)
        recipe = mul(14, n);
      else if (op == Op::drawdown_cycle_state)
        recipe = mul(6, n);
      else if (op == Op::between_events)
        recipe = mul(n, add(mul(2, n), 3));
      else if (op == Op::phase_direction)
        recipe = mul(n, add(mul(2, n), 1));
      else if (op == Op::drawdown_cycle_reference)
        recipe = add(mul(5, n), triangular(n));
      else if (op == Op::local_extrema) {
        auto l = bound(a[1], c, "extrema left", 5000),
             r = bound(a[2], c, "extrema right", 5000);
        bound(a[3], c, "extrema head", 5000);
        bound(a[4], c, "extrema tail", 5000);
        recipe = add(mul(n, add(add(8, std::max<N>(n, 1)),
                                mul(2, all_cells(std::min(n, add(l, r)))))),
                     n ? triangular(n - 1) : 0);
      } else if (op == Op::ps_filter) {
        const auto select = std::max<N>(n, 1), triangle = triangular(n);
        const auto alternate =
            add(mul(n, add(select, 4)), n ? triangular(n - 1) : 0);
        auto run = add(mul(n, add(2, mul(4, select))),
                       add(mul(add(13, mul(6, select)), triangle),
                           mul(2, add(triangle, mul(n, sum_cells(n))))));
        recipe =
            add(add(mul(3, n), alternate), mul(n, add(run, add(n, alternate))));
      } else
        throw Error(std::string("UNSUPPORTED_OPERATOR: symbolic budget ") +
                    spec.name);
    }
    c.steps(recipe);
    const bool aligned =
        spec.family == ops::Family::elementwise || op == Op::state_select;
    const bool mapped = op == Op::lag || op == Op::difference ||
                        op == Op::aligned_shift || op == Op::first ||
                        op == Op::last || op == Op::length;
    const bool positions = std::any_of(
        given.begin(), given.end(), [](const auto &v) { return v.positions; });
    if (graph_mode && isolate)
      c.steps(
          mul(o, (!out.shape.rank && out.kind == ops::Kind::number) ? 2 : 1));
    if (!((aligned || mapped) && positions))
      c.steps(o);
    c.steps(add(given.size(), 1));
    if (!aligned && !mapped)
      for (auto &v : given)
        c.steps(v.size());
    c.steps(o);
    if (aligned)
      c.steps(mul(o, given.size()));
    if (mapped && op != Op::length && positions)
      c.steps(mul(o, 2));
    out.positions = positions && (aligned || mapped);
    check(c);
    return out;
  }
  Result program(const graph::Program &, const std::vector<Value> &,
                 const std::vector<Value> &, N, unsigned = 0);
};
Result Analyzer::program(const graph::Program &p,
                         const std::vector<Value> &inputs,
                         const std::vector<Value> &parameters, N extent,
                         unsigned depth) {
  if (depth > 32)
    throw Error("OVER_BUDGET: symbolic scope depth");
  Result result;
  auto &cost = result.cost;
  cost.work = mul(p.nodes.size(), add(extent, 1));
  check(cost);
  std::vector<Value> nodes;
  nodes.reserve(p.nodes.size());
  for (N id = 0; id < p.nodes.size(); ++id) {
    const auto before = cost.cells;
    const auto &node = p.nodes[id];
    Value value;
    if (node.kind == graph::NodeKind::input)
      value = inputs.at(node.input_index);
    else if (node.kind == graph::NodeKind::parameter)
      value = parameters.at(node.input_index);
    else if (node.kind == graph::NodeKind::constant)
      value = literal(node.constant);
    else if (node.kind == graph::NodeKind::interval_tail) {
      auto v = nodes.at(node.parents[0]);
      value = slice(v, v.size() ? v.size() - 1 : 0);
    } else if (node.kind == graph::NodeKind::iteration_projection) {
      if (node.opcode != 2)
        value.kind = ops::Kind::integer;
    } else if (node.kind == graph::NodeKind::operation) {
      std::vector<Value> args;
      for (N j = 0; j < node.parent_count; ++j)
        args.push_back(nodes.at(node.parents[j]));
      value = operation(static_cast<Op>(node.opcode), args, cost, true,
                        p.isolate_errors);
    } else if (node.kind == graph::NodeKind::rolling_scope) {
      const auto &scope = p.rolling_scopes.at(node.input_index);
      auto width =
          bound(nodes.at(scope.width_node), cost, "rolling scope width");
      auto minimum =
          scope.has_min_periods
              ? bound(nodes.at(scope.min_periods_node), cost, "rolling minimum")
              : width;
      if (!width || !minimum || minimum > width)
        throw Error("INVALID_INPUT: rolling minimum/width");
      value.shape = ops::vector_shape(extent);
      value.positions = p.isolate_errors;
      const N available = scope.needs_preceding_observation
                              ? (extent ? extent - 1 : 0)
                              : extent;
      const N max = std::min(available, width);
      const N begin = scope.has_min_periods ? 1 : width;
      // Visit each distinct window geometry once. Full windows share a cost,
      // not state or workbook cells; multiply their independent instances.
      for (N count = begin; count <= max; ++count) {
        Cost local;
        std::vector<Value> in, params;
        for (N j = 0; j < scope.input_nodes.size(); ++j)
          in.push_back(slice(nodes.at(scope.input_nodes[j]),
                             add(count, j < scope.input_preceding.size() &&
                                                scope.input_preceding[j]
                                            ? 1
                                            : 0)));
        local.steps(
            add(mul(count, add(all_cells(in.size()), 1)), sum_cells(count)));
        if (scope.has_date_context)
          local.steps(1);
        for (auto &binding : scope.parameter_bindings) {
          if (binding.kind == graph::RollingParameterKind::outer_node)
            params.push_back(nodes.at(binding.node));
          else {
            params.push_back(Value{});
            if (binding.kind ==
                graph::RollingParameterKind::risk_free_return_window)
              local.steps(1);
            else if (binding.kind !=
                         graph::RollingParameterKind::window_elapsed_days &&
                     scope.has_returns && scope.has_min_periods &&
                     scope.returns_input >= 0)
              local.steps(all_cells(count));
          }
        }
        auto body = program(*scope.body, in, params, count, depth + 1);
        local.include(body.cost);
        local.steps(1);
        cost.include(local, count == width ? available - width + 1 : 1);
        check(cost);
      }
    } else if (node.kind == graph::NodeKind::apply_scope) {
      const auto &s = p.apply_scopes.at(node.input_index);
      std::vector<Value> captured, params;
      for (auto source : s.input_nodes)
        captured.push_back(nodes.at(source));
      for (auto source : s.parameter_nodes)
        params.push_back(source == UINT32_MAX ? Value{} : nodes.at(source));
      auto arg = [&](N i) -> const Value & {
        return nodes.at(s.argument_nodes.at(i));
      };
      auto eval = [&](const std::vector<Value> &in, N count) {
        return program(*s.body, in, params, count, depth + 1);
      };
      N count = captured.empty() ? extent : captured[0].size();
      if (s.kind == graph::ApplyKind::iterate) {
        value = arg(0);
        auto limit = bound(arg(2), cost, "iteration limit", 10000);
        if (!limit)
          throw Error("INVALID_INPUT: iteration limit");
        cost.steps(add(all_cells(value.size()), 2));
        // State may change on every iteration. Never propagate the seed as a
        // compile-time layout parameter or use observed early convergence.
        if (s.state_input_index >= 0) {
          captured.at(s.state_input_index) = value;
          captured.at(s.state_input_index).scalar.reset();
        }
        auto body = eval(captured, count);
        if (!(body.roots.at(0).shape == value.shape))
          throw Error("INVALID_INPUT: iteration state shape");
        body.cost.steps(
            add(add(mul(2, value.size()), all_cells(value.size())), 6));
        cost.include(body.cost, limit);
        value.scalar.reset();
      } else if (s.kind == graph::ApplyKind::bisect) {
        auto limit = bound(arg(3), cost, "bisection limit", 10000);
        if (!limit)
          throw Error("INVALID_INPUT: bisection limit");
        auto body = eval(captured, count);
        cost.include(body.cost, add(limit, 2));
        cost.steps(add(mul(9, limit), 3));
      } else if (s.kind == graph::ApplyKind::block) {
        auto width = bound(arg(0), cost, "block width");
        if (!width)
          throw Error("INVALID_INPUT: block width");
        value.shape = ops::vector_shape(count / width);
        if (count / width) {
          std::vector<Value> local;
          for (auto &v : captured)
            local.push_back(slice(v, width));
          cost.include(eval(local, width).cost, count / width);
        }
      } else if (s.kind == graph::ApplyKind::segment) {
        count = arg(0).shape.dim[0];
        value.shape = ops::vector_shape(count);
        // Bounds on all possible intervals; endpoints are never evaluated here.
        for (N length = 2; length <= count; ++length) {
          std::vector<Value> local;
          for (auto &v : captured)
            local.push_back(slice(v, length));
          auto body = eval(local, length);
          body.cost.steps(length - 1);
          cost.include(body.cost, count - length + 1);
          check(cost);
        }
      } else {
        const bool group = s.kind == graph::ApplyKind::group;
        if (!group && s.kind != graph::ApplyKind::filter)
          throw Error("UNSUPPORTED_OPERATOR: symbolic apply kind");
        count = arg(0).size();
        value.shape = group ? ops::vector_shape(count) : ops::Shape{};
        const N repeats = group ? count : 1;
        Cost per;
        per.steps(add(mul(2, count), product({captured.size(), count, count})));
        Cost lower;
        lower.include(per, repeats);
        check(lower);
        for (N length = 1; length <= count && repeats; ++length) {
          std::vector<Value> local;
          for (auto &v : captured)
            local.push_back(slice(v, length));
          per.include(eval(local, length).cost);
          per.steps(1);
          Cost current;
          current.include(per, repeats);
          check(current);
        }
        cost.include(per, repeats);
      }
      if (s.kind != graph::ApplyKind::iterate &&
          s.kind != graph::ApplyKind::bisect)
        value.positions = p.isolate_errors;
    } else
      throw Error("UNSUPPORTED_OPERATOR: symbolic graph node kind");
    if (p.isolate_errors && (node.kind == graph::NodeKind::apply_scope ||
                             node.kind == graph::NodeKind::rolling_scope)) {
      value.positions =
          p.execution_metadata->position_status_reachable.at(id) != 0;
      if (!value.positions)
        cost.steps(mul(2, value.size()));
    }
    cost.text(8); // Seven provenance columns plus at most one step label.
    check(cost);
    if (depth == 0)
      largest.push_back({id,
                         node.kind == graph::NodeKind::operation
                             ? ops::lookup(node.opcode).name
                             : compiler::kind_name(node.kind),
                         cost.cells - before});
    nodes.push_back(value);
  }
  for (auto root : p.roots)
    result.roots.push_back(nodes.at(root));
  return result;
}
} // namespace
Budget symbolic_budget(const compiler::CompiledGraph *graph,
                       std::optional<Op> op,
                       const std::vector<Snapshot> &inputs,
                       const std::vector<double> &parameters, Limits limits,
                       const std::function<void()> &checkpoint) {
  Analyzer a{limits, checkpoint};
  Budget report;
  Cost cost;
  std::vector<Value> in, params;
  N items = parameters.size(), bytes = mul(parameters.size(), 8);
  for (auto &s : inputs) {
    const auto &v = s.value;
    Value x;
    x.shape = v.shape;
    x.kind = v.kind;
    if (!v.shape.rank && v.kind == ops::Kind::number)
      x.scalar = v.scalar;
    items = add(items, x.size());
    bytes = add(bytes, mul(x.size(), v.kind == ops::Kind::mask ? 1 : 8));
    in.push_back(x);
  }
  if (bytes > limits.snapshot_bytes)
    throw Error("OVER_BUDGET: symbolic snapshot bytes");
  for (auto p : parameters)
    params.push_back(literal(p));
  // Four input columns, three possible formulas per input (two exact binary
  // constants + snapshot equality). Contents cannot enlarge this upper bound.
  cost.text(items);
  cost.steps(mul(3, items));
  cost.steps(all_cells(items));
  Result result;
  if (op) {
    result.roots.push_back(a.operation(*op, in, cost, false, false));
    cost.text(1);
  } else {
    if (!graph || in.size() != graph->program.input_count ||
        params.size() != graph->program.parameter_count)
      throw Error("INVALID_INPUT: symbolic graph binding count");
    N extent = 1;
    for (N i = 0; i < in.size(); ++i)
      if (in[i].shape.rank && graph->program.input_axes.at(i) != 1) {
        extent = in[i].shape.dim[0];
        break;
      }
    if (extent < graph->program.minimum_observations)
      throw Error("INVALID_INPUT: minimum observations unmet; native output "
                  "geometry is unknown");
    result = a.program(graph->program, in, params, extent);
    cost.include(result.cost);
    cost.text(graph->expressions.size());
  }
  cost.steps(all_cells(cost.guards));
  cost.text(19);
  cost.steps(3); // Readme, including reference state and tolerances.
  for (auto &v : result.roots) {
    if (v.shape.rank < 0)
      throw Error("UNKNOWN_BOUND: failed root has unknown geometry");
    cost.text(add(4, mul(2, v.size())));
    cost.steps(mul(3, v.size()));
  }
  a.check(cost);
  report.cells = cost.cells;
  report.formula_cells = cost.formulas;
  report.formula_characters = mul(cost.formulas, 8192);
  report.work_units = cost.work;
  report.snapshot_bytes = bytes;
  std::sort(a.largest.begin(), a.largest.end(),
            [](const auto &x, const auto &y) { return x.cells > y.cells; });
  if (a.largest.size() > 10)
    a.largest.resize(10);
  report.largest_nodes = std::move(a.largest);
  return report;
}
} // namespace calmetrics_engine::excel
