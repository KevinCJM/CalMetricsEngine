#include "excel_internal.hpp"

namespace calmetrics_engine::excel {
std::vector<Array> Builder::program(const graph::Program &p,
                                    const std::vector<Array> &inputs,
                                    const std::vector<Array> &parameters,
                                    std::size_t extent, unsigned depth) {
  if (depth > 32)
    throw Error("OVER_BUDGET: scope depth");
  struct Restore {
    bool &target;
    bool previous;
    ~Restore() { target = previous; }
  } restore{isolate, isolate};
  isolate = p.isolate_errors;
  cost({p.nodes.size(), extent + 1});
  const auto context = std::to_string(invocation++);
  std::vector<Array> nodes;
  nodes.reserve(p.nodes.size());
  for (std::size_t id = 0; id < p.nodes.size(); ++id) {
    const auto &node = p.nodes[id];
    Array value;
    label = "context " + context + " node " + std::to_string(id) + " depth " +
            std::to_string(depth) + " " + compiler::kind_name(node.kind);
    if (node.kind == graph::NodeKind::input)
      value = inputs.at(node.input_index);
    else if (node.kind == graph::NodeKind::parameter)
      value = parameters.at(node.input_index);
    else if (node.kind == graph::NodeKind::constant)
      value = literal(node.constant);
    else if (node.kind == graph::NodeKind::interval_tail) {
      auto &a = nodes.at(node.parents[0]);
      value = slice(a, a.size() ? 1 : 0, a.size() ? a.size() - 1 : 0);
    } else if (node.kind == graph::NodeKind::iteration_projection) {
      value = literal(0);
      value.scalar.reset();
      value.refs = {nodes.at(node.parents[0]).diagnostics.at(node.opcode)};
      if (node.opcode != 2)
        value.kind = ops::Kind::integer;
    } else if (node.kind == graph::NodeKind::operation) {
      std::vector<Array> args;
      for (std::size_t j = 0; j < node.parent_count; ++j)
        args.push_back(nodes.at(node.parents[j]));
      label += " " + F(ops::lookup(node.opcode).name);
      value = operation(static_cast<Op>(node.opcode), args);
    } else if (node.kind == graph::NodeKind::rolling_scope) {
      const auto &scope = p.rolling_scopes.at(node.input_index);
      auto width = bound(nodes.at(scope.width_node), "rolling scope width");
      if (!width)
        throw Error("INVALID_INPUT: rolling width");
      auto minimum =
          scope.has_min_periods
              ? bound(nodes.at(scope.min_periods_node), "rolling minimum")
              : width;
      if (!minimum || minimum > width)
        throw Error("INVALID_INPUT: rolling minimum");
      value = array(ops::vector_shape(extent));
      value.position_errors = p.isolate_errors;
      for (std::size_t right = 0; right < extent; ++right) {
        auto left = right + 1 > width ? right + 1 - width : 0;
        if (scope.has_min_periods && scope.needs_preceding_observation &&
            left == 0 && right > 0)
          left = 1;
        auto count = right + 1 - left;
        if ((!scope.has_min_periods && count != width) ||
            (scope.needs_preceding_observation && left == 0)) {
          value.refs[right] = q("NaN");
          continue;
        }
        std::vector<Array> local, params;
        std::vector<F> valid;
        for (std::size_t t = left; t <= right; ++t) {
          std::vector<F> row;
          for (auto source : scope.input_nodes)
            row.push_back(fin(nodes.at(source).at(t)));
          valid.push_back(formula("IF(" + all(row) + ",1,0)"));
        }
        F eligible = sum(valid);
        for (std::size_t j = 0; j < scope.input_nodes.size(); ++j) {
          auto &a = nodes.at(scope.input_nodes[j]);
          bool preceding =
              j < scope.input_preceding.size() && scope.input_preceding[j];
          local.push_back(slice(a, preceding ? left - 1 : left,
                                count + (preceding ? 1 : 0)));
        }
        F elapsed =
            scope.has_date_context
                ? formula(nodes.at(scope.dates_node).at(right) + "-" +
                          nodes.at(scope.dates_node)
                              .at(scope.needs_preceding_observation ? left - 1
                                                                    : left))
                : std::to_string(count ? count - 1 : 0);
        for (auto &binding : scope.parameter_bindings) {
          if (binding.kind == graph::RollingParameterKind::outer_node)
            params.push_back(nodes.at(binding.node));
          else {
            Array a = literal(0);
            a.scalar.reset();
            if (binding.kind ==
                graph::RollingParameterKind::window_elapsed_days)
              a.refs = {elapsed};
            else if (binding.kind ==
                     graph::RollingParameterKind::risk_free_return_window)
              a.refs = {formula("MAX(0,1+" +
                                nodes.at(scope.annual_rate_node).at() + ")^(" +
                                elapsed + "/365)-1")};
            else {
              F observations = std::to_string(scope.has_returns ? count
                                              : count           ? count - 1
                                                                : 0);
              if (scope.has_returns && scope.has_min_periods &&
                  scope.returns_input >= 0) {
                std::vector<F> counts;
                auto &src = nodes.at(scope.input_nodes.at(scope.returns_input));
                for (std::size_t t = left; t <= right; ++t)
                  counts.push_back(formula("IF(" + fin(src.at(t)) + ",1,0)"));
                observations = sum(counts);
              }
              a.refs = {observations};
            }
            params.push_back(a);
          }
        }
        auto result =
            program(*scope.body, local, params, count, depth + 1).at(0).at();
        value.refs[right] = formula(
            iff(eligible + ">=" + std::to_string(minimum), result, q("NaN")));
      }
    } else if (node.kind == graph::NodeKind::apply_scope) {
      const auto &scope = p.apply_scopes.at(node.input_index);
      std::vector<Array> captured, params;
      for (auto source : scope.input_nodes)
        captured.push_back(nodes.at(source));
      for (auto source : scope.parameter_nodes)
        params.push_back(source == UINT32_MAX ? literal(0) : nodes.at(source));
      auto arg = [&](std::size_t i) -> const Array & {
        return nodes.at(scope.argument_nodes.at(i));
      };
      auto eval = [&](const std::vector<Array> &in, std::size_t length,
                      const Array &solver = Array{}) {
        auto ps = params;
        for (std::size_t i = 0; i < scope.parameter_nodes.size(); ++i)
          if (scope.parameter_nodes[i] == UINT32_MAX)
            ps[i] = solver;
        return program(*scope.body, in, ps, length, depth + 1).at(0);
      };
      std::size_t count = captured.empty() ? extent : captured[0].size();
      if (scope.kind == graph::ApplyKind::iterate) {
        value = arg(0);
        auto limit = bound(arg(2), "iteration limit", 10000);
        if (!limit)
          throw Error("INVALID_INPUT: iteration limit");
        F tolerance = arg(1).at(), status = "1", iterations = "0",
          residual = q("+Inf");
        std::vector<F> finite;
        for (auto &x : value.refs)
          finite.push_back(fin(x));
        auto running = formula(all(finite));
        status = formula(iff(running, "1", "5"));
        for (std::size_t j = 0; j < limit; ++j) {
          if (scope.state_input_index >= 0)
            captured.at(scope.state_input_index) = value;
          auto next = eval(captured, count);
          if (!(next.shape == value.shape))
            throw Error("INVALID_INPUT: iteration state shape");
          std::vector<F> checks;
          F delta = "0";
          for (std::size_t k = 0; k < next.size(); ++k) {
            checks.push_back(fin(next.refs[k]));
            delta = formula("IFERROR(MAX(" + delta + ",ABS(" + next.refs[k] +
                            "-" + value.refs[k] + ")),\"+Inf\")");
          }
          auto good = formula(all(checks));
          auto converged =
              formula("AND(" + good + "," + delta + "<=" + tolerance + ")");
          for (std::size_t k = 0; k < value.size(); ++k)
            value.refs[k] = formula(iff("AND(" + running + "," + good + ")",
                                        next.refs[k], value.refs[k]));
          iterations = formula(iff(running, std::to_string(j + 1), iterations));
          residual =
              formula(iff(running, iff(good, delta, q("+Inf")), residual));
          status = formula(
              iff(running, iff(good, iff(converged, "0", "1"), "5"), status));
          running = formula("AND(" + running + "," + good + ",NOT(" +
                            converged + "))");
        }
        value.scalar.reset();
        value.diagnostics = {status, iterations, residual};
      } else if (scope.kind == graph::ApplyKind::bisect) {
        auto limit = bound(arg(3), "bisection limit", 10000);
        if (!limit)
          throw Error("INVALID_INPUT: bisection limit");
        F low = arg(0).at(), high = arg(1).at(), tol = arg(2).at();
        auto scalar = [&](const F &x) {
          auto s = literal(0);
          s.scalar.reset();
          s.refs = {x};
          return s;
        };
        F fl = eval(captured, count, scalar(low)).at(),
          fh = eval(captured, count, scalar(high)).at();
        F bad = formula("NOT(AND(" + fin(fl) + "," + fin(fh) + "," + low + "<" +
                        high + "," + tol + ">0))");
        F answer = formula(
            iff(bad, err("NONFINITE_RESULT"),
                iff(fl + "=0", low,
                    iff(fh + "=0", high,
                        iff("SIGN(" + fl + ")=SIGN(" + fh + ")",
                            err("ROOT_NOT_BRACKETED"), q("PENDING"))))));
        for (std::size_t j = 0; j < limit; ++j) {
          auto mid = formula(low + "*0.5+" + high + "*0.5"),
               stop = formula("OR(" + high + "-" + low + "<=" + tol + "," +
                              mid + "=" + low + "," + mid + "=" + high + ")");
          auto fm = eval(captured, count, scalar(mid)).at();
          auto active = formula(answer + "=" + q("PENDING"));
          answer = formula(iff(active,
                               iff(stop, mid,
                                   iff(fin(fm), iff(fm + "=0", mid, answer),
                                       err("NONFINITE_RESULT"))),
                               answer));
          auto change = formula("AND(" + active + ",NOT(" + stop + ")," +
                                fin(fm) + "," + fm + "<>0)"),
               left = formula("SIGN(" + fm + ")=SIGN(" + fl + ")");
          low = formula(iff("AND(" + change + "," + left + ")", mid, low));
          high =
              formula(iff("AND(" + change + ",NOT(" + left + "))", mid, high));
          fl = formula(iff("AND(" + change + "," + left + ")", fm, fl));
        }
        value = scalar(formula(
            iff(answer + "=" + q("PENDING"),
                iff(high + "-" + low + "<=" + tol,
                    low + "*0.5+" + high + "*0.5", err("NON_CONVERGENCE")),
                answer)));
      } else if (scope.kind == graph::ApplyKind::block) {
        auto width = bound(arg(0), "block width");
        if (!width)
          throw Error("INVALID_INPUT: block width");
        value = array(ops::vector_shape(count / width));
        for (std::size_t i = 0; i < value.size(); ++i) {
          std::vector<Array> local;
          for (auto &a : captured)
            local.push_back(slice(a, i * width, width));
          value.refs[i] = eval(local, width).at();
        }
      } else if (scope.kind == graph::ApplyKind::segment) {
        count = arg(0).shape.dim[0];
        value = array(ops::vector_shape(count));
        std::fill(value.refs.begin(), value.refs.end(), q("NaN"));
        for (std::size_t l = 0; l < count; ++l)
          for (std::size_t r = l + 1; r < count; ++r) {
            std::vector<Array> local;
            for (auto &a : captured)
              local.push_back(slice(a, l, r - l + 1));
            auto result = eval(local, r - l + 1).at();
            for (std::size_t t = l; t < r; ++t)
              value.refs[t] = formula(iff("AND(" + arg(0).refs.at(2 * t) + "=" +
                                              std::to_string(l) + "," +
                                              arg(0).refs.at(2 * t + 1) + "=" +
                                              std::to_string(r) + ")",
                                          result, value.refs[t]));
          }
      } else {
        // Compact dynamically, then expand every possible cardinality. This
        // preserves shape-changing filter/group edits without freezing choices.
        const bool group = scope.kind == graph::ApplyKind::group;
        count = arg(0).size();
        value = array(group ? ops::vector_shape(count) : ops::Shape{});
        for (std::size_t target = 0; target < (group ? count : 1); ++target) {
          std::vector<F> selected, positions;
          F length = "0";
          for (std::size_t i = 0; i < count; ++i) {
            auto take =
                group ? arg(0).at(i) + "=" + arg(0).at(target) : arg(0).at(i);
            selected.push_back(formula(take));
            positions.push_back(length);
            length = formula(length + "+IF(" + selected.back() + ",1,0)");
          }
          std::vector<Array> compact;
          for (auto &a : captured) {
            auto v = array(ops::vector_shape(count), a.kind);
            for (std::size_t j = 0; j < count; ++j) {
              F item = q("NaN");
              for (std::size_t i = 0; i < count; ++i)
                item = formula(iff("AND(" + selected[i] + "," + positions[i] +
                                       "=" + std::to_string(j) + ")",
                                   a.at(i), item));
              v.refs[j] = item;
            }
            compact.push_back(v);
          }
          F result = !group && scope.argument_nodes.size() == 2 ? arg(1).at()
                                                                : q("NaN");
          for (std::size_t size = 1; size <= count; ++size) {
            std::vector<Array> local;
            for (auto &a : compact)
              local.push_back(slice(a, 0, size));
            auto candidate = eval(local, size).at();
            result = formula(
                iff(length + "=" + std::to_string(size), candidate, result));
          }
          value.refs[target] = result;
        }
      }
      if (scope.kind != graph::ApplyKind::iterate &&
          scope.kind != graph::ApplyKind::bisect)
        value.position_errors = p.isolate_errors;
    } else
      throw Error("UNSUPPORTED_OPERATOR: graph node kind");
    if (p.isolate_errors && (node.kind == graph::NodeKind::apply_scope ||
                             node.kind == graph::NodeKind::rolling_scope)) {
      value.position_errors =
          p.execution_metadata->position_status_reachable.at(id) != 0;
      if (!value.position_errors) {
        F failure = "\"\"";
        for (auto &x : value.refs)
          failure = formula(
              iff("LEFT(" + x + ",6)=\"ERROR:\"", err("STATUS:4"), failure));
        value.failure = failure;
        for (auto &x : value.refs)
          x = formula(iff(failure + "<>\"\"", failure, x));
      }
    }
    // Logical provenance includes pure aliases and record projections, even
    // when they need no additional mathematical cell.
    const auto row = index++;
    const auto sheet =
        row < 1000000 ? "Nodes" : "Nodes" + std::to_string(row / 1000000);
    const auto local = row % 1000000;
    put(sheet, local, 0, "text", context + ":" + std::to_string(id));
    put(sheet, local, 1, "text",
        node.kind == graph::NodeKind::operation
            ? ops::lookup(node.opcode).name
            : compiler::kind_name(node.kind));
    put(sheet, local, 2, "text",
        value.refs.empty() ? "empty" : value.refs.front());
    F geometry = std::to_string(value.shape.rank) + ":";
    for (int j = 0; j < value.shape.rank; ++j)
      geometry += std::to_string(value.shape.dim[j]) + ",";
    put(sheet, local, 3, "text", geometry);
    F parents;
    for (std::size_t j = 0; j < node.parent_count; ++j) {
      if (j)
        parents += ",";
      parents += context + ":" + std::to_string(node.parents[j]);
    }
    put(sheet, local, 4, "text", parents);
    put(sheet, local, 5, "text",
        value.kind == ops::Kind::integer    ? "int64"
        : value.kind == ops::Kind::mask     ? "bool"
        : value.kind == ops::Kind::fit      ? "fit"
        : value.kind == ops::Kind::interval ? "interval"
                                            : "float64");
    put(sheet, local, 6, "text",
        value.refs.empty() ? "empty" : value.refs.back());
    nodes.push_back(std::move(value));
  }
  std::vector<Array> result;
  for (auto root : p.roots)
    result.push_back(nodes.at(root));
  return result;
}
} // namespace calmetrics_engine::excel
