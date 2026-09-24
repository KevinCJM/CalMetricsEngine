#include "excel_internal.hpp"

namespace calmetrics_engine::excel {
namespace {
F nan() { return q("NaN"); }
} // namespace
Array Builder::state_operation(Op op, const std::vector<Array> &a, Array out) {
  auto A = [&](std::size_t j, std::size_t i = 0) { return a.at(j).at(i); };
  auto cell = [&](const F &f) {
    return formula("IFERROR(" + f + "," + err("DOMAIN_ERROR") + ")");
  };
  auto store = [&](std::size_t i, const F &f) { out.refs.at(i) = cell(f); };
  const auto n = a[0].size();
  if (op == Op::transpose || op == Op::diag || op == Op::trace) {
    if (op == Op::transpose) {
      auto rows = a[0].shape.dim[0], cols = a[0].shape.dim[1];
      for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
          out.refs[j * rows + i] = A(0, i * cols + j);
    } else if (op == Op::diag && a[0].shape.rank == 1) {
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          store(i * n + j, i == j ? A(0, i) : "0");
    } else {
      std::vector<F> d;
      for (std::size_t i = 0;
           i < std::min(a[0].shape.dim[0], a[0].shape.dim[1]); ++i)
        d.push_back(A(0, i * (a[0].shape.dim[1] + 1)));
      if (op == Op::diag)
        out.refs = d;
      else
        store(0, sum(d));
    }
    return out;
  }
  if (op == Op::dot || op == Op::outer || op == Op::matmul ||
      op == Op::matvec || op == Op::portfolio_returns ||
      op == Op::quadratic_form) {
    if (op == Op::outer) {
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < a[1].size(); ++j)
          store(i * a[1].size() + j, A(0, i) + "*" + A(1, j));
      return out;
    }
    auto dot = [&](const std::vector<F> &x, const std::vector<F> &y) {
      F s = "0";
      for (std::size_t k = 0; k < x.size(); ++k)
        s = cell(s + "+" + x[k] + "*" + y[k]);
      return s;
    };
    if (op == Op::dot) {
      out.refs[0] = dot(a[0].refs, a[1].refs);
      return out;
    }
    if (op == Op::quadratic_form) {
      std::vector<F> v;
      for (std::size_t i = 0; i < n; ++i) {
        std::vector<F> col;
        for (std::size_t j = 0; j < n; ++j)
          col.push_back(A(1, j * n + i));
        v.push_back(dot(col, a[0].refs));
      }
      out.refs[0] = dot(v, a[0].refs);
      return out;
    }
    auto rows = a[0].shape.dim[0], inner = a[0].shape.dim[1],
         cols = op == Op::matmul ? a[1].shape.dim[1] : 1;
    for (std::size_t i = 0; i < rows; ++i)
      for (std::size_t j = 0; j < cols; ++j) {
        std::vector<F> x, y;
        for (std::size_t k = 0; k < inner; ++k) {
          x.push_back(A(0, i * inner + k));
          y.push_back(A(1, k * cols + j));
        }
        out.refs[i * cols + j] = dot(x, y);
      }
    return out;
  }
  if (op == Op::solve) {
    const auto size = a[1].size();
    auto m = a[0].refs, rhs = a[1].refs;
    F status = "FALSE";
    for (std::size_t k = 0; k < size; ++k) {
      F pivot = std::to_string(k), mag = cell("ABS(" + m[k * size + k] + ")");
      for (std::size_t i = k + 1; i < size; ++i) {
        auto test = cell("ABS(" + m[i * size + k] + ")");
        pivot = cell(iff(test + ">" + mag, std::to_string(i), pivot));
        mag = cell("MAX(" + mag + "," + test + ")");
      }
      status = cell("OR(" + status + "," + mag + "<=1E-14)");
      auto old = m, oldrhs = rhs;
      for (std::size_t j = 0; j < size; ++j) {
        std::vector<F> col;
        for (std::size_t i = 0; i < size; ++i)
          col.push_back(old[i * size + j]);
        auto picked = choose(col, pivot);
        m[k * size + j] = picked;
        for (std::size_t i = k + 1; i < size; ++i)
          m[i * size + j] = cell(iff(pivot + "=" + std::to_string(i),
                                     old[k * size + j], old[i * size + j]));
      }
      rhs[k] = choose(oldrhs, pivot);
      for (std::size_t i = k + 1; i < size; ++i)
        rhs[i] =
            cell(iff(pivot + "=" + std::to_string(i), oldrhs[k], oldrhs[i]));
      for (std::size_t i = k + 1; i < size; ++i) {
        auto f = cell(m[i * size + k] + "/" + m[k * size + k]);
        m[i * size + k] = "0";
        for (std::size_t j = k + 1; j < size; ++j)
          m[i * size + j] =
              cell(m[i * size + j] + "-" + f + "*" + m[k * size + j]);
        rhs[i] = cell(rhs[i] + "-" + f + "*" + rhs[k]);
      }
    }
    for (std::size_t rev = 0; rev < size; ++rev) {
      auto i = size - 1 - rev;
      F v = rhs[i];
      for (std::size_t j = i + 1; j < size; ++j)
        v = cell(v + "-" + m[i * size + j] + "*" + out.refs[j]);
      store(i, iff(status, err("SINGULAR_MATRIX"), v + "/" + m[i * size + i]));
    }
    return out;
  }
  if (op == Op::covariance || op == Op::correlation) {
    auto covariance = [&](const std::vector<F> &x, const std::vector<F> &y) {
      auto mx = reduce(Op::mean, x), my = reduce(Op::mean, y);
      F total = "0";
      for (std::size_t i = 0; i < x.size(); ++i)
        total = cell(total + "+(" + x[i] + "-" + mx + ")*(" + y[i] + "-" + my +
                     ")");
      return cell(x.size() < 2 ? err("INSUFFICIENT_SAMPLE")
                               : total + "/" + std::to_string(x.size() - 1));
    };
    if (a.size() == 2) {
      auto cov = covariance(a[0].refs, a[1].refs);
      if (op == Op::correlation)
        cov = cell(cov + "/(" + reduce(Op::std, a[0].refs) + "*" +
                   reduce(Op::std, a[1].refs) + ")");
      out.refs[0] = cov;
    } else {
      auto rows = a[0].shape.dim[0], cols = a[0].shape.dim[1];
      std::vector<std::vector<F>> columns(cols);
      for (std::size_t j = 0; j < cols; ++j)
        for (std::size_t i = 0; i < rows; ++i)
          columns[j].push_back(A(0, i * cols + j));
      std::vector<F> cov(cols * cols);
      for (std::size_t j = 0; j < cols; ++j)
        for (std::size_t k = j; k < cols; ++k)
          cov[j * cols + k] = cov[k * cols + j] =
              covariance(columns[j], columns[k]);
      for (std::size_t j = 0; j < cols; ++j)
        for (std::size_t k = 0; k < cols; ++k)
          store(j * cols + k, op == Op::covariance
                                  ? cov[j * cols + k]
                                  : cov[j * cols + k] + "/(SQRT(" +
                                        cov[j * cols + j] + ")*SQRT(" +
                                        cov[k * cols + k] + "))");
    }
    return out;
  }
  if (op == Op::linear_fit ||
      (op >= Op::linear_slope && op <= Op::regression_standard_error)) {
    const auto &y = a.size() == 1 ? a[0].refs : a[1].refs;
    std::vector<F> x;
    if (a.size() == 1)
      for (std::size_t i = 0; i < n; ++i)
        x.push_back(std::to_string(i));
    else
      x = a[0].refs;
    auto mx = cell(sum(x) + "/" + std::to_string(n)),
         my = cell(sum(y) + "/" + std::to_string(n));
    F xx = "0", xy = "0", yy = "0";
    for (std::size_t i = 0; i < n; ++i) {
      auto dx = cell(x[i] + "-" + mx), dy = cell(y[i] + "-" + my);
      xx = cell(xx + "+" + dx + "^2");
      xy = cell(xy + "+" + dx + "*" + dy);
      yy = cell(yy + "+" + dy + "^2");
    }
    auto slope = cell(iff(xx + ">0", xy + "/" + xx, err("DOMAIN_ERROR"))),
         intercept = cell(my + "-" + slope + "*" + mx);
    F rss = "0";
    for (std::size_t i = 0; i < n; ++i)
      rss = cell(rss + "+(" + y[i] + "-(" + intercept + "+" + slope + "*" +
                 x[i] + "))^2");
    if (op == Op::linear_fit)
      out.refs = {slope, intercept, rss, yy, std::to_string(n)};
    else
      store(0, op == Op::linear_slope       ? slope
               : op == Op::linear_intercept ? intercept
               : op == Op::linear_r_squared
                   ? iff(yy + ">0", "1-" + rss + "/" + yy, err("DOMAIN_ERROR"))
               : n > 2 ? "SQRT(" + rss + "/" + std::to_string(n - 2) + ")"
                       : err("DOMAIN_ERROR"));
    return out;
  }
  if (op >= Op::fit_slope && op <= Op::fit_observation_count) {
    out.refs[0] = a[0].refs.at(unsigned(op) - unsigned(Op::fit_slope));
    return out;
  }
  if (op == Op::state_estimate || op == Op::state_variance ||
      op == Op::continuous_state_values ||
      op == Op::continuous_state_evidence ||
      op == Op::continuous_state_pending || op == Op::segment_starts ||
      op == Op::segment_ends) {
    std::size_t col = op == Op::state_variance ||
                              op == Op::continuous_state_evidence ||
                              op == Op::segment_ends
                          ? 1
                      : op == Op::continuous_state_pending ? 2
                                                           : 0;
    for (std::size_t i = 0; i < out.size(); ++i)
      out.refs[i] = A(0, i * a[0].shape.dim[1] + col);
    return out;
  }
  if (op == Op::days_between) {
    store(0, iff("AND(" + fin(A(0)) + "," + fin(A(1)) + ")",
                 iff("AND(" + A(0) + "=INT(" + A(0) + ")," + A(1) + "=INT(" +
                         A(1) + ")," + A(1) + ">=" + A(0) + ")",
                     A(1) + "-" + A(0), nan()),
                 nan()));
    return out;
  }
  if (op == Op::require_positive || op == Op::require_nonnegative) {
    store(0, iff("AND(" + fin(A(0)) + "," + A(0) +
                     (op == Op::require_positive ? ">0" : ">=0") + ")",
                 A(0), err("DOMAIN_ERROR")));
    return out;
  }
  if (op == Op::state_select) {
    for (std::size_t i = 0; i < out.size(); ++i)
      store(i, iff(A(3, i), iff(A(0, i), A(1, i), A(2, i)), "-1"));
    return out;
  }
  if (op == Op::last_drawdown_interval) {
    F deepest = "0", latest = "0", peak = nan(), trough = nan(),
      recovery = nan(), valid = n ? A(0) + "=0" : "FALSE";
    for (std::size_t i = 0; i < n; ++i) {
      auto x = A(0, i), t = std::to_string(i);
      valid =
          cell("AND(" + valid + "," + fin(x) + "," + x + "<=0," + x + ">=-1)");
      auto lower = cell("AND(" + x + "<0," + x + "<=" + deepest + ")");
      latest = cell(iff(x + "=0", t, latest));
      peak = cell(iff(lower, latest, peak));
      trough = cell(iff(lower, t, trough));
      recovery = cell(iff(
          lower, nan(),
          iff("AND(" + x + "=0," + fin(trough) + ",NOT(" + fin(recovery) + "))",
              t, recovery)));
      deepest = cell(iff(lower, x, deepest));
    }
    out.refs = {cell(iff(valid, peak, nan())), cell(iff(valid, trough, nan())),
                cell(iff(valid, recovery, nan())),
                cell(iff(valid, "IF(" + deepest + "<0,1,0)", "-1"))};
    return out;
  }
  if (op >= Op::interval_start && op <= Op::interval_recovery) {
    out.refs[0] = a[0].refs.at(unsigned(op) - unsigned(Op::interval_start));
    return out;
  }
  if (op == Op::recursive_smooth || op == Op::recursive_filter ||
      op == Op::recursive_filter_adaptive || op == Op::linear_filter2 ||
      op == Op::scalar_kalman) {
    F previous = op == Op::recursive_filter_adaptive ? A(4)
                 : op == Op::recursive_smooth || op == Op::recursive_filter
                     ? A(2)
                     : "0";
    F initialized = op == Op::recursive_filter            ? A(4) + "<>2"
                    : op == Op::recursive_filter_adaptive ? A(5) + "=0"
                                                          : "FALSE";
    F count = "0", older = "0", previous_input = "0",
      variance = op == Op::scalar_kalman ? A(3) : "0";
    for (std::size_t i = 0; i < n; ++i) {
      auto x = A(0, i), valid = fin(x);
      if (op == Op::recursive_smooth) {
        previous = cell(
            iff(valid, "((" + A(1) + "-1)*" + previous + "+" + x + ")/" + A(1),
                previous));
        store(i, iff(valid, previous, nan()));
      } else if (op == Op::recursive_filter) {
        auto update = cell("AND(" + A(3, i) + "," + valid + "," +
                           (i ? "TRUE" : A(4) + "<>1") + ")");
        previous = cell(
            iff(update,
                iff(initialized,
                    "(1-" + A(1) + ")*" + previous + "+" + A(1) + "*" + x, x),
                previous));
        initialized = cell("OR(" + initialized + "," + update + ")");
        store(i, iff(i ? "FALSE" : A(4) + "=1", previous,
                     iff("AND(" + initialized + ",OR(" + update + "," + A(5) +
                             "=0))",
                         previous, nan())));
      } else if (op == Op::recursive_filter_adaptive) {
        auto reset = A(3, i);
        previous = cell(iff(reset, A(4), previous));
        initialized = cell(iff(reset, A(5) + "=0", initialized));
        count = cell(iff(reset, "0", count) + "+IF(" + valid + ",1,0)");
        auto seed = cell("AND(" + valid + ",NOT(" + initialized + "),OR(" +
                         A(5) + "=1," + A(2, i) + "))");
        auto update =
            cell("AND(" + valid + "," + initialized + "," + A(2, i) + ")");
        auto gain = A(1, i);
        previous = cell(iff(
            seed, x,
            iff(update,
                iff("AND(" + fin(gain) + "," + gain + ">=0," + gain + "<=1)",
                    previous + "+" + gain + "*(" + x + "-" + previous + ")",
                    err("INVALID_PARAMETER")),
                previous)));
        initialized = cell("OR(" + initialized + "," + seed + ")");
        store(i, iff("AND(" + initialized + "," + count + ">=" + A(6) + ",OR(" +
                         seed + "," + update + "," + A(7) + "=0))",
                     previous, nan()));
      } else if (op == Op::linear_filter2) {
        auto reset = A(5, i);
        previous = cell(iff(reset, "0", previous));
        older = cell(iff(reset, "0", older));
        previous_input = cell(iff(reset, "0", previous_input));
        count = cell(iff(reset, "0", count));
        auto next =
            cell(iff("AND(" + A(7) + "=1," + count + "<2)", x,
                     A(1) + "*" + x + "+" + A(2) + "*" + previous_input + "+" +
                         A(3) + "*" + previous + "+" + A(4) + "*" + older));
        auto good = cell("AND(" + valid + "," + fin(next) + ")");
        older = cell(iff(valid, iff(good, previous, "0"), older));
        previous = cell(iff(valid, iff(good, next, "0"), previous));
        previous_input = cell(iff(valid, iff(good, x, "0"), previous_input));
        count = cell(iff(valid, iff(good, count + "+1", "0"), count));
        store(i, iff("AND(" + good + "," + count + ">=" + A(6) + ")", previous,
                     nan()));
      } else {
        auto pred = cell(variance + "+" + A(1)),
             gain = cell(pred + "/(" + pred + "+" + A(2) + ")");
        previous = cell(
            iff(valid,
                iff(initialized,
                    previous + "+" + gain + "*(" + x + "-" + previous + ")", x),
                previous));
        variance = cell(iff("AND(" + valid + "," + initialized + ")",
                            pred + "*(1-" + gain + ")", variance));
        initialized = cell("OR(" + initialized + "," + valid + ")");
        store(i * 2, iff(valid, previous, nan()));
        store(i * 2 + 1, iff(valid, variance, nan()));
      }
    }
    return out;
  }
  if (op == Op::state_hysteresis) {
    F active = "1";
    for (std::size_t i = 0; i < n; ++i) {
      auto x = A(0, i);
      active = cell(iff(
          fin(x),
          iff(active + "=0",
              iff(x + "<=" + A(3), "2", iff(x + "<=" + A(2), "1", active)),
              iff(active + "=2",
                  iff(x + ">=" + A(1), "0", iff(x + ">=" + A(4), "1", active)),
                  iff(x + ">=" + A(1), "0",
                      iff(x + "<=" + A(3), "2", active)))),
          active));
      store(i, iff(fin(x), active, "-1"));
    }
    return out;
  }
  if (op == Op::state_confirm) {
    F active = "-1", candidate = "-1", count = "0", duration = "0";
    for (std::size_t i = 0; i < n; ++i) {
      auto x = A(0, i), old = active;
      duration = cell(duration + "+IF(" + active + ">=0,1,0)");
      auto pending = cell("AND(" + x + ">=0," + x + "<>" + active + ")");
      count =
          cell(iff(pending, iff(x + "=" + candidate, count + "+1", "1"), "0"));
      candidate = cell(iff(pending, x, "-1"));
      auto change = cell("AND(" + pending + "," + count + ">=" + A(1) + ",OR(" +
                         active + "<0," + duration + ">" + A(2) + "))");
      active = cell(iff(change, candidate, active));
      duration = cell(iff(change, "1", duration));
      auto clear = cell("AND(" + change + "," + old + ">=0)");
      candidate = cell(iff(clear, "-1", candidate));
      count = cell(iff(clear, "0", count));
      store(i, iff(x + "<0", "-1", active));
    }
    return out;
  }
  if (op == Op::state_continuous) {
    F active = n ? A(1) : "-1", proposed = "-1", count = "0",
      confirmed = "FALSE";
    for (std::size_t i = 0; i < n; ++i) {
      auto x = A(0, i);
      auto same = cell(x + "=" + active),
           pending = cell("AND(" + x + ">=0,NOT(" + same + "))");
      count =
          cell(iff(pending, iff(x + "=" + proposed, count + "+1", "1"), "0"));
      proposed = cell(iff(pending, x, "-1"));
      auto changed = cell("AND(" + pending + "," + count + ">=" + A(3) + ")"),
           accepted = cell("OR(" + same + "," + changed + ")");
      auto evidence = cell(
          iff(x + "<0", iff(confirmed, "2", "0"), iff(accepted, "1", "3")));
      active = cell(iff(changed, proposed, active));
      proposed = cell(iff(changed, "-1", proposed));
      count = cell(iff(changed, "0", count));
      confirmed = cell("OR(" + confirmed + "," + accepted + ")");
      store(i * 3, active);
      store(i * 3 + 1, evidence);
      store(i * 3 + 2, count);
    }
    return out;
  }
  if (op == Op::drawdown_cycle_state) {
    F active = "0", trough = nan();
    for (std::size_t i = 0; i < n; ++i) {
      auto x = A(0, i), d = A(1, i), old = active;
      auto valid = cell("AND(" + A(2, i) + "," + fin(x) + "," + x + ">0," +
                        fin(d) + ")");
      auto stress = d + "<=-" + A(3), exit = d + ">=-" + A(5);
      auto low = cell("IF(" + fin(trough) + "," + x + "<" + trough + ",TRUE)");
      auto update =
          cell("OR(AND(" + old + "<0,OR(" + stress + "," + exit + ")),AND(" +
               old + "=0," + stress + "),AND(" + old + "=2," + low + "),AND(" +
               old + "=1,OR(" + low + "," + exit + ")))");
      active = cell(
          iff(valid,
              iff(old + "<0", iff(stress, "2", iff(exit, "0", old)),
                  iff(old + "=0", iff(stress, "2", old),
                      iff(old + "=2",
                          iff(low, old,
                              iff(x + "/" + trough + "-1>=" + A(4), "1", old)),
                          iff(low, "2", iff(exit, "0", old))))),
              "-1"));
      trough = cell(iff(valid, iff(update, x, trough), nan()));
      store(i, active);
    }
    return out;
  }
  if (op == Op::between_events) {
    F previous = "-1";
    std::vector<F> left(n, "-1"), right(n, "-1");
    for (std::size_t t = 0; t < n; ++t) {
      auto kind = A(0, t);
      auto event = cell("OR(" + kind + "=1," + kind + "=-1)"),
           prior = choose(a[0].refs, previous, "0");
      auto complete = cell("AND(" + event + "," + previous + ">=0," + kind +
                           "<>" + prior + ")");
      for (std::size_t j = 0; j < t; ++j) {
        auto member =
            "AND(" + complete + "," + std::to_string(j) + ">=" + previous + ")";
        left[j] = cell(iff(member, previous, left[j]));
        right[j] = cell(iff(member, std::to_string(t), right[j]));
      }
      previous = cell(
          iff(kind + "=-2", "-1", iff(event, std::to_string(t), previous)));
    }
    for (std::size_t i = 0; i < n; ++i) {
      out.refs[2 * i] = left[i];
      out.refs[2 * i + 1] = right[i];
    }
    return out;
  }
  if (op == Op::phase_direction) {
    for (std::size_t i = 0; i < n; ++i) {
      auto l = A(1, 2 * i), r = A(1, 2 * i + 1), le = choose(a[0].refs, l, "0"),
           re = choose(a[0].refs, r, "0");
      store(i, iff(l + ">=0",
                   iff("AND(" + le + "=-1," + re + "=1)", "0",
                       iff("AND(" + le + "=1," + re + "=-1)", "1", "-1")),
                   "-1"));
    }
    return out;
  }
  if (op == Op::drawdown_cycle_reference) {
    F prior = "FALSE";
    std::vector<F> v(n, "-1");
    for (std::size_t t = 0; t < n; ++t) {
      auto l = A(2, 2 * t), r = A(2, 2 * t + 1), phase = A(0, t),
           change = A(1, t);
      auto start = cell("AND(" + l + "=" + std::to_string(t) + "," + r + ">" +
                        l + ")"),
           valid = cell("AND(OR(" + phase + "=0," + phase + "=1)," +
                        fin(change) + ")");
      auto stress = cell("AND(" + phase + "=1," + change + "<=-" + A(3) + ")");
      auto recovery = cell("AND(" + phase + "=0," + prior + ")");
      for (std::size_t j = t; j < n; ++j) {
        auto member = "AND(" + start + "," + valid + "," + std::to_string(j) +
                      "<=" + r + ")";
        F next = j == t ? iff(recovery, "2", iff(v[j] + "<0", "0", v[j]))
                        : iff(stress, "2", iff(recovery, "1", "0"));
        v[j] = cell(iff(member, next, v[j]));
      }
      prior = cell(iff(start, "AND(" + valid + "," + stress + ")", prior));
    }
    out.refs = v;
    return out;
  }
  if (op == Op::local_extrema || op == Op::ps_filter) {
    // Full-sample event editing is an acyclic unrolling. No result or position
    // is selected in C++ from observed prices: every decision remains a
    // formula.
    std::vector<F> valid(n), events(n, "0"), run_start(n), run_end(n);
    for (std::size_t i = 0; i < n; ++i)
      valid[i] =
          cell("AND(" + fin(A(0, i)) + "," + A(0, i) + ">0" +
               (op == Op::ps_filter ? "," + A(1, i) + "<>-2" : "") + ")");
    F start = "0";
    for (std::size_t i = 0; i < n; ++i) {
      start = cell(iff(valid[i], start, std::to_string(i + 1)));
      run_start[i] = start;
    }
    F end = std::to_string(n);
    for (std::size_t rev = 0; rev < n; ++rev) {
      auto i = n - 1 - rev;
      end = cell(iff(valid[i], end, std::to_string(i)));
      run_end[i] = end;
    }
    if (op == Op::local_extrema) {
      const auto left = bound(a[1], "extrema left", 5000),
                 right = bound(a[2], "extrema right", 5000),
                 head = std::max(left, bound(a[3], "extrema head", 5000)),
                 tail = std::max(right, bound(a[4], "extrema tail", 5000));
      F previous = "-1", kind = "0";
      for (std::size_t i = 0; i < n; ++i) {
        std::vector<F> peak, trough;
        for (std::size_t j = i > left ? i - left : 0; j < i; ++j) {
          peak.push_back(A(0, i) + ">" + A(0, j));
          trough.push_back(A(0, i) + "<" + A(0, j));
        }
        for (std::size_t j = i + 1; j < std::min(n, i + right + 1); ++j) {
          peak.push_back(A(0, i) + ">=" + A(0, j));
          trough.push_back(A(0, i) + "<=" + A(0, j));
        }
        auto p = all(peak), v = all(trough);
        auto candidate = cell(iff(
            "AND(" + valid[i] + "," + std::to_string(i) + ">=" + run_start[i] +
                "+" + std::to_string(head) + "," + std::to_string(i) + "<" +
                run_end[i] + "-" + std::to_string(tail) + ")",
            iff(p, "1", iff(v, "-1", "0")), "0"));
        auto prev_price = choose(a[0].refs, previous, "0");
        auto accept = cell(
            iff(valid[i],
                "AND(" + candidate + "<>0," +
                    iff(previous + "<" + run_start[i], "TRUE",
                        candidate + "*(" + A(0, i) + "-" + prev_price + ")>0") +
                    ")",
                "FALSE"));
        for (std::size_t j = 0; j < i; ++j)
          events[j] = cell(iff("AND(" + accept + "," + kind + "=" + candidate +
                                   "," + previous + "=" + std::to_string(j) +
                                   "," + previous + ">=" + run_start[i] + ")",
                               "0", events[j]));
        events[i] = cell(iff(valid[i], iff(accept, candidate, "0"), "-2"));
        previous = cell(iff(accept, std::to_string(i), previous));
        kind = cell(iff(accept, candidate, kind));
      }
    } else {
      events = a[1].refs;
      auto alternate = [&](std::vector<F> e) {
        F prev = "-1", pk = "0";
        std::vector<F> v(n, "0");
        for (std::size_t t = 0; t < n; ++t) {
          auto kind = e[t], price = choose(a[0].refs, prev, "0");
          auto accept =
              cell(iff(valid[t],
                       "AND(OR(" + kind + "=1," + kind + "=-1)," +
                           iff(prev + "<" + run_start[t], "TRUE",
                               kind + "*(" + A(0, t) + "-" + price + ")>0") +
                           ")",
                       "FALSE"));
          for (std::size_t j = 0; j < t; ++j)
            v[j] = cell(iff("AND(" + accept + "," + pk + "=" + kind + "," +
                                prev + "=" + std::to_string(j) + "," + prev +
                                ">=" + run_start[t] + ")",
                            "0", v[j]));
          v[t] = cell(iff(valid[t], iff(accept, kind, "0"), "-2"));
          prev = cell(iff(accept, std::to_string(t), prev));
          pk = cell(iff(accept, kind, pk));
        }
        return v;
      };
      events = alternate(events);
      // Each successful pass deletes at least one turn, so n passes suffice.
      for (std::size_t pass = 0; pass < n; ++pass) {
        std::vector<F> remove(n, "FALSE");
        F found = "FALSE";
        // Boundary removals take priority, then cycle, then phase; scan order
        // is the native order within each contiguous valid run.
        for (std::size_t run = 0; run < n; ++run) {
          auto starts =
              cell(run ? "AND(" + valid[run] + ",NOT(" + valid[run - 1] + "))"
                       : valid[run]);
          F first = "-1", last = "-1";
          std::vector<F> positions;
          for (std::size_t t = run; t < n; ++t) {
            auto take = cell("AND(" + starts + "," + std::to_string(t) + "<" +
                             run_end[run] + ",OR(" + events[t] + "=1," +
                             events[t] + "=-1))");
            first = cell(iff("AND(" + take + "," + first + "<0)",
                             std::to_string(t), first));
            last = cell(iff(take, std::to_string(t), last));
            positions.push_back(take);
          }
          auto first_kind = choose(events, first, "0"),
               last_kind = choose(events, last, "0"),
               first_price = choose(a[0].refs, first, "1"),
               last_price = choose(a[0].refs, last, "1");
          std::vector<F> before, after;
          for (std::size_t t = run; t < n; ++t) {
            before.push_back(
                iff(std::to_string(t) + "<" + first,
                    first_kind + "*(" + A(0, t) + "-" + first_price + ")>0",
                    "FALSE"));
            after.push_back(
                iff("AND(" + std::to_string(t) + ">" + last + "," +
                        std::to_string(t) + "<" + run_end[run] + ")",
                    last_kind + "*(" + A(0, t) + "-" + last_price + ")>0",
                    "FALSE"));
          }
          F target = cell(iff(
                "AND(" + starts + "," + first + ">=0)",
                iff(any(before), first, iff(any(after), last, "-1")), "-1")),
            second = "-1";
          std::vector<F> next(n, "-1"), previous(n, "-1");
          F prev = "-1", succ = "-1";
          for (std::size_t t = run; t < n; ++t) {
            previous[t] = prev;
            prev = cell(iff(positions[t - run], std::to_string(t), prev));
          }
          for (std::size_t rev = 0; rev < n - run; ++rev) {
            auto t = n - 1 - rev;
            next[t] = succ;
            succ = cell(iff(positions[t - run], std::to_string(t), succ));
          }
          for (std::size_t t = run; t < n; ++t) {
            auto mid = next[t], last2 = choose(next, mid, "-1");
            auto later = choose(a[0].refs, last2, "1");
            auto violation = cell("AND(" + positions[t - run] + "," + target +
                                  "<0," + last2 + ">=0," + last2 + "-" +
                                  std::to_string(t) + "<" + A(3) + ")");
            auto stronger =
                cell(events[t] + "*(" + later + "-" + A(0, t) + ")>0");
            target = cell(
                iff(violation, iff(stronger, std::to_string(t), mid), target));
            second = cell(iff(violation, iff(stronger, mid, last2), second));
          }
          for (std::size_t t = run; t < n; ++t) {
            auto right = next[t], rp = choose(a[0].refs, right, "1"),
                 pre = previous[t], following = choose(next, right, "-1");
            auto pp = choose(a[0].refs, pre, "1"),
                 fp = choose(a[0].refs, following, "1");
            auto violation =
                cell(iff(positions[t - run],
                         "AND(" + target + "<0," + right + ">=0," + right +
                             "-" + std::to_string(t) + "<" + A(2) + ",ABS(" +
                             rp + "/" + A(0, t) + "-1)<=" + A(4) + ")",
                         "FALSE"));
            auto at =
                cell(iff(right + "=" + last, right,
                         iff(pre + ">=0",
                             iff("ABS(LN(" + A(0, t) + ")-LN(" + pp +
                                     "))>=ABS(LN(" + fp + ")-LN(" + rp + "))",
                                 right, std::to_string(t)),
                             std::to_string(t))));
            target = cell(iff(violation, at, target));
          }
          for (std::size_t t = run; t < n; ++t)
            remove[t] = cell("OR(" + remove[t] + ",AND(" + starts + ",OR(" +
                             target + "=" + std::to_string(t) + "," + second +
                             "=" + std::to_string(t) + ")))");
        }
        for (std::size_t t = 0; t < n; ++t)
          events[t] = cell(iff(remove[t], "0", events[t]));
        events = alternate(events);
      }
    }
    out.refs = events;
    return out;
  }
  throw Error("UNSUPPORTED_OPERATOR: " +
              F(ops::lookup(static_cast<std::uint16_t>(op)).name));
}
} // namespace calmetrics_engine::excel
