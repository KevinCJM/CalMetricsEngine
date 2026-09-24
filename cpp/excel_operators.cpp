#include "excel_internal.hpp"

namespace calmetrics_engine::excel {
namespace {
F nan() { return q("NaN"); }
F numeric(const F &condition, const F &f, const F &otherwise = "\"NaN\"") {
  return iff(condition, "IFERROR(" + f + "," + err("DOMAIN_ERROR") + ")",
             otherwise);
}
Op base_reduction(Op op) {
  if (op >= Op::sum_where && op <= Op::quantile_where) {
    const Op map[]{Op::sum,       Op::mean,      Op::variance, Op::std,
                   Op::min_value, Op::max_value, Op::median,   Op::quantile};
    return map[unsigned(op) - unsigned(Op::sum_where)];
  }
  return op;
}
} // namespace
F Builder::reduce(Op raw, const std::vector<F> &x, const std::vector<F> &mask,
                  const F &ddof, const F &prob) {
  const auto op = base_reduction(raw);
  const bool masked = !mask.empty();
  if (x.empty())
    return err("INSUFFICIENT_SAMPLE");
  F count = "0", mean = "0", m2 = "0", total = "0", squares = "0",
    product = "1",
    // Excel rejects DBL_MAX as a formula literal and repairs away the formula.
    // Use the same exact binary decomposition as every other large constant.
      best = masked ? num(op == Op::min_value
                              ? std::numeric_limits<double>::max()
                              : -std::numeric_limits<double>::max())
                    : x[0],
    index = "0";
  std::vector<F> selected, finite_selected;
  for (std::size_t i = 0; i < x.size(); ++i) {
    auto take = masked ? mask[i] : "TRUE";
    auto n = formula(bin(count, "+", iff(take, "1", "0")));
    auto valid = "AND(" + take + "," + fin(x[i]) + ")";
    selected.push_back(take);
    finite_selected.push_back("AND(" + take + "," + fin(x[i]) + ")");
    if (op == Op::min_value || op == Op::max_value || op == Op::argmin ||
        op == Op::argmax) {
      auto better =
          "AND(" + take + "," + fin(x[i]) + "," + fin(best) + "," + x[i] +
          (op == Op::min_value || op == Op::argmin ? "<" : ">") + best + ")";
      best = formula(iff(better, x[i], best));
      index = formula(iff(better, std::to_string(i), index));
    } else if (op == Op::sum)
      total = formula(iff(take,
                          numeric("AND(" + fin(total) + "," + fin(x[i]) + ")",
                                  bin(total, "+", x[i])),
                          total));
    else if (op == Op::product)
      product =
          formula(iff(take,
                      numeric("AND(" + fin(product) + "," + fin(x[i]) + ")",
                              bin(product, "*", x[i])),
                      product));
    else if (op == Op::root_mean_square)
      squares =
          formula(iff(take,
                      numeric("AND(" + fin(squares) + "," + fin(x[i]) + ")",
                              squares + "+" + x[i] + "^2"),
                      squares));
    else if (op != Op::median && op != Op::quantile) {
      auto delta = formula(numeric("AND(" + fin(x[i]) + "," + fin(mean) + ")",
                                   bin(x[i], "-", mean)));
      auto next =
          formula(iff(take,
                      numeric("AND(" + fin(mean) + "," + fin(delta) + ")",
                              mean + "+" + delta + "/" + n),
                      mean));
      m2 = formula(iff(
          take,
          numeric("AND(" + fin(m2) + "," + fin(delta) + "," + fin(next) + ")",
                  m2 + "+" + delta + "*(" + x[i] + "-" + next + ")"),
          m2));
      mean = next;
    }
    count = n;
  }
  F result;
  if (op == Op::sum)
    result = total;
  else if (op == Op::product)
    result = product;
  else if (op == Op::mean)
    result = mean;
  else if (op == Op::min_value || op == Op::max_value)
    result = masked ? iff(any(finite_selected), best,
                          q(op == Op::min_value ? "+Inf" : "-Inf"))
                    : best;
  else if (op == Op::argmin || op == Op::argmax)
    result = index;
  else if (op == Op::variance || op == Op::std) {
    auto val = m2 + "/(" + count + "-" + ddof + ")";
    result = iff(count + ">" + ddof,
                 numeric(fin(m2), op == Op::std ? fn("SQRT", val) : val),
                 err(masked ? "INSUFFICIENT_SAMPLE" : "INVALID_PARAMETER"));
  } else if (op == Op::root_mean_square)
    result = numeric(fin(squares), "SQRT(" + squares + "/" + count + ")");
  else if (op == Op::median || op == Op::quantile) {
    // Stable rank expansion preserves NaNs at the end and counts selected NaNs.
    std::vector<F> ranks;
    for (std::size_t i = 0; i < x.size(); ++i) {
      std::vector<F> precedes;
      for (std::size_t j = 0; j < x.size(); ++j) {
        auto comparison = iff(
            fin(x[i]),
            "IF(" + fin(x[j]) + ",OR(" + x[j] + "<" + x[i] + ",AND(" + x[j] +
                "=" + x[i] + "," + (j < i ? "TRUE" : "FALSE") + ")),FALSE)",
            "OR(" + fin(x[j]) + "," + (j < i ? "TRUE" : "FALSE") + ")");
        precedes.push_back(formula(
            iff("AND(" + selected[j] + "," + comparison + ")", "1", "0")));
      }
      ranks.push_back(sum(precedes));
    }
    auto position =
        formula("(" + count + "-1)*" + (op == Op::median ? "0.5" : prob));
    auto pick = [&](const F &pos) {
      F r = nan();
      for (std::size_t i = 0; i < x.size(); ++i)
        r = formula(iff("AND(" + selected[i] + "," + ranks[i] + "=" + pos + ")",
                        x[i], r));
      return r;
    };
    auto lo = pick("INT(" + position + ")"),
         hi = pick("ROUNDUP(" + position + ",0)");
    auto fraction = "(" + position + "-INT(" + position + "))";
    result = numeric("AND(" + fin(lo) + "," + fin(hi) + ")",
                     lo + "*(1-" + fraction + ")+" + hi + "*" + fraction);
  } else {
    std::vector<F> terms;
    for (std::size_t i = 0; i < x.size(); ++i) {
      auto d = "(" + x[i] + "-" + mean + ")";
      auto term = op == Op::mean_absolute_deviation
                      ? fn("ABS", d)
                      : d + (op == Op::skewness ? "^3" : "^4");
      terms.push_back(formula(
          iff(selected[i],
              numeric("AND(" + fin(x[i]) + "," + fin(mean) + ")", term), "0")));
    }
    auto accum = sum(terms);
    if (op == Op::mean_absolute_deviation)
      result = numeric(fin(mean), accum + "/" + count);
    else if (op == Op::skewness)
      result =
          iff("AND(" + count + ">=3," + m2 + ">0)",
              numeric(fin(mean), "SQRT(" + count + "*(" + count + "-1))/(" +
                                     count + "-2)*(" + accum + "/" + count +
                                     ")/(" + m2 + "/" + count + ")^1.5"),
              err("INSUFFICIENT_SAMPLE"));
    else
      result = iff("AND(" + count + ">=4," + m2 + ">0)",
                   numeric(fin(mean), "(" + count + "-1)/((" + count + "-2)*(" +
                                          count + "-3))*((" + count + "+1)*(" +
                                          accum + "/" + count + "/(" + m2 +
                                          "/" + count + ")^2-3)+6)"),
                   err("INSUFFICIENT_SAMPLE"));
  }
  return formula(iff(count + ">0", result, err("INSUFFICIENT_SAMPLE")));
}

Array Builder::operation(Op op, const std::vector<Array> &given) {
  auto out = recipe(op, given);
  const auto &spec = ops::lookup(static_cast<std::uint16_t>(op));
  const bool aligned =
      spec.family == ops::Family::elementwise || op == Op::state_select;
  const bool mapped = op == Op::lag || op == Op::difference ||
                      op == Op::aligned_shift || op == Op::first ||
                      op == Op::last || op == Op::length;
  const bool positions =
      std::any_of(given.begin(), given.end(),
                  [](const auto &a) { return a.position_errors; });
  auto merge = [&](const F &left, const F &right) {
    if (graph_mode && isolate)
      return formula(iff("IFERROR(VALUE(MID(" + right +
                             ",14,8)),0)>IFERROR(VALUE(MID(" + left +
                             ",14,8)),0)",
                         right, iff(left + "<>\"\"", left, right)));
    return formula(iff(left + "<>\"\"", left, right));
  };
  if (graph_mode && isolate) {
    for (auto &x : out.refs) {
      F code = "4";
      if ((op == Op::divide && !given[1].shape.rank) ||
          (op == Op::reciprocal && !given[0].shape.rank))
        code = iff(x + "=\"ERROR:DIVIDE_BY_ZERO\"", "2", "4");
      if (!given[0].shape.rank &&
          (op == Op::sqrt || op == Op::log || op == Op::require_positive ||
           op == Op::require_nonnegative))
        code = iff(x + "=\"ERROR:DOMAIN_ERROR\"", "3", "4");
      x = formula(
          iff("LEFT(" + x + ",6)=\"ERROR:\"", "\"ERROR:STATUS:\"&" + code, x));
      if (!out.shape.rank && out.kind == ops::Kind::number) {
        F missing = op == Op::value_at || op == Op::days_between ? "7" : "4";
        if (op >= Op::interval_start && op <= Op::interval_recovery)
          missing = iff(given[0].refs.at(3) + "=0", "9", "10");
        x = formula(iff("OR(ISNUMBER(" + x + "),LEFT(" + x + ",6)=\"ERROR:\")",
                        x, "\"ERROR:STATUS:\"&" + missing));
      }
    }
  }
  F own = "\"\"";
  if (!((aligned || mapped) && positions))
    for (auto &x : out.refs)
      own = formula(iff(own + "<>\"\"", own,
                        iff("LEFT(" + x + ",6)=\"ERROR:\"", x, "\"\"")));
  F inherited = "\"\"";
  for (auto &a : given)
    inherited = merge(inherited, a.failure);
  if (!aligned && !mapped)
    for (auto &a : given)
      for (auto &x : a.refs)
        inherited =
            merge(inherited, iff("LEFT(" + x + ",6)=\"ERROR:\"", x, "\"\""));
  out.failure = formula(iff(inherited + "<>\"\"", inherited, own));
  out.position_errors = positions && (aligned || mapped);
  for (std::size_t i = 0; i < out.size(); ++i) {
    F error = out.failure;
    if (aligned)
      for (auto &a : given) {
        auto x = a.at(i);
        error = merge(error, iff("LEFT(" + x + ",6)=\"ERROR:\"", x, "\"\""));
      }
    if (mapped && op != Op::length && positions && !given[0].refs.empty()) {
      auto consume = [&](std::size_t index) {
        const auto &x = given[0].refs.at(index);
        error = merge(error, iff("LEFT(" + x + ",6)=\"ERROR:\"", x, "\"\""));
      };
      const auto periods =
          given.size() > 1
              ? static_cast<std::size_t>(given[1].scalar.value_or(0))
              : 0;
      if (op == Op::first)
        consume(0);
      else if (op == Op::last)
        consume(given[0].size() - 1);
      else if (op == Op::aligned_shift) {
        if (i >= periods)
          consume(i - periods);
      } else {
        consume(i);
        if (op == Op::difference)
          consume(i + periods);
      }
    }
    out.refs[i] = formula(iff(error + "<>\"\"", error, out.refs[i]));
  }
  return out;
}
Array Builder::recipe(Op op, const std::vector<Array> &given) {
  const auto &spec = ops::lookup(static_cast<std::uint16_t>(op));
  if (given.size() < spec.min_args || given.size() > spec.max_args)
    throw Error("INVALID_INPUT: operator arity");
  // Cheap checked polynomial bounds precede all helper-cell/string expansion.
  const auto extent = given[0].size();
  if (op == Op::ps_filter)
    cost({extent, extent, extent, extent, 100});
  else if (op == Op::argsort || op == Op::median || op == Op::quantile ||
           op == Op::median_where || op == Op::quantile_where ||
           op == Op::local_extrema || op == Op::between_events ||
           op == Op::distinct_count)
    cost({extent, extent, 24});
  else if (op == Op::solve)
    cost({given[1].size(), given[1].size(), given[1].size(), 32});
  else if (op == Op::matmul)
    cost({given[0].shape.dim[0], given[0].shape.dim[1], given[1].shape.dim[1],
          16});
  else if (op == Op::quadratic_form)
    cost({extent, extent, 16});
  else if ((op == Op::covariance || op == Op::correlation) &&
           given[0].shape.rank == 2)
    cost({given[0].shape.dim[0], given[0].shape.dim[1], given[0].shape.dim[1],
          32});
  else if (op == Op::phase_direction || op == Op::drawdown_cycle_reference)
    cost({extent, extent, 24});
  else if (op == Op::rolling_min || op == Op::rolling_max)
    cost({extent,
          std::min(extent, given[1].scalar && *given[1].scalar >= 0 &&
                                   *given[1].scalar <= 1000000
                               ? static_cast<std::size_t>(*given[1].scalar)
                               : extent),
          16});
  else
    cost({extent + 1, 200});
  auto a = given;
  std::vector<ops::Value> values;
  for (auto &x : a) {
    ops::Value v;
    v.kind = x.kind;
    v.shape = x.shape;
    v.scalar = x.scalar.value_or(0.0);
    values.push_back(v);
  }
  auto unavailable = [&](const F &failure) {
    std::uint32_t geometry = 0, payload = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      if (a[i].shape.rank >= 0)
        geometry |= 1u << i;
    }
    // No generated cell has an available native payload. This shared validator
    // still checks all independent ranks/dimensions, including after failure.
    auto structure = ops::validate_structure(spec, values.data(), values.size(),
                                             geometry, payload);
    Array result;
    result.kind = structure.output_kind;
    result.shape = structure.output_shape;
    if (!structure.geometry_known)
      result.shape.rank = -1;
    result.refs.assign(structure.geometry_known ? result.shape.size() : 1,
                       failure);
    result.failure = failure;
    result.unavailable = true;
    return result;
  };
  for (const auto &argument : a)
    if (argument.unavailable)
      return unavailable(argument.failure);
  ops::Prepared p;
  try {
    p = ops::prepare_geometry(spec, values.data(), values.size());
  } catch (const ops::Error &error) {
    const std::string code = error.what();
    if (graph_mode &&
        (code == "INVALID_PARAMETER" || code == "INSUFFICIENT_SAMPLE")) {
      for (const auto &argument : a)
        if (!argument.shape.rank) {
          if (!argument.scalar)
            throw Error("UNKNOWN_BOUND: unavailable scalar geometry");
          structural_guards.push_back(argument.at() + "=" +
                                      num(*argument.scalar));
        }
      return unavailable(err("STATUS:4"));
    }
    throw;
  }
  if (!p.output_geometry_known)
    throw Error("UNKNOWN_BOUND: operator geometry");
  for (std::size_t i = a.size(); i < spec.max_args; ++i)
    a.push_back(literal(p.args[i].scalar));
  Array out = array(p.output_shape, p.output_kind);
  if (out.kind == ops::Kind::fit || out.kind == ops::Kind::interval)
    out.refs.resize(5);
  auto A = [&](std::size_t arg, std::size_t i = 0) { return a.at(arg).at(i); };
  auto store = [&](std::size_t i, const F &f) { out.refs.at(i) = formula(f); };
  const auto n = a[0].size();
  if (spec.family == ops::Family::elementwise || op == Op::active_returns) {
    for (std::size_t i = 0; i < out.size(); ++i) {
      auto x = A(0, i), y = a.size() > 1 ? A(1, i) : "0",
           z = a.size() > 2 ? A(2, i) : "0";
      F f;
      switch (op) {
      case Op::add:
        f = x + "+" + y;
        break;
      case Op::subtract:
      case Op::active_returns:
        f = x + "-" + y;
        break;
      case Op::multiply:
        f = x + "*" + y;
        break;
      case Op::divide:
        f = iff(y + "=0", err("DIVIDE_BY_ZERO"),
                numeric("AND(" + fin(x) + "," + fin(y) + ")", x + "/" + y));
        break;
      case Op::power:
        f = iff("OR(" + y + "=0," + x + "=1)", "1",
                numeric("AND(" + fin(x) + "," + fin(y) + ")", x + "^" + y,
                        err("DOMAIN_ERROR")));
        break;
      case Op::minimum:
        f = iff("IF(" + fin(y) + "," + y + "<" + x + ",FALSE)", y, x);
        break;
      case Op::maximum:
        f = iff("IF(" + fin(y) + "," + x + "<" + y + ",FALSE)", y, x);
        break;
      case Op::negate:
        f = "-" + x;
        break;
      case Op::absolute:
        f = fn("ABS", x);
        break;
      case Op::sqrt:
        f = fn("SQRT", x);
        break;
      case Op::clip: {
        auto lower =
            formula(iff("IF(" + fin(y) + "," + x + "<" + y + ",FALSE)", y, x));
        f = iff(
            "AND(" + fin(y) + "," + fin(z) + "," + y + ">" + z + ")",
            err("INVALID_PARAMETER"),
            iff("IF(" + fin(z) + "," + z + "<" + lower + ",FALSE)", z, lower));
      } break;
      case Op::log:
        f = fn("LN", x);
        break;
      case Op::exp:
        f = numeric(fin(x), fn("EXP", x), err("DOMAIN_ERROR"));
        break;
      case Op::reciprocal:
        f = iff(x + "=0", err("DIVIDE_BY_ZERO"), "1/" + x);
        break;
      case Op::sign:
        f = iff(fin(x), fn("SIGN", x), "0");
        break;
      case Op::normal_pdf:
        f = "EXP(-0.5*" + x + "*" + x + ")/SQRT(2*PI())";
        break;
      case Op::normal_cdf:
        f = "_xlfn.NORM.S.DIST(" + x + ",TRUE)";
        break;
      case Op::normal_ppf: {
        // Same Acklam approximation as scalar_math; NORM.S.INV has a different
        // approximation and cannot silently replace this contract.
        auto tail = formula("OR(" + x + "<0.02425," + x + ">0.97575)");
        auto u = formula(iff(
            tail, "SQRT(-2*LN(IF(" + x + "<0.02425," + x + ",1-" + x + ")))",
            x + "-0.5"));
        auto r = formula(iff(tail, u, u + "*" + u));
        auto poly = [&](std::initializer_list<double> c) {
          F v;
          for (auto k : c)
            v = v.empty() ? num(k) : formula(v + "*" + r + "+" + num(k));
          return v;
        };
        auto tn = poly({-7.784894002430293e-3, -3.223964580411365e-1,
                        -2.400758277161838, -2.549732539343734,
                        4.374664141464968, 2.938163982698783});
        auto td = poly({7.784695709041462e-3, 3.224671290700398e-1,
                        2.445134137142996, 3.754408661907416, 1});
        auto cn = poly({-3.969683028665376e1, 2.209460984245205e2,
                        -2.759285104469687e2, 1.383577518672690e2,
                        -3.066479806614716e1, 2.506628277459239});
        auto cd = poly({-5.447609879822406e1, 1.615858368580409e2,
                        -1.556989798598866e2, 6.680131188771972e1,
                        -1.328068155288572e1, 1});
        f = iff("AND(" + x + ">0," + x + "<1)",
                iff(tail, "IF(" + x + "<0.02425,1,-1)*" + tn + "/" + td,
                    cn + "*" + u + "/" + cd),
                err("DOMAIN_ERROR"));
        break;
      }
      case Op::floor:
        f = fn("INT", x);
        break;
      case Op::cos:
        f = fn("COS", x);
        break;
      case Op::equal:
      case Op::not_equal:
      case Op::less_than:
      case Op::less_equal:
      case Op::greater_than:
      case Op::greater_equal: {
        const char *cmp[] = {"=", "<>", "<", "<=", ">", ">="};
        f = iff("AND(" + fin(x) + "," + fin(y) + ")",
                bin(x, cmp[unsigned(op) - unsigned(Op::equal)], y),
                op == Op::not_equal ? "TRUE" : "FALSE");
        break;
      }
      case Op::logical_and:
        f = "AND(" + x + "," + y + ")";
        break;
      case Op::logical_or:
        f = "OR(" + x + "," + y + ")";
        break;
      case Op::logical_not:
        f = fn("NOT", x);
        break;
      case Op::finite_mask:
        f = fin(x);
        break;
      case Op::where:
        f = iff(x, y, z);
        break;
      case Op::divide_or_default:
        f = iff("AND(" + fin(x) + "," + fin(y) + ")",
                iff("ABS(" + y + ")<1E-12", z, x + "/" + y), nan());
        break;
      default:
        throw Error("UNSUPPORTED_OPERATOR: elementwise");
      }
      const bool binary =
          spec.min_args == 2 && op != Op::logical_and && op != Op::logical_or;
      if (out.kind == ops::Kind::number && op != Op::where && op != Op::sign &&
          op != Op::divide_or_default && op != Op::power && op != Op::divide &&
          op != Op::exp)
        f = numeric(binary && op != Op::minimum && op != Op::maximum
                        ? "AND(" + fin(x) + "," + fin(y) + ")"
                        : fin(x),
                    f);
      store(i, f);
    }
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
    return out;
  }
  if (op >= Op::sum_time && op <= Op::max_asset) {
    bool time = op <= Op::max_time;
    const Op map[]{Op::sum, Op::mean,      Op::product,  Op::variance,
                   Op::std, Op::min_value, Op::max_value};
    auto base =
        map[unsigned(op) - unsigned(time ? Op::sum_time : Op::sum_asset)];
    const auto rows = a[0].shape.dim[0], cols = a[0].shape.dim[1];
    for (std::size_t i = 0; i < out.size(); ++i) {
      std::vector<F> v;
      for (std::size_t j = 0; j < (time ? rows : cols); ++j)
        v.push_back(A(0, time ? j * cols + i : i * cols + j));
      out.refs[i] = reduce(base, v);
    }
    return out;
  }
  if (spec.family == ops::Family::reduction) {
    if (op == Op::count_true || op == Op::max_consecutive_true) {
      F run = "0", longest = "0";
      for (auto &x : a[0].refs) {
        run = formula(iff(x, run + "+1", op == Op::count_true ? run : "0"));
        longest = formula("MAX(" + longest + "," + run + ")");
      }
      store(0, op == Op::count_true ? run : longest);
    } else if (op == Op::distinct_count) {
      std::vector<F> fresh;
      for (std::size_t i = 0; i < n; ++i) {
        std::vector<F> prev;
        for (std::size_t j = 0; j < i; ++j)
          prev.push_back("AND(" + (given.size() > 1 ? A(1, j) : "TRUE") + "," +
                         A(0, j) + "=" + A(0, i) + ")");
        fresh.push_back(
            formula(iff("AND(" + (given.size() > 1 ? A(1, i) : "TRUE") +
                            ",NOT(" + any(prev) + "))",
                        "1", "0")));
      }
      store(0, sum(fresh));
    } else {
      bool masked = op >= Op::sum_where && op <= Op::quantile_where;
      out.refs[0] = reduce(op, a[0].refs, masked ? a[1].refs : std::vector<F>{},
                           op == Op::variance || op == Op::std ? A(1) : "1",
                           op == Op::quantile         ? A(1)
                           : op == Op::quantile_where ? A(2)
                                                      : "0.5");
    }
    return out;
  }
  if (op == Op::first || op == Op::last || op == Op::length) {
    store(0, op == Op::length ? std::to_string(n)
             : n              ? A(0, op == Op::first ? 0 : n - 1)
                              : err("INSUFFICIENT_SAMPLE"));
    if (op == Op::length)
      out.scalar = double(n);
    return out;
  }
  if (op == Op::lag || op == Op::difference || op == Op::aligned_shift) {
    const auto periods = bound(a[1], "shift periods");
    for (std::size_t i = 0; i < out.size(); ++i)
      store(i, op == Op::lag ? A(0, i)
               : op == Op::difference
                   ? numeric("AND(" + fin(A(0, i + periods)) + "," +
                                 fin(A(0, i)) + ")",
                             A(0, i + periods) + "-" + A(0, i))
               : i < periods ? A(2)
                             : A(0, i - periods));
    return out;
  }
  if (op == Op::argsort) {
    std::vector<F> ranks;
    for (std::size_t i = 0; i < n; ++i) {
      std::vector<F> less;
      for (std::size_t j = 0; j < n; ++j)
        less.push_back(formula(iff(
            iff(fin(A(0, i)),
                "IF(" + fin(A(0, j)) + ",OR(" + A(0, j) + "<" + A(0, i) +
                    ",AND(" + A(0, j) + "=" + A(0, i) + "," +
                    (j < i ? "TRUE" : "FALSE") + ")),FALSE)",
                "OR(" + fin(A(0, j)) + "," + (j < i ? "TRUE" : "FALSE") + ")"),
            "1", "0")));
      ranks.push_back(sum(less));
    }
    for (std::size_t i = 0; i < n; ++i) {
      F ix = "-1";
      for (std::size_t j = 0; j < n; ++j)
        ix = formula(
            iff(ranks[j] + "=" + std::to_string(i), std::to_string(j), ix));
      out.refs[i] = ix;
    }
    return out;
  }
  if (op == Op::gather || op == Op::value_at) {
    for (std::size_t i = 0; i < out.size(); ++i)
      out.refs[i] =
          choose(a[0].refs, A(1, i),
                 op == Op::gather ? err("INDEX_OUT_OF_BOUNDS") : nan());
    return out;
  }
  if (op == Op::cumulative_sum || op == Op::cumulative_product ||
      op == Op::cumulative_return || op == Op::cumulative_max ||
      op == Op::cumulative_min || op == Op::drawdown_series ||
      op == Op::new_high_mask || op == Op::total_return ||
      op == Op::annualized_return) {
    F running = op == Op::cumulative_sum ? "0"
                : op == Op::cumulative_max || op == Op::cumulative_min ||
                        op == Op::drawdown_series || op == Op::new_high_mask
                    ? (n ? A(0) : nan())
                    : "1";
    for (std::size_t i = 0; i < n; ++i) {
      const auto x = A(0, i);
      F next;
      if (op == Op::cumulative_sum)
        next = numeric("AND(" + fin(running) + "," + fin(x) + ")",
                       running + "+" + x);
      else if (op == Op::cumulative_max || op == Op::drawdown_series ||
               op == Op::new_high_mask)
        next = iff("IF(" + fin(x) + ",IF(" + fin(running) + "," + x + ">" +
                       running + ",FALSE),FALSE)",
                   x, running);
      else if (op == Op::cumulative_min)
        next = iff("IF(" + fin(x) + ",IF(" + fin(running) + "," + x + "<" +
                       running + ",FALSE),FALSE)",
                   x, running);
      else
        next = numeric("AND(" + fin(running) + "," + fin(x) + ")",
                       running + "*(" + x +
                           (op == Op::cumulative_product ? ")" : "+1)"));
      auto old = running;
      running = formula(next);
      if (op == Op::total_return || op == Op::annualized_return)
        continue;
      store(i, op == Op::cumulative_return
                   ? numeric(fin(running), running + "-1")
               : op == Op::drawdown_series
                   ? numeric(fin(x), x + "/" + running + "-1")
               : op == Op::new_high_mask ? (i ? x + ">" + old : "TRUE")
                                         : running);
    }
    if (op == Op::total_return || op == Op::annualized_return)
      store(0, n ? numeric(fin(running), op == Op::total_return
                                             ? running + "-1"
                                             : running + "^(" + A(1) + "/" +
                                                   std::to_string(n) + ")-1")
                 : err("INSUFFICIENT_SAMPLE"));
    return out;
  }
  if (op == Op::rolling_mean || op == Op::rolling_std ||
      op == Op::rolling_min || op == Op::rolling_max) {
    const auto width = bound(a[1], "rolling window");
    if (width == 0)
      throw Error("INVALID_INPUT: zero window");
    const F minimum = op == Op::rolling_std ? A(3) : A(2),
            ddof = op == Op::rolling_std ? A(2) : "0";
    F total = "0", squares = "0", count = "0";
    for (std::size_t i = 0; i < n; ++i) {
      auto x = A(0, i), expired = i >= width ? A(0, i - width) : nan();
      total = formula(total + "+IF(" + fin(x) + "," + x + ",0)-IF(" +
                      fin(expired) + "," + expired + ",0)");
      squares = formula(squares + "+IF(" + fin(x) + "," + x + "^2,0)-IF(" +
                        fin(expired) + "," + expired + "^2,0)");
      count = formula(count + "+IF(" + fin(x) + ",1,0)-IF(" + fin(expired) +
                      ",1,0)");
      F value;
      if (op == Op::rolling_mean)
        value = total + "/" + count;
      else if (op == Op::rolling_std) {
        auto centered = formula(squares + "-" + total + "^2/" + count);
        value = "SQRT(IF(AND(" + centered + "<0," + centered + ">-1E-12),0," +
                centered + ")/(" + count + "-" + ddof + "))";
      } else {
        F best = nan();
        for (std::size_t j = i + 1 > width ? i + 1 - width : 0; j <= i; ++j)
          best = formula(iff(fin(A(0, j)),
                             iff(fin(best),
                                 (op == Op::rolling_min ? "MIN(" : "MAX(") +
                                     best + "," + A(0, j) + ")",
                                 A(0, j)),
                             best));
        value = best;
      }
      store(i, iff("AND(" + count + ">=" + minimum + "," + count + ">" + ddof +
                       ")",
                   "IFERROR(" + value + "," + nan() + ")", nan()));
    }
    return out;
  }
  // Matrix, fit and state recipes are kept separate from the production
  // kernels.
  if (spec.family == ops::Family::regression || op == Op::covariance ||
      op == Op::correlation)
    a.resize(given.size());
  return state_operation(op, a, std::move(out));
}
} // namespace calmetrics_engine::excel
