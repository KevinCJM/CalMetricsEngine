#include "excel_internal.hpp"

namespace calmetrics_engine::excel {
Array Builder::latent_operation(Op op, const std::vector<Array> &a, Array out) {
  auto A = [&](std::size_t i, std::size_t j = 0) { return a.at(i).at(j); };
  F active = "TRUE", failure = q("");
  auto cell = [&](const F &f) {
    return formula(iff(
        active, "IFERROR(" + f + "," + err("NON_FINITE_RESULT") + ")", "0"));
  };
  auto guard = [&](const F &test, const F &code) {
    failure = formula(iff(
        failure + "<>\"\"", failure,
        iff(active,
            "IFERROR(" + iff(test, q(""), err(code)) + "," + err(code) + ")",
            q(""))));
  };
  auto finite = [&](const std::vector<F> &v) {
    std::vector<F> checks;
    for (auto &x : v)
      checks.push_back(fin(x));
    guard(all(checks), "NON_FINITE_RESULT");
  };
  auto finish = [&]() {
    out.failure = failure;
    for (auto &x : out.refs)
      x = formula(iff(failure + "<>\"\"", failure, x));
    return out;
  };
  auto total = [&](const std::vector<F> &v) {
    F s = "0";
    for (auto &x : v)
      s = cell(s + "+" + x);
    return s;
  };
  auto maximum = [&](const std::vector<F> &v) {
    if (v.empty())
      return err("INSUFFICIENT_SAMPLE");
    F m = v[0];
    for (std::size_t i = 1; i < v.size(); ++i)
      m = cell("MAX(" + m + "," + v[i] + ")");
    return m;
  };
  auto lse = [&](const std::vector<F> &v) {
    finite(v); // MAX must not silently ignore a failed intermediate.
    auto m = maximum(v);
    std::vector<F> ex;
    for (auto &x : v)
      ex.push_back(cell("EXP(" + x + "-" + m + ")"));
    return cell(m + "+LN(MAX(" + total(ex) + ",1E-300))");
  };
  auto probabilities = [&](const std::vector<F> &v, std::size_t rows) {
    finite(v);
    std::vector<F> checks;
    for (auto &x : v)
      checks.push_back(x + ">=0");
    guard(all(checks), "DOMAIN_ERROR");
    const auto width = v.size() / rows;
    for (std::size_t i = 0; i < rows; ++i)
      guard(total(std::vector<F>(v.begin() + i * width,
                                 v.begin() + (i + 1) * width)) +
                ">0",
            "DOMAIN_ERROR");
  };
  const auto n = a[0].shape.dim[0];
  if (op >= Op::mixture_weights && op <= Op::hmm_variances) {
    const auto w = a[0].shape.dim[1];
    if (op >= Op::em_iterations && op <= Op::em_converged) {
      out.refs[0] = A(0, w - (op == Op::em_iterations       ? 3
                              : op == Op::em_log_likelihood ? 2
                                                            : 1));
      finite(out.refs);
      if (op == Op::em_iterations)
        guard("AND(" + out.refs[0] + ">=1," + out.refs[0] + "<=10000," +
                  out.refs[0] + "=INT(" + out.refs[0] + "))",
              "INVALID_PARAMETER");
      if (op == Op::em_converged) {
        guard("OR(" + out.refs[0] + "=0," + out.refs[0] + "=1)",
              "INVALID_MASK");
        out.refs[0] = cell(out.refs[0] + "=1");
      }
      return finish();
    }
    const bool hmm = op >= Op::hmm_initial;
    const auto d = (w - (hmm ? n : 0) - 4) / 2;
    for (std::size_t c = 0; c < n; ++c) {
      if (op == Op::mixture_weights || op == Op::hmm_initial)
        out.refs[c] = A(0, c * w);
      else {
        auto width = op == Op::hmm_transition ? n : d,
             offset = op == Op::hmm_transition
                          ? 1
                          : 1 + (hmm ? n : 0) +
                                ((op == Op::mixture_variances ||
                                  op == Op::hmm_variances)
                                     ? d
                                     : 0);
        for (std::size_t j = 0; j < width; ++j)
          out.refs[c * width + j] = A(0, c * w + offset + j);
      }
    }
    return out;
  }
  if (op == Op::sample_unique_indices) {
    auto k = bound(a[1], "sample count", 9007199254740991ULL);
    guard("AND(ISNUMBER(" + A(2) + "),ABS(" + A(2) + ")<=(2^53-1)," + A(2) +
              "=INT(" + A(2) + "))",
          "INVALID_PARAMETER");
    // Radix-65536 reduction keeps each intermediate below 2^47. A direct
    // MOD of a near-2^53 seed can round the quotient across an integer.
    auto magnitude = cell("ABS(" + A(2) + ")");
    F remainder = "0";
    for (int shift : {48, 32, 16, 0}) {
      const auto power = "2^" + std::to_string(shift);
      // Calc rounds INT/MOD near integer boundaries and approximate
      // subtraction can erase a unit near 2^53. Bias the quotient down;
      // its residual remains far from cancellation, then correct by one.
      auto digit = cell("MAX(0,INT(" + magnitude + "/" + power + "-0.25))");
      auto residual = cell(magnitude + "-" + digit + "*" + power);
      digit = cell(digit + "+IF(" + residual + ">=" + power + ",1,0)");
      magnitude =
          cell(iff(residual + ">=" + power, residual + "-" + power, residual));
      remainder = cell("MOD(" + remainder + "*65536+" + digit + ",2147483646)");
    }
    F state = cell(iff("AND(" + A(2) + "<0," + remainder + ">0)",
                       "2147483646-" + remainder, remainder) +
                   "+1");
    for (std::size_t i = 0; i < k; ++i) {
      state = cell("MOD(" + state + "*48271,2147483647)");
      auto row = cell("MOD(" + state + "," + std::to_string(n) + ")");
      for (std::size_t attempt = 0; attempt < i; ++attempt) {
        std::vector<F> used;
        for (std::size_t j = 0; j < i; ++j)
          used.push_back(row + "=" + out.refs[j]);
        row = cell(iff(any(used),
                       "MOD(" + row + "+1," + std::to_string(n) + ")", row));
      }
      out.refs[i] = row;
    }
    return finish();
  }
  if (op == Op::one_hot || op == Op::transition_counts ||
      op == Op::partition_mean || op == Op::scatter_rows) {
    bool category = op == Op::one_hot || op == Op::transition_counts;
    std::size_t id = category ? 0 : 1,
                k = bound(a[category ? 1 : 2], "output extent",
                          9007199254740991ULL),
                limit = op == Op::partition_mean ? n : k;
    std::vector<F> checks;
    for (auto &x : a[id].refs)
      checks.push_back("AND(ISNUMBER(" + x + ")," + x +
                       ">=" + (category ? "-1" : "0") + "," + x + "<" +
                       std::to_string(limit) + "," + x + "=INT(" + x + "))");
    guard(all(checks), "INDEX_OUT_OF_BOUNDS");
    if (!category) {
      checks.clear();
      for (std::size_t i = 0; i < a[id].size(); ++i)
        for (std::size_t j = 0; j < i; ++j)
          checks.push_back(a[id].at(i) + "<>" + a[id].at(j));
      guard(all(checks), "DUPLICATE_INDEX");
    }
    if (op == Op::one_hot) {
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t c = 0; c < k; ++c)
          out.refs[i * k + c] =
              cell(iff(A(0, i) + "=" + std::to_string(c), "1", "0"));
    } else if (op == Op::transition_counts) {
      for (std::size_t s = 0; s < k; ++s)
        for (std::size_t t = 0; t < k; ++t) {
          std::vector<F> values;
          for (std::size_t i = 1; i < n; ++i)
            values.push_back(
                cell(iff("AND(" + A(0, i - 1) + "=" + std::to_string(s) + "," +
                             A(0, i) + "=" + std::to_string(t) + ")",
                         "1", "0")));
          out.refs[s * k + t] = total(values);
        }
    } else if (op == Op::scatter_rows) {
      std::size_t width = 1;
      for (int i = 1; i < a[0].shape.rank; ++i)
        width *= a[0].shape.dim[i];
      for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < width; ++j) {
          F value = A(3);
          for (std::size_t r = 0; r < n; ++r)
            value = cell(iff(A(1, r) + "=" + std::to_string(i),
                             A(0, r * width + j), value));
          out.refs[i * width + j] = value;
        }
    } else {
      finite(a[0].refs);
      const auto d = a[0].shape.dim[1];
      std::size_t end = 0, remainder = 0;
      for (std::size_t c = 0; c < k; ++c) {
        auto start = end;
        end += n / k;
        remainder += n % k;
        if (remainder >= k) {
          ++end;
          remainder -= k;
        }
        for (std::size_t j = 0; j < d; ++j) {
          std::vector<F> col, selected;
          for (std::size_t r = 0; r < n; ++r)
            col.push_back(A(0, r * d + j));
          for (auto r = start; r < end; ++r)
            selected.push_back(choose(col, A(1, r)));
          out.refs[c * d + j] =
              cell(total(selected) + "/" + std::to_string(end - start));
        }
      }
    }
    return finish();
  }
  if (op >= Op::variance_axis && op <= Op::softmax_axis) {
    auto axis = bound(a[1], "axis", 2), count = a[0].shape.dim[axis];
    if (!out.size())
      return out;
    if (!count) {
      guard("FALSE", "INSUFFICIENT_SAMPLE");
      std::fill(out.refs.begin(), out.refs.end(), "0");
      return finish();
    }
    finite(a[0].refs);
    if (op == Op::variance_axis) {
      guard("AND(ISNUMBER(" + A(2) + ")," + A(2) + ">=0," + A(2) + "=INT(" +
                A(2) + "))",
            "INVALID_PARAMETER");
      guard(A(2) + "<" + std::to_string(count), "INSUFFICIENT_SAMPLE");
    }
    std::size_t outer = 1, inner = 1;
    for (std::size_t i = 0; i < axis; ++i)
      outer *= a[0].shape.dim[i];
    for (int i = int(axis) + 1; i < a[0].shape.rank; ++i)
      inner *= a[0].shape.dim[i];
    for (std::size_t i = 0; i < outer; ++i)
      for (std::size_t j = 0; j < inner; ++j) {
        std::vector<F> v;
        for (std::size_t c = 0; c < count; ++c)
          v.push_back(A(0, (i * count + c) * inner + j));
        if (op == Op::variance_axis) {
          auto mean = cell(total(v) + "/" + std::to_string(count));
          std::vector<F> squares;
          for (auto &x : v)
            squares.push_back(cell("(" + x + "-" + mean + ")^2"));
          out.refs[i * inner + j] = cell(
              total(squares) + "/(" + std::to_string(count) + "-" + A(2) + ")");
        } else if (op == Op::max_axis)
          out.refs[i * inner + j] = maximum(v);
        else if (op == Op::argmax_axis) {
          F best = v[0], index = "0";
          for (std::size_t c = 1; c < count; ++c) {
            index = cell(iff(v[c] + ">" + best, std::to_string(c), index));
            best = cell("MAX(" + best + "," + v[c] + ")");
          }
          out.refs[i * inner + j] = index;
        } else if (op == Op::logsumexp_axis)
          out.refs[i * inner + j] = lse(v);
        else {
          auto m = maximum(v);
          std::vector<F> ex;
          for (auto &x : v)
            ex.push_back(cell("EXP(" + x + "-" + m + ")"));
          auto s = total(ex);
          for (std::size_t c = 0; c < count; ++c)
            out.refs[(i * count + c) * inner + j] = cell(ex[c] + "/" + s);
        }
      }
    return finish();
  }
  const bool hmm = op == Op::gaussian_hmm_em,
             mix = op == Op::gaussian_mixture_em, fit = hmm || mix,
             inference = op == Op::hmm_filter || op == Op::hmm_smooth;
  const std::size_t mi = hmm   ? 3
                         : mix ? 2
                               : 1,
                    k = inference ? a[1].size() : a[mi].shape.dim[0],
                    d = inference ? 0 : a[0].shape.dim[1];
  if (!k || (!inference && !d) || (fit && !n)) {
    guard("FALSE", "INSUFFICIENT_SAMPLE");
    std::fill(out.refs.begin(), out.refs.end(), "0");
    return finish();
  }
  finite(a[0].refs);
  std::vector<F> means, vars, weights, transition;
  if (!inference) {
    means = a[mi].refs;
    vars = a[mi + 1].refs;
    finite(means);
    finite(vars);
    std::vector<F> valid;
    for (auto &x : vars)
      valid.push_back(x + ">=0");
    guard(all(valid), "DOMAIN_ERROR");
  }
  if (fit || inference) {
    weights = a[1].refs;
    probabilities(weights, 1);
  }
  if (hmm || inference) {
    transition = a[2].refs;
    probabilities(transition, k);
  }
  auto density = [&]() {
    std::vector<F> logs(k), em(n * k);
    for (std::size_t c = 0; c < k; ++c) {
      std::vector<F> terms;
      for (std::size_t j = 0; j < d; ++j)
        terms.push_back(cell("LN(MAX(" + vars[c * d + j] + ",1E-12))"));
      logs[c] = total(terms);
    }
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t c = 0; c < k; ++c) {
        F quadratic = "0";
        for (std::size_t j = 0; j < d; ++j)
          quadratic =
              cell(quadratic + "+(" + A(0, i * d + j) + "-" + means[c * d + j] +
                   ")^2/MAX(" + vars[c * d + j] + ",1E-12)");
        em[i * k + c] = cell(
            "-0.5*(" + num(double(d) * std::log(2 * 3.14159265358979323846)) +
            "+" + logs[c] + "+" + quadratic + ")");
      }
    return em;
  };
  auto fb = [&](const std::vector<F> &em, std::vector<F> &alpha,
                std::vector<F> &beta, std::vector<F> &log_a) {
    log_a.resize(k * k);
    for (std::size_t j = 0; j < k * k; ++j)
      log_a[j] = cell("LN(MAX(" + transition[j] + ",1E-12))");
    alpha.resize(n * k);
    beta.assign(n * k, "0");
    for (std::size_t c = 0; c < k; ++c)
      alpha[c] = cell("LN(MAX(" + weights[c] + ",1E-12))+" + em[c]);
    for (std::size_t i = 1; i < n; ++i)
      for (std::size_t t = 0; t < k; ++t) {
        std::vector<F> v;
        for (std::size_t s = 0; s < k; ++s)
          v.push_back(cell(alpha[(i - 1) * k + s] + "+" + log_a[s * k + t]));
        alpha[i * k + t] = cell(em[i * k + t] + "+" + lse(v));
      }
    for (std::size_t qv = 1; qv < n; ++qv) {
      auto i = n - 1 - qv;
      for (std::size_t s = 0; s < k; ++s) {
        std::vector<F> v;
        for (std::size_t t = 0; t < k; ++t)
          v.push_back(cell(log_a[s * k + t] + "+" + em[(i + 1) * k + t] + "+" +
                           beta[(i + 1) * k + t]));
        beta[i * k + s] = lse(v);
      }
    }
  };
  auto posterior = [&](const std::vector<F> &alpha,
                       const std::vector<F> &beta) {
    std::vector<F> gamma(n * k);
    for (std::size_t i = 0; i < n; ++i) {
      std::vector<F> v;
      for (std::size_t c = 0; c < k; ++c)
        v.push_back(cell(alpha[i * k + c] + "+" + beta[i * k + c]));
      auto z = lse(v);
      for (std::size_t c = 0; c < k; ++c)
        gamma[i * k + c] = cell("EXP(" + v[c] + "-" + z + ")");
    }
    return gamma;
  };
  if (op == Op::gaussian_log_density) {
    out.refs = density();
    return finish();
  }
  if (inference) {
    if (!n)
      return finish();
    if (op == Op::hmm_smooth) {
      std::vector<F> alpha, beta, la;
      fb(a[0].refs, alpha, beta, la);
      out.refs = posterior(alpha, beta);
    } else {
      guard("OR(" + A(3) + "=0," + A(3) + "=1)", "INVALID_PARAMETER");
      for (std::size_t i = 0; i < n; ++i) {
        std::vector<F> row(a[0].refs.begin() + i * k,
                           a[0].refs.begin() + (i + 1) * k),
            values;
        auto m = maximum(row);
        for (std::size_t t = 0; t < k; ++t) {
          std::vector<F> terms;
          for (std::size_t s = 0; s < k; ++s)
            terms.push_back(cell((i ? out.refs[(i - 1) * k + s] : weights[s]) +
                                 "*" + transition[s * k + t]));
          auto predicted = total(terms);
          if (!i)
            predicted = cell(iff(A(3) + "=0", weights[t], predicted));
          values.push_back(cell(predicted + "*EXP(" + row[t] + "-" + m + ")"));
        }
        auto sum = total(values);
        for (std::size_t t = 0; t < k; ++t)
          out.refs[i * k + t] = cell(values[t] + "/MAX(" + sum + ",1E-300)");
      }
    }
    return finish();
  }
  auto iterations = bound(a[hmm ? 5 : 4], "EM iterations", 10000);
  const auto tol = A(hmm ? 6 : 5);
  guard("AND(ISNUMBER(" + tol + ")," + tol + ">=0," + tol + "<=1E100)",
        "INVALID_PARAMETER");
  F done = "FALSE", completed = "0", last = "0";
  for (std::size_t iteration = 1; iteration <= iterations; ++iteration) {
    active = formula("NOT(" + done + ")");
    auto em = density();
    std::vector<F> gamma;
    F ll;
    auto next_weights = weights, next_transition = transition,
         next_means = means, next_vars = vars;
    if (mix) {
      gamma.resize(n * k);
      std::vector<F> likelihood;
      for (std::size_t i = 0; i < n; ++i) {
        std::vector<F> v;
        for (std::size_t c = 0; c < k; ++c)
          v.push_back(
              cell(em[i * k + c] + "+LN(MAX(" + weights[c] + ",1E-12))"));
        auto z = lse(v);
        likelihood.push_back(z);
        for (std::size_t c = 0; c < k; ++c)
          gamma[i * k + c] = cell("EXP(" + v[c] + "-" + z + ")");
      }
      ll = total(likelihood);
    } else {
      std::vector<F> alpha, beta, la;
      fb(em, alpha, beta, la);
      ll = lse(std::vector<F>(alpha.end() - k, alpha.end()));
      gamma = posterior(alpha, beta);
      std::vector<F> xi(k * k, "0");
      for (std::size_t i = 0; i + 1 < n; ++i) {
        std::vector<F> v;
        for (std::size_t s = 0; s < k; ++s)
          for (std::size_t t = 0; t < k; ++t)
            v.push_back(cell(alpha[i * k + s] + "+" + la[s * k + t] + "+" +
                             em[(i + 1) * k + t] + "+" +
                             beta[(i + 1) * k + t]));
        F z = v[0];
        for (std::size_t j = 1; j < v.size(); ++j)
          z = cell("MAX(" + z + "," + v[j] + ")+LN(1+EXP(-ABS(" + z + "-" +
                   v[j] + ")))");
        for (std::size_t j = 0; j < k * k; ++j)
          xi[j] = cell(xi[j] + "+EXP(" + v[j] + "-" + z + ")");
      }
      finite(xi);
      std::vector<F> initial;
      for (std::size_t c = 0; c < k; ++c)
        initial.push_back(cell("MAX(" + gamma[c] + ",1E-8)"));
      auto isum = total(initial);
      for (std::size_t c = 0; c < k; ++c)
        next_weights[c] = cell(initial[c] + "/" + isum);
      for (std::size_t s = 0; s < k; ++s) {
        std::vector<F> row;
        for (std::size_t t = 0; t < k; ++t)
          row.push_back(cell("MAX(" + xi[s * k + t] + ",1E-8)"));
        auto rsum = total(row);
        for (std::size_t t = 0; t < k; ++t)
          next_transition[s * k + t] = cell(row[t] + "/" + rsum);
      }
    }
    for (std::size_t c = 0; c < k; ++c) {
      std::vector<F> mass;
      for (std::size_t i = 0; i < n; ++i)
        mass.push_back(gamma[i * k + c]);
      auto effective = cell("MAX(" + total(mass) + ",1E-8)");
      if (mix)
        next_weights[c] = cell(effective + "/" + std::to_string(n));
      for (std::size_t j = 0; j < d; ++j) {
        std::vector<F> terms;
        for (std::size_t i = 0; i < n; ++i)
          terms.push_back(cell(gamma[i * k + c] + "*" + A(0, i * d + j)));
        next_means[c * d + j] = cell(total(terms) + "/" + effective);
        terms.clear();
        for (std::size_t i = 0; i < n; ++i)
          terms.push_back(cell(gamma[i * k + c] + "*(" + A(0, i * d + j) + "-" +
                               next_means[c * d + j] + ")^2"));
        const auto raw_variance = cell(total(terms) + "/" + effective);
        finite({raw_variance});
        next_vars[c * d + j] = cell("MAX(" + raw_variance + ",1E-5)");
      }
    }
    finite(next_weights);
    finite(next_transition);
    finite(next_means);
    finite(next_vars);
    finite({ll});
    auto freeze = [&](std::vector<F> &old, const std::vector<F> &next) {
      for (std::size_t i = 0; i < old.size(); ++i)
        old[i] = formula(iff(active, next[i], old[i]));
    };
    freeze(weights, next_weights);
    freeze(transition, next_transition);
    freeze(means, next_means);
    freeze(vars, next_vars);
    auto hit = iteration == 1 ? F("FALSE")
                              : cell("ABS(" + ll + "-" + last + ")<" + tol);
    completed = formula(iff(active, std::to_string(iteration), completed));
    last = formula(iff(active, ll, last));
    done = formula("OR(" + done + "," + hit + ")");
  }
  active = "TRUE";
  const auto width = out.shape.dim[1], base = 1 + (hmm ? k : 0);
  for (std::size_t c = 0; c < k; ++c) {
    out.refs[c * width] = weights[c];
    if (hmm)
      for (std::size_t t = 0; t < k; ++t)
        out.refs[c * width + 1 + t] = transition[c * k + t];
    for (std::size_t j = 0; j < d; ++j) {
      out.refs[c * width + base + j] = means[c * d + j];
      out.refs[c * width + base + d + j] = vars[c * d + j];
    }
    out.refs[c * width + width - 3] = completed;
    out.refs[c * width + width - 2] = last;
    out.refs[c * width + width - 1] = formula(iff(done, "1", "0"));
  }
  return finish();
}
} // namespace calmetrics_engine::excel
