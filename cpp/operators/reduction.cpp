#include "calmetrics_engine/operators.hpp"
#include <algorithm>
#include <cmath>

namespace calmetrics_engine::ops {
namespace {
Op unmasked_op(Op op) {
  switch (op) {
  case Op::sum_where:
    return Op::sum;
  case Op::mean_where:
    return Op::mean;
  case Op::variance_where:
    return Op::variance;
  case Op::std_where:
    return Op::std;
  case Op::min_where:
    return Op::min_value;
  case Op::max_where:
    return Op::max_value;
  case Op::median_where:
    return Op::median;
  case Op::quantile_where:
    return Op::quantile;
  default:
    return op;
  }
}
Op axis_op(Op op) {
  const int index = static_cast<int>(op) -
                    (op <= Op::max_time ? static_cast<int>(Op::sum_time)
                                        : static_cast<int>(Op::sum_asset));
  constexpr Op mapping[]{Op::sum, Op::mean,      Op::product,  Op::variance,
                         Op::std, Op::min_value, Op::max_value};
  require(index >= 0 && index < 7, "INVALID_AXIS_REDUCTION");
  return mapping[index];
}
double ordered_stat(Op op, const Value &x, const Value *mask,
                    double probability, Workspace &work) {
  auto *ordered = work.doubles.data();
  std::size_t count = 0;
  for (std::size_t i = 0; i < x.size(); ++i)
    if (!mask || mask->u(i))
      ordered[count++] = x.f(i);
  require(count > 0, "INSUFFICIENT_SAMPLE");
  // A contiguous numeric scratch copy is deliberate algorithm workspace.
  // It keeps the caller input read-only while avoiding cache-hostile indirect
  // comparisons through an index array.
  std::sort(ordered, ordered + count, [](double a, double b) {
    if (std::isnan(a))
      return false;
    if (std::isnan(b))
      return true;
    return a < b;
  });
  if (op == Op::median) {
    const auto middle = count / 2;
    return count % 2 ? ordered[middle]
                     : (ordered[middle - 1] + ordered[middle]) * 0.5;
  }
  const double position = static_cast<double>(count - 1) * probability;
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction;
}
} // namespace

double reduce_value(Op raw_op, const Value &x, const Value *mask,
                    std::size_t ddof, double probability, Workspace &work) {
  const Op op = unmasked_op(raw_op);
  if (op == Op::median || op == Op::quantile)
    return ordered_stat(op, x, mask, probability, work);
  if (op == Op::product && !mask)
    return product_value(x, false);
  require(x.size() > 0, "INSUFFICIENT_SAMPLE");
  if (!mask && (op == Op::variance || op == Op::std))
    require(ddof < x.size(), "INVALID_PARAMETER");
  std::size_t count = 0, best_index = 0;
  double mean = 0.0, m2 = 0.0, total = 0.0, square_total = 0.0;
  double best = 0.0;
  if (mask && op == Op::min_value)
    best = std::numeric_limits<double>::infinity();
  if (mask && op == Op::max_value)
    best = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (mask && !mask->u(i))
      continue;
    const double value = x.f(i);
    if (!mask && count == 0) {
      best = value;
      best_index = i;
    }
    ++count;
    if (op == Op::min_value || op == Op::argmin) {
      if (value < best) {
        best = value;
        best_index = i;
      }
    } else if (op == Op::max_value || op == Op::argmax) {
      if (value > best) {
        best = value;
        best_index = i;
      }
    } else {
      total += value;
      square_total += value * value;
      const double delta = value - mean;
      mean += delta / static_cast<double>(count);
      m2 += delta * (value - mean);
    }
  }
  require(count > 0, "INSUFFICIENT_SAMPLE");
  switch (op) {
  case Op::min_value:
  case Op::max_value:
    return best;
  case Op::argmin:
  case Op::argmax:
    return static_cast<double>(best_index);
  case Op::sum:
    return total;
  case Op::mean:
    return mean;
  case Op::variance:
  case Op::std: {
    require(count > ddof, mask ? "INSUFFICIENT_SAMPLE" : "INVALID_PARAMETER");
    const double variance = m2 / static_cast<double>(count - ddof);
    return op == Op::std ? std::sqrt(variance) : variance;
  }
  case Op::root_mean_square:
    return std::sqrt(square_total / static_cast<double>(count));
  default:
    break;
  }
  const double n = static_cast<double>(count);
  double accum = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (mask && !mask->u(i))
      continue;
    const double d = x.f(i) - mean;
    if (op == Op::mean_absolute_deviation)
      accum += std::abs(d);
    else if (op == Op::skewness)
      accum += d * d * d;
    else
      accum += (d * d) * (d * d);
  }
  if (op == Op::mean_absolute_deviation)
    return accum / n;
  if (op == Op::skewness) {
    require(count >= 3 && !(m2 <= 0), "INSUFFICIENT_SAMPLE");
    return std::sqrt(n * (n - 1)) / (n - 2) * (accum / n) /
           std::pow(m2 / n, 1.5);
  }
  require(op == Op::excess_kurtosis, "INVALID_REDUCTION");
  require(count >= 4 && !(m2 <= 0), "INSUFFICIENT_SAMPLE");
  const double second = m2 / n, excess = accum / n / (second * second) - 3.0;
  return (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * excess + 6.0);
}

void reduction(const Prepared &p, Output &out, Workspace &work, Audit &audit) {
  const auto op = p.spec->op;
  const auto &x = p.args[0];
  if (op == Op::distinct_count) {
    auto &indices = work.indices;
    std::size_t count = 0;
    for (std::size_t i = 0; i < x.size(); ++i)
      if (p.args[1].u(i))
        indices[count++] = i;
    std::sort(indices.begin(), indices.begin() + count,
              [&](std::size_t left, std::size_t right) { return x.i(left) < x.i(right); });
    std::int64_t distinct = count > 0 ? 1 : 0;
    for (std::size_t i = 1; i < count; ++i)
      distinct += x.i(indices[i]) != x.i(indices[i - 1]);
    // Counts are numeric quantities, like count_true, rather than category IDs.
    require(distinct <= 9007199254740992LL, "INEXACT_CARDINALITY");
    out.set(0, static_cast<double>(distinct));
    return;
  }
  if (op == Op::count_true || op == Op::max_consecutive_true) {
    std::size_t count = 0, longest = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
      if (x.u(i))
        ++count;
      else if (op == Op::max_consecutive_true)
        count = 0;
      longest = std::max(longest, count);
    }
    out.set(0, static_cast<double>(op == Op::count_true ? count : longest));
    return;
  }
  if (op >= Op::sum_time && op <= Op::max_asset) {
    if (op >= Op::sum_asset)
      require(x.shape.dim[0] && x.shape.dim[1], "INSUFFICIENT_SAMPLE");
    const auto base = axis_op(op);
    for (std::size_t i = 0; i < out.shape.size(); ++i) {
      const auto view = op <= Op::max_time ? x.column(i) : x.row(i);
      out.set(i, reduce_value(base, view, nullptr, 1, 0.5, work));
    }
    return;
  }
  const bool masked = op >= Op::sum_where && op <= Op::quantile_where;
  const auto ddof = (op == Op::variance || op == Op::std)
                        ? static_cast<std::size_t>(p.args[1].scalar)
                        : 1;
  const double q = op == Op::quantile         ? p.args[1].scalar
                   : op == Op::quantile_where ? p.args[2].scalar
                                              : 0.5;
  if (op == Op::median || op == Op::quantile || op == Op::median_where ||
      op == Op::quantile_where) {
    std::size_t copied = x.size();
    if (masked) {
      copied = 0;
      for (std::size_t i = 0; i < x.size(); ++i)
        copied += p.args[1].u(i) != 0;
    }
    audit.algorithm_copy_bytes = copied * sizeof(double);
  }
  out.set(0, reduce_value(op, x, masked ? &p.args[1] : nullptr, ddof, q, work));
}
} // namespace calmetrics_engine::ops
