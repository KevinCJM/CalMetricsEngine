#include "excel_internal.hpp"
#include "operators/axis_primitives.hpp"
#include <limits>

namespace calmetrics_engine::excel {
namespace {
using N = std::size_t;

N checked_add(N a, N b) {
  if (b > std::numeric_limits<N>::max() - a)
    throw Error("OVER_BUDGET: axis recipe integer addition overflow");
  return a + b;
}
N checked_mul(N a, N b) {
  if (b && a > std::numeric_limits<N>::max() / b)
    throw Error("OVER_BUDGET: axis recipe integer multiplication overflow");
  return a * b;
}
N sum_helpers(N n) {
  N cells = 0;
  while (n > 1) {
    n = n / 32 + (n % 32 != 0);
    cells = checked_add(cells, n);
  }
  return cells;
}
struct AxisGeometry {
  N before = 1, length = 0, after = 1;
};
AxisGeometry geometry(const ops::Shape &shape, N axis) {
  AxisGeometry result;
  result.length = shape.dim[axis];
  for (N d = 0; d < axis; ++d)
    result.before = checked_mul(result.before, shape.dim[d]);
  for (N d = axis + 1; d < N(shape.rank); ++d)
    result.after = checked_mul(result.after, shape.dim[d]);
  return result;
}
F representable(const F &condition, const F &expression) {
  return iff(condition,
             "IFERROR(" + expression + "," + err("UNREPRESENTABLE_VALUE") +
                 ")",
             q("NaN"));
}
void check_intermediate(double value) {
  const auto magnitude = std::abs(value);
  if (std::isinf(value) || magnitude > max_reproduction_number ||
      (std::isfinite(value) && value != 0 &&
       magnitude < std::numeric_limits<double>::min()))
    throw Error("UNREPRESENTABLE_VALUE: quantiles_axis intermediate exceeds "
                "Excel numeric domain");
}
} // namespace

Array Builder::axis_path_recipe(Op op, const std::vector<Array> &a, Array out) {
  const bool quantiles = op == Op::quantiles_axis;
  const auto axis = bound(a[quantiles ? 2 : 1], "axis", 2);
  if (!quantiles) {
    // Geometry/configuration has already been checked. Empty output consumes
    // neither a seed nor a source payload, and creates no recurrence cells.
    if (out.refs.empty()) return out;
    const auto g = geometry(a[0].shape, axis);
    cost({checked_add(out.size(), checked_mul(g.before, g.after)), 16});
    for (N outer = 0; outer < g.before; ++outer)
      for (N inner = 0; inner < g.after; ++inner) {
        F prior = formula(a[2].at());
        for (N t = 0; t < g.length; ++t) {
          const auto index = (outer * g.length + t) * g.after + inner;
          const auto current = a[0].at(index);
          if (op == Op::cumulative_product_axis)
            prior = formula(representable(
                "AND(" + fin(prior) + "," + fin(current) + ")",
                prior + "*" + current));
          else
            // Both numeric guards precede the comparison. Current NaN keeps
            // state; a NaN seed stays NaN, including when later input is finite.
            prior = formula(iff(fin(current),
                                iff(fin(prior),
                                    iff(current + ">" + prior, current, prior),
                                    prior),
                                prior));
          out.refs[index] = prior;
        }
      }
    return out;
  }

  const N count = a[1].size();
  N recipe = checked_add(checked_add(checked_mul(2, count), sum_helpers(count)),
                         1);
  AxisGeometry g;
  if (!out.refs.empty()) {
    g = geometry(a[0].shape, axis);
    const auto ranks = checked_mul(
        g.length, checked_add(g.length, sum_helpers(g.length)));
    const auto picks = checked_mul(count, checked_add(checked_mul(2, g.length), 5));
    recipe = checked_add(recipe, checked_mul(checked_mul(g.before, g.after),
                                             checked_add(ranks, picks)));
  }
  cost({recipe, 16});
  std::vector<F> valid;
  valid.reserve(count);
  for (N i = 0; i < count; ++i) {
    const auto probability = a[1].at(i);
    valid.push_back(formula(iff(fin(probability),
                               "AND(" + probability + ">=0," + probability +
                                   "<=1)",
                               "FALSE")));
  }
  const auto invalid = formula(iff(all(valid), q(""), err("INVALID_PARAMETER")));
  // This guard is retained even with Q0 or an empty outer axis. The generic
  // operation wrapper carries failures; each real result also exposes it.
  out.failure = invalid;
  if (out.refs.empty()) return out;
  if (g.length == 0) {
    // A nonempty distribution output needs a sample. Do not form N-1 or
    // interpolation indices; edited probability errors still take precedence.
    for (auto &value : out.refs)
      value = formula(iff(invalid + "<>\"\"", invalid,
                          err("INSUFFICIENT_SAMPLE")));
    return out;
  }

  for (N outer = 0; outer < g.before; ++outer)
    for (N inner = 0; inner < g.after; ++inner) {
      std::vector<F> sources, ranks;
      sources.reserve(g.length);
      ranks.reserve(g.length);
      for (N t = 0; t < g.length; ++t)
        sources.push_back(a[0].at((outer * g.length + t) * g.after + inner));
      // One transparent stable NaN-last rank network per line, shared by all Q.
      for (N target = 0; target < g.length; ++target) {
        std::vector<F> precedes;
        precedes.reserve(g.length);
        for (N source = 0; source < g.length; ++source) {
          const F earlier = source < target ? "TRUE" : "FALSE";
          const auto comparison = iff(
              fin(sources[target]),
              iff(fin(sources[source]),
                  "OR(" + sources[source] + "<" + sources[target] + ",AND(" +
                      sources[source] + "=" + sources[target] + "," + earlier +
                      "))",
                  "FALSE"),
              "OR(" + fin(sources[source]) + "," + earlier + ")");
          precedes.push_back(formula(iff(comparison, "1", "0")));
        }
        ranks.push_back(sum(precedes));
      }
      for (N probability_index = 0; probability_index < count; ++probability_index) {
        const auto probability = a[1].at(probability_index);
        // Invalid edited q never enters arithmetic/INT. Its public result is
        // INVALID_PARAMETER, while inactive helpers remain safe to calculate.
        const auto position = formula(iff(valid[probability_index],
            std::to_string(g.length - 1) + "*" + probability, "0"));
        auto pick = [&](const F &selected_rank) {
          F value = q("NaN");
          for (N source = 0; source < g.length; ++source)
            value = formula(iff(ranks[source] + "=" + selected_rank,
                                sources[source], value));
          return value;
        };
        const auto lower = pick("INT(" + position + ")");
        const auto upper = pick("ROUNDUP(" + position + ",0)");
        const auto fraction = formula(position + "-INT(" + position + ")");
        const auto direct = fraction + "=0";
        // Direct ranks (including N1) must not evaluate 0*Inf/NaN or execute a
        // weighted interpolation. Each non-direct product is an explicit cell.
        const auto low_product = formula(iff(direct, "0",
            representable(fin(lower), lower + "*(1-" + fraction + ")")));
        const auto high_product = formula(iff(direct, "0",
            representable(fin(upper), upper + "*" + fraction)));
        const auto value = iff(direct, lower,
            iff("LEFT(" + low_product + ",6)=\"ERROR:\"", low_product,
                iff("LEFT(" + high_product + ",6)=\"ERROR:\"", high_product,
                    representable("AND(" + fin(low_product) + "," +
                                      fin(high_product) + ")",
                                  low_product + "+" + high_product))));
        out.refs[(outer * count + probability_index) * g.after + inner] =
            formula(iff(invalid + "<>\"\"", invalid, value));
      }
    }
  return out;
}

void axis_path_validate(Op op, const ops::Value *given, std::size_t count) {
  const auto prepared = ops::prepare(
      ops::lookup(static_cast<std::uint16_t>(op)), given, count);
  // Every prefix state is an output and is checked by the existing export
  // observer. No duplicate recurrence or retained numeric state is necessary.
  if (op != Op::quantiles_axis || prepared.output_shape.size() == 0) return;
  const auto &values = prepared.args[0], &probabilities = prepared.args[1];
  const auto axis = static_cast<N>(prepared.args[2].scalar);
  const auto g = geometry(values.shape, axis);
  ops::require(g.length > 0, "INSUFFICIENT_SAMPLE");
  std::vector<double> sorted(g.length);
  for (N outer = 0; outer < g.before; ++outer)
    for (N inner = 0; inner < g.after; ++inner) {
      for (N t = 0; t < g.length; ++t)
        sorted[t] = values.f((outer * g.length + t) * g.after + inner);
      std::sort(sorted.begin(), sorted.end(), ops::axis_path_detail::NaNLast{});
      for (N i = 0; i < probabilities.size(); ++i) {
        const double probability = probabilities.f(i);
        if (g.length == 1 || probability == 0.0 || probability == 1.0) continue;
        const double position = static_cast<double>(g.length - 1) * probability;
        const auto lower = static_cast<N>(std::floor(position));
        const auto upper = static_cast<N>(std::ceil(position));
        if (lower == upper) continue;
        const double fraction = position - static_cast<double>(lower);
        // Check the actual consumed terms even if the other term is NaN and
        // canonical final NaN would otherwise conceal Excel underflow/overflow.
        check_intermediate(sorted[lower] * (1.0 - fraction));
        check_intermediate(sorted[upper] * fraction);
        check_intermediate(ops::axis_path_detail::sorted_quantile(
            sorted.data(), sorted.size(), probability));
      }
    }
}
} // namespace calmetrics_engine::excel
