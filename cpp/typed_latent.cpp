#include "calmetrics_engine/typed_ir.hpp"
#include <cmath>

namespace calmetrics_engine::typed {
namespace {
void check(bool ok, const char *why) {
  if (!ok)
    throw Error(why);
}
ValueType shape(ValueType t, std::vector<std::string> axes,
                std::vector<std::string> dims) {
  t.axes = std::move(axes);
  t.shape = std::move(dims);
  t.record_tag.clear();
  t.fields.clear();
  t.kind = t.rank() == 3   ? ValueKind::tensor
           : t.rank() == 2 ? ValueKind::matrix
           : t.rank() == 1
               ? (t.axes[0] == "time" ? ValueKind::series : ValueKind::vector)
           : t.dtype == DType::float64 ? ValueKind::scalar
                                       : ValueKind::value;
  t.validate();
  return t;
}
const std::vector<std::string> mixture_fields{"weights",        "means",
                                              "variances",      "iterations",
                                              "log_likelihood", "converged"};
const std::vector<std::string> hmm_fields{
    "initial",    "transition",     "means",    "variances",
    "iterations", "log_likelihood", "converged"};
} // namespace
ValueType infer_latent(const ops::Spec &spec, const std::vector<ValueType> &v,
                       const std::vector<double> &constants) {
  using O = ops::Op;
  const auto op = spec.op;
  auto out = v[0];
  auto scalar = [&](std::size_t i) {
    check(v[i].is_scalar() && v[i].is_numeric() &&
              (v[i].semantic_dimension == "dimensionless" ||
               v[i].semantic_dimension == "count"),
          "TYPE_MISMATCH: expected numeric scalar option");
  };
  auto literal = [&](std::size_t i, std::size_t low, std::size_t high) {
    scalar(i);
    check(i < constants.size() && std::isfinite(constants[i]) &&
              constants[i] >= double(low) && constants[i] <= double(high) &&
              constants[i] == std::floor(constants[i]),
          "CONSTANT_LAYOUT_REQUIRED: layout requires integer literal");
    return std::size_t(constants[i]);
  };
  auto numeric = [&](std::size_t i, std::size_t low, std::size_t high) {
    check(v[i].is_numeric() && v[i].rank() >= low && v[i].rank() <= high,
          "TYPE_MISMATCH: invalid numeric array");
  };
  auto codes = [&](std::size_t i) {
    check(v[i].rank() == 1 && v[i].dtype == DType::int64 &&
              v[i].record_tag.empty(),
          "TYPE_MISMATCH: expected int64 vector");
  };
  auto dim = [&](std::size_t i, std::size_t ai, std::size_t j, std::size_t aj) {
    check(v[i].shape[ai] == v[j].shape[aj],
          "SHAPE_MISMATCH: dimensions differ");
    check(v[i].axes[ai] == v[j].axes[aj], "AXIS_MISMATCH: axes differ");
  };
  if (op >= O::mixture_weights && op <= O::hmm_variances) {
    const bool hmm = out.record_tag == "gaussian_hmm",
               mix = out.record_tag == "gaussian_mixture";
    check(out.kind == ValueKind::matrix && out.dtype == DType::float64 &&
              (hmm || mix) && out.fields == (hmm ? hmm_fields : mixture_fields),
          "TYPE_MISMATCH: expected fitted Gaussian model");
    if (op >= O::em_iterations && op <= O::em_converged) {
      if (op == O::em_converged)
        return ValueType::mask();
      out.dtype = op == O::em_iterations  ? DType::int64
                  : op == O::em_converged ? DType::boolean
                                          : DType::float64;
      return shape(out, {}, {})
          .with_semantics(op == O::em_iterations ? "count" : "dimensionless");
    }
    check((op >= O::hmm_initial) == hmm, "TYPE_MISMATCH: model family differs");
    // Packed storage is an internal nominal type. Preserve the original
    // feature extent in its canonical symbolic expression, never infer it
    // from an unrelated runtime matrix supplied by a caller.
    const std::string prefix = "(2*(",
                      suffix = ")" + (hmm ? "+" + out.shape[0] : "") + "+4)";
    check(out.shape[1].rfind(prefix, 0) == 0 &&
              out.shape[1].size() > prefix.size() + suffix.size() &&
              out.shape[1].substr(out.shape[1].size() - suffix.size()) ==
                  suffix,
          "TYPE_MISMATCH: invalid model layout");
    const auto d = out.shape[1].substr(
        prefix.size(), out.shape[1].size() - prefix.size() - suffix.size());
    if (op == O::mixture_weights || op == O::hmm_initial)
      return shape(out, {out.axes[0]}, {out.shape[0]})
          .with_semantics("dimensionless");
    if (op == O::hmm_transition)
      return shape(out, {out.axes[0], out.axes[0]},
                   {out.shape[0], out.shape[0]})
          .with_semantics("dimensionless");
    return shape(out, out.axes, {out.shape[0], d})
        .with_semantics("dimensionless");
  }
  if (op >= O::variance_axis && op <= O::softmax_axis) {
    numeric(0, 1, 3);
    auto axis = literal(1, 0, out.rank() - 1);
    if (v.size() > 2)
      scalar(2);
    if (op == O::softmax_axis)
      return out.with_semantics("dimensionless");
    auto axes = out.axes, dims = out.shape;
    axes.erase(axes.begin() + axis);
    dims.erase(dims.begin() + axis);
    if (op == O::argmax_axis)
      out.dtype = DType::int64;
    out = shape(out, axes, dims);
    if (op == O::argmax_axis)
      return out.with_semantics("index");
    if (op == O::logsumexp_axis)
      return out.with_semantics("dimensionless");
    if (op == O::variance_axis && out.semantic_dimension != "dimensionless")
      return out.with_semantics("squared:" + out.semantic_dimension);
    return out;
  }
  if (op == O::transition_counts || op == O::one_hot) {
    codes(0);
    auto k = std::to_string(literal(1, 1, 9007199254740991ULL));
    out.dtype = DType::float64;
    return shape(out,
                 op == O::one_hot
                     ? std::vector<std::string>{out.axes[0], "component"}
                     : std::vector<std::string>{"component", "component"},
                 {op == O::one_hot ? out.shape[0] : k, k})
        .with_semantics(op == O::one_hot ? "dimensionless" : "count");
  }
  if (op == O::sample_unique_indices) {
    numeric(0, 1, 3);
    auto k = literal(1, 0, 9007199254740991ULL);
    if (v.size() > 2)
      scalar(2);
    out.dtype = DType::int64;
    return shape(out, {out.axes[0]}, {std::to_string(k)})
        .with_semantics("index");
  }
  if (op == O::partition_mean || op == O::scatter_rows) {
    numeric(0, op == O::partition_mean ? 2 : 1,
            op == O::partition_mean ? 2 : 3);
    codes(1);
    dim(0, 0, 1, 0);
    auto k = literal(2, op == O::partition_mean ? 1 : 0, 9007199254740991ULL);
    if (v.size() > 3)
      scalar(3);
    auto axes = out.axes, dims = out.shape;
    dims[0] = std::to_string(k);
    if (op == O::partition_mean)
      axes[0] = "component";
    return shape(out, axes, dims);
  }
  numeric(0, 2, 2);
  const bool hmm = op == O::gaussian_hmm_em, mix = op == O::gaussian_mixture_em,
             inference = op == O::hmm_filter || op == O::hmm_smooth;
  if (hmm || mix || inference)
    numeric(1, 1, 1);
  if (hmm || inference) {
    numeric(2, 2, 2);
    dim(1, 0, 2, 0);
    dim(1, 0, 2, 1);
  }
  if (inference) {
    dim(0, 1, 1, 0);
    if (v.size() > 3)
      scalar(3);
    return out.with_semantics("dimensionless");
  }
  const std::size_t m = hmm ? 3 : mix ? 2 : 1;
  numeric(m, 2, 2);
  numeric(m + 1, 2, 2);
  dim(m, 0, m + 1, 0);
  dim(m, 1, m + 1, 1);
  dim(0, 1, m, 1);
  if (mix || hmm)
    dim(1, 0, m, 0);
  if (!mix && !hmm)
    return shape(out, {out.axes[0], v[m].axes[0]},
                 {out.shape[0], v[m].shape[0]})
        .with_semantics("dimensionless");
  for (std::size_t i = m + 2; i < v.size(); ++i)
    scalar(i);
  out = shape(out, v[m].axes,
              {v[m].shape[0], "(2*(" + out.shape[1] + ")" +
                                  (hmm ? "+" + v[m].shape[0] : "") + "+4)"})
            .with_semantics("dimensionless");
  out.record_tag = hmm ? "gaussian_hmm" : "gaussian_mixture";
  out.fields = hmm ? hmm_fields : mixture_fields;
  return out;
}
} // namespace calmetrics_engine::typed
