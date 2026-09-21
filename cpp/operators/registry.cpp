#include "calmetrics_engine/operators.hpp"
#include <algorithm>
#include <cmath>

namespace calmetrics_engine::ops {

Value Value::row(std::size_t i) const {
  Value v = *this;
  v.shape = vector_shape(shape.dim[1]);
  if (v.size()) {
    const auto offset = static_cast<std::ptrdiff_t>(i) * stride[0];
    if (kind == Kind::integer)
      v.data = static_cast<const std::int64_t *>(data) + offset;
    else if (kind == Kind::mask)
      v.data = static_cast<const std::uint8_t *>(data) + offset;
    else
      v.data = static_cast<const double *>(data) + offset;
  }
  v.stride = {stride[1], 1};
  return v;
}
Value Value::column(std::size_t i) const {
  Value v = *this;
  v.shape = vector_shape(shape.dim[0]);
  if (v.size()) {
    const auto offset = static_cast<std::ptrdiff_t>(i) * stride[1];
    if (kind == Kind::integer)
      v.data = static_cast<const std::int64_t *>(data) + offset;
    else if (kind == Kind::mask)
      v.data = static_cast<const std::uint8_t *>(data) + offset;
    else
      v.data = static_cast<const double *>(data) + offset;
  }
  v.stride = {stride[0], 1};
  return v;
}

const std::array<Spec, operator_count> &registry() {
  static constexpr std::array<Spec, operator_count> table{{
#define OP(id, name, family, lo, hi, params)                                   \
  {Op::name, #name, Family::family, lo, hi, params},
#include "calmetrics_engine/operators.def"
#undef OP
  }};
  return table;
}
const Spec &lookup(std::string_view name) {
  for (const auto &spec : registry())
    if (name == spec.name)
      return spec;
  throw Error("UNKNOWN_OPERATOR");
}
const Spec &lookup(std::uint16_t id) {
  require(id > 0 && id <= operator_count, "UNKNOWN_OPCODE");
  return registry()[id - 1];
}
const char *family_name(Family f) {
  switch (f) {
#define F(name)                                                                \
  case Family::name:                                                           \
    return #name;
    F(elementwise)
    F(reduction)
    F(sequence) F(rolling) F(matrix) F(regression) F(state) F(composite)
#undef F
  }
  throw Error("UNKNOWN_FAMILY");
}

namespace {
bool is_between(Op o, Op first, Op last) {
  return static_cast<unsigned>(o) >= static_cast<unsigned>(first) &&
         static_cast<unsigned>(o) <= static_cast<unsigned>(last);
}
void numeric(const Value &v, int min_rank = 0, int max_rank = 2) {
  require(v.kind == Kind::number, "DTYPE_MISMATCH");
  require(v.shape.rank >= min_rank && v.shape.rank <= max_rank,
          "RANK_MISMATCH");
}
void mask(const Value &v, int min_rank = 0, int max_rank = 2) {
  require(v.kind == Kind::mask, "DTYPE_MISMATCH");
  require(v.shape.rank >= min_rank && v.shape.rank <= max_rank,
          "RANK_MISMATCH");
}
void integer_values(const Value &v, int min_rank = 1, int max_rank = 1) {
  require(v.kind == Kind::integer, "DTYPE_MISMATCH");
  require(v.shape.rank >= min_rank && v.shape.rank <= max_rank,
          "RANK_MISMATCH");
}
Shape broadcast(const Shape &a, const Shape &b) {
  if (a.rank == 0)
    return b;
  if (b.rank == 0)
    return a;
  require(a == b, "SHAPE_MISMATCH");
  return a;
}
std::size_t integer(const Value &v, bool allow_zero) {
  numeric(v, 0, 0);
  const double x = v.scalar;
  require(std::isfinite(x) && x == std::floor(x) && x >= (allow_zero ? 0 : 1) &&
              x < static_cast<double>(PTRDIFF_MAX),
          "INVALID_PARAMETER");
  return static_cast<std::size_t>(x);
}
void elementwise_shape(Prepared &p) {
  const Op o = p.spec->op;
  auto &a = p.args;
  if (o == Op::finite_mask) {
    numeric(a[0], 1, 1);
    p.output_kind = Kind::mask;
    p.output_shape = a[0].shape;
    return;
  }
  if (o == Op::logical_and || o == Op::logical_or || o == Op::logical_not) {
    mask(a[0]);
    if (o != Op::logical_not) {
      mask(a[1]);
      require(a[0].shape == a[1].shape, "SHAPE_MISMATCH");
    }
    p.output_kind = Kind::mask;
    p.output_shape = a[0].shape;
    return;
  }
  if (o == Op::where) {
    mask(a[0]);
    numeric(a[1]);
    numeric(a[2]);
    p.output_shape = broadcast(a[0].shape, broadcast(a[1].shape, a[2].shape));
    return;
  }
  numeric(a[0]);
  p.output_shape = a[0].shape;
  if (o == Op::divide_or_default) {
    numeric(a[0], 1, 1);
    numeric(a[1], 1, 1);
    numeric(a[2], 0, 0);
    require(a[0].shape == a[1].shape && std::isfinite(a[2].scalar),
            "INVALID_PARAMETER");
    return;
  }
  if (o == Op::clip) {
    numeric(a[1], 0, 0);
    numeric(a[2], 0, 0);
    return;
  }
  if (p.count == 2) {
    numeric(a[1]);
    p.output_shape = broadcast(a[0].shape, a[1].shape);
  }
  if (is_between(o, Op::equal, Op::greater_equal))
    p.output_kind = Kind::mask;
}
void reduction_shape(Prepared &p) {
  const auto o = p.spec->op;
  auto &a = p.args;
  if (o == Op::distinct_count) {
    integer_values(a[0]);
    if (p.count == 1) {
      a[1] = Value::number(1);
      a[1].kind = Kind::mask;
    } else {
      mask(a[1], 1, 1);
      require(a[0].shape == a[1].shape, "SHAPE_MISMATCH");
    }
    p.scratch_indices = a[0].size();
    return;
  }
  if (o == Op::count_true || o == Op::max_consecutive_true) {
    mask(a[0], 1, o == Op::count_true ? 2 : 1);
    return;
  }
  numeric(a[0], 1, 2);
  if (is_between(o, Op::sum_time, Op::max_asset)) {
    numeric(a[0], 2, 2);
    p.output_shape = vector_shape(a[0].shape.dim[o <= Op::max_time ? 1 : 0]);
    return;
  }
  if (is_between(o, Op::sum_where, Op::quantile_where)) {
    mask(a[1], 1, 2);
    require(a[0].shape == a[1].shape, "SHAPE_MISMATCH");
  }
  if (o == Op::variance || o == Op::std) {
    if (p.count == 1)
      a[1] = Value::number(1);
    integer(a[1], true);
  }
  if (o == Op::quantile || o == Op::quantile_where) {
    const auto &probability = a[o == Op::quantile ? 1 : 2];
    numeric(probability, 0, 0);
    require(std::isfinite(probability.scalar) && probability.scalar > 0 &&
                probability.scalar < 1,
            "INVALID_PARAMETER");
  }
  if (o == Op::median || o == Op::quantile || o == Op::median_where ||
      o == Op::quantile_where)
    p.scratch_doubles = a[0].size();
}
void sequence_shape(Prepared &p) {
  auto &a = p.args;
  const auto o = p.spec->op;
  if (o == Op::argsort) {
    require(a[0].kind == Kind::number || a[0].kind == Kind::integer,
            "DTYPE_MISMATCH");
    require(a[0].shape.rank == 1, "RANK_MISMATCH");
    p.output_shape = a[0].shape;
    p.output_kind = Kind::integer;
    p.scratch_indices = a[0].size();
    return;
  }
  if (o == Op::gather) {
    require(a[0].kind == Kind::number || a[0].kind == Kind::mask ||
                a[0].kind == Kind::integer,
            "DTYPE_MISMATCH");
    require(a[0].shape.rank == 1, "RANK_MISMATCH");
    integer_values(a[1]);
    p.output_shape = a[1].shape;
    p.output_kind = a[0].kind;
    return;
  }
  numeric(a[0], 1, 1);
  if (o == Op::first || o == Op::last || o == Op::length)
    return;
  p.output_shape = a[0].shape;
  if (o == Op::aligned_shift) {
    if (p.count < 2)
      a[1] = Value::number(1);
    if (p.count < 3)
      a[2] = Value::number(std::numeric_limits<double>::quiet_NaN());
    integer(a[1], true);
    numeric(a[2], 0, 0);
    require(std::isfinite(a[2].scalar) || std::isnan(a[2].scalar),
            "INVALID_PARAMETER");
    return;
  }
  if (o == Op::recursive_filter) {
    numeric(a[1], 0, 0);
    numeric(a[2], 0, 0);
    mask(a[3], 1, 1);
    require(a[3].shape == a[0].shape, "SHAPE_MISMATCH");
    require(std::isfinite(a[1].scalar) && a[1].scalar >= 0 &&
                a[1].scalar <= 1 && std::isfinite(a[2].scalar),
            "INVALID_PARAMETER");
    if (p.count < 5)
      a[4] = Value::number(0);
    if (p.count < 6)
      a[5] = Value::number(0);
    require(integer(a[4], true) <= 2 && integer(a[5], true) <= 1,
            "INVALID_PARAMETER");
    return;
  }
  if (o == Op::new_high_mask)
    p.output_kind = Kind::mask;
  if (o == Op::lag || o == Op::difference) {
    if (p.count == 1)
      a[1] = Value::number(1);
    const auto periods = integer(a[1], o == Op::lag);
    require(periods < a[0].size(), "INVALID_PARAMETER");
    p.output_shape = vector_shape(a[0].size() - periods);
    if (o == Op::lag) {
      p.borrowed = true;
      p.view = a[0];
      p.view.shape = p.output_shape;
    }
  }
}
void rolling_shape(Prepared &p) {
  auto &a = p.args;
  const auto o = p.spec->op;
  numeric(a[0], 1, 1);
  p.output_shape = a[0].shape;
  const auto width = integer(a[1], false);
  if (o == Op::recursive_smooth) {
    numeric(a[2], 0, 0);
    require(std::isfinite(a[2].scalar), "INVALID_PARAMETER");
    return;
  }
  if (o == Op::rolling_std) {
    if (p.count < 3)
      a[2] = Value::number(0);
    if (p.count < 4)
      a[3] = a[1];
    integer(a[2], true);
  } else if (p.count < 3)
    a[2] = a[1];
  const auto minimum = integer(a[o == Op::rolling_std ? 3 : 2], false);
  require(minimum <= width, "INVALID_PARAMETER");
  if (o == Op::rolling_min || o == Op::rolling_max)
    p.scratch_indices = std::min(a[0].size(), width);
}
void infer_matrix_shape(Prepared &p) {
  auto &a = p.args;
  const auto o = p.spec->op;
  numeric(a[0], 1, 2);
  if (o == Op::transpose) {
    numeric(a[0], 2, 2);
    p.borrowed = true;
    p.view = a[0];
    std::swap(p.view.shape.dim[0], p.view.shape.dim[1]);
    std::swap(p.view.stride[0], p.view.stride[1]);
    p.output_shape = p.view.shape;
    return;
  }
  if (o == Op::diag) {
    if (a[0].shape.rank == 1)
      p.output_shape = matrix_shape(a[0].size(), a[0].size());
    else {
      p.output_shape =
          vector_shape(std::min(a[0].shape.dim[0], a[0].shape.dim[1]));
      p.borrowed = true;
      p.view = a[0];
      p.view.shape = p.output_shape;
      if (p.output_shape.size() <= 1)
        p.view.stride[0] = 1;
      else {
        const auto left = a[0].stride[0], right = a[0].stride[1];
        require(!(right > 0 && left > PTRDIFF_MAX - right) &&
                    !(right < 0 && left < PTRDIFF_MIN - right),
                "STRIDE_OVERFLOW");
        p.view.stride[0] = left + right;
      }
    }
    return;
  }
  if (o == Op::trace) {
    numeric(a[0], 2, 2);
    return;
  }
  if ((o == Op::covariance || o == Op::correlation) && p.count == 1) {
    numeric(a[0], 2, 2);
    const auto columns = a[0].shape.dim[1];
    p.output_shape = matrix_shape(columns, columns);
    p.scratch_doubles = columns * 2;
    return;
  }
  numeric(a[1], 1, 2);
  if (o == Op::dot || o == Op::outer || o == Op::covariance ||
      o == Op::correlation) {
    numeric(a[0], 1, 1);
    numeric(a[1], 1, 1);
    if (o == Op::outer)
      p.output_shape = matrix_shape(a[0].size(), a[1].size());
    else
      require(a[0].size() == a[1].size(), "SHAPE_MISMATCH");
    return;
  }
  numeric(a[0], 2, 2);
  if (o == Op::matmul) {
    numeric(a[1], 2, 2);
    require(a[0].shape.dim[1] == a[1].shape.dim[0], "SHAPE_MISMATCH");
    p.output_shape = matrix_shape(a[0].shape.dim[0], a[1].shape.dim[1]);
    return;
  }
  numeric(a[1], 1, 1);
  require(a[0].shape.dim[1] == a[1].size(), "SHAPE_MISMATCH");
  p.output_shape = vector_shape(a[0].shape.dim[0]);
  if (o == Op::solve) {
    require(a[0].shape.dim[0] == a[1].size(), "SHAPE_MISMATCH");
    require(a[0].size() <=
                (static_cast<std::size_t>(PTRDIFF_MAX) / sizeof(double)) -
                    a[1].size(),
            "SHAPE_OVERFLOW");
    p.scratch_doubles = a[0].size() + a[1].size();
  }
}
void state_shape(Prepared &p) {
  auto &a = p.args;
  const auto o = p.spec->op;
  if (o == Op::last_drawdown_interval) {
    numeric(a[0], 1, 1);
    p.output_kind = Kind::interval;
    return;
  }
  if (is_between(o, Op::interval_start, Op::interval_recovery)) {
    require(a[0].kind == Kind::interval, "RECORD_TYPE_MISMATCH");
    return;
  }
  if (is_between(o, Op::fit_slope, Op::fit_observation_count)) {
    require(a[0].kind == Kind::fit, "RECORD_TYPE_MISMATCH");
    return;
  }
  if (o == Op::value_at) {
    numeric(a[0], 1, 1);
    numeric(a[1], 0, 0);
    return;
  }
  numeric(a[0], 0, 0);
  if (p.count == 2)
    numeric(a[1], 0, 0);
}
void regression_shape(Prepared &p) {
  numeric(p.args[0], 1, 1);
  if (p.count == 2) {
    numeric(p.args[1], 1, 1);
    require(p.args[0].size() == p.args[1].size(), "SHAPE_MISMATCH");
  }
  if (p.spec->op == Op::linear_fit)
    p.output_kind = Kind::fit;
}
void composite_shape(Prepared &p) {
  auto &a = p.args;
  const auto o = p.spec->op;
  if (o == Op::active_returns) {
    numeric(a[0]);
    numeric(a[1]);
    p.output_shape = broadcast(a[0].shape, a[1].shape);
    return;
  }
  if (o == Op::portfolio_returns) {
    numeric(a[0], 2, 2);
    numeric(a[1], 1, 1);
    require(a[0].shape.dim[1] == a[1].size(), "SHAPE_MISMATCH");
    p.output_shape = vector_shape(a[0].shape.dim[0]);
    return;
  }
  numeric(a[0], 1, 1);
  if (o == Op::quadratic_form) {
    numeric(a[1], 2, 2);
    require(a[1].shape.dim[0] == a[0].size() &&
                a[1].shape.dim[1] == a[0].size(),
            "SHAPE_MISMATCH");
    p.scratch_doubles = a[0].size();
    return;
  }
  if (o == Op::cumulative_return)
    p.output_shape = a[0].shape;
  if (o == Op::annualized_return)
    numeric(a[1], 0, 0);
}
} // namespace

Prepared prepare(const Spec &spec, const Value *args, std::size_t count) {
  require(count >= spec.min_args && count <= spec.max_args && count <= 8,
          "ARITY_MISMATCH");
  Prepared p;
  p.spec = &spec;
  p.count = count;
  for (std::size_t i = 0; i < count; ++i) {
    require(args[i].shape.rank >= 0 && args[i].shape.rank <= 2,
            "RANK_MISMATCH");
    require(args[i].size() == 0 || args[i].shape.rank == 0 ||
                args[i].data != nullptr,
            "NULL_INPUT");
    p.args[i] = args[i];
  }
  switch (spec.family) {
  case Family::elementwise:
    elementwise_shape(p);
    break;
  case Family::reduction:
    reduction_shape(p);
    break;
  case Family::sequence:
    sequence_shape(p);
    break;
  case Family::rolling:
    rolling_shape(p);
    break;
  case Family::matrix:
    infer_matrix_shape(p);
    break;
  case Family::state:
    state_shape(p);
    break;
  case Family::regression:
    regression_shape(p);
    break;
  case Family::composite:
    composite_shape(p);
    break;
  }
  require(p.output_shape.size() <=
              static_cast<std::size_t>(PTRDIFF_MAX) / sizeof(double),
          "SHAPE_OVERFLOW");
  return p;
}

void execute(const Prepared &p, Output &out, Workspace &work, Isa isa,
             Audit &audit) {
  require(p.spec && out.kind == p.output_kind && out.shape == p.output_shape,
          "OUTPUT_MISMATCH");
  require(out.shape.rank == 0 || out.shape.size() == 0 || out.data,
          "NULL_OUTPUT");
  require(supports_isa(isa), "UNSUPPORTED_ISA");
  audit = {};
  for (std::size_t arg = 0; arg < p.count; ++arg) {
    const auto &value = p.args[arg];
    if (value.kind != Kind::mask)
      continue;
    if (value.shape.rank == 0)
      require(value.scalar == 0.0 || value.scalar == 1.0, "INVALID_MASK");
    else
      for (std::size_t i = 0; i < value.size(); ++i)
        require(value.u(i) <= 1, "INVALID_MASK");
  }
  work.ensure(p.scratch_doubles, p.scratch_indices);
  audit.scratch_bytes = p.scratch_doubles * sizeof(double) +
                        p.scratch_indices * sizeof(std::size_t);
  if (p.borrowed) {
    for (std::size_t i = 0; i < p.view.size(); ++i)
      out.set(i, p.view.f(i));
    return;
  }
  switch (p.spec->family) {
  case Family::elementwise:
    elementwise(p, out, work, isa, audit);
    break;
  case Family::reduction:
    reduction(p, out, work, audit);
    break;
  case Family::sequence:
  case Family::rolling:
    sequence(p, out, work, audit);
    break;
  case Family::matrix:
    matrix(p, out, work, isa, audit);
    break;
  case Family::regression:
    regression(p, out, work, audit);
    break;
  case Family::state:
    state(p, out, work, audit);
    break;
  case Family::composite:
    composite(p, out, work, isa, audit);
    break;
  }
}

const char *shape_rule(const Spec &s) {
  switch (s.op) {
  case Op::aligned_shift:
    return "float64 rank 1 -> same length; nonnegative integer periods; prefix fill";
  case Op::recursive_filter:
    return "float64 rank 1 and equal-shaped mask -> same length; scalar alpha/initial/policies";
  case Op::argsort:
    return "float64/int64 rank 1 -> equal-length int64 indices; ascending stable order";
  case Op::gather:
    return "float64/int64/mask rank 1 plus int64 rank 1 indices -> source dtype, indices length";
  case Op::distinct_count:
    return "int64 rank 1, optional equal-shaped mask -> float64 exact-cardinality scalar";
  default:
    break;
  }
  switch (s.family) {
  case Family::elementwise:
    return "numeric/mask rank 0..2; equal shapes or explicit scalar broadcast; "
           "logical "
           "requires equal masks; finite_mask and divide_or_default require "
           "rank 1";
  case Family::reduction:
    return "rank 1/2 -> scalar; *_time reduces axis 0; *_asset reduces axis 1; "
           "max_consecutive_true rank 1";
  case Family::sequence:
    return "rank 1; lag/difference length n-periods; first/last/length scalar; "
           "other scans "
           "preserve length";
  case Family::rolling:
    return "rank 1 -> same length, trailing windows, positions preserved";
  case Family::matrix:
    return "explicit vector/matrix ranks and inner-dimension checks; "
           "covariance/correlation "
           "single matrix reduces time axis 0";
  case Family::regression:
    return "one rank-1 y or equal-length rank-1 x,y; linear_fit returns "
           "FitState; projections "
           "scalar";
  case Family::state:
    return "typed FitState/DrawdownInterval projections or explicit "
           "scalar/vector access";
  case Family::composite:
    return "declared composition with the same operand and output shape as its "
           "primitive "
           "expansion";
  }
  return "invalid";
}
const char *missing_policy(const Spec &s) {
  switch (s.op) {
  case Op::normal_cdf:
    return "NaN -> NaN; -Inf -> 0; +Inf -> 1; erfc stable tail evaluation";
  case Op::floor:
    return "IEEE floor; NaN/+Inf/-Inf preserved; output remains float64";
  case Op::aligned_shift:
    return "prefix uses explicit fill (default NaN); remaining values copied unchanged";
  case Op::recursive_filter:
    return "update only mask=true and finite input; hold state across gaps; emit_policy=0 holds, 1 emits NaN; before first-valid seed emit NaN";
  case Op::argsort:
    return "NaN last; infinities ordered; equal values including signed zero keep original order";
  case Op::gather:
    return "selected values preserved exactly; negative or out-of-range indices fail before writes";
  case Op::distinct_count:
    return "mask=false excluded; every int64 code otherwise valid including negative codes; empty selection returns 0";
  default:
    break;
  }
  if (s.family == Family::rolling)
    return "skip nonfinite observations, preserve positions; recursive state "
           "carries across "
           "gaps";
  if (s.op == Op::divide_or_default)
    return "nonfinite pair -> NaN; abs(denominator)<1e-12 -> default";
  if (s.op == Op::finite_mask)
    return "finite -> true; NaN/+Inf/-Inf -> false";
  if (s.op == Op::drawdown_series || s.op == Op::new_high_mask ||
      s.family == Family::regression)
    return "reject nonfinite/domain-invalid values";
  if (s.family == Family::state)
    return "explicit state/domain contract; missing positions/dates -> NaN, "
           "invalid interval "
           "status=-1";
  if (s.op >= Op::sum_where && s.op <= Op::quantile_where)
    return "mask=0 excluded without numeric compaction; selected values retain "
           "source "
           "IEEE/reduction semantics";
  return "source IEEE arithmetic/reduction semantics; no implicit drop/fill; "
         "caller Typed DAG "
         "applies finite-input/result gates";
}
const char *composition(const Spec &s) {
  switch (s.op) {
  case Op::total_return:
    return "product(add(returns,1))-1";
  case Op::cumulative_return:
    return "cumulative_product(add(returns,1))-1";
  case Op::annualized_return:
    return "power(require_nonnegative((product(returns+1)-1)+1),require_"
           "positive(periods_per_"
           "year)/length(returns))-1";
  case Op::active_returns:
    return "subtract(lhs,rhs)";
  case Op::portfolio_returns:
    return "matvec(asset_returns,asset_weights)";
  case Op::quadratic_form:
    return "dot(matvec(transpose(matrix),vector),vector)";
  case Op::linear_slope:
    return "fit_slope(linear_fit(x,y))";
  case Op::linear_intercept:
    return "fit_intercept(linear_fit(x,y))";
  case Op::linear_r_squared:
    return "1-fit_residual_sum_squares(fit)/"
           "require_positive(fit_total_sum_squares(fit))";
  case Op::regression_standard_error:
    return "sqrt(fit_residual_sum_squares(fit)/"
           "require_positive(fit_observation_count(fit)-2))";
  default:
    return "";
  }
}
std::vector<std::string> parameter_names(const Spec &spec, std::size_t arity) {
  require(arity >= spec.min_args && arity <= spec.max_args, "ARITY_MISMATCH");
  if (arity == 1 && spec.family == Family::regression)
    return {"values"};
  if (arity == 1 && (spec.op == Op::covariance || spec.op == Op::correlation))
    return {"asset_returns"};
  std::vector<std::string> names;
  const std::string_view text = spec.parameters;
  std::size_t start = 0;
  while (start < text.size() && names.size() < arity) {
    const auto end = text.find(',', start);
    names.emplace_back(
        text.substr(start, end == std::string_view::npos ? end : end - start));
    if (end == std::string_view::npos)
      break;
    start = end + 1;
  }
  require(names.size() == arity, "INVALID_REGISTRY_PARAMETERS");
  return names;
}

const char *default_rule(const Spec &s) {
  switch (s.op) {
  case Op::aligned_shift:
    return "periods=1, fill=NaN; periods>=0 may exceed length; fill finite or NaN";
  case Op::recursive_filter:
    return "seed_mode=0 consume first with initial, 1 seed row0 without consumption, 2 seed first eligible; emit_policy=0 hold, 1 NaN; alpha in [0,1], initial finite";
  case Op::distinct_count:
    return "mask omitted selects every identifier; no implicit missing category code";
  case Op::variance:
  case Op::std:
    return "ddof=1; finite integer 0<=ddof<n";
  case Op::lag:
  case Op::difference:
    return "periods=1; lag permits 0, difference requires >0; periods<n";
  case Op::rolling_std:
    return "ddof=0, min_periods=window; window>=1, 1<=min_periods<=window";
  case Op::rolling_mean:
  case Op::rolling_min:
  case Op::rolling_max:
    return "min_periods=window; window>=1, 1<=min_periods<=window";
  default:
    return "no implicit optional parameters";
  }
}
} // namespace calmetrics_engine::ops
