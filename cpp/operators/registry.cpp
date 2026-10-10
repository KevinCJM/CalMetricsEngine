#include "mask_validation.hpp"
#include "calmetrics_engine/operators.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

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
  // Immutable canonical names use a fixed index, without per-name heap nodes.
  // Keep registry/opcode order untouched; the standard library sorts only IDs.
  static const auto by_name = [] {
    std::array<std::uint16_t, operator_count> ids{};
    std::iota(ids.begin(), ids.end(), std::uint16_t(0));
    std::sort(ids.begin(), ids.end(), [](auto left, auto right) {
      return std::string_view(registry()[left].name) < std::string_view(registry()[right].name);
    });
    return ids;
  }();
  const auto it = std::lower_bound(by_name.begin(), by_name.end(), name,
      [](auto id, std::string_view candidate) { return std::string_view(registry()[id].name) < candidate; });
  if (it != by_name.end() && name == registry()[*it].name)
    return registry()[*it];
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
Shape broadcast(const Shape &a, const Shape &b) {
  if (a.rank == 0)
    return b;
  if (b.rank == 0)
    return a;
  require(a == b, "SHAPE_MISMATCH");
  return a;
}
void numeric_arg(const Prepared &p, std::size_t i, int lo = 0, int hi = 3) {
  require(p.args[i].kind == Kind::number, "DTYPE_MISMATCH");
  if (p.geometry(i)) require(p.args[i].shape.rank >= lo && p.args[i].shape.rank <= hi, "RANK_MISMATCH");
}
void mask_arg(const Prepared &p, std::size_t i, int lo = 0, int hi = 3) {
  require(p.args[i].kind == Kind::mask, "DTYPE_MISMATCH");
  if (p.geometry(i)) require(p.args[i].shape.rank >= lo && p.args[i].shape.rank <= hi, "RANK_MISMATCH");
}
void integer_arg(const Prepared &p, std::size_t i, int lo = 1, int hi = 1) {
  require(p.args[i].kind == Kind::integer, "DTYPE_MISMATCH");
  if (p.geometry(i)) require(p.args[i].shape.rank >= lo && p.args[i].shape.rank <= hi, "RANK_MISMATCH");
}
void equal_shapes(const Prepared &p, std::size_t a, std::size_t b) {
  if (p.geometry(a) && p.geometry(b)) require(p.args[a].shape == p.args[b].shape, "SHAPE_MISMATCH");
}
void output_from(Prepared &p, std::size_t i) {
  p.output_geometry_known = p.geometry(i);
  if (p.output_geometry_known) p.output_shape = p.args[i].shape;
}
// A malformed numerical configuration retains the existing isolate policy. In
// structural-only validation it makes its own value unavailable, so unrelated
// structural checks continue without inventing a valid replacement parameter.
bool configuration(Prepared &p, std::size_t i, bool valid) {
  if (!p.payload(i)) return false;
  if (valid) return true;
  if (!p.structure_only) throw Error("INVALID_PARAMETER");
  p.payload_available_mask &= static_cast<std::uint32_t>(~(1u << i));
  return false;
}
bool integer_arg_value(Prepared &p, std::size_t i, bool zero) {
  numeric_arg(p, i, 0, 0);
  if (!p.payload(i)) return false;
  const auto x = p.args[i].scalar;
  return configuration(p, i, std::isfinite(x) && x == std::floor(x) &&
      x >= (zero ? 0 : 1) && x < static_cast<double>(PTRDIFF_MAX));
}
void broadcast_output(Prepared &p, std::initializer_list<std::size_t> ids) {
  Shape shape;
  bool known = true;
  for (const auto i : ids) {
    if (!p.geometry(i)) { known = false; continue; }
    shape = broadcast(shape, p.args[i].shape);
  }
  p.output_geometry_known = known;
  if (known) p.output_shape = shape;
}
void elementwise_shape(Prepared &p) {
  const auto o = p.spec->op;
  auto &a = p.args;
  if ((o == Op::equal || o == Op::not_equal) &&
      (a[0].kind == Kind::integer || a[1].kind == Kind::integer)) {
    for (std::size_t i = 0; i < 2; ++i) {
      if (a[i].kind == Kind::integer) continue;
      numeric_arg(p, i, 0, 0);
      if (p.payload(i)) require(std::isfinite(a[i].scalar) && a[i].scalar == std::floor(a[i].scalar) &&
          std::abs(a[i].scalar) <= 9007199254740991.0, "INVALID_INTEGER_SCALAR");
    }
    broadcast_output(p, {0, 1}); p.output_kind = Kind::mask; return;
  }
  if (o == Op::finite_mask) {
    numeric_arg(p, 0, 0, 3); p.output_kind = Kind::mask; output_from(p, 0); return;
  }
  if (o == Op::logical_and || o == Op::logical_or || o == Op::logical_not) {
    mask_arg(p, 0);
    if (o != Op::logical_not) { mask_arg(p, 1); equal_shapes(p, 0, 1); }
    p.output_kind = Kind::mask; output_from(p, 0); return;
  }
  if (o == Op::where) {
    mask_arg(p, 0); numeric_arg(p, 1); numeric_arg(p, 2);
    broadcast_output(p, {0, 1, 2}); return;
  }
  numeric_arg(p, 0); output_from(p, 0);
  if (o == Op::divide_or_default) {
    numeric_arg(p, 0, 1, 1); numeric_arg(p, 1, 1, 1); numeric_arg(p, 2, 0, 0);
    if (p.geometry(0) && p.geometry(1)) {
      // Preserve this operator's existing numerical-parameter classification.
      if (!(a[0].shape == a[1].shape)) {
        if (!p.structure_only) throw Error("INVALID_PARAMETER");
      }
    }
    if (p.payload(2)) configuration(p, 2, std::isfinite(a[2].scalar));
    return;
  }
  if (o == Op::clip) { numeric_arg(p, 1, 0, 0); numeric_arg(p, 2, 0, 0); return; }
  if (p.count == 2) { numeric_arg(p, 1); broadcast_output(p, {0, 1}); }
  if (is_between(o, Op::equal, Op::greater_equal)) p.output_kind = Kind::mask;
}
void reduction_shape(Prepared &p) {
  const auto o = p.spec->op; auto &a = p.args;
  if (o == Op::distinct_count) {
    integer_arg(p, 0);
    if (p.count == 1) { a[1] = Value::number(1); a[1].kind = Kind::mask; }
    else { mask_arg(p, 1, 1, 1); equal_shapes(p, 0, 1); }
    if (p.geometry(0)) p.scratch_indices = a[0].size();
    return;
  }
  if (o == Op::count_true || o == Op::max_consecutive_true) {
    mask_arg(p, 0, 1, o == Op::count_true ? 3 : 1); return;
  }
  numeric_arg(p, 0, 1, 3);
  if (is_between(o, Op::sum_time, Op::max_asset)) {
    numeric_arg(p, 0, 2, 2); p.output_geometry_known = p.geometry(0);
    if (p.output_geometry_known) p.output_shape = vector_shape(a[0].shape.dim[o <= Op::max_time ? 1 : 0]);
    return;
  }
  if (is_between(o, Op::sum_where, Op::quantile_where)) { mask_arg(p, 1, 1, 3); equal_shapes(p, 0, 1); }
  if (o == Op::variance || o == Op::std) {
    if (p.count == 1) a[1] = Value::number(1);
    integer_arg_value(p, 1, true);
  }
  if (o == Op::quantile || o == Op::quantile_where) {
    const auto i = o == Op::quantile ? 1u : 2u; numeric_arg(p, i, 0, 0);
    if (p.payload(i)) configuration(p, i, std::isfinite(a[i].scalar) && a[i].scalar > 0 && a[i].scalar < 1);
  }
  if ((o == Op::median || o == Op::quantile || o == Op::median_where || o == Op::quantile_where) && p.geometry(0))
    p.scratch_doubles = a[0].size();
}
void sequence_shape(Prepared &p) {
  auto &a = p.args; const auto o = p.spec->op;
  if (o == Op::argsort || o == Op::gather) {
    require(a[0].kind == Kind::number || a[0].kind == Kind::integer ||
        (o == Op::gather && a[0].kind == Kind::mask), "DTYPE_MISMATCH");
    if (p.geometry(0)) require(a[0].shape.rank == 1, "RANK_MISMATCH");
    if (o == Op::gather) { integer_arg(p, 1); output_from(p, 1); p.output_kind = a[0].kind; }
    else { output_from(p, 0); p.output_kind = Kind::integer; if (p.geometry(0)) p.scratch_indices = a[0].size(); }
    return;
  }
  numeric_arg(p, 0, 1, 1);
  if (o == Op::first || o == Op::last || o == Op::length) return;
  output_from(p, 0);
  if (o == Op::recursive_filter_adaptive) {
    numeric_arg(p, 1, 0, 1);
    if (p.geometry(0) && p.geometry(1)) require(a[1].shape.rank == 0 || a[1].shape == a[0].shape, "SHAPE_MISMATCH");
    for (std::size_t i = 2; i <= 3; ++i) { mask_arg(p, i, 1, 1); equal_shapes(p, 0, i); }
    equal_shapes(p, 2, 3);
    if (p.count < 5) a[4] = Value::number(0);
    if (p.count < 6) a[5] = Value::number(1);
    if (p.count < 7) a[6] = Value::number(1);
    if (p.count < 8) a[7] = Value::number(1);
    numeric_arg(p, 4, 0, 0);
    if (p.payload(4)) configuration(p, 4, std::isfinite(a[4].scalar));
    if (integer_arg_value(p, 5, true)) configuration(p, 5, a[5].scalar <= 2);
    integer_arg_value(p, 6, false);
    if (integer_arg_value(p, 7, true)) configuration(p, 7, a[7].scalar <= 1);
    return;
  }
  if (o == Op::linear_filter2) {
    for (std::size_t i = 1; i <= 4; ++i) { numeric_arg(p, i, 0, 0); if (p.payload(i)) configuration(p, i, std::isfinite(a[i].scalar)); }
    mask_arg(p, 5, 1, 1); equal_shapes(p, 0, 5);
    if (p.count < 7) a[6] = Value::number(1);
    if (p.count < 8) a[7] = Value::number(1);
    integer_arg_value(p, 6, false);
    if (integer_arg_value(p, 7, true)) configuration(p, 7, a[7].scalar <= 1);
    return;
  }
  if (o == Op::scalar_kalman) {
    if (p.count < 4) a[3] = Value::number(1);
    for (std::size_t i = 1; i <= 3; ++i) { numeric_arg(p, i, 0, 0); if (p.payload(i)) configuration(p, i, std::isfinite(a[i].scalar) && (i == 2 ? a[i].scalar > 0 : a[i].scalar >= 0)); }
    if (p.geometry(0)) p.output_shape = matrix_shape(a[0].size(), 2);
    return;
  }
  if (o == Op::aligned_shift) {
    if (p.count < 2) a[1] = Value::number(1);
    if (p.count < 3) a[2] = Value::number(std::numeric_limits<double>::quiet_NaN());
    integer_arg_value(p, 1, true); numeric_arg(p, 2, 0, 0);
    if (p.payload(2)) configuration(p, 2, std::isfinite(a[2].scalar) || std::isnan(a[2].scalar));
    return;
  }
  if (o == Op::recursive_filter) {
    numeric_arg(p, 1, 0, 0); numeric_arg(p, 2, 0, 0); mask_arg(p, 3, 1, 1); equal_shapes(p, 0, 3);
    if (p.payload(1)) configuration(p, 1, std::isfinite(a[1].scalar) && a[1].scalar >= 0 && a[1].scalar <= 1);
    if (p.payload(2)) configuration(p, 2, std::isfinite(a[2].scalar));
    if (p.count < 5) a[4] = Value::number(0);
    if (p.count < 6) a[5] = Value::number(0);
    if (integer_arg_value(p, 4, true)) configuration(p, 4, a[4].scalar <= 2);
    if (integer_arg_value(p, 5, true)) configuration(p, 5, a[5].scalar <= 1);
    return;
  }
  if (o == Op::new_high_mask) p.output_kind = Kind::mask;
  if (o == Op::lag || o == Op::difference) {
    if (p.count == 1) a[1] = Value::number(1);
    const auto valid = integer_arg_value(p, 1, o == Op::lag);
    p.output_geometry_known = p.geometry(0) && valid;
    if (p.output_geometry_known) {
      const auto periods = static_cast<std::size_t>(a[1].scalar);
      if (!configuration(p, 1, periods < a[0].size())) { p.output_geometry_known = false; return; }
      p.output_shape = vector_shape(a[0].size() - periods);
      if (o == Op::lag && p.payload(0)) { p.borrowed = true; p.view = a[0]; p.view.shape = p.output_shape; }
    }
  }
}
void rolling_shape(Prepared &p) {
  auto &a = p.args; const auto o = p.spec->op;
  numeric_arg(p, 0, 1, 1); output_from(p, 0);
  const auto width = integer_arg_value(p, 1, false);
  if (o == Op::recursive_smooth) { numeric_arg(p, 2, 0, 0); if (p.payload(2)) configuration(p, 2, std::isfinite(a[2].scalar)); return; }
  if (o == Op::rolling_std) {
    if (p.count < 3) a[2] = Value::number(0);
    if (p.count < 4) { a[3] = a[1];
      if (!p.payload(1)) p.payload_available_mask &= static_cast<std::uint32_t>(~(1u << 3));
      if (!p.geometry(1)) p.geometry_known_mask &= static_cast<std::uint32_t>(~(1u << 3)); }
    integer_arg_value(p, 2, true);
  } else if (p.count < 3) { a[2] = a[1];
    if (!p.payload(1)) p.payload_available_mask &= static_cast<std::uint32_t>(~(1u << 2));
    if (!p.geometry(1)) p.geometry_known_mask &= static_cast<std::uint32_t>(~(1u << 2)); }
  const auto minimum_index = o == Op::rolling_std ? 3u : 2u;
  const auto minimum = integer_arg_value(p, minimum_index, false);
  if (width && minimum) configuration(p, minimum_index, a[minimum_index].scalar <= a[1].scalar);
  if ((o == Op::rolling_min || o == Op::rolling_max) && p.geometry(0) && width)
    p.scratch_indices = std::min(a[0].size(), static_cast<std::size_t>(a[1].scalar));
}
void infer_matrix_shape(Prepared &p) {
  if (axis_path_operator(p.spec->op)) { axis_path_prepare(p); return; }
  if (latent_operator(p.spec->op)) { prepare_latent(p); return; }
  if (matrix_foundation(p.spec->op)) { prepare_foundation(p); return; }
  auto &a = p.args; const auto o = p.spec->op;
  if (o == Op::affine_contract) {
    numeric_arg(p, 0, 2, 3);
    numeric_arg(p, 1, 2, 2);
    numeric_arg(p, 2, 0, 1);
    if (p.geometry(0) && p.geometry(1))
      require(a[0].shape.dim[a[0].shape.rank - 1] == a[1].shape.dim[0], "SHAPE_MISMATCH");
    if (p.geometry(1) && p.geometry(2) && a[2].shape.rank == 1)
      require(a[2].shape.dim[0] == a[1].shape.dim[1], "SHAPE_MISMATCH");
    p.output_geometry_known = p.geometry(0) && p.geometry(1);
    if (p.output_geometry_known) {
      p.output_shape = a[0].shape;
      p.output_shape.dim[p.output_shape.rank - 1] = a[1].shape.dim[1];
    }
    return;
  }
  numeric_arg(p, 0, 1, 2);
  if (o == Op::transpose) {
    numeric_arg(p, 0, 2, 2); p.output_geometry_known = p.geometry(0);
    if (p.geometry(0)) {
      p.output_shape = matrix_shape(a[0].shape.dim[1], a[0].shape.dim[0]);
      if (p.payload(0)) { p.borrowed = true; p.view = a[0]; std::swap(p.view.shape.dim[0], p.view.shape.dim[1]); std::swap(p.view.stride[0], p.view.stride[1]); }
    }
    return;
  }
  if (o == Op::diag) {
    p.output_geometry_known = p.geometry(0); if (!p.geometry(0)) return;
    if (a[0].shape.rank == 1) p.output_shape = matrix_shape(a[0].size(), a[0].size());
    else {
      p.output_shape = vector_shape(std::min(a[0].shape.dim[0], a[0].shape.dim[1]));
      auto stride = std::ptrdiff_t{1};
      if (p.output_shape.size() > 1) {
        const auto left = a[0].stride[0], right = a[0].stride[1];
        require(!(right > 0 && left > PTRDIFF_MAX - right) && !(right < 0 && left < PTRDIFF_MIN - right), "STRIDE_OVERFLOW");
        stride = left + right;
      }
      if (p.payload(0)) { p.borrowed = true; p.view = a[0]; p.view.shape = p.output_shape; p.view.stride[0] = stride; }
    }
    return;
  }
  if (o == Op::trace) { numeric_arg(p, 0, 2, 2); return; }
  if ((o == Op::covariance || o == Op::correlation) && p.count == 1) {
    numeric_arg(p, 0, 2, 2); p.output_geometry_known = p.geometry(0);
    if (p.geometry(0)) { const auto cols = a[0].shape.dim[1]; p.output_shape = matrix_shape(cols, cols); }
    return;
  }
  numeric_arg(p, 1, 1, 2);
  if (o == Op::dot || o == Op::outer || o == Op::covariance || o == Op::correlation) {
    numeric_arg(p, 0, 1, 1); numeric_arg(p, 1, 1, 1);
    if (o == Op::outer) {
      p.output_geometry_known = p.geometry(0) && p.geometry(1);
      if (p.output_geometry_known) p.output_shape = matrix_shape(a[0].size(), a[1].size());
    } else equal_shapes(p, 0, 1);
    return;
  }
  numeric_arg(p, 0, 2, 2);
  if (o == Op::matmul) {
    numeric_arg(p, 1, 2, 2); p.output_geometry_known = p.geometry(0) && p.geometry(1);
    if (p.output_geometry_known) {
      require(a[0].shape.dim[1] == a[1].shape.dim[0], "SHAPE_MISMATCH");
      p.output_shape = matrix_shape(a[0].shape.dim[0], a[1].shape.dim[1]);
      p.scratch_doubles = matmul_scratch_doubles(a[0].shape, a[1].shape);
    }
    return;
  }
  numeric_arg(p, 1, 1, 1);
  if (p.geometry(0) && p.geometry(1)) require(a[0].shape.dim[1] == a[1].size(), "SHAPE_MISMATCH");
  p.output_geometry_known = p.geometry(0);
  if (p.geometry(0)) p.output_shape = vector_shape(a[0].shape.dim[0]);
  if (o == Op::solve && p.geometry(0) && p.geometry(1)) {
    require(a[0].shape.dim[0] == a[1].size(), "SHAPE_MISMATCH");
    require(a[0].size() <= (static_cast<std::size_t>(PTRDIFF_MAX) / sizeof(double)) - a[1].size(), "SHAPE_OVERFLOW");
    p.scratch_doubles = a[0].size() + a[1].size();
  }
}
void state_shape(Prepared &p) {
  auto &a = p.args; const auto o = p.spec->op;
  if (o == Op::state_select) {
    mask_arg(p, 0, 1, 1); mask_arg(p, 3, 1, 1); equal_shapes(p, 0, 3);
    for (std::size_t j = 1; j <= 2; ++j) {
      if (a[j].kind == Kind::integer) {
        integer_arg(p, j); equal_shapes(p, 0, j); equal_shapes(p, 3, j);
        if (p.payload(j)) for (std::size_t i = 0; i < a[j].size(); ++i) require(a[j].i(i) >= -1, "INVALID_STATE_CODE");
      } else {
        numeric_arg(p, j, 0, 0);
        if (p.payload(j)) require(std::isfinite(a[j].scalar) && a[j].scalar >= -1 && a[j].scalar <= 9007199254740991.0 && a[j].scalar == std::floor(a[j].scalar), "INVALID_STATE_CODE");
      }
    }
    if (a[1].kind == Kind::integer && a[2].kind == Kind::integer) equal_shapes(p, 1, 2);
    p.output_kind = Kind::integer; output_from(p, p.geometry(0) ? 0 : 3); return;
  }
  if (is_between(o, Op::state_hysteresis, Op::drawdown_cycle_reference)) { prepare_state_events(p); return; }
  if (o == Op::state_estimate || o == Op::state_variance) {
    numeric_arg(p, 0, 2, 2); p.output_geometry_known = p.geometry(0);
    if (p.geometry(0)) { require(a[0].shape.dim[1] == 2, "SHAPE_MISMATCH"); p.output_shape = vector_shape(a[0].shape.dim[0]);
      if (p.payload(0)) { p.borrowed = true; p.view = a[0].column(o == Op::state_estimate ? 0 : 1); } }
    return;
  }
  if (o == Op::last_drawdown_interval) { numeric_arg(p, 0, 1, 1); p.output_kind = Kind::interval; return; }
  if (is_between(o, Op::interval_start, Op::interval_recovery)) { require(a[0].kind == Kind::interval, "RECORD_TYPE_MISMATCH"); return; }
  if (is_between(o, Op::fit_slope, Op::fit_observation_count)) { require(a[0].kind == Kind::fit, "RECORD_TYPE_MISMATCH"); return; }
  if (o == Op::value_at) { numeric_arg(p, 0, 1, 1); numeric_arg(p, 1, 0, 0); return; }
  numeric_arg(p, 0, 0, 0); if (p.count == 2) numeric_arg(p, 1, 0, 0);
}
void regression_shape(Prepared &p) {
  numeric_arg(p, 0, 1, 1);
  if (p.count == 2) { numeric_arg(p, 1, 1, 1); equal_shapes(p, 0, 1); }
  if (p.spec->op == Op::linear_fit) p.output_kind = Kind::fit;
}
void composite_shape(Prepared &p) {
  auto &a = p.args; const auto o = p.spec->op;
  if (o == Op::active_returns) { numeric_arg(p, 0); numeric_arg(p, 1); broadcast_output(p, {0, 1}); return; }
  if (o == Op::portfolio_returns) {
    numeric_arg(p, 0, 2, 2); numeric_arg(p, 1, 1, 1);
    if (p.geometry(0) && p.geometry(1)) require(a[0].shape.dim[1] == a[1].size(), "SHAPE_MISMATCH");
    p.output_geometry_known = p.geometry(0); if (p.geometry(0)) p.output_shape = vector_shape(a[0].shape.dim[0]); return;
  }
  numeric_arg(p, 0, 1, 1);
  if (o == Op::quadratic_form) {
    numeric_arg(p, 1, 2, 2);
    if (p.geometry(0) && p.geometry(1)) require(a[1].shape.dim[0] == a[0].size() && a[1].shape.dim[1] == a[0].size(), "SHAPE_MISMATCH");
    if (p.geometry(0)) p.scratch_doubles = a[0].size();
    return;
  }
  if (o == Op::cumulative_return) output_from(p, 0);
  if (o == Op::annualized_return) numeric_arg(p, 1, 0, 0);
}
} // namespace

static std::size_t validate_data_structure(const Prepared &p,
                                         bool count_selected = false) {
  std::size_t selected = 0;
  for (std::size_t arg = 0; arg < p.count; ++arg) {
    if (!p.payload(arg)) continue;
    const auto &value = p.args[arg];
    if (value.kind != Kind::mask) continue;
    if (value.shape.rank == 0)
      require(value.scalar == 0.0 || value.scalar == 1.0, "INVALID_MASK");
#if defined(CALMETRICS_HAVE_HIGHWAY_ORDER)
    else if (value.size() >= 128 && value.c_contiguous()) {
      std::size_t count = 0;
      const bool counting = count_selected && arg == 1;
      require(mask_detail::validate_mask_bytes_highway(
          static_cast<const std::uint8_t *>(value.data), value.size(),
          counting ? &count : nullptr), "INVALID_MASK");
      selected += count;
    }
#endif
    else if (count_selected && arg == 1) {
      const auto *data = value.c_contiguous()
          ? static_cast<const std::uint8_t *>(value.data) : nullptr;
      const auto n = value.size();
      for (std::size_t i = 0; i < n; ++i) {
        const auto byte = data ? data[i] : value.u(i);
        require(byte <= 1, "INVALID_MASK");
        selected += byte;
      }
    } else {
      const auto n = value.size();
      if (n && value.c_contiguous()) {
        const auto *data = static_cast<const std::uint8_t *>(value.data);
        require(std::all_of(data, data + n,
            [](std::uint8_t byte) { return byte <= 1; }), "INVALID_MASK");
      } else {
        for (std::size_t i = 0; i < n; ++i)
          require(value.u(i) <= 1, "INVALID_MASK");
      }
    }
  }
  if (p.spec->op == Op::gather && p.payload(1) &&
      p.args[1].kind == Kind::integer && p.args[1].shape.rank == 1) {
    const auto &indices = p.args[1];
    for (std::size_t i = 0; i < indices.size(); ++i) {
      require(indices.i(i) >= 0, "INDEX_OUT_OF_BOUNDS");
      if (p.geometry(0) && p.args[0].shape.rank == 1)
        require(static_cast<std::uint64_t>(indices.i(i)) < p.args[0].size(), "INDEX_OUT_OF_BOUNDS");
    }
  }
  return selected;
}

static Prepared prepare_impl(const Spec &spec, const Value *args, std::size_t count,
    std::uint32_t geometry_known, std::uint32_t payload_available, bool structure_only) {
  require(count >= spec.min_args && count <= spec.max_args && count <= max_operator_arguments,
          "ARITY_MISMATCH");
  Prepared p(std::max<std::size_t>(count, spec.max_args));
  p.spec = &spec;
  p.count = count;
  const auto provided = (count == 32 ? UINT32_MAX : (1u << count) - 1);
  require((payload_available & provided & ~geometry_known) == 0, "ARGUMENT_AVAILABILITY");
  p.geometry_known_mask = geometry_known | static_cast<std::uint32_t>(~provided);
  p.payload_available_mask = payload_available | static_cast<std::uint32_t>(~provided);
  p.structure_only = structure_only;
  for (std::size_t i = 0; i < count; ++i) {
    if (p.geometry(i)) require(args[i].shape.rank >= 0 && args[i].shape.rank <= 3, "RANK_MISMATCH");
    if (p.payload(i)) require(args[i].size() == 0 || args[i].shape.rank == 0 || args[i].data != nullptr, "NULL_INPUT");
    p.args[i] = args[i];
  }
  // This preflight cannot stop at an unrelated invalid numerical parameter.
  // Normal execution invokes this same payload validator once in execute().
  if (structure_only) validate_data_structure(p);
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
  if (p.output_geometry_known)
    require(p.output_shape.size() <= static_cast<std::size_t>(PTRDIFF_MAX) / sizeof(double), "SHAPE_OVERFLOW");
  return p;
}

void validate_payload_structure(const Prepared &p) {
  (void)validate_data_structure(p);
}

Prepared prepare(const Spec &spec, const Value *args, std::size_t count) {
  return prepare_impl(spec, args, count, UINT32_MAX, UINT32_MAX, false);
}
Prepared prepare_geometry(const Spec &spec, const Value *args, std::size_t count) {
  std::uint32_t payload = 0;
  for (std::size_t i = 0; i < count; ++i)
    if (!args[i].shape.rank || args[i].data) payload |= (1u << i);
  return prepare_impl(spec, args, count, UINT32_MAX, payload, false);
}
StructureResult validate_structure(const Spec &spec, const Value *args, std::size_t count,
    std::uint32_t geometry_known, std::uint32_t payload_available) {
  const auto p = prepare_impl(spec, args, count, geometry_known, payload_available, true);
  return {p.output_kind, p.output_shape, p.output_geometry_known};
}

void execute(const Prepared &p, Output &out, Workspace &work, Isa isa,
             Audit &audit, const LatentFitScratch *fit_scratch) {
  require(p.spec && out.kind == p.output_kind && out.shape == p.output_shape,
          "OUTPUT_MISMATCH");
  require(out.shape.rank == 0 || out.shape.size() == 0 || out.data,
          "NULL_OUTPUT");
  require(out.order == MemoryOrder::c || out.shape.rank < 2 || p.borrowed ||
          p.spec->family == Family::elementwise || p.spec->op == Op::matmul ||
          p.spec->op == Op::affine_contract ||
          axis_path_operator(p.spec->op),
          "UNSUPPORTED_OUTPUT_ORDER");
  require(supports_isa(isa), "UNSUPPORTED_ISA");
  audit = {};
  const bool masked_order = p.spec->op == Op::median_where ||
                            p.spec->op == Op::quantile_where;
  const auto selected = validate_data_structure(p, masked_order);
  const bool skip_panels =
      (p.spec->op == Op::matmul &&
       (isa != Isa::automatic || !matmul_packed_eligible(p.args[0], p.args[1]))) ||
      (p.spec->op == Op::cholesky && isa != Isa::automatic);
  const auto doubles = skip_panels ? 0 : masked_order ? selected : p.scratch_doubles;
  if (fit_scratch) {
    require(!p.borrowed && (p.spec->op == Op::gaussian_mixture_em ||
        p.spec->op == Op::gaussian_hmm_em), "FIT_SCRATCH_CONTRACT");
    const auto cells = p.args[0].shape.dim[0] * p.args[1].size();
    const auto matrices = latent_fit_matrix_count(p.spec->op, p.args[0].shape.dim[0], p.args[1].size());
    const auto full_scratch = p.spec->op == Op::gaussian_hmm_em ?
        latent_hmm_full_scratch(p.args[0].shape.dim[0],p.args[1].size()) : p.scratch_doubles;
    require(fit_scratch->cells >= cells && fit_scratch->em && fit_scratch->alpha &&
        fit_scratch->beta && fit_scratch->small &&
        fit_scratch->small_count >= full_scratch - matrices * cells &&
        p.scratch_indices == 0, "FIT_SCRATCH_CONTRACT");
    if (p.spec->op == Op::gaussian_hmm_em)
      require(fit_scratch->alpha != fit_scratch->em &&
          fit_scratch->beta != fit_scratch->alpha &&
          (matrices == 2 || fit_scratch->beta != fit_scratch->em),
          "FIT_SCRATCH_CONTRACT");
    // Graph reserved the live-value saves, missing matrices and small scratch
    // together before acquiring pointers. Never resize it within this fit.
  } else work.ensure(doubles, p.scratch_indices);
  audit.scratch_bytes = doubles * sizeof(double) +
                        p.scratch_indices * sizeof(std::size_t);
  if (p.borrowed) {
    for (std::size_t i = 0; i < p.view.size(); ++i) {
      if (p.view.kind == Kind::integer)
        out.set_integer(i, p.view.i(i));
      else if (p.view.kind == Kind::mask)
        out.set_mask(i, p.view.u(i));
      else
        out.set(i, p.view.f(i));
    }
    return;
  }
  if (latent_operator(p.spec->op)) { latent_execute(p, out, work, isa, audit, fit_scratch); return; }
  if (axis_path_operator(p.spec->op)) { axis_path_execute(p, out, work, isa, audit); return; }
  if (matrix_foundation(p.spec->op)) {
    matrix_foundation_execute(p, out, work, isa, audit);
    return;
  }
  if (is_between(p.spec->op, Op::recursive_filter_adaptive, Op::scalar_kalman)) {
    recurrence(p, out, work, audit);
    return;
  }
  if (is_between(p.spec->op, Op::state_hysteresis, Op::drawdown_cycle_reference)) {
    state_events(p, out, work, audit);
    return;
  }
  switch (p.spec->family) {
  case Family::elementwise:
    elementwise(p, out, work, isa, audit);
    break;
  case Family::reduction:
    reduction(p, out, work, isa, audit);
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
  if (s.op == Op::quantiles_axis)
    return "float64 rank 1..3 values, float64 rank 1 probabilities, explicit axis -> same rank with that axis replaced by probability count";
  if (axis_path_operator(s.op))
    return "float64 rank 1..3 values, explicit axis, float64 scalar initial -> same shape; ascending inclusive prefix on each independent line";
  if (latent_operator(s.op)) {
    if(s.op>=Op::variance_axis && s.op<=Op::softmax_axis) return "rank 1..3; explicit axis; reduction removes axis, softmax preserves shape; argmax int64";
    if(s.op==Op::gaussian_log_density) return "X[N,D], means/variances[K,D] -> log density[N,K]";
    if(s.op==Op::gaussian_mixture_em) return "X[N,D], weights[K], means/variances[K,D] -> nominal Gaussian mixture[K,2D+4]";
    if(s.op==Op::gaussian_hmm_em) return "X[N,D], initial[K], transition[K,K], means/variances[K,D] -> nominal Gaussian HMM[K,2D+K+4]";
    if(latent_view(s.op)) return "nominal fitted model -> borrowed readonly parameter vector/matrix";
    if(s.op>=Op::em_iterations && s.op<=Op::em_converged) return "nominal fitted model -> int64 / float64 / bool scalar diagnostic";
    if(s.op==Op::hmm_filter || s.op==Op::hmm_smooth) return "log emissions[N,K], initial[K], transition[K,K] -> posterior[N,K]";
    if(s.op==Op::one_hot) return "int64 codes[N], literal K -> float64[N,K]";
    if(s.op==Op::transition_counts) return "int64 codes[N], literal K -> float64[K,K]";
    if(s.op==Op::partition_mean) return "float64[N,D], int64 permutation[N], literal K<=N -> float64[K,D]";
    if(s.op==Op::sample_unique_indices) return "float64 rank 1..3 template, count<=axis0 -> int64[count]; template payload ignored";
    return "float64 rank 1..3 values, unique int64 row indices, literal output length -> same trailing shape";
  }
  switch (s.op) {
  case Op::affine_contract:
    return "float64 X[...,F] rank 2/3, coefficients[F,A], scalar or initial[A] -> X leading axes plus A";
  case Op::expm1:
    return "float64 scalar or rank 1..3 -> same shape and axes";
  case Op::cholesky:
    return "float64 square matrix -> same-shaped lower triangular matrix";
  case Op::solve_multi:
  case Op::triangular_solve:
    return "square float64 coefficient matrix; matching RHS rows; solve_multi requires matrix RHS, triangular_solve permits vector/matrix -> RHS shape";
  case Op::symmetric_eigh:
    return "float64 square matrix -> shared [N,N+1] eigenpair rows; Typed IR nominal symmetric_eigh";
  case Op::eigenvalues:
  case Op::eigenvectors:
    return "nominal symmetric_eigh [N,N+1] -> borrowed eigenvalues [N] / column eigenvectors [N,N]";
  case Op::take_axis:
    return "rank 1..3 and int64 rank 1 indices -> selected axis length equals index count";
  case Op::slice_axis:
    return "rank 1..3; explicit axis,start,stop,step -> borrowed strict half-open slice";
  case Op::concatenate:
    return "two same-dtype rank 1..3 arrays; non-concatenated dimensions equal";
  case Op::stack:
    return "two equal-shaped same-dtype rank 0..2 values; insert length-two axis";
  case Op::broadcast_like:
    return "same-dtype rank 1..3 arrays; right-aligned unit/equal dimensions -> borrowed template-shaped view";
  case Op::sum_axis:
  case Op::mean_axis:
    return "float64 rank 1..3; remove explicit axis, without keepdims";
  case Op::swap_axes:
    return "rank 1..3; swap explicit axes -> borrowed view";
  case Op::state_select:
    return "equal-shaped rank 1 condition/valid masks; branches scalar code or equal-shaped int64 series -> int64 state series";
  case Op::equal:
  case Op::not_equal:
    return "numeric rank 0..3 comparison; exact int64 with int64 or integral float64 scalar |x|<=2^53-1; equal shapes or scalar broadcast -> mask";
  case Op::recursive_filter_adaptive:
    return "float64 rank 1; scalar/equal-shaped alpha; equal-shaped update/reset masks -> same length";
  case Op::linear_filter2:
    return "float64 rank 1; finite scalar b0/b1/a1/a2; equal-shaped reset mask -> same length";
  case Op::scalar_kalman:
    return "float64 rank 1 plus scalar variances -> float64 [T,2] shared estimate/variance state";
  case Op::state_estimate:
  case Op::state_variance:
    return "float64 [T,2] Kalman state -> borrowed rank 1 field; Typed IR requires kalman_series nominal type";
  case Op::state_hysteresis:
  case Op::drawdown_cycle_state:
    return "aligned float64 input sequences and scalar thresholds -> int64 time series state";
  case Op::state_confirm:
    return "int64 state series plus scalar confirmation/min_hold -> int64 time series state";
  case Op::state_continuous:
    return "aligned int64 candidate/initial and float64 observed -> int64 [T,3] state/evidence/pending";
  case Op::continuous_state_values:
  case Op::continuous_state_evidence:
  case Op::continuous_state_pending:
    return "int64 [T,3] continuous state -> borrowed rank 1 field; Typed IR requires nominal type";
  case Op::local_extrema:
    return "float64 price series plus scalar left/right/head/tail -> int64 aligned extrema events";
  case Op::ps_filter:
    return "float64 prices, aligned int64 events and scalar constraints -> int64 aligned events";
  case Op::between_events:
    return "int64 event series -> int64 [T,2] start/end interval boundaries";
  case Op::segment_starts:
  case Op::segment_ends:
    return "int64 [T,2] segment boundaries -> borrowed rank 1 field; Typed IR requires nominal type";
  case Op::phase_direction:
    return "int64 event series and int64 [T,2] segments -> int64 time series phase direction";
  case Op::drawdown_cycle_reference:
    return "int64 phases, float64 changes, int64 [T,2] segments and scalar stress -> int64 time series drawdown cycle";
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
    return "numeric/mask rank 0..3; equal shapes or explicit scalar broadcast; "
           "logical "
           "requires equal masks; divide_or_default requires "
           "rank 1";
  case Family::reduction:
    return "rank 1/2/3 -> scalar; *_time/*_asset require matrix and reduce axis 0/1; "
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
  case Op::cumulative_product_axis:
    return "initial-seeded ascending inclusive multiplication; IEEE NaN/Inf/signed zero propagate; no dropping, reseeding or reassociation";
  case Op::cumulative_max_axis:
    return "initial-seeded strict current>state update; ties retain state, current NaN does not update, initial NaN persists; no implicit zero seed";
  case Op::quantiles_axis:
    return "one NaN-last sort per line, without filtering; finite probabilities in [0,1]; integer rank/singleton returns value directly, otherwise lo*(1-f)+hi*f; empty axis with real output -> INSUFFICIENT_SAMPLE";
  case Op::affine_contract:
    return "initial-first ascending last-axis sum; separate coefficients*values then add; IEEE NaN/Inf/overflow propagate; empty contracted axis returns initial";
  case Op::expm1:
    return "stable exp(x)-1 via std::expm1; signed zero preserved; NaN/+Inf preserved, -Inf -> -1; finite overflow -> +Inf";
  case Op::cholesky:
    return "finite exactly symmetric SPD input; no regularization; invalid symmetry/pivot -> DOMAIN_ERROR";
  case Op::solve_multi:
    return "finite coefficients/RHS; partial pivoting; absolute pivot<=1e-14 -> SINGULAR_MATRIX";
  case Op::triangular_solve:
    return "only consumed triangle is validated; unit diagonal ignores stored diagonal; finite RHS; absolute pivot<=1e-14 -> SINGULAR_MATRIX";
  case Op::symmetric_eigh:
    return "finite exactly symmetric input; scaled cyclic Jacobi; ascending stable eigenvalues; deterministic vector signs; exhausted sweeps -> NON_CONVERGENCE";
  case Op::sum_axis:
  case Op::mean_axis:
    return "IEEE ordered sum, no drop/fill; empty sum=0; empty mean with nonempty output -> INSUFFICIENT_SAMPLE";
  case Op::state_select:
    return "valid=false emits -1; otherwise selects exact integer branch, including valid zero or explicit -1; no float-array coercion";
  case Op::cos:
    return "radians; NaN/+Inf/-Inf -> NaN; finite input uses scalar std::cos";
  case Op::recursive_filter_adaptive:
    return "reset mask acts before row; missing input skips update; finite-input count independent of update mask; only consumed alpha validated; emit_policy=0 hold/1 NaN; no implicit gap reset";
  case Op::linear_filter2:
    return "reset mask acts before row; nonfinite input emits NaN without advancing state; nonfinite recurrence resets state and warmup; no implicit input-gap reset";
  case Op::scalar_kalman:
    return "first finite observation seeds estimate, retains initial variance; missing emits NaN for both fields, holds state, does not advance variance; nonfinite update fails";
  case Op::state_estimate:
  case Op::state_variance:
    return "exact field view; preserves all values including NaN; keeps underlying shared state owner alive";
  case Op::state_hysteresis:
    return "nonfinite emits unknown=-1 and holds active state; entry/exit thresholds inclusive; neutral upper entry takes precedence";
  case Op::state_confirm:
    return "unknown=-1 emits unknown, clears pending but retains active/duration; confirmation>=threshold and active_duration>min_hold; no backdating";
  case Op::state_continuous:
    return "candidate=-1 means no proposal and holds state; observed must be finite positive; confirmed switch at count>=confirmation; evidence and pending are separate fields";
  case Op::drawdown_cycle_state:
    return "invalid/missing input resets; unknown=-1; stress drawdown<=-stress, recovery rebound>=threshold, exit drawdown>=-exit; one transition per row";
  case Op::continuous_state_values:
  case Op::continuous_state_evidence:
  case Op::continuous_state_pending:
    return "exact int64 field view with shared owner; unknown state=-1; evidence and pending retain their own field semantics";
  case Op::local_extrema:
  case Op::ps_filter:
    return "missing=-2, no event=0, extrema events=-1/+1; full-input retrospective event selection";
  case Op::between_events:
  case Op::segment_starts:
  case Op::segment_ends:
    return "unknown segment endpoint=-1; complete event-pair boundaries retained; future-dependent segment knowledge";
  case Op::phase_direction:
    return "unknown=-1, up=0, down=1; uses complete phases with shared endpoints";
  case Op::drawdown_cycle_reference:
    return "unknown=-1, normal=0, recovery=1, stress=2; complete phases fill (left,right]; stress threshold includes equality";
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
  if (latent_operator(s.op))
    return "finite numeric inputs required except scatter preserves values and sampling ignores template payload; category -1 unknown; Gaussian variance floor 1e-12; EM mass floor 1e-8 and variance floor 1e-5; no implicit probability normalization";
  if (matrix_foundation(s.op))
    return "exact dtype/value preservation; negative/out-of-range indices rejected; no implicit axis alignment or fill";
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
  if(s.op==Op::gaussian_mixture_em) return "max_iterations=100 [1,10000]; tolerance=1e-6 >=0; explicit initialization; zero tolerance disables early stop";
  if(s.op==Op::gaussian_hmm_em) return "max_iterations=100 [1,10000]; tolerance=1e-5 >=0; explicit initialization; zero tolerance disables early stop";
  if(s.op==Op::variance_axis) return "ddof=0; nonnegative integer < axis length";
  if(s.op==Op::hmm_filter) return "predict_first=0; 0 consumes first observation directly, 1 first applies transition; subsequent observations always transition";
  if(s.op==Op::scatter_rows) return "fill=NaN; exact row indices unique and in range";
  if(s.op==Op::sample_unique_indices) return "seed=0; integer within +/- (2^53-1); local Park-Miller and deterministic linear collision advance";
  switch (s.op) {
  case Op::symmetric_eigh:
    return "tolerance=1e-12 in [1e-15,1e-3]; max_sweeps=32 integer [1,128]";
  case Op::triangular_solve:
    return "lower=1, unit_diagonal=0; both integer 0/1";
  case Op::slice_axis:
    return "all required; 0<=start<=stop<=axis extent; step>=1; no negative indices/clamping";
  case Op::state_select:
    return "all required; scalar branch finite integer [-1,2^53-1], int64 branch values>=-1; -1 is explicit unknown/no proposal; validity stays separate";
  case Op::recursive_filter_adaptive:
    return "initial=0 finite; seed_mode=1 (0 initial consumed, 1 first finite, 2 first eligible); min_periods=1 >=1; emit_policy=1 (0 hold, 1 NaN); consumed alpha finite in [0,1]; update previous+=alpha*(x-previous)";
  case Op::linear_filter2:
    return "min_periods=1 >=1; bootstrap=1 (first two finite values seed outputs), 0 uses zero initial states; y=b0*x+b1*previous_x+a1*previous_y+a2*older_y";
  case Op::scalar_kalman:
    return "initial_variance=1; finite process_variance>=0, measurement_variance>0, initial_variance>=0; no implicit floors or observation warmup";
  case Op::state_hysteresis:
    return "all finite thresholds required; initial neutral=1, upper=0, lower=2; thresholds retain caller ordering";
  case Op::state_confirm:
    return "confirmation,min_hold required integers in [1,2^31-1]; codes>=-1 exact int64";
  case Op::state_continuous:
    return "all required; confirmation integer [1,252], state_count integer [2,12]; initial[0]>=0; candidate/initial in [-1,state_count-1]";
  case Op::drawdown_cycle_state:
    return "all required; 0<=exit<stress<1, 0<rebound<1; valid is caller-composed complete-window mask; drawdown in [-1,0]";
  case Op::local_extrema:
    return "all required; left/right integer [1,5000], head/tail [0,5000]; strict preceding and inclusive following comparisons choose earliest plateau";
  case Op::ps_filter:
    return "all required; min_phase integer [1,10000], min_cycle [2,20000], amplitude>=0 finite; amplitude exemption is strict >";
  case Op::drawdown_cycle_reference:
    return "all required; 0<stress<1; completed decline<=-stress marks stress and following rise recovery; membership (left,right]";
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

const char *granularity(const Spec &s) {
  if (axis_path_operator(s.op)) return "primitive";
  if (s.op == Op::affine_contract || s.op == Op::expm1) return "primitive";
  if(s.op==Op::gaussian_mixture_em || s.op==Op::gaussian_hmm_em || s.op==Op::hmm_filter || s.op==Op::hmm_smooth) return "coupled_kernel";
  if(latent_operator(s.op)) return "primitive";
  if (s.op >= Op::cholesky && s.op <= Op::symmetric_eigh) return "coupled_kernel";
  if (matrix_foundation(s.op)) return "primitive";
  switch (s.op) {
  case Op::recursive_filter_adaptive:
  case Op::linear_filter2:
  case Op::scalar_kalman:
  case Op::state_hysteresis:
  case Op::state_confirm:
  case Op::state_continuous:
  case Op::drawdown_cycle_state:
  case Op::local_extrema:
  case Op::ps_filter:
  case Op::drawdown_cycle_reference:
    return "coupled_kernel";
  case Op::cos:
  case Op::state_estimate:
  case Op::state_variance:
  case Op::continuous_state_values:
  case Op::continuous_state_evidence:
  case Op::continuous_state_pending:
  case Op::between_events:
  case Op::segment_starts:
  case Op::segment_ends:
  case Op::phase_direction:
  case Op::state_select:
    return "primitive";
  default:
    return "unspecified";
  }
}

bool series_record_projection(Op op) noexcept {
  switch (op) {
  case Op::state_estimate:
  case Op::state_variance:
  case Op::continuous_state_values:
  case Op::continuous_state_evidence:
  case Op::continuous_state_pending:
  case Op::segment_starts:
  case Op::segment_ends:
    return true;
  default:
    return false;
  }
}

const char *temporal_dependency(const Spec &s) {
  if (axis_path_operator(s.op))
    return s.op == Op::quantiles_axis ? "full_input" : "causal";
  if (s.op == Op::affine_contract || s.op == Op::expm1) return "causal";
  if(s.op==Op::hmm_filter) return "causal";
  if(latent_operator(s.op)) return "full_input";
  if (matrix_foundation(s.op)) return "full_input";
  if (s.op == Op::state_select) return "causal";
  if (s.op >= Op::local_extrema && s.op <= Op::drawdown_cycle_reference)
    return "full_input";
  if (s.op >= Op::cos && s.op <= Op::drawdown_cycle_state)
    return "causal";
  return "unspecified";
}
} // namespace calmetrics_engine::ops
