#include "calmetrics_engine/typed_ir.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace calmetrics_engine::typed {
namespace {
using O = ops::Op;

[[noreturn]] void fail(const std::string &code, const std::string &message) {
  throw Error(code + ": " + message);
}

const std::unordered_set<std::string> &supported_semantics() {
  static const std::unordered_set<std::string> values{
      "dimensionless",          "return_decimal", "rate_decimal",
      "adjusted_nav",           "adjusted_market_price",
      "reported_nav",           "raw_market_price",
      "volume",                 "currency_amount", "count",
      "calendar_days",          "date",            "mask"};
  return values;
}
bool derived_semantic(const std::string &value) {
  return value.rfind("derived:", 0) == 0 || value.rfind("squared:", 0) == 0 ||
         value.rfind("inverse:", 0) == 0;
}
bool price_semantic(const std::string &value) {
  return value == "adjusted_nav" || value == "adjusted_market_price" ||
         value == "reported_nav" || value == "raw_market_price";
}
bool additive_rate(const std::string &value) {
  return value == "return_decimal" || value == "rate_decimal";
}
bool path_level(const std::string &value) {
  return value == "dimensionless" || price_semantic(value);
}
bool one_dimensional(const ValueType &value) {
  return value.kind == ValueKind::series || value.kind == ValueKind::vector;
}
bool numeric_array(const ValueType &value) {
  return value.is_numeric() && !value.is_scalar();
}
ValueType from_axes(const std::vector<std::string> &axes,
                    const std::vector<std::string> &shape,
                    const std::string &semantic,
                    const std::string &basis = {}, DType dtype = DType::float64) {
  if (axes.empty()) {
    auto out = ValueType::scalar(semantic, basis);
    out.dtype = dtype;
    if (dtype == DType::boolean) {
      out.semantic_dimension = "mask";
      out.price_basis.clear();
    }
    out.validate();
    return out;
  }
  ValueType out;
  if (axes == std::vector<std::string>{"time"})
    out = ValueType::series(shape.at(0), semantic, basis);
  else if (axes == std::vector<std::string>{"asset"})
    out = ValueType::vector(shape.at(0), semantic, basis);
  else if (axes == std::vector<std::string>{"time", "window"})
    out = ValueType::window(shape.at(0), shape.at(1), semantic, basis);
  else
    out = ValueType::matrix(axes, shape, semantic, basis);
  out.dtype = dtype;
  if (dtype == DType::boolean) {
    out.semantic_dimension = "mask";
    out.price_basis.clear();
  }
  out.validate();
  return out;
}
ValueType elementwise_structure(const std::string &op, const ValueType &left,
                                const ValueType &right) {
  if (!left.is_numeric() || !right.is_numeric())
    fail("TYPE_MISMATCH", op + " requires numeric inputs");
  if (left.is_scalar())
    return right;
  if (right.is_scalar())
    return left;
  if (left.axes != right.axes)
    fail("AXIS_MISMATCH", op + " named axes do not match");
  if (left.shape != right.shape)
    fail("SHAPE_MISMATCH", op + " symbolic shapes do not match");
  return left;
}
std::pair<std::string, std::string>
compatible_semantics(const std::string &op, const ValueType &left,
                     const ValueType &right) {
  std::string semantic;
  if (left.is_scalar() && left.semantic_dimension == "dimensionless")
    semantic = right.semantic_dimension;
  else if (right.is_scalar() && right.semantic_dimension == "dimensionless")
    semantic = left.semantic_dimension;
  else if (left.semantic_dimension == right.semantic_dimension)
    semantic = left.semantic_dimension;
  else if (additive_rate(left.semantic_dimension) &&
           additive_rate(right.semantic_dimension))
    semantic = "return_decimal";
  else
    fail("SEMANTIC_DIMENSION_MISMATCH",
         op + " semantic dimensions are incompatible");
  std::string basis = !left.price_basis.empty() ? left.price_basis
                                                 : right.price_basis;
  if (price_semantic(semantic) && !left.price_basis.empty() &&
      !right.price_basis.empty() && left.price_basis != right.price_basis)
    fail("PRICE_BASIS_MISMATCH", op + " price bases do not match");
  return {semantic, basis};
}
ValueType additive(const std::string &op, const ValueType &left,
                   const ValueType &right) {
  auto structure = elementwise_structure(op, left, right);
  auto semantics = compatible_semantics(op, left, right);
  return structure.with_semantics(semantics.first, semantics.second);
}
std::pair<std::string, std::string> product_semantics(const ValueType &left,
                                                       const ValueType &right) {
  if (left.semantic_dimension == "dimensionless")
    return {right.semantic_dimension, right.price_basis};
  if (right.semantic_dimension == "dimensionless")
    return {left.semantic_dimension, left.price_basis};
  if (left.semantic_dimension == right.semantic_dimension)
    return {"squared:" + left.semantic_dimension, {}};
  return {"derived:" + left.semantic_dimension + "*" + right.semantic_dimension,
          {}};
}
void require_scalar_parameter(const ValueType &value, const std::string &name,
                              bool allow_count = true) {
  if (!value.is_scalar() || !value.is_numeric())
    fail("TYPE_MISMATCH", name + " must be a numeric scalar");
  if (value.semantic_dimension != "dimensionless" &&
      (!allow_count || value.semantic_dimension != "count"))
    fail("SEMANTIC_DIMENSION_MISMATCH",
         name + " must be dimensionless/count");
}
ValueType reduce_all(const ValueType &value) {
  if (!numeric_array(value))
    fail("TYPE_MISMATCH", "reduction requires a numeric array");
  return ValueType::scalar(value.semantic_dimension, value.price_basis);
}
void require_same_one_dimensional(const std::string &name,
                                  const ValueType &left,
                                  const ValueType &right) {
  if (!one_dimensional(left) || !one_dimensional(right) ||
      !same_type(left, right) || !left.is_numeric())
    fail("TYPE_MISMATCH", name + " requires matching one-dimensional inputs");
}
std::string shrink(const std::string &dimension, bool explicit_period) {
  if (!explicit_period) {
    try {
      const auto value = std::stoll(dimension);
      if (value < 2)
        fail("INSUFFICIENT_SAMPLE", "lag/difference requires two values");
      return std::to_string(value - 1);
    } catch (const std::invalid_argument &) {
    } catch (const std::out_of_range &) {
    }
  }
  return dimension + (explicit_period ? "-n" : "-1");
}
} // namespace

ValueType ValueType::scalar(std::string semantic, std::string basis) {
  ValueType out;
  out.kind = ValueKind::scalar;
  out.semantic_dimension = normalize_semantic_dimension(std::move(semantic));
  out.price_basis = std::move(basis);
  out.validate();
  return out;
}
ValueType ValueType::series(std::string length, std::string semantic,
                            std::string basis) {
  ValueType out;
  out.kind = ValueKind::series;
  out.axes = {"time"};
  out.shape = {std::move(length)};
  out.semantic_dimension = normalize_semantic_dimension(std::move(semantic));
  out.price_basis = std::move(basis);
  out.validate();
  return out;
}
ValueType ValueType::vector(std::string length, std::string semantic,
                            std::string basis) {
  ValueType out;
  out.kind = ValueKind::vector;
  out.axes = {"asset"};
  out.shape = {std::move(length)};
  out.semantic_dimension = normalize_semantic_dimension(std::move(semantic));
  out.price_basis = std::move(basis);
  out.validate();
  return out;
}
ValueType ValueType::matrix(std::vector<std::string> a,
                            std::vector<std::string> s, std::string semantic,
                            std::string basis) {
  ValueType out;
  out.kind = ValueKind::matrix;
  out.axes = std::move(a);
  out.shape = std::move(s);
  out.semantic_dimension = normalize_semantic_dimension(std::move(semantic));
  out.price_basis = std::move(basis);
  out.validate();
  return out;
}
ValueType ValueType::window(std::string length, std::string width,
                            std::string semantic, std::string basis) {
  ValueType out;
  out.kind = ValueKind::window;
  out.axes = {"time", "window"};
  out.shape = {std::move(length), std::move(width)};
  out.semantic_dimension = normalize_semantic_dimension(std::move(semantic));
  out.price_basis = std::move(basis);
  out.validate();
  return out;
}
ValueType ValueType::mask(std::vector<std::string> a,
                          std::vector<std::string> s) {
  ValueType out;
  out.kind = a.empty() ? ValueKind::scalar
                       : (a == std::vector<std::string>{"time"}
                              ? ValueKind::series
                              : (a == std::vector<std::string>{"asset"}
                                     ? ValueKind::vector
                                     : ValueKind::matrix));
  out.dtype = DType::boolean;
  out.axes = std::move(a);
  out.shape = std::move(s);
  out.semantic_dimension = "mask";
  out.validate();
  return out;
}
ValueType ValueType::record(std::string tag, std::vector<std::string> names) {
  ValueType out;
  out.kind = ValueKind::record;
  out.record_tag = std::move(tag);
  out.fields = std::move(names);
  out.validate();
  return out;
}
void ValueType::validate() const {
  static const std::unordered_set<std::string> axes_allowed{"time", "asset",
                                                             "window"};
  if (axes.size() != shape.size())
    fail("TYPE_MISMATCH", "axes and symbolic shape ranks differ");
  for (const auto &axis : axes)
    if (!axes_allowed.count(axis))
      fail("AXIS_MISMATCH", "unsupported named axis: " + axis);
  for (const auto &dimension : shape)
    if (dimension.empty())
      fail("SHAPE_MISMATCH", "symbolic dimensions cannot be empty");
  if (semantic_dimension.empty() ||
      (!supported_semantics().count(semantic_dimension) &&
       !derived_semantic(semantic_dimension)))
    fail("SEMANTIC_DIMENSION_MISMATCH",
         "unsupported semantic dimension: " + semantic_dimension);
  if (dtype == DType::boolean &&
      (semantic_dimension != "mask" || !price_basis.empty()))
    fail("TYPE_MISMATCH", "boolean values must use mask semantics");
  const auto rank = axes.size();
  if (kind == ValueKind::scalar && rank != 0)
    fail("RANK_MISMATCH", "scalar must be rank zero");
  if (kind == ValueKind::series &&
      (axes != std::vector<std::string>{"time"}))
    fail("AXIS_MISMATCH", "series must use the time axis");
  if (kind == ValueKind::vector &&
      (axes != std::vector<std::string>{"asset"}))
    fail("AXIS_MISMATCH", "vector must use the asset axis");
  if (kind == ValueKind::matrix && rank != 2)
    fail("RANK_MISMATCH", "matrix must be rank two");
  if (kind == ValueKind::window &&
      (axes != std::vector<std::string>{"time", "window"} ||
       dtype != DType::float64))
    fail("TYPE_MISMATCH", "window must be numeric time/window rank two");
  if (kind == ValueKind::record &&
      (!axes.empty() || fields.empty() || record_tag.empty()))
    fail("TYPE_MISMATCH", "record requires named scalar fields");
}
ValueType ValueType::with_semantics(std::string semantic,
                                    std::string basis) const {
  auto out = *this;
  out.semantic_dimension = normalize_semantic_dimension(std::move(semantic));
  out.price_basis = std::move(basis);
  out.validate();
  return out;
}
ValueType ValueType::as_mask() const { return mask(axes, shape); }

const char *kind_name(ValueKind kind) noexcept {
  switch (kind) {
  case ValueKind::scalar:
    return "scalar";
  case ValueKind::series:
    return "series";
  case ValueKind::vector:
    return "vector";
  case ValueKind::matrix:
    return "matrix";
  case ValueKind::window:
    return "window";
  case ValueKind::record:
    return "record";
  }
  return "invalid";
}
const char *dtype_name(DType dtype) noexcept {
  return dtype == DType::float64 ? "float64" : "bool";
}
std::string display(const ValueType &value) {
  if (value.kind == ValueKind::record)
    return "record<" + value.record_tag + ">";
  if (value.is_scalar())
    return value.is_mask() ? "mask" : "scalar";
  std::string result = value.is_mask() ? "mask" : kind_name(value.kind);
  result += "<";
  for (std::size_t i = 0; i < value.axes.size(); ++i) {
    if (i)
      result += ",";
    result += value.axes[i];
  }
  result += ">[";
  for (std::size_t i = 0; i < value.shape.size(); ++i) {
    if (i)
      result += ",";
    result += value.shape[i];
  }
  return result + "]";
}
std::string normalize_semantic_dimension(std::string value) {
  static const std::unordered_map<std::string, std::string> aliases{
      {"number", "dimensionless"}, {"return", "return_decimal"},
      {"weight", "dimensionless"}, {"price", "raw_market_price"},
      {"price_change", "raw_market_price"}, {"currency", "currency_amount"}};
  if (const auto it = aliases.find(value); it != aliases.end())
    value = it->second;
  if (!supported_semantics().count(value) && !derived_semantic(value))
    fail("SEMANTIC_DIMENSION_MISMATCH",
         "unsupported semantic dimension: " + value);
  return value;
}

std::string canonical_operator_name(std::string_view name) {
  static const std::unordered_map<std::string_view, std::string_view> aliases{
      {"sub", "subtract"},
      {"mul", "multiply"},
      {"safe_divide", "divide"},
      {"elementwise_min", "minimum"},
      {"elementwise_max", "maximum"},
      {"abs", "absolute"},
      {"eq", "equal"},
      {"ne", "not_equal"},
      {"lt", "less_than"},
      {"le", "less_equal"},
      {"gt", "greater_than"},
      {"ge", "greater_equal"},
      {"sequence_sum", "sum"},
      {"prod", "product"},
      {"sequence_prod", "product"},
      {"sequence_mean", "mean"},
      {"min", "min_value"},
      {"max", "max_value"},
      {"var", "variance"},
      {"sequence_std", "std"},
      {"cumulative_maximum", "cumulative_max"},
      {"cumulative_minimum", "cumulative_min"},
      {"skew", "skewness"},
      {"kurtosis_excess", "excess_kurtosis"},
      {"mad", "mean_absolute_deviation"},
      {"rms", "root_mean_square"},
      {"masked_sum", "sum_where"},
      {"masked_mean", "mean_where"},
      {"masked_variance", "variance_where"},
      {"masked_std", "std_where"},
      {"masked_min", "min_where"},
      {"masked_max", "max_where"},
      {"masked_median", "median_where"},
      {"masked_quantile", "quantile_where"},
      {"masked_count", "count_true"},
      {"cov", "covariance"},
      {"corr", "correlation"}};
  if (const auto it = aliases.find(name); it != aliases.end())
    return std::string(it->second);
  return std::string(name);
}

bool same_structure(const ValueType &left, const ValueType &right) noexcept {
  return left.kind == right.kind && left.axes == right.axes &&
         left.shape == right.shape && left.dtype == right.dtype;
}
bool same_type(const ValueType &left, const ValueType &right) noexcept {
  return same_structure(left, right) &&
         left.semantic_dimension == right.semantic_dimension &&
         left.price_basis == right.price_basis &&
         left.record_tag == right.record_tag && left.fields == right.fields;
}

ValueType infer(const ops::Spec &spec, const std::vector<ValueType> &v) {
  const auto op = spec.op;
  if (v.size() < spec.min_args || v.size() > spec.max_args)
    fail("ARITY_MISMATCH", std::string(spec.name) + " has invalid arity");

  if (spec.family == ops::Family::elementwise) {
    if (op == O::logical_and || op == O::logical_or) {
      if (!v[0].is_mask() || !same_type(v[0], v[1]))
        fail("TYPE_MISMATCH", "logical operators require matching masks");
      return v[0];
    }
    if (op == O::logical_not) {
      if (!v[0].is_mask())
        fail("TYPE_MISMATCH", "logical_not requires a mask");
      return v[0];
    }
    if (op == O::finite_mask) {
      if (!v[0].is_numeric())
        fail("TYPE_MISMATCH", "finite_mask requires numeric input");
      return v[0].as_mask();
    }
    if (op == O::where) {
      if (!v[0].is_mask())
        fail("TYPE_MISMATCH", "where requires a mask");
      auto out = additive("where", v[1], v[2]);
      if (v[0].is_scalar())
        return out;
      if (out.is_scalar())
        return from_axes(v[0].axes, v[0].shape, out.semantic_dimension,
                         out.price_basis);
      if (v[0].axes != out.axes)
        fail("AXIS_MISMATCH", "where mask axes do not match");
      if (v[0].shape != out.shape)
        fail("SHAPE_MISMATCH", "where mask shape does not match");
      return out;
    }
    if (op == O::clip) {
      if (!v[0].is_numeric() || !v[1].is_scalar() || !v[2].is_scalar())
        fail("TYPE_MISMATCH", "clip requires numeric, scalar, scalar");
      compatible_semantics("clip", v[0], v[1]);
      compatible_semantics("clip", v[0], v[2]);
      return v[0];
    }
    if (op == O::divide_or_default) {
      if (v.size() != 3 || v[0].kind != ValueKind::series ||
          v[1].kind != ValueKind::series || !v[2].is_scalar())
        fail("TYPE_MISMATCH",
             "divide_or_default requires series, series, scalar");
      auto out = infer(ops::lookup("divide"), {v[0], v[1]});
      compatible_semantics("divide_or_default", out, v[2]);
      return out;
    }
    if (op == O::equal || op == O::not_equal || op == O::less_than ||
        op == O::less_equal || op == O::greater_than ||
        op == O::greater_equal) {
      auto out = elementwise_structure("comparison", v[0], v[1]);
      compatible_semantics("comparison", v[0], v[1]);
      return out.as_mask();
    }
    if (op == O::add || op == O::subtract || op == O::minimum ||
        op == O::maximum)
      return additive(spec.name, v[0], v[1]);
    if (op == O::multiply) {
      auto out = elementwise_structure("multiply", v[0], v[1]);
      auto semantics = product_semantics(v[0], v[1]);
      return out.with_semantics(semantics.first, semantics.second);
    }
    if (op == O::divide) {
      auto out = elementwise_structure("divide", v[0], v[1]);
      if (v[1].is_scalar() &&
          (v[1].semantic_dimension == "dimensionless" ||
           v[1].semantic_dimension == "count")) {
        if (v[0].semantic_dimension == "count" &&
            v[1].semantic_dimension == "count")
          return out.with_semantics("dimensionless");
        return out.with_semantics(v[0].semantic_dimension, v[0].price_basis);
      }
      if (v[0].semantic_dimension == v[1].semantic_dimension) {
        if (!v[0].price_basis.empty() && !v[1].price_basis.empty() &&
            v[0].price_basis != v[1].price_basis)
          fail("PRICE_BASIS_MISMATCH", "divide price bases do not match");
        return out.with_semantics("dimensionless");
      }
      return out.with_semantics("derived:" + v[0].semantic_dimension + "/" +
                                v[1].semantic_dimension);
    }
    if (op == O::power) {
      elementwise_structure("power", v[0], v[1]);
      if (!v[1].is_scalar() || v[1].semantic_dimension != "dimensionless")
        fail("TYPE_MISMATCH", "power exponent must be dimensionless scalar");
      return v[0];
    }
    if (v.size() != 1 || !v[0].is_numeric())
      fail("TYPE_MISMATCH", std::string(spec.name) + " requires numeric input");
    if (op == O::sign)
      return v[0].with_semantics("dimensionless");
    if (op == O::normal_pdf || op == O::normal_ppf) {
      if (v[0].semantic_dimension != "dimensionless")
        fail("SEMANTIC_DIMENSION_MISMATCH",
             "normal_pdf/normal_ppf require dimensionless input");
      return v[0].with_semantics("dimensionless");
    }
    if (op == O::log || op == O::exp) {
      if (v[0].semantic_dimension != "dimensionless" &&
          v[0].semantic_dimension != "return_decimal" &&
          v[0].semantic_dimension != "rate_decimal")
        fail("SEMANTIC_DIMENSION_MISMATCH",
             "log/exp require dimensionless decimal input");
      return v[0].with_semantics("dimensionless");
    }
    if (op == O::reciprocal)
      return v[0].with_semantics(
          v[0].semantic_dimension == "dimensionless"
              ? "dimensionless"
              : "inverse:" + v[0].semantic_dimension);
    if (op == O::sqrt) {
      auto semantic = v[0].semantic_dimension;
      if (semantic.rfind("squared:", 0) == 0)
        semantic = semantic.substr(8);
      return v[0].with_semantics(semantic, v[0].price_basis);
    }
    return v[0];
  }

  if (spec.family == ops::Family::reduction) {
    if (op == O::count_true || op == O::max_consecutive_true) {
      if (!v[0].is_mask() || v[0].is_scalar())
        fail("TYPE_MISMATCH", "mask reduction requires a mask array");
      return ValueType::scalar("count");
    }
    if (std::string(spec.name).find("_where") != std::string::npos) {
      if (!numeric_array(v[0]) || !v[1].is_mask() ||
          v[0].axes != v[1].axes || v[0].shape != v[1].shape)
        fail("TYPE_MISMATCH", "masked reduction inputs do not match");
      if (v.size() == 3)
        require_scalar_parameter(v[2], "probability", false);
      auto out = ValueType::scalar(v[0].semantic_dimension, v[0].price_basis);
      if (op == O::variance_where)
        out = out.with_semantics("squared:" + v[0].semantic_dimension);
      return out;
    }
    if (op == O::sum_time || op == O::mean_time ||
        op == O::product_time || op == O::variance_time ||
        op == O::std_time || op == O::min_time || op == O::max_time) {
      if (v[0].kind != ValueKind::matrix ||
          v[0].axes != std::vector<std::string>{"time", "asset"})
        fail("TYPE_MISMATCH", "time reduction requires time/asset matrix");
      auto out = ValueType::vector(v[0].shape[1], v[0].semantic_dimension,
                                   v[0].price_basis);
      if (op == O::variance_time)
        out = out.with_semantics("squared:" + v[0].semantic_dimension);
      return out;
    }
    if (op == O::sum_asset || op == O::mean_asset ||
        op == O::product_asset || op == O::variance_asset ||
        op == O::std_asset || op == O::min_asset || op == O::max_asset) {
      if (v[0].kind != ValueKind::matrix ||
          v[0].axes != std::vector<std::string>{"time", "asset"})
        fail("TYPE_MISMATCH", "asset reduction requires time/asset matrix");
      auto out = ValueType::series(v[0].shape[0], v[0].semantic_dimension,
                                   v[0].price_basis);
      if (op == O::variance_asset)
        out = out.with_semantics("squared:" + v[0].semantic_dimension);
      return out;
    }
    if (v[0].kind == ValueKind::window) {
      if (op != O::mean && op != O::std && op != O::variance &&
          op != O::min_value && op != O::max_value)
        fail("TYPE_MISMATCH",
             "rolling_window supports mean/std/variance/min/max reducers");
      auto out = ValueType::series(v[0].shape[0], v[0].semantic_dimension,
                                   v[0].price_basis);
      if (op == O::variance)
        out = out.with_semantics("squared:" + v[0].semantic_dimension);
      return out;
    }
    if (v.size() > 1)
      require_scalar_parameter(v[1], "reduction parameter");
    auto out = reduce_all(v[0]);
    if (op == O::variance)
      out = out.with_semantics("squared:" + v[0].semantic_dimension);
    if (op == O::skewness || op == O::excess_kurtosis)
      out = out.with_semantics("dimensionless");
    if (op == O::argmin || op == O::argmax)
      out = out.with_semantics("count");
    return out;
  }

  if (spec.family == ops::Family::sequence) {
    if (!one_dimensional(v[0]) || !v[0].is_numeric())
      fail("TYPE_MISMATCH", "sequence operator requires numeric 1D input");
    if (op == O::first || op == O::last)
      return ValueType::scalar(v[0].semantic_dimension, v[0].price_basis);
    if (op == O::length)
      return ValueType::scalar("count");
    if (op == O::lag || op == O::difference) {
      if (v.size() == 2)
        require_scalar_parameter(v[1], "periods");
      auto out = v[0];
      out.shape[0] = shrink(v[0].shape[0], v.size() == 2);
      return out;
    }
    if (op == O::new_high_mask) {
      if (!path_level(v[0].semantic_dimension))
        fail("SEMANTIC_DIMENSION_MISMATCH",
             "new_high_mask requires level semantics");
      return v[0].as_mask();
    }
    if (op == O::drawdown_series) {
      if (!path_level(v[0].semantic_dimension))
        fail("SEMANTIC_DIMENSION_MISMATCH",
             "drawdown_series requires level semantics");
      return from_axes(v[0].axes, v[0].shape, "return_decimal");
    }
    if (op == O::cumulative_return)
      return v[0].with_semantics("return_decimal");
    return v[0];
  }

  if (spec.family == ops::Family::rolling) {
    if (v[0].kind != ValueKind::series || !v[0].is_numeric())
      fail("TYPE_MISMATCH", "rolling operator requires numeric time series");
    for (std::size_t i = 1; i < v.size(); ++i)
      require_scalar_parameter(v[i], "rolling parameter");
    return v[0];
  }

  if (spec.family == ops::Family::matrix) {
    if (op == O::dot) {
      require_same_one_dimensional("dot", v[0], v[1]);
      auto semantics = product_semantics(v[0], v[1]);
      return ValueType::scalar(semantics.first, semantics.second);
    }
    if (op == O::outer) {
      if (v[0].kind != ValueKind::vector || v[1].kind != ValueKind::vector)
        fail("TYPE_MISMATCH", "outer requires asset vectors");
      auto semantics = product_semantics(v[0], v[1]);
      return ValueType::matrix({"asset", "asset"},
                               {v[0].shape[0], v[1].shape[0]},
                               semantics.first);
    }
    if (op == O::transpose) {
      if (v[0].kind != ValueKind::matrix)
        fail("TYPE_MISMATCH", "transpose requires a matrix");
      return ValueType::matrix({v[0].axes[1], v[0].axes[0]},
                               {v[0].shape[1], v[0].shape[0]},
                               v[0].semantic_dimension, v[0].price_basis);
    }
    if (op == O::matmul) {
      if (v[0].kind != ValueKind::matrix || v[1].kind != ValueKind::matrix)
        fail("TYPE_MISMATCH", "matmul requires matrices");
      if (v[0].axes[1] != v[1].axes[0])
        fail("AXIS_MISMATCH", "matmul inner axes differ");
      if (v[0].shape[1] != v[1].shape[0])
        fail("SHAPE_MISMATCH", "matmul inner dimensions differ");
      auto semantics = product_semantics(v[0], v[1]);
      return ValueType::matrix({v[0].axes[0], v[1].axes[1]},
                               {v[0].shape[0], v[1].shape[1]},
                               semantics.first, semantics.second);
    }
    if (op == O::matvec) {
      if (v[0].kind != ValueKind::matrix || v[1].rank() != 1)
        fail("TYPE_MISMATCH", "matvec requires matrix and 1D vector");
      if (v[0].axes[1] != v[1].axes[0])
        fail("AXIS_MISMATCH", "matvec inner axes differ");
      if (v[0].shape[1] != v[1].shape[0])
        fail("SHAPE_MISMATCH", "matvec inner dimensions differ");
      auto semantics = product_semantics(v[0], v[1]);
      return from_axes({v[0].axes[0]}, {v[0].shape[0]}, semantics.first,
                       semantics.second);
    }
    if (op == O::diag) {
      if (v[0].kind == ValueKind::vector)
        return ValueType::matrix({"asset", "asset"},
                                 {v[0].shape[0], v[0].shape[0]},
                                 v[0].semantic_dimension, v[0].price_basis);
      if (v[0].kind == ValueKind::matrix &&
          v[0].axes == std::vector<std::string>{"asset", "asset"} &&
          v[0].shape[0] == v[0].shape[1])
        return ValueType::vector(v[0].shape[0], v[0].semantic_dimension,
                                 v[0].price_basis);
      fail("TYPE_MISMATCH", "diag requires asset vector or square matrix");
    }
    if (op == O::trace) {
      if (v[0].kind != ValueKind::matrix || v[0].axes[0] != v[0].axes[1] ||
          v[0].shape[0] != v[0].shape[1])
        fail("TYPE_MISMATCH", "trace requires square same-axis matrix");
      return ValueType::scalar(v[0].semantic_dimension, v[0].price_basis);
    }
    if (op == O::solve) {
      if (v[0].kind != ValueKind::matrix ||
          v[0].axes != std::vector<std::string>{"asset", "asset"} ||
          v[1].kind != ValueKind::vector ||
          v[0].shape[0] != v[0].shape[1] ||
          v[0].shape[1] != v[1].shape[0])
        fail("TYPE_MISMATCH", "solve dimensions do not match");
      return v[1].with_semantics("derived:" + v[1].semantic_dimension + "/" +
                                 v[0].semantic_dimension);
    }
    if (op == O::covariance || op == O::correlation) {
      if (v.size() == 1) {
        if (v[0].kind != ValueKind::matrix ||
            v[0].axes != std::vector<std::string>{"time", "asset"})
          fail("TYPE_MISMATCH",
               "covariance/correlation matrix requires time/asset matrix");
        return ValueType::matrix(
            {"asset", "asset"}, {v[0].shape[1], v[0].shape[1]},
            op == O::correlation ? "dimensionless"
                                 : "squared:" + v[0].semantic_dimension);
      }
      require_same_one_dimensional(spec.name, v[0], v[1]);
      return ValueType::scalar(
          op == O::correlation
              ? "dimensionless"
              : (v[0].semantic_dimension == v[1].semantic_dimension
                     ? "squared:" + v[0].semantic_dimension
                     : "derived:" + v[0].semantic_dimension + "*" +
                           v[1].semantic_dimension));
    }
    fail("TYPE_MISMATCH",
         std::string(spec.name) + " typed matrix signature unsupported");
  }

  if (spec.family == ops::Family::regression) {
    if (v.empty() || !one_dimensional(v.back()) || !v.back().is_numeric())
      fail("TYPE_MISMATCH", "linear fit requires numeric 1D input");
    if (v.size() == 2)
      require_same_one_dimensional("linear_fit", v[0], v[1]);
    if (op == O::linear_fit)
      return ValueType::record(
          "fit", {"slope", "intercept", "residual_sum_squares",
                  "total_sum_squares", "observation_count"});
    return ValueType::scalar("dimensionless");
  }

  if (spec.family == ops::Family::state) {
    if (op == O::last_drawdown_interval) {
      if (v[0].kind != ValueKind::series)
        fail("TYPE_MISMATCH", "drawdown interval requires time series");
      return ValueType::record(
          "interval", {"start", "trough", "recovery", "has_drawdown"});
    }
    if (op == O::interval_start || op == O::interval_trough ||
        op == O::interval_recovery) {
      if (v[0].kind != ValueKind::record ||
          v[0].record_tag != "interval")
        fail("TYPE_MISMATCH", "interval field requires interval record");
      return ValueType::scalar("count");
    }
    if (op == O::fit_slope || op == O::fit_intercept ||
        op == O::fit_residual_sum_squares ||
        op == O::fit_total_sum_squares ||
        op == O::fit_observation_count) {
      if (v[0].kind != ValueKind::record || v[0].record_tag != "fit")
        fail("TYPE_MISMATCH", "fit field requires fit record");
      return ValueType::scalar(op == O::fit_observation_count ? "count"
                                                               : "dimensionless");
    }
    if (op == O::value_at) {
      if (!one_dimensional(v[0]) || !v[0].is_numeric() || !v[1].is_scalar())
        fail("TYPE_MISMATCH", "value_at requires 1D numeric input and index");
      return ValueType::scalar(v[0].semantic_dimension, v[0].price_basis);
    }
    if (op == O::days_between)
      return ValueType::scalar("calendar_days");
    if (op == O::require_positive || op == O::require_nonnegative)
      return v[0];
    fail("TYPE_MISMATCH", std::string(spec.name) + " typed state unsupported");
  }

  if (spec.family == ops::Family::composite) {
    if (op == O::active_returns)
      return additive("active_returns", v[0], v[1]);
    if (op == O::portfolio_returns) {
      if (v[0].kind != ValueKind::matrix ||
          v[0].axes != std::vector<std::string>{"time", "asset"} ||
          v[1].kind != ValueKind::vector ||
          v[0].shape[1] != v[1].shape[0])
        fail("TYPE_MISMATCH",
             "portfolio_returns requires time/asset matrix and asset vector");
      auto semantics = product_semantics(v[0], v[1]);
      return ValueType::series(v[0].shape[0], semantics.first,
                               semantics.second);
    }
    if (op == O::quadratic_form) {
      if (v[0].kind != ValueKind::vector ||
          v[1].kind != ValueKind::matrix ||
          v[1].axes != std::vector<std::string>{"asset", "asset"} ||
          v[1].shape != std::vector<std::string>{v[0].shape[0],
                                                 v[0].shape[0]})
        fail("TYPE_MISMATCH", "quadratic_form dimensions do not match");
      auto semantics = product_semantics(v[0], v[1]);
      return ValueType::scalar(semantics.first, semantics.second);
    }
    if (op == O::cumulative_return) {
      if (v[0].kind != ValueKind::series)
        fail("TYPE_MISMATCH", "cumulative_return requires time series");
      return v[0].with_semantics("return_decimal");
    }
    if (op == O::total_return) {
      if (v[0].kind != ValueKind::series)
        fail("TYPE_MISMATCH", "total_return requires time series");
      return ValueType::scalar("return_decimal");
    }
    if (op == O::annualized_return) {
      if (v[0].kind != ValueKind::series || v.size() != 2)
        fail("TYPE_MISMATCH",
             "annualized_return requires series and scalar periods");
      require_scalar_parameter(v[1], "periods_per_year");
      return ValueType::scalar("return_decimal");
    }
  }
  fail("TYPE_MISMATCH",
       std::string(spec.name) + " typed signature is not supported");
}

} // namespace calmetrics_engine::typed
