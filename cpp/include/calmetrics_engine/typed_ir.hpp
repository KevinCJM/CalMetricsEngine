#pragma once

#include "calmetrics_engine/operators.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace calmetrics_engine::typed {

struct Error : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

enum class ValueKind : std::uint8_t {
  scalar,
  series,
  vector,
  matrix,
  window,
  record
};

enum class DType : std::uint8_t { float64, boolean, int64 };

struct ValueType {
  ValueKind kind = ValueKind::scalar;
  DType dtype = DType::float64;
  std::vector<std::string> axes;
  std::vector<std::string> shape;
  std::string semantic_dimension = "dimensionless";
  std::string price_basis;
  std::string record_tag;
  std::vector<std::string> fields;

  static ValueType scalar(std::string semantic = "dimensionless",
                          std::string price_basis = {});
  static ValueType series(std::string length = "T",
                          std::string semantic = "dimensionless",
                          std::string price_basis = {});
  static ValueType vector(std::string length = "N",
                          std::string semantic = "dimensionless",
                          std::string price_basis = {});
  static ValueType matrix(std::vector<std::string> axes = {"time", "asset"},
                          std::vector<std::string> shape = {"T", "N"},
                          std::string semantic = "dimensionless",
                          std::string price_basis = {});
  static ValueType window(std::string length = "T",
                          std::string width = "W",
                          std::string semantic = "dimensionless",
                          std::string price_basis = {});
  static ValueType mask(std::vector<std::string> axes = {},
                        std::vector<std::string> shape = {});
  static ValueType record(std::string tag, std::vector<std::string> fields);

  bool is_scalar() const noexcept { return kind == ValueKind::scalar; }
  bool is_integer() const noexcept { return dtype == DType::int64; }
  bool is_mask() const noexcept { return dtype == DType::boolean; }
  bool is_numeric() const noexcept {
    return dtype == DType::float64 && kind != ValueKind::window &&
           kind != ValueKind::record;
  }
  std::size_t rank() const noexcept { return axes.size(); }
  void validate() const;
  ValueType with_semantics(std::string semantic,
                           std::string basis = {}) const;
  ValueType as_mask() const;
};

struct Variable {
  std::string name;
  ValueType type;
};

const char *kind_name(ValueKind kind) noexcept;
const char *dtype_name(DType dtype) noexcept;
std::string display(const ValueType &value);
std::string normalize_semantic_dimension(std::string value);

std::string canonical_operator_name(std::string_view name);
ValueType infer(const ops::Spec &spec, const std::vector<ValueType> &inputs);

bool same_structure(const ValueType &left, const ValueType &right) noexcept;
bool same_type(const ValueType &left, const ValueType &right) noexcept;

} // namespace calmetrics_engine::typed
