#pragma once
#include "calmetrics_engine/excel.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace calmetrics_engine::excel {
using Op = ops::Op;
using F = std::string;
inline F num(double x) {
  if (std::isnan(x))
    return "\"NaN\"";
  if (std::isinf(x))
    return x > 0 ? "\"+Inf\"" : "\"-Inf\"";
  // XLSX writers and Excel input parsing may round a long decimal literal.
  // Reconstruct binary64 from exact short integers, keeping cancellation cases
  // (e.g. variance of 1e12 + small binary fractions) reproducible.
  if (std::floor(x) != x || std::abs(x) > 999999999999999.0) {
    int exponent = 0;
    double fraction = std::frexp(std::abs(x), &exponent);
    auto significand = static_cast<std::uint64_t>(std::ldexp(fraction, 53));
    return "(" + std::string(x < 0 ? "-" : "") + "(" +
           std::to_string(significand >> 27) + "*134217728+" +
           std::to_string(significand & 134217727) + ")/2^52*2^(" +
           std::to_string(exponent - 1) + "))";
  }
  std::ostringstream s;
  s.imbue(std::locale::classic());
  s << std::setprecision(17) << x;
  return s.str();
}
inline F q(const F &s) {
  F r = "\"";
  for (char c : s) {
    r += c;
    if (c == '\"')
      r += '\"';
  }
  return r + '\"';
}
inline F iff(const F &c, const F &a, const F &b) {
  return "IF(" + c + "," + a + "," + b + ")";
}
inline F fn(const F &n, const F &a) { return n + "(" + a + ")"; }
inline F fin(const F &a) { return "ISNUMBER(" + a + ")"; }
inline F err(const F &code) { return q("ERROR:" + code); }
inline F bin(const F &a, const F &op, const F &b) {
  return "(" + a + op + b + ")";
}
struct Array {
  ops::Shape shape;
  ops::Kind kind = ops::Kind::number;
  std::vector<F> refs;
  // Only literal/configuration expressions have a compile-time scalar value.
  std::optional<double> scalar;
  std::array<F, 3> diagnostics{};
  F failure = "\"\"";
  bool position_errors = false;
  bool unavailable = false;
  F at(std::size_t i = 0) const { return refs.at(shape.rank ? i : 0); }
  std::size_t size() const { return refs.size(); }
};
struct Builder {
  Limits limits;
  Report report;
  std::function<void(const Cell &)> sink;
  std::size_t step = 0, input = 0, result = 0, index = 0;
  std::size_t invocation = 0;
  std::string label;
  std::string prefix;
  bool graph_mode = false, isolate = false;
  std::function<void()> checkpoint;
  std::chrono::steady_clock::time_point started =
      std::chrono::steady_clock::now();
  std::vector<F> structural_guards;
  explicit Builder(Limits l, std::function<void(const Cell &)> s,
                   std::string p = "")
      : limits(l), sink(std::move(s)), prefix(std::move(p)) {}
  void put(const F &sheet, std::size_t row, std::size_t col, const F &type,
           const F &value);
  F formula(const F &formula);
  Array array(const ops::Shape &shape, ops::Kind kind = ops::Kind::number);
  Array literal(double x);
  Array input_array(const Snapshot &, const F &name);
  Array operation(Op, const std::vector<Array> &);
  Array recipe(Op, const std::vector<Array> &);
  Array state_operation(Op, const std::vector<Array> &, Array output);
  std::vector<Array> program(const graph::Program &, const std::vector<Array> &,
                             const std::vector<Array> &, std::size_t extent,
                             unsigned depth = 0);
  F sum(const std::vector<F> &);
  F all(const std::vector<F> &);
  F any(const std::vector<F> &);
  F choose(const std::vector<F> &, const F &index,
           const F &fallback = "\"NaN\"");
  F reduce(Op, const std::vector<F> &, const std::vector<F> & = {},
           const F &ddof = "1", const F &probability = "0.5");
  Array slice(const Array &, std::size_t, std::size_t);
  std::size_t bound(const Array &, const F &name,
                    std::size_t maximum = 1000000);
  void cost(std::initializer_list<std::size_t> factors);
};
// Metadata-only abstract interpretation: no formulas, addresses, element
// arrays, numerical execution or file writes. Used before Builder exists.
Budget symbolic_budget(const compiler::CompiledGraph *, std::optional<Op>,
                       const std::vector<Snapshot> &,
                       const std::vector<double> &, Limits,
                       const std::function<void()> & = {});
} // namespace calmetrics_engine::excel
