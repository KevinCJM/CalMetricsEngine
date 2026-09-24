#include "calmetrics_engine/scheduler.hpp"
#include "excel_internal.hpp"
#include <filesystem>
#include <fstream>
#include <set>

namespace calmetrics_engine::excel {
namespace {
F address(const F &sheet, std::size_t row, std::size_t column) {
  F col;
  for (auto n = column + 1; n; n = (n - 1) / 26)
    col.insert(col.begin(), char('A' + (n - 1) % 26));
  return "'" + sheet + "'!$" + col + "$" + std::to_string(row + 1);
}
F json(const F &s) {
  F r = "\"";
  for (unsigned char c : s) {
    if (c == '\"' || c == '\\') {
      r += '\\';
      r += char(c);
    } else if (c < 32) {
      const char *hex = "0123456789abcdef";
      r += "\\u00";
      r += hex[c >> 4];
      r += hex[c & 15];
    } else
      r += char(c);
  }
  return r + '\"';
}
F dtype(ops::Kind k) {
  return k == ops::Kind::mask       ? "bool"
         : k == ops::Kind::integer  ? "int64"
         : k == ops::Kind::fit      ? "fit"
         : k == ops::Kind::interval ? "interval"
                                    : "float64";
}
void validate_numeric_domain(Op op, const ops::Value &value, bool output) {
  if (value.shape.rank < 0 || value.kind == ops::Kind::mask)
    return;
  const auto count = value.kind == ops::Kind::fit ? 5u
                     : value.kind == ops::Kind::interval ? 4u : value.size();
  for (std::size_t i = 0; i < count; ++i) {
    if (value.kind == ops::Kind::integer) {
      const auto x = value.i(i);
      if (x < -999999999999999LL || x > 999999999999999LL)
        throw Error("UNREPRESENTABLE_VALUE: computed integer exceeds Excel domain");
      continue;
    }
    const double x = value.kind == ops::Kind::fit || value.kind == ops::Kind::interval
                         ? value.record[i] : value.f(i);
    // Masked extrema deliberately return an infinity category when every
    // selected value is NaN. Their recipe represents that sentinel as text;
    // consuming it in another numeric operator still requires rejection.
    const bool sentinel = output && (op == Op::min_where || op == Op::max_where);
    if ((!sentinel && std::isinf(x)) ||
        (x != 0 && std::isfinite(x) && std::abs(x) < std::numeric_limits<double>::min()))
      throw Error("UNREPRESENTABLE_VALUE: arithmetic exceeds Excel numeric domain in " +
                  F(ops::lookup(static_cast<std::uint16_t>(op)).name));
  }
}
} // namespace
ops::Value Snapshot::view() const {
  auto v = value;
  if (v.shape.rank) {
    v.data = v.kind == ops::Kind::integer
                 ? static_cast<const void *>(integers.data())
             : v.kind == ops::Kind::mask
                 ? static_cast<const void *>(masks.data())
                 : numbers.data();
    v.set_contiguous_strides();
  }
  return v;
}
void Builder::put(const F &sheet, std::size_t row, std::size_t col,
                  const F &type, const F &value) {
  if ((report.cells & 1023) == 0) {
    if (checkpoint)
      checkpoint();
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      started)
            .count() > limits.timeout_seconds)
      throw Error("TIMEOUT: Excel planning/writing");
  }
  const auto name = prefix + sheet;
  if (row >= 1048576 || col >= 16384)
    throw Error("OVER_BUDGET: Excel row/column limit");
  if (report.cells >= limits.cells)
    throw Error("OVER_BUDGET: whole workbook cell limit " +
                std::to_string(limits.cells));
  ++report.cells;
  if (type == "formula") {
    if (value.size() > 8192)
      throw Error("OVER_BUDGET: Excel formula length");
    if (value.size() > limits.formula_characters - report.formula_characters)
      throw Error("OVER_BUDGET: total formula characters");
    report.formula_characters += value.size();
    // Count parentheses outside quoted literals; conservative with grouping,
    // so the Excel 64-function nesting bound cannot be exceeded silently.
    bool quoted = false;
    unsigned nesting = 0;
    for (char c : value) {
      if (c == '"')
        quoted = !quoted;
      else if (!quoted && c == '(') {
        if (++nesting > 64)
          throw Error("OVER_BUDGET: formula nesting");
      } else if (!quoted && c == ')' && nesting)
        --nesting;
    }
  }
  if (std::find(report.sheets.begin(), report.sheets.end(), name) ==
      report.sheets.end())
    report.sheets.push_back(name);
  if (sink)
    sink({name, row, col, type, value});
}
void Builder::cost(std::initializer_list<std::size_t> factors) {
  std::size_t amount = 1;
  for (auto f : factors) {
    if (f && amount > limits.work_units / f)
      throw Error("OVER_BUDGET: conservative expansion work bound");
    amount *= f;
  }
  if (amount > limits.work_units - report.estimated_work)
    throw Error("OVER_BUDGET: conservative expansion work bound");
  report.estimated_work += amount;
}
F Builder::formula(const F &f) {
  const auto i = step++;
  const auto sheet = "Steps" + std::to_string(i / 1000000);
  const auto row = i % 1000000;
  if (!label.empty()) {
    put(sheet, row, 0, "text", label);
    label.clear();
  }
  put(sheet, row, 1, "formula", f);
  return address(prefix + sheet, row, 1);
}
Array Builder::array(const ops::Shape &s, ops::Kind k) {
  if (s.size() > limits.cells - report.cells)
    throw Error("OVER_BUDGET: output geometry");
  Array a;
  a.shape = s;
  a.kind = k;
  a.refs.resize(s.size());
  return a;
}
Array Builder::literal(double x) {
  Array a;
  a.refs = {num(x)};
  a.scalar = x;
  return a;
}
Array Builder::input_array(const Snapshot &s, const F &name) {
  auto v = s.view();
  auto a = array(v.shape, v.kind);
  if (v.kind == ops::Kind::fit || v.kind == ops::Kind::interval)
    a.refs.resize(v.kind == ops::Kind::fit ? 5 : 4);
  for (std::size_t j = 0; j < a.size(); ++j) {
    const auto i = input++, row = i % 1000000;
    const F sheet = "Inputs" + std::to_string(i / 1000000);
    F value, type = "number";
    if (v.kind == ops::Kind::integer) {
      auto n = v.i(j);
      // Excel stores only 15 significant decimal digits through file/UI paths.
      if (n < -999999999999999LL || n > 999999999999999LL)
        throw Error(
            "UNREPRESENTABLE_VALUE: int64 exceeds exact Excel export domain");
      value = std::to_string(n);
    } else if (v.kind == ops::Kind::mask) {
      type = "boolean";
      value = v.u(j) ? "1" : "0";
    } else {
      double x = (v.kind == ops::Kind::fit || v.kind == ops::Kind::interval)
                     ? v.record[j]
                     : v.f(j);
      if (std::isinf(x))
        throw Error("UNREPRESENTABLE_VALUE: infinite input");
      if (x != 0 && std::abs(x) < std::numeric_limits<double>::min())
        throw Error("UNREPRESENTABLE_VALUE: subnormal input");
      if (std::isnan(x)) {
        type = "text";
        value = "NaN";
      } else {
        value = num(x);
        if (value.front() == '(')
          type = "formula";
      }
    }
    put(sheet, row, 0, "text", name + "[" + std::to_string(j) + "]");
    put(sheet, row, 1, type, value);
    put(sheet, row, 2, type, value);
    put(sheet, row, 3, "formula",
        address(prefix + sheet, row, 1) + "=" +
            address(prefix + sheet, row, 2));
    a.refs[j] = address(prefix + sheet, row, 1);
  }
  if (!v.shape.rank && v.kind == ops::Kind::number)
    a.scalar = v.scalar;
  return a;
}
F Builder::sum(const std::vector<F> &items) {
  // Bounded fan-in gives a predictable formula length and exposes
  // intermediates.
  if (items.empty())
    return "0";
  std::vector<F> level = items;
  while (level.size() > 1) {
    std::vector<F> next;
    for (std::size_t i = 0; i < level.size(); i += 32) {
      F f = "SUM(";
      for (std::size_t j = i; j < std::min(i + 32, level.size()); ++j) {
        if (j > i)
          f += ",";
        f += level[j];
      }
      next.push_back(formula(f + ")"));
    }
    level = std::move(next);
  }
  return level[0];
}
F Builder::all(const std::vector<F> &v) {
  std::vector<F> a;
  for (auto &s : v)
    a.push_back(formula("--(" + s + ")"));
  return bin(sum(a), "=", std::to_string(v.size()));
}
F Builder::any(const std::vector<F> &v) {
  std::vector<F> a;
  for (auto &s : v)
    a.push_back(formula("--(" + s + ")"));
  return bin(sum(a), ">", "0");
}
F Builder::choose(const std::vector<F> &v, const F &i, const F &fallback) {
  // No volatile INDIRECT/OFFSET and no whole-column references. Tree depth is
  // logarithmic and each branch is a visible formula cell.
  std::function<F(std::size_t, std::size_t)> select = [&](std::size_t l,
                                                          std::size_t r) -> F {
    if (l == r)
      return fallback;
    if (r - l == 1)
      return v[l];
    auto m = l + (r - l) / 2;
    auto left = select(l, m), right = select(m, r);
    return formula(iff(bin(i, "<", std::to_string(m)), left, right));
  };
  auto value = select(0, v.size());
  return formula(
      iff(fin(i),
          iff("AND(" + i + ">=0," + i + "<" + std::to_string(v.size()) + "," +
                  i + "=INT(" + i + "))",
              value, fallback),
          fallback));
}
Array Builder::slice(const Array &a, std::size_t first, std::size_t count) {
  if (a.shape.rank != 1 || first > a.size() || count > a.size() - first)
    throw Error("INVALID_INPUT: slice geometry");
  Array r = a;
  r.shape = ops::vector_shape(count);
  r.refs.assign(a.refs.begin() + first, a.refs.begin() + first + count);
  r.scalar.reset();
  return r;
}
std::size_t Builder::bound(const Array &a, const F &name, std::size_t max) {
  if (!a.scalar || !std::isfinite(*a.scalar) || *a.scalar < 0 ||
      *a.scalar > double(max) || std::floor(*a.scalar) != *a.scalar)
    throw Error("UNKNOWN_BOUND: " + name +
                " needs a finite bounded configuration scalar");
  // Layout parameters are frozen. Editing requires a fresh plan; the workbook
  // snapshot indicator changes, and affected outputs carry a layout guard.
  structural_guards.push_back(a.at() + "=" + num(*a.scalar));
  return static_cast<std::size_t>(*a.scalar);
}
Plan::Plan(compiler::CompiledGraph g, std::vector<Snapshot> in,
           std::vector<double> p, Limits l, std::string build)
    : graph_(std::move(g)), inputs_(std::move(in)), parameters_(std::move(p)),
      limits_(l), build_identity_(std::move(build)) {
  initialize();
}
Plan::Plan(Op op, std::vector<Snapshot> args, Limits l, std::string build)
    : inputs_(std::move(args)), operator_(op), limits_(l),
      build_identity_(std::move(build)) {
  initialize();
}
void Plan::initialize() {
  if (!std::isfinite(limits_.atol) || !std::isfinite(limits_.rtol) ||
      limits_.atol < 0 || limits_.rtol < 0 ||
      !std::isfinite(limits_.timeout_seconds) || limits_.timeout_seconds <= 0)
    throw Error("INVALID_INPUT: tolerance/time budget");
  if (limits_.cells == 0 || limits_.cells > 2000000 ||
      limits_.formula_characters == 0)
    throw Error("INVALID_INPUT: limits must be positive; max_cells cannot "
                "exceed 2000000");
  // Deep-copy mutable shared scope programs. Physical kernels are reconstructed
  // by the existing decoder, never reverse translated into formulas.
  if (!operator_)
    graph_.program =
        compiler::decode_program(compiler::encode_program(graph_.program));
  budget_ =
      symbolic_budget(operator_ ? nullptr : &graph_, operator_, inputs_,
                      parameters_, limits_, [this] { check_cancelled(); });
  report_ = generate({});
  if (report_.cells > budget_.cells ||
      report_.formula_characters > budget_.formula_characters ||
      report_.snapshot_bytes > budget_.snapshot_bytes ||
      report_.estimated_work > budget_.work_units)
    throw Error(
        "SYMBOLIC_BOUND_VIOLATION: formula expansion exceeded its bound");
  // The symbolic pass remains metadata-only and runs first. Before READY,
  // replay the frozen inputs through canonical kernels to reject unsupported
  // numeric intermediates even when a later comparison hides their range.
  compute_reference(true);
  std::uint64_t hash = 1469598103934665603ULL;
  auto mix = [&](const void *p, std::size_t n) {
    auto *b = static_cast<const unsigned char *>(p);
    while (n--) {
      hash ^= *b++;
      hash *= 1099511628211ULL;
    }
  };
  mix(version, std::char_traits<char>::length(version));
  mix(build_identity_.data(), build_identity_.size());
  auto strings = [&](const std::vector<std::string> &items) {
    const auto count = items.size();
    mix(&count, sizeof(count));
    for (const auto &item : items) {
      const auto length = item.size();
      mix(&length, sizeof(length));
      mix(item.data(), item.size());
    }
  };
  if (operator_) {
    auto id = static_cast<std::uint16_t>(*operator_);
    mix(&id, sizeof(id));
  } else {
    auto bytes = compiler::encode_program(graph_.program);
    mix(bytes.data(), bytes.size());
    strings(graph_.input_names);
    strings(graph_.parameter_names);
    strings(graph_.output_names);
    strings(graph_.expressions);
  }
  for (auto &s : inputs_) {
    auto v = s.view();
    auto kind = static_cast<int>(v.kind);
    mix(&kind, sizeof(kind));
    mix(&v.shape.rank, sizeof(v.shape.rank));
    mix(v.shape.dim.data(), sizeof(v.shape.dim));
    if (v.shape.rank) {
      const auto bytes = v.size() * (v.kind == ops::Kind::mask ? 1 : 8);
      mix(v.data, bytes);
    } else {
      mix(&v.scalar, sizeof(v.scalar));
      mix(&v.integer, sizeof(v.integer));
      mix(v.record.data(), sizeof(v.record));
    }
  }
  mix(parameters_.data(), parameters_.size() * sizeof(double));
  mix(&limits_.cells, sizeof(limits_.cells));
  mix(&limits_.formula_characters, sizeof(limits_.formula_characters));
  mix(&limits_.snapshot_bytes, sizeof(limits_.snapshot_bytes));
  mix(&limits_.work_units, sizeof(limits_.work_units));
  mix(&limits_.stream_bytes, sizeof(limits_.stream_bytes));
  mix(&limits_.timeout_seconds, sizeof(limits_.timeout_seconds));
  mix(&limits_.atol, sizeof(limits_.atol));
  mix(&limits_.rtol, sizeof(limits_.rtol));
  std::ostringstream os;
  os << std::hex << hash;
  identity_ = os.str();
}
Report Plan::generate(const std::function<void(const Cell &)> &sink,
                      const F &prefix) const {
  Builder b(limits_, sink, prefix);
  b.graph_mode = !operator_;
  b.isolate = !operator_ && graph_.program.isolate_errors;
  b.checkpoint = [this] { check_cancelled(); };
  b.put("Readme", 0, 0, "text", "CalMetricsEngine Excel calculation steps");
  b.put("Readme", 1, 0, "text", "Recalculation evidence");
  b.put("Readme", 1, 1, "text", "NOT_RECALCULATED");
  b.put("Readme", 2, 0, "text", "Formula contract");
  b.put("Readme", 2, 1, "text", version);
  b.put("Readme", 3, 0, "text", "Export identity (non-cryptographic)");
  b.put("Readme", 3, 1, "text", identity_);
  b.put("Readme", 4, 0, "text",
        "Inputs: B editable, C reference snapshot; changed structural "
        "parameters require re-export.");
  b.put("Readme", 5, 0, "text",
        "NaN is the explicit text NaN; ERROR:code is an execution failure. "
        "Neither is zero or blank.");
  std::vector<Array> in, params;
  for (std::size_t i = 0; i < graph_.expressions.size(); ++i)
    b.put("Definitions", i, 0, "text", graph_.expressions[i]);
  for (std::size_t i = 0; i < inputs_.size(); ++i) {
    auto v = inputs_[i].view();
    auto bytes = (v.kind == ops::Kind::fit        ? 5
                  : v.kind == ops::Kind::interval ? 4
                                                  : v.size()) *
                 (v.kind == ops::Kind::mask ? 1 : 8);
    if (bytes > limits_.snapshot_bytes - b.report.snapshot_bytes)
      throw Error("OVER_BUDGET: snapshot bytes");
    b.report.snapshot_bytes += bytes;
    in.push_back(b.input_array(inputs_[i], operator_
                                               ? "argument " + std::to_string(i)
                                               : graph_.input_names.at(i)));
  }
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    if (8 > limits_.snapshot_bytes - b.report.snapshot_bytes)
      throw Error("OVER_BUDGET: parameter snapshot");
    b.report.snapshot_bytes += 8;
    Snapshot s;
    s.value = ops::Value::number(parameters_[i]);
    params.push_back(b.input_array(s, graph_.parameter_names.at(i)));
  }
  std::vector<Array> roots;
  if (operator_) {
    b.label = ops::lookup(static_cast<std::uint16_t>(*operator_)).name;
    roots.push_back(b.operation(*operator_, in));
  } else {
    if (in.size() != graph_.program.input_count ||
        params.size() != graph_.program.parameter_count)
      throw Error("INVALID_INPUT: graph binding count");
    std::size_t extent = 1;
    for (std::size_t i = 0; i < in.size(); ++i)
      if (in[i].shape.rank && graph_.program.input_axes.at(i) != 1) {
        extent = in[i].shape.dim[0];
        break;
      }
    if (extent < graph_.program.minimum_observations)
      throw Error("INVALID_INPUT: minimum observations unmet; native output "
                  "geometry is unknown");
    roots = b.program(graph_.program, in, params, extent);
  }
  std::vector<F> changed;
  for (std::size_t i = 0; i < b.input; ++i)
    changed.push_back(address(prefix + "Inputs" + std::to_string(i / 1000000),
                              i % 1000000, 3));
  auto fresh = b.all(changed);
  auto layout_valid = b.all(b.structural_guards);
  b.put("Readme", 6, 0, "text", "Reference snapshot");
  b.put("Readme", 6, 1, "formula",
        iff(fresh, q("UNCHANGED"), q("REFERENCE_STALE")));
  for (std::size_t i = 0; i < roots.size(); ++i) {
    auto &a = roots[i];
    if (a.shape.rank < 0)
      throw Error("UNKNOWN_BOUND: failed root has unknown geometry");
    Range r;
    r.name = operator_
                 ? ops::lookup(static_cast<std::uint16_t>(*operator_)).name
             : graph_.output_names.empty() ? "output_" + std::to_string(i)
                                           : graph_.output_names.at(i);
    r.dtype = dtype(a.kind);
    r.shape = a.shape;
    if (!operator_)
      r.axes = graph_.nodes.at(graph_.program.roots.at(i)).inferred_type.axes;
    else
      for (int axis = 0; axis < a.shape.rank; ++axis)
        r.axes.push_back("axis_" + std::to_string(axis));
    if (a.kind == ops::Kind::fit)
      r.fields = {"slope", "intercept", "residual_sum_squares",
                  "total_sum_squares", "count"};
    if (a.kind == ops::Kind::interval)
      r.fields = {"peak", "trough", "recovery", "status"};
    F geometry;
    for (int axis = 0; axis < a.shape.rank; ++axis) {
      if (axis)
        geometry += ",";
      geometry += r.axes.at(axis) + ":" + std::to_string(a.shape.dim[axis]);
    }
    b.put("Outputs", i, 0, "text", r.name);
    b.put("Outputs", i, 1, "text", r.dtype);
    b.put("Outputs", i, 2, "text",
          geometry.empty() ? "scalar/record" : geometry);
    b.put("Outputs", i, 3, "text",
          "Flat row-major order; Results columns: name, Excel, frozen C++, "
          "absolute error, comparison");
    for (std::size_t j = 0; j < a.size(); ++j) {
      const auto k = b.result++, row = k % 1000000;
      const F sheet = "Results" + std::to_string(k / 1000000);
      auto value = a.refs[j];
      if (!operator_)
        value = iff("AND(LEFT(" + value + ",6)=\"ERROR:\",LEFT(" + value +
                        ",13)<>\"ERROR:STATUS:\")",
                    q("ERROR:STATUS:4"), value);
      if (!operator_ && graph_.program.isolate_errors &&
          a.kind == ops::Kind::number)
        value = iff("OR(ISNUMBER(" + value + "),LEFT(" + value +
                        ",13)=\"ERROR:STATUS:\")",
                    value, q("ERROR:STATUS:4"));
      b.put(sheet, row, 0, "text", r.name + "[" + std::to_string(j) + "]");
      b.put(sheet, row, 1, "formula",
            iff(layout_valid, value, q("LAYOUT_STALE: re-export required")));
      b.put(sheet, row, 2, "reference",
            std::to_string(i) + ":" + std::to_string(j));
      auto actual = address(prefix + sheet, row, 1),
           reference = address(prefix + sheet, row, 2);
      b.put(sheet, row, 3, "formula",
            iff("AND(ISNUMBER(" + actual + "),ISNUMBER(" + reference + "))",
                "ABS(" + actual + "-" + reference + ")", q("not finite")));
      auto comparison =
          (a.kind == ops::Kind::integer || a.kind == ops::Kind::mask)
              ? actual + "=" + reference
              : iff("AND(ISNUMBER(" + actual + "),ISNUMBER(" + reference + "))",
                    "ABS(" + actual + "-" + reference +
                        ")<=" + num(limits_.atol) + "+" + num(limits_.rtol) +
                        "*ABS(" + reference + ")",
                    actual + "=" + reference);
      b.put(sheet, row, 4, "formula",
            "IFERROR(" +
                iff(fresh, iff(comparison, q("MATCH"), q("MISMATCH")),
                    q("REFERENCE_STALE")) +
                ",\"MISMATCH\")");
      r.addresses.push_back(address(prefix + sheet, row, 1));
    }
    b.report.outputs.push_back(std::move(r));
  }
  b.put("Readme", 8, 0, "text", "Engine build identity");
  b.put("Readme", 8, 1, "text", build_identity_);
  b.put("Readme", 9, 0, "text", "Whole workbook cells (including metadata)");
  b.put("Readme", 9, 1, "number", std::to_string(b.report.cells + 8));
  b.put("Readme", 10, 0, "text", "Cell budget");
  b.put("Readme", 10, 1, "number", std::to_string(limits_.cells));
  b.put("Readme", 11, 0, "text", "Float comparison atol / rtol");
  b.put("Readme", 11, 1, "formula", num(limits_.atol));
  b.put("Readme", 11, 2, "formula", num(limits_.rtol));
  b.put(
      "Readme", 12, 0, "text",
      "Input fractions use exact binary decomposition; edit B to change "
      "inputs. Numeric overflow or unsupported magnitudes require re-export.");
  b.put("Readme", 13, 0, "text",
        "Nodes: context:node, operation, first address, shape, parents, dtype, "
        "last address. Order is flat row-major.");
  return b.report;
}
void Plan::write(const std::string &path, const F &prefix) const {
  if (prefix.size() > 12 ||
      !std::all_of(prefix.begin(), prefix.end(), [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '_';
      }))
    throw Error("INVALID_INPUT: sheet prefix");
  // Prefix changes reference lengths; re-preflight before opening the stream.
  check_cancelled();
  auto expected = prefix.empty() ? report_ : generate({}, prefix);
  std::ofstream out(std::filesystem::u8path(path),
                    std::ios::binary | std::ios::trunc);
  if (!out)
    throw Error("cannot open export stream");
  auto r = generate(
      [&](const Cell &c) {
        out << "[" << json(c.sheet) << "," << c.row << "," << c.column << ","
            << json(c.type) << "," << json(c.value) << "]\n";
        if (!out)
          throw Error("export stream write failed");
        if (static_cast<std::uint64_t>(out.tellp()) > limits_.stream_bytes)
          throw Error("OVER_BUDGET: stream bytes");
      },
      prefix);
  if (r.cells != expected.cells ||
      r.formula_characters != expected.formula_characters ||
      r.cells > budget_.cells ||
      r.formula_characters > budget_.formula_characters)
    throw Error("EXPORT_PLAN_MISMATCH");
  out.close();
  if (!out)
    throw Error("export stream close failed");
}
std::vector<Reference> Plan::reference() const {
  return compute_reference(false);
}
std::vector<Reference> Plan::compute_reference(bool validate_domain) const {
  check_cancelled();
  std::vector<Reference> result;
  // Existing CPU admission and canonical execution only. Export recipes never
  // calculate the C++ reference or use it to select formula branches.
  native::NativeScheduler::instance().parallel_for(
      1, 1, [&](std::size_t, std::size_t) {
        std::vector<ops::Value> inputs;
        for (auto &s : inputs_)
          inputs.push_back(s.view());
        if (operator_) {
          Reference r;
          try {
            auto p = ops::prepare(
                ops::lookup(static_cast<std::uint16_t>(*operator_)),
                inputs.data(), inputs.size());
            r.shape = p.output_shape;
            r.kind = p.output_kind;
            std::vector<std::uint64_t> storage(r.shape.size());
            ops::Output output;
            output.shape = r.shape;
            output.kind = r.kind;
            if (r.shape.rank)
              output.data = storage.data();
            ops::Workspace work;
            ops::Audit audit;
            ops::execute(p, output, work, ops::Isa::automatic, audit);
            if (validate_domain) {
              auto value = p.output_shape.rank ? ops::Value{} : ops::Value::number(output.scalar);
              value.kind = output.kind;
              value.shape = output.shape;
              value.data = output.data;
              value.integer = output.integer;
              value.record = output.record;
              value.set_contiguous_strides();
              validate_numeric_domain(*operator_, value, true);
            }
            if (r.kind == ops::Kind::fit || r.kind == ops::Kind::interval)
              r.numbers.assign(output.record.begin(),
                               output.record.begin() +
                                   (r.kind == ops::Kind::fit ? 5 : 4));
            else
              for (std::size_t i = 0; i < r.shape.size(); ++i) {
                if (r.kind == ops::Kind::integer)
                  r.integers.push_back(
                      r.shape.rank
                          ? reinterpret_cast<std::int64_t *>(storage.data())[i]
                          : output.integer);
                else if (r.kind == ops::Kind::mask)
                  r.integers.push_back(
                      r.shape.rank
                          ? reinterpret_cast<std::uint8_t *>(storage.data())[i]
                          : static_cast<std::int64_t>(output.scalar));
                else
                  r.numbers.push_back(r.shape.rank ? reinterpret_cast<double *>(
                                                         storage.data())[i]
                                                   : output.scalar);
              }
          } catch (const ops::Error &e) {
            r.error = e.what();
          }
          result.push_back(std::move(r));
          return;
        }
        auto program = graph_.program;
        program.output_kind = graph::OutputKind::typed;
        program.root_outputs.clear();
        for (auto &r : report_.outputs)
          program.root_outputs.push_back(
              {r.dtype == "int64"  ? graph::OutputDType::int64
               : r.dtype == "bool" ? graph::OutputDType::boolean
                                   : graph::OutputDType::float64,
               static_cast<std::uint8_t>(r.shape.rank)});
        program.finalize();
        std::int64_t start = 0, end = 1;
        for (std::size_t i = 0; i < inputs.size(); ++i)
          if (inputs[i].shape.rank && program.input_axes.at(i) != 1) {
            end = static_cast<std::int64_t>(inputs[i].shape.dim[0]);
            break;
          }
        auto layout = graph::result_layout(program, inputs, &start, &end, 1);
        std::vector<std::uint64_t> storage((layout.bytes() + 7) / 8);
        auto audit = validate_domain
            ? graph::execute_observed(program, inputs, parameters_.data(), parameters_.size(),
                &start, &end, 1, storage.data(), program.roots.size(), &layout, validate_numeric_domain)
            : graph::execute(program, inputs, parameters_.data(), parameters_.size(), &start,
                &end, 1, storage.data(), program.roots.size(), &layout);
        for (std::size_t j = 0; j < program.roots.size(); ++j) {
          Reference r;
          r.shape = audit.result_shapes.at(j);
          const auto dtype = program.root_outputs[j].dtype;
          r.kind = dtype == graph::OutputDType::int64     ? ops::Kind::integer
                   : dtype == graph::OutputDType::boolean ? ops::Kind::mask
                                                          : ops::Kind::number;
          if (r.shape.rank < 0)
            r.error = "STATUS:" + std::to_string(audit.root_statuses[j]);
          else {
            auto *ptr =
                reinterpret_cast<const unsigned char *>(storage.data()) +
                layout.slots.at(j).byte_offset;
            for (std::size_t i = 0; i < r.shape.size(); ++i) {
              if (r.kind == ops::Kind::number)
                r.numbers.push_back(reinterpret_cast<const double *>(ptr)[i]);
              else
                r.integers.push_back(
                    r.kind == ops::Kind::integer
                        ? reinterpret_cast<const std::int64_t *>(ptr)[i]
                        : ptr[i]);
              r.statuses.push_back(
                  audit.statuses.empty()
                      ? 0
                      : audit.statuses.at(layout.slots.at(j).status_offset +
                                          i));
            }
          }
          result.push_back(std::move(r));
        }
      });
  check_cancelled();
  return result;
}
std::vector<std::pair<std::string, std::string>> coverage() {
  std::vector<std::pair<std::string, std::string>> r;
  for (auto &s : ops::registry())
    r.emplace_back(s.name, static_cast<std::uint16_t>(s.op) <= 146
                               ? "implemented_unverified"
                               : "unsupported");
  return r;
}
} // namespace calmetrics_engine::excel
