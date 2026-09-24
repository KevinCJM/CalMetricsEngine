#pragma once

#include "calmetrics_engine/compiler.hpp"
#include <atomic>
#include <chrono>
#include <functional>
#include <optional>
#include <string>

namespace calmetrics_engine::excel {
inline constexpr const char *version = "excel-formulas-1";
struct Limits {
  std::size_t cells = 2000000;
  std::size_t formula_characters = 128 * 1024 * 1024;
  std::size_t snapshot_bytes = 64 * 1024 * 1024;
  std::size_t work_units = 100000000;
  std::size_t stream_bytes = 512 * 1024 * 1024;
  double timeout_seconds = 180;
  double atol = 1e-12, rtol = 1e-10;
};
struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct Cell {
  std::string sheet;
  std::size_t row = 0, column = 0;
  // number, boolean, text, formula. Formula text has no leading '='.
  std::string type, value;
};
struct Range {
  std::string name, dtype;
  ops::Shape shape;
  std::vector<std::string> addresses;
  std::vector<std::string> axes, fields;
};
struct Snapshot {
  ops::Value value;
  std::vector<double> numbers;
  std::vector<std::int64_t> integers;
  std::vector<std::uint8_t> masks;
  ops::Value view() const;
};
struct Report {
  std::size_t cells = 0, formula_characters = 0, snapshot_bytes = 0;
  std::size_t estimated_work = 0;
  std::vector<Range> outputs;
  std::vector<std::string> sheets;
};
struct BudgetNode {
  std::size_t node = 0;
  std::string kind;
  std::size_t cells = 0;
};
struct Budget {
  std::size_t cells = 0, formula_cells = 0, formula_characters = 0;
  std::size_t snapshot_bytes = 0, work_units = 0;
  std::vector<BudgetNode> largest_nodes;
};
struct Reference {
  ops::Shape shape;
  ops::Kind kind = ops::Kind::number;
  std::vector<double> numbers;
  std::vector<std::int64_t> integers;
  std::vector<std::int16_t> statuses;
  std::string error;
};
// All input bytes are copied once on this explicit export path. No Python
// owners, callbacks, or borrowed input pointers survive construction.
class Plan {
public:
  Plan(compiler::CompiledGraph graph, std::vector<Snapshot> inputs,
       std::vector<double> parameters, Limits limits,
       std::string build_identity = "standalone-native");
  Plan(ops::Op op, std::vector<Snapshot> arguments, Limits limits,
       std::string build_identity = "standalone-native");
  const Report &report() const { return report_; }
  const Budget &budget() const { return budget_; }
  const Limits &limits() const { return limits_; }
  // Replay the bounded layout into a native stream; no per-cell PyBind calls.
  void write(const std::string &path,
             const std::string &sheet_prefix = "") const;
  const std::string &identity() const { return identity_; }
  const std::string &build_identity() const { return build_identity_; }
  std::vector<Reference> reference() const;
  void cancel() { cancelled_.store(true); }
  void check_cancelled() const {
    if (cancelled_.load())
      throw Error("CANCELLED: Excel export");
  }

private:
  compiler::CompiledGraph graph_;
  std::vector<Snapshot> inputs_;
  std::vector<double> parameters_;
  std::optional<ops::Op> operator_;
  Limits limits_;
  Report report_;
  Budget budget_;
  std::string identity_;
  std::string build_identity_;
  std::atomic<bool> cancelled_{false};
  Report generate(const std::function<void(const Cell &)> &sink,
                  const std::string &prefix = "") const;
  void initialize();
};
std::vector<std::pair<std::string, std::string>> coverage();
} // namespace calmetrics_engine::excel
