#include "../cpp/excel_internal.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

using namespace calmetrics_engine;

namespace {
excel::Snapshot snapshot(ops::Shape shape, std::initializer_list<double> values) {
  excel::Snapshot result;
  result.value.shape = shape;
  result.numbers = values;
  return result;
}

void axis_observer_isolation_contracts() {
  const std::vector<typed::Variable> variables{
      {"x", typed::ValueType::matrix({"time", "scenario"}, {"T", "P"})},
      {"y", typed::ValueType::matrix({"time", "scenario"}, {"T", "D"})},
      {"good_q", typed::ValueType::vector("Q")},
      {"bad_q", typed::ValueType::vector("Q")}};
  for (bool invalid_probability : {false, true}) {
    for (bool same_operation : {false, true}) {
      // Establish every healthy consumer parent before the failed quantile.
      // No input, constant or output callback occurs between the failure and
      // the next operation's arguments to incidentally reset the collector.
      const std::vector<std::string> expressions{
          "cumulative_product_axis(y,0,1)",
          "cumulative_max_axis(good_q,0,1)",
          "quantiles_axis(x,bad_q,1)",
          same_operation ? "quantiles_axis(y,good_q,0)"
                         : "cumulative_max_axis(y,0,1)"};
      auto graph = compiler::compile(expressions, variables, true, {}, {}, 0,
                                     100000000, true);
      if (graph->program.roots[3] != graph->program.roots[2] + 1)
        throw std::runtime_error("axis observer regression needs adjacent operations");
      const auto x = invalid_probability
          ? snapshot(ops::matrix_shape(2, 3), {1, 2, 3, 3, 4, 5})
          : snapshot(ops::matrix_shape(2, 0), {});
      std::vector<excel::Snapshot> inputs{
          x, snapshot(ops::matrix_shape(2, 3), {1, 2, 3, 3, 4, 5}),
          snapshot(ops::vector_shape(1), {.5}),
          snapshot(ops::vector_shape(1), {invalid_probability ? -.1 : .5})};
      // Construction invokes the export-only observed executor; reference()
      // subsequently checks that failure isolation retained healthy results.
      excel::Plan plan(*graph, std::move(inputs), {}, {});
      const auto references = plan.reference();
      if (references.size() != 4 || plan.report().outputs.size() != 4)
        throw std::runtime_error("axis observer isolated output count");
      if (references[0].numbers != std::vector<double>{1, 2, 3, 3, 8, 15} ||
          references[1].numbers != std::vector<double>{1})
        throw std::runtime_error("axis observer prebuilt healthy parents");
      const auto &failed = references[2];
      if (failed.error.empty() &&
          (failed.statuses.empty() ||
           !std::all_of(failed.statuses.begin(), failed.statuses.end(),
                        [](auto status) { return status == 4; })))
        throw std::runtime_error("axis observer quantile failure was lost");
      const auto &healthy = references[3];
      const auto expected_shape = same_operation ? ops::matrix_shape(1, 3)
                                                : ops::matrix_shape(2, 3);
      const std::vector<double> expected = same_operation
          ? std::vector<double>{2, 3, 4}
          : std::vector<double>{1, 2, 3, 3, 4, 5};
      if (!healthy.error.empty() || !(healthy.shape == expected_shape) ||
          healthy.numbers != expected ||
          !std::all_of(healthy.statuses.begin(), healthy.statuses.end(),
                       [](auto status) { return status == 0; }))
        throw std::runtime_error("axis observer failed-to-healthy collector reuse");
      if (plan.report().cells > plan.budget().cells)
        throw std::runtime_error("axis observer symbolic bound");
    }
  }
}

void empty_root_cells(const excel::Plan &plan) {
  const auto path = std::filesystem::temp_directory_path() /
      ("calmetrics-axis-empty-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".jsonl");
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  } cleanup{path};
  plan.write(path.string());
  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("empty root cell stream unavailable");
  std::vector<bool> states(plan.report().outputs.size(), false);
  std::size_t cells = 0, formulas = 0;
  for (std::string line; std::getline(stream, line);) {
    ++cells;
    formulas += line.find(",\"formula\",") != std::string::npos;
    if (line.rfind("[\"Results", 0) == 0)
      throw std::runtime_error("empty root must not invent a result element");
    for (std::size_t root = 0; root < states.size(); ++root) {
      const auto prefix = "[\"Outputs\"," + std::to_string(root) +
                          ",4,\"formula\",";
      if (line.rfind(prefix, 0) != 0) continue;
      if (states[root] || line.find("EMPTY") == std::string::npos ||
          line.find("ERROR:STATUS:4") == std::string::npos ||
          line.find("LAYOUT_STALE") == std::string::npos)
        throw std::runtime_error("empty root formula state missing or duplicated");
      states[root] = true;
    }
  }
  if (!stream.eof() || cells != plan.report().cells ||
      formulas > plan.budget().formula_cells ||
      !std::all_of(states.begin(), states.end(), [](bool present) { return present; }))
    throw std::runtime_error("empty root state stream or symbolic count");
}

void empty_typed_root_contracts() {
  const std::vector<typed::Variable> variables{
      {"x", typed::ValueType::matrix({"time", "scenario"}, {"T", "P"})},
      {"bad_q", typed::ValueType::vector("Q")},
      {"good_q", typed::ValueType::vector("Q")}};
  auto graph = compiler::compile(
      {"quantiles_axis(x,bad_q,1)", "quantiles_axis(x,good_q,1)",
       "cumulative_product_axis(x,1,2)", "cumulative_max_axis(x,1,2)"},
      variables, true, {}, {}, 0, 100000000, true);
  std::size_t upper_cells = 0;
  for (double probability : {.5, -.1, 1.1,
                             std::numeric_limits<double>::quiet_NaN()}) {
    const std::vector<excel::Snapshot> inputs{
        snapshot(ops::matrix_shape(0, 3), {}),
        snapshot(ops::vector_shape(1), {probability}),
        snapshot(ops::vector_shape(1), {.5})};
    // No root has a result byte. Plan construction and reference() must still
    // provide a nonnull batch destination and retain root-level failure state.
    excel::Plan plan(*graph, inputs, {}, {});
    const auto references = plan.reference();
    if (references.size() != 4 || plan.report().outputs.size() != 4)
      throw std::runtime_error("empty typed root output count");
    const bool invalid = !std::isfinite(probability) || probability < 0 || probability > 1;
    for (std::size_t root = 0; root < references.size(); ++root) {
      const auto expected_shape = ops::matrix_shape(0, root < 2 ? 1 : 3);
      const auto expected_error = root == 0 && invalid ? "STATUS:4" : "";
      const auto &reference = references[root];
      if (!(reference.shape == expected_shape) ||
          !(plan.report().outputs[root].shape == expected_shape) ||
          reference.error != expected_error || !reference.numbers.empty() ||
          !reference.integers.empty() || !reference.statuses.empty() ||
          !plan.report().outputs[root].addresses.empty())
        throw std::runtime_error("empty typed root shape, failure or element ownership");
    }
    empty_root_cells(plan);
    if (plan.report().cells > plan.budget().cells ||
        plan.report().formula_characters > plan.budget().formula_characters ||
        plan.report().estimated_work > plan.budget().work_units)
      throw std::runtime_error("empty typed root complete symbolic bounds");
    if (!upper_cells) upper_cells = plan.budget().cells;
    if (plan.budget().cells != upper_cells)
      throw std::runtime_error("empty typed root budget must not depend on probability contents");
    excel::Limits boundary;
    boundary.cells = upper_cells;
    excel::Plan accepted(*graph, inputs, {}, boundary);
    if (accepted.report().cells != plan.report().cells)
      throw std::runtime_error("empty typed root exact symbolic boundary");
    --boundary.cells;
    bool rejected = false;
    try {
      excel::Plan over_budget(*graph, inputs, {}, boundary);
    } catch (const excel::Error &error) {
      const std::string code = error.what();
      rejected = code.find("OVER_BUDGET") != std::string::npos &&
                 code.find("symbolic") != std::string::npos;
    }
    if (!rejected)
      throw std::runtime_error("empty typed root symbolic boundary minus one");
  }
  // A nonempty input with Q0 also has a zero-byte typed result; there is no
  // line sorting or sample consumption merely to make the destination nonnull.
  const std::vector<typed::Variable> q0_variables{
      {"x", typed::ValueType::matrix({"time", "scenario"}, {"T", "P"})},
      {"q", typed::ValueType::vector("Q")}};
  auto q0_graph = compiler::compile({"quantiles_axis(x,q,1)"}, q0_variables,
                                   false, {}, {}, 0, 100000000, true);
  excel::Plan q0_plan(*q0_graph,
                     {snapshot(ops::matrix_shape(2, 3), {1, 2, 3, 3, 4, 5}),
                      snapshot(ops::vector_shape(0), {})}, {}, {});
  const auto q0 = q0_plan.reference()[0];
  if (!(q0.shape == ops::matrix_shape(2, 0)) || !q0.error.empty() ||
      !q0.numbers.empty() || !q0.statuses.empty())
    throw std::runtime_error("empty Q0 typed root reference");
  empty_root_cells(q0_plan);
}
} // namespace

int main() {
  try {
    excel::Builder count({}, {});
    for (std::size_t i = 0; i < 2000000; ++i)
      count.put(i < 1000000 ? "A" : "B", i % 1000000, 0, "number", "1");
    if (count.report.cells != 2000000)
      throw std::runtime_error("exact workbook budget");
    bool rejected = false;
    try {
      count.put("B", 1000000, 0, "number", "1");
    } catch (const excel::Error &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("cell 2000001 must fail");
    excel::Builder work({}, {});
    rejected = false;
    try {
      work.cost({SIZE_MAX, SIZE_MAX});
    } catch (const excel::Error &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("checked cost multiplication");
    // Geometry-only snapshots intentionally have no readable payload. A
    // symbolic bound must not evaluate data or allocate per-element formulas.
    excel::Snapshot geometry;
    geometry.value.shape = ops::vector_shape(1000);
    auto symbolic =
        excel::symbolic_budget(nullptr, ops::Op::mean, {geometry}, {}, {});
    if (!symbolic.cells || symbolic.snapshot_bytes != 8000)
      throw std::runtime_error("metadata-only symbolic analysis");
    excel::Limits boundary;
    boundary.cells = symbolic.cells;
    excel::symbolic_budget(nullptr, ops::Op::mean, {geometry}, {}, boundary);
    --boundary.cells;
    rejected = false;
    try {
      excel::symbolic_budget(nullptr, ops::Op::mean, {geometry}, {}, boundary);
    } catch (const excel::Error &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("symbolic cell boundary");
    geometry.value.shape.rank = 2;
    geometry.value.shape.dim = {SIZE_MAX, 2, 0};
    rejected = false;
    try {
      excel::symbolic_budget(nullptr, ops::Op::transpose, {geometry}, {}, {});
    } catch (const excel::Error &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("symbolic shape multiplication overflow");
    // Validate constants independently of platform-specific stream parsing.
    for (double value : {1e-320, -1e-320}) {
      rejected = false;
      try {
        excel::Builder constants({}, {});
        constants.literal(value);
      } catch (const excel::Error &) {
        rejected = true;
      }
      if (!rejected)
        throw std::runtime_error("subnormal graph constant must fail");
    }
    excel::Snapshot input;
    input.value.shape = ops::vector_shape(3);
    input.numbers = {1, 2, 3};
    excel::Plan plan(ops::Op::mean, {input}, {});
    input.numbers[0] = -999;
    if (plan.reference()[0].numbers[0] != 2)
      throw std::runtime_error("owned snapshot");
    plan.cancel();
    rejected = false;
    try {
      plan.reference();
    } catch (const excel::Error &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("cancelled plan");
    axis_observer_isolation_contracts();
    empty_typed_root_contracts();
    std::cout
        << "Excel native budget, overflow, ownership, cancellation, axis observer isolation and empty roots passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
