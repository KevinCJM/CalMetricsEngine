#include "calmetrics_engine/compiler.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace c = calmetrics_engine::compiler;
namespace g = calmetrics_engine::graph;
namespace o = calmetrics_engine::ops;
namespace t = calmetrics_engine::typed;

static void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}

static o::Value numbers(const std::vector<double> &values) {
  o::Value view;
  view.shape = o::vector_shape(values.size());
  view.data = values.data();
  return view;
}

static std::vector<double> execute(const c::CompiledGraph &graph,
                                  const std::vector<o::Value> &inputs,
                                  std::int64_t length, g::Audit *audit = nullptr) {
  // Every new scope and dtype travels through the standalone worker wire format.
  auto decoded = c::decode_program(c::encode_program(graph.program));
  require(decoded.scope_work_budget == graph.program.scope_work_budget, "budget serialization");
  std::int64_t start = 0;
  std::vector<double> out(graph.program.roots.size() *
      (graph.output_kind == g::OutputKind::series ? static_cast<std::size_t>(length) : 1));
  auto result = g::execute(decoded, inputs, nullptr, 0, &start, &length, 1,
                          out.data(), graph.program.roots.size());
  if (audit) *audit = std::move(result);
  return out;
}

int main() {
  try {
    std::vector<double> values{1, NAN, 3, 7, 2, 9, 99};
    std::vector<t::Variable> variables{{"x", t::ValueType::series()}};
    auto scopes = c::compile({
        "filter_apply(sum(difference(x)),finite_mask(x))",
        "sum(block_apply(sum_where(x,finite_mask(x)),2))",
        "bisect(solve_x*solve_x-2,0,2,1e-12,100)",
        "filter_apply(7,x>100,0)",
        "sum(block_apply(2,2))"}, variables);
    auto result = execute(*scopes, {numbers(values)}, values.size());
    require(result[0] == 98 && result[1] == 22 && result[3] == 0 && result[4] == 6,
            "block/filter sequence semantics");
    require(std::abs(result[2] - std::sqrt(2.0)) < 1e-12, "bounded solve");

    auto type = t::ValueType::series();
    type.dtype = t::DType::int64;
    std::vector<t::Variable> grouped_variables{{"x", t::ValueType::series()}, {"key", type}};
    std::vector<double> group_values{4, -1, 4, -1};
    std::vector<std::int64_t> keys{1LL << 60, (1LL << 60) + 1, 1LL << 60, (1LL << 60) + 1};
    o::Value key_view;
    key_view.kind = o::Kind::integer;
    key_view.shape = o::vector_shape(keys.size());
    key_view.data = keys.data();
    auto groups = c::compile({"group_apply(bisect(solve_x*solve_x-mean(x),0,10,1e-12,100),key)"},
                            grouped_variables, true);
    g::Audit audit;
    result = execute(*groups, {numbers(group_values), key_view}, keys.size(), &audit);
    require(std::abs(result[0] - 2) < 1e-12 && std::abs(result[2] - 2) < 1e-12 &&
            std::isnan(result[1]) && std::isnan(result[3]), "group isolation or exact IDs");
    require(audit.statuses == std::vector<std::int16_t>({0, 4, 0, 4}), "group row statuses");

    auto order = c::compile({"sum(gather(x,argsort(x)))", "distinct_count(key)"}, grouped_variables);
    result = execute(*order, {numbers(group_values), key_view}, keys.size());
    require(result[0] == 6 && result[1] == 2 && order->program.integer_slots > 0,
            "integer arena and gather");

    std::vector<double> matrix{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto matrix_view = numbers(matrix);
    matrix_view.shape = o::matrix_shape(3, 3);
    matrix_view.stride = {3, 1};
    auto matrix_graph = c::compile({
        "sum(transpose(x+1)+transpose(x+2))",
        "sum(diag(x+1)+diag(x+2))"},
        std::vector<t::Variable>{{"x", t::ValueType::matrix({"asset", "asset"}, {"N", "N"})}});
    result = execute(*matrix_graph, {matrix_view}, 3);
    require(result[0] == 117 && result[1] == 39, "borrowed transpose/diag lifetime");
    for (const auto &branch : matrix_graph->branches) {
      const std::int64_t start = 0, end = 3;
      std::vector<double> out(branch.program.roots.size());
      g::execute(branch.program, {matrix_view}, nullptr, 0, &start, &end, 1, out.data(), out.size());
      for (std::size_t i = 0; i < out.size(); ++i)
        require(out[i] == result[branch.root_indices[i]], "branch borrowed lifetime");
    }

    auto budget = c::compile({"filter_apply(sum(x),finite_mask(x))"}, variables, false, {}, {}, 0, 1);
    bool rejected = false;
    try { execute(*budget, {numbers(values)}, values.size()); }
    catch (const o::Error &error) { rejected = std::string(error.what()) == "SCOPE_COMPUTE_BUDGET_EXCEEDED"; }
    require(rejected, "scope work budget");

    auto corrupt = c::encode_program(groups->program);
    corrupt.pop_back();
    rejected = false;
    try { c::decode_program(corrupt); } catch (const std::exception &) { rejected = true; }
    require(rejected, "truncated nested native plan");
    std::cout << "native mathematical scopes passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
