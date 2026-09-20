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

void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
std::vector<double> run(const std::vector<std::string> &formulas) {
  auto compiled = c::compile(formulas, {{"x", "series"}});
  std::vector<double> x{1, 2, 3, 4, 5, 6};
  o::Value v;
  v.shape = o::vector_shape(x.size());
  v.data = x.data();
  v.stride[0] = 1;
  std::int64_t start = 0, end = static_cast<std::int64_t>(x.size());
  std::vector<double> result(formulas.size());
  auto encoded = c::encode_program(compiled->program);
  auto decoded = c::decode_program(encoded);
  g::execute(decoded, {v}, nullptr, 0, &start, &end, 1, result.data(),
             formulas.size());
  return result;
}
int main() {
  try {
    auto values =
        run({"-2**2", "2**-2", "2**3**2", "(2+3)*4", "mean(x)", "std(x,1)"});
    require(values[0] == -4 && values[1] == 0.25 && values[2] == 512 &&
                values[3] == 20,
            "precedence");
    require(values[4] == 3.5 && std::abs(values[5] - std::sqrt(3.5)) < 1e-14,
            "reference parity");
    auto aliased = run({"mean(lag(x+1,1))", "mean(x*2)", "std(lag(x+1,1),1)"});
    require(aliased[0] == 4 && aliased[1] == 7 &&
                std::abs(aliased[2] - std::sqrt(2.5)) < 1e-14,
            "borrowed lifetime");
    auto graph = c::compile({"mean(x)", "mean(x)/std(x,1)", "std(x,1)"},
                            {{"x", "series"}});
    std::size_t means = 0;
    for (const auto &node : graph->nodes)
      means += node.node.opcode == static_cast<std::uint16_t>(o::Op::mean);
    require(means == 1 && graph->raw_node_count > graph->nodes.size(), "CSE");
    require(graph->physical_cost.linear_per_observation > 0 &&
                graph->branches.size() == 1,
            "physical metadata");

    auto physical =
        c::compile({"mean(x)", "std(x,1)", "median(x)", "quantile(x,.25)",
                    "-min_value(drawdown_series(x))", "linear_slope(x)",
                    "linear_r_squared(x)"},
                   {{"x", "series"}});
    require(physical->branches.size() >= 3, "independent branches");
    for (const auto &branch : physical->branches) {
      require(branch.program.numeric_slots <= physical->program.numeric_slots &&
                  branch.program.mask_slots <= physical->program.mask_slots,
              "branch arena compaction");
      require(!branch.root_indices.empty() && !branch.source_nodes.empty(),
              "branch metadata");
    }
    for (const auto &source :
         {"x[0]", "x.real", "__import__('os')", "(lambda x:x)(x)",
          "mean(x)<2<3", "fit_slope(1)", "mean(x,1)", "logical_and(x,x)"}) {
      bool rejected = false;
      try {
        c::compile({source}, {{"x", "series"}});
      } catch (const std::exception &) {
        rejected = true;
      }
      require(rejected, "invalid AST accepted");
    }
    bool limited = false;
    try {
      c::compile({std::string(200, '(') + "mean(x)" + std::string(200, ')')},
                 {{"x", "series"}});
    } catch (const c::CompileError &) {
      limited = true;
    }
    require(limited, "depth limit");
    std::vector<t::Variable> typed_variables{
        {"x", t::ValueType::series("T", "return_decimal")}};
    auto alias_graph = c::compile({"mean(sub(x,0))"}, typed_variables);
    auto canonical_graph = c::compile({"mean(subtract(x,0))"}, typed_variables);
    require(alias_graph->fingerprint == canonical_graph->fingerprint,
            "typed alias canonicalization");

    auto rolling =
        c::compile({"rolling_apply(mean(x),3)"}, typed_variables);
    require(rolling->program.output_kind == g::OutputKind::series &&
                rolling->program.rolling_scopes.size() == 1,
            "rolling scope compile");
    std::vector<double> input{1, 2, 3, 4, 5, 6};
    o::Value input_view;
    input_view.shape = o::vector_shape(input.size());
    input_view.data = input.data();
    input_view.stride[0] = 1;
    std::int64_t rolling_start = 0,
                 rolling_end = static_cast<std::int64_t>(input.size());
    std::vector<double> rolling_result(input.size());
    auto rolling_decoded = c::decode_program(c::encode_program(rolling->program));
    g::execute(rolling_decoded, {input_view}, nullptr, 0, &rolling_start,
               &rolling_end, 1, rolling_result.data(), 1);
    require(std::isnan(rolling_result[0]) && std::isnan(rolling_result[1]) &&
                rolling_result[2] == 2 && rolling_result[3] == 3 &&
                rolling_result[4] == 4 && rolling_result[5] == 5,
            "rolling scope execution");

    bool semantic_rejected = false;
    try {
      c::compile(
          {"add(a,b)"},
          std::vector<t::Variable>{
              {"a", t::ValueType::series("T", "adjusted_nav", "hfq")},
              {"b", t::ValueType::series("T", "adjusted_nav", "qfq")}});
    } catch (const c::CompileError &) {
      semantic_rejected = true;
    }
    require(semantic_rejected, "typed price-basis mismatch");

    auto truncated = c::encode_program(graph->program);
    truncated.pop_back();
    bool rejected = false;
    try {
      c::decode_program(truncated);
    } catch (const std::exception &) {
      rejected = true;
    }
    require(rejected, "truncated program");
    std::cout << "native compiler tests passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
