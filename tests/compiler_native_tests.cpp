#include "calmetrics_engine/compiler.hpp"
#include "calmetrics_engine/planner.hpp"
#include "compact_path_native_cases.hpp"
#include "node_type_sharing_native_cases.hpp"
#include "raw_prefix_native_cases.hpp"
#include <atomic>
#include <cmath>
#include <thread>
#include <utility>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace c = calmetrics_engine::compiler;
namespace g = calmetrics_engine::graph;
namespace o = calmetrics_engine::ops;
namespace t = calmetrics_engine::typed;
namespace p = calmetrics_engine::planner;

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
// Pin public catalog behavior independently of the production search tables.
// The first concurrent lookup also exercises initialization of the fixed index.
void catalog_contracts() {
  std::atomic<unsigned> ready{0};
  std::atomic<bool> start{false}, valid{true};
  std::vector<std::thread> workers;
  for (unsigned thread = 0; thread < 6; ++thread) {
    workers.emplace_back([&, thread] {
      ready.fetch_add(1);
      while (!start.load()) std::this_thread::yield();
      try {
        for (unsigned repeat = 0; repeat < 8; ++repeat)
          for (std::size_t i = 0; i < o::operator_count; ++i) {
            const auto &expected = o::registry()[(i + thread) % o::operator_count];
            if (&o::lookup(expected.name) != &expected ||
                &o::lookup(static_cast<std::uint16_t>(expected.op)) != &expected)
              valid.store(false);
          }
      } catch (...) { valid.store(false); }
    });
  }
  while (ready.load() != 6) std::this_thread::yield();
  start.store(true);
  for (auto &worker : workers) worker.join();
  require(valid.load(), "concurrent canonical name/opcode lookup");
  const auto rejects_operator = [](const std::string &name) {
    try { (void)o::lookup(std::string_view(name)); }
    catch (const o::Error &error) { return std::string(error.what()) == "UNKNOWN_OPERATOR"; }
    return false;
  };
  for (const auto &entry : o::registry()) {
    require(rejects_operator(std::string(entry.name) + "!"), "operator suffix accepted");
    require(rejects_operator("!" + std::string(entry.name)), "operator prefix accepted");
  }
  for (const auto &name : {std::string(), std::string("Mean"), std::string("mean\0extra", 10)})
    require(rejects_operator(name), "invalid operator lookup accepted");
  const std::pair<const char *, const char *> aliases[]{
      {"abs", "absolute"},
      {"corr", "correlation"},
      {"cov", "covariance"},
      {"cumulative_maximum", "cumulative_max"},
      {"cumulative_minimum", "cumulative_min"},
      {"elementwise_max", "maximum"},
      {"elementwise_min", "minimum"},
      {"eq", "equal"},
      {"ge", "greater_equal"},
      {"gt", "greater_than"},
      {"kurtosis_excess", "excess_kurtosis"},
      {"le", "less_equal"},
      {"lt", "less_than"},
      {"mad", "mean_absolute_deviation"},
      {"masked_count", "count_true"},
      {"masked_max", "max_where"},
      {"masked_mean", "mean_where"},
      {"masked_median", "median_where"},
      {"masked_min", "min_where"},
      {"masked_quantile", "quantile_where"},
      {"masked_std", "std_where"},
      {"masked_sum", "sum_where"},
      {"masked_variance", "variance_where"},
      {"max", "max_value"},
      {"min", "min_value"},
      {"mul", "multiply"},
      {"ne", "not_equal"},
      {"prod", "product"},
      {"rms", "root_mean_square"},
      {"safe_divide", "divide"},
      {"sequence_mean", "mean"},
      {"sequence_prod", "product"},
      {"sequence_std", "std"},
      {"sequence_sum", "sum"},
      {"skew", "skewness"},
      {"sub", "subtract"},
      {"var", "variance"}};
  for (const auto &entry : aliases)
    require(t::canonical_operator_name(entry.first) == entry.second,
            "historical operator alias changed");
  for (const auto &name : {std::string("undefined"), std::string("Sub"), std::string("sub\0x", 5)})
    require(t::canonical_operator_name(name) == name, "unknown alias changed");
  const char *semantics[]{"adjusted_market_price",
      "adjusted_nav",
      "calendar_days",
      "category",
      "count",
      "currency_amount",
      "date",
      "dimensionless",
      "event",
      "index",
      "mask",
      "phase",
      "rate_decimal",
      "raw_market_price",
      "reported_nav",
      "return_decimal",
      "state",
      "volume"};
  for (const auto *name : semantics) {
    require(t::normalize_semantic_dimension(name) == name, "semantic normalization changed");
    require(t::ValueType::scalar(name).semantic_dimension == name, "semantic validation changed");
  }
  const std::pair<const char *, const char *> semantic_aliases[]{
      {"currency", "currency_amount"},
      {"number", "dimensionless"},
      {"price", "raw_market_price"},
      {"price_change", "raw_market_price"},
      {"return", "return_decimal"},
      {"weight", "dimensionless"}};
  for (const auto &entry : semantic_aliases)
    require(t::normalize_semantic_dimension(entry.first) == entry.second, "semantic alias changed");
  for (const auto *name : {"derived:", "derived:count*volume", "squared:dimensionless", "inverse:rate_decimal"})
    require(t::normalize_semantic_dimension(name) == name, "derived semantic changed");
  for (const auto &name : {std::string("Dimensionless"), std::string("time"), std::string("dimensionless\0x", 15)}) {
    bool rejected = false;
    try { (void)t::normalize_semantic_dimension(name); }
    catch (const t::Error &error) {
      rejected = std::string(error.what()).find("SEMANTIC_DIMENSION_MISMATCH:") == 0;
    }
    require(rejected, "unsupported semantic accepted");
  }
  const char *axes[]{"asset",
      "component",
      "factor",
      "kalman_field",
      "model",
      "month",
      "path",
      "scenario",
      "segment_field",
      "state_field",
      "time",
      "window"};
  for (const auto *axis : axes)
    (void)t::ValueType::matrix({axis, "asset"}, {"N", "D"});
  for (const auto &axis : {std::string(), std::string("Asset"), std::string("asset!"), std::string("asset\0x", 7)}) {
    bool rejected = false;
    try { (void)t::ValueType::matrix({axis, "asset"}, {"N", "D"}); }
    catch (const t::Error &error) { rejected = std::string(error.what()).find("AXIS_MISMATCH:") == 0; }
    require(rejected, "unsupported axis accepted");
  }
}

void prebound_cholesky_budget_contract() {
  constexpr std::size_t n = 33, elements = n * n, matrix_bytes = elements * sizeof(double);
  for (bool covariance_pair : {false, true}) {
    std::vector<t::Variable> variables{{"x", t::ValueType::matrix({"asset", "asset"}, {"N", "N"})}};
    if (covariance_pair)
      variables.push_back({"samples", t::ValueType::matrix({"time", "asset"}, {"T", "N"})});
    const auto compile_iteration = [&] {
      return c::compile({covariance_pair ? "iterate(cholesky(covariance(samples)),x,0,1)"
                                       : "iterate(cholesky(iterate_x),x,0,1)"}, variables,
          false, {}, {}, 0, 100000000, true);
    };
    auto c_graph = compile_iteration();
    auto f_graph = compile_iteration();
    auto &body = *f_graph->program.apply_scopes[0].body;
    body.root_outputs[0].order = g::OutputOrder::f;
    body.finalize();
    f_graph->program.finalize();
    // F child schemas are a valid native/wire contract even though the public
    // compiler currently emits C iteration bodies.
    f_graph->program = c::decode_program(c::encode_program(f_graph->program));
    const auto &decoded_body = *f_graph->program.apply_scopes[0].body;
    require(decoded_body.root_outputs[0].order == g::OutputOrder::f &&
            decoded_body.execution_metadata->root_destinations[decoded_body.roots[0]] >= 0,
            "native F Cholesky child schema was not preserved");
    std::vector<double> input(elements, 0.);
    for (std::size_t i = 0; i < n; ++i) input[i * n + i] = 4.;
    input[1] = input[n] = 2.;
    input[n + 1] = 5.;
    o::Value view; view.data = input.data(); view.shape = o::matrix_shape(n, n);
    view.set_contiguous_strides(o::MemoryOrder::f);
    std::vector<double> samples((n + 1) * n, 0.);
    for (std::size_t i = 0; i < n; ++i) samples[i * (n + 1) + i] = 1.;
    const auto saved_samples = samples;
    o::Value sample_view; sample_view.data = samples.data(); sample_view.shape = o::matrix_shape(n + 1, n);
    sample_view.set_contiguous_strides(o::MemoryOrder::f);
    std::vector<o::Value> inputs;
    std::vector<std::size_t> input_sizes;
    for (const auto &name : c_graph->input_names) {
      inputs.push_back(name == "samples" ? sample_view : view);
      input_sizes.push_back(inputs.back().size());
    }
    const std::vector<o::Value> body_inputs{covariance_pair ? sample_view : view};
    const auto capacity = covariance_pair ? samples.size() : elements;
    std::vector<o::Shape> shapes;
    g::required_array_capacity(decoded_body, body_inputs, capacity, &shapes);
    g::ArrayArenaLayout ordinary, prebound;
    g::plan_array_arena(decoded_body, capacity, shapes, ordinary, true);
    g::plan_array_arena(decoded_body, capacity, shapes, prebound, true, true);
    require(ordinary.elements[0] == 0 && prebound.elements[0] == elements,
            "F Cholesky child downgrade must reserve exactly one C matrix");
    const std::int64_t start = 0, end = covariance_pair ? n + 1 : 0;
    const auto plan = [&](const std::shared_ptr<c::CompiledGraph> &graph,
                          std::optional<std::size_t> budget = {}) {
      return p::make_plan(graph, p::Config{}, input_sizes, &start, &end, 1,
          nullptr, 1, budget, false, false, false, &inputs);
    };
    const auto c_plan = plan(c_graph), f_plan = plan(f_graph);
    require(f_plan->estimated_worker_scratch_bytes == c_plan->estimated_worker_scratch_bytes + matrix_bytes &&
            f_plan->estimated_total_memory_bytes == c_plan->estimated_total_memory_bytes + matrix_bytes,
            "Planner omitted the prebound F Cholesky C arena");
    bool rejected = false;
    try { (void)plan(f_graph, c_plan->estimated_total_memory_bytes); }
    catch (const std::bad_alloc &) { rejected = true; }
    require(rejected, "F child was admitted without its C arena budget");
    require(plan(f_graph, f_plan->estimated_total_memory_bytes)->estimated_total_memory_bytes ==
            f_plan->estimated_total_memory_bytes, "exact F child budget was rejected");
    std::vector<double> pair_expected(elements);
    if (covariance_pair)
      g::execute(c_graph->program, inputs, nullptr, 0, &start, &end,
          1, pair_expected.data(), 1, c_plan->result_layout.get());
    std::vector<double> output(elements);
    const auto audit = g::execute(f_graph->program, inputs, nullptr, 0, &start, &end,
        1, output.data(), 1, f_plan->result_layout.get());
    // F input needs both C state buffers; the extra arena cannot hide in an
    // unused iteration-state reservation.
    require(audit.operator_workspace_capacity_bytes >= 3 * matrix_bytes &&
            audit.operator_workspace_capacity_bytes <= f_plan->estimated_worker_scratch_bytes &&
            audit.root_copy_bytes == matrix_bytes,
            "actual prebound Cholesky storage is not covered by Planner");
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        const double expected = covariance_pair ? pair_expected[i * n + j]
            : i == j ? 2. : (i == 1 && j == 0 ? 1. : 0.);
        const double original = i == j ? (i == 1 ? 5. : 4.) :
            ((i == 0 && j == 1) || (i == 1 && j == 0) ? 2. : 0.);
        require(output[i * n + j] == expected, "F child changed Cholesky result or transposed its factor");
        require(input[i * n + j] == original, "F child modified its input");
      }
    require(samples == saved_samples, "F covariance child modified captured observations");
  }
}

void compact_metadata_sharing_contract() {
  namespace compact = compact_path_tests;
  for (auto order : {o::MemoryOrder::c, o::MemoryOrder::f}) {
    compact::Spec spec;
    spec.projected = true;
    spec.groups = 2;
    spec.duplicates = 1;
    auto compiled = compact::compile(spec, order);
    const auto metadata = compiled->program.execution_metadata;
    require(metadata && metadata->compact_path && compiled->branches.size() == 1,
            "projected compact roots should form one proved component");
    const auto &branch = compiled->branches.front();
    require(branch.program.execution_metadata == metadata &&
            branch.program.execution_metadata->compact_path == metadata->compact_path,
            "identity compact branch did not share its immutable proof");
    require(branch.root_indices.size() == compiled->program.roots.size() &&
            branch.source_nodes.size() == compiled->program.nodes.size(),
            "shared branch lost complete roots or source nodes");
    for (std::size_t i = 0; i < branch.root_indices.size(); ++i)
      require(branch.root_indices[i] == i, "shared branch root mapping changed");
    for (std::size_t i = 0; i < branch.source_nodes.size(); ++i)
      require(branch.source_nodes[i] == i, "shared branch node mapping changed");
    require(c::encode_program(branch.program) == c::encode_program(compiled->program),
            "shared metadata changed the public native program");

    compact::Fixture fixture(spec);
    const std::vector<std::int64_t> starts{0}, ends{static_cast<std::int64_t>(spec.steps)};
    const auto expected = compact::run(*compiled, fixture, true, starts, ends);
    const auto actual = compact::run(*compiled, fixture, false, starts, ends);
    compact::compare(actual, expected);
    require(actual.audit.compact_path_groups == spec.groups,
            "shared graph did not execute the compact body");
    auto separate = std::make_shared<c::CompiledGraph>(*compiled);
    separate->program.finalize();
    separate->branches.front().program.finalize();
    require(separate->program.execution_metadata != separate->branches.front().program.execution_metadata,
            "independent finalization did not rebuild its own metadata");
    const auto inputs = fixture.inputs(*compiled);
    std::vector<std::size_t> input_sizes;
    for (const auto &value : inputs) input_sizes.push_back(value.size());
    const auto plan = [&](const std::shared_ptr<c::CompiledGraph> &graph) {
      return p::make_plan(graph, p::Config{}, input_sizes, starts.data(), ends.data(), 1,
          nullptr, 1, {}, false, false, false, &inputs);
    };
    {
      const auto shared_plan = plan(compiled), separate_plan = plan(separate);
      require(shared_plan->estimated_worker_scratch_bytes == separate_plan->estimated_worker_scratch_bytes &&
              shared_plan->estimated_total_memory_bytes == separate_plan->estimated_total_memory_bytes &&
              shared_plan->estimated_work_units == separate_plan->estimated_work_units,
              "metadata ownership sharing changed execution reservation or work");
    }
    c::CompiledGraph independent = *compiled;
    independent.program = branch.program;
    compiled.reset();
    const auto after_release = compact::run(independent, fixture, false, starts, ends);
    compact::compare(after_release, expected);
    require(after_release.data() != actual.data(), "shared proof aliased independent result storage");

    // A real selected root has no full consumer closure. Its branch must keep
    // independently finalized metadata rather than inheriting a stale proof.
    const auto selected = c::compile({independent.expressions.front()}, independent.variable_types,
        false, {independent.root_bindings.front()}, {}, 0, 100000000, true, {},
        order == o::MemoryOrder::f ? g::OutputOrder::f : g::OutputOrder::c);
    require(!selected->program.execution_metadata->compact_path && selected->branches.size() == 1 &&
            selected->branches.front().program.execution_metadata != selected->program.execution_metadata,
            "selected root inherited a complete compact proof");
    compact::compare(compact::run(*selected, fixture, false, starts, ends),
                     compact::run(*selected, fixture, true, starts, ends));
  }

  // Two independent period inputs form two proper root/node subsets and
  // compact their slots. Each finalized child proof must use its own IDs.
  compact::Spec periods;
  periods.groups = 2;
  auto compiled = compact::compile(periods);
  require(compiled->program.execution_metadata->compact_path && compiled->branches.size() == 2,
          "independent compact groups should retain separate branches");
  compact::Fixture fixture(periods);
  const std::vector<std::int64_t> starts{0}, ends{static_cast<std::int64_t>(periods.steps)};
  bool compacted_slots = false;
  for (const auto &branch : compiled->branches) {
    require(branch.root_indices.size() < compiled->program.roots.size() &&
            branch.source_nodes.size() < compiled->program.nodes.size() &&
            branch.program.execution_metadata != compiled->program.execution_metadata &&
            branch.program.execution_metadata->compact_path &&
            branch.program.execution_metadata->compact_path != compiled->program.execution_metadata->compact_path,
            "proper branch subset reused parent node/root metadata");
    compacted_slots |= branch.program.numeric_slots < compiled->program.numeric_slots ||
                       branch.program.mask_slots < compiled->program.mask_slots ||
                       branch.program.integer_slots < compiled->program.integer_slots;
    c::CompiledGraph child = *compiled;
    child.program = branch.program;
    compact::compare(compact::run(child, fixture, false, starts, ends),
                     compact::run(child, fixture, true, starts, ends));
  }
  require(compacted_slots, "nonidentity slot-remapping fixture was not exercised");
}

void compact_parent_storage_contracts() {
  const g::ParentSlots empty;
  require(empty.size() == 32 && empty.prefix(0).empty() && empty.overflow_bytes() == 0,
          "parent slot maximum or empty allocation changed");
  for (std::size_t i = 0; i < empty.size(); ++i) require(empty[i] == 0, "default parent slot not zero");
  require(&empty[8] == &empty[31], "absent parent tail must refer to the shared immutable zero");
  g::ParentSlots values;
  for (std::size_t i = 0; i < 8; ++i) values[i] = static_cast<std::uint32_t>(i + 1);
  require(values.overflow_bytes() == 0, "low-arity parent slots allocate a tail");
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = static_cast<std::uint32_t>(i + 1);
  require(values.overflow_bytes() == 24 * sizeof(std::uint32_t), "parent tail allocation size changed");
  const auto &const_values = values;
  for (std::size_t i = 0; i < values.size(); ++i)
    require(&const_values[i] == &values[i], "const parent access returned a temporary reference");
  require(values.prefix(32).size() == 32 && values.contains(32, 32) &&
          !values.contains(9, 8), "parent prefix or dependency membership changed");
  auto copy = values;
  copy[7] = 71; copy[8] = 81; copy[31] = 311;
  require(values[7] == 8 && values[8] == 9 && values[31] == 32 && copy != values,
          "parent copy aliases inline or overflow slots");
  copy = copy;
  require(copy[31] == 311, "parent self-copy lost tail");
  g::ParentSlots assigned;
  assigned = values;
  assigned[8] = 82;
  require(values[8] == 9, "parent assignment aliases tail");
  assigned = empty;
  require(assigned == empty, "parent assignment did not release old tail");
  auto moved = std::move(copy);
  require(moved[7] == 71 && moved[8] == 81 && moved[31] == 311, "parent move lost slots");
  assigned = std::move(moved);
  require(assigned[31] == 311, "parent move assignment lost tail");
  g::ParentSlots implicit_zero, explicit_zero;
  explicit_zero[31] = 0;
  require(implicit_zero == explicit_zero && explicit_zero.overflow_bytes() == 96,
          "unused zero tail changed identity comparison or resource visibility");
  explicit_zero[31] = 1;
  require(implicit_zero != explicit_zero, "unused nonzero tail omitted from identity comparison");

  // Thirty scalar captures + x + width exercise all 32 real dependencies,
  // not merely an artificial container with an unused overflow value.
  std::vector<std::pair<std::string, std::string>> variables{{"x", "series"}};
  std::string body = "sum(x)";
  std::vector<double> parameters;
  for (std::size_t i = 0; i < 30; ++i) {
    const auto name = "p" + std::to_string(i);
    variables.emplace_back(name, "scalar"); body += "+" + name;
    parameters.push_back(static_cast<double>(i + 1));
  }
  auto compiled = c::compile({"sum(block_apply(" + body + ",2))"}, variables);
  const auto &scope_node = compiled->program.nodes[compiled->program.roots[0] - 1];
  require(scope_node.kind == g::NodeKind::apply_scope && scope_node.parent_count == 32,
          "full dependency scope fixture did not use 32 parents");
  const auto encoded = c::encode_program(compiled->program);
  auto decoded = c::decode_program(encoded);
  require(c::encode_program(decoded) == encoded, "32-parent encoded graph round trip changed");
  auto independent = decoded;
  const auto scope_id = compiled->program.roots[0] - 1;
  const auto saved_parent = decoded.nodes[scope_id].parents[31];
  independent.nodes[scope_id].parents[31] = 0;
  require(decoded.nodes[scope_id].parents[31] == saved_parent, "Program copy shares mutable parent tail");
  std::vector<double> data{1, 2, 3, 4, 5, 6};
  o::Value input; input.shape = o::vector_shape(data.size()); input.data = data.data(); input.stride[0] = 1;
  std::int64_t start = 0, end = 6;
  double result = 0;
  g::execute(decoded, {input}, parameters.data(), parameters.size(), &start, &end, 1, &result, 1);
  require(result == 21 + 3 * 465, "32-parent block result changed");
  decoded.nodes[scope_id].parent_count = 33;
  bool rejected = false;
  try { decoded.validate(); } catch (const o::Error &) { rejected = true; }
  require(rejected, "33-parent malformed Program was accepted");

  compact_path_tests::Spec spec;
  auto compact = compact_path_tests::compile(spec);
  compact_path_tests::Fixture fixture(spec);
  const auto inputs = fixture.inputs(*compact);
  const auto baseline = g::compact_path_budget(compact->program,
      *compact->program.execution_metadata->compact_path, inputs, {spec.steps},
      p::canonical_child_scratch_bytes);
  auto tailed = compact->program;
  for (auto &node : tailed.nodes) node.parents[31] = 0;
  tailed.finalize();
  require(c::encode_program(tailed) == c::encode_program(compact->program),
          "unused zero parent storage changed encoded semantics");
  const auto with_tail = g::compact_path_budget(tailed,
      *tailed.execution_metadata->compact_path, inputs, {spec.steps}, p::canonical_child_scratch_bytes);
  require(baseline && with_tail && with_tail->metadata_bytes > baseline->metadata_bytes &&
          with_tail->total_bytes - baseline->total_bytes ==
            with_tail->metadata_bytes - baseline->metadata_bytes,
          "retained unused parent tail was omitted from compact resource admission");
}

int main() {
  try {
    catalog_contracts();
    prebound_cholesky_budget_contract();
    compact_metadata_sharing_contract();
    compact_parent_storage_contracts();
    node_type_sharing::contracts();
    test_raw_prefix_contracts();
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
