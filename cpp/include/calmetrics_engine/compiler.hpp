#pragma once

#include "calmetrics_engine/graph.hpp"
#include "calmetrics_engine/typed_ir.hpp"
#include <memory>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace calmetrics_engine::compiler {

enum class ValueClass : std::uint8_t {
  scalar,
  series,
  mask_scalar,
  mask_series,
  integer_series,
  fit,
  interval
};
struct CompileError : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

struct NodeInfo {
  std::uint32_t node_id = 0;
  graph::Node node;
  ValueClass value_class = ValueClass::scalar;
  typed::ValueType inferred_type = typed::ValueType::scalar();
  std::uint32_t last_use = 0;
  std::string cost_model = "constant";
  bool simd_eligible = false;
};

struct PhysicalCost {
  double constant_per_row = 0.0;
  double linear_per_observation = 0.0;
  double sort_nlogn = 0.0;
};

struct BranchInfo {
  std::vector<std::uint32_t> root_indices;
  std::vector<std::uint32_t> source_nodes;
  graph::Program program;
  PhysicalCost cost;
};

struct CompiledGraph {
  graph::Program program;
  std::vector<NodeInfo> nodes;
  std::vector<std::string> expressions;
  std::vector<std::string> source_contracts;
  std::vector<std::map<std::string, std::string>> root_bindings;
  std::vector<std::pair<std::string, std::string>> variables;
  std::vector<typed::Variable> variable_types;
  std::vector<std::string> input_names;
  std::vector<std::string> parameter_names;
  std::vector<std::string> output_names;
  std::string fingerprint;
  graph::OutputKind output_kind = graph::OutputKind::scalar;
  std::size_t raw_node_count = 0;
  PhysicalCost physical_cost;
  std::vector<BranchInfo> branches;
};

const char *class_name(ValueClass type) noexcept;
const char *kind_name(graph::NodeKind kind) noexcept;
const char *storage_name(graph::StorageKind storage) noexcept;

// Never invokes Python, eval, a JIT or a callback. Limits are enforced before
// recursion/allocation.
std::shared_ptr<CompiledGraph>
compile(const std::vector<std::string> &expressions,
        const std::vector<std::pair<std::string, std::string>> &variables);

std::shared_ptr<CompiledGraph>
compile(const std::vector<std::string> &expressions,
        const std::vector<typed::Variable> &variables,
        bool isolate_errors = false,
        const std::vector<std::map<std::string, std::string>> &root_bindings = {},
        const std::vector<std::string> &source_contracts = {},
        std::uint32_t minimum_observations = 0,
        std::uint64_t scope_work_budget = 100000000,
        bool typed_results = false,
        const std::vector<std::string> &output_names = {});

// Versioned, pointer-free plan representation shared by bindings and native
// workers.
std::vector<std::uint8_t> encode_program(const graph::Program &program);
graph::Program decode_program(const std::vector<std::uint8_t> &bytes);

} // namespace calmetrics_engine::compiler
