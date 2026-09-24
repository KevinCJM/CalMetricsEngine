#pragma once

#include "calmetrics_engine/operators.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <atomic>
#include <chrono>
#include <functional>
#include <vector>

namespace calmetrics_engine::graph {

enum class NodeKind : std::uint8_t {
  input = 0,
  parameter = 1,
  constant = 2,
  operation = 3,
  rolling_scope = 4,
  interval_tail = 5,
  apply_scope = 6,
  iteration_projection = 7
};

enum class OutputKind : std::uint8_t { scalar = 0, series = 1, typed = 2 };
inline const char *output_kind_name(OutputKind kind) noexcept {
  return kind == OutputKind::typed ? "typed" : kind == OutputKind::series ? "series" : "scalar";
}
// Public scalar results retain float64. Aligned series preserve one exact
// dtype for every root in the graph, including across native worker transport.
enum class OutputDType : std::uint8_t { float64 = 0, boolean = 1, int64 = 2 };
inline std::size_t output_itemsize(OutputDType dtype) noexcept {
  return dtype == OutputDType::boolean ? 1 : 8;
}
inline const char *output_dtype_name(OutputDType dtype) noexcept {
  switch (dtype) {
  case OutputDType::float64: return "float64";
  case OutputDType::boolean: return "bool";
  case OutputDType::int64: return "int64";
  }
  return "invalid";
}
inline void *output_offset(void *data, std::size_t elements,
                           OutputDType dtype) noexcept {
  return data ? static_cast<std::uint8_t *>(data) + elements * output_itemsize(dtype)
              : nullptr;
}
// Per-root native schema. Typed storage never coerces integer or mask values.
struct RootOutput { OutputDType dtype = OutputDType::float64; std::uint8_t rank = 0; };
struct ResultSlot { std::size_t byte_offset = 0, status_offset = 0, capacity = 0; };
struct ResultLayout {
  std::size_t columns = 0;
  std::vector<ResultSlot> slots;
  std::vector<std::size_t> row_bytes, row_statuses;
  std::size_t bytes() const { return row_bytes.empty() ? 0 : row_bytes.back(); }
  std::size_t status_count() const { return row_statuses.empty() ? 0 : row_statuses.back(); }
};
enum class RollingParameterKind : std::uint8_t {
  outer_node = 0,
  observation_count = 1,
  window_elapsed_days = 2,
  risk_free_return_window = 3
};
enum class StorageKind : std::uint8_t {
  inline_value = 0,
  numeric = 1,
  mask = 2,
  integer = 3
};

struct Node {
  NodeKind kind = NodeKind::constant;
  std::uint16_t opcode = 0;
  std::uint16_t input_index = 0;
  double constant = 0.0;
  std::array<std::uint32_t, 32> parents{};
  std::uint8_t parent_count = 0;
  StorageKind storage = StorageKind::inline_value;
  std::uint32_t slot = 0;
};

struct Program;

struct RollingParameterBinding {
  RollingParameterKind kind = RollingParameterKind::outer_node;
  std::uint32_t node = 0;
};

struct RollingScope {
  std::shared_ptr<Program> body;
  std::vector<std::uint32_t> input_nodes;
  std::vector<std::uint8_t> input_preceding;
  std::vector<RollingParameterBinding> parameter_bindings;
  std::uint32_t width_node = 0;
  std::uint32_t min_periods_node = 0;
  std::uint32_t dates_node = 0;
  std::uint32_t annual_rate_node = 0;
  bool has_min_periods = false;
  bool has_date_context = false;
  bool has_returns = false;
  bool needs_preceding_observation = false;
  std::int32_t returns_input = -1;
  std::size_t body_node_count = 0;
  std::size_t array_count = 0;
};

enum class ApplyKind : std::uint8_t { block = 0, filter = 1, group = 2, bisect = 3, segment = 4, iterate = 5 };
enum class SolverStatus : std::int64_t {
  converged = 0, iteration_limit = 1, no_feasible_start = 2,
  infeasible = 3, local_stop = 4, numerical_failure = 5
};
struct ApplyScope {
  ApplyKind kind = ApplyKind::block;
  std::shared_ptr<Program> body;
  std::vector<std::uint32_t> input_nodes;
  // UINT32_MAX binds the local solver variable rather than an outer node.
  std::vector<std::uint32_t> parameter_nodes;
  std::vector<std::uint32_t> argument_nodes;
  std::size_t body_node_count = 0;
  std::int32_t state_input_index = -1;
};

// Process-local AOT kernels are rebuilt by finalize(), never serialized.
struct PointwiseRegion {
  std::uint32_t source = 0, first = 0, second = UINT32_MAX, predicate = UINT32_MAX;
  std::array<std::uint32_t, 3> scalars{UINT32_MAX, UINT32_MAX, UINT32_MAX};
  std::array<bool, 3> scalar_first{};
  ops::ScalarChainKernel kernel = nullptr;
  bool retain_first = true;
  std::uint32_t end() const { return predicate != UINT32_MAX ? predicate : second; }
};
struct ExecutionMetadata {
  bool pointwise = false;
  bool numeric_roots_only = false;
  std::vector<std::int32_t> root_destinations;
  bool requires_shape_planning = false;
  std::vector<const ops::Spec *> specs;
  std::vector<std::uint8_t> summary_consumers;
  std::vector<std::uint8_t> order_consumers;
  // Keep optional provenance after the existing hot execution metadata.
  std::size_t position_status_nodes = 0;
  std::vector<std::uint8_t> position_status_reachable;
  bool contains_segment_scope = false;
  std::vector<PointwiseRegion> pointwise_regions;
  std::size_t pointwise_numeric_slots = 0, pointwise_mask_slots = 0;
  bool pointwise_complete_region = false;
  std::vector<std::size_t> pointwise_root_copies;
};

struct Program {
  std::vector<Node> nodes;
  std::vector<std::uint32_t> roots;
  std::size_t input_count = 0;
  std::size_t parameter_count = 0;
  std::size_t numeric_slots = 0;
  std::size_t mask_slots = 0;
  std::size_t integer_slots = 0;
  // 0: temporal vector, 1: static vector/matrix, 2: time x asset matrix.
  std::vector<std::uint8_t> input_axes;
  OutputKind output_kind = OutputKind::scalar;
  OutputDType output_dtype = OutputDType::float64;
  std::vector<RootOutput> root_outputs; // present only for typed results
  bool isolate_errors = false;
  std::uint32_t minimum_observations = 0;
  std::uint64_t scope_work_budget = 100000000;
  std::vector<RollingScope> rolling_scopes;
  std::vector<ApplyScope> apply_scopes;
  std::shared_ptr<const ExecutionMetadata> execution_metadata;

  void validate() const;
  void finalize();
};

bool summary_fusion_eligible(ops::Op op) noexcept;
void release_graph_workspace();
bool order_fusion_eligible(ops::Op op) noexcept;

std::size_t required_array_capacity(const Program &program,
                                   const std::vector<ops::Value> &inputs,
                                   std::size_t max_window,
                                   std::vector<ops::Shape> *node_shapes = nullptr);

ResultLayout result_layout(const Program &, const std::vector<ops::Value> &,
                           const std::int64_t *starts, const std::int64_t *ends, std::size_t rows);

struct Audit {
  std::size_t fused_pointwise_calls = 0, pointwise_tiles = 0, vector_elements = 0;
  std::size_t root_copy_bytes = 0, direct_output_bytes = 0, state_copy_bytes = 0;
  std::size_t pointwise_workspace_capacity_bytes = 0;
  std::vector<ops::Shape> result_shapes; // row/root order; rank=-1 means unknown on failure
  std::vector<std::int16_t> root_statuses;
  // Present only for isolate mode; row-major, with the same shape as values.
  std::vector<std::int16_t> statuses;
  std::size_t rows = 0;
  std::size_t nodes = 0;
  std::size_t max_window = 0;
  std::size_t numeric_arena_bytes = 0;
  std::size_t mask_arena_bytes = 0;
  std::size_t operator_workspace_capacity_bytes = 0;
  std::size_t input_copy_bytes = 0;
  std::size_t fused_scalar_calls = 0;
  std::size_t summary_source_scans = 0;
  std::size_t order_stat_sorts = 0;
  std::size_t order_scratch_capacity_bytes = 0;
  std::size_t algorithm_copy_bytes = 0;
};

Audit execute(const Program &program, const std::vector<ops::Value> &inputs,
              const double *parameters, std::size_t parameter_count,
              const std::int64_t *starts, const std::int64_t *ends,
              std::size_t rows, void *output, std::size_t output_columns,
              const ResultLayout *layout = nullptr, std::size_t result_row = 0,
              std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max(),
              const std::atomic<bool> *cancelled = nullptr);

// Explicit diagnostic replay observes canonical operands/results, including
// nested scopes. Ordinary execution does not scan values or invoke callbacks.
using ValueObserver = std::function<void(ops::Op, const ops::Value &, bool)>;
Audit execute_observed(const Program &program, const std::vector<ops::Value> &inputs,
                       const double *parameters, std::size_t parameter_count,
                       const std::int64_t *starts, const std::int64_t *ends,
                       std::size_t rows, void *output, std::size_t output_columns,
                       const ResultLayout *layout, const ValueObserver &observer);

} // namespace calmetrics_engine::graph
