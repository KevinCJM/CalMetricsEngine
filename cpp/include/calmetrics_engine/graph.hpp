#pragma once

#include "calmetrics_engine/operators.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace calmetrics_engine::graph {

enum class NodeKind : std::uint8_t {
  input = 0,
  parameter = 1,
  constant = 2,
  operation = 3
};
enum class StorageKind : std::uint8_t {
  inline_value = 0,
  numeric = 1,
  mask = 2
};

struct Node {
  NodeKind kind = NodeKind::constant;
  std::uint16_t opcode = 0;
  std::uint16_t input_index = 0;
  double constant = 0.0;
  std::array<std::uint32_t, 4> parents{};
  std::uint8_t parent_count = 0;
  StorageKind storage = StorageKind::inline_value;
  std::uint32_t slot = 0;
};

struct ExecutionMetadata {
  std::vector<const ops::Spec *> specs;
  std::vector<std::uint8_t> summary_consumers;
  std::vector<std::uint8_t> order_consumers;
};

struct Program {
  std::vector<Node> nodes;
  std::vector<std::uint32_t> roots;
  std::size_t input_count = 0;
  std::size_t parameter_count = 0;
  std::size_t numeric_slots = 0;
  std::size_t mask_slots = 0;
  std::shared_ptr<const ExecutionMetadata> execution_metadata;

  void validate() const;
  void finalize();
};

bool summary_fusion_eligible(ops::Op op) noexcept;
bool order_fusion_eligible(ops::Op op) noexcept;

struct Audit {
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
              std::size_t rows, double *output, std::size_t output_columns);

} // namespace calmetrics_engine::graph
