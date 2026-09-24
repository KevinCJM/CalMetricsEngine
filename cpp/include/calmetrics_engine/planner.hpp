#pragma once
#include "calmetrics_engine/compiler.hpp"
#include <optional>

namespace calmetrics_engine::planner {
struct Config {
  double thread_work_units = 250000.0;
  double dag_branch_work_units = 5000000.0;
  double process_work_units = 4000000000.0;
  std::size_t process_input_threshold_bytes = 256 * 1024 * 1024;
  std::size_t shared_memory_threshold_bytes = 8 * 1024 * 1024;
  std::size_t min_rows_per_worker = 16;
  std::size_t simd_min_elements = 128;
  std::size_t max_processes = 8;
  std::size_t max_async_jobs = 8;
  void validate() const;
};
struct Chunk {
  std::size_t begin = 0, end = 0;
};
struct Geometry {
  std::size_t rows = 0, observations = 0, max_window = 0;
  std::size_t products = 0, max_intervals = 0;
  double sort_work = 0;
  std::uint64_t signature = 0;
  std::vector<Chunk> groups;
  std::vector<std::size_t> weights;
  std::vector<std::size_t> row_lengths;
  std::vector<double> row_work_units;
};

struct BranchTask {
  std::size_t row = 0;
  std::uint32_t branch_index = 0;
};
struct Plan {
  std::string model_identity;
  std::shared_ptr<compiler::CompiledGraph> graph;
  std::string lane = "single";
  std::size_t process_count = 1, thread_count = 1, threads_per_process = 1;
  bool use_shared_memory = false, async_orchestration = false,
       hard_stop = false;
  bool inputs_already_shared = false;
  double estimated_work_units = 0;
  double estimated_logical_work_units = 0;
  double estimated_typed_array_work_units = 0;
  std::size_t estimated_input_bytes = 0, estimated_output_bytes = 0;
  std::size_t estimated_status_bytes = 0;
  std::size_t estimated_result_metadata_bytes = 0;
  std::shared_ptr<const graph::ResultLayout> result_layout;
  std::size_t estimated_worker_scratch_bytes = 0,
              estimated_total_memory_bytes = 0;
  std::size_t row_count = 0, interval_observations = 0, product_count = 0;
  std::size_t max_intervals_per_product = 0, max_window = 0, cpu_budget = 1;
  std::optional<std::size_t> memory_budget_bytes;
  std::string parallel_dimension = "interval";
  std::vector<std::uint32_t> simd_nodes;
  std::vector<std::string> reason_codes;
  std::size_t tensor_elements = 0;
  std::vector<Chunk> chunks;
  std::vector<BranchTask> branch_tasks;
  std::vector<std::size_t> input_sizes;
  std::uint64_t geometry_signature = 0;
};
std::size_t checked_add(std::size_t a, std::size_t b);
std::size_t checked_mul(std::size_t a, std::size_t b);
Geometry inspect(const std::vector<std::size_t> &sizes,
                 const std::int64_t *starts, const std::int64_t *ends,
                 std::size_t rows, const std::int64_t *product_ids,
                 bool build_groups = true);
std::vector<Chunk>
partition(const Geometry &geometry, std::size_t workers, bool by_product,
          const compiler::PhysicalCost *physical_cost = nullptr);
std::shared_ptr<Plan>
make_plan(std::shared_ptr<compiler::CompiledGraph> graph, const Config &config,
          const std::vector<std::size_t> &sizes, const std::int64_t *starts,
          const std::int64_t *ends, std::size_t rows,
          const std::int64_t *product_ids, std::size_t cpu,
          std::optional<std::size_t> memory_budget = {}, bool hard_stop = false,
          bool async_io = false, bool already_shared = false,
          const std::vector<ops::Value> *inputs = nullptr, std::size_t owned_input_bytes = 0);
void validate_plan(const Plan &plan, const std::vector<std::size_t> &sizes,
                   const std::int64_t *starts, const std::int64_t *ends,
                   std::size_t rows, const std::int64_t *product_ids,
                   std::size_t engine_cpu,
                   const std::vector<ops::Value> *inputs = nullptr);
} // namespace calmetrics_engine::planner
