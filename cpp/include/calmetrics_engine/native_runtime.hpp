#pragma once
#include "calmetrics_engine/planner.hpp"
#include "calmetrics_engine/scheduler.hpp"
#include "calmetrics_engine/shared_memory.hpp"
#include "calmetrics_engine/inline_vector.hpp"
#include <atomic>
#include <chrono>
#include <mutex>
#include <utility>

namespace calmetrics_engine::native {
using Clock = std::chrono::steady_clock;
using Deadline = Clock::time_point;
struct Timeout : std::runtime_error {
  using std::runtime_error::runtime_error;
};
inline void check_deadline(Deadline deadline) {
  if (deadline != Deadline::max() && Clock::now() >= deadline)
    throw Timeout("native execution deadline exceeded");
}

struct Batch {
  std::string model_identity;
  std::size_t owned_input_bytes = 0;
  std::vector<ops::Value> inputs;
  const double *parameters = nullptr;
  std::size_t parameter_count = 0;
  const std::int64_t *starts = nullptr, *ends = nullptr, *product_ids = nullptr;
  std::size_t rows = 0;
  std::vector<SharedDescriptor> shared_inputs;
  std::vector<std::shared_ptr<SharedRegion>> shared_owners;
  std::vector<std::size_t> input_sizes() const;
};
struct ExecutionAudit {
  InlineVector<graph::Audit, 1> chunks;
  double queue_wait_ms = 0, compute_ms = 0, elapsed_ms = 0;
  std::size_t cpu_tokens = 0, cpu_budget = 0;
  std::size_t shared_memory_bytes = 0, boundary_copy_bytes = 0,
              output_copy_bytes = 0;
  std::size_t native_threads = 0, native_processes = 0;
  // Optional completed shared output. Its owner can be pinned directly by a
  // Python ndarray after all workers have settled; no return-path copy.
  std::shared_ptr<SharedRegion> output_owner;
};
class ProcessPool;
class Engine {
public:
  // One request only: metadata/intervals must remain unchanged until execution.
  // Not a cached proof; every public request must obtain a fresh check.
  class CheckedBatch {
    friend class Engine;
    const Engine *owner_;
    const planner::Plan *plan_;
    const Batch *batch_;
    Deadline started_;
    CheckedBatch(const Engine *owner, const planner::Plan &plan, const Batch &batch, Deadline started)
        : owner_(owner), plan_(&plan), batch_(&batch), started_(started) {}
  public:
    CheckedBatch(const CheckedBatch &) = delete;
    CheckedBatch &operator=(const CheckedBatch &) = delete;
    CheckedBatch(CheckedBatch &&other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), plan_(other.plan_), batch_(other.batch_), started_(other.started_) {}
    CheckedBatch &operator=(CheckedBatch &&) = delete;
  };
  Engine(std::size_t cpu, planner::Config config, std::string worker_path);
  ~Engine();
  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;
  std::shared_ptr<planner::Plan>
  plan(std::shared_ptr<compiler::CompiledGraph> graph, const Batch &batch,
       std::optional<std::size_t> memory_budget = {}, bool hard_stop = false,
       bool async_io = false) const;
  ExecutionAudit execute(const planner::Plan &plan, const Batch &batch,
                         void *output,
                         std::optional<double> timeout_seconds = {},
                         bool return_shared_output = false,
                         bool allow_workspace_loan = false);
  CheckedBatch check(const planner::Plan &, const Batch &) const;
  ExecutionAudit execute(CheckedBatch &&, void *output,
                         std::optional<double> timeout_seconds = {},
                         bool return_shared_output = false,
                         bool allow_workspace_loan = false);
  void close();
  bool closed() const noexcept { return closed_.load(); }
  std::size_t cpu() const noexcept { return cpu_limit_; }
  std::size_t peak_active() const { return budget_.peak_active(); }
  const planner::Config &config() const noexcept { return config_; }

private:
  ProcessPool &processes();
  planner::Config config_;
  const std::size_t cpu_limit_;
  CpuBudget budget_;
  std::string worker_path_;
  std::mutex process_mutex_, close_mutex_;
  std::unique_ptr<ProcessPool> processes_;
  std::atomic<bool> closed_{false};
};
} // namespace calmetrics_engine::native
