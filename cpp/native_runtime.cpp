#include "calmetrics_engine/native_runtime.hpp"
#include "calmetrics_engine/pointwise.hpp"
#include "calmetrics_engine/native_process.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace calmetrics_engine::native {
namespace {
struct Lease {
  CpuBudget &budget;
  std::size_t count;
  Lease(CpuBudget &b, std::size_t n, Deadline deadline,
        const char *timeout_message)
      : budget(b), count(n) {
    if (!budget.acquire(n, deadline))
      throw Timeout(timeout_message);
  }
  ~Lease() { budget.release(count); }
};
Deadline deadline_from(std::optional<double> seconds, Deadline started) {
  if (!seconds)
    return Deadline::max();
  if (!std::isfinite(*seconds) || *seconds < 0 || *seconds > 365.0 * 86400)
    throw std::invalid_argument(
        "timeout must be finite and between zero and one year");
  return started + std::chrono::duration_cast<Clock::duration>(
                            std::chrono::duration<double>(*seconds));
}
double milliseconds(Deadline from, Deadline to) {
  return std::chrono::duration<double, std::milli>(to - from).count();
}
std::size_t output_row_offset(const graph::Program &program, const Batch &batch,
                              std::size_t row) {
  if (program.output_kind == graph::OutputKind::scalar)
    return row;
  std::size_t offset = 0;
  for (std::size_t i = 0; i < row; ++i)
    offset = planner::checked_add(
        offset, static_cast<std::size_t>(batch.ends[i] - batch.starts[i]));
  return offset;
}
} // namespace
std::vector<std::size_t> Batch::input_sizes() const {
  std::vector<std::size_t> result;
  result.reserve(inputs.size());
  for (const auto &input : inputs)
    result.push_back(input.size());
  return result;
}
Engine::Engine(std::size_t cpu, planner::Config config, std::string worker_path)
    : config_(config), cpu_limit_(cpu), budget_(cpu), worker_path_(std::move(worker_path)) {
  config_.validate();
  NativeScheduler::instance().ensure_capacity(cpu);
}
Engine::~Engine() { close(); }
ProcessPool &Engine::processes() {
  std::lock_guard<std::mutex> lock(process_mutex_);
  if (!processes_)
    processes_ = std::make_unique<ProcessPool>(
        worker_path_, std::min(budget_.total(), config_.max_processes));
  return *processes_;
}
std::shared_ptr<planner::Plan>
Engine::plan(std::shared_ptr<compiler::CompiledGraph> graph, const Batch &batch,
             std::optional<std::size_t> memory_budget, bool hard_stop,
             bool async_io) const {
  if (closed())
    throw std::runtime_error("native engine is closed");
  auto result = planner::make_plan(std::move(graph), config_, batch.input_sizes(),
                            batch.starts, batch.ends, batch.rows,
                            batch.product_ids, budget_.total(), memory_budget,
                            hard_stop, async_io, !batch.shared_inputs.empty(), &batch.inputs, batch.owned_input_bytes);
  result->model_identity = batch.model_identity;
  return result;
}
ExecutionAudit Engine::execute(const planner::Plan &p, const Batch &batch,
                               void *output,
                               std::optional<double> timeout_seconds,
                               bool return_shared_output, bool allow_workspace_loan) {
  // Preserve admission error precedence for the ordinary native entry point.
  if (timeout_seconds) (void)deadline_from(timeout_seconds, Deadline{});
  return execute(check(p, batch), output, timeout_seconds, return_shared_output, allow_workspace_loan);
}
Engine::CheckedBatch Engine::check(const planner::Plan &p, const Batch &batch) const {
  const auto started = Clock::now();
  if (closed()) throw std::runtime_error("native engine is closed");
  if (p.input_sizes.size() != batch.inputs.size())
    throw std::invalid_argument("stale execution plan input shape");
  for (std::size_t i = 0; i < batch.inputs.size(); ++i)
    if (p.input_sizes[i] != batch.inputs[i].size())
      throw std::invalid_argument("stale execution plan input shape");
  // Geometry reads the actual views. The size comparison above avoids allocating
  // a second vector merely to compare it with this immutable plan metadata.
  planner::validate_plan(p, p.input_sizes, batch.starts, batch.ends,
                        batch.rows, batch.product_ids, cpu(), &batch.inputs);
  return CheckedBatch(this, p, batch, started);
}
ExecutionAudit Engine::execute(CheckedBatch &&checked, void *output,
                               std::optional<double> timeout_seconds,
                               bool return_shared_output, bool allow_workspace_loan) {
  if (std::exchange(checked.owner_, nullptr) != this)
    throw std::invalid_argument("invalid or consumed execution check");
  const auto &p = *checked.plan_;
  const auto &batch = *checked.batch_;
  const auto started = checked.started_;
  const auto deadline = deadline_from(timeout_seconds, started);
  if (closed())
    throw std::runtime_error("native engine is closed");
  if (p.model_identity != batch.model_identity)
    throw std::invalid_argument("model payload does not match plan identity");
  if (batch.parameter_count != p.graph->program.parameter_count)
    throw std::invalid_argument("parameter count does not match graph");
  if (return_shared_output && (p.lane != "process" || !p.use_shared_memory))
    throw std::invalid_argument(
        "shared output requires the shared process lane");
  if (batch.rows && !output && !return_shared_output)
    throw std::invalid_argument("missing output buffer");
  const auto count = p.lane == "process" ? p.process_count : p.thread_count;
  Lease local_lease(budget_, count, deadline,
                    "native engine CPU admission deadline exceeded");
  auto &shared_scheduler = NativeScheduler::instance();
  // Construction already admits this engine's immutable limit. Global capacity
  // only grows; acquiring both leases below still enforces current CPU usage.
  if (count > 1)
    shared_scheduler.ensure_threads(count);
  Lease global_lease(shared_scheduler.budget(), count, deadline,
                     "global native CPU admission deadline exceeded");
  const auto admitted = Clock::now();
  ExecutionAudit audit;
  audit.queue_wait_ms = milliseconds(started, admitted);
  audit.cpu_tokens = count;
  audit.cpu_budget = cpu();
  if (p.lane == "single" || (p.lane == "thread" && count == 1)) {
    audit.chunks.push_back(
        graph::execute(p.graph->program, batch.inputs, batch.parameters,
                       batch.parameter_count, batch.starts, batch.ends,
                       batch.rows, output, p.graph->program.roots.size(), p.result_layout.get(), 0, deadline, nullptr, allow_workspace_loan, p.expanded_matmul_roots, p.row_subviews));
    check_deadline(deadline);
    audit.native_threads = 1;
  } else {
    const bool process = p.lane == "process";
    if (!process && p.lane != "thread")
      throw std::invalid_argument("invalid native execution lane");
    std::unique_ptr<ProcessTransport> transport;
    ProcessPool *process_pool = nullptr;
    if (process) {
      transport = std::make_unique<ProcessTransport>(p, batch);
      process_pool = &processes();
      audit.shared_memory_bytes = transport->shared_memory_bytes;
      audit.boundary_copy_bytes = transport->boundary_copy_bytes;
      audit.output_copy_bytes = transport->output_copy_bytes;
      audit.native_processes = p.process_count;
    } else
      audit.native_threads = count;
    auto execute_tasks = [&](std::size_t task_count, auto &&task_function) {
      audit.chunks.resize(task_count);
      std::atomic<bool> cancelled{false};
      auto run_task = [&](std::size_t i) {
        if (cancelled.load())
          throw Timeout("native batch cancelled");
        check_deadline(deadline);
        auto result = task_function(i, cancelled);
        check_deadline(deadline);
        return result;
      };
      std::exception_ptr failure;
      // A branch plan can contain more tasks than the memory-adjusted CPU
      // lease. Execute in bounded waves so this request never runs more native
      // tasks concurrently than the tokens it actually owns.
      for (std::size_t base = 0; base < task_count && !failure; base += count) {
        const auto end = std::min(task_count, base + count);
        std::vector<std::future<graph::Audit>> futures;
        futures.reserve(end > base ? end - base - 1 : 0);
        try {
          for (std::size_t i = base + 1; i < end; ++i)
            futures.push_back(shared_scheduler.threads().submit(
                [&, i] { return run_task(i); }));
          audit.chunks[base] = run_task(base);
        } catch (...) {
          failure = std::current_exception();
          cancelled.store(true);
        }
        // Never release buffers/tokens/mappings while submitted work can still
        // use them.
        for (std::size_t i = 0; i < futures.size(); ++i) {
          try {
            audit.chunks[base + i + 1] = futures[i].get();
          } catch (...) {
            if (!failure)
              failure = std::current_exception();
            cancelled.store(true);
          }
        }
      }
      if (failure)
        std::rethrow_exception(failure);
    };

    if (!process && p.parallel_dimension == "tensor") {
      execute_tasks(p.chunks.size(), [&](std::size_t i, const std::atomic<bool> &cancelled) {
        graph::Audit result;
        const auto chunk = p.chunks[i];
        if (!graph::execute_pointwise(p.graph->program, batch.inputs, batch.parameters,
            batch.parameter_count, batch.starts, batch.ends, batch.rows, output,
            p.result_layout.get(), 0, result, deadline, &cancelled, false, chunk.begin, chunk.end,
            nullptr, nullptr, nullptr, nullptr, i == 0))
          throw std::invalid_argument("tensor plan no longer matches pointwise geometry");
        return result;
      });
    } else if (!process && p.parallel_dimension == "array_row") {
      struct Failure { std::size_t node = std::numeric_limits<std::size_t>::max(); std::string text; };
      std::vector<Failure> failures(p.chunks.size());
      execute_tasks(p.chunks.size(), [&](std::size_t i, const std::atomic<bool> &cancelled) {
        const auto chunk = p.chunks[i];
        std::size_t node = std::numeric_limits<std::size_t>::max();
        try {
          return graph::execute_row_batch_range(p.graph->program, batch.inputs,
              batch.parameters, batch.parameter_count, batch.starts, batch.ends,
              output, *p.result_layout, chunk.begin, chunk.end, i == 0, node,
              deadline, &cancelled, allow_workspace_loan, p.expanded_matmul_roots);
        } catch (const ops::Error &error) {
          if (node == std::numeric_limits<std::size_t>::max()) throw;
          // Numerical failures do not cancel later ranges: a later row can
          // fail at an earlier logical node. Resource/timeout errors still use
          // execute_tasks' existing cancellation and drain protocol.
          failures[i].node = node; failures[i].text = error.what();
          return graph::Audit{};
        }
      });
      const auto first = std::min_element(failures.begin(), failures.end(),
          [](const Failure &a, const Failure &b) { return a.node < b.node; });
      if (first != failures.end() && first->node != std::numeric_limits<std::size_t>::max())
        throw ops::Error(first->text.c_str());
    } else if (!process && p.parallel_dimension == "dag_branch") {
      execute_tasks(p.branch_tasks.size(), [&](std::size_t i,
                                               const std::atomic<bool> &cancelled) {
        const auto task = p.branch_tasks[i];
        if (task.branch_index >= p.graph->branches.size() ||
            task.row >= batch.rows)
          throw std::invalid_argument("invalid DAG branch task");
        const auto &branch = p.graph->branches[task.branch_index];
        std::vector<double> local(branch.root_indices.size());
        auto result = graph::execute(
            branch.program, batch.inputs, batch.parameters,
            batch.parameter_count, batch.starts + task.row,
            batch.ends + task.row, 1, local.data(), local.size(), nullptr, 0, deadline, &cancelled, false, p.expanded_matmul_roots, false);
        const auto columns = p.graph->program.roots.size();
        for (std::size_t root = 0; root < branch.root_indices.size(); ++root)
          static_cast<double *>(output)[task.row * columns + branch.root_indices[root]] = local[root];
        return result;
      });
    } else {
      execute_tasks(p.chunks.size(), [&](std::size_t i,
                                         const std::atomic<bool> &cancelled) {
        const auto chunk = p.chunks[i];
        if (process) {
          auto request = transport->request(chunk);
          auto response =
              process_pool->transact(request, deadline, p.hard_stop, cancelled);
          return transport->response(response, chunk, output);
        }
        return graph::execute(p.graph->program, batch.inputs, batch.parameters,
                              batch.parameter_count, batch.starts + chunk.begin,
                              batch.ends + chunk.begin, chunk.end - chunk.begin,
                              p.result_layout ? output : graph::output_offset(output, output_row_offset(
                                           p.graph->program, batch, chunk.begin) *
                                           p.graph->program.roots.size(), p.graph->program.output_dtype),
                              p.graph->program.roots.size(), p.result_layout.get(), chunk.begin, deadline, &cancelled, allow_workspace_loan, p.expanded_matmul_roots, p.row_subviews);
      });
    }
    if (transport) {
      if (return_shared_output) {
        audit.output_owner = transport->output_owner();
        audit.output_copy_bytes = 0;
      } else
        transport->finish(output);
    }
  }
  const auto completed = Clock::now();
  audit.compute_ms = milliseconds(admitted, completed);
  audit.elapsed_ms = milliseconds(started, completed);
  return audit;
}
void Engine::close() {
  std::lock_guard<std::mutex> close_lock(close_mutex_);
  if (closed_.exchange(true))
    return;
  budget_.close_admission();
  budget_.wait_idle();
  if (processes_)
    processes_->close();
}
} // namespace calmetrics_engine::native
