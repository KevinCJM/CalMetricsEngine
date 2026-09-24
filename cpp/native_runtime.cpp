#include "calmetrics_engine/native_runtime.hpp"
#include "calmetrics_engine/pointwise.hpp"
#include "calmetrics_engine/native_process.hpp"
#include <algorithm>
#include <cmath>

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
Deadline deadline_from(std::optional<double> seconds) {
  if (!seconds)
    return Deadline::max();
  if (!std::isfinite(*seconds) || *seconds < 0 || *seconds > 365.0 * 86400)
    throw std::invalid_argument(
        "timeout must be finite and between zero and one year");
  return Clock::now() + std::chrono::duration_cast<Clock::duration>(
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
    : config_(config), budget_(cpu), worker_path_(std::move(worker_path)) {
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
                               bool return_shared_output) {
  const auto started = Clock::now();
  const auto deadline = deadline_from(timeout_seconds);
  if (closed())
    throw std::runtime_error("native engine is closed");
  planner::validate_plan(p, batch.input_sizes(), batch.starts, batch.ends,
                         batch.rows, batch.product_ids, cpu(), &batch.inputs);
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
  shared_scheduler.ensure_capacity(cpu());
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
                       batch.rows, output, p.graph->program.roots.size(), p.result_layout.get(), 0, deadline));
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
            p.result_layout.get(), 0, result, deadline, &cancelled, false, chunk.begin, chunk.end))
          throw std::invalid_argument("tensor plan no longer matches pointwise geometry");
        // Each disjoint task computes the same shape contract; expose it once.
        if (i) { result.result_shapes.clear(); result.root_statuses.clear(); }
        return result;
      });
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
            batch.ends + task.row, 1, local.data(), local.size(), nullptr, 0, deadline, &cancelled);
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
                              p.graph->program.roots.size(), p.result_layout.get(), chunk.begin, deadline, &cancelled);
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
