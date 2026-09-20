#include "calmetrics_engine/planner.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace calmetrics_engine::planner {
namespace {
void require(bool ok, const char *message) {
  if (!ok)
    throw std::invalid_argument(message);
}
void mix(std::uint64_t &hash, std::uint64_t value) {
  hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
}
double row_work(const compiler::PhysicalCost &cost, std::size_t observations) {
  const auto n = static_cast<double>(observations);
  const auto sort =
      n *
      std::log2(static_cast<double>(std::max<std::size_t>(observations, 2)));
  return cost.constant_per_row + cost.linear_per_observation * n +
         cost.sort_nlogn * sort;
}
double geometry_work(const compiler::PhysicalCost &cost, const Geometry &g) {
  return cost.constant_per_row * static_cast<double>(g.rows) +
         cost.linear_per_observation * static_cast<double>(g.observations) +
         cost.sort_nlogn * g.sort_work;
}
std::size_t scratch_estimate(const graph::Program &program,
                             std::size_t window) {
  std::size_t estimate =
      checked_mul(window, checked_add(checked_mul(program.numeric_slots, 8),
                                      program.mask_slots));
  estimate = checked_add(
      estimate, checked_mul(program.nodes.size(), sizeof(ops::Value) + 160));
  std::unordered_set<std::uint32_t> order_sources;
  bool operator_scratch = false;
  for (const auto &node : program.nodes) {
    if (node.kind != graph::NodeKind::operation)
      continue;
    const auto op = ops::lookup(node.opcode).op;
    if (op == ops::Op::median || op == ops::Op::quantile)
      order_sources.insert(node.parents[0]);
    if (op == ops::Op::median || op == ops::Op::quantile ||
        op == ops::Op::median_where || op == ops::Op::quantile_where ||
        op == ops::Op::rolling_min || op == ops::Op::rolling_max)
      operator_scratch = true;
  }
  estimate = checked_add(
      estimate, checked_mul(checked_mul(order_sources.size(), window), 8));
  if (operator_scratch)
    estimate = checked_add(estimate, checked_mul(window, 8));
  return estimate;
}
std::size_t total_memory(const Plan &p, std::size_t workers) {
  auto bytes = checked_add(p.estimated_input_bytes, p.estimated_output_bytes);
  bytes = checked_add(bytes, checked_mul(p.row_count, 16));
  bytes = checked_add(bytes,
                      checked_mul(workers, p.estimated_worker_scratch_bytes));
  if (p.lane == "process") {
    bytes = checked_add(bytes, p.estimated_output_bytes);
    if (p.use_shared_memory) {
      if (!p.inputs_already_shared)
        bytes = checked_add(bytes, p.estimated_input_bytes);
      bytes = checked_add(bytes, checked_mul(p.row_count, 16));
    } else {
      bytes = checked_add(
          bytes, checked_mul(p.estimated_input_bytes, checked_mul(workers, 2)));
    }
  }
  return bytes;
}
} // namespace

std::size_t checked_add(std::size_t a, std::size_t b) {
  if (a > static_cast<std::size_t>(PTRDIFF_MAX) - b ||
      b > static_cast<std::size_t>(PTRDIFF_MAX))
    throw std::overflow_error("execution size overflow");
  return a + b;
}
std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (b && a > static_cast<std::size_t>(PTRDIFF_MAX) / b)
    throw std::overflow_error("execution size overflow");
  return a * b;
}
void Config::validate() const {
  require(std::isfinite(thread_work_units) && thread_work_units > 0 &&
              std::isfinite(dag_branch_work_units) &&
              dag_branch_work_units > 0 && std::isfinite(process_work_units) &&
              process_work_units > 0,
          "work thresholds must be finite and positive");
  require(min_rows_per_worker > 0 && max_processes > 0 &&
              max_processes <= 1024 && max_async_jobs > 0 &&
              max_async_jobs <= 1024,
          "invalid planner worker limits");
}

Geometry inspect(const std::vector<std::size_t> &sizes,
                 const std::int64_t *starts, const std::int64_t *ends,
                 std::size_t rows, const std::int64_t *product_ids,
                 bool build_groups) {
  require(rows == 0 || (starts && ends), "missing interval arrays");
  for (auto size : sizes)
    require(size == sizes[0],
            "graph inputs must share an aligned observation axis");
  Geometry g;
  g.rows = rows;
  g.signature = 0x434d454e41544956ull;
  mix(g.signature, rows);
  mix(g.signature, sizes.size());
  mix(g.signature, product_ids != nullptr);
  for (auto size : sizes)
    mix(g.signature, size);
  const auto available = sizes.empty() ? 0 : sizes[0];
  std::size_t group_start = 0, group_weight = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    require(starts[row] >= 0 && ends[row] >= starts[row] &&
                static_cast<std::size_t>(ends[row]) <= available,
            "intervals must satisfy 0 <= start <= end <= input length");
    if (product_ids)
      require(product_ids[row] >= 0 &&
                  (row == 0 || product_ids[row] >= product_ids[row - 1]),
              "product_ids must be nonnegative and nondecreasing");
    const auto n = static_cast<std::size_t>(ends[row] - starts[row]);
    g.row_lengths.push_back(n);
    g.observations = checked_add(g.observations, n);
    g.max_window = std::max(g.max_window, n);
    g.sort_work += static_cast<double>(n) *
                   std::log2(static_cast<double>(std::max<std::size_t>(n, 2)));
    mix(g.signature, static_cast<std::uint64_t>(starts[row]));
    mix(g.signature, static_cast<std::uint64_t>(ends[row]));
    if (product_ids)
      mix(g.signature, static_cast<std::uint64_t>(product_ids[row]));
    const bool new_group =
        row > 0 && (!product_ids || product_ids[row] != product_ids[row - 1]);
    if (new_group) {
      ++g.products;
      g.max_intervals = std::max(g.max_intervals, row - group_start);
      if (build_groups) {
        g.groups.push_back({group_start, row});
        g.weights.push_back(group_weight);
      }
      group_start = row;
      group_weight = 0;
    }
    group_weight = checked_add(group_weight, n);
  }
  if (rows) {
    ++g.products;
    g.max_intervals = std::max(g.max_intervals, rows - group_start);
    if (build_groups) {
      g.groups.push_back({group_start, rows});
      g.weights.push_back(group_weight);
    }
  }
  return g;
}
std::vector<Chunk> partition(const Geometry &g, std::size_t workers,
                             bool by_product,
                             const compiler::PhysicalCost *physical_cost) {
  if (!g.rows)
    return {};
  const auto units = by_product ? g.groups.size() : g.rows;
  workers = std::max<std::size_t>(1, std::min(workers, units));

  std::vector<long double> weights(units, 0.0L);
  if (by_product) {
    for (std::size_t group = 0; group < g.groups.size(); ++group) {
      const auto chunk = g.groups[group];
      for (std::size_t row = chunk.begin; row < chunk.end; ++row)
        weights[group] +=
            physical_cost ? static_cast<long double>(
                                row_work(*physical_cost, g.row_lengths[row]))
                          : static_cast<long double>(
                                g.row_lengths[row] ? g.row_lengths[row] : 1);
    }
  } else {
    for (std::size_t row = 0; row < g.rows; ++row)
      weights[row] = physical_cost
                         ? static_cast<long double>(
                               row_work(*physical_cost, g.row_lengths[row]))
                         : static_cast<long double>(
                               g.row_lengths[row] ? g.row_lengths[row] : 1);
  }

  std::vector<long double> cumulative;
  cumulative.reserve(units);
  long double total = 0.0L;
  for (auto weight : weights) {
    total += weight > 0 ? weight : 1.0L;
    cumulative.push_back(total);
  }

  std::vector<Chunk> result;
  std::size_t previous = 0;
  for (std::size_t worker = 1; worker < workers; ++worker) {
    const auto target = total * static_cast<long double>(worker) /
                        static_cast<long double>(workers);
    auto cut =
        static_cast<std::size_t>(
            std::lower_bound(cumulative.begin(), cumulative.end(), target) -
            cumulative.begin()) +
        1;
    cut = std::max(cut, previous + 1);
    cut = std::min(cut, units - (workers - worker));
    if (by_product)
      result.push_back({g.groups[previous].begin, g.groups[cut - 1].end});
    else
      result.push_back({previous, cut});
    previous = cut;
  }
  if (by_product)
    result.push_back({g.groups[previous].begin, g.groups.back().end});
  else
    result.push_back({previous, g.rows});
  return result;
}

std::shared_ptr<Plan>
make_plan(std::shared_ptr<compiler::CompiledGraph> graph, const Config &config,
          const std::vector<std::size_t> &sizes, const std::int64_t *starts,
          const std::int64_t *ends, std::size_t rows,
          const std::int64_t *product_ids, std::size_t cpu,
          std::optional<std::size_t> memory_budget, bool hard_stop,
          bool async_io, bool already_shared) {
  config.validate();
  require(cpu > 0 && cpu <= 1024, "cpu_budget must be in 1..1024");
  require(!memory_budget || *memory_budget > 0,
          "memory budget must be positive");
  require(graph && sizes.size() == graph->program.input_count,
          "graph input count mismatch");
  const auto geometry = inspect(sizes, starts, ends, rows, product_ids);
  auto p = std::make_shared<Plan>();
  p->graph = std::move(graph);
  p->cpu_budget = cpu;
  p->memory_budget_bytes = memory_budget;
  p->row_count = rows;
  p->max_window = geometry.max_window;
  p->interval_observations = geometry.observations;
  p->product_count = geometry.products;
  p->max_intervals_per_product = geometry.max_intervals;
  p->geometry_signature = geometry.signature;
  p->input_sizes = sizes;
  p->inputs_already_shared = already_shared;
  p->async_orchestration = async_io;
  p->hard_stop = hard_stop;
  p->parallel_dimension =
      product_ids &&
              geometry.products >= std::min(cpu, std::max<std::size_t>(1, rows))
          ? "product"
          : "interval";
  for (auto size : sizes)
    p->estimated_input_bytes =
        checked_add(p->estimated_input_bytes, checked_mul(size, 8));
  p->estimated_output_bytes =
      checked_mul(checked_mul(rows, p->graph->program.roots.size()), 8);
  p->estimated_worker_scratch_bytes =
      scratch_estimate(p->graph->program, geometry.max_window);
  for (const auto &node : p->graph->nodes) {
    if (node.node.kind != graph::NodeKind::operation)
      continue;
    double work = 0;
    if (node.cost_model == "constant")
      work = static_cast<double>(rows);
    else if (node.cost_model == "sort")
      work = 2 * geometry.sort_work;
    else {
      double factor = 1;
      if (node.cost_model == "elementwise")
        factor = node.simd_eligible ? .65 : 1;
      else if (node.cost_model == "sequence")
        factor = 1.2;
      else if (node.cost_model == "reduction" || node.cost_model == "composite")
        factor = 1.5;
      else if (node.cost_model == "rolling" || node.cost_model == "matrix")
        factor = 2;
      else if (node.cost_model == "regression")
        factor = 3.5;
      work = factor * static_cast<double>(geometry.observations);
    }
    p->estimated_logical_work_units += work;
    if (node.simd_eligible && geometry.max_window >= config.simd_min_elements)
      p->simd_nodes.push_back(node.node_id);
  }
  p->estimated_work_units = geometry_work(p->graph->physical_cost, geometry);
  const bool process =
      hard_stop ||
      (rows >= std::max<std::size_t>(2, config.min_rows_per_worker) &&
       (p->estimated_work_units >= config.process_work_units ||
        p->estimated_input_bytes >= config.process_input_threshold_bytes));

  std::size_t heavy_branches = 0;
  for (const auto &branch : p->graph->branches)
    heavy_branches +=
        geometry_work(branch.cost, geometry) >= config.thread_work_units;
  const auto branch_task_count = checked_mul(rows, p->graph->branches.size());
  const bool dag_branch =
      !process && cpu > 1 && rows > 0 && rows < cpu &&
      p->graph->branches.size() > 1 && heavy_branches >= 2 &&
      p->estimated_work_units >= config.dag_branch_work_units &&
      branch_task_count > rows;

  if (process) {
    p->lane = "process";
    p->reason_codes.push_back(hard_stop ? "hard_stop_requires_process_isolation"
                              : p->estimated_input_bytes >=
                                      config.process_input_threshold_bytes
                                  ? "input_exceeds_process_isolation_threshold"
                                  : "work_exceeds_process_threshold");
    p->process_count =
        std::min({cpu, config.max_processes,
                  std::max<std::size_t>(1, rows / config.min_rows_per_worker)});
  } else if (dag_branch) {
    p->lane = "thread";
    p->parallel_dimension = "dag_branch";
    p->thread_count = std::min(cpu, branch_task_count);
    p->estimated_worker_scratch_bytes = 0;
    for (const auto &branch : p->graph->branches) {
      const auto branch_output =
          checked_mul(branch.root_indices.size(), sizeof(double));
      p->estimated_worker_scratch_bytes = std::max(
          p->estimated_worker_scratch_bytes,
          checked_add(scratch_estimate(branch.program, geometry.max_window),
                      branch_output));
    }
    p->reason_codes.push_back("heavy_independent_dag_branches");
  } else if (p->estimated_work_units >= config.thread_work_units && rows >= 2 &&
             cpu > 1) {
    p->lane = "thread";
    p->thread_count = std::min(cpu, rows);
    p->reason_codes.push_back("work_exceeds_thread_threshold");
  } else
    p->reason_codes.push_back("parallel_overhead_expected_to_dominate");

  if (p->lane != "single")
    p->reason_codes.push_back("parallelize_by_" + p->parallel_dimension);
  p->use_shared_memory = process && (hard_stop || already_shared ||
                                     p->estimated_input_bytes >=
                                         config.shared_memory_threshold_bytes);
  if (p->use_shared_memory)
    p->reason_codes.push_back(
        hard_stop        ? "hard_stop_uses_shared_process_transport"
        : already_shared ? "inputs_already_shared"
                         : "process_input_exceeds_shared_memory_threshold");

  auto workers = process ? p->process_count : p->thread_count;
  p->estimated_total_memory_bytes = total_memory(*p, workers);
  bool reduced = false;
  while (memory_budget && p->estimated_total_memory_bytes > *memory_budget &&
         workers > 1) {
    --workers;
    reduced = true;
    p->estimated_total_memory_bytes = total_memory(*p, workers);
  }
  if (memory_budget && p->estimated_total_memory_bytes > *memory_budget)
    throw std::bad_alloc();
  if (process)
    p->process_count = workers;
  else
    p->thread_count = workers;
  if (reduced)
    p->reason_codes.push_back("memory_budget_reduced_parallelism");
  if (!p->simd_nodes.empty())
    p->reason_codes.push_back(
        "eligible_nodes_use_native_runtime_simd_dispatch");
  if (async_io)
    p->reason_codes.push_back("coroutine_orchestration_for_async_boundary");

  if (p->parallel_dimension == "dag_branch") {
    struct WeightedTask {
      BranchTask task;
      double work = 0;
    };
    std::vector<WeightedTask> tasks;
    tasks.reserve(branch_task_count);
    for (std::size_t row = 0; row < rows; ++row)
      for (std::uint32_t branch = 0; branch < p->graph->branches.size();
           ++branch)
        tasks.push_back({{row, branch},
                         row_work(p->graph->branches[branch].cost,
                                  geometry.row_lengths[row])});
    std::stable_sort(tasks.begin(), tasks.end(),
                     [](const WeightedTask &a, const WeightedTask &b) {
                       return a.work > b.work;
                     });
    for (const auto &task : tasks)
      p->branch_tasks.push_back(task.task);
  } else {
    p->chunks = partition(geometry, workers, p->parallel_dimension == "product",
                          &p->graph->physical_cost);
  }
  return p;
}
void validate_plan(const Plan &p, const std::vector<std::size_t> &sizes,
                   const std::int64_t *starts, const std::int64_t *ends,
                   std::size_t rows, const std::int64_t *product_ids,
                   std::size_t engine_cpu) {
  require(p.cpu_budget <= engine_cpu,
          "execution plan exceeds this engine CPU budget");
  require(p.input_sizes == sizes && p.row_count == rows,
          "stale execution plan input shape");
  const auto geometry = inspect(sizes, starts, ends, rows, product_ids, false);
  require(geometry.signature == p.geometry_signature,
          "stale execution plan interval/product geometry");
}
} // namespace calmetrics_engine::planner
