#include "calmetrics_engine/native_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>

namespace n = calmetrics_engine::native;
namespace c = calmetrics_engine::compiler;
namespace p = calmetrics_engine::planner;
namespace o = calmetrics_engine::ops;
void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  try {
    auto readonly_region = n::SharedRegion::create(sizeof(double));
    *static_cast<double *>(readonly_region->data()) = 42.0;
    readonly_region->make_readonly();
    require(!readonly_region->writable(), "creator shared input readonly");
    auto readonly_attached =
        n::SharedRegion::attach(readonly_region->name(), sizeof(double));
    require(!readonly_attached->writable() &&
                *static_cast<const double *>(readonly_attached->data()) == 42.0,
            "attached shared input readonly");

    auto graph =
        c::compile({"mean(x)", "std(x,1)", "median(x)", "quantile(x,.25)"},
                   {{"x", "series"}});
    std::vector<double> x(800);
    std::iota(x.begin(), x.end(), 1.0);
    std::vector<std::int64_t> starts{0, 25, 100, 125, 200, 225, 300, 325},
        ends{100, 100, 200, 200, 300, 300, 400, 400};
    std::vector<std::int64_t> products{0, 0, 1, 1, 2, 2, 3, 3};
    n::Batch batch;
    o::Value input;
    input.shape = o::vector_shape(x.size());
    input.data = x.data();
    batch.inputs = {input};
    batch.starts = starts.data();
    batch.ends = ends.data();
    batch.product_ids = products.data();
    batch.rows = starts.size();
    p::Config single;
    single.thread_work_units = 1e99;
    single.process_work_units = 1e100;
    n::Engine serial(4, single, argv[1]);
    auto serial_plan = serial.plan(graph, batch);
    std::vector<double> reference(starts.size() * 4), actual(reference.size());
    serial.execute(*serial_plan, batch, reference.data());
    require(reference[0] == 50.5, "serial mean");
    p::Config config;
    config.thread_work_units = 1;
    config.process_work_units = 1e100;
    config.min_rows_per_worker = 1;
    n::Engine threaded(4, config, argv[1]);
    auto thread_plan = threaded.plan(graph, batch);
    require(thread_plan->lane == "thread", "thread plan");
    for (int repeat = 0; repeat < 3; ++repeat) {
      auto audit = threaded.execute(*thread_plan, batch, actual.data());
      require(actual == reference && audit.native_threads == 4,
              "native thread parity");
      require(n::NativeScheduler::instance().budget().peak_active() >= 4,
              "graph runtime must use global scheduler CPU admission");
    }
    require(threaded.peak_active() <= 4, "CPU budget");
    config.process_work_units = 1;
    config.shared_memory_threshold_bytes = 1;
    n::Engine processes(4, config, argv[1]);
    auto process_plan = processes.plan(graph, batch);
    require(process_plan->lane == "process" && process_plan->use_shared_memory,
            "process plan");
    for (int repeat = 0; repeat < 2; ++repeat) {
      auto audit = processes.execute(*process_plan, batch, actual.data(), 10.0);
      require(actual == reference && audit.native_processes == 4,
              "native process parity");
      require(audit.shared_memory_bytes >= x.size() * 8, "shared transport");
    }
    auto shared_output =
        processes.execute(*process_plan, batch, nullptr, 10.0, true);
    require(shared_output.output_owner && shared_output.output_copy_bytes == 0,
            "native-owned shared output without return copy");
    const auto *shared_values =
        static_cast<const double *>(shared_output.output_owner->data());
    require(std::equal(reference.begin(), reference.end(), shared_values),
            "zero-copy output parity");
    auto isolation = processes.plan(graph, batch, {}, true);
    bool expired = false;
    try {
      processes.execute(*isolation, batch, actual.data(), 0.000001);
    } catch (const n::Timeout &) {
      expired = true;
    }
    require(expired, "hard deadline");
    processes.execute(*process_plan, batch, actual.data(), 10.0);
    require(actual == reference, "recovery after timeout");
    config.shared_memory_threshold_bytes = 1ull << 30;
    n::Engine inline_process(2, config, argv[1]);
    auto inline_plan = inline_process.plan(graph, batch);
    require(!inline_plan->use_shared_memory, "inline process lane");
    inline_process.execute(*inline_plan, batch, actual.data(), 10.0);
    require(actual == reference, "inline process parity");
    auto memory = n::SharedRegion::create(x.size() * 8);
    std::memcpy(memory->data(), x.data(), x.size() * 8);
    batch.inputs[0].data = memory->data();
    batch.shared_inputs = {{memory->name(), memory->size(), 0}};
    batch.shared_owners = {memory};
    auto shared_plan = processes.plan(graph, batch);
    auto audit = processes.execute(*shared_plan, batch, actual.data(), 10.0);
    require(actual == reference &&
                audit.boundary_copy_bytes == starts.size() * 16,
            "reusable shared input");
    starts[0] = 1;
    bool stale = false;
    try {
      processes.execute(*shared_plan, batch, actual.data());
    } catch (const std::invalid_argument &) {
      stale = true;
    }
    require(stale, "stale plan");
    starts[0] = 0;

    // A single heavy row with independent root branches should fork/join in
    // the native thread pool without duplicating shared operation prefixes.
    auto branch_graph =
        c::compile({"mean(x)", "median(x)", "-min_value(drawdown_series(x))",
                    "linear_r_squared(x)"},
                   {{"x", "series"}});
    std::vector<double> branch_values(200000);
    for (std::size_t i = 0; i < branch_values.size(); ++i)
      branch_values[i] = 1.0 + static_cast<double>(i) / branch_values.size();
    std::int64_t branch_start = 0,
                 branch_end = static_cast<std::int64_t>(branch_values.size());
    n::Batch branch_batch;
    o::Value branch_input;
    branch_input.shape = o::vector_shape(branch_values.size());
    branch_input.data = branch_values.data();
    branch_batch.inputs = {branch_input};
    branch_batch.starts = &branch_start;
    branch_batch.ends = &branch_end;
    branch_batch.rows = 1;
    p::Config branch_serial_config;
    branch_serial_config.thread_work_units = 1e99;
    branch_serial_config.dag_branch_work_units = 1e99;
    branch_serial_config.process_work_units = 1e100;
    n::Engine branch_serial(4, branch_serial_config, argv[1]);
    auto branch_serial_plan = branch_serial.plan(branch_graph, branch_batch);
    std::vector<double> branch_reference(4), branch_actual(4);
    branch_serial.execute(*branch_serial_plan, branch_batch,
                          branch_reference.data());

    p::Config branch_config;
    branch_config.process_work_units = 1e100;
    branch_config.dag_branch_work_units = 1000000;
    n::Engine branch_engine(4, branch_config, argv[1]);
    auto branch_plan = branch_engine.plan(branch_graph, branch_batch);
    require(branch_plan->lane == "thread" &&
                branch_plan->parallel_dimension == "dag_branch" &&
                branch_plan->branch_tasks.size() >= 2,
            "DAG branch plan");
    auto branch_audit =
        branch_engine.execute(*branch_plan, branch_batch, branch_actual.data());
    require(branch_actual == branch_reference &&
                branch_audit.native_threads == branch_plan->thread_count &&
                branch_engine.peak_active() <= 4,
            "DAG branch parity/CPU budget");

    threaded.close();
    bool closed = false;
    try {
      threaded.execute(*thread_plan, batch, actual.data());
    } catch (const std::exception &) {
      closed = true;
    }
    require(closed, "close admission");
    std::cout << "native compiler/planner/threads/processes/shared-memory "
                 "tests passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
