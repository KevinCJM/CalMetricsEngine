#include "calmetrics_engine/native_runtime.hpp"
#include "../cpp/operators/numeric_line_sort.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>

namespace n = calmetrics_engine::native;
namespace c = calmetrics_engine::compiler;
namespace p = calmetrics_engine::planner;
namespace o = calmetrics_engine::ops;
void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
namespace {
using SingleAudit = calmetrics_engine::InlineVector<calmetrics_engine::graph::Audit, 1>;
static_assert(sizeof(SingleAudit) ==
              sizeof(std::variant<calmetrics_engine::graph::Audit,
                                  std::vector<calmetrics_engine::graph::Audit>>),
              "single-chunk metadata must not store a redundant length");
static_assert(std::is_nothrow_move_constructible_v<SingleAudit> &&
              std::is_nothrow_move_assignable_v<SingleAudit>);

struct ThrowingMetadata {
  inline static bool fail_default = false, fail_copy = false;
  std::shared_ptr<int> owner;
  ThrowingMetadata() {
    if (fail_default) throw std::runtime_error("metadata default construction");
  }
  explicit ThrowingMetadata(int value) : owner(std::make_shared<int>(value)) {}
  ThrowingMetadata(const ThrowingMetadata &other) : owner(other.owner) {
    if (fail_copy) throw std::runtime_error("metadata copy construction");
  }
  ThrowingMetadata &operator=(const ThrowingMetadata &) = default;
  ThrowingMetadata(ThrowingMetadata &&) noexcept = default;
  ThrowingMetadata &operator=(ThrowingMetadata &&) noexcept = default;
};

void check_single_metadata() {
  using Single = calmetrics_engine::InlineVector<std::shared_ptr<int>, 1>;
  Single values;
  require(values.empty() && values.begin() == values.end(), "empty single metadata range");
  bool rejected = false;
  try { values.at(0); } catch (const std::out_of_range &) { rejected = true; }
  require(rejected, "empty single metadata bounds");
  auto first = std::make_shared<int>(7);
  std::weak_ptr<int> first_owner = first;
  values.push_back(first); first.reset();
  values.resize(1);
  const auto &constant = values;
  require(constant.size() == 1 && *constant.at(0) == 7 &&
          constant.end() - constant.begin() == 1, "single metadata const access");
  auto copied = values;
  auto moved = std::move(values);
  require(values.empty() && *moved[0] == 7, "single metadata move empties source");
  values.resize(4);
  values[3] = std::make_shared<int>(19);
  std::weak_ptr<int> last_owner = values[3];
  Single heap_copy;
  heap_copy = values;
  require(heap_copy.size() == 4 && *heap_copy[3] == 19, "single metadata heap copy");
  auto heap_moved = std::move(values);
  require(values.empty() && *heap_moved[3] == 19, "single metadata heap move");
  Single heap_target;
  heap_target.push_back(std::make_shared<int>(17));
  std::weak_ptr<int> heap_replaced = heap_target[0];
  heap_target = std::move(heap_moved);
  require(heap_moved.empty() && heap_replaced.expired() && *heap_target[3] == 19,
          "single metadata heap move assignment replaces inline storage");
  heap_moved = std::move(heap_target);
  values.resize(2); values[0] = std::make_shared<int>(23);
  std::weak_ptr<int> replaced_owner = values[0];
  values = std::move(moved);
  require(moved.empty() && replaced_owner.expired() && *values[0] == 7,
          "single metadata move assignment releases previous owner");
  auto *same = &values;
  values = std::move(*same);
  values = *same;
  require(values.size() == 1 && *values[0] == 7, "single metadata self assignment");
  values.resize(5);
  require(*values[0] == 7 && !values[4], "single metadata growth initializes new entries");
  values[4] = std::make_shared<int>(29);
  std::weak_ptr<int> tail_owner = values[4];
  values.resize(1);
  require(tail_owner.expired() && *values[0] == 7, "single metadata shrink releases tail");
  heap_moved.resize(1); heap_copy.clear();
  require(last_owner.expired(), "single metadata heap shrink releases owners");
  copied.clear(); values.clear();
  require(first_owner.expired(), "single metadata clear releases inline owner");
  moved.push_back(std::make_shared<int>(31));
  moved.resize(3);
  require(*moved[0] == 31, "single metadata moved source can be reused");
  moved.clear(); moved.clear();
  require(moved.empty() && moved.begin() == moved.end(), "single metadata repeated clear");

  using Guarded = calmetrics_engine::InlineVector<ThrowingMetadata, 1>;
  Guarded guarded;
  ThrowingMetadata::fail_default = true;
  rejected = false;
  try { guarded.resize(1); } catch (const std::runtime_error &) { rejected = true; }
  require(rejected && guarded.empty(), "failed inline construction leaves empty metadata");
  ThrowingMetadata::fail_default = false;
  guarded.push_back(ThrowingMetadata(41));
  ThrowingMetadata::fail_default = true;
  rejected = false;
  try { guarded.resize(3); } catch (const std::runtime_error &) { rejected = true; }
  require(rejected && guarded.size() == 1 && *guarded[0].owner == 41,
          "failed inline growth preserves owner");
  ThrowingMetadata::fail_default = false;
  guarded.resize(3);
  guarded[2] = ThrowingMetadata(43);
  ThrowingMetadata::fail_default = true;
  rejected = false;
  try { guarded.resize(7); } catch (const std::runtime_error &) { rejected = true; }
  require(rejected && guarded.size() == 3 && *guarded[2].owner == 43,
          "failed heap growth preserves metadata");
  ThrowingMetadata::fail_default = false;
  Guarded target;
  target.push_back(ThrowingMetadata(47));
  std::weak_ptr<int> target_owner = target[0].owner;
  ThrowingMetadata::fail_copy = true;
  rejected = false;
  try { target = guarded; } catch (const std::runtime_error &) { rejected = true; }
  require(rejected && target.size() == 1 && *target[0].owner == 47 &&
          guarded.size() == 3 && *guarded[2].owner == 43,
          "failed heap copy assignment preserves both containers");
  guarded.resize(1);
  rejected = false;
  try { target = guarded; } catch (const std::runtime_error &) { rejected = true; }
  require(rejected && *target[0].owner == 47 && *guarded[0].owner == 41,
          "failed inline copy assignment preserves both owners");
  ThrowingMetadata::fail_default = true;
  auto transferred = std::move(target);
  require(target.empty() && *transferred[0].owner == 47,
          "moving single metadata does not default construct its payload");
  transferred.clear(); guarded.clear();
  require(target_owner.expired(), "clear releases owners without payload construction");
  ThrowingMetadata::fail_default = ThrowingMetadata::fail_copy = false;
}
void numeric_line_sort_runtime_contracts(const char *worker) {
  namespace t = calmetrics_engine::typed;
  constexpr std::size_t length = 2048, rows = 4;
  const auto available = o::axis_path_detail::numeric_line_sort_available();
  const auto reserve = o::axis_path_detail::native_sort_library_reserve_bytes;
  // block_apply captures are aligned rank-1 values; keep Q the same length as x.
  std::vector<double> data(length), probabilities(length, .5);
  std::iota(data.begin(), data.end(), 0.0);
  o::Value values; values.data = data.data(); values.shape = o::vector_shape(length);
  values.set_contiguous_strides();
  o::Value q; q.data = probabilities.data(); q.shape = o::vector_shape(probabilities.size());
  q.set_contiguous_strides();
  const std::int64_t starts[rows]{}, ends[rows]{length, length, length, length};
  const auto has_reserve = [](const p::Plan &plan) {
    return std::count(plan.reason_codes.begin(), plan.reason_codes.end(),
                      "native_numeric_line_sort_resource_reserve");
  };
  // Both roots are scalar. The scope case keeps the large sort entirely inside
  // a captured child, so neither admission nor audit may depend on root size.
  for (const bool scoped : {false, true}) {
    const std::string expression = scoped
        ? "mean(block_apply(mean(quantiles_axis(x,q,0)),2048))"
        : "mean(quantiles_axis(x,q,0))";
    auto graph = c::compile({expression}, std::vector<t::Variable>{
        {"x", t::ValueType::series()}, {"q", t::ValueType::vector("Q")}});
    n::Batch batch;
    for (const auto &name : graph->input_names)
      batch.inputs.push_back(name == "x" ? values : q);
    batch.starts = starts; batch.ends = ends; batch.rows = rows;
    for (const int lane : {0, 1, 2, 3}) {
      p::Config config;
      config.min_rows_per_worker = 1;
      config.thread_work_units = lane == 0 ? 1e99 : 1;
      config.process_work_units = lane >= 2 ? 1 : 1e100;
      config.shared_memory_threshold_bytes = lane == 3 ? 1 : 1ull << 30;
      n::Engine engine(4, config, worker);
      auto plan = engine.plan(graph, batch);
      require(plan->lane == (lane == 0 ? "single" : lane == 1 ? "thread" : "process"),
              "numeric-line sort runtime lane");
      require(lane < 2 || plan->use_shared_memory == (lane == 3),
              "numeric-line sort transport lane");
      require(has_reserve(*plan) == (available ? 1 : 0),
              "numeric-line sort scope admission must charge once");
      require(!available || plan->estimated_worker_scratch_bytes >= reserve,
              "numeric-line sort fixed reserve missing from worker admission");
      std::vector<double> first(rows, -912.0), second(rows, -913.0);
      const auto check_audit = [&](const n::ExecutionAudit &audit, bool used_library) {
        std::size_t calls = 0;
        for (const auto &chunk : audit.chunks) {
          calls += chunk.highway_vqsort_calls;
          require(chunk.native_library_workspace_reserve_bytes ==
                      (used_library ? reserve : 0),
                  "numeric-line sort estimated reserve IPC/child audit");
        }
        require(calls == (used_library ? rows : 0),
                "numeric-line sort actual library calls IPC/child audit");
      };
      const auto first_audit = engine.execute(*plan, batch, first.data(), 10.0);
      check_audit(first_audit, available);
      require(std::all_of(first.begin(), first.end(), [](double v) { return v == 1023.5; }),
              "numeric-line sort scalar consumer value");
      // The NaN is outside every requested rank. It changes physical admission,
      // never plan budget; the healthy scalar result and prior result survive.
      data.back() = std::numeric_limits<double>::quiet_NaN();
      const auto second_audit = engine.execute(*plan, batch, second.data(), 10.0);
      check_audit(second_audit, false);
      require(second == first && data.front() == 0.0 && std::isnan(data.back()),
              "numeric-line sort fallback/input/previous-output ownership");
      std::iota(data.begin(), data.end(), 0.0);
      auto same_geometry = engine.plan(graph, batch);
      require(same_geometry->estimated_total_memory_bytes == plan->estimated_total_memory_bytes,
              "numeric-line sort reserve must not depend on numerical guard");
      if (lane == 0) {
        bool rejected = false;
        try { engine.plan(graph, batch, plan->estimated_total_memory_bytes - 1); }
        catch (const std::bad_alloc &) { rejected = true; }
        require(rejected, "numeric-line sort exact memory rejection");
        const auto exact = engine.plan(graph, batch, plan->estimated_total_memory_bytes);
        require(exact->estimated_total_memory_bytes == plan->estimated_total_memory_bytes,
                "numeric-line sort exact memory admission");
        auto empty_batch = batch; empty_batch.rows = 0;
        const auto empty = engine.plan(graph, empty_batch);
        require(has_reserve(*empty) == 0, "zero rows need no numeric-line sort reserve");
      }
    }
  }
}
} // namespace
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  try {
    check_single_metadata();
    numeric_line_sort_runtime_contracts(argv[1]);
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
    const auto geometry = p::inspect(batch.input_sizes(), batch.starts, batch.ends,
        batch.rows, batch.product_ids, false);
    require(geometry.row_lengths.size() == batch.rows && geometry.groups.empty() &&
        geometry.observations > 0 && geometry.sort_work > 0,
        "public geometry inspection retains statistics without group materialization");
    p::Plan overflow_plan;
    overflow_plan.graph = graph;
    overflow_plan.cpu_budget = 1;
    overflow_plan.row_count = 2;
    overflow_plan.input_sizes = {static_cast<std::size_t>(PTRDIFF_MAX)};
    const std::int64_t overflow_starts[]{0, 0}, overflow_ends[]{PTRDIFF_MAX, PTRDIFF_MAX};
    bool overflow_rejected = false;
    try {
      p::validate_plan(overflow_plan, overflow_plan.input_sizes, overflow_starts, overflow_ends, 2, nullptr, 1);
    } catch (const std::overflow_error &) { overflow_rejected = true; }
    require(overflow_rejected, "signature-only validation preserves observation overflow checks");
    std::vector<double> reference(starts.size() * 4), actual(reference.size());
    serial.execute(*serial_plan, batch, reference.data());
    require(reference[0] == 50.5, "serial mean");
    auto checked = serial.check(*serial_plan, batch);
    auto moved_check = std::move(checked);
    bool consumed = false;
    try { serial.execute(std::move(checked), actual.data()); }
    catch (const std::invalid_argument &) { consumed = true; }
    require(consumed, "moved execution proof cannot be reused");
    serial.execute(std::move(moved_check), actual.data());
    require(actual == reference, "checked execution parity");
    consumed = false;
    try { serial.execute(std::move(moved_check), actual.data()); }
    catch (const std::invalid_argument &) { consumed = true; }
    require(consumed, "execution proof is single use");
    auto changed_batch = batch;
    changed_batch.inputs[0].shape.dim[0] -= 1;
    bool changed = false;
    try { serial.check(*serial_plan, changed_batch); }
    catch (const std::invalid_argument &) { changed = true; }
    require(changed, "fresh checks reject changed input extent");
    // Inline metadata must preserve values across storage transitions, copies,
    // moves and worker-sized batches, and release retained owners on clear.
    calmetrics_engine::InlineVector<std::shared_ptr<int>, 4> small;
    auto owned = std::make_shared<int>(7);
    std::weak_ptr<int> weak = owned;
    small.push_back(owned); owned.reset();
    small.resize(12); small[11] = std::make_shared<int>(19);
    auto copy = small;
    auto moved = std::move(small);
    require(small.empty() && *moved[0] == 7 && *copy[11] == 19, "metadata move/copy");
    small.resize(2); small[1] = std::make_shared<int>(23);
    require(*small[1] == 23, "moved metadata can be reused inline");
    small.resize(8);
    require(*small[1] == 23, "reused metadata can grow back to heap");
    small.clear();
    moved.resize(1); copy.clear();
    require(!weak.expired() && *moved[0] == 7, "metadata heap to inline");
    moved.clear();
    require(weak.expired(), "metadata clear releases owners");
    p::Config config;
    config.thread_work_units = 1;
    config.process_work_units = 1e100;
    config.min_rows_per_worker = 1;
    n::Engine threaded(4, config, argv[1]);
    auto wrong_owner = serial.check(*serial_plan, batch);
    bool owner_rejected = false;
    try { threaded.execute(std::move(wrong_owner), actual.data()); }
    catch (const std::invalid_argument &) { owner_rejected = true; }
    require(owner_rejected, "execution proof cannot cross engines");
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
