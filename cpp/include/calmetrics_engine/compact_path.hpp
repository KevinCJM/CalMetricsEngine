#pragma once

#include "calmetrics_engine/graph.hpp"

namespace calmetrics_engine::graph {

// Private lowering of an explicitly editable scan/order/compact-consumer DAG.
// A proved raw affine -> scale -> mean prefix shares one row across all groups;
// when the existing state bound needs multiple batches, admission uses all
// groups only if their extra state fits one transformed tensor's extent.
// The proof owns metadata only: no input addresses, numeric caches or workers.
struct CompactPath;
struct CompactPathBudget {
  std::size_t total_bytes = 0;
  std::size_t workspace_doubles = 0;
  std::size_t metadata_bytes = 0;
  std::size_t child_scratch_bytes = 0;
  std::size_t batch_size = 0;
  double physical_work = 0.0;
};

// The child has only static, precisely shaped inputs. The estimate must include
// all child Scratch arenas, operator scratch and descriptor retained capacity.
using CompactChildBudget = std::function<std::size_t(
    const Program &, const std::vector<ops::Value> &, std::size_t logical_window)>;

// Root executes the unchanged canonical post nodes with a distinct child
// Scratch. original_roots maps each child root to its original layout column.
// A non-null deferred_errors[node] is replayed inside that node's strict try,
// before any numeric work. Checkpoint constants have no real consumers.
using CompactPostExecutor = std::function<void(
    const Program &post_program, const std::vector<ops::Value> &synthetic_inputs,
    const double *parameters, std::size_t parameter_count,
    const std::vector<const char *> &deferred_errors,
    void *output, const ResultLayout &original_layout, std::size_t result_row,
    const std::vector<std::size_t> &original_roots, Audit &audit,
    std::chrono::steady_clock::time_point deadline,
    const std::atomic<bool> *cancelled)>;

std::shared_ptr<const CompactPath> plan_compact_path(const Program &);

// Every row length participates in admission. An unsupported/unknown geometry
// returns nullopt; callers must then retain the full canonical memory estimate.
std::optional<CompactPathBudget> compact_path_budget(
    const Program &, const CompactPath &, const std::vector<ops::Value> &inputs,
    const std::vector<std::size_t> &row_lengths, const CompactChildBudget &);

// Admission, interval/layout bounds and maximum capacity are checked before any
// result write. It never falls back on numerical values after budget admission.
bool execute_compact_path(
    const Program &, const CompactPath &, const std::vector<ops::Value> &inputs,
    const double *parameters, std::size_t parameter_count,
    const std::int64_t *starts, const std::int64_t *ends, std::size_t rows,
    void *output, const ResultLayout *, std::size_t result_row,
    ops::Workspace &, Audit &, const CompactChildBudget &,
    const CompactPostExecutor &,
    std::chrono::steady_clock::time_point deadline,
    const std::atomic<bool> *cancelled);

} // namespace calmetrics_engine::graph
