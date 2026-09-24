#pragma once
#include "calmetrics_engine/graph.hpp"

namespace calmetrics_engine::graph {
// A physical cache tile, independent of the logical dtype/axis contract.
inline constexpr std::size_t pointwise_tile_elements = 2048;
struct InplacePointwiseChain {
  ops::ScalarChainKernel kernel = nullptr;
  ops::ScalarChain chain;
  ops::Value input;
  double *output = nullptr;
  std::size_t count = 0;
  bool output_is_first = false;
};
bool pointwise_eligible(const Program &);
void plan_pointwise_regions(const Program &, ExecutionMetadata &);
std::size_t pointwise_workspace_bytes(const Program &, std::size_t);
void release_pointwise_workspace();
// Sufficient proof only: finite state -> finite state for a short scalar chain.
// It never authorizes in-place execution of an arbitrary iterative subgraph.
bool pointwise_iteration_inplace_safe(const Program &, const std::vector<ops::Value> &,
    const double *, std::size_t state_input);
// Returns false before writing anything when this invocation requires the
// general executor (isolation, mixed geometry, or scalar roots).
bool execute_pointwise(const Program &, const std::vector<ops::Value> &,
    const double *, std::size_t, const std::int64_t *, const std::int64_t *,
    std::size_t, void *, const ResultLayout *, std::size_t, Audit &,
    std::chrono::steady_clock::time_point, const std::atomic<bool> *, bool prebound = false,
    std::size_t tile_begin = 0, std::size_t tile_end = SIZE_MAX,
    const double *previous = nullptr, double *residual = nullptr, bool *finite = nullptr,
    InplacePointwiseChain *capture = nullptr);
// Reuse within one proven finite-preserving iteration scope only, after its
// first full execution validated all geometry, parameters and output bounds.
void execute_inplace_pointwise_chain(const InplacePointwiseChain &, Audit &,
    double &, bool &, std::chrono::steady_clock::time_point, const std::atomic<bool> *);
}
