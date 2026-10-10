#pragma once

#include "calmetrics_engine/graph.hpp"

namespace calmetrics_engine::graph {

// Process-local proof of one private affine/expm1 producer and its complete
// editable consumers. No addresses, array extents or numerical values are kept.
struct AffinePeriodRoot {
  std::uint32_t root = 0, weights = 0;
  bool swapped = false;
  std::size_t contribution_begin = 0, contribution_end = 0;
  // Contiguous assets from zero map to contiguous contribution indices. The
  // runtime additionally requires the link count to equal the actual asset axis.
  std::size_t dense_contribution_begin = SIZE_MAX;
};
struct AffineContributionRoot {
  std::uint32_t root = 0, weights = 0, divisor = 0, sum = 0;
  std::size_t asset = 0;
};
struct AffineContributionLink {
  std::size_t asset = 0, contribution = 0;
};
struct AffineProjection {
  std::uint32_t affine = 0, returns = 0;
  std::size_t asset_axis = 0;
  std::vector<AffinePeriodRoot> periods;
  std::vector<AffineContributionRoot> contributions;
  // A contribution has exactly one carrier with the same weights node, or
  // belongs to the unmatched list. Links are sparse and asset-ordered.
  std::vector<AffineContributionLink> contribution_links;
  std::vector<std::size_t> unmatched_contributions;
};

// Internal producer shared by bounded mathematical consumer proofs. Packing
// and ownership remain the caller's responsibility; packet coefficients must
// have stride 1 on the asset axis. The scalar/Eigen fold has one implementation.
std::size_t affine_producer_coefficient_reserve(std::size_t rows,
    std::size_t factors, std::size_t assets);
bool affine_producer_packets_enabled(std::size_t coefficient_reserve);
// The same ascending-factor fold without an expm1 epilogue. Private row
// producers can retain distinct logical affine, scale and mean nodes.
void affine_producer_linear_row(std::size_t factors, std::size_t outputs,
    const ops::Value &coefficients, const ops::Value &initial,
    const double *source, std::ptrdiff_t base, std::ptrdiff_t factor_stride,
    double *destination, bool use_packets, Audit &);
void affine_producer_return_row(std::size_t factors, std::size_t assets,
    const ops::Value &coefficients, const ops::Value &initial,
    const double *source, std::ptrdiff_t base, std::ptrdiff_t factor_stride,
    double *destination, bool use_packets, Audit &);

// This proof never changes the logical DAG or its canonical arena slots.
std::shared_ptr<const AffineProjection> plan_affine_projection(const Program &);

// Same geometry admission as execution, without reading numerical payloads.
// Includes the bounded tile, contribution state, per-node descriptors, and
// the shared proof's actual metadata capacities (conservative per worker).
// nullopt means that the normal executor and its complete budget are required.
std::optional<std::size_t> affine_projection_workspace_bytes(
    const Program &, const AffineProjection &, const std::vector<ops::Value> &,
    std::size_t logical_window);

// Caller additionally excludes observers, prebound scopes and failed captures.
// false is returned before any output write for unsupported geometry. Strict
// mathematical failures remain canonical errors, not fallback conditions.
// Workspace belongs to the existing scheduler/executor; there is no private pool.
bool execute_affine_projection(
    const Program &, const AffineProjection &, const std::vector<ops::Value> &,
    const double *parameters, std::size_t parameter_count,
    const std::int64_t *starts, const std::int64_t *ends, std::size_t rows,
    void *output, const ResultLayout *, std::size_t result_row,
    ops::Workspace &, Audit &,
    std::chrono::steady_clock::time_point deadline,
    const std::atomic<bool> *cancelled);

} // namespace calmetrics_engine::graph
