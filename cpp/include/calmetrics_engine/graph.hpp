#pragma once

#include "calmetrics_engine/operators.hpp"

#include <array>
#include <algorithm>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <atomic>
#include <chrono>
#include <functional>
#include <optional>
#include <vector>
#include <utility>

namespace calmetrics_engine::graph {

enum class NodeKind : std::uint8_t {
  input = 0,
  parameter = 1,
  constant = 2,
  operation = 3,
  rolling_scope = 4,
  interval_tail = 5,
  apply_scope = 6,
  iteration_projection = 7
};

enum class OutputKind : std::uint8_t { scalar = 0, series = 1, typed = 2 };
inline const char *output_kind_name(OutputKind kind) noexcept {
  return kind == OutputKind::typed ? "typed" : kind == OutputKind::series ? "series" : "scalar";
}
// Public scalar results retain float64. Aligned series preserve one exact
// dtype for every root in the graph, including across native worker transport.
enum class OutputDType : std::uint8_t { float64 = 0, boolean = 1, int64 = 2 };
inline std::size_t output_itemsize(OutputDType dtype) noexcept {
  return dtype == OutputDType::boolean ? 1 : 8;
}
inline const char *output_dtype_name(OutputDType dtype) noexcept {
  switch (dtype) {
  case OutputDType::float64: return "float64";
  case OutputDType::boolean: return "bool";
  case OutputDType::int64: return "int64";
  }
  return "invalid";
}
inline void *output_offset(void *data, std::size_t elements,
                           OutputDType dtype) noexcept {
  return data ? static_cast<std::uint8_t *>(data) + elements * output_itemsize(dtype)
              : nullptr;
}
// Per-root native schema. Typed storage never coerces integer or mask values.
enum class OutputOrder : std::uint8_t { c = 0, f = 1, automatic = 2 };
struct RootOutput {
  OutputDType dtype = OutputDType::float64;
  std::uint8_t rank = 0;
  OutputOrder order = OutputOrder::c; // Legacy native programs retain C.
};
struct ResultSlot { std::size_t byte_offset = 0, status_offset = 0, capacity = 0; };
// Layout flags for the common small-root plan do not need a heap allocation.
// Larger plans retain one byte per root, within the existing metadata budget.
class RootOrders {
  std::size_t count_ = 0;
  std::bitset<64> bits_;
  std::unique_ptr<ops::MemoryOrder[]> large_;
public:
  RootOrders() = default;
  RootOrders(const RootOrders &other) : count_(other.count_), bits_(other.bits_) {
    if (other.large_) {
      large_ = std::make_unique<ops::MemoryOrder[]>(count_);
      std::copy_n(other.large_.get(), count_, large_.get());
    }
  }
  RootOrders(RootOrders &&other) noexcept
      : count_(std::exchange(other.count_, 0)), bits_(other.bits_), large_(std::move(other.large_)) {}
  void swap(RootOrders &other) noexcept {
    std::swap(count_, other.count_); std::swap(bits_, other.bits_); large_.swap(other.large_);
  }
  RootOrders &operator=(const RootOrders &other) {
    if (this != &other) { RootOrders next(other); swap(next); }
    return *this;
  }
  RootOrders &operator=(RootOrders &&other) noexcept {
    if (this != &other) { RootOrders next(std::move(other)); swap(next); }
    return *this;
  }
  bool empty() const noexcept { return count_ == 0; }
  std::size_t size() const noexcept { return count_; }
  void assign(std::size_t count, ops::MemoryOrder value) {
    ops::require(value == ops::MemoryOrder::c || value == ops::MemoryOrder::f, "RESULT_LAYOUT_ORDER");
    RootOrders next; next.count_ = count;
    if (count > 64) {
      next.large_ = std::make_unique<ops::MemoryOrder[]>(count);
      std::fill_n(next.large_.get(), count, value);
    } else if (value == ops::MemoryOrder::f) next.bits_.set();
    swap(next);
  }
  ops::MemoryOrder operator[](std::size_t root) const noexcept {
    return large_ ? large_[root] : (bits_[root] ? ops::MemoryOrder::f : ops::MemoryOrder::c);
  }
  ops::MemoryOrder at(std::size_t root) const {
    if (root >= count_) throw std::out_of_range("result layout root");
    return (*this)[root];
  }
  void set(std::size_t root, ops::MemoryOrder value) {
    if (root >= count_) throw std::out_of_range("result layout root");
    ops::require(value == ops::MemoryOrder::c || value == ops::MemoryOrder::f, "RESULT_LAYOUT_ORDER");
    if (large_) large_[root] = value;
    else bits_.set(root, value == ops::MemoryOrder::f);
  }
};
struct ResultLayout {
  std::size_t columns = 0;
  std::vector<ResultSlot> slots;
  // Empty for all-C/legacy/internal scope layouts, otherwise one flag per root.
  RootOrders orders;
  ops::MemoryOrder order(std::size_t root) const {
    return orders.empty() ? ops::MemoryOrder::c : orders.at(root);
  }
  std::vector<std::size_t> row_bytes, row_statuses;
  std::size_t bytes() const { return row_bytes.empty() ? 0 : row_bytes.back(); }
  std::size_t status_count() const { return row_statuses.empty() ? 0 : row_statuses.back(); }
};
enum class RollingParameterKind : std::uint8_t {
  outer_node = 0,
  observation_count = 1,
  window_elapsed_days = 2,
  risk_free_return_window = 3
};
enum class StorageKind : std::uint8_t {
  inline_value = 0,
  numeric = 1,
  mask = 2,
  integer = 3
};

// Most mathematical nodes use at most eight inputs. Keep the original 32-slot
// value contract, including unused slots, without storing its cold tail in
// every logical, compiler and physical node. Overflow is node-owned and copied
// deeply; no mutable dependency storage is shared between programs.
class ParentSlots {
  static constexpr std::size_t inline_count = 8, total_count = 32;
  std::array<std::uint32_t, inline_count> local_{};
  std::unique_ptr<std::array<std::uint32_t, total_count - inline_count>> tail_;
  inline static constexpr std::uint32_t zero_ = 0;
public:
  ParentSlots() = default;
  ParentSlots(const ParentSlots &other) : local_(other.local_) {
    if (other.tail_) tail_ = std::make_unique<decltype(tail_)::element_type>(*other.tail_);
  }
  ParentSlots &operator=(const ParentSlots &other) {
    if (this != &other) {
      auto next = other.tail_ ? std::make_unique<decltype(tail_)::element_type>(*other.tail_) : nullptr;
      local_ = other.local_;
      tail_ = std::move(next);
    }
    return *this;
  }
  ParentSlots(ParentSlots &&) noexcept = default;
  ParentSlots &operator=(ParentSlots &&) noexcept = default;
  static constexpr std::size_t size() noexcept { return total_count; }
  std::size_t overflow_bytes() const noexcept { return tail_ ? sizeof(*tail_) : 0; }
  const std::uint32_t &operator[](std::size_t index) const noexcept {
    // Explicit branches preserve the lvalue in MSVC's default conditional mode.
    if (index < inline_count) return local_[index];
    if (tail_) return (*tail_)[index - inline_count];
    return zero_;
  }
  std::uint32_t &operator[](std::size_t index) {
    if (index < inline_count) return local_[index];
    if (!tail_) tail_ = std::make_unique<decltype(tail_)::element_type>();
    return (*tail_)[index - inline_count];
  }
  bool contains(std::uint32_t value, std::size_t count) const noexcept {
    for (std::size_t i = 0; i < count; ++i) if ((*this)[i] == value) return true;
    return false;
  }
  std::vector<std::uint32_t> prefix(std::size_t count) const {
    std::vector<std::uint32_t> result(count);
    for (std::size_t i = 0; i < count; ++i) result[i] = (*this)[i];
    return result;
  }
  friend bool operator==(const ParentSlots &a, const ParentSlots &b) noexcept {
    if (a.local_ != b.local_) return false;
    for (std::size_t i = inline_count; i < total_count; ++i)
      if (a[i] != b[i]) return false;
    return true;
  }
  friend bool operator!=(const ParentSlots &a, const ParentSlots &b) noexcept { return !(a == b); }
};

struct Node {
  NodeKind kind = NodeKind::constant;
  std::uint16_t opcode = 0;
  std::uint16_t input_index = 0;
  double constant = 0.0;
  ParentSlots parents;
  std::uint8_t parent_count = 0;
  StorageKind storage = StorageKind::inline_value;
  std::uint32_t slot = 0;
};

struct Program;

struct RollingParameterBinding {
  RollingParameterKind kind = RollingParameterKind::outer_node;
  std::uint32_t node = 0;
};

struct RollingScope {
  std::shared_ptr<Program> body;
  std::vector<std::uint32_t> input_nodes;
  std::vector<std::uint8_t> input_preceding;
  std::vector<RollingParameterBinding> parameter_bindings;
  std::uint32_t width_node = 0;
  std::uint32_t min_periods_node = 0;
  std::uint32_t dates_node = 0;
  std::uint32_t annual_rate_node = 0;
  bool has_min_periods = false;
  bool has_date_context = false;
  bool has_returns = false;
  bool needs_preceding_observation = false;
  std::int32_t returns_input = -1;
  std::size_t body_node_count = 0;
  std::size_t array_count = 0;
};

enum class ApplyKind : std::uint8_t { block = 0, filter = 1, group = 2, bisect = 3, segment = 4, iterate = 5 };
enum class SolverStatus : std::int64_t {
  converged = 0, iteration_limit = 1, no_feasible_start = 2,
  infeasible = 3, local_stop = 4, numerical_failure = 5
};
struct ApplyScope {
  ApplyKind kind = ApplyKind::block;
  std::shared_ptr<Program> body;
  std::vector<std::uint32_t> input_nodes;
  // UINT32_MAX binds the local solver variable rather than an outer node.
  std::vector<std::uint32_t> parameter_nodes;
  std::vector<std::uint32_t> argument_nodes;
  std::size_t body_node_count = 0;
  std::int32_t state_input_index = -1;
};

// Process-local AOT kernels are rebuilt by finalize(), never serialized.
struct PointwiseRegion {
  std::uint32_t source = 0, first = 0, second = UINT32_MAX, predicate = UINT32_MAX;
  std::array<std::uint32_t, 3> scalars{UINT32_MAX, UINT32_MAX, UINT32_MAX};
  std::array<bool, 3> scalar_first{};
  ops::ScalarChainKernel kernel = nullptr;
  bool retain_first = true;
  std::uint32_t end() const { return predicate != UINT32_MAX ? predicate : second; }
};
struct MatmulSummaryRegion {
  std::uint32_t producer = 0;
  std::vector<std::uint32_t> consumers;
  std::vector<std::uint32_t> quantiles;
  double quantile_scale = 0.;
  bool trajectory_library = true;
  bool moments = false;
  std::size_t maximum_ddof = 0;
  std::array<double, 3> linear_scale{}, sqrt_scale{};
};
struct AffineProjection;
struct CompactPath;
struct RowBatchPlan;
struct ScanReductionRegion {
  std::uint32_t nav = 0, peak = 0, division = 0, subtraction = 0, reduction = 0;
};
struct SquareReductionRegion {
  std::uint32_t subtraction = 0, square = 0, selection = 0, reduction = 0;
};
// Physical tiling of private comparisons and scalar state decisions. Logical
// nodes, preparation order and the canonical mathematical kernels stay intact.
struct StateDecisionRegion {
  struct Step {
    std::uint32_t comparison = 0, selection = 0;
    std::int8_t previous_argument = -1;
  };
  std::uint32_t source = 0, valid = 0;
  bool loan_source = false; // Private argmax may use the independent terminal state root.
  std::vector<Step> steps;
};
inline constexpr std::size_t state_decision_tile_elements = 4096;
inline constexpr std::size_t state_root_loan_min_elements = 131072;
// Immutable structural candidates. Index contents are proved again per call.
struct RowSubviewRegion {
  std::uint32_t selected = 0, full_root = 0, take = 0;
};
inline constexpr std::size_t row_subview_min_elements = 262144;
struct ExecutionMetadata {
  bool pointwise = false;
  std::vector<std::int32_t> root_destinations;
  bool requires_shape_planning = false;
  std::vector<const ops::Spec *> specs;
  std::vector<std::uint8_t> summary_consumers;
  std::vector<std::uint8_t> order_consumers;
  // Keep optional provenance after the existing hot execution metadata.
  std::size_t position_status_nodes = 0;
  std::vector<std::uint8_t> position_status_reachable;
  bool contains_segment_scope = false;
  std::vector<PointwiseRegion> pointwise_regions;
  std::vector<std::uint32_t> pointwise_slots;
  std::size_t pointwise_numeric_slots = 0, pointwise_mask_slots = 0;
  // Deferred pairs read persistent backing, a saved short broadcast, or an
  // owned C array whose slot is proven untouched until its fused consumer.
  std::vector<std::int32_t> deferred_to, fused_from;
  std::vector<std::int8_t> deferred_broadcast_argument, deferred_private_argument;
  // A private full filter can materialize only its sole, proven last-row slice.
  std::vector<std::int32_t> filter_tail_source, filter_tail_slice;
  std::vector<std::int32_t> gaussian_filter_source, gaussian_filter_consumer;
  // Deferred private concatenations expand persistent leaves into one C destination.
  std::vector<std::uint8_t> concat_tree_roots;
  // A sole materialized coefficient consumer may use its own dead arena as LU workspace.
  std::vector<std::int32_t> solve_private_source;
  // Terminal floating solve and its full private arithmetic producer.
  std::vector<std::int32_t> solve_library_source, solve_library_consumer;
  std::vector<std::uint8_t> solve_copy_roots;
  bool pointwise_complete_region = false;
  std::vector<std::size_t> pointwise_root_copies;
  // A private Jacobi record can write its two independent F projections.
  std::int32_t split_eigh_source = -1, split_eigh_values = -1, split_eigh_vectors = -1;
  std::int32_t solve_pair_first = -1, solve_pair_second = -1;
  // Only these two terminal roots may consume this private coefficient arena.
  std::int32_t solve_pair_private_source = -1;
  std::vector<MatmulSummaryRegion> matmul_summaries;
  std::vector<std::int32_t> matmul_summary_producer, matmul_summary_consumer;
  std::shared_ptr<const AffineProjection> affine_projection;
  std::shared_ptr<const CompactPath> compact_path;
  std::shared_ptr<const RowBatchPlan> row_batch;
  // Internal strict independent-axis child only. Original nodes/error order
  // remain visible; each private where stores its sole sum consumer's N values.
  std::vector<std::int32_t> where_sum_to, where_sum_from;
  std::vector<ScanReductionRegion> scan_reductions;
  std::vector<std::int32_t> scan_members;
  std::vector<SquareReductionRegion> square_reductions;
  std::vector<std::int32_t> square_members;
  std::vector<StateDecisionRegion> state_decisions;
  std::vector<std::int32_t> state_decision_members;
  // Single-use matrix/scalar arithmetic shares this call's independent root.
  std::vector<std::uint32_t> matmul_scalar_chain;
  // A shared product remains in the last logical branch's independent root.
  // All other branches finish before that branch can overwrite the product.
  std::vector<std::vector<std::uint32_t>> matmul_shared_chains;
  std::int32_t matmul_shared_backing_root = -1;
  std::vector<std::uint32_t> row_subview_takes;
  std::vector<RowSubviewRegion> row_subview_reuses;
};

// A physical partition of the same logical graph, not a numerical operator.
struct RowBatchPlan {
  std::uint32_t input = 0, producer = 0;
  std::vector<std::int8_t> axes;
  std::shared_ptr<const ExecutionMetadata> canonical_metadata;
};

struct Program {
  std::vector<Node> nodes;
  std::vector<std::uint32_t> roots;
  std::size_t input_count = 0;
  std::size_t parameter_count = 0;
  std::size_t numeric_slots = 0;
  std::size_t mask_slots = 0;
  std::size_t integer_slots = 0;
  // 0: temporal vector, 1: static vector/matrix, 2: time x asset matrix.
  std::vector<std::uint8_t> input_axes;
  OutputKind output_kind = OutputKind::scalar;
  OutputDType output_dtype = OutputDType::float64;
  std::vector<RootOutput> root_outputs; // present only for typed results
  bool isolate_errors = false;
  std::uint32_t minimum_observations = 0;
  std::uint64_t scope_work_budget = 100000000;
  std::vector<RollingScope> rolling_scopes;
  std::vector<ApplyScope> apply_scopes;
  std::shared_ptr<const ExecutionMetadata> execution_metadata;

  void validate() const;
  void finalize();
};

// Structural admission shared by result layout and graph execution.
bool independent_transcendental_root(const Program &, std::size_t);
bool summary_fusion_eligible(ops::Op op) noexcept;
void release_graph_workspace();
bool order_fusion_eligible(ops::Op op) noexcept;

std::size_t required_array_capacity(const Program &program,
                                   const std::vector<ops::Value> &inputs,
                                   std::size_t max_window,
                                   std::vector<ops::Shape> *node_shapes = nullptr);

bool row_batch_inputs(const Program &, const std::vector<ops::Value> &,
    std::size_t logical_window, std::vector<ops::Value> &, std::size_t &batch,
    std::size_t &count);

// Maximum sequential matrix panel workspace; unknown dimensions use capacity.
std::size_t matrix_workspace_doubles(const Program &, const std::vector<ops::Shape> &,
                                     std::size_t capacity, bool expanded_matmul_roots = true);

bool matmul_summary_geometry(const Program &, std::size_t,
                             const std::vector<ops::Shape> &);
bool gaussian_filter_stream_geometry(const Program &, std::size_t,
                                     const std::vector<ops::Shape> &);
std::size_t matmul_summary_cache_doubles(const Program &,
    const std::vector<ops::Shape> &, std::vector<std::size_t> *offsets = nullptr);
// Existing liveness slots keep their ownership; only their byte extents change.
// The planner and executor must use the same per-node shape upper bounds.
// Prebound scopes reserve the canonical C arena when a direct F writer is disabled.
struct ArrayArenaLayout {
  std::array<std::vector<std::size_t>, 3> offsets;
  std::array<std::size_t, 3> elements{};
};
void plan_array_arena(const Program &, std::size_t capacity,
                      const std::vector<ops::Shape> &, ArrayArenaLayout &, bool fuse = false,
                      bool prebound_inputs = false, bool summary_fusion = true,
                      bool private_where_sums = false, bool allow_matmul_chain = true,
                      const std::vector<std::uint8_t> *row_subviews = nullptr,
                      std::int32_t loaned_state_source = -1);

ResultLayout result_layout(const Program &, const std::vector<ops::Value> &,
                           const std::int64_t *starts, const std::int64_t *ends, std::size_t rows);

struct Audit {
  std::size_t compact_path_groups = 0, compact_path_batches = 0, compact_path_steps = 0;
  std::size_t compact_path_raw_rows = 0, compact_path_raw_scale_elements = 0;
  std::size_t compact_path_raw_pack_bytes = 0;
  std::size_t highway_vqsort_calls = 0;
  std::size_t eigen_rotation_calls = 0;
  std::size_t eigen_matvec_calls = 0;
  std::size_t eigen_covariance_calls = 0;
  std::size_t eigen_density_calls = 0;
  std::size_t gaussian_filter_streams = 0;
  std::size_t accelerate_covchol_calls = 0;
  std::size_t accelerate_matmul_calls = 0, matmul_zero_calls = 0;
  std::size_t accelerate_covariance_attempts = 0;
  std::size_t accelerate_covariance_calls = 0;
  std::size_t covariance_syrk_calls = 0;
  std::size_t covariance_replayed_cells = 0;
  std::size_t covariance_fallbacks = 0;
  std::size_t covariance_zero_calls = 0;
  std::size_t accelerate_vforce_calls = 0;
  std::size_t accelerate_lu_attempts = 0;
  std::size_t accelerate_lu_calls = 0;
  std::size_t accelerate_lu_fallbacks = 0;
  std::size_t solve_residual_materialization_bytes = 0;
  std::size_t covariance_panel_materialization_bytes = 0;
  std::size_t native_library_workspace_reserve_bytes = 0;
  std::size_t fused_pointwise_calls = 0, pointwise_tiles = 0, vector_elements = 0;
  std::size_t root_copy_bytes = 0, direct_output_bytes = 0, state_copy_bytes = 0;
  std::size_t pointwise_workspace_capacity_bytes = 0;
  std::vector<ops::Shape> result_shapes; // row/root order; rank=-1 means unknown on failure
  std::vector<std::int16_t> root_statuses;
  // Present only for isolate mode; row-major, with the same shape as values.
  std::vector<std::int16_t> statuses;
  std::size_t rows = 0;
  std::size_t nodes = 0;
  std::size_t max_window = 0;
  std::size_t numeric_arena_bytes = 0;
  std::size_t mask_arena_bytes = 0;
  std::size_t operator_workspace_capacity_bytes = 0;
  std::size_t input_copy_bytes = 0;
  std::size_t fused_scalar_calls = 0;
  std::size_t summary_source_scans = 0;
  std::size_t order_stat_sorts = 0;
  std::size_t order_scratch_capacity_bytes = 0;
  std::size_t algorithm_copy_bytes = 0;
};

Audit execute(const Program &program, const std::vector<ops::Value> &inputs,
              const double *parameters, std::size_t parameter_count,
              const std::int64_t *starts, const std::int64_t *ends,
              std::size_t rows, void *output, std::size_t output_columns,
              const ResultLayout *layout = nullptr, std::size_t result_row = 0,
              std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max(),
              const std::atomic<bool> *cancelled = nullptr,
              bool allow_workspace_loan = false, bool expanded_matmul_roots = true,
              bool row_subviews = true);

// Internal range of the proved independent matrix-row closure. The normal
// executor and original logical nodes remain the only numerical implementation.
Audit execute_row_batch_range(const Program &, const std::vector<ops::Value> &,
    const double *parameters, std::size_t parameter_count,
    const std::int64_t *starts, const std::int64_t *ends, void *output,
    const ResultLayout &, std::size_t first, std::size_t last,
    bool publish_metadata, std::size_t &failed_node,
    std::chrono::steady_clock::time_point deadline,
    const std::atomic<bool> *cancelled, bool allow_workspace_loan,
    bool expanded_matmul_roots = true);

// Explicit diagnostic replay observes canonical operands/results, including
// nested scopes. Ordinary execution does not scan values or invoke callbacks.
// A missing opcode denotes an input/constant or structural scope value.
using ValueObserver = std::function<void(std::optional<ops::Op>, const ops::Value &, bool)>;
Audit execute_observed(const Program &program, const std::vector<ops::Value> &inputs,
                       const double *parameters, std::size_t parameter_count,
                       const std::int64_t *starts, const std::int64_t *ends,
                       std::size_t rows, void *output, std::size_t output_columns,
                       const ResultLayout *layout, const ValueObserver &observer);

} // namespace calmetrics_engine::graph
