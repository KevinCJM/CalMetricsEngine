#include "numeric_line_sort.hpp"
#include "axis_primitives.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>

#if defined(CALMETRICS_HAVE_HIGHWAY_ORDER)
#include "hwy/contrib/sort/vqsort.h"
#include "hwy/per_target.h"
#endif

namespace calmetrics_engine::ops::axis_path_detail {
namespace {
bool library_line_domain(const double *keys, std::size_t count) {
  bool has_nan = false, has_subnormal = false, negative_zero = false;
  for (std::size_t i = 0; i < count; ++i) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, keys + i, sizeof(bits));
    const auto magnitude = bits & UINT64_C(0x7fffffffffffffff);
    has_nan |= magnitude > UINT64_C(0x7ff0000000000000);
    has_subnormal |= magnitude != 0 && magnitude < UINT64_C(0x0010000000000000);
    negative_zero |= bits == UINT64_C(0x8000000000000000);
  }
  return !has_nan && !has_subnormal && !negative_zero;
}
} // namespace
bool numeric_line_sort_available() noexcept {
#if defined(CALMETRICS_HAVE_HIGHWAY_ORDER)
  return true;
#else
  return false;
#endif
}

bool sort_numeric_line(double *keys, std::size_t count, Isa requested,
                       Audit *audit) {
  if (count <= 1) return false; // Avoid even null + 0 for unused lines.
#if defined(CALMETRICS_HAVE_HIGHWAY_ORDER)
  if (requested == Isa::automatic && count >= 1024 && hwy::HaveFloat64()) {
    // SIMD min/max can canonicalize NaNs or flush subnormals; VQSort can also
    // raise an input-denormal flag for negative zero with ARM flush enabled.
    // Preserve both exact bits and caller flags through the original path.
    if (library_line_domain(keys, count)) {
      hwy::VQSort(keys, count, hwy::SortAscending{});
      if (audit) {
        audit->isa = "highway_vqsort";
        ++audit->highway_vqsort_calls;
        audit->native_library_workspace_reserve_bytes = std::max(
            audit->native_library_workspace_reserve_bytes, native_sort_library_reserve_bytes);
      }
      return true;
    }
  }
#else
  (void)requested;
  (void)audit;
#endif
  std::sort(keys, keys + count, NaNLast{});
  return false;
}

bool partition_quantile_line(double *keys, std::size_t count, double probability,
                             Isa requested, Audit *audit) {
  // Short histories can occur in hundreds of thousands of independent lines.
  // Reuse the existing SIMD selection where available, with standard-library fallback.
  if (requested != Isa::automatic || count < 128 ||
      !library_line_domain(keys, count)) return false;
  const double position = static_cast<double>(count - 1) * probability;
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  if (upper >= count) return false;
  const bool lower_needed = std::floor(position) < static_cast<double>(upper);
  const auto select = [&](std::size_t rank) {
#if defined(CALMETRICS_HAVE_HIGHWAY_ORDER)
    if (hwy::HaveFloat64()) {
      hwy::VQSelect(keys, count, rank, hwy::SortAscending{});
      if (audit) {
        audit->isa = "highway_vqselect";
        audit->native_library_workspace_reserve_bytes = std::max(
            audit->native_library_workspace_reserve_bytes, native_sort_library_reserve_bytes);
      }
      return;
    }
#endif
    std::nth_element(keys, keys + rank, keys + count);
    if (audit) audit->isa = "std-nth-element-quantile";
  };
  // Existing quantile selection strategy: preserve both adjacent order
  // statistics, using the shorter side for the one required neighbor scan.
  if (lower_needed && upper > count / 2) {
    select(upper - 1);
    keys[upper] = *std::min_element(keys + upper, keys + count);
  } else {
    select(upper);
    if (lower_needed) keys[upper - 1] = *std::max_element(keys, keys + upper);
  }
  return true;
}
} // namespace calmetrics_engine::ops::axis_path_detail
