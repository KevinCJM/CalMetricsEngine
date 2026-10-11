#pragma once

#include "calmetrics_engine/operators.hpp"

namespace calmetrics_engine::ops::axis_path_detail {
// Estimated fixed library resources, separate from the existing N-double
// caller-owned scratch. Includes a conservative allowance above the pinned
// release library's measured own stack and TLS; it is not a malloc capacity
// or a strict bound on external OS frames.
inline constexpr std::size_t native_sort_library_reserve_bytes = 16 * 1024;
bool numeric_line_sort_available() noexcept;

// Sort only mutable caller-owned scratch, never borrowed input. Exceptional
// bit patterns retain the canonical NaNLast standard-library path.
bool sort_numeric_line(double *keys, std::size_t count, Isa requested,
                       Audit *audit = nullptr);
// Private scratch needs only the adjacent ranks consumed by sorted_quantile.
// False means the caller must retain the canonical complete sorting path.
bool partition_quantile_line(double *keys, std::size_t count, double probability,
                             Isa requested, Audit *audit = nullptr);
} // namespace calmetrics_engine::ops::axis_path_detail
