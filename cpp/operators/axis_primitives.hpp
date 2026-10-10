#pragma once

#include <cmath>
#include <cstddef>

namespace calmetrics_engine::ops::axis_path_detail {

inline double ordered_sum_step(double total, double current) noexcept {
    return total + current;
}

// The same narrow mathematical bodies are used by canonical axis operators
// and eligible private consumers. Callers own geometry, budgets and lifetimes.
inline double product_step(double state, double current) noexcept {
    return state * current;
}

inline double strict_max_step(double state, double current) noexcept {
    return current > state ? current : state;
}

struct NaNLast {
    bool operator()(double left, double right) const noexcept {
        if (std::isnan(left)) return false;
        if (std::isnan(right)) return true;
        return left < right;
    }
};

// count>0, sorted uses NaNLast, and probability is finite in [0,1]. No
// stable tie/NaN-payload order is promised. Each product is materialized before
// the final add under the engine's no-contraction compilation policy.
inline double sorted_quantile(const double *sorted, std::size_t count,
                              double probability) noexcept {
    if (count == 1 || probability == 0.0) return sorted[0];
    if (probability == 1.0) return sorted[count - 1];
    const double position = static_cast<double>(count - 1) * probability;
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    if (lower == upper) return sorted[lower];
    const double fraction = position - static_cast<double>(lower);
    const double lower_product = sorted[lower] * (1.0 - fraction);
    const double upper_product = sorted[upper] * fraction;
    return lower_product + upper_product;
}

} // namespace calmetrics_engine::ops::axis_path_detail
