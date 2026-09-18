#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace calmetrics_engine {

inline constexpr double nan = std::numeric_limits<double>::quiet_NaN();
inline constexpr std::int64_t not_recovered = 1'000'000;

inline double nan_mean(const std::vector<double>& values) {
    double sum = 0.0;
    std::size_t count = 0;
    for (const double value : values) {
        if (!std::isnan(value)) {
            sum += value;
            ++count;
        }
    }
    return count ? sum / static_cast<double>(count) : nan;
}

// Mutates caller-owned scratch only; Python inputs remain read-only.
inline double nan_median(std::vector<double>& values) {
    values.erase(
        std::remove_if(
            values.begin(),
            values.end(),
            [](double value) { return std::isnan(value); }
        ),
        values.end()
    );
    if (values.empty()) return nan;

    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    if (values.size() & 1u) return *middle;
    return 0.5 * (*middle + *std::max_element(values.begin(), middle));
}

}  // namespace calmetrics_engine
