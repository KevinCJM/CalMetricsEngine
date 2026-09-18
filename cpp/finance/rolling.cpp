#include "calmetrics_engine/finance.hpp"
#include "calmetrics_engine/calendar.hpp"

namespace calmetrics_engine {
void rolling_gain(MatrixView values, VectorView<std::int64_t> starts, VectorView<std::int64_t> ends,
                  VectorView<std::int64_t> dates, int months,
                  const std::array<double*, 9>& outputs, unsigned threads) {
    if (values.cols == 0) return;
    if (values.rows == 0) {
        for (auto* output : outputs) std::fill_n(output, values.cols, nan);
        return;
    }
    std::vector<std::size_t> future(values.rows, values.rows - 1);
    std::size_t cursor = 0;
    for (std::size_t row = 0; row < values.rows; ++row) {
        const auto target = add_months_ns(dates[row], months);
        while (cursor < values.rows && dates[cursor] < target) ++cursor;
        future[row] = cursor < values.rows ? cursor : values.rows - 1;
    }
    const auto tail = static_cast<std::size_t>(
        std::count(future.begin(), future.end(), values.rows - 1));
    // The original tail+1 rule is retained, but must never write before ret[0].
    const auto cutoff = tail < values.rows ? values.rows - tail - 1 : 0;
    parallel_columns(values.cols, threads, [&](std::size_t first, std::size_t step) {
        std::vector<double> net_value(values.rows), returns(values.rows);
        for (std::size_t col = first; col < values.cols; col += step) {
            double accumulated = 1.0;
            bool started = false;
            for (std::size_t row = 0; row < values.rows; ++row) {
                const auto signed_row = static_cast<std::int64_t>(row);
                if (signed_row < starts[col] || signed_row > ends[col]) {
                    net_value[row] = nan;
                    continue;
                }
                const auto value = values(row, col);
                if (std::isnan(value)) { net_value[row] = accumulated; continue; }
                if (!started) { accumulated = 1.0; started = true; }
                accumulated *= 1.0 + value;
                net_value[row] = accumulated;
            }
            for (std::size_t row = 0; row < values.rows; ++row) {
                const auto later = future[row];
                returns[row] = row >= cutoff || std::isnan(net_value[row])
                             || std::isnan(net_value[later]) ? nan
                             : net_value[later] / net_value[row] - 1.0;
            }
            std::size_t count = 0, positive = 0;
            std::array<std::size_t, 6> buckets{};
            for (const auto value : returns) {
                if (std::isnan(value)) continue;
                ++count;
                if (value > 0.0) ++positive;
                if (0.0 < value && value <= 0.05) ++buckets[0];
                else if (0.05 < value && value <= 0.10) ++buckets[1];
                else if (value > 0.10) ++buckets[2];
                else if (-0.05 <= value && value <= 0.0) ++buckets[3];
                else if (-0.10 <= value && value < -0.05) ++buckets[4];
                else if (value < -0.10) ++buckets[5];
            }
            outputs[0][col] = nan_mean(returns);
            outputs[1][col] = nan_median(returns);
            outputs[2][col] = count ? static_cast<double>(positive) / count : nan;
            for (std::size_t k = 0; k < buckets.size(); ++k) {
                outputs[k + 3][col] = count ? static_cast<double>(buckets[k]) / count : nan;
            }
            // nan_median compacts scratch storage; restore its size for the next column.
            returns.resize(values.rows);
        }
    });
}
}  // namespace calmetrics_engine
