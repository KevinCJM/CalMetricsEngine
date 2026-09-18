#include "calmetrics_engine/finance.hpp"
#include "calmetrics_engine/calendar.hpp"

namespace calmetrics_engine {
void longest_recovery(MatrixView values, std::int64_t* result, unsigned threads) {
    parallel_columns(values.cols, threads, [&](std::size_t first, std::size_t step) {
        for (std::size_t col = first; col < values.cols; col += step) {
            double accumulated = 1.0, maximum = 1.0;
            std::int64_t current = 0, longest = 0;
            for (std::size_t row = 0; row < values.rows; ++row) {
                const auto value = values(row, col);
                if (!std::isnan(value)) accumulated *= 1.0 + value;
                maximum = std::max(maximum, accumulated);
                const bool in_drawdown = (accumulated - maximum) / maximum != 0.0;
                if (in_drawdown) { ++current; longest = std::max(longest, current); }
                else current = 0;
            }
            result[col] = longest;
        }
    });
}

void max_drawdown(MatrixView values, VectorView<std::int64_t> dates, double* result,
                  std::vector<std::string>& result_dates, std::int64_t* recovery,
                  unsigned threads) {
    parallel_columns(values.cols, threads, [&](std::size_t first, std::size_t step) {
        std::vector<double> cumulative(values.rows), running_max(values.rows);
        for (std::size_t col = first; col < values.cols; col += step) {
            result[col] = nan;
            recovery[col] = not_recovered;
            if (values.rows == 0) continue;
            double accumulated = 1.0;
            for (std::size_t row = 0; row < values.rows; ++row) {
                const auto value = values(row, col);
                if (std::isnan(value)) cumulative[row] = nan;
                else { accumulated *= 1.0 + value; cumulative[row] = accumulated; }
            }
            // Preserve legacy bfill, including its internal-NaN behavior.
            double last = nan;
            for (std::size_t row = values.rows; row-- > 0;) {
                if (std::isnan(cumulative[row])) cumulative[row] = last;
                else last = cumulative[row];
            }
            if (std::isnan(cumulative[0])) {
                result[col] = 0.0;
                result_dates[col] = date_string(dates[0]);
                continue;
            }
            double maximum = cumulative[0], minimum_drawdown = 0.0;
            std::size_t minimum_index = 0;
            for (std::size_t row = 0; row < values.rows; ++row) {
                maximum = std::max(maximum, cumulative[row]);
                running_max[row] = maximum;
                const double drawdown = (cumulative[row] - maximum) / maximum;
                if (drawdown < minimum_drawdown) {
                    minimum_drawdown = drawdown;
                    minimum_index = row;
                }
            }
            result[col] = std::abs(minimum_drawdown);
            result_dates[col] = date_string(dates[minimum_index]);
            for (std::size_t row = minimum_index + 1; row < values.rows; ++row) {
                if (cumulative[row] >= running_max[row]) {
                    recovery[col] = static_cast<std::int64_t>(row - minimum_index);
                    break;
                }
            }
            if (minimum_index == 0) recovery[col] = 0;
        }
    });
}
}  // namespace calmetrics_engine
