#include "my_ctools/kernels.hpp"
#include "my_ctools/calendar.hpp"

namespace my_ctools {
void largest_streak(MatrixView values, const std::int64_t* dates, bool positive,
                    double* result, std::int64_t* periods,
                    std::vector<std::string>& starts, std::vector<std::string>& ends,
                    unsigned threads) {
    parallel_columns(values.cols, threads, [&](std::size_t first, std::size_t step) {
        std::vector<double> ratios(values.rows);
        for (std::size_t col = first; col < values.cols; col += step) {
            result[col] = 0.0;
            periods[col] = 0;
            double accumulated = 1.0, last_selected = 1.0, denominator = 1.0;
            double extreme = positive ? -std::numeric_limits<double>::infinity()
                                      : std::numeric_limits<double>::infinity();
            std::size_t last_index = 0;
            for (std::size_t row = 0; row < values.rows; ++row) {
                const auto value = values(row, col);
                // Both historical modes select positive values. Do not reinterpret here.
                const bool selected = value > 0.0;
                if (selected) {
                    accumulated *= 1.0 + (positive ? value : -value);
                    last_selected = accumulated;
                    ratios[row] = accumulated / denominator;
                    if (ratios[row] == 1.0) ratios[row] = nan;
                } else {
                    denominator = last_selected;
                    ratios[row] = nan;
                }
                const auto ratio = ratios[row];
                if (!std::isnan(ratio) &&
                    ((positive && ratio >= extreme) || (!positive && ratio <= extreme))) {
                    extreme = ratio;
                    last_index = row;
                }
            }
            if (std::isinf(extreme)) continue;
            auto start = last_index;
            while (start > 0 && !std::isnan(ratios[start - 1])) --start;
            result[col] = extreme - 1.0;
            periods[col] = static_cast<std::int64_t>(last_index - start + 1);
            starts[col] = date_string(dates[start]);
            ends[col] = date_string(dates[last_index]);
        }
    });
}

void longest_streak(MatrixView values, const std::int64_t* dates, bool positive,
                    double* result, std::int64_t* periods,
                    std::vector<std::string>& starts, std::vector<std::string>& ends,
                    unsigned threads) {
    parallel_columns(values.cols, threads, [&](std::size_t first, std::size_t step) {
        std::vector<double> cumulative(values.rows);
        for (std::size_t col = first; col < values.cols; col += step) {
            std::size_t best_length = 0, current = 0, best_end = 0;
            double accumulated = 1.0;
            for (std::size_t row = 0; row < values.rows; ++row) {
                const auto value = values(row, col);
                if (value > 0.0) {
                    ++current;
                    if (current >= best_length) { best_length = current; best_end = row; }
                } else current = 0;
                if (!std::isnan(value)) accumulated *= 1.0 + (positive ? value : -value);
                cumulative[row] = accumulated;
            }
            result[col] = 0.0;
            periods[col] = static_cast<std::int64_t>(best_length);
            if (best_length == 0) continue;
            const auto start = best_end - best_length + 1;
            // Preserve the original row_before=-1 behavior; changing it is a financial API change.
            const auto before = start == 0 ? values.rows - 1 : start - 1;
            result[col] = cumulative[before] != 0.0
                        ? cumulative[best_end] / cumulative[before] - 1.0 : nan;
            starts[col] = date_string(dates[start]);
            ends[col] = date_string(dates[best_end]);
        }
    });
}
}  // namespace my_ctools
