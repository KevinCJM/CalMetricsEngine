#include "calmetrics_engine/finance.hpp"
#include <map>

namespace calmetrics_engine {
void column_statistics(MatrixView values, double* stds, double* means, unsigned threads) {
    parallel_columns(values.cols, threads, [&](std::size_t first, std::size_t step) {
        for (std::size_t col = first; col < values.cols; col += step) {
            double sum = 0.0, sum_squares = 0.0;
            std::size_t count = 0;
            for (std::size_t row = 0; row < values.rows; ++row) {
                const double value = values(row, col);
                if (!std::isnan(value)) {
                    sum += value;
                    sum_squares += value * value;
                    ++count;
                }
            }
            stds[col] = nan;
            if (means) means[col] = nan;
            if (count > 1) {
                const double mean = sum / static_cast<double>(count);
                stds[col] = std::sqrt((sum_squares - count * mean * mean) / (count - 1));
                if (means) means[col] = mean;
            }
        }
    });
}

void cpr(MatrixView values, VectorView<std::int32_t> types, double* result, unsigned threads) {
    std::map<std::int32_t, std::vector<std::size_t>> grouped;
    for (std::size_t col = 0; col < values.cols; ++col) grouped[types[col]].push_back(col);
    std::vector<const std::vector<std::size_t>*> groups;
    groups.reserve(grouped.size());
    for (const auto& group : grouped) groups.push_back(&group.second);
    // Fuse median/comparison/transition passes: O(columns) scratch, not O(rows*columns).
    parallel_columns(groups.size(), threads, [&](std::size_t first, std::size_t step) {
        std::vector<double> buffer;
        for (std::size_t group = first; group < groups.size(); group += step) {
            const auto& columns = *groups[group];
            std::vector<std::int8_t> previous(columns.size(), -1);
            std::vector<std::size_t> same(columns.size(), 0), changed(columns.size(), 0);
            buffer.reserve(columns.size());
            for (std::size_t row = 0; row < values.rows; ++row) {
                buffer.clear();
                for (const auto col : columns) {
                    const auto value = values(row, col);
                    if (!std::isnan(value)) buffer.push_back(value);
                }
                const auto median = nan_median(buffer);
                if (std::isnan(median)) continue;
                for (std::size_t k = 0; k < columns.size(); ++k) {
                    const auto value = values(row, columns[k]);
                    if (std::isnan(value)) continue;
                    const std::int8_t current = value >= median;
                    if (previous[k] != -1) {
                        if (previous[k] == current) ++same[k];
                        else ++changed[k];
                    }
                    previous[k] = current;
                }
            }
            for (std::size_t k = 0; k < columns.size(); ++k) {
                result[columns[k]] = changed[k] ? static_cast<double>(same[k]) / changed[k] : nan;
            }
        }
    });
}
}  // namespace calmetrics_engine
