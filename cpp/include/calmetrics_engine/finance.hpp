#pragma once

#include "calmetrics_engine/array_view.hpp"
#include "calmetrics_engine/numeric.hpp"
#include "calmetrics_engine/parallel.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace calmetrics_engine {

void column_statistics(MatrixView values, double* stds, double* means, unsigned threads);

void cpr(
    MatrixView values,
    VectorView<std::int32_t> types,
    double* result,
    unsigned threads
);

void longest_recovery(MatrixView values, std::int64_t* result, unsigned threads);

void max_drawdown(
    MatrixView values,
    VectorView<std::int64_t> dates,
    double* result,
    std::vector<std::string>& result_dates,
    std::int64_t* recovery,
    unsigned threads
);

void largest_streak(
    MatrixView values,
    VectorView<std::int64_t> dates,
    bool positive,
    double* result,
    std::int64_t* periods,
    std::vector<std::string>& starts,
    std::vector<std::string>& ends,
    unsigned threads
);

void longest_streak(
    MatrixView values,
    VectorView<std::int64_t> dates,
    bool positive,
    double* result,
    std::int64_t* periods,
    std::vector<std::string>& starts,
    std::vector<std::string>& ends,
    unsigned threads
);

void rolling_gain(
    MatrixView values,
    VectorView<std::int64_t> starts,
    VectorView<std::int64_t> ends,
    VectorView<std::int64_t> dates,
    int months,
    const std::array<double*, 9>& outputs,
    unsigned threads
);

}  // namespace calmetrics_engine
