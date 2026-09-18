#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace my_ctools {
inline constexpr double nan = std::numeric_limits<double>::quiet_NaN();
inline constexpr std::int64_t not_recovered = 1'000'000;

struct MatrixView {
    const double* data;
    std::size_t rows;
    std::size_t cols;
    double operator()(std::size_t row, std::size_t col) const noexcept {
        return data[row * cols + col];
    }
};

// Each worker owns its scratch memory and disjoint output columns.
// Every exit path joins started threads; C++ exceptions reach Python safely.
template <class Worker>
void parallel_columns(std::size_t columns, unsigned requested, Worker&& worker) {
    if (columns == 0) return;
    const unsigned detected = std::max(1u, std::thread::hardware_concurrency());
    const auto count = std::min(columns, static_cast<std::size_t>(
        requested == 0 ? std::min(detected, 256u) : requested));
    if (count == 1) {
        worker(0, 1);
        return;
    }
    std::exception_ptr failure;
    std::mutex failure_mutex;
    auto invoke = [&](std::size_t first) {
        try {
            worker(first, count);
        } catch (...) {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure) failure = std::current_exception();
        }
    };
    std::vector<std::thread> threads;
    threads.reserve(count - 1);
    try {
        for (std::size_t i = 1; i < count; ++i) threads.emplace_back(invoke, i);
    } catch (...) {
        for (auto& thread : threads) thread.join();
        throw;
    }
    invoke(0);
    for (auto& thread : threads) thread.join();
    if (failure) std::rethrow_exception(failure);
}

inline double nan_mean(const std::vector<double>& values) {
    double sum = 0.0;
    std::size_t count = 0;
    for (double value : values) {
        if (!std::isnan(value)) { sum += value; ++count; }
    }
    return count ? sum / static_cast<double>(count) : nan;
}

// Mutates only caller-owned scratch storage, never a Python input.
inline double nan_median(std::vector<double>& values) {
    values.erase(std::remove_if(values.begin(), values.end(),
                               [](double value) { return std::isnan(value); }), values.end());
    if (values.empty()) return nan;
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    if (values.size() & 1u) return *middle;
    return 0.5 * (*middle + *std::max_element(values.begin(), middle));
}
}  // namespace my_ctools
