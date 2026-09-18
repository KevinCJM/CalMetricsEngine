#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace calmetrics_engine {

// Workers own scratch storage and write disjoint output ranges.
// Every exit path joins started threads before propagating failures.
template <class Worker>
void parallel_columns(std::size_t columns, unsigned requested, Worker&& worker) {
    if (columns == 0) return;
    const unsigned detected = std::max(1u, std::thread::hardware_concurrency());
    const auto count = std::min(
        columns,
        static_cast<std::size_t>(requested == 0 ? std::min(detected, 256u) : requested)
    );
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
        for (std::size_t index = 1; index < count; ++index) {
            threads.emplace_back(invoke, index);
        }
    } catch (...) {
        for (auto& thread : threads) thread.join();
        throw;
    }

    invoke(0);
    for (auto& thread : threads) thread.join();
    if (failure) std::rethrow_exception(failure);
}

}  // namespace calmetrics_engine
