#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace calmetrics_engine::native {

using SchedulerClock = std::chrono::steady_clock;
using SchedulerDeadline = SchedulerClock::time_point;

class CpuBudget {
public:
  explicit CpuBudget(std::size_t total);
  bool acquire(std::size_t count,
               SchedulerDeadline deadline = SchedulerDeadline::max());
  void release(std::size_t count) noexcept;
  void close_admission();
  void wait_idle();
  void grow(std::size_t total);
  std::size_t total() const;
  std::size_t peak_active() const;

private:
  std::size_t total_, available_, peak_ = 0;
  bool closing_ = false;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
};

class ThreadPool {
public:
  explicit ThreadPool(std::size_t workers = 0);
  ~ThreadPool();

  template <class Function>
  auto submit(Function &&function)
      -> std::future<std::invoke_result_t<std::decay_t<Function>>> {
    using Result = std::invoke_result_t<std::decay_t<Function>>;
    auto task = std::make_shared<std::packaged_task<Result()>>(
        std::forward<Function>(function));
    auto future = task->get_future();
    enqueue([task] { (*task)(); });
    return future;
  }

  void grow(std::size_t workers);
  void close();
  std::size_t size() const;

private:
  void enqueue(std::function<void()> function);
  void work();

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool stopping_ = false;
};

class NativeScheduler {
public:
  static NativeScheduler &instance();

  NativeScheduler(const NativeScheduler &) = delete;
  NativeScheduler &operator=(const NativeScheduler &) = delete;

  void ensure_capacity(std::size_t total);
  void ensure_threads(std::size_t total_concurrency);
  CpuBudget &budget() noexcept { return budget_; }
  ThreadPool &threads() noexcept { return threads_; }
  std::size_t capacity() const { return budget_.total(); }

  void
  parallel_for(std::size_t items, unsigned requested,
               const std::function<void(std::size_t, std::size_t)> &worker);

private:
  NativeScheduler();
  static std::size_t detected_cpu();

  mutable std::mutex growth_mutex_;
  CpuBudget budget_;
  ThreadPool threads_;
};

} // namespace calmetrics_engine::native
