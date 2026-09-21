#include "calmetrics_engine/scheduler.hpp"

#include <limits>
#include <stdexcept>

namespace calmetrics_engine::native {
namespace {
void require_count(std::size_t count) {
  if (!count || count > 1024)
    throw std::invalid_argument("CPU budget must be in 1..1024");
}
} // namespace

CpuBudget::CpuBudget(std::size_t total) : total_(total), available_(total) {
  require_count(total);
}

bool CpuBudget::acquire(std::size_t count, SchedulerDeadline deadline) {
  if (!count)
    throw std::invalid_argument("CPU request must be positive");
  std::unique_lock<std::mutex> lock(mutex_);
  if (count > total_)
    throw std::invalid_argument("CPU request exceeds scheduler capacity");
  const auto ready = [&] { return closing_ || available_ >= count; };
  // A maximum steady-clock deadline can overflow when older libstdc++
  // converts it to the realtime clock, spinning while holding this mutex.
  // Infinite admission must use the untimed wait so releases can acquire it.
  if (deadline == SchedulerDeadline::max())
    cv_.wait(lock, ready);
  else if (!cv_.wait_until(lock, deadline, ready))
    return false;
  if (closing_)
    throw std::runtime_error("native scheduler is closed");
  available_ -= count;
  peak_ = std::max(peak_, total_ - available_);
  return true;
}

void CpuBudget::release(std::size_t count) noexcept {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    available_ += count;
  }
  cv_.notify_all();
}

void CpuBudget::close_admission() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    closing_ = true;
  }
  cv_.notify_all();
}

void CpuBudget::wait_idle() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [&] { return available_ == total_; });
}

void CpuBudget::grow(std::size_t total) {
  require_count(total);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closing_)
      throw std::runtime_error("native scheduler is closed");
    if (total <= total_)
      return;
    available_ += total - total_;
    total_ = total;
  }
  cv_.notify_all();
}

std::size_t CpuBudget::total() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return total_;
}

std::size_t CpuBudget::peak_active() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return peak_;
}

ThreadPool::ThreadPool(std::size_t workers) { grow(workers); }
ThreadPool::~ThreadPool() { close(); }

void ThreadPool::enqueue(std::function<void()> function) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_)
      throw std::runtime_error("native thread pool closed");
    tasks_.push(std::move(function));
  }
  cv_.notify_one();
}

void ThreadPool::work() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return stopping_ || !tasks_.empty(); });
      if (tasks_.empty()) {
        if (stopping_)
          return;
        continue;
      }
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    task();
  }
}

void ThreadPool::grow(std::size_t workers) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (stopping_)
    throw std::runtime_error("native thread pool closed");
  while (workers_.size() < workers)
    workers_.emplace_back([this] { work(); });
}

void ThreadPool::close() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_)
      return;
    stopping_ = true;
  }
  cv_.notify_all();
  for (auto &worker : workers_)
    if (worker.joinable())
      worker.join();
}

std::size_t ThreadPool::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return workers_.size();
}

std::size_t NativeScheduler::detected_cpu() {
  const auto detected = std::thread::hardware_concurrency();
  return std::min<std::size_t>(
      1024, std::max<std::size_t>(1, detected ? detected : 1));
}

NativeScheduler::NativeScheduler() : budget_(detected_cpu()), threads_(0) {}

NativeScheduler &NativeScheduler::instance() {
  static NativeScheduler scheduler;
  return scheduler;
}

void NativeScheduler::ensure_capacity(std::size_t total) {
  require_count(total);
  std::lock_guard<std::mutex> lock(growth_mutex_);
  if (total > budget_.total())
    budget_.grow(total);
}

void NativeScheduler::ensure_threads(std::size_t total_concurrency) {
  require_count(total_concurrency);
  std::lock_guard<std::mutex> lock(growth_mutex_);
  if (total_concurrency > budget_.total())
    throw std::invalid_argument(
        "thread concurrency exceeds scheduler capacity");
  threads_.grow(total_concurrency > 1 ? total_concurrency - 1 : 0);
}

void NativeScheduler::parallel_for(
    std::size_t items, unsigned requested,
    const std::function<void(std::size_t, std::size_t)> &worker) {
  if (!items)
    return;
  const auto available = detected_cpu();
  auto desired = requested == 0
                     ? available
                     : std::min(static_cast<std::size_t>(requested), available);
  require_count(desired);
  desired = std::min(desired, items);
  ensure_threads(desired);

  if (!budget_.acquire(desired))
    throw std::runtime_error("native CPU admission failed");
  struct Release {
    CpuBudget &budget;
    std::size_t count;
    ~Release() { budget.release(count); }
  } release{budget_, desired};

  if (desired == 1) {
    worker(0, 1);
    return;
  }

  std::vector<std::future<void>> futures;
  futures.reserve(desired - 1);
  std::exception_ptr failure;
  try {
    for (std::size_t index = 1; index < desired; ++index)
      futures.push_back(
          threads_.submit([&, index] { worker(index, desired); }));
    worker(0, desired);
  } catch (...) {
    failure = std::current_exception();
  }
  for (auto &future : futures) {
    try {
      future.get();
    } catch (...) {
      if (!failure)
        failure = std::current_exception();
    }
  }
  if (failure)
    std::rethrow_exception(failure);
}

} // namespace calmetrics_engine::native
