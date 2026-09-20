#pragma once

#include "calmetrics_engine/scheduler.hpp"

#include <cstddef>
#include <functional>
#include <utility>

namespace calmetrics_engine {

// Compatibility helper for the legacy column-oriented finance APIs.
//
// Concurrency is owned by the process-wide native C++ Scheduler. requested is
// only this request's upper bound; it never creates threads directly and cannot
// bypass global CPU admission.
template <class Worker>
void parallel_columns(std::size_t columns, unsigned requested,
                      Worker &&worker) {
  if (!columns)
    return;
  native::NativeScheduler::instance().parallel_for(
      columns, requested,
      std::function<void(std::size_t, std::size_t)>(
          std::forward<Worker>(worker)));
}

} // namespace calmetrics_engine
