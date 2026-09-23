#pragma once
#include "calmetrics_engine/native_runtime.hpp"

namespace calmetrics_engine::native {
inline constexpr std::size_t max_process_frame_bytes = 256 * 1024 * 1024;
// Complete successful response, including audit fields and all blob headers.
std::size_t process_response_bytes(std::size_t output_bytes,
                                   std::size_t status_bytes,
                                   std::size_t result_slots);
class ProcessPool {
public:
  ProcessPool(std::string executable, std::size_t capacity);
  ~ProcessPool();
  std::vector<std::uint8_t> transact(const std::vector<std::uint8_t> &request,
                                     Deadline deadline, bool disposable,
                                     const std::atomic<bool> &cancelled);
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
class ProcessTransport {
public:
  ProcessTransport(const planner::Plan &plan, const Batch &batch);
  std::vector<std::uint8_t> request(planner::Chunk chunk) const;
  graph::Audit response(const std::vector<std::uint8_t> &bytes,
                        planner::Chunk chunk, void *out);
  void finish(void *out);
  std::shared_ptr<SharedRegion> output_owner() const { return output_; }
  std::size_t shared_memory_bytes = 0, boundary_copy_bytes = 0,
              output_copy_bytes = 0;

private:
  const planner::Plan &plan_;
  const Batch &batch_;
  std::vector<std::uint8_t> program_bytes_;
  std::vector<SharedDescriptor> input_descriptors_;
  std::vector<std::shared_ptr<SharedRegion>> owners_;
  std::shared_ptr<SharedRegion> starts_, ends_, output_;
};
// Standalone worker entry: uses only this C++ library and OS facilities, never
// Python.
int worker_main();
} // namespace calmetrics_engine::native
