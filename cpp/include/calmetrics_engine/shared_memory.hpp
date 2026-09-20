#pragma once
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>

namespace calmetrics_engine::native {

// Mapping ownership is independent of Python. Live views retain
// shared_ptr<SharedRegion>.
class SharedRegion {
public:
  static std::shared_ptr<SharedRegion> create(std::size_t bytes);
  static std::shared_ptr<SharedRegion>
  attach(const std::string &name, std::size_t bytes, bool writable = false);
  ~SharedRegion() noexcept;
  SharedRegion(const SharedRegion &) = delete;
  SharedRegion &operator=(const SharedRegion &) = delete;
  void *data() const noexcept { return data_; }
  std::size_t size() const noexcept { return logical_bytes_; }
  bool writable() const noexcept { return writable_; }
  const std::string &name() const noexcept { return name_; }
  void make_readonly();
  void unlink();

private:
  SharedRegion() = default;
  void *data_ = nullptr;
#ifdef _WIN32
  void *handle_ = nullptr;
#endif
  std::size_t logical_bytes_ = 0, mapped_bytes_ = 0;
  std::string name_;
  bool creator_ = false, unlinked_ = false;
  bool writable_ = true;
  std::mutex mutex_;
};
struct SharedDescriptor {
  std::string name;
  std::size_t bytes = 0;
  std::size_t offset = 0;
};
} // namespace calmetrics_engine::native
