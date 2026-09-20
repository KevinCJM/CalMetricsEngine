#include "calmetrics_engine/shared_memory.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace calmetrics_engine::native {
namespace {
std::string unique_name() {
  static std::atomic<unsigned long> sequence{0};
  std::random_device random;
  std::ostringstream out;
#ifdef _WIN32
  out << "cme_" << std::hex << GetCurrentProcessId();
#else
  out << "cme_" << std::hex << getpid();
#endif
  out << '_' << random() << '_' << sequence.fetch_add(1);
  return out.str();
}
void valid_name(const std::string &name) {
  if (name.empty() || name.size() > 240 ||
      name.find_first_not_of(
          "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") !=
          std::string::npos)
    throw std::invalid_argument("invalid shared-memory name");
}
#ifdef _WIN32
std::wstring wide(const std::string &s) {
  return std::wstring(s.begin(), s.end());
}
std::runtime_error os_error(const char *what) {
  return std::runtime_error(std::string(what) +
                            ": win32=" + std::to_string(GetLastError()));
}
#else
std::runtime_error os_error(const char *what) {
  return std::runtime_error(std::string(what) + ": " + std::strerror(errno));
}
#endif
} // namespace

std::shared_ptr<SharedRegion> SharedRegion::create(std::size_t bytes) {
  if (bytes > static_cast<std::size_t>(PTRDIFF_MAX))
    throw std::overflow_error("shared mapping size overflow");
  auto region = std::shared_ptr<SharedRegion>(new SharedRegion());
  region->logical_bytes_ = bytes;
  region->mapped_bytes_ = std::max<std::size_t>(1, bytes);
  for (int attempt = 0; attempt < 8; ++attempt) {
    region->name_ = unique_name();
#ifdef _WIN32
    const auto n = static_cast<unsigned long long>(region->mapped_bytes_);
    HANDLE handle =
        CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                           static_cast<DWORD>(n >> 32), static_cast<DWORD>(n),
                           wide(region->name_).c_str());
    if (!handle)
      throw os_error("CreateFileMapping");
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
      CloseHandle(handle);
      continue;
    }
    region->handle_ = handle;
    region->creator_ = true;
    region->data_ =
        MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, region->mapped_bytes_);
    if (!region->data_)
      throw os_error("MapViewOfFile");
#else
    const auto name = "/" + region->name_;
    int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
      if (errno == EEXIST)
        continue;
      throw os_error("shm_open create");
    }
    region->creator_ = true;
    if (ftruncate(fd, static_cast<off_t>(region->mapped_bytes_)) != 0) {
      auto error = os_error("ftruncate");
      ::close(fd);
      throw error;
    }
    void *data = mmap(nullptr, region->mapped_bytes_, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, 0);
    const int saved = errno;
    ::close(fd);
    errno = saved;
    if (data == MAP_FAILED)
      throw os_error("mmap create");
    region->data_ = data;
#endif
    return region;
  }
  throw std::runtime_error("could not allocate unique shared-memory name");
}
std::shared_ptr<SharedRegion> SharedRegion::attach(const std::string &name,
                                                   std::size_t bytes,
                                                   bool writable) {
  valid_name(name);
  if (bytes > static_cast<std::size_t>(PTRDIFF_MAX))
    throw std::overflow_error("shared mapping size overflow");
  auto region = std::shared_ptr<SharedRegion>(new SharedRegion());
  region->name_ = name;
  region->writable_ = writable;
  region->logical_bytes_ = bytes;
  region->mapped_bytes_ = std::max<std::size_t>(bytes, 1);
#ifdef _WIN32
  HANDLE handle =
      OpenFileMappingW(writable ? FILE_MAP_ALL_ACCESS : FILE_MAP_READ, FALSE,
                       wide(name).c_str());
  if (!handle)
    throw os_error("OpenFileMapping");
  region->handle_ = handle;
  region->data_ =
      MapViewOfFile(handle, writable ? FILE_MAP_ALL_ACCESS : FILE_MAP_READ, 0,
                    0, region->mapped_bytes_);
  if (!region->data_)
    throw os_error("MapViewOfFile attach");
#else
  int fd = shm_open(("/" + name).c_str(), writable ? O_RDWR : O_RDONLY, 0600);
  if (fd < 0)
    throw os_error("shm_open attach");
  struct stat info {};
  if (fstat(fd, &info) != 0 || info.st_size < 0 ||
      static_cast<std::uint64_t>(info.st_size) < region->mapped_bytes_) {
    ::close(fd);
    throw std::invalid_argument("shared descriptor exceeds mapping size");
  }
  void *data = mmap(nullptr, region->mapped_bytes_,
                    PROT_READ | (writable ? PROT_WRITE : 0), MAP_SHARED, fd, 0);
  const int saved = errno;
  ::close(fd);
  errno = saved;
  if (data == MAP_FAILED)
    throw os_error("mmap attach");
  region->data_ = data;
#endif
  return region;
}

void SharedRegion::make_readonly() {
  std::lock_guard<std::mutex> guard(mutex_);
  if (!writable_)
    return;
#ifdef _WIN32
  DWORD old_protection = 0;
  if (!VirtualProtect(data_, mapped_bytes_, PAGE_READONLY, &old_protection))
    throw os_error("VirtualProtect readonly");
#else
  if (mprotect(data_, mapped_bytes_, PROT_READ) != 0)
    throw os_error("mprotect readonly");
#endif
  writable_ = false;
}

void SharedRegion::unlink() {
  std::lock_guard<std::mutex> guard(mutex_);
  if (!creator_ || unlinked_)
    return;
#ifndef _WIN32
  if (shm_unlink(("/" + name_).c_str()) != 0 && errno != ENOENT)
    throw os_error("shm_unlink");
#endif
  unlinked_ = true;
}
SharedRegion::~SharedRegion() noexcept {
  // Creator-side cleanup still occurs if construction or a child task fails.
  try {
    unlink();
  } catch (...) { /* destructor cannot report; explicit unlink reports errors */
  }
#ifdef _WIN32
  if (data_)
    UnmapViewOfFile(data_);
  if (handle_)
    CloseHandle(static_cast<HANDLE>(handle_));
#else
  if (data_)
    munmap(data_, mapped_bytes_);
#endif
}
} // namespace calmetrics_engine::native
