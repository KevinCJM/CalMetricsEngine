#include "calmetrics_engine/native_process.hpp"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

namespace calmetrics_engine::native {
namespace {
constexpr std::size_t max_frame = 256 * 1024 * 1024;
constexpr std::uint64_t request_magic = 0x434d453300000001ull;
constexpr std::uint64_t response_magic = 0x434d455300000001ull;
using Bytes = std::vector<std::uint8_t>;
void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
struct Writer {
  Bytes bytes;
  void number(std::uint64_t x) {
    for (unsigned i = 0; i < 8; ++i)
      bytes.push_back(static_cast<std::uint8_t>(x >> (8 * i)));
  }
  void raw(const void *data, std::size_t n) {
    require(n <= max_frame && bytes.size() <= max_frame - n,
            "IPC frame too large; use shared-memory transport");
    if (n) {
      const auto *p = static_cast<const std::uint8_t *>(data);
      bytes.insert(bytes.end(), p, p + n);
    }
  }
  void blob(const void *data, std::size_t n) {
    number(n);
    raw(data, n);
  }
  void blob(const Bytes &data) { blob(data.data(), data.size()); }
  void string(const std::string &s) { blob(s.data(), s.size()); }
};
struct Reader {
  const Bytes &bytes;
  std::size_t position = 0;
  std::uint64_t number() {
    require(position <= bytes.size() && bytes.size() - position >= 8,
            "truncated native IPC");
    std::uint64_t x = 0;
    for (unsigned i = 0; i < 8; ++i)
      x |= static_cast<std::uint64_t>(bytes[position++]) << (8 * i);
    return x;
  }
  Bytes blob() {
    auto n = number();
    require(n <= bytes.size() - position, "invalid native IPC blob length");
    Bytes out(bytes.begin() + static_cast<std::ptrdiff_t>(position),
              bytes.begin() + static_cast<std::ptrdiff_t>(position + n));
    position += static_cast<std::size_t>(n);
    return out;
  }
  template <class T> std::vector<T> array(std::size_t count) {
    auto n = number();
    require(count <= max_frame / sizeof(T) && n == count * sizeof(T) &&
                n <= bytes.size() - position,
            "invalid native IPC array");
    std::vector<T> out(count);
    if (n)
      std::memcpy(out.data(), bytes.data() + position,
                  static_cast<std::size_t>(n));
    position += static_cast<std::size_t>(n);
    return out;
  }
  std::string string() {
    auto data = blob();
    return std::string(data.begin(), data.end());
  }
  void end() {
    require(position == bytes.size(), "unexpected native IPC trailing bytes");
  }
};
void put_descriptor(Writer &out, const SharedDescriptor &d) {
  out.string(d.name);
  out.number(d.bytes);
  out.number(d.offset);
}
SharedDescriptor get_descriptor(Reader &in) {
  SharedDescriptor d;
  d.name = in.string();
  d.bytes = static_cast<std::size_t>(in.number());
  d.offset = static_cast<std::size_t>(in.number());
  require(d.offset <= d.bytes, "invalid shared descriptor offset");
  return d;
}
SharedDescriptor descriptor(const std::shared_ptr<SharedRegion> &region) {
  return {region->name(), region->size(), 0};
}
std::size_t graph::Audit::*const audit_fields[] = {
    &graph::Audit::rows,
    &graph::Audit::nodes,
    &graph::Audit::max_window,
    &graph::Audit::numeric_arena_bytes,
    &graph::Audit::mask_arena_bytes,
    &graph::Audit::operator_workspace_capacity_bytes,
    &graph::Audit::input_copy_bytes,
    &graph::Audit::fused_scalar_calls,
    &graph::Audit::summary_source_scans,
    &graph::Audit::order_stat_sorts,
    &graph::Audit::order_scratch_capacity_bytes,
    &graph::Audit::algorithm_copy_bytes};
void write_audit(Writer &out, const graph::Audit &a) {
  for (auto member : audit_fields)
    out.number(a.*member);
}
graph::Audit read_audit(Reader &in) {
  graph::Audit a;
  for (auto member : audit_fields)
    a.*member = static_cast<std::size_t>(in.number());
  return a;
}

void check_cancel(Deadline deadline, const std::atomic<bool> &cancelled) {
  if (cancelled.load(std::memory_order_relaxed))
    throw Timeout("native worker request cancelled");
  check_deadline(deadline);
}

class Worker {
public:
  explicit Worker(const std::string &executable) { start(executable); }
  ~Worker() { stop(); }
  Worker(const Worker &) = delete;
  Worker &operator=(const Worker &) = delete;
  Bytes call(const Bytes &request, Deadline deadline,
             const std::atomic<bool> &cancelled) {
    Writer header;
    header.number(request.size());
    transfer(header.bytes.data(), header.bytes.size(), true, deadline,
             cancelled);
    transfer(const_cast<std::uint8_t *>(request.data()), request.size(), true,
             deadline, cancelled);
    Bytes size(8);
    transfer(size.data(), size.size(), false, deadline, cancelled);
    Reader count{size};
    auto n = count.number();
    require(n <= max_frame, "native worker response too large");
    Bytes response(static_cast<std::size_t>(n));
    transfer(response.data(), response.size(), false, deadline, cancelled);
    return response;
  }

private:
#ifdef _WIN32
  HANDLE channel_ = INVALID_HANDLE_VALUE, process_ = nullptr;
  static std::wstring wide(const std::string &s) {
    int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.c_str(),
                                    -1, nullptr, 0);
    if (!count)
      throw std::runtime_error("invalid native worker UTF-8 path");
    std::wstring out(static_cast<std::size_t>(count), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.c_str(), -1,
                        out.data(), count);
    out.pop_back();
    return out;
  }
  void start(const std::string &executable) {
    std::random_device random;
    const auto name = L"\\\\.\\pipe\\cme_" +
                      std::to_wstring(GetCurrentProcessId()) + L"_" +
                      std::to_wstring(random());
    channel_ = CreateNamedPipeW(name.c_str(),
                                PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED |
                                    FILE_FLAG_FIRST_PIPE_INSTANCE,
                                PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                                1, 65536, 65536, 0, nullptr);
    if (channel_ == INVALID_HANDLE_VALUE)
      throw std::runtime_error("CreateNamedPipe failed");
    HANDLE client = INVALID_HANDLE_VALUE;
    HANDLE event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    std::vector<std::uint8_t> attribute_storage;
    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof(startup);
    bool attributes_ready = false;
    try {
      OVERLAPPED connect{};
      connect.hEvent = event;
      ConnectNamedPipe(channel_, &connect);
      SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
      client = CreateFileW(name.c_str(), GENERIC_READ | GENERIC_WRITE, 0,
                           &security, OPEN_EXISTING, 0, nullptr);
      if (client == INVALID_HANDLE_VALUE)
        throw std::runtime_error("CreateFile native pipe failed");
      if (WaitForSingleObject(event, 5000) != WAIT_OBJECT_0)
        throw std::runtime_error("native pipe connect failed");
      SIZE_T bytes = 0;
      InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
      attribute_storage.resize(bytes);
      startup.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(
          attribute_storage.data());
      if (!InitializeProcThreadAttributeList(startup.lpAttributeList, 1, 0,
                                             &bytes))
        throw std::runtime_error("process attribute init failed");
      attributes_ready = true;
      if (!UpdateProcThreadAttribute(startup.lpAttributeList, 0,
                                     PROC_THREAD_ATTRIBUTE_HANDLE_LIST, &client,
                                     sizeof(client), nullptr, nullptr))
        throw std::runtime_error("process handle list failed");
      startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
      startup.StartupInfo.hStdInput = client;
      startup.StartupInfo.hStdOutput = client;
      startup.StartupInfo.hStdError = nullptr;
      auto application = wide(executable);
      auto command = L"\"" + application + L"\" --stdio";
      PROCESS_INFORMATION process{};
      if (!CreateProcessW(application.c_str(), command.data(), nullptr, nullptr,
                          TRUE, CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT,
                          nullptr, nullptr, &startup.StartupInfo, &process))
        throw std::runtime_error("CreateProcess native worker failed: " +
                                 std::to_string(GetLastError()));
      process_ = process.hProcess;
      CloseHandle(process.hThread);
      DeleteProcThreadAttributeList(startup.lpAttributeList);
      attributes_ready = false;
      CloseHandle(client);
      client = INVALID_HANDLE_VALUE;
      CloseHandle(event);
    } catch (...) {
      if (attributes_ready)
        DeleteProcThreadAttributeList(startup.lpAttributeList);
      if (client != INVALID_HANDLE_VALUE)
        CloseHandle(client);
      CancelIoEx(channel_, nullptr);
      if (event)
        CloseHandle(event);
      stop();
      throw;
    }
  }
  void transfer(std::uint8_t *data, std::size_t bytes, bool sending,
                Deadline deadline, const std::atomic<bool> &cancelled) {
    while (bytes) {
      check_cancel(deadline, cancelled);
      HANDLE event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
      OVERLAPPED overlapped{};
      overlapped.hEvent = event;
      DWORD done = 0;
      const DWORD count =
          static_cast<DWORD>(std::min<std::size_t>(bytes, 1 << 20));
      BOOL ok = sending ? WriteFile(channel_, data, count, &done, &overlapped)
                        : ReadFile(channel_, data, count, &done, &overlapped);
      const auto error = GetLastError();
      if (!ok && error != ERROR_IO_PENDING) {
        CloseHandle(event);
        throw std::runtime_error("native worker pipe failed");
      }
      try {
        if (!ok) {
          while (WaitForSingleObject(event, 20) == WAIT_TIMEOUT)
            check_cancel(deadline, cancelled);
          if (!GetOverlappedResult(channel_, &overlapped, &done, FALSE))
            throw std::runtime_error("native worker pipe disconnected");
        }
      } catch (...) {
        CancelIoEx(channel_, &overlapped);
        WaitForSingleObject(event, INFINITE);
        CloseHandle(event);
        throw;
      }
      CloseHandle(event);
      require(done > 0, "native worker closed transport");
      data += done;
      bytes -= done;
    }
  }
  void stop() noexcept {
    if (channel_ != INVALID_HANDLE_VALUE) {
      CloseHandle(channel_);
      channel_ = INVALID_HANDLE_VALUE;
    }
    if (process_) {
      if (WaitForSingleObject(process_, 100) != WAIT_OBJECT_0) {
        TerminateProcess(process_, 124);
        WaitForSingleObject(process_, INFINITE);
      }
      CloseHandle(process_);
      process_ = nullptr;
    }
  }
#else
  int socket_ = -1;
  pid_t pid_ = -1;
  void start(const std::string &executable) {
    int sockets[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0)
      throw std::runtime_error("socketpair failed");
    for (int fd : sockets)
      fcntl(fd, F_SETFD, FD_CLOEXEC);
#ifdef __APPLE__
    int one = 1;
    setsockopt(sockets[0], SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    posix_spawn_file_actions_t actions;
    int error = posix_spawn_file_actions_init(&actions);
    if (error) {
      ::close(sockets[0]);
      ::close(sockets[1]);
      throw std::runtime_error("spawn actions init failed");
    }
    error =
        posix_spawn_file_actions_adddup2(&actions, sockets[1], STDIN_FILENO);
    if (!error)
      error =
          posix_spawn_file_actions_adddup2(&actions, sockets[1], STDOUT_FILENO);
    if (!error)
      error = posix_spawn_file_actions_addclose(&actions, sockets[0]);
    if (!error && sockets[1] != STDIN_FILENO && sockets[1] != STDOUT_FILENO)
      error = posix_spawn_file_actions_addclose(&actions, sockets[1]);
    char *arguments[] = {const_cast<char *>(executable.c_str()),
                         const_cast<char *>("--stdio"), nullptr};
    if (!error)
      error = posix_spawn(&pid_, executable.c_str(), &actions, nullptr,
                          arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    ::close(sockets[1]);
    if (error) {
      ::close(sockets[0]);
      pid_ = -1;
      throw std::runtime_error(std::string("posix_spawn native worker: ") +
                               std::strerror(error));
    }
    socket_ = sockets[0];
    fcntl(socket_, F_SETFL, fcntl(socket_, F_GETFL) | O_NONBLOCK);
  }
  void transfer(std::uint8_t *data, std::size_t bytes, bool sending,
                Deadline deadline, const std::atomic<bool> &cancelled) {
    while (bytes) {
      check_cancel(deadline, cancelled);
      pollfd descriptor{socket_, static_cast<short>(sending ? POLLOUT : POLLIN),
                        0};
      auto wait = 20;
      if (deadline != Deadline::max()) {
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline -
                                                                  Clock::now())
                .count();
        wait = static_cast<int>(
            std::max<std::int64_t>(1, std::min<std::int64_t>(wait, remaining)));
      }
      const int ready = poll(&descriptor, 1, wait);
      if (ready < 0 && errno == EINTR)
        continue;
      if (ready < 0)
        throw std::runtime_error("poll native worker failed");
      if (!ready)
        continue;
      ssize_t transferred;
      if (sending) {
#ifdef MSG_NOSIGNAL
        transferred = send(socket_, data, bytes, MSG_NOSIGNAL);
#else
        transferred = send(socket_, data, bytes, 0);
#endif
      } else
        transferred = recv(socket_, data, bytes, 0);
      if (transferred < 0 && (errno == EAGAIN || errno == EINTR))
        continue;
      if (transferred <= 0)
        throw std::runtime_error("native worker exited or closed IPC");
      data += transferred;
      bytes -= static_cast<std::size_t>(transferred);
    }
  }
  void stop() noexcept {
    if (socket_ >= 0) {
      ::close(socket_);
      socket_ = -1;
    }
    if (pid_ > 0) {
      int status = 0;
      pid_t result;
      do {
        result = waitpid(pid_, &status, WNOHANG);
      } while (result < 0 && errno == EINTR);
      if (result == 0) {
        kill(pid_, SIGKILL);
        do {
          result = waitpid(pid_, &status, 0);
        } while (result < 0 && errno == EINTR);
      }
      pid_ = -1;
    }
  }
#endif
};

Bytes execute_request(const Bytes &bytes, Bytes &cached_bytes,
                      graph::Program &cached_program) {
  Writer response;
  response.number(response_magic);
  try {
    Reader in{bytes};
    require(in.number() == request_magic,
            "unsupported native worker request version");
    auto program = in.blob();
    if (program != cached_bytes) {
      cached_program = compiler::decode_program(program);
      cached_bytes = std::move(program);
    }
    const bool shared = in.number() != 0;
    auto params = in.array<double>(cached_program.parameter_count);
    const auto input_count = in.number();
    require(input_count == cached_program.input_count,
            "native worker input count mismatch");
    std::vector<ops::Value> inputs;
    std::vector<std::vector<double>> storage;
    std::vector<std::shared_ptr<SharedRegion>> mappings;
    storage.reserve(static_cast<std::size_t>(input_count));
    for (std::size_t i = 0; i < input_count; ++i) {
      const auto count = static_cast<std::size_t>(in.number());
      ops::Value value;
      value.kind = ops::Kind::number;
      value.shape = ops::vector_shape(count);
      value.stride[0] = 1;
      if (shared) {
        const auto d = get_descriptor(in);
        require(d.offset % alignof(double) == 0 &&
                    planner::checked_mul(count, 8) <= d.bytes - d.offset,
                "native input descriptor bounds");
        auto mapping = SharedRegion::attach(d.name, d.bytes);
        value.data =
            static_cast<const std::uint8_t *>(mapping->data()) + d.offset;
        mappings.push_back(std::move(mapping));
      } else {
        storage.push_back(in.array<double>(count));
        value.data = storage.back().data();
      }
      inputs.push_back(value);
    }
    const auto total_rows = static_cast<std::size_t>(in.number());
    const auto begin = static_cast<std::size_t>(in.number()),
               end = static_cast<std::size_t>(in.number());
    require(begin <= end && end <= total_rows, "native worker chunk bounds");
    std::vector<std::int64_t> starts_storage, ends_storage;
    std::vector<double> output_storage;
    const std::int64_t *starts = nullptr, *ends = nullptr;
    double *output = nullptr;
    if (shared) {
      auto a = get_descriptor(in), b = get_descriptor(in),
           c = get_descriptor(in);
      require(a.offset == 0 && b.offset == 0 && c.offset == 0,
              "native interval mapping offset must be zero");
      require(
          planner::checked_mul(total_rows, 8) <= a.bytes &&
              planner::checked_mul(total_rows, 8) <= b.bytes &&
              planner::checked_mul(
                  planner::checked_mul(total_rows, cached_program.roots.size()),
                  8) <= c.bytes,
          "native interval/output mapping bounds");
      auto sa = SharedRegion::attach(a.name, a.bytes),
           sb = SharedRegion::attach(b.name, b.bytes),
           so = SharedRegion::attach(c.name, c.bytes, true);
      starts = static_cast<const std::int64_t *>(sa->data()) + begin;
      ends = static_cast<const std::int64_t *>(sb->data()) + begin;
      output = static_cast<double *>(so->data()) +
               begin * cached_program.roots.size();
      mappings.push_back(sa);
      mappings.push_back(sb);
      mappings.push_back(so);
    } else {
      starts_storage = in.array<std::int64_t>(end - begin);
      ends_storage = in.array<std::int64_t>(end - begin);
      output_storage.resize(
          planner::checked_mul(end - begin, cached_program.roots.size()));
      starts = starts_storage.data();
      ends = ends_storage.data();
      output = output_storage.data();
    }
    in.end();
    auto audit = graph::execute(cached_program, inputs, params.data(),
                                params.size(), starts, ends, end - begin,
                                output, cached_program.roots.size());
    response.number(0);
    write_audit(response, audit);
    response.blob(output_storage.data(),
                  output_storage.size() * sizeof(double));
  } catch (const std::exception &error) {
    response.bytes.resize(8);
    response.number(1);
    response.string(error.what());
  }
  return response.bytes;
}
} // namespace

struct ProcessPool::Impl {
  std::string executable;
  std::vector<std::unique_ptr<Worker>> workers;
  std::vector<bool> busy;
  std::mutex mutex;
  std::condition_variable cv;
  bool closed = false;
  Impl(std::string path, std::size_t count)
      : executable(std::move(path)), workers(count), busy(count, false) {}
};
ProcessPool::ProcessPool(std::string executable, std::size_t capacity)
    : impl_(std::make_unique<Impl>(std::move(executable), capacity)) {}
ProcessPool::~ProcessPool() { close(); }
Bytes ProcessPool::transact(const Bytes &request, Deadline deadline,
                            bool disposable,
                            const std::atomic<bool> &cancelled) {
  check_cancel(deadline, cancelled);
  if (disposable) {
    Worker worker(impl_->executable);
    return worker.call(request, deadline, cancelled);
  }
  std::size_t slot = 0;
  {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    while (true) {
      require(!impl_->closed, "native process pool is closed");
      for (slot = 0; slot < impl_->busy.size(); ++slot)
        if (!impl_->busy[slot])
          break;
      if (slot != impl_->busy.size()) {
        impl_->busy[slot] = true;
        break;
      }
      impl_->cv.wait_for(lock, std::chrono::milliseconds(20));
      check_cancel(deadline, cancelled);
    }
  }
  try {
    if (!impl_->workers[slot])
      impl_->workers[slot] = std::make_unique<Worker>(impl_->executable);
    auto response = impl_->workers[slot]->call(request, deadline, cancelled);
    {
      std::lock_guard<std::mutex> lock(impl_->mutex);
      impl_->busy[slot] = false;
    }
    impl_->cv.notify_all();
    return response;
  } catch (...) {
    impl_->workers[slot].reset();
    {
      std::lock_guard<std::mutex> lock(impl_->mutex);
      impl_->busy[slot] = false;
    }
    impl_->cv.notify_all();
    throw;
  }
}
void ProcessPool::close() {
  std::unique_lock<std::mutex> lock(impl_->mutex);
  impl_->closed = true;
  impl_->cv.notify_all();
  impl_->cv.wait(lock, [&] {
    return std::none_of(impl_->busy.begin(), impl_->busy.end(),
                        [](bool x) { return x; });
  });
  impl_->workers.clear();
}

ProcessTransport::ProcessTransport(const planner::Plan &plan,
                                   const Batch &batch)
    : plan_(plan), batch_(batch),
      program_bytes_(compiler::encode_program(plan.graph->program)) {
  if (plan.use_shared_memory) {
    if (!batch.shared_inputs.empty()) {
      require(batch.shared_inputs.size() == batch.inputs.size(),
              "incomplete shared input descriptors");
      input_descriptors_ = batch.shared_inputs;
      for (const auto &d : input_descriptors_)
        shared_memory_bytes += d.bytes;
    } else
      for (const auto &input : batch.inputs) {
        const auto bytes = planner::checked_mul(input.size(), 8);
        auto region = SharedRegion::create(bytes);
        if (bytes)
          std::memcpy(region->data(), input.data, bytes);
        region->make_readonly();
        input_descriptors_.push_back(descriptor(region));
        owners_.push_back(region);
        shared_memory_bytes += bytes;
        boundary_copy_bytes += bytes;
      }
    starts_ = SharedRegion::create(batch.rows * 8);
    ends_ = SharedRegion::create(batch.rows * 8);
    output_ = SharedRegion::create(plan.estimated_output_bytes);
    if (batch.rows) {
      std::memcpy(starts_->data(), batch.starts, batch.rows * 8);
      std::memcpy(ends_->data(), batch.ends, batch.rows * 8);
    }
    starts_->make_readonly();
    ends_->make_readonly();
    boundary_copy_bytes += batch.rows * 16;
    shared_memory_bytes += batch.rows * 16 + plan.estimated_output_bytes;
    output_copy_bytes = plan.estimated_output_bytes;
  } else {
    boundary_copy_bytes =
        planner::checked_mul(plan.estimated_input_bytes, plan.chunks.size());
    output_copy_bytes = plan.estimated_output_bytes;
  }
}
Bytes ProcessTransport::request(planner::Chunk chunk) const {
  Writer out;
  out.number(request_magic);
  out.blob(program_bytes_);
  out.number(plan_.use_shared_memory);
  out.blob(batch_.parameters, batch_.parameter_count * sizeof(double));
  out.number(batch_.inputs.size());
  for (std::size_t i = 0; i < batch_.inputs.size(); ++i) {
    out.number(batch_.inputs[i].size());
    if (plan_.use_shared_memory)
      put_descriptor(out, input_descriptors_[i]);
    else
      out.blob(batch_.inputs[i].data, batch_.inputs[i].size() * 8);
  }
  out.number(batch_.rows);
  out.number(chunk.begin);
  out.number(chunk.end);
  if (plan_.use_shared_memory) {
    put_descriptor(out, descriptor(starts_));
    put_descriptor(out, descriptor(ends_));
    put_descriptor(out, descriptor(output_));
  } else {
    out.blob(batch_.starts + chunk.begin, (chunk.end - chunk.begin) * 8);
    out.blob(batch_.ends + chunk.begin, (chunk.end - chunk.begin) * 8);
  }
  return out.bytes;
}
graph::Audit ProcessTransport::response(const Bytes &bytes,
                                        planner::Chunk chunk, double *output) {
  Reader in{bytes};
  require(in.number() == response_magic,
          "unsupported native worker response version");
  if (in.number())
    throw std::runtime_error("native worker: " + in.string());
  auto audit = read_audit(in);
  const auto count =
      plan_.use_shared_memory
          ? 0
          : (chunk.end - chunk.begin) * plan_.graph->program.roots.size();
  auto values = in.array<double>(count);
  in.end();
  require(audit.rows == chunk.end - chunk.begin,
          "native worker result shape mismatch");
  if (count)
    std::memcpy(output + chunk.begin * plan_.graph->program.roots.size(),
                values.data(), count * 8);
  return audit;
}
void ProcessTransport::finish(double *output) {
  if (plan_.use_shared_memory && plan_.estimated_output_bytes)
    std::memcpy(output, output_->data(), plan_.estimated_output_bytes);
}

int worker_main() {
#ifdef _WIN32
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
#endif
  Bytes cached_bytes;
  graph::Program cached_program;
  while (true) {
    Bytes size(8);
    std::cin.read(reinterpret_cast<char *>(size.data()), 8);
    if (std::cin.gcount() == 0)
      return 0;
    if (std::cin.gcount() != 8)
      return 2;
    Reader header{size};
    auto count = header.number();
    if (count > max_frame)
      return 2;
    Bytes request(static_cast<std::size_t>(count));
    std::cin.read(reinterpret_cast<char *>(request.data()),
                  static_cast<std::streamsize>(count));
    if (!std::cin)
      return 2;
    auto response = execute_request(request, cached_bytes, cached_program);
    Writer out;
    out.number(response.size());
    std::cout.write(reinterpret_cast<const char *>(out.bytes.data()), 8);
    std::cout.write(reinterpret_cast<const char *>(response.data()),
                    static_cast<std::streamsize>(response.size()));
    std::cout.flush();
    if (!std::cout)
      return 2;
  }
}
} // namespace calmetrics_engine::native
