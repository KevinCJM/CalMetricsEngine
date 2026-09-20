#include "calmetrics_engine/operators.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#endif

namespace ops = calmetrics_engine::ops;

namespace {

struct Arguments {
  std::string nav_path;
  std::string starts_path;
  std::string ends_path;
  std::string output_path;
  std::size_t rows = 0;
  std::size_t history = 0;
  int repeats = 5;
  int threads = 1;
};

Arguments parse(int argc, char **argv) {
  Arguments args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc)
        throw std::runtime_error("missing argument for " + key);
      return argv[++i];
    };
    if (key == "--nav")
      args.nav_path = next();
    else if (key == "--starts")
      args.starts_path = next();
    else if (key == "--ends")
      args.ends_path = next();
    else if (key == "--output")
      args.output_path = next();
    else if (key == "--rows")
      args.rows = static_cast<std::size_t>(std::stoull(next()));
    else if (key == "--history")
      args.history = static_cast<std::size_t>(std::stoull(next()));
    else if (key == "--repeats")
      args.repeats = std::stoi(next());
    else if (key == "--threads")
      args.threads = std::stoi(next());
    else
      throw std::runtime_error("unknown argument: " + key);
  }
  if (args.nav_path.empty() || args.starts_path.empty() ||
      args.ends_path.empty() || args.output_path.empty() || args.rows == 0 ||
      args.history < 2 || args.repeats < 1 || args.threads < 1) {
    throw std::runtime_error("invalid benchmark arguments");
  }
  return args;
}

template <class T> std::vector<T> read_binary(const std::string &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot open " + path);
  const auto bytes = stream.tellg();
  if (bytes < 0 || static_cast<std::size_t>(bytes) % sizeof(T) != 0)
    throw std::runtime_error("invalid binary size");
  std::vector<T> values(static_cast<std::size_t>(bytes) / sizeof(T));
  stream.seekg(0);
  stream.read(reinterpret_cast<char *>(values.data()), bytes);
  if (!stream)
    throw std::runtime_error("cannot read " + path);
  return values;
}

template <class T>
void write_binary(const std::string &path, const std::vector<T> &values) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("cannot create " + path);
  stream.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!stream)
    throw std::runtime_error("cannot write " + path);
}

std::size_t peak_rss_bytes() {
#if defined(__APPLE__)
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  return static_cast<std::size_t>(usage.ru_maxrss);
#elif defined(__linux__)
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  return static_cast<std::size_t>(usage.ru_maxrss) * 1024u;
#else
  return 0;
#endif
}

ops::Value vector_value(const double *data, std::size_t size) {
  ops::Value value;
  value.kind = ops::Kind::number;
  value.shape = ops::vector_shape(size);
  value.data = data;
  value.stride[0] = 1;
  return value;
}

ops::Value mask_value(const std::uint8_t *data, std::size_t size) {
  ops::Value value;
  value.kind = ops::Kind::mask;
  value.shape = ops::vector_shape(size);
  value.data = data;
  value.stride[0] = 1;
  return value;
}

double scalar_call(ops::Op op, const std::array<ops::Value, 4> &args,
                   std::size_t count, ops::Workspace &workspace) {
  const auto &spec = ops::lookup(static_cast<std::uint16_t>(op));
  const auto plan = ops::prepare(spec, args.data(), count);
  ops::Output output;
  output.kind = plan.output_kind;
  output.shape = plan.output_shape;
  ops::Audit audit;
  ops::execute(plan, output, workspace, ops::Isa::automatic, audit);
  return output.scalar;
}

void array_call(ops::Op op, const std::array<ops::Value, 4> &args,
                std::size_t count, void *output_data, ops::Kind output_kind,
                ops::Workspace &workspace) {
  const auto &spec = ops::lookup(static_cast<std::uint16_t>(op));
  const auto plan = ops::prepare(spec, args.data(), count);
  ops::Output output;
  output.kind = output_kind;
  output.shape = plan.output_shape;
  output.data = output_data;
  ops::Audit audit;
  ops::execute(plan, output, workspace, ops::Isa::automatic, audit);
}

struct Worker {
  ops::Workspace workspace;
  std::vector<double> returns;
  std::vector<double> temp;
  std::vector<std::uint8_t> mask;

  explicit Worker(std::size_t history)
      : returns(history > 0 ? history - 1 : 0), temp(history), mask(history) {}
};

inline std::array<ops::Value, 4> one(const ops::Value &a) {
  return {a, {}, {}, {}};
}
inline std::array<ops::Value, 4> two(const ops::Value &a, const ops::Value &b) {
  return {a, b, {}, {}};
}

void compute_row(const std::vector<double> &nav, std::int64_t start,
                 std::int64_t end, double *output, Worker &worker) {
  if (start < 0 || end - start < 2 ||
      static_cast<std::size_t>(end) > nav.size())
    throw std::runtime_error("invalid interval");

  const auto nav_size = static_cast<std::size_t>(end - start);
  const auto return_size = nav_size - 1;
  const double *nav_ptr = nav.data() + start;
  for (std::size_t i = 0; i < return_size; ++i)
    worker.returns[i] = nav_ptr[i + 1] / nav_ptr[i] - 1.0;

  const auto nav_value = vector_value(nav_ptr, nav_size);
  const auto returns_value = vector_value(worker.returns.data(), return_size);
  const auto one_value = ops::Value::number(1.0);
  const auto zero_value = ops::Value::number(0.0);
  const auto p005 = ops::Value::number(0.05);
  const auto p252 = ops::Value::number(252.0);

  std::size_t metric = 0;
  output[metric++] = scalar_call(ops::Op::total_return, one(returns_value), 1,
                                 worker.workspace);
  output[metric++] = scalar_call(ops::Op::annualized_return,
                                 two(returns_value, p252), 2, worker.workspace);
  output[metric++] =
      scalar_call(ops::Op::mean, one(returns_value), 1, worker.workspace);
  output[metric++] = scalar_call(ops::Op::std, two(returns_value, one_value), 2,
                                 worker.workspace);
  output[metric++] =
      scalar_call(ops::Op::median, one(returns_value), 1, worker.workspace);
  output[metric++] =
      scalar_call(ops::Op::min_value, one(returns_value), 1, worker.workspace);
  output[metric++] =
      scalar_call(ops::Op::max_value, one(returns_value), 1, worker.workspace);
  output[metric++] = scalar_call(ops::Op::quantile, two(returns_value, p005), 2,
                                 worker.workspace);
  output[metric++] = scalar_call(ops::Op::mean_absolute_deviation,
                                 one(returns_value), 1, worker.workspace);
  output[metric++] = scalar_call(ops::Op::root_mean_square, one(returns_value),
                                 1, worker.workspace);

  array_call(ops::Op::drawdown_series, one(nav_value), 1, worker.temp.data(),
             ops::Kind::number, worker.workspace);
  const auto drawdown_value = vector_value(worker.temp.data(), nav_size);
  const double minimum_drawdown =
      scalar_call(ops::Op::min_value, one(drawdown_value), 1, worker.workspace);
  output[metric++] = -minimum_drawdown;

  array_call(ops::Op::new_high_mask, one(nav_value), 1, worker.mask.data(),
             ops::Kind::mask, worker.workspace);
  const auto high_mask = mask_value(worker.mask.data(), nav_size);
  const double highs =
      scalar_call(ops::Op::count_true, one(high_mask), 1, worker.workspace);
  output[metric++] = highs / static_cast<double>(nav_size);

  output[metric++] =
      scalar_call(ops::Op::linear_slope, one(nav_value), 1, worker.workspace);
  output[metric++] = scalar_call(ops::Op::linear_r_squared, one(nav_value), 1,
                                 worker.workspace);

  array_call(ops::Op::greater_than, two(returns_value, zero_value), 2,
             worker.mask.data(), ops::Kind::mask, worker.workspace);
  const auto positive_mask = mask_value(worker.mask.data(), return_size);
  output[metric++] =
      scalar_call(ops::Op::mean_where, two(returns_value, positive_mask), 2,
                  worker.workspace);

  array_call(ops::Op::less_than, two(returns_value, zero_value), 2,
             worker.mask.data(), ops::Kind::mask, worker.workspace);
  const auto negative_mask = mask_value(worker.mask.data(), return_size);
  output[metric++] =
      scalar_call(ops::Op::std_where, two(returns_value, negative_mask), 2,
                  worker.workspace);

  if (metric != 16)
    throw std::runtime_error("metric count mismatch");
}

void compute_all(const std::vector<double> &nav,
                 const std::vector<std::int64_t> &starts,
                 const std::vector<std::int64_t> &ends,
                 std::vector<double> &output,
                 std::vector<std::unique_ptr<Worker>> &workers,
                 int thread_count) {
  const auto rows = starts.size();
  auto run = [&](std::size_t worker_index) {
    const std::size_t begin =
        rows * worker_index / static_cast<std::size_t>(thread_count);
    const std::size_t finish =
        rows * (worker_index + 1) / static_cast<std::size_t>(thread_count);
    auto &worker = *workers[worker_index];
    for (std::size_t row = begin; row < finish; ++row)
      compute_row(nav, starts[row], ends[row], output.data() + row * 16,
                  worker);
  };

  if (thread_count == 1) {
    run(0);
    return;
  }
  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(thread_count) - 1);
  for (int index = 1; index < thread_count; ++index)
    threads.emplace_back(run, static_cast<std::size_t>(index));
  run(0);
  for (auto &thread : threads)
    thread.join();
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto args = parse(argc, argv);
    auto nav = read_binary<double>(args.nav_path);
    auto starts = read_binary<std::int64_t>(args.starts_path);
    auto ends = read_binary<std::int64_t>(args.ends_path);
    if (starts.size() != args.rows || ends.size() != args.rows)
      throw std::runtime_error("row count mismatch");
    if (nav.size() % args.history != 0)
      throw std::runtime_error("history shape mismatch");

    const auto threads =
        std::min<int>(args.threads, static_cast<int>(args.rows));
    std::vector<std::unique_ptr<Worker>> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    for (int i = 0; i < threads; ++i)
      workers.emplace_back(std::make_unique<Worker>(args.history));
    std::vector<double> output(args.rows * 16);

    compute_all(nav, starts, ends, output, workers, threads);
    const auto rss_before = peak_rss_bytes();

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(args.repeats));
    for (int repeat = 0; repeat < args.repeats; ++repeat) {
      const auto started = std::chrono::steady_clock::now();
      compute_all(nav, starts, ends, output, workers, threads);
      const auto finished = std::chrono::steady_clock::now();
      samples.push_back(
          std::chrono::duration<double, std::milli>(finished - started)
              .count());
    }
    std::sort(samples.begin(), samples.end());
    const auto median = samples[samples.size() / 2];
    const auto rss_after = peak_rss_bytes();
    std::size_t workspace_capacity = 0;
    for (const auto &worker : workers)
      workspace_capacity += worker->workspace.capacity_bytes();

    write_binary(args.output_path, output);
    const double checksum = std::accumulate(output.begin(), output.end(), 0.0);

    std::cout << std::setprecision(17) << "{"
              << "\"median_ms\":" << median << ",\"threads\":" << threads
              << ",\"rows\":" << args.rows << ",\"metrics\":16"
              << ",\"output_checksum\":" << checksum
              << ",\"workspace_capacity_bytes\":" << workspace_capacity
              << ",\"worker_buffer_bytes\":"
              << static_cast<std::size_t>(threads) *
                     ((args.history - 1) * sizeof(double) +
                      args.history * sizeof(double) +
                      args.history * sizeof(std::uint8_t))
              << ",\"peak_rss_before_bytes\":" << rss_before
              << ",\"peak_rss_after_bytes\":" << rss_after
              << ",\"peak_rss_growth_bytes\":"
              << (rss_after > rss_before ? rss_after - rss_before : 0) << "}"
              << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
