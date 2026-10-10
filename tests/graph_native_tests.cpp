#include "matmul_shared_roots_native_cases.hpp"
#include "take_scatter_native_cases.hpp"
#include "matmul_scalar_chain_native_cases.hpp"
#include "state_decision_native_cases.hpp"
#include "calmetrics_engine/graph.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include <cfenv>
#include "affine_projection_native_cases.hpp"
#include "compact_path_native_cases.hpp"
#include "matmul_statistics_native_cases.hpp"
#include "matmul_root_native_cases.hpp"
#include "row_batch_native_cases.hpp"
#include "derived_row_native_cases.hpp"
#include "mixed_row_native_cases.hpp"
#include "masked_reduce_native_cases.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <iostream>
#include <vector>

namespace graph = calmetrics_engine::graph;
namespace ops = calmetrics_engine::ops;

namespace {

graph::Node input_node(std::uint16_t index) {
  graph::Node node;
  node.kind = graph::NodeKind::input;
  node.input_index = index;
  return node;
}

graph::Node constant_node(double value) {
  graph::Node node;
  node.kind = graph::NodeKind::constant;
  node.constant = value;
  return node;
}

graph::Node
operation(ops::Op op, std::initializer_list<std::uint32_t> parents,
          graph::StorageKind storage = graph::StorageKind::inline_value,
          std::uint32_t slot = 0) {
  graph::Node node;
  node.kind = graph::NodeKind::operation;
  node.opcode = static_cast<std::uint16_t>(op);
  node.parent_count = static_cast<std::uint8_t>(parents.size());
  std::size_t index = 0;
  for (auto parent : parents)
    node.parents[index++] = parent;
  node.storage = storage;
  node.slot = slot;
  return node;
}

ops::Value series(const std::vector<double> &values) {
  ops::Value value;
  value.kind = ops::Kind::number;
  value.shape = ops::vector_shape(values.size());
  value.data = values.data();
  value.stride[0] = 1;
  return value;
}

void require_close(double actual, double expected, double tolerance = 1e-12) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << "expected " << expected << " got " << actual << "\n";
    std::exit(1);
  }
}

} // namespace


void test_scan_packet_exception_and_lifetime() {
  std::vector<double> values(7 * 257, 1.0);
  ops::Value input;
  input.shape = ops::matrix_shape(7, 257);
  input.data = values.data();
  input.set_contiguous_strides();
  const auto saved = values;
  const auto check = [](bool ok, const char *message) {
    if (!ok) { std::cerr << "scan packet: " << message << "\n"; std::exit(1); }
  };
  for (double seed : {-1.0, 0.0, 1.0}) {
    const auto saved_env = std::fegetround();
    std::fenv_t environment;
    std::fegetenv(&environment);
    double totals[7]{987,987,987,987,987,987,987};
    unsigned errors = 0;
    std::size_t checkpoints = 0;
    std::feclearexcept(FE_ALL_EXCEPT);
    const bool used = ops::library_scan_reduce_quad(input, 2, seed, 0.0, 1.0,
        totals, errors, &checkpoints, [](void *p) { ++*static_cast<std::size_t *>(p); });
    const auto raised = std::fetestexcept(FE_ALL_EXCEPT);
    std::fesetenv(&environment);
    check(std::fegetround() == saved_env, "changed rounding mode");
#if defined(__APPLE__) && defined(__aarch64__)
    if (__builtin_available(macOS 15.0, *)) {
      const auto expected = std::getenv("CALMETRICS_TEST_EXPECT_EIGEN");
      if (expected) check(used == (std::strcmp(expected,"1") == 0), "unexpected actual dispatch");
    }
#endif
    check(values == saved, "changed borrowed input");
    check(totals[0] == 987 && totals[1] == 987 && totals[6] == 987, "overwrote output boundary");
    if (!used) {
      check(errors == 0 && checkpoints == 0, "inactive path changed state");
      for (auto total : totals) check(total == 987, "inactive path wrote output");
      continue;
    }
    check(checkpoints == 1, "wrong checkpoint count");
    check(!(raised & FE_DIVBYZERO), "masked zero denominator still divided");
    check(errors == (seed > 0 ? 0u : 3u), "changed original-node flags");
    for (std::size_t i = 2; i < 6; ++i)
      check(seed > 0 ? totals[i] == 0.0 : std::isinf(totals[i]) && totals[i] < 0,
            "changed ordered maximum");
    double cancelled[7]{987,987,987,987,987,987,987};
    unsigned untouched = 16;
    try {
      ops::library_scan_reduce_quad(input, 2, 1, 1, 1, cancelled, untouched,
          nullptr, [](void *) { throw std::runtime_error("scan-checkpoint"); });
      check(false, "lost checkpoint exception");
    } catch (const std::runtime_error &e) {
      check(std::strcmp(e.what(), "scan-checkpoint") == 0, "changed checkpoint exception");
    }
    for (auto total : cancelled) check(total == 987, "cancelled path wrote output");
    check(untouched == 16, "cancelled path changed flags");
  }
}

int main() {
  matmul_scalar_chain_native_cases::run();
  matmul_shared_roots_native_cases::run();
  take_scatter_native_cases::run();
  state_decision_native_cases::run();
  matmul_root_native_cases::run();
  masked_reduce_native_cases::run();
  test_scan_packet_exception_and_lifetime();
  row_batch_native_cases::run();
  derived_row_native_cases::run();
  mixed_row_native_cases::run();
  matmul_statistics_cases::run();
    matmul_statistics_cases::run_quantiles();
  {
    graph::Program program;
    program.input_count = 2;
    program.input_axes = {1, 1};
    program.numeric_slots = 2;
    program.output_kind = graph::OutputKind::typed;
    program.root_outputs = {{graph::OutputDType::float64, 1, graph::OutputOrder::c},
                            {graph::OutputDType::float64, 0, graph::OutputOrder::c}};
    program.nodes = {
        input_node(0), input_node(1), constant_node(0),
        operation(ops::Op::concatenate, {0, 1, 2}, graph::StorageKind::numeric, 0),
        constant_node(-1), operation(ops::Op::log, {4}),
        operation(ops::Op::concatenate, {3, 0, 2}, graph::StorageKind::numeric, 1)};
    program.roots = {6, 5};
    program.finalize();
    std::vector<double> numbers{1, 2};
    std::vector<std::int64_t> integers{1, 2};
    std::vector<std::uint8_t> masks{0, 1};
    const std::int64_t begin = 0, end = 0;
    for (auto kind : {ops::Kind::integer, ops::Kind::mask, ops::Kind::number}) {
      ops::Value value;
      value.kind = kind; value.shape = ops::vector_shape(2); value.set_contiguous_strides();
      value.data = kind == ops::Kind::integer ? static_cast<const void *>(integers.data()) :
          kind == ops::Kind::mask ? static_cast<const void *>(masks.data()) :
                                   static_cast<const void *>(numbers.data());
      const std::vector<ops::Value> inputs{value, value};
      const auto layout = graph::result_layout(program, inputs, &begin, &end, 1);
      std::vector<double> output((layout.bytes() + sizeof(double) - 1) / sizeof(double), 987.0);
      const auto run = [&](bool observed) {
        try {
          if (observed)
            graph::execute_observed(program, inputs, nullptr, 0, &begin, &end, 1,
                output.data(), 2, &layout, [](std::optional<ops::Op>, const ops::Value &, bool) {});
          else
            graph::execute(program, inputs, nullptr, 0, &begin, &end, 1, output.data(), 2, &layout);
        } catch (const ops::Error &error) { return std::string(error.what()); }
        return std::string("OK");
      };
      const auto canonical = run(true);
      const auto canonical_output = output;
      std::fill(output.begin(), output.end(), 987.0);
      const auto actual = run(false);
      const auto expected = kind == ops::Kind::number ? "DOMAIN_ERROR" : "GRAPH_STORAGE_KIND";
      // Both entries initialize failure output slots. Preserve those original
      // bytes; a failed call does not promise to leave caller output untouched.
      if (canonical != expected || actual != canonical ||
          std::memcmp(output.data(), canonical_output.data(), output.size() * sizeof(double)) != 0) {
        std::cerr << "concat native first error: " << canonical << " / " << actual << '\n';
        return 1;
      }
    }
  }
  {
    graph::Program program;
    program.input_count = 1;
    program.nodes = {
        input_node(0),
        operation(ops::Op::mean, {0}),
        constant_node(1.0),
        operation(ops::Op::std, {0, 2}),
        operation(ops::Op::divide, {1, 3}),
    };
    program.roots = {1, 3, 4};
    program.validate();

    std::vector<double> values{1, 2, 3, 4, 5, 10, 12, 14};
    std::vector<ops::Value> inputs{series(values)};
    std::int64_t starts[]{0, 5};
    std::int64_t ends[]{5, 8};
    double output[6]{};
    const auto audit =
        graph::execute(program, inputs, nullptr, 0, starts, ends, 2, output, 3);

    require_close(output[0], 3.0);
    require_close(output[1], std::sqrt(2.5));
    require_close(output[2], 3.0 / std::sqrt(2.5));
    require_close(output[3], 12.0);
    require_close(output[4], 2.0);
    require_close(output[5], 6.0);
    if (audit.input_copy_bytes != 0 || audit.rows != 2 || audit.nodes != 5)
      return 2;
  }

  {
    graph::Program program;
    program.input_count = 1;
    program.numeric_slots = 2;
    program.nodes = {
        input_node(0),
        constant_node(1.0),
        operation(ops::Op::difference, {0, 1}, graph::StorageKind::numeric, 0),
        operation(ops::Op::lag, {0, 1}),
        operation(ops::Op::divide, {2, 3}, graph::StorageKind::numeric, 1),
        operation(ops::Op::mean, {4}),
    };
    program.roots = {5};
    program.validate();

    std::vector<double> nav{1.0, 1.1, 1.05, 1.2, 1.3};
    std::vector<ops::Value> inputs{series(nav)};
    std::int64_t starts[]{0};
    std::int64_t ends[]{5};
    double output[1]{};
    const auto audit =
        graph::execute(program, inputs, nullptr, 0, starts, ends, 1, output, 1);

    double expected = 0.0;
    for (std::size_t index = 0; index + 1 < nav.size(); ++index)
      expected += (nav[index + 1] - nav[index]) / nav[index];
    expected /= static_cast<double>(nav.size() - 1);
    require_close(output[0], expected);
    if (audit.numeric_arena_bytes < 2 * nav.size() * sizeof(double))
      return 3;
  }

  {
    constexpr std::size_t n = 128, columns1 = 2, columns2 = 3;
    constexpr double sentinel = -912.125;
    const auto check_pair = [](bool ok, const char *message) {
      if (!ok) {
        std::cerr << "solve pair failure buffers: " << message << '\n';
        std::exit(1);
      }
    };
    graph::Program program;
    program.input_count = 3;
    program.input_axes = {1, 1, 1};
    program.numeric_slots = 2;
    program.output_kind = graph::OutputKind::typed;
    program.root_outputs = {
        {graph::OutputDType::float64, 2, graph::OutputOrder::c},
        {graph::OutputDType::float64, 2, graph::OutputOrder::c}};
    program.nodes = {
        input_node(0), input_node(1),
        operation(ops::Op::solve_multi, {0, 1}, graph::StorageKind::numeric, 0),
        input_node(2), // Ordinary compiler order: B2 is bound between solves.
        operation(ops::Op::solve_multi, {0, 3}, graph::StorageKind::numeric, 1)};
    program.roots = {2, 4};
    program.finalize();
    const std::int64_t start = 0, end = 0;
    const auto matrix = [&](const std::vector<double> &data, std::size_t columns) {
      ops::Value value;
      value.kind = ops::Kind::number;
      value.shape = ops::matrix_shape(n, columns);
      value.data = data.data();
      value.set_contiguous_strides();
      return value;
    };

    for (const bool second_backsolve_overflow : {false, true}) {
      std::vector<double> a(n * n, 0.0), b1(n * columns1), b2(n * columns2);
      for (std::size_t i = 0; i < n; ++i) {
        a[i * n + i] = 1.0;
        b1[i * columns1] = 3.0;
        b1[i * columns1 + 1] = 7.0;
        for (std::size_t j = 0; j < columns2; ++j)
          b2[i * columns2 + j] = 2.0 * (j + 1);
      }
      if (!second_backsolve_overflow) {
        a[(n - 1) * n + n - 1] = 0.0;
        b2[(n / 2) * columns2 + 1] = std::numeric_limits<double>::quiet_NaN();
      } else {
        // All elimination inputs/intermediates are finite. Back substitution
        // finishes the last row, divides column 0 of the previous row, then
        // overflows at column 1, leaving column 2 undivided.
        a[(n - 2) * n + n - 2] = 0.5;
        b2[(n - 2) * columns2] = 6.0;
        b2[(n - 2) * columns2 + 1] = std::numeric_limits<double>::max();
        b2[(n - 2) * columns2 + 2] = 10.0;
        for (std::size_t j = 0; j < columns2; ++j)
          b2[(n - 1) * columns2 + j] = 8.0 + 4.0 * j;
      }
      const auto saved_a = a, saved_b1 = b1, saved_b2 = b2;
      const std::vector<ops::Value> inputs{matrix(a, n), matrix(b1, columns1), matrix(b2, columns2)};
      const auto layout = graph::result_layout(program, inputs, &start, &end, 1);
      const auto storage_size = (layout.bytes() + sizeof(double) - 1) / sizeof(double) + 2;
      std::vector<double> actual(storage_size, sentinel), canonical(storage_size, sentinel);
      const auto run = [&](std::vector<double> &storage, bool observed) {
        try {
          if (observed) {
            graph::execute_observed(program, inputs, nullptr, 0, &start, &end, 1,
                storage.data() + 1, 2, &layout,
                [](std::optional<ops::Op>, const ops::Value &, bool) {});
          } else {
            graph::execute(program, inputs, nullptr, 0, &start, &end, 1,
                storage.data() + 1, 2, &layout);
          }
        } catch (const ops::Error &error) {
          return std::string(error.what());
        }
        return std::string("NO_ERROR");
      };
      // Observer mode executes the original nodes separately; compare every
      // caller-owned result byte, including the incomplete second result.
      const auto expected_error = run(canonical, true);
      const auto actual_error = run(actual, false);
      check_pair(expected_error == (second_backsolve_overflow ? "NON_FINITE_RESULT" : "SINGULAR_MATRIX"),
                 "fixture did not reach its intended failure");
      check_pair(actual_error == expected_error, "error priority changed");
      check_pair(std::memcmp(actual.data() + 1, canonical.data() + 1, layout.bytes()) == 0,
                 "caller-owned failure output differs from canonical replay");
      check_pair(actual.front() == sentinel && actual.back() == sentinel &&
                 canonical.front() == sentinel && canonical.back() == sentinel,
                 "output canary overwritten");
      check_pair(std::memcmp(a.data(), saved_a.data(), a.size() * sizeof(double)) == 0 &&
                 std::memcmp(b1.data(), saved_b1.data(), b1.size() * sizeof(double)) == 0 &&
                 std::memcmp(b2.data(), saved_b2.data(), b2.size() * sizeof(double)) == 0,
                 "input changed");
      const auto *bytes = reinterpret_cast<const unsigned char *>(actual.data() + 1);
      const auto *first = reinterpret_cast<const double *>(bytes + layout.slots[0].byte_offset);
      const auto *second = reinterpret_cast<const double *>(bytes + layout.slots[1].byte_offset);
      if (!second_backsolve_overflow) {
        for (std::size_t i = 0; i < n * columns2; ++i)
          check_pair(second[i] == 0.0 && !std::signbit(second[i]),
                     "unexecuted second root must remain positive zero");
      } else {
        for (std::size_t i = 0; i < n; ++i) {
          const double factor = i == n - 2 ? 2.0 : 1.0;
          check_pair(first[i * columns1] == 3.0 * factor &&
                     first[i * columns1 + 1] == 7.0 * factor,
                     "first root not fully completed before second failure");
        }
        check_pair(second[(n - 2) * columns2] == 12.0 &&
                   std::isinf(second[(n - 2) * columns2 + 1]) &&
                   second[(n - 2) * columns2 + 1] > 0.0 &&
                   second[(n - 2) * columns2 + 2] == 10.0,
                   "second root lost its original within-row failure prefix");
        for (std::size_t j = 0; j < columns2; ++j)
          check_pair(second[(n - 1) * columns2 + j] == 8.0 + 4.0 * j,
                     "second root lost its already completed last row");
      }
    }
  }

  test_affine_projection_contracts();
  test_compact_path_contracts();
  std::cout << "native graph tests passed\n";
  return 0;
}
