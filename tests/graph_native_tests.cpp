#include "calmetrics_engine/graph.hpp"

#include <cmath>
#include <cstdint>
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

int main() {
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

  std::cout << "native graph tests passed\n";
  return 0;
}
