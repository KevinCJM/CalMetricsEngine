#include "../cpp/excel_internal.hpp"
#include <iostream>

using namespace calmetrics_engine;

int main() {
  try {
    excel::Builder count({}, {});
    for (std::size_t i = 0; i < 2000000; ++i)
      count.put(i < 1000000 ? "A" : "B", i % 1000000, 0, "number", "1");
    if (count.report.cells != 2000000)
      throw std::runtime_error("exact workbook budget");
    bool rejected = false;
    try {
      count.put("B", 1000000, 0, "number", "1");
    } catch (const excel::Error &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("cell 2000001 must fail");
    excel::Builder work({}, {});
    rejected = false;
    try {
      work.cost({SIZE_MAX, SIZE_MAX});
    } catch (const excel::Error &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("checked cost multiplication");
    // Geometry-only snapshots intentionally have no readable payload. A
    // symbolic bound must not evaluate data or allocate per-element formulas.
    excel::Snapshot geometry;
    geometry.value.shape = ops::vector_shape(1000);
    auto symbolic =
        excel::symbolic_budget(nullptr, ops::Op::mean, {geometry}, {}, {});
    if (!symbolic.cells || symbolic.snapshot_bytes != 8000)
      throw std::runtime_error("metadata-only symbolic analysis");
    excel::Limits boundary;
    boundary.cells = symbolic.cells;
    excel::symbolic_budget(nullptr, ops::Op::mean, {geometry}, {}, boundary);
    --boundary.cells;
    rejected = false;
    try {
      excel::symbolic_budget(nullptr, ops::Op::mean, {geometry}, {}, boundary);
    } catch (const excel::Error &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("symbolic cell boundary");
    geometry.value.shape.rank = 2;
    geometry.value.shape.dim = {SIZE_MAX, 2, 0};
    rejected = false;
    try {
      excel::symbolic_budget(nullptr, ops::Op::transpose, {geometry}, {}, {});
    } catch (const excel::Error &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("symbolic shape multiplication overflow");
    // Validate constants independently of platform-specific stream parsing.
    for (double value : {1e-320, -1e-320}) {
      rejected = false;
      try {
        excel::Builder constants({}, {});
        constants.literal(value);
      } catch (const excel::Error &) {
        rejected = true;
      }
      if (!rejected)
        throw std::runtime_error("subnormal graph constant must fail");
    }
    excel::Snapshot input;
    input.value.shape = ops::vector_shape(3);
    input.numbers = {1, 2, 3};
    excel::Plan plan(ops::Op::mean, {input}, {});
    input.numbers[0] = -999;
    if (plan.reference()[0].numbers[0] != 2)
      throw std::runtime_error("owned snapshot");
    plan.cancel();
    rejected = false;
    try {
      plan.reference();
    } catch (const excel::Error &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("cancelled plan");
    std::cout
        << "Excel native budget, overflow, ownership and cancellation passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
