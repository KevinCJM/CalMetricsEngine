#include "calmetrics_engine/compiler.hpp"
#include "calmetrics_engine/native_process.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace c = calmetrics_engine::compiler;
namespace g = calmetrics_engine::graph;
namespace o = calmetrics_engine::ops;
namespace t = calmetrics_engine::typed;
namespace n = calmetrics_engine::native;
namespace p = calmetrics_engine::planner;
void check(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
template<class F> void rejects(F action, const char *message) {
  bool failed = false;
  try { action(); } catch (const std::exception &) { failed = true; }
  check(failed, message);
}
int main(int argc, char **argv) {
  if (argc != 2) return 2;
  try {
    auto ids_type = t::ValueType::vector();
    ids_type.dtype = t::DType::int64;
    auto graph = c::compile({"transpose(x)", "mean_time(x)", "ids", "sum_asset(x)>0", "mean(sum_asset(x))"},
        std::vector<t::Variable>{{"x", t::ValueType::matrix()}, {"ids", ids_type}}, true);
    auto program = c::decode_program(c::encode_program(graph->program));
    check(program.output_kind == g::OutputKind::typed && program.root_outputs.size() == 5,
          "typed schema serialization");
    std::vector<double> x{1,2,3,4,5,6,7,8,9,10,11,12};
    std::vector<std::int64_t> ids{9007199254740993LL, INT64_MIN, INT64_MAX};
    o::Value matrix; matrix.data = x.data(); matrix.shape = o::matrix_shape(4,3); matrix.stride = {3,1};
    o::Value integer; integer.kind = o::Kind::integer; integer.data = ids.data(); integer.shape = o::vector_shape(3);
    std::vector<o::Value> inputs{matrix, integer};
    std::int64_t starts[]{0,1}, ends[]{2,4};
    auto layout = g::result_layout(program, inputs, starts, ends, 2);
    std::vector<std::uint64_t> output(layout.bytes()/8 + 1, 0xdeadbeef);
    const auto audit = g::execute(program, inputs, nullptr, 0, starts, ends, 2, output.data(), 5, &layout);
    check(audit.result_shapes.size() == 10 && audit.root_statuses.size() == 10, "root metadata count");
    check(audit.result_shapes[0] == o::matrix_shape(3,2) && audit.result_shapes[5] == o::matrix_shape(3,3), "per-interval shapes");
    for (std::size_t row = 0; row < 2; ++row) {
      const auto *exact = reinterpret_cast<const std::uint8_t *>(output.data()) + layout.slots[row*5+2].byte_offset;
      for (std::size_t i = 0; i < 3; ++i) {
        std::int64_t value;
        std::memcpy(&value, exact + i * sizeof(value), sizeof(value));
        check(value == ids[i], "exact integer payload");
      }
      for (std::size_t root = 0; root < 5; ++root) {
        check(layout.slots[row*5+root].byte_offset % 8 == 0, "root alignment");
        check(audit.root_statuses[row*5+root] == 0, "healthy root status");
      }
    }
    check(output.back() == 0xdeadbeef, "payload end sentinel");
    // Corrupt real worker responses: validation must finish before copying any payload.
    n::Batch batch; batch.inputs = inputs; batch.starts = starts; batch.ends = ends; batch.rows = 2;
    p::Config config; config.process_work_units = 1; config.min_rows_per_worker = 1;
    config.shared_memory_threshold_bytes = 1ull << 30;
    n::Engine engine(2, config, argv[1]);
    auto plan = engine.plan(graph, batch);
    check(!plan->use_shared_memory, "inline transport for corruption tests");
    n::ProcessTransport transport(*plan, batch);
    n::ProcessPool pool(argv[1], 1);
    std::atomic<bool> cancelled{false};
    const auto response = pool.transact(transport.request({0,2}), n::Clock::now() + std::chrono::seconds(10), false, cancelled);
    transport.response(response, {0,2}, output.data());
    const auto status_bytes = layout.status_count() * sizeof(std::int16_t);
    check(response.size() == n::process_response_bytes(layout.bytes(), status_bytes, 10),
          "response size estimate differs from actual worker serialization");
    const auto headers = n::process_response_bytes(0, 0, 0);
    check(n::process_response_bytes(n::max_process_frame_bytes - headers, 0, 0) == n::max_process_frame_bytes,
          "exact IPC frame boundary");
    check(n::process_response_bytes(n::max_process_frame_bytes - headers, 2, 1) > n::max_process_frame_bytes,
          "statuses and shapes must count toward frame limit");
    rejects([&] { n::process_response_bytes(0, 0, std::numeric_limits<std::size_t>::max()); },
            "response metadata size overflow accepted");
    const std::size_t shape_count = 128 + status_bytes; // v5: header, 12 audit fields, status count/blob.
    const std::size_t first_shape = shape_count + 8;
    const std::size_t root_status = first_shape + 10 * 24 + 8;
    const std::size_t kind_offset = response.size() - layout.bytes() - 16;
    for (const auto mutation : std::vector<std::pair<std::size_t, std::uint8_t>>{
             {0,4}, {shape_count,9}, {first_shape,4}, {first_shape+8,255},
             {root_status,8}, {128,8}, {kind_offset,0}, {kind_offset-8,99}}) {
      auto corrupt = response; corrupt.at(mutation.first) = mutation.second;
      std::fill(output.begin(), output.end(), 0xdeadbeef);
      rejects([&] { transport.response(corrupt, {0,2}, output.data()); }, "malformed worker response accepted");
      check(std::all_of(output.begin(), output.end(), [](auto x) { return x == 0xdeadbeef; }), "rejected response copied output");
    }
    auto truncated = response; truncated.pop_back();
    rejects([&] { transport.response(truncated, {0,2}, output.data()); }, "truncated payload accepted");
    auto encoded = c::encode_program(program);
    encoded[38] = 99; // First root dtype follows the version-6 fixed header.
    rejects([&] { c::decode_program(encoded); }, "invalid serialized dtype accepted");
    // A v5 scalar definition has the same header except for the new schema count.
    auto legacy_graph = c::compile({"mean(x)"}, std::vector<t::Variable>{{"x", t::ValueType::series()}});
    auto legacy_bytes = c::encode_program(legacy_graph->program);
    legacy_bytes.erase(legacy_bytes.begin()+34, legacy_bytes.begin()+38);
    legacy_bytes[4] = 5;
    auto legacy = c::decode_program(legacy_bytes);
    auto series = matrix; series.shape = o::vector_shape(x.size()); series.stride = {1,1};
    double legacy_result = 0;
    g::execute(legacy, {series}, nullptr, 0, starts, ends, 1, &legacy_result, 1);
    check(legacy.output_kind == g::OutputKind::scalar && legacy_result == 1.5, "v5 scalar compatibility");
    auto broken = program;
    broken.root_outputs.pop_back();
    rejects([&] { g::result_layout(broken, inputs, starts, ends, 2); }, "invalid root count accepted");
    auto huge = inputs;
    huge[1].shape = o::vector_shape(static_cast<std::size_t>(PTRDIFF_MAX));
    rejects([&] { g::result_layout(program, huge, starts, ends, 2); }, "payload size overflow accepted");
    rejects([&] { g::result_layout(program, inputs, starts, ends, std::numeric_limits<std::size_t>::max()); }, "row count overflow accepted");
    std::int64_t bad_end = 5;
    rejects([&] { g::result_layout(program, inputs, starts, &bad_end, 1); }, "invalid time bound accepted");
    std::vector<std::uint8_t> unaligned(layout.bytes()+8);
    rejects([&] { g::execute(program, inputs, nullptr, 0, starts, ends, 2, unaligned.data()+1, 5, &layout); }, "unaligned destination accepted");
    // Empty dimensions remain known and distinct from minimum-sample failures.
    inputs[0].shape = o::matrix_shape(4,0);
    inputs[1].shape = o::vector_shape(0);
    auto empty_graph = c::compile({"x", "ids"},
        std::vector<t::Variable>{{"x", t::ValueType::matrix()}, {"ids", ids_type}}, true);
    auto empty_layout = g::result_layout(empty_graph->program, inputs, starts, ends, 2);
    check(empty_layout.bytes() == 0, "empty output allocation");
    auto empty_audit = g::execute(empty_graph->program, inputs, nullptr, 0, starts, ends, 2, output.data(), 2, &empty_layout);
    check(empty_audit.result_shapes[0] == o::matrix_shape(2,0) && empty_audit.root_statuses[0] == 0, "known empty shape");
    empty_graph->program.minimum_observations = 5;
    auto failed = g::execute(empty_graph->program, inputs, nullptr, 0, starts, ends, 2, output.data(), 2, &empty_layout);
    check(failed.result_shapes[0].rank == -1 && failed.root_statuses[0] == 1, "empty capacity failed root");
    std::cout << "native heterogeneous result contracts passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
