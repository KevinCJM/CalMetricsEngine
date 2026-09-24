#include "calmetrics_engine/compiler.hpp"
#include "calmetrics_engine/native_process.hpp"
#include "calmetrics_engine/model_payload.hpp"
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
void foundation_contracts() {
  // Register lowering is checked against the separate canonical arithmetic
  // functions, including a value that would change under multiply/add FMA.
  const std::vector<double> chain_input{1.-std::ldexp(1.,-27),-0.,0.,1.,-1.,
      std::numeric_limits<double>::max(),std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()};
  for (auto isa : {o::Isa::scalar,o::Isa::neon,o::Isa::sse2,o::Isa::avx2}) {
    if (!o::supports_isa(isa)) continue;
    for (auto first : {o::Op::add,o::Op::subtract,o::Op::multiply})
      for (auto second : {o::Op::add,o::Op::subtract,o::Op::multiply})
        for (int sides = 0; sides < 4; ++sides)
          for (std::ptrdiff_t stride : {-1,0,1}) {
            o::Value v; v.data = chain_input.data()+(stride < 0 ? chain_input.size()-1 : 0);
            v.shape = o::vector_shape(chain_input.size()); v.stride = {stride,1,1};
            std::vector<double> a(chain_input.size()),b(chain_input.size());
            std::vector<std::uint8_t> mask(chain_input.size());
            o::ScalarChain chain;
            chain.constants = {1.+std::ldexp(1.,-27),-1.}; chain.constant_first = {bool(sides&1),bool(sides&2)};
            chain.first_output = a.data(); chain.value_output = b.data(); chain.mask_output = mask.data();
            const auto kernel = o::scalar_chain_kernel(first,second,o::Op::greater_than,isa);
            check(kernel != nullptr,"missing AOT chain kernel");
            kernel(v,0,chain_input.size(),chain);
            for (std::size_t i = 0; i < chain_input.size(); ++i) {
              const auto x = v.f(i);
              const auto y = sides&1 ? o::scalar_math(first,chain.constants[0],x) : o::scalar_math(first,x,chain.constants[0]);
              const auto z = sides&2 ? o::scalar_math(second,chain.constants[1],y) : o::scalar_math(second,y,chain.constants[1]);
              const auto same = [](double lhs,double rhs) { return (std::isnan(lhs)&&std::isnan(rhs)) ||
                  (lhs == rhs && (lhs != 0 || std::signbit(lhs) == std::signbit(rhs))); };
              check(same(a[i],y)&&same(b[i],z)&&mask[i] == (z>0),"register chain changed canonical arithmetic");
            }
          }
    std::vector<double> inplace{8.,-8.,0.,std::numeric_limits<double>::max(),1.};
    const auto original = inplace;
    o::Value v; v.data = inplace.data(); v.shape = o::vector_shape(inplace.size());
    o::ScalarChain chain; chain.constants[0] = .5; chain.first_output = inplace.data(); chain.previous = inplace.data();
    double residual = 0; bool finite = true; chain.residual = &residual; chain.finite = &finite;
    o::scalar_chain_kernel(o::Op::multiply,o::Op(0),o::Op(0),isa)(v,0,inplace.size(),chain);
    for (std::size_t i = 0; i < inplace.size(); ++i) check(inplace[i] == original[i]*.5,"in-place chain value");
    check(finite && residual == std::numeric_limits<double>::max()*.5,"in-place scan lost old values");
  }
  for (std::size_t count : {0u, 1u, 2u, 3u, 31u, 32u, 33u, 65u}) {
    std::vector<double> before(count), candidate(count);
    for (std::size_t i = 0; i < count; ++i) {
      before[i] = i % 2 ? -static_cast<double>(i) : static_cast<double>(i);
      candidate[i] = before[i] * .125;
    }
    for (auto exceptional : {0.0, std::numeric_limits<double>::max(),
         std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
      if (count) candidate[count / 2] = exceptional;
      bool expected_finite = true, actual_finite = true;
      double expected_residual = 0, actual_residual = 0;
      for (std::size_t i = 0; i < count; ++i) {
        expected_finite = expected_finite && std::isfinite(candidate[i]);
        expected_residual = std::max(expected_residual, std::abs(candidate[i] - before[i]));
      }
      o::iteration_residual(before.data(), candidate.data(), count, actual_residual, actual_finite);
      check(actual_finite == expected_finite && actual_residual == expected_residual,
            "vector residual differs from ordered scalar max/finite contract");
    }
  }
  auto tensor = t::ValueType::tensor({"scenario", "path", "asset"}, {"S", "P", "N"});
  auto compiled = c::compile({"iterate(iterate_x*0.5,x,1e-6,100)",
      "iteration_status(iterate(iterate_x*0.5,x,1e-6,100))"}, std::vector<t::Variable>{{"x", tensor}}, true);
  auto program = c::decode_program(c::encode_program(compiled->program));
  check(program.apply_scopes.size() == 1, "iteration CSE");
  std::vector<double> source(24, 8.0);
  o::Value input; input.shape = o::tensor_shape(2,3,4); input.data = source.data(); input.set_contiguous_strides();
  const std::int64_t start = 0, end = 1;
  auto scalar_type = t::ValueType::scalar();
  scalar_type.kind = t::ValueKind::value;
  const std::map<std::string, std::string> scalar_binding{{"a", "iterate(iterate_x+1,x,0,1)"}};
  auto scalar_graph = c::compile({"a", "iterate(iterate_x+a,x,0,1)"},
      std::vector<t::Variable>{{"x", scalar_type}}, false, {scalar_binding, scalar_binding});
  double initial_scalar = 2.0;
  auto scalar_input = o::Value::number(initial_scalar);
  scalar_input.data = &initial_scalar;
  auto scalar_layout = g::result_layout(scalar_graph->program, {scalar_input}, &start, &end, 1);
  std::vector<std::uint64_t> scalar_output((scalar_layout.bytes()+7)/8);
  g::execute(scalar_graph->program, {scalar_input}, nullptr, 0, &start, &end, 1,
      scalar_output.data(), 2, &scalar_layout);
  const auto *scalar_values = reinterpret_cast<const double *>(scalar_output.data());
  check(scalar_values[0] == 3 && scalar_values[1] == 5,
      "captured scalar iteration result refreshed from initial input");
  auto layout = g::result_layout(program, {input}, &start, &end, 1);
  std::vector<std::uint64_t> output((layout.bytes()+7)/8);
  auto audit = g::execute(program, {input}, nullptr, 0, &start, &end, 1, output.data(), 2, &layout);
  check(audit.result_shapes[0] == input.shape && audit.root_statuses[0] == 0, "iteration tensor shape");
  const auto *values = reinterpret_cast<const double *>(output.data());
  check(values[0] == std::ldexp(8.0, -23), "iteration convergence value");
  auto wrong_dtype = program;
  wrong_dtype.root_outputs[0].dtype = g::OutputDType::boolean;
  auto narrow_layout = g::result_layout(wrong_dtype, {input}, &start, &end, 1);
  std::vector<std::uint64_t> narrow(narrow_layout.bytes()/8 + 1, 0xdeadbeef);
  rejects([&] { g::execute(wrong_dtype, {input}, nullptr, 0, &start, &end, 1,
      narrow.data(), 2, &narrow_layout); }, "iteration wrote incompatible destination dtype");
  check(narrow.back() == 0xdeadbeef, "iteration exceeded malformed destination capacity");
  check(*reinterpret_cast<const std::int64_t *>(reinterpret_cast<const std::uint8_t *>(output.data()) +
       layout.slots[1].byte_offset) == 0, "iteration solver status");
  auto invalid = program;
  invalid.apply_scopes[0].state_input_index = 100;
  rejects([&] { invalid.validate(); }, "invalid state binding accepted");
  invalid = program;
  invalid.apply_scopes[0].body = std::make_shared<g::Program>(*program.apply_scopes[0].body);
  invalid.apply_scopes[0].body->root_outputs.clear();
  rejects([&] { invalid.validate(); }, "invalid child schema accepted");
  rejects([&] { g::execute(program, {input}, nullptr, 0, &start, &end, 1, output.data(), 2, &layout, 0,
      n::Clock::now() - std::chrono::seconds(1)); }, "iteration deadline not checked");
  std::atomic<bool> cancelled{true};
  rejects([&] { g::execute(program, {input}, nullptr, 0, &start, &end, 1, output.data(), 2, &layout, 0,
      n::Deadline::max(), &cancelled); }, "iteration cancellation not checked");
  const std::map<std::string,std::string> metadata{{"schema","fixture-1"}, {"algorithm","fixture"},
      {"algorithm_version","1"}, {"source_identity","test"}, {"training_options","{}"}, {"random_version","none"}};
  auto model = n::ModelPayload::create({{"x",tensor}}, {input}, metadata);
  auto encoded = model->encode();
  check(n::ModelPayload::decode(encoded)->identity() == model->identity(), "model round trip");
  auto wrong = tensor; wrong.shape = {"3","3","4"};
  rejects([&] { n::ModelPayload::create({{"x",wrong}}, {input}, metadata); }, "model fixed shape ignored");
  auto peer = input; peer.shape = o::tensor_shape(1,3,4);
  rejects([&] { n::ModelPayload::create({{"x",tensor},{"y",tensor}}, {input,peer}, metadata); }, "model symbolic shape ignored");
  for (auto length : {std::size_t(0), std::size_t(7), encoded.size()-1}) {
    auto truncated = encoded; truncated.resize(length);
    rejects([&] { n::ModelPayload::decode(truncated); }, "truncated model accepted");
  }
  // The empty third dimension must not be read or converted into a scalar.
  input.shape = o::tensor_shape(2,3,0); input.data = nullptr; input.set_contiguous_strides();
  auto empty = n::ModelPayload::create({{"x",tensor}}, {input}, metadata);
  check(empty->bytes() == 0 && n::ModelPayload::decode(empty->encode())->fields()[0].size() == 0,
        "empty model field");
  o::Arguments<o::Value> arguments(32);
  for (std::size_t i = 0; i < arguments.size(); ++i) arguments[i] = o::Value::number(i);
  auto copied = arguments;
  check(copied[31].scalar == 31 && arguments.data() != copied.data(), "large argument ownership");
  rejects([&] { o::Arguments<o::Value> oversized(33); }, "argument limit ignored");
  auto fused = c::compile({"x*2+1", "x*2+1>3"}, std::vector<t::Variable>{{"x", tensor}});
  input.shape = o::tensor_shape(2,3,4); input.data = source.data(); input.set_contiguous_strides();
  auto fused_layout = g::result_layout(fused->program, {input}, &start, &end, 1);
  std::vector<std::uint64_t> fused_output(fused_layout.bytes()/8 + 1, 0xdeadbeef);
  rejects([&] { g::execute(fused->program, {input}, nullptr, 0, &start, &end, 1,
      fused_output.data(), 1, &fused_layout); }, "fused output column validation omitted");
  auto bad_fused = fused->program;
  for (auto &node : bad_fused.nodes)
    if (node.kind == g::NodeKind::operation) { node.parent_count = 3; node.parents[2] = 0; break; }
  rejects([&] { g::execute(bad_fused, {input}, nullptr, 0, &start, &end, 1,
      fused_output.data(), 2, &fused_layout); }, "fused invalid arity accepted");
  bad_fused = fused->program;
  bad_fused.root_outputs[0].dtype = g::OutputDType::int64;
  rejects([&] { g::execute(bad_fused, {input}, nullptr, 0, &start, &end, 1,
      fused_output.data(), 2, &fused_layout); }, "fused numeric output accepted integer schema");
  auto fused_audit = g::execute(fused->program, {input}, nullptr, 0, &start, &end, 1,
      fused_output.data(), 2, &fused_layout);
  check(fused_audit.fused_pointwise_calls == 1 && fused_audit.root_copy_bytes == 0, "physical pointwise path missing");
  check(fused_output.back() == 0xdeadbeef, "fused output end sentinel");
  bad_fused = fused->program;
  for (auto &node : bad_fused.nodes)
    if (node.kind == g::NodeKind::operation) { node.storage = g::StorageKind::inline_value; break; }
  bad_fused.finalize();
  rejects([&] { g::execute(bad_fused, {input}, nullptr, 0, &start, &end, 1,
      fused_output.data(), 2, &fused_layout); }, "pointwise array accepted inline storage");
  auto regions = c::compile({"((x*2+1)*3-2)*4+1", "((x*2+1)*3-2)*4+1>0"},
      std::vector<t::Variable>{{"x", tensor}});
  auto restored = c::decode_program(c::encode_program(regions->program));
  check(restored.execution_metadata->pointwise_regions.size() == 3, "regions not rebuilt on decode");
  for (const auto &region : restored.execution_metadata->pointwise_regions)
    check(region.kernel && !region.retain_first, "unused region intermediate materialized");
}
int main(int argc, char **argv) {
  if (argc != 2) return 2;
  try {
    foundation_contracts();
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
    batch.model_identity = "model-1-" + std::string(32, 'a'); // Exercise identity even on inline transport.
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
    const std::size_t status_blob = headers - 88; // v7 fixed framing outside the audit/status prefix.
    const std::size_t shape_count = status_blob + status_bytes;
    const std::size_t first_shape = shape_count + 8;
    const std::size_t root_status = first_shape + 10 * 32 + 8;
    const std::size_t kind_offset = response.size() - layout.bytes() - 16 - 48;
    for (const auto mutation : std::vector<std::pair<std::size_t, std::uint8_t>>{
             {0,4}, {shape_count,9}, {first_shape,5}, {first_shape+8,255},
             {root_status,8}, {status_blob,8}, {kind_offset,0}, {kind_offset-8,99}, {response.size()-1,1}}) {
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
