#include "calmetrics_engine/compiler.hpp"
#include "calmetrics_engine/native_process.hpp"
#include "calmetrics_engine/model_payload.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include <cfenv>
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
#include "state_root_loan_native_cases.hpp"
#include "latent_unwritten_result_native_cases.hpp"

void dense_layout_contracts(const char *worker) {
  for (auto shape : {o::Shape{}, o::vector_shape(7), o::matrix_shape(5,3),
                     o::tensor_shape(4,3,2), o::tensor_shape(1,3,1), o::tensor_shape(4,0,2)}) {
    for (auto order : {o::MemoryOrder::c, o::MemoryOrder::f}) {
      o::Value value; value.shape = shape; value.set_contiguous_strides(order);
      check(value.contiguous(order) && value.dense_contiguous(), "dense geometry classification");
      if (shape.size() == 0 || shape.rank <= 1 || shape == o::tensor_shape(1,3,1))
        check(value.c_contiguous() && value.f_contiguous() && value.packed_order() == o::MemoryOrder::c,
              "ambiguous layout must normalize to C");
      std::vector<double> storage(shape.size()); value.data = storage.data();
      for (std::size_t i = 0; i < value.size(); ++i) storage[value.offset(i)] = double(i);
      for (std::size_t i = 0; i < value.size(); ++i) check(value.f(i) == double(i), "dense logical indexing");
    }
  }
  o::Value unusual; unusual.shape = o::tensor_shape(1,3,1); unusual.stride = {-19,1,0};
  check(unusual.c_contiguous() && unusual.f_contiguous() && !unusual.contiguous(),
        "dense singleton rule must not change legacy C-only predicate");
  unusual.shape = o::matrix_shape(5,3);
  for (auto strides : {std::array<std::ptrdiff_t,3>{4,1,1}, {1,6,1}, {-3,1,1}, {0,1,1}, {1,1,1}}) {
    unusual.stride = strides;
    check(!unusual.dense_contiguous(), "padded/reversed/overlapping input is not dense");
  }
  unusual.shape = {4,{2,3,4}};
  check(!unusual.dense_contiguous(), "invalid dense rank accepted");
  rejects([&] { unusual.set_contiguous_strides(o::MemoryOrder::f); }, "invalid layout rank accepted");
  unusual.shape = o::matrix_shape(static_cast<std::size_t>(PTRDIFF_MAX), 2);
  rejects([&] { unusual.set_contiguous_strides(o::MemoryOrder::f); }, "layout size overflow accepted");
  unusual.shape = o::tensor_shape(static_cast<std::size_t>(PTRDIFF_MAX), 2, 0);
  unusual.set_contiguous_strides(o::MemoryOrder::f);
  check(unusual.size() == 0 && unusual.dense_contiguous(), "empty layout overflowed unused product");
  rejects([&] { unusual.set_contiguous_strides(static_cast<o::MemoryOrder>(2)); }, "unknown memory order accepted");

  // Inspect the real request, then execute it. Logical parity alone cannot prove
  // that the parent did not silently repack a column-major matrix as row-major.
  std::vector<double> physical(12);
  o::Value input; input.shape = o::matrix_shape(4,3); input.data = physical.data();
  input.set_contiguous_strides(o::MemoryOrder::f);
  for (std::size_t i=0; i<input.size(); ++i) physical[input.offset(i)] = double(i+1);
  auto graph = c::compile({"x"}, std::vector<t::Variable>{{"x",t::ValueType::matrix()}});
  std::int64_t starts[]{0,1}, ends[]{2,4};
  n::Batch batch; batch.inputs = {input}; batch.starts=starts; batch.ends=ends; batch.rows=2;
  p::Config config; config.process_work_units=1; config.min_rows_per_worker=1;
  config.shared_memory_threshold_bytes=1ull<<30;
  n::Engine engine(2,config,worker);
  auto plan=engine.plan(graph,batch);
  n::ProcessTransport transport(*plan,batch);
  auto request=transport.request({0,2});
  const auto word = [&](std::size_t offset) {
    check(offset+8 <= request.size(), "test request word bounds");
    std::uint64_t v=0;
    for (unsigned j=0;j<8;++j) v |= std::uint64_t(request[offset+j]) << (8*j);
    return v;
  };
  check(word(0) == 0x434d45330000001eull, "layout, wide stream and F row-view admission wire version");
  std::size_t cursor=8;
  cursor += 8+word(cursor); // program
  check(word(cursor)==0,"layout test must use inline inputs"); cursor+=8;
  cursor += 8+word(cursor); // parameters
  check(word(cursor)==1,"layout input count"); cursor+=8;
  const auto descriptor_offset=cursor;
  check(word(cursor)==(std::uint64_t(o::Kind::number)|(1ull<<8)),"worker request lost F layout"); cursor+=8;
  check(word(cursor)==2 && word(cursor+8)==4 && word(cursor+16)==3,"worker request transposed shape"); cursor+=24;
  check(word(cursor)==physical.size()*8,"physical input byte count"); cursor+=8;
  check(std::memcmp(request.data()+cursor,physical.data(),physical.size()*8)==0,"worker repacked F bytes");
  const auto &layout=*plan->result_layout;
  std::vector<std::uint64_t> expected((layout.bytes()+7)/8), actual(expected.size());
  g::execute(graph->program,batch.inputs,nullptr,0,starts,ends,2,expected.data(),1,&layout);
  n::ProcessPool pool(worker,1); std::atomic<bool> cancelled{false};
  for (auto corrupt : {0,4,1,2,3,5,6,7,0}) {
    auto current=request;
    if (corrupt==1) current.at(descriptor_offset+1)=2;
    if (corrupt==2) current.at(0)=0x18; // Previous request protocol must be rejected.
    if (corrupt==3) current.at(8+8+word(8))=8; // Unknown mode bits must be rejected.
    if (corrupt==4) current.at(8+8+word(8))=4; // Current original-row geometry is valid.
    if (corrupt==5) current.at(0)=0x17; // Older audit request protocol is also rejected.
    if (corrupt==6) current.at(8+8+word(8))=15; // Known flags do not mask an unknown bit.
    if (corrupt==7) current.at(0)=0x19; // Previous worker audit table must be rejected before execution.
    auto response=pool.transact(current,n::Clock::now()+std::chrono::seconds(10),false,cancelled);
    if (corrupt && corrupt!=4) {
      std::fill(actual.begin(),actual.end(),0xdeadbeef);
      rejects([&] { transport.response(response,{0,2},actual.data()); },"invalid layout or old audit wire accepted");
      check(std::all_of(actual.begin(),actual.end(),[](auto v){return v==0xdeadbeef;}),"bad layout wrote output");
    } else {
      if (corrupt==0) {
        auto stale=response;
        stale.at(0)=0x18; // Previous response lacks the Gaussian/filter audit field.
        std::fill(actual.begin(),actual.end(),0xdeadbeef);
        std::string error;
        try { transport.response(stale,{0,2},actual.data()); }
        catch (const std::exception &e) { error=e.what(); }
        check(error=="unsupported native worker response version","old response audit table accepted");
        check(std::all_of(actual.begin(),actual.end(),[](auto v){return v==0xdeadbeef;}),
              "old response wrote output before version validation");
      }
      transport.response(response,{0,2},actual.data());
      check(actual==expected,"F worker decoding changed logical values");
    }
  }
  // A negative first-axis stride must be packed as logical F coordinates,
  // rather than advertised as F over a C-ordered byte stream.
  batch.inputs[0].data=physical.data()+3;
  batch.inputs[0].stride[0]=-1;
  auto reverse_plan=engine.plan(graph,batch);
  n::ProcessTransport reversed(*reverse_plan,batch);
  request=reversed.request({0,2});
  check(word(descriptor_offset)==(std::uint64_t(o::Kind::number)|(1ull<<8)),
        "reversed column transport lost F packing");
  for (std::size_t j=0;j<3;++j) for (std::size_t i=0;i<4;++i) {
    double packed=0;std::memcpy(&packed,request.data()+cursor+(j*4+i)*8,8);
    check(packed==physical[j*4+3-i],"reversed F payload has wrong coordinates");
  }
  g::execute(graph->program,batch.inputs,nullptr,0,starts,ends,2,expected.data(),1,reverse_plan->result_layout.get());
  const auto response=pool.transact(request,n::Clock::now()+std::chrono::seconds(10),false,cancelled);
  reversed.response(response,{0,2},actual.data());
  check(actual==expected,"reversed F worker changed logical values");
}

void result_order_contracts(const char *worker) {
  for (std::size_t count : {0u,1u,63u,64u,65u,129u}) {
    g::RootOrders orders;orders.assign(count,o::MemoryOrder::c);
    for (std::size_t i=0;i<count;++i) if (i%3==0) orders.set(i,o::MemoryOrder::f);
    g::RootOrders copied=orders, assigned;assigned=copied;
    auto moved=std::move(copied);
    check(copied.empty(),"moved layout flags retain a live length");
    g::RootOrders moved_assignment;moved_assignment=std::move(assigned);
    check(assigned.empty(),"move-assigned layout flags retain a live length");
    for (std::size_t i=0;i<count;++i) {
      const auto expected=i%3==0 ? o::MemoryOrder::f : o::MemoryOrder::c;
      check(orders.at(i)==expected && moved.at(i)==expected && moved_assignment.at(i)==expected,
            "copied layout flags lost an inline/overflow bit");
    }
    rejects([&]{orders.at(count);},"layout flag index overflow accepted");
    rejects([&]{orders.set(count,o::MemoryOrder::f);},"layout flag write overflow accepted");
    moved.assign(2,o::MemoryOrder::f);moved.set(1,o::MemoryOrder::c);
    check(moved.at(0)==o::MemoryOrder::f && moved.at(1)==o::MemoryOrder::c,"layout flag reset failed");
    check(orders.size()==count,"layout flag copy aliases source size");
    rejects([&]{moved.assign(3,static_cast<o::MemoryOrder>(2));},"unknown inline layout accepted");
    check(moved.size()==2 && moved.at(0)==o::MemoryOrder::f,"failed layout assignment changed storage");
  }
  for (const auto shape : {o::matrix_shape(5,3),o::tensor_shape(4,3,2)}) {
    std::vector<double> values(shape.size()+2,-99);
    o::Output out; out.shape=shape; out.order=o::MemoryOrder::f; out.data=values.data()+1;
    o::Value view; view.shape=shape; view.data=out.data; view.set_contiguous_strides(o::MemoryOrder::f);
    for (std::size_t i=0;i<shape.size();++i) out.set(i,double(i));
    for (std::size_t i=0;i<shape.size();++i) check(view.f(i)==double(i),"F writer changed logical coordinates");
    check(values.front()==-99 && values.back()==-99,"F writer exceeded capacity");
  }
  auto old=c::compile({"x"},std::vector<t::Variable>{{"x",t::ValueType::matrix()}});
  auto graph=c::compile({"x"},std::vector<t::Variable>{{"x",t::ValueType::matrix()}},
      false,{}, {},0,100000000,true,{},g::OutputOrder::f);
  check(c::encode_program(old->program)[4]==7,"legacy plan no longer uses v7");
  auto encoded=c::encode_program(graph->program);
  check(encoded[4]==8 && encoded[40]==1,"new plan output order missing");
  check(c::decode_program(encoded).root_outputs[0].order==g::OutputOrder::f,"plan lost F policy");
  encoded[40]=3;
  rejects([&]{c::decode_program(encoded);},"invalid plan output order accepted");
  std::vector<double> input(12);
  for (std::size_t i=0;i<input.size();++i) input[i]=double(i+1);
  o::Value x; x.shape=o::matrix_shape(4,3); x.data=input.data(); x.set_contiguous_strides();
  std::int64_t starts[]{0,1},ends[]{2,4};
  n::Batch batch; batch.inputs={x};batch.starts=starts;batch.ends=ends;batch.rows=2;
  p::Config config;config.process_work_units=1;config.min_rows_per_worker=1;
  config.shared_memory_threshold_bytes=1ull<<30;
  n::Engine engine(2,config,worker); auto plan=engine.plan(graph,batch);
  const auto &layout=*plan->result_layout;check(layout.order(0)==o::MemoryOrder::f,"lost resolved F order");
  n::ProcessTransport transport(*plan,batch);n::ProcessPool pool(worker,1);std::atomic<bool> cancelled{false};
  const auto response=pool.transact(transport.request({0,2}),n::Clock::now()+std::chrono::seconds(10),false,cancelled);
  // Tail: shape records, status blob, dtype/kind, output blob, 40-byte model id.
  const auto shape_offset=response.size()-(2*32+8+2*2+16+8+layout.bytes()+48);
  check(response.at(shape_offset)==3 && response.at(shape_offset+1)==1,"response lacks actual F order");
  std::vector<std::uint64_t> output(layout.bytes()/8,0xdeadbeef);
  for (const auto wrong : {0,2}) {
    auto bad=response;bad.at(shape_offset+1)=wrong;
    rejects([&]{transport.response(bad,{0,2},output.data());},"invalid/mismatched response order accepted");
    check(std::all_of(output.begin(),output.end(),[](auto v){return v==0xdeadbeef;}),"bad response wrote output");
  }
  auto audit=transport.response(response,{0,2},output.data());
  for (std::size_t row=0;row<2;++row) {
    o::Value v;v.shape=audit.result_shapes[row];v.set_contiguous_strides(o::MemoryOrder::f);
    v.data=reinterpret_cast<const std::uint8_t *>(output.data())+layout.slots[row].byte_offset;
    for (std::size_t i=0;i<v.size();++i) check(v.f(i)==input[starts[row]*3+i],"F worker output transposed values");
  }
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
void covariance_reuse_contracts() {
  constexpr std::uint64_t sentinel=0x5a7f31be19d04c62ULL;
  for (const auto dimensions : {std::array<std::size_t,2>{33,9}, {33,17},
                                {129,9}, {129,17}, {34,10}, {32,12}}) {
    const auto rows=dimensions[0], cols=dimensions[1];
    const std::size_t elements=cols*cols;
    const std::int64_t start=0, end=rows;
    for (int layout : {0,1,2}) {
      const bool fortran=layout!=0;
      const std::size_t padding=layout==2 ? 1 : 0, pitch=rows+(layout==2 ? 3 : 0);
      std::vector<double> source(fortran ? pitch*cols+padding : rows*cols, -9876.5);
      const auto offset=[&](std::size_t i,std::size_t j) {
        return fortran ? padding+j*pitch+i : i*cols+j;
      };
      for (std::size_t i=0;i<rows;++i) for (std::size_t j=0;j<cols;++j)
        source[offset(i,j)]=static_cast<int>((i*11+j*7)%23)-11+double(i)/32;
      o::Value input; input.data=source.data()+padding; input.shape=o::matrix_shape(rows,cols);
      input.stride=fortran ? std::array<std::ptrdiff_t,3>{1,static_cast<std::ptrdiff_t>(pitch),1}
                          : std::array<std::ptrdiff_t,3>{static_cast<std::ptrdiff_t>(cols),1,1};
      for (bool constant : {false,true}) {
        if (constant) for (std::size_t i=0;i<rows;++i) source[offset(i,0)]=7.0;
        const auto before=source;
        std::vector<double> covariance(elements),correlation(elements);
        o::Workspace work;
        const auto cov_plan=o::prepare(o::lookup("covariance"),&input,1);
        o::Output cov_out; cov_out.kind=cov_plan.output_kind; cov_out.shape=cov_plan.output_shape;
        cov_out.data=covariance.data(); o::Audit cov_audit;
        o::execute(cov_plan,cov_out,work,o::Isa::automatic,cov_audit);
        std::vector<double> scalar_covariance(elements);
        auto scalar_out=cov_out; scalar_out.data=scalar_covariance.data(); o::Audit scalar_audit;
        o::execute(cov_plan,scalar_out,work,o::Isa::scalar,scalar_audit);
        for (std::size_t i=0;i<elements;++i)
          check(std::abs(covariance[i]-scalar_covariance[i])<=1e-12+1e-12*std::abs(scalar_covariance[i]),
                "covariance panel differs from scalar reference");
        const auto corr_plan=o::prepare(o::lookup("correlation"),&input,1);
        o::Output corr_out; corr_out.kind=corr_plan.output_kind; corr_out.shape=corr_plan.output_shape;
        corr_out.data=correlation.data(); o::Audit corr_audit;
        if (constant) {
          bool failed=false;
          try { o::execute(corr_plan,corr_out,work,o::Isa::automatic,corr_audit); }
          catch (const o::Error &error) { failed=std::strcmp(error.what(),"NON_FINITE_RESULT")==0; }
          check(failed,"constant-column direct correlation did not preserve its error");
        } else o::execute(corr_plan,corr_out,work,o::Isa::automatic,corr_audit);
        auto graph=c::compile({"covariance(x)","correlation(x)","covariance(x)"},
            std::vector<t::Variable>{{"x",t::ValueType::matrix()}},true);
        check(graph->program.output_kind==g::OutputKind::typed &&
              graph->program.roots[0]==graph->program.roots[2],"covariance duplicate-root fixture");
        const auto layout=g::result_layout(graph->program,{input},&start,&end,1);
        const auto words=(layout.bytes()+7)/8;
        std::vector<std::uint64_t> storage(words+2,sentinel);
        auto *output=storage.data()+1;
        const auto audit=g::execute(graph->program,{input},nullptr,0,&start,&end,1,output,3,&layout);
        check(storage.front()==sentinel && storage.back()==sentinel,"covariance reuse result canary");
        check(source==before,"covariance reuse mutated its C/F input");
        check(audit.algorithm_copy_bytes==elements*sizeof(double),"covariance reuse copy audit");
        // The direct control distinguishes an available Eigen backend from OFF
        // builds and unsupported hosts without changing the expected shared count.
        check(cov_audit.eigen_covariance_calls<=1 &&
              audit.eigen_covariance_calls==cov_audit.eigen_covariance_calls,
              "covariance reuse executed its core more than once");
        for (std::size_t root=0;root<3;++root) {
          const auto &slot=layout.slots[root];
          check(slot.capacity==elements && slot.byte_offset+elements*sizeof(double)<=layout.bytes(),
                "covariance reuse invalid slot range");
          if (root) check(layout.slots[root-1].byte_offset+elements*sizeof(double)<=slot.byte_offset,
                          "covariance reuse roots overlap");
          check(audit.result_shapes[root]==o::matrix_shape(cols,cols),"covariance reuse result shape");
          const bool failed=constant && root==1;
          check(audit.root_statuses[root]==(failed ? 4 : 0),"correlation failure corrupted covariance roots");
          const auto *values=reinterpret_cast<const double *>(
              reinterpret_cast<const std::uint8_t *>(output)+slot.byte_offset);
          const auto &expected=root==1 ? correlation : covariance;
          for (std::size_t i=0;i<elements;++i) {
            check(audit.statuses[slot.status_offset+i]==(failed ? 4 : 0),"covariance reuse position status");
            if (failed) check(std::isnan(values[i]),"failed correlation exposed partial normalization");
            else check(std::isfinite(values[i]) &&
                       std::abs(values[i]-expected[i])<=1e-12+1e-12*std::abs(expected[i]),
                       "covariance reuse differs from independent direct operators");
          }
        }
      }
    }
  }
}
void softmax_library_contracts() {
  constexpr std::size_t rows=32, cols=9;
  constexpr double sentinel=-987654.125;
  for (auto bad : {std::numeric_limits<double>::quiet_NaN(),
                   std::numeric_limits<double>::infinity(),
                   -std::numeric_limits<double>::infinity()}) {
    std::vector<double> source(rows*cols, 0.0), destination(rows*cols+2,sentinel);
    source[cols+cols-1]=bad;
    o::Value args[2]; args[0].data=source.data();
    args[0].shape=o::matrix_shape(rows,cols); args[0].stride={cols,1,1};
    args[1]=o::Value::number(1);
    const auto prepared=o::prepare(o::lookup("softmax_axis"),args,2);
    o::Workspace workspace; o::Audit audit; o::Output output;
    output.shape=prepared.output_shape; output.data=destination.data()+1;
    bool failed=false;
    try { o::execute(prepared,output,workspace,o::Isa::automatic,audit); }
    catch (const o::Error &error) {
      failed=std::strcmp(error.what(),"NON_FINITE_RESULT")==0;
    }
    check(failed,"softmax nonfinite row lost its error");
    for (std::size_t j=0;j<cols;++j)
      check(destination[j+1]==1.0/double(cols),"softmax healthy preceding row");
    for (std::size_t j=cols+1;j<destination.size();++j)
      check(destination[j]==sentinel,"softmax writes failing or following row");
    check(destination.front()==sentinel,"softmax prefix canary");
    check(workspace.doubles.empty() && workspace.indices.empty(),"softmax workspace growth");
  }
}
void wide_em_library_contracts() {
  constexpr std::size_t rows=33, states=3;
  constexpr double sentinel=-987654.125;
  for (std::size_t cols : {32,64,128,256}) {
    std::vector<double> x(rows*cols), means(states*cols), vars(states*cols,2.0);
    std::vector<double> weights(states,1.0/double(states)), transition(states*states,1.0/double(states));
    for (std::size_t i=0;i<x.size();++i) x[i]=std::sin(double(i)*0.17);
    std::copy(x.begin(),x.begin()+means.size(),means.begin());
    const auto saved=x;
    const auto matrix=[](const double *data,std::size_t n,std::size_t d) {
      o::Value value; value.data=data; value.shape=o::matrix_shape(n,d);
      value.set_contiguous_strides(); return value;
    };
    for (bool hmm : {false,true}) {
      o::Value args[7]; args[0]=matrix(x.data(),rows,cols);
      args[1].data=weights.data(); args[1].shape=o::vector_shape(states);
      args[1].set_contiguous_strides();
      const int mi=hmm ? 3 : 2;
      if (hmm) args[2]=matrix(transition.data(),states,states);
      args[mi]=matrix(means.data(),states,cols); args[mi+1]=matrix(vars.data(),states,cols);
      args[mi+2]=o::Value::number(5); args[mi+3]=o::Value::number(0);
      const auto prepared=o::prepare(o::lookup(hmm ? "gaussian_hmm_em" : "gaussian_mixture_em"),args,mi+4);
      const auto count=prepared.output_shape.size();
      std::vector<double> expected(count+2,sentinel),actual(count+2,sentinel);
      o::Workspace scalar_work,auto_work; o::Audit scalar_audit,auto_audit;
      o::Output output; output.shape=prepared.output_shape; output.data=expected.data()+1;
      o::execute(prepared,output,scalar_work,o::Isa::scalar,scalar_audit);
      output.data=actual.data()+1;
      o::execute(prepared,output,auto_work,o::Isa::automatic,auto_audit);
      check(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(double))==0,
            "wide EM differs from scalar or writes canaries");
      check(actual.front()==sentinel && actual.back()==sentinel,"wide EM canary");
      check(auto_work.capacity_bytes()==scalar_work.capacity_bytes(),"wide EM workspace growth");
      check(x==saved,"wide EM changed input");
    }
  }
}

void variance_library_failure_prefix() {
  constexpr std::size_t rows=33, cols=17;
  constexpr double sentinel=-987654.125;
  for (std::size_t column : {0,7,8,15,16}) {
    for (double bad : {std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity(),1e308}) {
      std::vector<double> x(rows*cols,1.0); x[(rows-1)*cols+column]=bad;
      o::Value args[3];args[0].data=x.data();args[0].shape=o::matrix_shape(rows,cols);
      args[0].set_contiguous_strides();args[1]=o::Value::number(0);args[2]=o::Value::number(0);
      const auto prepared=o::prepare(o::lookup("variance_axis"),args,3);
      std::vector<double> expected(cols+2,sentinel),actual(cols+2,sentinel);
      o::Workspace work; o::Audit audit;
      for (auto isa : {o::Isa::scalar,o::Isa::automatic}) {
        o::Output output;output.shape=prepared.output_shape;
        output.data=(isa==o::Isa::scalar ? expected : actual).data()+1;
        bool failed=false;
        try {o::execute(prepared,output,work,isa,audit);}
        catch (const std::exception &e) {failed=std::string(e.what())=="NON_FINITE_RESULT";}
        check(failed,"variance lost invalid/overflow error");
        check(work.capacity_bytes()==0,"variance introduced workspace");
      }
      check(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(double))==0,
            "variance changed failing output prefix");
      check(actual.front()==sentinel && actual.back()==sentinel,"variance canary");
    }
  }
}

void f_variance_library_failure_prefix() {
  constexpr std::size_t rows=33, cols=17;
  constexpr double sentinel=-987654.125;
  for (std::size_t column : {0,7,8,15,16}) {
    for (double bad : {std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity(),1e308}) {
      std::vector<double> x(rows*cols,1.0); x[column*rows+rows-1]=bad;
      o::Value args[3];args[0].data=x.data();args[0].shape=o::matrix_shape(rows,cols);
      args[0].stride={1,std::ptrdiff_t(rows),1};args[1]=o::Value::number(0);args[2]=o::Value::number(0);
      const auto prepared=o::prepare(o::lookup("variance_axis"),args,3);
      std::vector<double> expected(cols+2,sentinel),actual(cols+2,sentinel);
      o::Workspace work; o::Audit audit;
      for (auto isa : {o::Isa::scalar,o::Isa::automatic}) {
        o::Output output;output.shape=prepared.output_shape;
        output.data=(isa==o::Isa::scalar ? expected : actual).data()+1;
        bool failed=false;
        try {o::execute(prepared,output,work,isa,audit);}
        catch (const std::exception &e) {failed=std::string(e.what())=="NON_FINITE_RESULT";}
        check(failed,"variance lost invalid/overflow error");
        check(work.capacity_bytes()==0,"variance introduced workspace");
      }
      check(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(double))==0,
            "variance changed failing output prefix");
      check(actual.front()==sentinel && actual.back()==sentinel,"variance canary");
    }
  }
}


void f_column_pair_failure_prefix() {
  constexpr std::size_t rows=65,cols=17;
  constexpr double sentinel=-987654.125;
  for(std::size_t bad : {0,7,8,16}) for(bool left : {false,true}) {
    std::vector<double> x(rows*cols,3.),shift(cols,1.),scale(cols,2.);
    if(left) scale[bad]=0.; else x[bad*rows]=1.;
    o::Value input;input.data=x.data();input.shape=o::matrix_shape(rows,cols);input.stride={1,std::ptrdiff_t(rows),1};
    o::Value a=input;a.data=shift.data();a.stride={0,1,1};
    o::Value b=a;b.data=scale.data();
    o::Value initial[]{input,a};const auto first=o::prepare(o::lookup("subtract"),initial,2);
    auto middle=input;middle.set_contiguous_strides();
    o::Value final[]{left ? middle : b,left ? b : middle};
    const auto second=o::prepare(o::lookup("divide"),final,2);
    std::vector<double> expected(rows*cols+2,sentinel),actual(rows*cols+2,sentinel);
    bool reference_failed=false,actual_failed=false;
    try {for(std::size_t i=0;i<rows*cols;++i) {
      const auto m=o::scalar_math(o::Op::subtract,input.f(i),a.f(i));
      expected[i+1]=left ? o::scalar_math(o::Op::divide,m,b.f(i)) : o::scalar_math(o::Op::divide,b.f(i),m);
    }} catch(const std::exception &e) {reference_failed=std::string(e.what())=="DIVIDE_BY_ZERO";}
    o::Output output;output.shape=second.output_shape;output.data=actual.data()+1;
    try {o::elementwise_pair(first,second,output,left);}
    catch(const std::exception &e) {actual_failed=std::string(e.what())=="DIVIDE_BY_ZERO";}
    check(reference_failed && actual_failed,"column pair divide error");
    check(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(double))==0,
          "column pair changed failure prefix or canary");
  }
}

void f_column_pair_tile_boundaries() {
  constexpr double sentinel=-987654.125;
  const char *names[]={"add","subtract","multiply","divide"};
  for(const auto shape : {std::array<std::size_t,2>{32,8}, {33,12}, {36,13}, {35,15}}) {
    const auto rows=shape[0],cols=shape[1],pitch=rows+3;
    std::vector<double> x(pitch*cols+2,sentinel),p(cols),q(cols);
    for(std::size_t c=0;c<cols;++c) {
      p[c]=0.5+double(c)/64; q[c]=1.125+double(c)/128;
      for(std::size_t i=0;i<rows;++i) x[1+c*pitch+i]=2.+double((i*7+c*11)%31)/32;
    }
    const auto saved=x;
    o::Value input;input.data=x.data()+1;input.shape=o::matrix_shape(rows,cols);
    input.stride={1,std::ptrdiff_t(pitch),1};
    auto a=input;a.data=p.data()+cols-1;a.stride={0,-1,1};
    auto b=input;b.data=q.data();b.stride={0,1,1};
    for(int first=0;first<3;++first) for(int second=0;second<4;++second)
      for(bool source_left : {false,true}) for(bool middle_left : {false,true}) {
        o::Value initial[]{source_left ? input : a,source_left ? a : input};
        auto one=o::prepare(o::lookup(names[first]),initial,2);
        auto middle=input;middle.set_contiguous_strides();
        o::Value final[]{middle_left ? middle : b,middle_left ? b : middle};
        auto two=o::prepare(o::lookup(names[second]),final,2);
        std::vector<double> expected(rows*cols+2,sentinel),actual(rows*cols+2,sentinel);
        for(std::size_t i=0;i<rows*cols;++i) {
          const double m=source_left ? o::scalar_math(one.spec->op,input.f(i),a.f(i))
                                     : o::scalar_math(one.spec->op,a.f(i),input.f(i));
          expected[i+1]=middle_left ? o::scalar_math(two.spec->op,m,b.f(i))
                                    : o::scalar_math(two.spec->op,b.f(i),m);
        }
        o::Output output;output.shape=two.output_shape;output.data=actual.data()+1;
        o::elementwise_pair(one,two,output,middle_left);
        check(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(double))==0,
              "column pair tile/tail differs from scalar or overwrites canary");
        check(x==saved,"column pair tile wrote padded input");
      }
  }
}

void covariance_f_observer_contracts() {
  constexpr std::size_t rows=33, cols=9;
  constexpr std::uint64_t sentinel=0x5a7f31be19d04c62ULL;
  for (auto order : {o::MemoryOrder::c, o::MemoryOrder::f}) {
    for (const auto *name : {"covariance", "correlation"}) for (int pattern=0;pattern<3;++pattern) {
      if (std::strcmp(name,"correlation")==0 && pattern==1) continue; // zero variance is an error
      std::vector<double> source(rows*cols);
      o::Value input; input.data=source.data(); input.shape=o::matrix_shape(rows,cols);
      input.set_contiguous_strides(order);
      for (std::size_t i=0;i<source.size();++i)
        source[input.offset(i)]=pattern==1 ? -0. : double(int((i*7)%19)-9);
      if (pattern==2) source[input.offset(7)]=std::numeric_limits<double>::quiet_NaN();
      const auto before=source;
      auto prepared=o::prepare(o::lookup(name),&input,1);
      std::vector<double> expected(cols*cols);
      o::Output direct; direct.kind=prepared.output_kind; direct.shape=prepared.output_shape;
      direct.data=expected.data(); o::Workspace work; o::Audit direct_audit;
      o::execute(prepared,direct,work,o::Isa::automatic,direct_audit);
      auto compiled=c::compile({std::string(name)+"(x)"},std::vector<t::Variable>{{"x",t::ValueType::matrix()}},
          false,{}, {},0,100000000,true,{},g::OutputOrder::f);
      auto program=c::decode_program(c::encode_program(compiled->program));
      const std::int64_t start=0,end=rows;
      const auto layout=g::result_layout(program,{input},&start,&end,1);
      std::vector<std::uint64_t> storage((layout.bytes()+7)/8+2,sentinel);
      std::size_t observed=0;
      const auto audit=g::execute_observed(program,{input},nullptr,0,&start,&end,1,storage.data()+1,1,&layout,
          [&](std::optional<o::Op> op,const o::Value &value,bool result) {
            if (!result || op!=prepared.spec->op) return;
            check(value.shape==o::matrix_shape(cols,cols) && value.f_contiguous(),
                  "covariance observer did not receive actual F strides");
            for (std::size_t i=0;i<expected.size();++i) {
              const auto v=value.f(i);
              check(std::memcmp(&v,&expected[i],sizeof(v))==0,"F observer changed canonical covariance bits");
            }
            ++observed;
          });
      check(observed==1 && audit.numeric_arena_bytes==0 && audit.root_copy_bytes==0 &&
            audit.direct_output_bytes==expected.size()*sizeof(double),"F observer allocated or copied a root matrix");
      check(audit.accelerate_covariance_attempts==0,"observer entered guarded covariance candidate");
      check(storage.front()==sentinel && storage.back()==sentinel,"F covariance overwrote root canary");
      check(std::memcmp(source.data(),before.data(),source.size()*sizeof(double))==0,"F observer mutated source");
    }
  }
}

void covcorr_f_observer_contracts() {
  constexpr std::size_t rows=33, cols=9, elements=cols*cols;
  constexpr std::uint64_t sentinel=0x5a7f31be19d04c62ULL;
  for (auto order : {o::MemoryOrder::c, o::MemoryOrder::f}) {
    std::vector<double> source(rows*cols), expected(2*elements);
    o::Value input; input.data=source.data(); input.shape=o::matrix_shape(rows,cols);
    input.set_contiguous_strides(order);
    for (std::size_t i=0;i<source.size();++i)
      source[input.offset(i)]=double(int((i*7)%19)-9);
    const auto before=source;
    for (std::size_t r=0;r<2;++r) {
      auto prepared=o::prepare(o::lookup(r ? "correlation" : "covariance"),&input,1);
      o::Output out; out.kind=prepared.output_kind; out.shape=prepared.output_shape;
      out.data=expected.data()+r*elements; o::Workspace work; o::Audit audit;
      o::execute(prepared,out,work,o::Isa::automatic,audit);
    }
    auto compiled=c::compile({"covariance(x)","correlation(x)"},
        std::vector<t::Variable>{{"x",t::ValueType::matrix()}},false,{}, {},0,100000000,true,{},g::OutputOrder::f);
    auto program=c::decode_program(c::encode_program(compiled->program));
    const std::int64_t start=0,end=rows;
    const auto layout=g::result_layout(program,{input},&start,&end,1);
    std::vector<std::uint64_t> storage((layout.bytes()+7)/8+2,sentinel);
    std::size_t observed=0;
    const auto audit=g::execute_observed(program,{input},nullptr,0,&start,&end,1,storage.data()+1,2,&layout,
        [&](std::optional<o::Op> op,const o::Value &value,bool result) {
          if (!result || (op!=o::Op::covariance && op!=o::Op::correlation)) return;
          const auto root=op==o::Op::correlation ? 1 : 0;
          check(value.f_contiguous(),"paired observer lost F strides");
          for (std::size_t i=0;i<elements;++i) {
            const auto v=value.f(i);
            check(std::memcmp(&v,&expected[root*elements+i],sizeof(v))==0,"paired F observer changed bits");
          }
          ++observed;
        });
    check(observed==2 && audit.numeric_arena_bytes==0 && audit.root_copy_bytes==0 &&
          audit.direct_output_bytes==2*elements*sizeof(double),"paired F observer allocated root matrices");
    check(audit.accelerate_covariance_attempts==0,"paired observer entered guarded library");
    check(source==before && storage.front()==sentinel && storage.back()==sentinel,
          "paired F observer changed input or canary");
  }
}

void covariance_descending_rows_contracts() {
  constexpr std::size_t rows=65, cols=257, pitch=2*(cols+3), elements=cols*cols;
  constexpr std::uint64_t sentinel=0x5a7f31be19d04c62ULL;
  for (bool reverse_f : {false,true}) {
    std::vector<double> source((reverse_f ? cols*(rows+3) : rows*pitch)+2, -91.0), expected(2*elements);
    auto *data=source.data()+1+(rows-1)*(reverse_f ? 1 : pitch);
    o::Value input; input.data=data;
    input.shape=o::matrix_shape(rows,cols);
    input.stride=reverse_f ? std::array<std::ptrdiff_t,3>{-1,rows+3,1}
                            : std::array<std::ptrdiff_t,3>{-static_cast<std::ptrdiff_t>(pitch),1,1};
    std::uint64_t random=90317;
    for (std::size_t i=0;i<rows*cols;++i) {
      random=random*6364136223846793005ULL+1442695040888963407ULL;
      data[input.offset(i)]=double(random>>40)*0x1p-24-.5;
    }
    const auto before=source;
    for (std::size_t root=0;root<2;++root) {
      auto prepared=o::prepare(o::lookup(root ? "correlation" : "covariance"),&input,1);
      o::Output out; out.kind=prepared.output_kind; out.shape=prepared.output_shape;
      out.data=expected.data()+root*elements; o::Workspace work; o::Audit audit;
      o::execute(prepared,out,work,o::Isa::automatic,audit);
      if (reverse_f) {
        std::vector<double> scalar(elements);
        auto reference=out; reference.data=scalar.data();
        o::Audit scalar_audit;
        o::execute(prepared,reference,work,o::Isa::scalar,scalar_audit);
        check(std::memcmp(out.data,scalar.data(),elements*sizeof(double))==0,
              "descending F ordered library changed canonical bits");
        if (o::covariance_accelerate_reserve_doubles(input.shape))
          check(audit.eigen_covariance_calls==1,"descending F bypassed ordered Eigen tiles");
      }
    }
    for (auto order : {g::OutputOrder::c,g::OutputOrder::f}) {
      auto compiled=c::compile({"covariance(x)","correlation(x)"},
          std::vector<t::Variable>{{"x",t::ValueType::matrix()}},false,{}, {},0,100000000,true,{},order);
      auto program=c::decode_program(c::encode_program(compiled->program));
      const std::int64_t start=0,end=rows;
      const auto layout=g::result_layout(program,{input},&start,&end,1);
      std::vector<std::uint64_t> storage((layout.bytes()+7)/8+2,sentinel);
      const auto audit=g::execute(program,{input},nullptr,0,&start,&end,1,storage.data()+1,2,&layout);
      if (o::covariance_accelerate_reserve_doubles(input.shape))
        check(audit.accelerate_covariance_calls==1,"descending rows bypassed guarded library");
      for (std::size_t root=0;root<2;++root) {
        const auto *actual=reinterpret_cast<const double *>(
            reinterpret_cast<const std::uint8_t *>(storage.data()+1)+layout.slots[root].byte_offset);
        for (std::size_t i=0;i<elements;++i) {
          const auto value=actual[i], reference=expected[root*elements+i];
          check(std::abs(value-reference)<=1e-12+1e-12*std::abs(reference) &&
                std::signbit(value)==std::signbit(reference) && (value==0)==(reference==0),
                "descending rows changed covariance/correlation contract");
          if (i/cols==i%cols)
            check(std::memcmp(&value,&reference,sizeof(value))==0,"descending rows changed diagonal");
        }
      }
      check(source==before && storage.front()==sentinel && storage.back()==sentinel,
            "descending rows changed source or output canary");
    }
  }
}

void cholesky_f_observer_contracts() {
  constexpr std::size_t n=3, elements=n*n;
  constexpr std::uint64_t sentinel=0x47a917bf80ce21d6ULL;
  for (bool covariance_pair : {false,true}) {
    const std::vector<double> source=covariance_pair
        ? std::vector<double>{1.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.}
        : std::vector<double>{4.,2.,1.,2.,5.,.5,1.,.5,3.};
    const auto saved=source;
    o::Value input; input.data=source.data(); input.shape=o::matrix_shape(covariance_pair ? n+1 : n,n);
    input.set_contiguous_strides();
    o::Value factor_input=input;
    std::vector<double> covariance(elements);
    o::Workspace work; o::Audit direct_audit;
    if (covariance_pair) {
      const auto plan=o::prepare(o::lookup("covariance"),&input,1);
      o::Output out; out.kind=plan.output_kind; out.shape=plan.output_shape; out.data=covariance.data();
      o::execute(plan,out,work,o::Isa::scalar,direct_audit);
      factor_input.data=covariance.data(); factor_input.shape=out.shape;
      factor_input.set_contiguous_strides();
    }
    const auto prepared=o::prepare(o::lookup("cholesky"),&factor_input,1);
    std::vector<double> expected(elements);
    o::Output direct; direct.kind=prepared.output_kind; direct.shape=prepared.output_shape;
    direct.data=expected.data();
    o::execute(prepared,direct,work,o::Isa::scalar,direct_audit);
    const auto type=covariance_pair ? t::ValueType::matrix({"time","asset"},{"T","N"})
                                    : t::ValueType::matrix({"asset","asset"},{"N","N"});
    auto compiled=c::compile({covariance_pair ? "cholesky(covariance(x))" : "cholesky(x)"},
        std::vector<t::Variable>{{"x",type}},
        false,{}, {},0,100000000,true,{},g::OutputOrder::f);
    auto program=c::decode_program(c::encode_program(compiled->program));
    const std::int64_t start=0,end=covariance_pair ? n+1 : 0;
    const auto layout=g::result_layout(program,{input},&start,&end,1);
    std::vector<std::uint64_t> storage((layout.bytes()+7)/8+2,sentinel);
    std::size_t observed=0,observed_covariance=0;
    const auto audit=g::execute_observed(program,{input},nullptr,0,&start,&end,1,storage.data()+1,1,&layout,
        [&](std::optional<o::Op> op,const o::Value &value,bool result) {
          if (result && op==o::Op::covariance) {
            check(covariance_pair && value.shape==o::matrix_shape(n,n) && value.c_contiguous(),
                  "F covariance pair observer lost its canonical covariance descriptor");
            for (std::size_t i=0;i<elements;++i) {
              const auto v=value.f(i);
              check(std::memcmp(&v,&covariance[i],sizeof(v))==0,
                    "F covariance pair observer changed covariance bits");
            }
            ++observed_covariance;
          }
          if (!result || op!=o::Op::cholesky) return;
          check(value.shape==o::matrix_shape(n,n) && value.c_contiguous() && !value.f_contiguous(),
                "Cholesky observer did not restore the canonical C arena descriptor");
          for (std::size_t i=0;i<elements;++i) {
            const auto v=value.f(i);
            check(std::memcmp(&v,&expected[i],sizeof(v))==0,
                  "Cholesky observer changed canonical C bits");
          }
          ++observed;
        });
    check(observed==1 && observed_covariance==std::size_t(covariance_pair) &&
          audit.numeric_arena_bytes>=elements*sizeof(double) &&
          audit.root_copy_bytes==elements*sizeof(double) && audit.direct_output_bytes==0,
          "F Cholesky observer did not restore its budgeted C arena and root copy");
    check(layout.order(0)==o::MemoryOrder::f,"Cholesky observer changed public F order");
    const auto *payload=reinterpret_cast<const double *>(
        reinterpret_cast<const unsigned char *>(storage.data()+1)+layout.slots[0].byte_offset);
    for (std::size_t i=0;i<n;++i) for (std::size_t j=0;j<n;++j)
      check(std::memcmp(payload+j*n+i,&expected[i*n+j],sizeof(double))==0,
            "Cholesky observer F materialization changed canonical bits");
    check(storage.front()==sentinel && storage.back()==sentinel,
          "F Cholesky observer overwrote root canary");
    // The downgrade owns separate metadata; the next ordinary call must still
    // use the original direct F root, rather than retaining the observer arena.
    const auto ordinary=g::execute(program,{input},nullptr,0,&start,&end,1,storage.data()+1,1,&layout);
    check(ordinary.numeric_arena_bytes==0 && ordinary.root_copy_bytes==0 &&
          ordinary.direct_output_bytes==elements*sizeof(double),
          "Cholesky observer downgrade leaked into later ordinary execution");
    for (std::size_t i=0;i<n;++i) for (std::size_t j=0;j<n;++j)
      check(std::memcmp(payload+j*n+i,&expected[i*n+j],sizeof(double))==0,
            "Cholesky ordinary F root differs after observer downgrade");
    check(storage.front()==sentinel && storage.back()==sentinel &&
          std::memcmp(source.data(),saved.data(),source.size()*sizeof(double))==0,
          "F covariance pair observer recovery changed input or canary");
  }
}

void transcendental_cross_layout_contracts() {
  for (const auto op : {o::Op::exp, o::Op::log}) for (const bool input_f : {false, true}) {
    const auto &spec=o::lookup(op == o::Op::exp ? "exp" : "log");
    constexpr std::size_t rows=129,d=33,k=17,n=rows*d*k;
    const std::size_t pitch=input_f ? rows+3 : d*k+3;
    std::vector<double> backing(input_f ? pitch*d*k : pitch*rows,
                                std::numeric_limits<double>::quiet_NaN());
    o::Value x; x.shape=o::tensor_shape(rows,d,k); x.data=backing.data();
    x.stride=input_f ? std::array<std::ptrdiff_t,3>{1,std::ptrdiff_t(pitch),std::ptrdiff_t(pitch*d)} :
                      std::array<std::ptrdiff_t,3>{std::ptrdiff_t(pitch),std::ptrdiff_t(k),1};
    for (std::size_t i=0;i<n;++i) backing[x.offset(i)]=op == o::Op::exp ? double(i%17)/8-1 : double(i%17+1)/8;
    const auto prepared=o::prepare(spec,&x,1);
    std::vector<double> guarded(n+2,912.);
    o::Output out; out.shape=x.shape; out.data=guarded.data()+1;
    out.order=input_f ? o::MemoryOrder::c : o::MemoryOrder::f;
    o::Workspace work; o::Audit audit;
    const bool accepted=o::try_accelerate_transcendental(prepared,out,work,audit);
    if (accepted) {
      check(audit.accelerate_vforce_calls>0 && audit.vector_elements==n,"cross-layout actual calls");
      check(audit.algorithm_copy_bytes==2*n*sizeof(double),"cross-layout copy audit");
      check(work.capacity_bytes()<=65536 && work.capacity_bytes()>0,"cross-layout scratch bound");
      check(work.capacity_bytes()<=o::transcendental_accelerate_scratch_doubles(x.shape)*8,
            "cross-layout shape reserve underestimated");
      for (std::size_t i=0;i<n;++i) {
        const auto row=i/(d*k), col=(i/k)%d, plane=i%k;
        const auto offset=input_f ? i : row+(col+d*plane)*rows;
        const double expected=op == o::Op::exp ? std::exp(x.f(i)) : std::log(x.f(i));
        check(std::abs(guarded[1+offset]-expected)<=2e-14+2e-12*std::abs(expected),
              "cross-layout tensor coordinates");
      }
    } else check(o::transcendental_accelerate_scratch_doubles(x.shape)==0,
                 "eligible cross-layout rejected on enabled backend");
    check(guarded.front()==912. && guarded.back()==912.,"cross-layout result canary");
    std::fill(guarded.begin(),guarded.end(),912.);
    backing[x.offset(n-1)]=std::numeric_limits<double>::quiet_NaN();
    check(!o::try_accelerate_transcendental(prepared,out,work,audit),"cross-layout accepted late fault");
    check(std::all_of(guarded.begin(),guarded.end(),[](double v){return v==912.;}),
          "cross-layout wrote before consumed-payload validation");
    backing[x.offset(n-1)]=.5;
    const int rounding=std::fegetround(); std::fesetround(FE_UPWARD);
    const bool rounded=o::try_accelerate_transcendental(prepared,out,work,audit);
    std::fesetround(rounding);
    check(!rounded,"cross-layout accepted non-default rounding");
    check(std::all_of(guarded.begin(),guarded.end(),[](double v){return v==912.;}),
          "cross-layout rounding rejection wrote result");
  }
}
void transcendental_library_contracts() {
  for (const auto op : {o::Op::exp, o::Op::log}) {
    const auto &spec=o::lookup(op == o::Op::exp ? "exp" : "log");
    // Padded first-axis columns, deliberately only double-aligned output.
    std::vector<double> backing(2*1034, -9876.), guarded(2*1031+2, 912.);
    for (std::size_t j=0;j<2;++j) for (std::size_t i=0;i<1031;++i)
      backing[j*1034+i]=op == o::Op::exp ? double(i%17)/8-1 : double(i%17+1)/8;
    o::Value x; x.shape=o::matrix_shape(1031,2); x.data=backing.data(); x.stride={1,1034,1};
    const auto prepared=o::prepare(spec,&x,1);
    o::Output out; out.shape=prepared.output_shape; out.data=guarded.data()+1; out.order=o::MemoryOrder::f;
    o::Workspace work; o::Audit audit;
    const bool accepted=o::try_accelerate_transcendental(prepared,out,work,audit);
    if (accepted) {
      check(audit.accelerate_vforce_calls==2 && audit.vector_elements==2062,"vForce actual run audit");
      check(work.capacity_bytes()==0,"vForce allocated numerical scratch");
      for (std::size_t j=0;j<2;++j) for (std::size_t i=0;i<1031;++i) {
        const double v=backing[j*1034+i];
        const double expected=op == o::Op::exp ? std::exp(v) : std::log(v);
        check(std::abs(guarded[1+j*1031+i]-expected)<=1e-12+1e-12*std::abs(expected),"vForce column coordinates");
      }
    }
    check(guarded.front()==912. && guarded.back()==912.,"vForce result canary");
    std::fill(guarded.begin(),guarded.end(),912.);
    backing[1034+1030]=std::numeric_limits<double>::quiet_NaN();
    check(!o::try_accelerate_transcendental(prepared,out,work,audit),"vForce accepted late fault");
    check(std::all_of(guarded.begin(),guarded.end(),[](double v){return v==912.;}),"vForce wrote before full validation");
    backing[1034+1030]=.5;
    const int rounding=std::fegetround();
    std::fesetround(FE_UPWARD);
    const bool rounded=o::try_accelerate_transcendental(prepared,out,work,audit);
    std::fesetround(rounding);
    check(!rounded,"vForce accepted unqualified rounding mode");
    check(std::all_of(guarded.begin(),guarded.end(),[](double v){return v==912.;}),"vForce rounding rejection wrote output");

    const auto compiled=c::compile({op == o::Op::exp ? "exp(x)" : "log(x)"},
        std::vector<t::Variable>{{"x",t::ValueType::matrix()}});
    const std::vector<o::Value> inputs{x}; const std::int64_t start=0,end=1031;
    const auto layout=g::result_layout(compiled->program,inputs,&start,&end,1);
    std::vector<std::uint64_t> bytes((layout.bytes()+7)/8);
    std::size_t observed=0;
    const auto observed_audit=g::execute_observed(compiled->program,inputs,nullptr,0,&start,&end,1,
        bytes.data(),1,&layout,[&](std::optional<o::Op> operation,const o::Value &value,bool result) {
          if (result && operation==op) {
            for (std::size_t i=0;i<x.size();++i) {
              const double expected=op == o::Op::exp ? std::exp(x.f(i)) : std::log(x.f(i));
              check(value.f(i)==expected,"observer lost canonical transcendental math");
            }
            ++observed;
          }
        });
    check(observed==1 && observed_audit.accelerate_vforce_calls==0,"observer entered private vForce path");
  }
}
void gaussian_filter_stream_observer_contracts() {
  constexpr std::size_t rows=43691, cols=2, states=3;
  constexpr std::uint64_t canary=0x8a31759fe14b3c02ULL;
  std::vector<double> x(rows*cols), means(states*cols), variances(states*cols, .7);
  std::vector<double> initial{.2,.3,.5}, transition{.6,.2,.2,.1,.7,.2,.2,.1,.7};
  for (std::size_t i=0;i<x.size();++i) x[i]=.01*double(int(i%29)-14);
  for (std::size_t i=0;i<means.size();++i) means[i]=.1*double(int(i)-2);
  const auto before=x;
  const auto matrix=[](const std::vector<double> &a,std::size_t n,std::size_t d) {
    o::Value v;v.data=a.data();v.shape=o::matrix_shape(n,d);v.set_contiguous_strides();return v;
  };
  const auto vector=[](const std::vector<double> &a) {
    o::Value v;v.data=a.data();v.shape=o::vector_shape(a.size());return v;
  };
  const std::vector<o::Value> inputs{matrix(x,rows,cols),matrix(means,states,cols),
      matrix(variances,states,cols),vector(initial),matrix(transition,states,states)};
  const auto run=[](const char *name,const std::vector<o::Value> &args) {
    const auto prepared=o::prepare(o::lookup(name),args.data(),args.size());
    std::vector<double> result(prepared.output_shape.size());
    o::Output out;out.shape=prepared.output_shape;out.kind=prepared.output_kind;out.data=result.data();
    o::Workspace work;o::Audit audit;o::execute(prepared,out,work,o::Isa::automatic,audit);return result;
  };
  const auto density=run("gaussian_log_density",{inputs[0],inputs[1],inputs[2]});
  const auto filtered=run("hmm_filter",{matrix(density,rows,states),inputs[3],inputs[4]});
  auto component=t::ValueType::vector("3");component.axes={"component"};
  const auto model=t::ValueType::matrix({"component","factor"},{"3","2"});
  const auto compiled=c::compile({"sum_axis(slice_axis(hmm_filter(gaussian_log_density(x,m,v),i,t),0,43690,43691,1),0)"},
      std::vector<t::Variable>{{"x",t::ValueType::matrix({"time","factor"},{"43691","2"})},
      {"m",model},{"v",model},{"i",component},{"t",t::ValueType::matrix({"component","component"},{"3","3"})}},
      false,{}, {},0,100000000,true);
  const auto program=c::decode_program(c::encode_program(compiled->program));
  const std::int64_t start=0,end=rows;
  const auto layout=g::result_layout(program,inputs,&start,&end,1);
  std::vector<std::uint64_t> output((layout.bytes()+7)/8+2,canary);
  std::size_t observed=0;
  const auto audit=g::execute_observed(program,inputs,nullptr,0,&start,&end,1,output.data()+1,1,&layout,
      [&](std::optional<o::Op> op,const o::Value &value,bool result) {
        if (!result || (op!=o::Op::gaussian_log_density && op!=o::Op::hmm_filter)) return;
        const auto &expected=op==o::Op::hmm_filter ? filtered : density;
        check(value.shape==o::matrix_shape(rows,states),"observer lost full Gaussian/filter geometry");
        for (std::size_t j=0;j<expected.size();++j) {
          const auto actual=value.f(j);
          check(std::memcmp(&actual,&expected[j],sizeof(actual))==0,"observer changed Gaussian/filter history");
        }
        ++observed;
      });
  check(observed==2 && audit.gaussian_filter_streams==0,"observer entered private Gaussian stream");
  const auto *tail=reinterpret_cast<const std::uint8_t *>(output.data()+1)+layout.slots[0].byte_offset;
  check(std::memcmp(tail,filtered.data()+(rows-1)*states,states*sizeof(double))==0,"observed filter tail changed");
  const auto ordinary=g::execute(program,inputs,nullptr,0,&start,&end,1,output.data()+1,1,&layout);
  check(ordinary.gaussian_filter_streams==1,"normal observer control did not enter stream");
  check(std::memcmp(tail,filtered.data()+(rows-1)*states,states*sizeof(double))==0,"ordinary stream tail changed");
  check(x==before && output.front()==canary && output.back()==canary,"stream observer mutated input or output canary");
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  try {
    gaussian_filter_stream_observer_contracts();
    transcendental_cross_layout_contracts();
    transcendental_library_contracts();
    state_root_loan_native_contracts();
    latent_unwritten_result_native_contracts();
    dense_layout_contracts(argv[1]);
    result_order_contracts(argv[1]);
    foundation_contracts();
    softmax_library_contracts();
    wide_em_library_contracts();
    variance_library_failure_prefix();
    f_variance_library_failure_prefix();
    f_column_pair_failure_prefix();
    f_column_pair_tile_boundaries();
    covariance_reuse_contracts();
    covariance_f_observer_contracts();
    covcorr_f_observer_contracts();
    covariance_descending_rows_contracts();
    cholesky_f_observer_contracts();
    {
      auto index_type=t::ValueType::vector(); index_type.dtype=t::DType::int64;
      auto compiled=c::compile({"sum_axis(take_axis(x,ids,0),0)","(1-x)/2"},
          std::vector<t::Variable>{{"x",t::ValueType::matrix()},{"ids",index_type}});
      auto mixed=c::decode_program(c::encode_program(compiled->program));
      std::vector<double> x{1,2,3,4,5,6,7,8,9,10,11,12};
      std::vector<std::int64_t> ids{3,1};
      o::Value a; a.data=x.data(); a.shape=o::matrix_shape(4,3); a.set_contiguous_strides();
      o::Value ix; ix.kind=o::Kind::integer; ix.data=ids.data(); ix.shape=o::vector_shape(2);
      const std::vector<o::Value> inputs{a,ix};
      std::int64_t start=0,end=4;
      const auto layout=g::result_layout(mixed,inputs,&start,&end,1);
      std::vector<std::uint64_t> output((layout.bytes()+7)/8);
      auto audit=g::execute(mixed,inputs,nullptr,0,&start,&end,1,output.data(),2,&layout);
      check(audit.numeric_arena_bytes==0 && audit.fused_pointwise_calls==2,
            "mixed fusion did not eliminate intermediate arrays");
      const auto *bytes=reinterpret_cast<const std::uint8_t *>(output.data());
      const auto *sum=reinterpret_cast<const double *>(bytes+layout.slots[0].byte_offset);
      const auto *values=reinterpret_cast<const double *>(bytes+layout.slots[1].byte_offset);
      for (std::size_t j=0;j<3;++j) check(sum[j]==14+2*j,"take-reduce fusion value");
      for (std::size_t j=0;j<12;++j) check(values[j]==-double(j)/2,"mixed arithmetic value");
      std::size_t observed=0;
      g::execute_observed(mixed,inputs,nullptr,0,&start,&end,1,output.data(),2,&layout,
          [&](std::optional<o::Op> op,const o::Value &value,bool result) {
            if (result && op==o::Op::take_axis) {
              check(value.shape==o::matrix_shape(2,3) && value.f(0)==10 && value.f(3)==4,
                    "observer lost canonical intermediate");
              ++observed;
            }
          });
      check(observed==1,"observer did not see selection");
      ids[0]=4;
      rejects([&] {g::execute(mixed,inputs,nullptr,0,&start,&end,1,output.data(),2,&layout);},
              "fused selection skipped bounds checks");
    }

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
