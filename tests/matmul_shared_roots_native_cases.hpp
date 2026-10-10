#pragma once
#include "matmul_scalar_chain_native_cases.hpp"
#include <cstring>
#include <limits>
#include <stdexcept>
namespace matmul_shared_roots_native_cases {
namespace g=calmetrics_engine::graph;
namespace o=calmetrics_engine::ops;
using matmul_scalar_chain_native_cases::input;
using matmul_scalar_chain_native_cases::constant;
using matmul_scalar_chain_native_cases::operation;
inline g::Program program(g::OutputOrder order,bool reverse=false) {
 g::Program p;p.input_count=3;p.input_axes={1,1,1};p.numeric_slots=3;
 p.output_kind=g::OutputKind::typed;
 p.root_outputs={{g::OutputDType::float64,2,order},{g::OutputDType::float64,2,order}};
 p.nodes={input(0),input(1),operation(o::Op::matmul,0,1,0),constant(.5),
          operation(o::Op::multiply,2,3,1),constant(.25),operation(o::Op::add,4,5,1),
          input(2),operation(o::Op::multiply,2,7,2),constant(.125),
          operation(o::Op::subtract,8,9,2)};
 p.roots=reverse ? std::vector<std::uint32_t>{10,6} : std::vector<std::uint32_t>{6,10};p.finalize();
 // Exercise the direct raw writer. Row-batch children have a different,
 // separately covered whole-tile commit prefix.
 auto direct=std::make_shared<g::ExecutionMetadata>(*p.execution_metadata);
 direct->row_batch.reset();p.execution_metadata=std::move(direct);
 if(p.execution_metadata->matmul_shared_chains.size()!=2 ||
    p.execution_metadata->matmul_shared_backing_root!=(reverse?0:1))
  throw std::runtime_error("shared MatMul backing not proven from logical last branch");
 return p;
}
inline void run() {
 constexpr std::size_t m=256,k=64,n=256;
 std::vector<double> a(m*k,1.),b(k*n,1.);
 o::Value x,y;x.data=a.data();x.shape=o::matrix_shape(m,k);x.set_contiguous_strides();
 y.data=b.data();y.shape=o::matrix_shape(k,n);y.set_contiguous_strides();
 const auto before_a=a,before_b=b;
 for(auto order:{g::OutputOrder::automatic,g::OutputOrder::c,g::OutputOrder::f})
 for(bool reverse:{false,true})
 for(std::size_t rows:{1,2})
 for(int fault=0;fault<5;++fault) {
  auto p=program(order,reverse);auto canonical=p;
  auto metadata=std::make_shared<g::ExecutionMetadata>(*p.execution_metadata);
  metadata->matmul_shared_chains.clear();metadata->matmul_shared_backing_root=-1;
  canonical.execution_metadata=std::move(metadata);
  auto scale=o::Value::number(.25);
  if(fault==1){scale.kind=o::Kind::integer;scale.integer=1;}
  const std::vector<o::Value> inputs{x,y,scale};
  const std::int64_t starts[2]={0,0},ends[2]={0,0};
  auto layout=g::result_layout(p,inputs,starts,ends,rows);
  if(fault==2) {layout.slots[reverse?0:1].capacity-=1;}
  if(fault==3) {layout.orders.assign(2,o::MemoryOrder::c);layout.orders.set(1,o::MemoryOrder::f);}
  if(fault==4) {
   // Padding cannot use lazy clearing: every unused byte keeps its canonical zero.
   for(std::size_t row=0;row<rows;++row) {
    layout.slots[row*2].byte_offset+=row*16;
    layout.slots[row*2].capacity+=1;
    layout.slots[row*2+1].byte_offset+=row*16+8;
   }
   for(std::size_t row=1;row<=rows;++row)layout.row_bytes[row]+=row*16;
  }
  std::vector<double> expected(layout.bytes()/8,987.),actual(expected.size(),987.);
  const auto execute=[&](const g::Program &graph,std::vector<double> &out) {
   try{g::execute(graph,inputs,nullptr,0,starts,ends,rows,out.data(),2,&layout);}
   catch(const o::Error &e){return std::string(e.what());}
   return std::string("OK");
  };
  const auto error=execute(canonical,expected),optimized_error=execute(p,actual);
  if(error!=optimized_error || std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(double)) ||
     a!=before_a || b!=before_b)
   throw std::runtime_error("shared product changed raw failure prefix, root order, mixed layout or inputs");
  if(fault==1 && (actual[layout.slots[reverse?1:0].byte_offset/8]!=32.25 ||
                 actual[layout.slots[reverse?0:1].byte_offset/8]!=0.))
   { std::cerr<<"late-fault order="<<int(order)<<" reverse="<<reverse<<" error="<<error
      <<" first="<<actual[layout.slots[reverse?1:0].byte_offset/8]
      <<" last="<<actual[layout.slots[reverse?0:1].byte_offset/8]
      <<" rowbatch="<<bool(p.execution_metadata->row_batch)<<'\n';
     throw std::runtime_error("late failed leaf erased completed shared root or exposed private product"); }
 }
 // Match the existing library-qualified single-root stop contract. Small/raw
 // fallback kernels have no guaranteed internal checkpoint, so do not pretend
 // that an expired clock must interrupt those paths.
 for(auto order:{g::OutputOrder::automatic,g::OutputOrder::c,g::OutputOrder::f})
 for(bool reverse:{false,true})
 for(std::size_t rows:{1,2})
 for(bool cancelled:{false,true}) {
  constexpr std::size_t sm=512,sk=256,sn=512;
  std::vector<double> left(sm*sk,1.),right(sk*sn,1.);
  o::Value sx,sy;sx.data=left.data();sx.shape=o::matrix_shape(sm,sk);sx.set_contiguous_strides();
  sy.data=right.data();sy.shape=o::matrix_shape(sk,sn);sy.set_contiguous_strides();
  auto p=program(order,reverse),single=p;
  single.nodes.resize(7);single.roots={6};single.root_outputs.resize(1);single.finalize();
  auto metadata=std::make_shared<g::ExecutionMetadata>(*single.execution_metadata);
  metadata->row_batch.reset();single.execution_metadata=std::move(metadata);
  if(single.execution_metadata->matmul_scalar_chain.empty())
   throw std::runtime_error("legacy single-root cancellation control was not admitted");
  const std::vector<o::Value> inputs{sx,sy,o::Value::number(.25)};
  const std::int64_t starts[2]={0,0},ends[2]={0,0};
  const auto layout=g::result_layout(p,inputs,starts,ends,rows);
  const auto control_layout=g::result_layout(single,inputs,starts,ends,rows);
  std::vector<double> expected(control_layout.bytes()/8,987.),actual(layout.bytes()/8,987.);
  std::atomic<bool> stop(cancelled);
  const auto deadline=cancelled ? std::chrono::steady_clock::time_point::max()
      : std::chrono::steady_clock::now()-std::chrono::seconds(1);
  const auto execute=[&](const g::Program &graph,std::vector<double> &out,const g::ResultLayout &result) {
   try{g::execute(graph,inputs,nullptr,0,starts,ends,rows,out.data(),graph.roots.size(),&result,0,deadline,&stop);}
   catch(const calmetrics_engine::native::Timeout &e){return std::string(e.what());}
   return std::string("OK");
  };
  const auto error=execute(single,expected,control_layout),optimized_error=execute(p,actual,layout);
  const bool library=o::matmul_accelerate_reserve_doubles(sx.shape,sy.shape)>0;
  if(error!=optimized_error || (library && error=="OK"))
   throw std::runtime_error("shared-root stop differs from existing qualified single-root contract");
  if(library) {
   const auto zero=[](double v){return v==0. && !std::signbit(v);};
   if(!std::all_of(expected.begin(),expected.end(),zero) || !std::all_of(actual.begin(),actual.end(),zero))
    throw std::runtime_error("shared-root early stop exposed an unpublished product or uninitialized row");
  } else {
   for(std::size_t row=0;row<rows;++row)
    for(std::size_t root=0;root<2;++root) {
     const auto *data=actual.data()+layout.slots[row*2+root].byte_offset/8;
     const auto value=(root==(reverse?1u:0u)) ? sk*.5+.25 : sk*.25-.125;
     if(!std::all_of(data,data+sm*sn,[&](double v){return v==value;}))
      throw std::runtime_error("noninterruptible fallback changed its completed output");
    }
  }
  if(!std::all_of(left.begin(),left.end(),[](double v){return v==1.;}) ||
     !std::all_of(right.begin(),right.end(),[](double v){return v==1.;}))
   throw std::runtime_error("early stop changed shared inputs");
 }
 // Mixed declared orders are legitimate native schemas. Planner and
 // executor must reserve the same canonical storage before admission.
 for(auto first:{g::OutputOrder::automatic,g::OutputOrder::c,g::OutputOrder::f})
 for(auto second:{g::OutputOrder::automatic,g::OutputOrder::c,g::OutputOrder::f})
 for(bool reverse:{false,true}) {
  if(first==second) continue;
  auto p=program(first,reverse);p.root_outputs[1].order=second;p.finalize();
  if(!p.execution_metadata->matmul_shared_chains.empty())
   throw std::runtime_error("mixed declared root orders omitted canonical Planner arena");
  auto direct=std::make_shared<g::ExecutionMetadata>(*p.execution_metadata);
  direct->row_batch.reset();p.execution_metadata=std::move(direct);
  const std::vector<o::Value> inputs{x,y,o::Value::number(.25)};
  std::vector<o::Shape> shapes;const auto capacity=g::required_array_capacity(p,inputs,0,&shapes);
  g::ArrayArenaLayout arena;g::plan_array_arena(p,capacity,shapes,arena,true);
  const std::int64_t z[1]={0};const auto layout=g::result_layout(p,inputs,z,z,1);
  std::vector<double> out(layout.bytes()/8,987.);
  const auto audit=g::execute(p,inputs,nullptr,0,z,z,1,out.data(),2,&layout);
  if(!arena.elements[0] || audit.numeric_arena_bytes>arena.elements[0]*sizeof(double))
   throw std::runtime_error("mixed root layout execution exceeds planned canonical arena");
  for(std::size_t root=0;root<2;++root) {
   const auto value=(root==(reverse?1u:0u)) ? 32.25 : 15.875;
   const auto *data=out.data()+layout.slots[root].byte_offset/8;
   if(!std::all_of(data,data+m*n,[&](double v){return v==value;}))
    throw std::runtime_error("mixed declared root orders changed canonical values");
  }
 }
 // A raw interleaved backing branch cannot overwrite a still-live producer.
 {
  auto p=program(g::OutputOrder::c);
  // Construct a valid interleaving rather than changing parent chronology.
  p.nodes={input(0),input(1),operation(o::Op::matmul,0,1,0),constant(.5),
           operation(o::Op::multiply,2,3,1),input(2),operation(o::Op::multiply,2,5,2),
           constant(.25),operation(o::Op::add,4,7,1),constant(.125),operation(o::Op::subtract,6,9,2)};
  p.roots={8,10};p.finalize();
  if(!p.execution_metadata->matmul_shared_chains.empty())
   throw std::runtime_error("interleaved last branch can overwrite a shared product before another root consumes it");
 }
 {
  auto p=program(g::OutputOrder::c);
  if(g::matrix_workspace_doubles(p,{},256,true)<o::matmul_accelerate_reserve_doubles(
      o::matrix_shape(1024,256),o::matrix_shape(256,1024)))
   throw std::runtime_error("unknown shared matrix geometry omits library reserve");
 }
}
}
