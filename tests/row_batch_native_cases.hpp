#pragma once
#include "calmetrics_engine/graph.hpp"
#include "calmetrics_engine/native_runtime.hpp"
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace row_batch_native_cases {
inline void check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
inline void run() {
  namespace g = calmetrics_engine::graph;
  namespace o = calmetrics_engine::ops;
  namespace n = calmetrics_engine::native;
  const auto input = [](unsigned i) { g::Node x; x.kind=g::NodeKind::input; x.input_index=i; return x; };
  const auto constant = [](double v) { g::Node x; x.kind=g::NodeKind::constant; x.constant=v; return x; };
  const auto op = [](o::Op code, std::initializer_list<std::uint32_t> parents, unsigned slot) {
    g::Node x; x.kind=g::NodeKind::operation; x.opcode=std::uint16_t(code);
    x.parent_count=parents.size(); x.storage=g::StorageKind::numeric; x.slot=slot;
    unsigned j=0; for (auto p:parents) x.parents[j++]=p; return x;
  };
  g::Program p; p.input_count=2; p.input_axes={1,2}; p.numeric_slots=8;
  p.output_kind=g::OutputKind::typed;
  p.root_outputs={{g::OutputDType::float64,1,g::OutputOrder::c}};
  p.nodes={input(0),input(1),op(o::Op::transpose,{1},0),op(o::Op::matmul,{0,2},1),
    constant(1),op(o::Op::add,{4,3},2),op(o::Op::cumulative_product_axis,{5,4,4},3),
    op(o::Op::cumulative_max_axis,{6,4,4},4),op(o::Op::divide,{6,7},5),
    op(o::Op::subtract,{4,8},6),op(o::Op::max_axis,{9,4},7)};
  p.roots={10}; p.finalize();
  check(bool(p.execution_metadata->row_batch),"row partition proof missing");
  constexpr std::size_t rows=193, times=257, assets=7;
  std::vector<double> weights(rows*assets), returns(times*assets);
  for(std::size_t i=0;i<weights.size();++i) weights[i]=(.1+double(i%13)/100)/assets;
  for(std::size_t i=0;i<returns.size();++i) returns[i]=double(int(i%19)-9)/1000;
  const auto matrix=[](const std::vector<double> &v,std::size_t r,std::size_t c) {
    o::Value x; x.kind=o::Kind::number; x.shape=o::matrix_shape(r,c); x.data=v.data();
    x.set_contiguous_strides(); return x;
  };
  std::vector<o::Value> values{matrix(weights,rows,assets),matrix(returns,times,assets)};
  const auto original_weights=weights, original_returns=returns;
  const std::int64_t start=0,end=times;
  auto layout=g::result_layout(p,values,&start,&end,1);
  const auto payload=layout.bytes()/sizeof(double);
  std::vector<double> output(payload+8,-777), canonical(payload+8,-777);
  const auto observer=[](std::optional<o::Op>,const o::Value &,bool) {};
  const auto audit=g::execute(p,values,nullptr,0,&start,&end,1,output.data(),1,&layout);
  const auto full=g::execute_observed(p,values,nullptr,0,&start,&end,1,canonical.data(),1,&layout,observer);
  check(audit.max_window==times,"row partition window audit wrong");
  check(audit.numeric_arena_bytes<full.numeric_arena_bytes,"row partition restored full arena");
  for(std::size_t i=0;i<payload;++i) check(output[i]==canonical[i],"row partition math changed");
  for(std::size_t i=payload;i<output.size();++i) check(output[i]==-777,"row partition overwrote guard");
  check(weights==original_weights && returns==original_returns,"row partition mutated input");

  // A selected row fragment writes only its own part of the global root and
  // keeps global metadata. Guards exercise the same entry used by threads.
  std::size_t failed_node = 0;
  std::fill(output.begin(),output.end(),-777);
  const auto fragment=g::execute_row_batch_range(p,values,nullptr,0,&start,&end,
      output.data(),layout,10,rows,true,failed_node,n::Deadline::max(),nullptr,false);
  check(fragment.result_shapes.size()==1 && fragment.result_shapes[0]==o::vector_shape(rows),
      "row fragment published a local shape");
  check(fragment.root_copy_bytes==0 && fragment.direct_output_bytes==(rows-10)*8,
      "row fragment copied or wrote the wrong result extent");
  for(std::size_t i=0;i<output.size();++i)
    check(output[i]==(i>=10 && i<rows ? canonical[i] : -777),
        "row fragment wrote outside its selected range");
  const auto reject_range=[&](std::size_t first,std::size_t last,const g::ResultLayout &destination) {
    std::fill(output.begin(),output.end(),-777); bool rejected=false;
    try {g::execute_row_batch_range(p,values,nullptr,0,&start,&end,output.data(),
        destination,first,last,false,failed_node,n::Deadline::max(),nullptr,false);}
    catch(const o::Error &) {rejected=true;}
    check(rejected,"malformed row fragment was admitted");
    check(failed_node==std::numeric_limits<std::size_t>::max(),
        "malformed row fragment reported a stale numerical node");
    for(const auto value:output) check(value==-777,"malformed row fragment wrote output");
  };
  reject_range(0,127,layout); reject_range(0,rows+1,layout);
  auto range_layout=layout; --range_layout.slots[0].capacity;
  reject_range(0,rows,range_layout);
  // Empty contracting axes may have either a null or nonnull input pointer.
  // Row clipping must not flatten an empty shape (which would divide by zero).
  for (bool null_pointer : {false,true}) for (auto row_stride : {std::ptrdiff_t(0),
      std::numeric_limits<std::ptrdiff_t>::max(), std::numeric_limits<std::ptrdiff_t>::min()}) {
    auto empty=values; empty[0].shape=o::matrix_shape(rows,0); empty[0].set_contiguous_strides();
    empty[1].shape=o::matrix_shape(times,0); empty[1].set_contiguous_strides();
    if(null_pointer) empty[0].data=empty[1].data=nullptr;
    empty[0].stride[0]=row_stride;
    const auto empty_layout=g::result_layout(p,empty,&start,&end,1);
    std::fill(output.begin(),output.end(),-777);
    const auto part=g::execute_row_batch_range(p,empty,nullptr,0,&start,&end,
        output.data(),empty_layout,10,rows,false,failed_node,n::Deadline::max(),nullptr,false);
    check(part.result_shapes.empty() && part.root_statuses.empty(),"secondary fragment published metadata");
    for(std::size_t i=0;i<rows;++i) check(output[i]==(i<10 ? -777 : 0),"empty fragment value or extent wrong");
    for(std::size_t i=rows;i<output.size();++i) check(output[i]==-777,"empty fragment overwrote guard");
  }

  auto malformed=layout; --malformed.slots[0].capacity;
  std::fill(output.begin(),output.end(),-777);
  bool capacity_failed=false;
  try {g::execute(p,values,nullptr,0,&start,&end,1,output.data(),1,&malformed);}
  catch(const o::Error &e) {capacity_failed=std::string(e.what())=="RESULT_CAPACITY_MISMATCH";}
  check(capacity_failed,"row partition bypassed root capacity");
  for(std::size_t i=payload;i<output.size();++i) check(output[i]==-777,"malformed root overwrote guard");

  values[0].data=weights.data()+(rows-1)*assets; values[0].stride[0]=-std::ptrdiff_t(assets);
  g::execute(p,values,nullptr,0,&start,&end,1,output.data(),1,&layout);
  for(std::size_t i=0;i<rows;++i) check(output[i]==canonical[rows-1-i],"negative row stride changed correspondence");
  std::atomic<bool> cancelled{true}; bool stopped=false;
  try {g::execute(p,values,nullptr,0,&start,&end,1,output.data(),1,&layout,0,n::Deadline::max(),&cancelled);}
  catch(const n::Timeout &) {stopped=true;}
  check(stopped,"row partition ignored cancellation");
  cancelled=false;
  g::execute(p,values,nullptr,0,&start,&end,1,output.data(),1,&layout,0,n::Deadline::max(),&cancelled);
  for(std::size_t i=0;i<rows;++i) check(output[i]==canonical[rows-1-i],"cancelled partition did not recover");
  // Raw native callers can supply a malformed scalar bool without Python's
  // normalization. Fusion must run the canonical payload preflight even on
  // an empty trajectory, at the original selection node.
  g::Program masked;
  masked.input_count=3; masked.input_axes={1,2,1}; masked.numeric_slots=4;
  masked.output_kind=g::OutputKind::typed;
  masked.root_outputs={{g::OutputDType::float64,1,g::OutputOrder::c}};
  masked.nodes={input(0),input(1),input(2),op(o::Op::transpose,{1},0),
      op(o::Op::matmul,{0,3},1),constant(0),op(o::Op::where,{2,4,5},2),
      constant(1),op(o::Op::sum_axis,{6,7},3)};
  masked.roots={8}; masked.finalize();
  check(bool(masked.execution_metadata->row_batch),"scalar mask batching proof missing");
  auto invalid=o::Value::number(2); invalid.kind=o::Kind::mask;
  for (const auto count : {times,std::size_t(0)}) {
    auto right=matrix(returns,count,assets);
    std::vector<o::Value> args{matrix(weights,rows,assets),right,invalid};
    const std::int64_t stop=count;
    const auto shape=g::result_layout(masked,args,&start,&stop,1);
    std::vector<double> destination(shape.bytes()/8+8,-777);
    bool failed=false;
    try {g::execute(masked,args,nullptr,0,&start,&stop,1,destination.data(),1,&shape);}
    catch(const o::Error &e) {failed=std::string(e.what())=="INVALID_MASK";}
    check(failed,"private where skipped scalar INVALID_MASK");
    for(std::size_t j=shape.bytes()/8;j<destination.size();++j)
      check(destination[j]==-777,"private mask failure overwrote canary");
    args[2].scalar=1;
    g::execute(masked,args,nullptr,0,&start,&stop,1,destination.data(),1,&shape);
    if (!count) for(std::size_t j=0;j<rows;++j)
      check(destination[j]==0,"empty private sum did not recover after invalid mask");
  }
  g::release_graph_workspace();
}
}
