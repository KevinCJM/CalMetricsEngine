#pragma once
#include "calmetrics_engine/graph.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include "calmetrics_engine/native_runtime.hpp"
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace matmul_statistics_cases {
inline void check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
inline void run() {
  namespace g=calmetrics_engine::graph;
  namespace o=calmetrics_engine::ops;
  namespace n=calmetrics_engine::native;
  g::Program p; p.input_count=2; p.input_axes={1,1}; p.numeric_slots=3;
  p.output_kind=g::OutputKind::typed;
  p.root_outputs={{g::OutputDType::float64,1,g::OutputOrder::c},
                  {g::OutputDType::float64,1,g::OutputOrder::c}};
  auto input=[](unsigned index) { g::Node n; n.kind=g::NodeKind::input; n.input_index=index; return n; };
  auto literal=[](double v) { g::Node n; n.kind=g::NodeKind::constant; n.constant=v; return n; };
  auto op=[](o::Op code,std::initializer_list<std::uint32_t> parents,unsigned slot) {
    g::Node node; node.kind=g::NodeKind::operation; node.opcode=static_cast<std::uint16_t>(code);
    node.parent_count=parents.size(); node.storage=g::StorageKind::numeric; node.slot=slot;
    std::size_t index=0;
    for (auto parent:parents) node.parents[index++]=parent;
    return node;
  };
  p.nodes={input(0),input(1),op(o::Op::matmul,{0,1},0),literal(1),
           op(o::Op::mean_axis,{2,3},1),literal(0),op(o::Op::variance_axis,{2,3,5},2)};
  p.roots={4,6}; p.finalize();
  check(p.execution_metadata->matmul_summaries.size()==1,"private statistics proof missing");
  std::vector<double> w(32*2,.5),r(2*32),result(64);
  for (std::size_t i=0;i<32;++i) { r[i]=double(i)/1000; r[32+i]=double(i)/2000; }
  const auto view=[](const std::vector<double> &data,std::size_t a,std::size_t b) {
    o::Value v;v.kind=o::Kind::number;v.shape=o::matrix_shape(a,b);v.data=data.data();v.set_contiguous_strides();return v;
  };
  std::vector<o::Value> values{view(w,32,2),view(r,2,32)};
  std::vector<o::Shape> shapes;
  g::required_array_capacity(p,values,0,&shapes);
  check(g::matmul_summary_geometry(p,2,shapes),"shared summary geometry missing");
  const std::int64_t begin=0,end=0;
  const auto audit=g::execute(p,values,nullptr,0,&begin,&end,1,result.data(),2);
  check(audit.numeric_arena_bytes<32*32*8,"private trajectory was materialized");
  double sum=0,squared=0;
  for (std::size_t i=0;i<32;++i) sum+=double(i)*.00075;
  const double mean=sum/32;
  for (std::size_t i=0;i<32;++i) { const auto d=double(i)*.00075-mean;squared+=d*d; }
  for (std::size_t i=0;i<32;++i) {
    check(std::abs(result[i]-mean)<4e-5,"mean differs from independent formula");
    check(std::abs(result[32+i]-squared/32)<4e-5,"variance differs from independent formula");
  }
  std::atomic<bool> cancelled{true}; bool caught=false;
  try { g::execute(p,values,nullptr,0,&begin,&end,1,result.data(),2,nullptr,0,
                    n::Deadline::max(),&cancelled); }
  catch (const n::Timeout &) { caught=true; }
  check(caught,"statistics did not honour cancellation");
  cancelled=false;
  g::execute(p,values,nullptr,0,&begin,&end,1,result.data(),2,nullptr,0,n::Deadline::max(),&cancelled);
  bool saw_trajectory=false;
  const auto observer=[&](std::optional<o::Op> opcode,const o::Value &value,bool output) {
    if (output && opcode && *opcode==o::Op::matmul) {
      check(value.shape==o::matrix_shape(32,32),"observer geometry changed");
      check(std::abs(value.at(31,31)-.02325)<1e-15,"observer received placeholder values");
      saw_trajectory=true;
    }
  };
  g::execute_observed(p,values,nullptr,0,&begin,&end,1,result.data(),2,nullptr,observer);
  check(saw_trajectory,"observer skipped logical matrix");
  g::release_graph_workspace();
}

inline void run_quantiles() {
  namespace g=calmetrics_engine::graph;namespace o=calmetrics_engine::ops;
  g::Program p;p.input_count=3;p.input_axes={1,1,1};p.numeric_slots=3;
  p.output_kind=g::OutputKind::typed;
  p.root_outputs={{g::OutputDType::float64,1,g::OutputOrder::c},
                  {g::OutputDType::float64,2,g::OutputOrder::c}};
  auto input=[](unsigned index){g::Node node;node.kind=g::NodeKind::input;node.input_index=index;return node;};
  auto literal=[](double value){g::Node node;node.kind=g::NodeKind::constant;node.constant=value;return node;};
  auto op=[](o::Op code,std::initializer_list<std::uint32_t> parents,unsigned slot){
    g::Node node;node.kind=g::NodeKind::operation;node.opcode=static_cast<std::uint16_t>(code);
    node.parent_count=parents.size();node.storage=g::StorageKind::numeric;node.slot=slot;
    std::size_t i=0;for (auto parent:parents)node.parents[i++]=parent;return node;
  };
  // q is deliberately a later static input, as in ordinary AST construction.
  p.nodes={input(0),input(1),op(o::Op::matmul,{0,1},0),literal(1),
           op(o::Op::mean_axis,{2,3},1),input(2),op(o::Op::quantiles_axis,{2,5,3},2)};
  p.roots={4,6};p.finalize();
  check(p.execution_metadata->matmul_summaries.size()==1,"quantile physical proof missing");
  check(p.execution_metadata->matmul_summaries[0].quantiles.size()==1,"quantile consumer missing");
  std::vector<double>w(65*2,.5),r(2*63),q{0.,.25,1.},result(65*4);
  for(std::size_t i=0;i<63;++i){r[i]=double(i)/1000;r[63+i]=double(i)/2000;}
  auto matrix=[](const std::vector<double>&data,std::size_t rows,std::size_t cols){
    o::Value value;value.kind=o::Kind::number;value.shape=o::matrix_shape(rows,cols);
    value.data=data.data();value.set_contiguous_strides();return value;
  };
  o::Value probabilities;probabilities.kind=o::Kind::number;probabilities.shape=o::vector_shape(q.size());
  probabilities.data=q.data();probabilities.set_contiguous_strides();
  std::vector<o::Value>values{matrix(w,65,2),matrix(r,2,63),probabilities};
  const std::int64_t start=0,end=0;
  const auto audit=g::execute(p,values,nullptr,0,&start,&end,1,result.data(),2);
  check(audit.numeric_arena_bytes<65*63*8,"quantile materialized full trajectory");
  for(std::size_t row=0;row<65;++row){
    check(std::abs(result[row]-.02325)<4e-5,"quantile-task mean wrong");
    check(std::abs(result[65+row*3])<4e-5,"quantile q0 wrong");
    check(std::abs(result[65+row*3+1]-.011625)<4e-5,"quantile interpolation wrong");
    check(std::abs(result[65+row*3+2]-.0465)<4e-5,"quantile q1 wrong");
  }
  bool observed=false;
  const auto observer=[&](std::optional<o::Op>code,const o::Value&value,bool output){
    if(output&&code&&*code==o::Op::matmul){
      check(value.shape==o::matrix_shape(65,63),"observed quantile matrix geometry wrong");
      check(std::abs(value.at(64,62)-.0465)<1e-15,"observed quantile matrix is placeholder");observed=true;
    }
  };
  g::execute_observed(p,values,nullptr,0,&start,&end,1,result.data(),2,nullptr,observer);
  check(observed,"quantile observer missing");
  // The producer must not write a later malformed root before that root's
  // original capacity validation. Keep a canary-filled output in ASan runs.
  auto malformed=g::result_layout(p,values,&start,&end,1);
  malformed.slots[1].capacity-=1;
  const auto payload=malformed.row_bytes.back()/sizeof(double);
  result.resize(payload+8);
  std::fill(result.begin(),result.end(),-777.);
  bool capacity_caught=false;
  try{g::execute(p,values,nullptr,0,&start,&end,1,result.data(),2,&malformed);}
  catch(const o::Error&error){capacity_caught=std::string(error.what())=="RESULT_CAPACITY_MISMATCH";}
  check(capacity_caught,"direct quantile root skipped original capacity validation");
  // Typed execute clears the payload first. No mathematical producer values
  // may enter this malformed root, and no byte may enter the trailing guards.
  for(std::size_t i=malformed.slots[1].byte_offset/sizeof(double);i<payload;++i)
    check(result[i]==0.,"producer wrote malformed quantile root before validation");
  for(std::size_t i=payload;i<result.size();++i)
    check(result[i]==-777.,"malformed quantile root overwrote trailing guard");
  g::Program independent;independent.input_count=3;independent.input_axes={1,1,1};
  independent.output_kind=g::OutputKind::typed;independent.numeric_slots=5;
  independent.root_outputs={{g::OutputDType::float64,1,g::OutputOrder::c},
      {g::OutputDType::float64,1,g::OutputOrder::c},{g::OutputDType::float64,2,g::OutputOrder::c}};
  independent.nodes={input(0),input(1),op(o::Op::matmul,{0,1},0),literal(1),
      op(o::Op::mean_axis,{2,3},1),op(o::Op::sum_axis,{0,3},2),literal(0),
      op(o::Op::divide,{5,6},3),input(2),op(o::Op::quantiles_axis,{2,8,3},4)};
  independent.roots={4,7,9};independent.finalize();
  auto late=g::result_layout(independent,values,&start,&end,1);late.slots[2].capacity-=1;
  const auto failed_payload=late.row_bytes.back()/sizeof(double);
  std::vector<double> failed(failed_payload+8,-777.);bool original_error=false;
  try{g::execute(independent,values,nullptr,0,&start,&end,1,failed.data(),3,&late);}
  catch(const o::Error&error){original_error=std::string(error.what())=="DIVIDE_BY_ZERO";}
  check(original_error,"later root capacity changed independent error priority");
  for(std::size_t i=late.slots[2].byte_offset/sizeof(double);i<failed_payload;++i)
    check(failed[i]==0.,"producer wrote later malformed root before independent error");
  for(std::size_t i=failed_payload;i<failed.size();++i)
    check(failed[i]==-777.,"later malformed quantile root overwrote trailing guard");
  q[1]=std::numeric_limits<double>::quiet_NaN();bool caught=false;
  try{g::execute(p,values,nullptr,0,&start,&end,1,result.data(),2);}
  catch(const o::Error&error){caught=std::string(error.what())=="INVALID_PARAMETER";}
  check(caught,"quantile invalid probability was lost");g::release_graph_workspace();

  // Capacity must cover every shorter interval, even when floor(target/T)
  // increases its batch width. Exercise the million-observation bound too.
  const auto a=o::matrix_shape(65,3);
  const auto upper=o::matmul_quantiles_scratch_doubles(a,o::matrix_shape(3,100000));
  check(upper>=64*2520+64*3,"shorter interval exceeds quantile scratch bound");
  check(upper<170000,"quantile upper bound retained full trajectories");
  const auto longest=o::matmul_quantiles_scratch_doubles(a,o::matrix_shape(3,1000000));
  check(longest==1000000+64*3,"million-observation scratch is not one bounded line");
}
} // namespace matmul_statistics_cases
