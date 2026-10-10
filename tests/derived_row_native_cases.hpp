#pragma once
#include "calmetrics_engine/graph.hpp"
#include <climits>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace derived_row_native_cases {
inline void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
inline void run() {
  namespace g = calmetrics_engine::graph;
  namespace o = calmetrics_engine::ops;
  const auto input = [](unsigned i) { g::Node x; x.kind=g::NodeKind::input; x.input_index=i; return x; };
  const auto constant = [](double v) { g::Node x; x.kind=g::NodeKind::constant; x.constant=v; return x; };
  const auto op = [](o::Op code, std::initializer_list<std::uint32_t> parents, unsigned slot) {
    g::Node x; x.kind=g::NodeKind::operation; x.opcode=std::uint16_t(code);
    x.parent_count=parents.size(); x.storage=g::StorageKind::numeric; x.slot=slot;
    unsigned j=0; for (auto p:parents) x.parents[j++]=p; return x;
  };
  g::Program p; p.input_count=2; p.input_axes={1,2}; p.numeric_slots=4;
  p.output_kind=g::OutputKind::typed;
  p.root_outputs={{g::OutputDType::float64,1,g::OutputOrder::c}};
  p.nodes={input(0),input(1),constant(2),op(o::Op::multiply,{0,2},0),
           op(o::Op::transpose,{1},1),op(o::Op::matmul,{3,4},2),
           constant(1),op(o::Op::sum_axis,{5,6},3)};
  p.roots={7}; p.finalize();
  require(bool(p.execution_metadata->row_batch), "derived row proof missing");
  require(p.execution_metadata->row_batch->input==0, "wrong derived source");
  constexpr std::size_t rows=257, times=129, columns=7;
  std::vector<double> weights(rows*columns), returns(times*columns);
  for (std::size_t i=0;i<weights.size();++i) weights[i]=double(i%5)*.125;
  for (std::size_t i=0;i<returns.size();++i) returns[i]=double(int(i%7)-3)*.0625;
  const auto matrix=[](const double *data,std::size_t r,std::size_t c) {
    o::Value x; x.kind=o::Kind::number; x.shape=o::matrix_shape(r,c);
    x.data=data; x.stride[0]=c; x.stride[1]=1; return x;
  };
  for (const bool reverse : {false,true}) {
    auto left=matrix(weights.data(),rows,columns);
    if (reverse) {left.data=weights.data()+(rows-1)*columns; left.stride[0]=-std::ptrdiff_t(columns);}
    std::vector<o::Value> args{left,matrix(returns.data(),times,columns)};
    const std::int64_t start=0,end=times;
    auto layout=g::result_layout(p,args,&start,&end,1);
    std::vector<double> out(layout.bytes()/8+8,-777.);
    g::execute(p,args,nullptr,0,&start,&end,1,out.data(),1,&layout);
    for (std::size_t i=0;i<rows;++i) {
      double expected=0.;
      for (std::size_t t=0;t<times;++t) {
        double value=0.;
        for (std::size_t k=0;k<columns;++k) value+=(left.at(i,k)*2.)*returns[t*columns+k];
        expected+=value;
      }
      require(out[i]==expected,"derived native fold changed");
    }
    for (std::size_t i=layout.bytes()/8;i<out.size();++i) require(out[i]==-777.,"derived result canary overwritten");
  }
  // Empty raw storage with non-null data and extreme strides must never be advanced.
  double sentinel=3.;
  for (const auto stride : {std::ptrdiff_t(PTRDIFF_MAX),std::ptrdiff_t(PTRDIFF_MIN)}) {
    auto left=matrix(&sentinel,rows,0); left.stride[0]=stride;
    auto right=matrix(&sentinel,times,0);
    const std::int64_t start=0,end=times;
    std::vector<o::Value> args{left,right};
    auto layout=g::result_layout(p,args,&start,&end,1);
    std::vector<double> out(layout.bytes()/8+8,-777.);
    g::execute(p,args,nullptr,0,&start,&end,1,out.data(),1,&layout);
    for (std::size_t i=0;i<rows;++i) require(out[i]==0. && !std::signbit(out[i]),"empty derived matrix changed zero");
    for (std::size_t i=layout.bytes()/8;i<out.size();++i) require(out[i]==-777.,"empty derived canary overwritten");
    require(sentinel==3.,"empty input mutated");
  }
  g::release_graph_workspace();
}
}
