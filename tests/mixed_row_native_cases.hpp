#pragma once
#include "calmetrics_engine/graph.hpp"
#include <climits>
#include <stdexcept>
#include <vector>

namespace mixed_row_native_cases {
inline void check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
inline void run() {
  namespace g = calmetrics_engine::graph;
  namespace o = calmetrics_engine::ops;
  const auto input = [](unsigned index) { g::Node x; x.kind=g::NodeKind::input; x.input_index=index; return x; };
  const auto constant = [](double value) { g::Node x; x.kind=g::NodeKind::constant; x.constant=value; return x; };
  const auto op = [](o::Op code, std::initializer_list<std::uint32_t> parents, unsigned slot) {
    g::Node x; x.kind=g::NodeKind::operation; x.opcode=std::uint16_t(code);
    x.parent_count=parents.size(); x.storage=g::StorageKind::numeric; x.slot=slot;
    unsigned j=0; for (auto parent:parents) x.parents[j++]=parent; return x;
  };
  constexpr std::size_t rows=257, times=129;
  std::vector<double> weights(rows*7), returns(times*7);
  for (std::size_t i=0;i<weights.size();++i) weights[i]=double(i%5)*.125;
  for (std::size_t i=0;i<returns.size();++i) returns[i]=double(int(i%7)-3)*.0625;
  const auto matrix=[](const double *data,std::size_t r,std::size_t c) {
    o::Value x; x.kind=o::Kind::number; x.shape=o::matrix_shape(r,c);
    x.data=data; x.stride[0]=c; x.stride[1]=1; return x;
  };
  const std::int64_t start=0,end=times;
  for (const auto order : {g::OutputOrder::c,g::OutputOrder::f})
  for (const auto columns : {std::size_t(0),std::size_t(7)})
  for (const bool reverse : {false,true}) {
    g::Program p; p.input_count=2; p.input_axes={1,2}; p.numeric_slots=4;
    p.output_kind=g::OutputKind::typed;
    p.root_outputs={{g::OutputDType::float64,2,order},{g::OutputDType::float64,1,g::OutputOrder::c}};
    p.nodes={input(0),input(1),constant(2),op(o::Op::multiply,{0,2},0),
             op(o::Op::transpose,{1},1),op(o::Op::matmul,{3,4},2),
             constant(1),op(o::Op::sum_axis,{5,6},3)};
    p.roots={3,7}; p.finalize();
    check(bool(p.execution_metadata->row_batch),"mixed row proof missing");
    double empty=3.;
    auto left=matrix(columns ? weights.data() : &empty,rows,columns);
    if (reverse && columns) {
      left.data=weights.data()+(rows-1)*columns;
      left.stride[0]=-std::ptrdiff_t(columns);
    } else if (!columns) left.stride[0]=reverse ? PTRDIFF_MIN : PTRDIFF_MAX;
    auto right=matrix(columns ? returns.data() : &empty,times,columns);
    std::vector<o::Value> args{left,right};
    auto layout=g::result_layout(p,args,&start,&end,1);
    const auto elements=layout.bytes()/sizeof(double);
    std::vector<double> serial(elements+8,-777.), ranged(elements+8,-777.);
    auto audit=g::execute(p,args,nullptr,0,&start,&end,1,serial.data(),2,&layout);
    check(audit.result_shapes[0]==o::matrix_shape(rows,columns),"mixed root shape lost");
    check(audit.result_shapes[1]==o::vector_shape(rows),"mixed vector shape lost");
    std::size_t failed=0;
    const auto deadline=std::chrono::steady_clock::time_point::max();
    auto a=g::execute_row_batch_range(p,args,nullptr,0,&start,&end,ranged.data(),layout,0,129,true,failed,deadline,nullptr,false);
    auto b=g::execute_row_batch_range(p,args,nullptr,0,&start,&end,ranged.data(),layout,129,257,false,failed,deadline,nullptr,false);
    check(a.result_shapes==audit.result_shapes && b.result_shapes.empty(),"mixed range metadata mismatch");
    for (std::size_t i=0;i<elements;++i) check(ranged[i]==serial[i],"F full-stride range misplaced");
    for (std::size_t i=0;i<rows;++i) {
      for (std::size_t j=0;j<columns;++j) {
        const auto index=order==g::OutputOrder::f ? j*rows+i : i*columns+j;
        check(serial[index]==2.*left.at(i,j),"mixed matrix value changed");
      }
      double expected=0.;
      for (std::size_t t=0;t<times;++t) {
        double value=0.;
        for (std::size_t k=0;k<columns;++k) value+=(left.at(i,k)*2.)*returns[t*columns+k];
        expected+=value;
      }
      check(serial[layout.slots[1].byte_offset/8+i]==expected,"mixed vector fold changed");
    }
    for (std::size_t i=elements;i<serial.size();++i)
      check(serial[i]==-777. && ranged[i]==-777.,"mixed output canary overwritten");
    if (columns) {
      auto invalid=layout; invalid.slots[0].capacity=rows;
      std::vector<double> untouched(elements+8,-777.);
      bool rejected=false;
      try {g::execute_row_batch_range(p,args,nullptr,0,&start,&end,untouched.data(),invalid,0,129,true,failed,deadline,nullptr,false);}
      catch (const o::Error &) {rejected=true;}
      check(rejected,"matrix root capacity mismatch admitted");
      for (const auto x:untouched) check(x==-777.,"invalid range wrote before validation");
    }
  }
}
}
