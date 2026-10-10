#pragma once
#include "calmetrics_engine/graph.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
namespace take_scatter_native_cases {
namespace g=calmetrics_engine::graph;namespace o=calmetrics_engine::ops;
g::Node input(unsigned id){g::Node n;n.kind=g::NodeKind::input;n.input_index=id;return n;}
g::Node constant(double v){g::Node n;n.constant=v;return n;}
g::Node operation(o::Op op,std::initializer_list<unsigned> parents,unsigned slot){g::Node n;n.kind=g::NodeKind::operation;n.opcode=static_cast<unsigned>(op);n.storage=g::StorageKind::numeric;n.slot=slot;n.parent_count=parents.size();unsigned i=0;for(auto id:parents)n.parents[i++]=id;return n;}
inline void run(){
 g::Program p;p.input_count=3;p.input_axes={1,1,1};p.numeric_slots=2;p.output_kind=g::OutputKind::typed;
 p.nodes={input(0),input(1),input(2),constant(1),constant(3),constant(0),operation(o::Op::take_axis,{0,1,3},0),operation(o::Op::scatter_rows,{6,2,4,5},1)};
 p.roots={7};p.root_outputs={{g::OutputDType::float64,2,g::OutputOrder::c}};p.finalize();
 std::vector<double> numbers{1,2,3,4};std::vector<std::int64_t> integers{1,2,3,4},indices{0,1};std::vector<unsigned char> masks{0,1,0,1};
 o::Value x;x.kind=o::Kind::number;x.data=numbers.data();x.shape=o::matrix_shape(2,2);x.set_contiguous_strides();
 o::Value ix;ix.kind=o::Kind::integer;ix.data=indices.data();ix.shape=o::vector_shape(2);ix.set_contiguous_strides();
 const std::int64_t bound=0;const auto layout=g::result_layout(p,{x,ix,ix},&bound,&bound,1);
 for(auto kind:{o::Kind::number,o::Kind::integer,o::Kind::mask}){
  x.kind=kind;x.data=kind==o::Kind::number?static_cast<const void*>(numbers.data()):kind==o::Kind::integer?static_cast<const void*>(integers.data()):static_cast<const void*>(masks.data());
  std::vector<o::Value> values{x,ix,ix};std::vector<double> out((layout.bytes()+7)/8,987.),reference;std::string expected,actual;
  auto run=[&](bool observe){try{if(observe)g::execute_observed(p,values,nullptr,0,&bound,&bound,1,out.data(),1,&layout,[](std::optional<o::Op>,const o::Value&,bool){});else g::execute(p,values,nullptr,0,&bound,&bound,1,out.data(),1,&layout);return std::string("OK");}catch(const o::Error&e){return std::string(e.what());}};
  expected=run(true);reference=out;std::fill(out.begin(),out.end(),987.);actual=run(false);
  const auto required=kind==o::Kind::number?"OK":"GRAPH_STORAGE_KIND";
  if(expected!=required || actual!=expected || std::memcmp(out.data(),reference.data(),out.size()*8)!=0)
    throw std::runtime_error("private take/scatter changed raw dtype error or prefix: "+expected+" / "+actual);
 }
}

} // namespace take_scatter_native_cases
