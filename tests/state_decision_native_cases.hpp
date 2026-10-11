#pragma once
#include "calmetrics_engine/graph.hpp"
#include <vector>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <cstring>
namespace state_decision_native_cases {
namespace g=calmetrics_engine::graph; namespace o=calmetrics_engine::ops;
inline g::Node input(unsigned id){g::Node n;n.kind=g::NodeKind::input;n.input_index=id;return n;}
inline g::Node constant(double v){g::Node n;n.constant=v;return n;}
inline g::Node operation(o::Op op,std::initializer_list<unsigned> parents,g::StorageKind storage,unsigned slot){g::Node n;n.kind=g::NodeKind::operation;n.opcode=static_cast<unsigned>(op);n.storage=storage;n.slot=slot;n.parent_count=parents.size();unsigned j=0;for(auto id:parents)n.parents[j++]=id;return n;}
inline void run(){
 g::Program p;p.input_count=2;p.input_axes={1,1};p.mask_slots=3;p.integer_slots=2;p.output_kind=g::OutputKind::typed;
 p.nodes={input(0),constant(2),operation(o::Op::equal,{0,1},g::StorageKind::mask,0),constant(1),operation(o::Op::equal,{0,3},g::StorageKind::mask,1),constant(0),operation(o::Op::equal,{0,5},g::StorageKind::mask,2),constant(-1),input(1),operation(o::Op::state_select,{6,5,7,8},g::StorageKind::integer,0),operation(o::Op::state_select,{4,3,9,8},g::StorageKind::integer,1),operation(o::Op::state_select,{2,1,10,8},g::StorageKind::integer,0)};
 p.roots={11};p.root_outputs={{g::OutputDType::int64,1,g::OutputOrder::c}};p.finalize();
 if(p.execution_metadata->state_decisions.size()!=1)throw std::runtime_error("state decision not recognized");
 std::vector<std::int64_t> integer_source{0,1,2,3,2};
 const std::int64_t bound=0;std::vector<double>x{0,1,2,3,2};std::vector<unsigned char>valid{1,1,1,1,0};
 o::Value a;a.kind=o::Kind::number;a.data=x.data();a.shape=o::vector_shape(5);a.set_contiguous_strides();o::Value v;v.kind=o::Kind::mask;v.data=valid.data();v.shape=a.shape;v.set_contiguous_strides();
 const auto layout=g::result_layout(p,{a,v},&bound,&bound,1);std::vector<std::int64_t>out((layout.bytes()+7)/8),canonical(out.size());
 auto audit=g::execute(p,{a,v},nullptr,0,&bound,&bound,1,out.data(),1,&layout);
 g::execute_observed(p,{a,v},nullptr,0,&bound,&bound,1,canonical.data(),1,&layout,[](std::optional<o::Op>,const o::Value&,bool){});
 if(audit.fused_pointwise_calls!=1 || audit.mask_arena_bytes || audit.numeric_arena_bytes ||std::memcmp(out.data(),canonical.data(),out.size()*8))throw std::runtime_error("state decision fusion/parity failed");
 for(int fault=0;fault<10;++fault){
  auto candidate=p;auto source=a,available=v;auto bad=valid;
  if(fault==0){bad[2]=2;available.data=bad.data();}
  if(fault==1)candidate.nodes[3].constant=1.5;
  if(fault==2){source.kind=o::Kind::integer;source.data=integer_source.data();candidate.nodes[3].constant=1.5;}
  if(fault==3){source.data=nullptr;}
  if(fault==4){available.shape=o::vector_shape(4);}
  if(fault==5){source.data=nullptr;candidate.nodes[3].constant=1.5;}
  if(fault==6){available.data=nullptr;candidate.nodes[3].constant=1.5;}
  if(fault==7){available.data=nullptr;}
  if(fault==8){source.data=nullptr;bad[2]=2;available.data=bad.data();}
  if(fault==9){available.data=nullptr;candidate.nodes[5].constant=0.5;}
  candidate.finalize();
  auto capture=[&](bool observe,std::vector<std::int64_t>& result){
   std::fill(result.begin(),result.end(),987.);
   try{if(observe)g::execute_observed(candidate,{source,available},nullptr,0,&bound,&bound,1,result.data(),1,&layout,[](std::optional<o::Op>,const o::Value&,bool){});
       else g::execute(candidate,{source,available},nullptr,0,&bound,&bound,1,result.data(),1,&layout);
       return std::string("OK");}catch(const o::Error&e){return std::string(e.what());}
  };
  const auto expected=capture(true,canonical),actual=capture(false,out);
  if(expected=="OK" || actual!=expected ||std::memcmp(out.data(),canonical.data(),out.size()*8))throw std::runtime_error("state decision raw failure or prefix mismatch: "+expected+" / "+actual);
 }
}
}
