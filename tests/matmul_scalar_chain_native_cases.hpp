#pragma once
#include "calmetrics_engine/graph.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include "calmetrics_engine/native_runtime.hpp"
#include <new>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <vector>

namespace matmul_scalar_chain_native_cases {
namespace g=calmetrics_engine::graph;
namespace o=calmetrics_engine::ops;

inline void run_scalar_pair_failure_layouts() {
    constexpr std::size_t rows=257,cols=263;
    // This is the raw writer contract, including its partial failure prefix.
    // No graph/result wrapper is allowed to hide a changed prefix by clearing it.
    for(int layout=0;layout<6;++layout)
    for(auto order:{o::MemoryOrder::c,o::MemoryOrder::f})
    for(bool left:{false,true}) {
        const bool column=layout<3,pitched=layout%3==1,reversed=layout%3==2;
        const auto pitch=(column?rows:cols)+(pitched?3:0);
        std::vector<double> owner(pitch*(column?cols:rows),987.0);
        o::Value x;x.shape=o::matrix_shape(rows,cols);
        x.stride={column?1:std::ptrdiff_t(pitch),column?std::ptrdiff_t(pitch):1,1};
        auto *start=owner.data();
        if(reversed) {start+=std::ptrdiff_t(rows-1)*x.stride[0];x.stride[0]*=-1;}
        x.data=start;
        for(std::size_t row=0;row<rows;++row)for(std::size_t col=0;col<cols;++col)
            start[std::ptrdiff_t(row)*x.stride[0]+std::ptrdiff_t(col)*x.stride[1]]=1.+double((row+col)%5)*.25;
        start[3*x.stride[0]+2*x.stride[1]]=0.;
        const auto before=owner;
        const o::Value first_args[2]={x,o::Value::number(.5)};
        const auto first=o::prepare(o::lookup("multiply"),first_args,2);
        std::vector<double> middle(rows*cols),expected(rows*cols,987.),actual(rows*cols,987.);
        o::Output intermediate;intermediate.shape=x.shape;intermediate.data=middle.data();
        o::Workspace workspace;o::Audit audit;
        o::execute(first,intermediate,workspace,o::Isa::automatic,audit);
        o::Value value;value.shape=x.shape;value.data=middle.data();value.set_contiguous_strides();
        const auto scalar=o::Value::number(left?0.:.5);
        const o::Value second_args[2]={left?value:scalar,left?scalar:value};
        const auto second=o::prepare(o::lookup("divide"),second_args,2);
        o::Output destination;destination.shape=x.shape;destination.order=order;
        const auto error=[&](bool pair) {
            destination.data=pair?actual.data():expected.data();
            try {
                if(pair)o::elementwise_pair(first,second,destination,left);
                else for(std::size_t i=0;i<middle.size();++i)
                    destination.set(i,o::scalar_math(o::Op::divide,
                        left?middle[i]:scalar.scalar,left?scalar.scalar:middle[i]));
            } catch(const o::Error &e) {return std::string(e.what());}
            return std::string("OK");
        };
        const auto canonical_error=error(false),pair_error=error(true);
        if(canonical_error!="DIVIDE_BY_ZERO" || pair_error!=canonical_error ||
           std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(double)) || owner!=before)
            throw std::runtime_error("scalar pair changed a strided/cross-layout division failure prefix");
    }
}

inline g::Node input(std::size_t index) {
    g::Node n;n.kind=g::NodeKind::input;n.input_index=index;return n;
}
inline g::Node constant(double x) {
    g::Node n;n.kind=g::NodeKind::constant;n.constant=x;return n;
}
inline g::Node operation(o::Op op,std::uint32_t a,std::uint32_t b,std::size_t slot) {
    g::Node n;n.kind=g::NodeKind::operation;n.opcode=std::uint16_t(op);
    n.parent_count=2;n.parents[0]=a;n.parents[1]=b;
    n.storage=g::StorageKind::numeric;n.slot=slot;return n;
}
inline void run_terminal_failure_cases() {
    constexpr std::size_t m=512,k=256,n=512;
    if (!o::matmul_accelerate_reserve_doubles(o::matrix_shape(m,k),o::matrix_shape(k,n))) return;
    bool failures=false;
    for (auto order:{o::MemoryOrder::c,o::MemoryOrder::f})
    for (bool private_chain:{false,true})
    for (bool resource:{false,true}) {
        std::vector<double> a(m*k,.01),b(k*n,.01),out(m*n,987.0);
        const auto old_a=a,old_b=b;
        o::Value args[2];args[0].data=a.data();args[1].data=b.data();
        args[0].shape=o::matrix_shape(m,k);args[1].shape=o::matrix_shape(k,n);
        args[0].set_contiguous_strides();args[1].set_contiguous_strides();
        const auto prepared=o::prepare(o::lookup("matmul"),args,2);
        o::Output output;output.kind=o::Kind::number;output.shape=o::matrix_shape(m,n);
        output.data=out.data();output.order=order;
        o::Workspace workspace;o::Audit audit;
        o::MatmulTerminalProof proof;proof.source=&prepared;
        proof.left.maximum=proof.right.maximum=.01;
        struct Hook {double *data;bool resource;bool fired=false;} hook{out.data(),resource};
        const auto checkpoint=[](void *context) {
            auto &state=*static_cast<Hook*>(context);
            // Synchronous same-thread callback after DGEMM. No clock race or
            // cross-thread read of a matrix is used to force this exception.
            if (state.data[0]!=987.0) {
                state.fired=true;
                if(state.resource) throw std::bad_alloc();
                throw calmetrics_engine::native::Timeout("terminal proof deterministic cancellation");
            }
        };
        bool caught=false;
        try {
            o::try_accelerate_matmul_root(prepared,output,workspace,audit,&hook,checkpoint,true,
                                         private_chain?&proof:nullptr);
        } catch(const std::bad_alloc &) {
            if(!resource) throw;caught=true;
        } catch(const calmetrics_engine::native::Timeout &) {
            if(resource) throw;caught=true;
        }
        const bool restored=private_chain
            ? std::all_of(out.begin(),out.end(),[](double x){return x==0.0&&!std::signbit(x);})
            : std::all_of(out.begin(),out.end(),[](double x){return std::isfinite(x)&&x>0.0&&x!=987.0;});
        if(!caught || !hook.fired || !restored || a!=old_a || b!=old_b) {
            failures=true;
            std::cerr<<"terminal-failure order="<<int(order)<<" private="<<private_chain
                     <<" resource="<<resource<<" fired="<<hook.fired
                     <<" caught="<<caught<<" restored="<<restored<<'\n';
        }
    }
    if(failures)throw std::runtime_error("terminal proof cancellation/resource exception exposed an uncommitted matrix");
}
inline void run() {
    run_scalar_pair_failure_layouts();
    run_terminal_failure_cases();
    {
        g::Program p;p.input_count=3;p.input_axes={1,1,1};p.numeric_slots=2;
        p.output_kind=g::OutputKind::typed;
        p.root_outputs={{g::OutputDType::float64,2,g::OutputOrder::c}};
        p.nodes={input(0),input(1),operation(o::Op::matmul,0,1,0),
                 input(2),operation(o::Op::multiply,2,3,1)};
        p.roots={4};p.finalize();
        const auto a=o::matrix_shape(512,256),b=o::matrix_shape(256,512),c=o::matrix_shape(512,512);
        const std::vector<o::Shape> shapes{a,b,c,c,c};
        if(g::matrix_workspace_doubles(p,shapes,c.size(),true)!=o::matmul_scratch_doubles(a,b))
            throw std::runtime_error("known non-scalar matrix chain reserves an ineligible library path");
    }
    {
        g::Program p;p.input_count=2;p.input_axes={1,1};p.numeric_slots=2;
        p.output_kind=g::OutputKind::typed;
        p.root_outputs={{g::OutputDType::float64,2,g::OutputOrder::c}};
        p.nodes={input(0),input(1),operation(o::Op::matmul,0,1,0),
                 constant(.5),operation(o::Op::multiply,2,3,1)};
        p.roots={4};p.finalize();
        const auto reserve=o::matmul_accelerate_reserve_doubles(
            o::matrix_shape(1024,256),o::matrix_shape(256,1024));
        if(g::matrix_workspace_doubles(p,{},256,true)<reserve)
            throw std::runtime_error("unknown shape private matrix chain omits mature library reserve");
    }
    {
        constexpr std::size_t m=512,k=256,n=512;
        std::vector<double> a(m*k,.01),b(k*n,.01),out(m*n,987.0);
        o::Value args[2];args[0].data=a.data();args[1].data=b.data();
        args[0].shape=o::matrix_shape(m,k);args[1].shape=o::matrix_shape(k,n);
        args[0].set_contiguous_strides();args[1].set_contiguous_strides();
        const auto prepared=o::prepare(o::lookup("matmul"),args,2);
        const auto other=prepared;
        o::Output output;output.kind=o::Kind::number;output.shape=o::matrix_shape(m,n);output.data=out.data();
        o::Workspace workspace;o::Audit audit;
        o::MatmulTerminalProof proof;proof.source=&other;
        proof.left.maximum=proof.right.maximum=.01;
        if(o::try_accelerate_matmul_root(prepared,output,workspace,audit,nullptr,nullptr,true,&proof) ||
           !std::all_of(out.begin(),out.end(),[](double x){return x==987.0;}))
            throw std::runtime_error("different Prepared instance consumed a terminal proof");
    }
    for (auto order:{g::OutputOrder::c,g::OutputOrder::f})
    for (bool blocked:{false,true})
    for (bool left:{false,true})
    for (bool later:{false,true}) {
        constexpr std::size_t m=256,k=4,n=256;
        std::vector<double> a(m*k,1.0),b(k*n,1.0);
        if (left) for(std::size_t i=0;i<k;++i)b[i*n+3]=0.0;
        o::Value x,y;x.data=a.data();y.data=b.data();
        x.shape=o::matrix_shape(m,k);y.shape=o::matrix_shape(k,n);
        x.set_contiguous_strides();y.set_contiguous_strides();
        const auto before_a=a,before_b=b;
        g::Program p;p.input_count=2;p.input_axes={1,1};p.numeric_slots=2;
        p.output_kind=g::OutputKind::typed;
        p.root_outputs={{g::OutputDType::float64,2,order}};
        p.nodes={input(0),input(1),operation(o::Op::matmul,0,1,0),
                 constant(left?1.0:0.0),operation(o::Op::divide,left?3:2,left?2:3,1)};
        if(later){p.nodes.push_back(constant(.25));p.nodes.push_back(operation(o::Op::add,4,5,0));}
        p.roots={std::uint32_t(p.nodes.size()-1)};p.finalize();
        if(!blocked) {
            auto metadata=std::make_shared<g::ExecutionMetadata>(*p.execution_metadata);
            metadata->row_batch.reset();p.execution_metadata=std::move(metadata);
        }
        const std::vector<o::Value> inputs{x,y};const std::int64_t start=0,end=0;
        const auto layout=g::result_layout(p,inputs,&start,&end,1);
        std::vector<double> out(layout.bytes()/8,987.0);
        const auto execute=[&](bool observed) {
            try {
                if(observed)g::execute_observed(p,inputs,nullptr,0,&start,&end,1,
                    out.data(),1,&layout,[](std::optional<o::Op>,const o::Value &,bool){});
                else g::execute(p,inputs,nullptr,0,&start,&end,1,out.data(),1,&layout);
            } catch(const o::Error &e){return std::string(e.what());}
            return std::string("OK");
        };
        const auto error=execute(true);
        // Observation disables direct backing. Its failed bytes need not be
        // the original direct-root prefix, especially for F divide. Rebuild
        // that prefix with the existing canonical divide operator instead.
        std::vector<double> expected(out.size(),0.0),product(m*n,double(k));
        if(left)for(std::size_t row=0;row<m;++row)product[row*n+3]=0.0;
        // Existing row batching commits a tile only after its whole child
        // succeeds. This fixture fails in the first tile, so the parent root
        // stays zero. A plain F writer can retain its canonical partial prefix.
        if(!p.execution_metadata->row_batch && !later && p.execution_metadata->root_destinations[p.roots[0]]>=0) {
            o::Value v;v.data=product.data();v.shape=o::matrix_shape(m,n);v.set_contiguous_strides();
            const auto scalar=o::Value::number(left?1.0:0.0);
            const o::Value args[2]={left?scalar:v,left?v:scalar};
            const auto prepared=o::prepare(o::lookup("divide"),args,2);
            o::Output destination;destination.kind=o::Kind::number;destination.shape=v.shape;
            destination.data=expected.data();destination.order=layout.order(0);
            o::Workspace workspace;o::Audit audit;
            try{o::execute(prepared,destination,workspace,o::Isa::automatic,audit);}
            catch(const o::Error &e){if(std::string(e.what())!=error)throw;}
        }
        std::fill(out.begin(),out.end(),987.0);
        const auto actual_error=execute(false);
        if(error!="DIVIDE_BY_ZERO" || actual_error!=error ||
           std::memcmp(out.data(),expected.data(),layout.bytes())!=0 ||
           a!=before_a || b!=before_b) {
            const auto mismatch=std::mismatch(out.begin(),out.end(),expected.begin());
            std::cerr<<"order="<<int(order)<<" blocked="<<blocked<<" left="<<left<<" later="<<later
                     <<" error="<<error<<" actual="<<actual_error
                     <<" first_difference="<<std::distance(out.begin(),mismatch.first)
                     <<" root_destination="<<p.execution_metadata->root_destinations[p.roots[0]]
                     <<" layout_order="<<int(layout.order(0))
                     <<" affine="<<bool(p.execution_metadata->affine_projection)
                     <<" compact="<<bool(p.execution_metadata->compact_path)
                     <<" rowbatch="<<bool(p.execution_metadata->row_batch)
                     <<" actual0="<<out[0]<<" expected0="<<expected[0]<<'\n';
            throw std::runtime_error("matmul scalar-chain failure buffer/order differs from canonical");
        }
        if(!p.execution_metadata->matmul_scalar_chain.empty())
            throw std::runtime_error("throwing divide region admitted a private root loan");
    }
}
}
