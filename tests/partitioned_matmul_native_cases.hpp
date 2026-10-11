#pragma once
#include "calmetrics_engine/operators.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>
#if defined(CALMETRICS_EXPECT_ACCELERATE_THREADING) && CALMETRICS_EXPECT_EIGEN_INPUT_SCAN
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif
namespace partitioned_matmul_native_cases {
namespace o=calmetrics_engine::ops;
inline void check(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
inline void run() {
#if defined(CALMETRICS_EXPECT_ACCELERATE_THREADING) && CALMETRICS_EXPECT_EIGEN_INPUT_SCAN
    const std::size_t m=513,k=1025,n=515;
    if(!o::matmul_accelerate_reserve_doubles(o::matrix_shape(m,k),o::matrix_shape(k,n)))return;
    const auto caller_threading=BLASGetThreading();
    check(BLASSetThreading(BLAS_THREADING_MULTI_THREADED)==0,"cannot set test caller threading");
    const auto restore=[&]{check(BLASSetThreading(caller_threading)==0 && BLASGetThreading()==caller_threading,"test caller threading restore");};
    try {
        std::vector<double> a(m*k),b(k*n),saved_a,saved_b;
        for(std::size_t i=0;i<a.size();++i)a[i]=double(int(i%13)-6)*128.;
        for(std::size_t i=0;i<b.size();++i)b[i]=double(int(i%11)-5)*128.;
        saved_a=a;saved_b=b;
        o::Value left,right;left.data=a.data();left.shape=o::matrix_shape(m,k);left.stride={std::ptrdiff_t(k),1};
        right.data=b.data();right.shape=o::matrix_shape(k,n);right.stride={std::ptrdiff_t(n),1};
        const o::Value args[]={left,right};const auto prepared=o::prepare(o::lookup("matmul"),args,2);
        for(auto order:{o::MemoryOrder::c,o::MemoryOrder::f}) {
            std::vector<double> storage(m*n+2,-993.);
            o::Output output;output.shape=prepared.output_shape;output.order=order;output.data=storage.data()+1;
            o::Workspace work;work.ensure(64,8);o::Audit audit;
            struct Stop:std::runtime_error {Stop():std::runtime_error("test stop after first DGEMM") {}};
            struct Context {double *first;bool after_write=false;};Context context{storage.data()+1};
            bool stopped=false;
            try {
                o::try_accelerate_matmul_root(prepared,output,work,audit,&context,[](void *pointer){
                    auto &c=*static_cast<Context*>(pointer);
                    if(*c.first!=-993.) {c.after_write=true;throw Stop();}
                });
            } catch(const Stop&) {stopped=true;}
            check(stopped&&context.after_write,"checkpoint did not stop after real DGEMM wrote private output");
            check(BLASGetThreading()==BLAS_THREADING_MULTI_THREADED,"exception failed to restore caller BLAS threading");
            check(storage.front()==-993.&&storage.back()==-993.,"cancelled matrix crossed private output canary");
            check(o::try_accelerate_matmul_root(prepared,output,work,audit),"root cannot run after cancellation");
            check(audit.accelerate_matmul_calls>1,"high-amplitude native root did not partition");
            check(BLASGetThreading()==BLAS_THREADING_MULTI_THREADED,"successful root failed to restore caller BLAS threading");
            for(std::size_t i=0;i<m;i+=127)for(std::size_t j=0;j<n;j+=129) {
                const auto actual=storage[1+(order==o::MemoryOrder::f?j*m+i:i*n+j)];
                check(actual==o::dot_value(left.row(i),right.column(j)),"exact binary native fixture differs");
            }
            check(a==saved_a&&b==saved_b,"native root changed readonly operands");
            for(auto &x:a)x*=0x1p30;
            std::fill(storage.begin(),storage.end(),-993.);
            work.doubles.assign(64,17.);work.indices.assign(8,19);
            const auto previous_d=work.doubles;const auto previous_i=work.indices;const auto capacity=work.capacity_bytes();audit.isa="untouched";
            check(!o::try_accelerate_matmul_root(prepared,output,work,audit),"over-budget root admitted");
            check(work.capacity_bytes()==capacity&&work.doubles==previous_d&&work.indices==previous_i,"preflight rejection changed caller workspace");
            check(std::strcmp(audit.isa,"untouched")==0,"preflight rejection changed caller audit");
            check(std::all_of(storage.begin(),storage.end(),[](double x){return x==-993.;}),"preflight rejection wrote private output");
            check(BLASGetThreading()==BLAS_THREADING_MULTI_THREADED,"preflight rejection changed caller threading");
            std::copy(saved_a.begin(),saved_a.end(),a.begin());
        }
        restore();
    } catch(...) {restore();throw;}
#endif
}
}
