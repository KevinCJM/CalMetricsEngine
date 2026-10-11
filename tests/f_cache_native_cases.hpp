#pragma once
#include "calmetrics_engine/operators.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include <cstring>
#include <stdexcept>
#include <vector>
namespace f_cache_native_cases {
namespace o = calmetrics_engine::ops;
inline void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
inline void run() {
    const bool library = o::matmul_scratch_doubles(o::matrix_shape(32,33),
                                                  o::matrix_shape(33,128)) != 0;
    for (std::size_t m : {32U,35U}) for (std::size_t n : {32U,35U})
      for (std::size_t k : {683U,1025U}) for (unsigned layout=0;layout<4;++layout) {
        const bool af=layout&1,bf=layout&2;
        std::vector<double> left(m*k),right(k*n),storage(m*n+2,-993.),reference(m*n);
        for (std::size_t i=0;i<m;++i) for (std::size_t t=0;t<k;++t)
            left[af ? t*m+i : i*k+t]=double(int(i%5)-2)*(t%2 ? -1. : 1.)*1024.;
        for (std::size_t t=0;t<k;++t) for (std::size_t j=0;j<n;++j)
            right[bf ? j*k+t : t*n+j]=double(int(j%7)-3)*(t%2 ? -1. : 1.)*1024.;
        const auto before_left=left,before_right=right;
        o::Value a,b;a.data=left.data();b.data=right.data();
        a.shape=o::matrix_shape(m,k);b.shape=o::matrix_shape(k,n);
        a.stride=af ? std::array<std::ptrdiff_t,3>{1,std::ptrdiff_t(m),1}
                    : std::array<std::ptrdiff_t,3>{std::ptrdiff_t(k),1,1};
        b.stride=bf ? std::array<std::ptrdiff_t,3>{1,std::ptrdiff_t(k),1}
                    : std::array<std::ptrdiff_t,3>{std::ptrdiff_t(n),1,1};
        const o::Value args[]={a,b};const auto p=o::prepare(o::lookup("matmul"),args,2);
        check(p.scratch_doubles==0,"cache F allocated panels outside packing admission");
        o::Workspace scalar_work,work;o::Audit scalar_audit,audit;
        o::Output scalar;scalar.shape=p.output_shape;scalar.data=reference.data();
        o::execute(p,scalar,scalar_work,o::Isa::scalar,scalar_audit);
        o::Output out;out.shape=p.output_shape;out.order=o::MemoryOrder::f;out.data=storage.data()+1;
        o::execute(p,out,work,o::Isa::automatic,audit);
        for (std::size_t i=0;i<m;++i) for (std::size_t j=0;j<n;++j)
            check(storage[1+j*m+i]==reference[i*n+j],"cache F changed ordered scalar result");
        check(storage.front()==-993.&&storage.back()==-993.,"cache F boundary write");
        check(left==before_left&&right==before_right,"cache F input modified");
        check(work.doubles.empty()&&work.indices.empty()&&audit.algorithm_copy_bytes==0,
              "cache F introduced workspace or copy");
        if (library) check(std::strcmp(audit.isa,"eigen-block-matmul-f-root")==0,
                           "cache F missed native dispatch");
      }
}
}
