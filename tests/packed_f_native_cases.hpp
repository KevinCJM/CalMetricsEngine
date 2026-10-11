#pragma once
#include "calmetrics_engine/operators.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include <cstring>
#include <stdexcept>
#include <vector>
namespace packed_f_native_cases {
namespace o = calmetrics_engine::ops;
inline void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
inline void run() {
    for (std::size_t m : {32U, 35U}) for (std::size_t n : {128U, 131U})
      for (unsigned layout=0; layout<4; ++layout) {
        constexpr std::size_t k=33;
        const bool af=layout&1, bf=layout&2;
        std::vector<double> left(m*k),right(k*n),storage(m*n+2,-993.0);
        for(std::size_t i=0;i<m;++i) for(std::size_t t=0;t<k;++t)
            left[af ? t*m+i : i*k+t]=double(int(i%5)-2)*(t%2 ? -1.0 : 1.0)*1024.;
        for(std::size_t t=0;t<k;++t) for(std::size_t j=0;j<n;++j)
            right[bf ? j*k+t : t*n+j]=double(int(j%7)-3)*(t%2 ? -1.0 : 1.0)*1024.;
        const auto saved_left=left,saved_right=right;
        o::Value a,b;a.data=left.data();b.data=right.data();
        a.shape=o::matrix_shape(m,k);b.shape=o::matrix_shape(k,n);
        a.stride=af ? std::array<std::ptrdiff_t,3>{1,std::ptrdiff_t(m),1} : std::array<std::ptrdiff_t,3>{std::ptrdiff_t(k),1,1};
        b.stride=bf ? std::array<std::ptrdiff_t,3>{1,std::ptrdiff_t(k),1} : std::array<std::ptrdiff_t,3>{std::ptrdiff_t(n),1,1};
        const o::Value args[]={a,b};const auto p=o::prepare(o::lookup("matmul"),args,2);
        o::Workspace work;work.ensure(p.scratch_doubles,p.scratch_indices);
        o::Output out;out.shape=p.output_shape;out.order=o::MemoryOrder::f;out.data=storage.data()+1;
        o::Audit audit;o::execute(p,out,work,o::Isa::automatic,audit);
        for(std::size_t i=0;i<m;++i) for(std::size_t j=0;j<n;++j) {
            const double expected=double(int(i%5)-2)*double(int(j%7)-3)*double(k)*1024.*1024.;
            check(storage[1+j*m+i]==expected,"packed F native independent rectangular formula");
        }
        check(storage.front()==-993.&&storage.back()==-993.,"packed F native canary");
        check(std::memcmp(left.data(),saved_left.data(),left.size()*8)==0&&
              std::memcmp(right.data(),saved_right.data(),right.size()*8)==0,"packed F native input changed");
        if(o::matmul_packed_eligible(a,b)) {
            check(std::strcmp(audit.isa,"eigen-packed-matmul-f-root")==0,"packed F native dispatch");
            const auto prefix=m-m%4,bulk=n-n%6;
            check(audit.algorithm_copy_bytes==(prefix+((prefix+127)/128)*bulk)*k*8,
                  "packed F native exact copy audit");
        }
      }
}
}
