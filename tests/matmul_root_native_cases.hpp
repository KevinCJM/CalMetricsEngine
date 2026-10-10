#pragma once
#include "calmetrics_engine/operators.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <climits>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace matmul_root_native_cases {
namespace o = calmetrics_engine::ops;
inline void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
inline bool same(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0 || (std::isnan(a) && std::isnan(b));
}
inline o::Value view(const double *data, std::size_t m, std::size_t n, bool f = false) {
    o::Value v; v.data = data; v.shape = o::matrix_shape(m, n);
    v.stride = f ? std::array<std::ptrdiff_t, 3>{1, std::ptrdiff_t(m), 1}
                 : std::array<std::ptrdiff_t, 3>{std::ptrdiff_t(n), 1, 1};
    return v;
}
inline void run() {
    // Native F writer, including panel tails, unsupported strides, empty K,
    // and a nonnull empty descriptor whose row stride must never be advanced.
    for (const auto shape : {std::array<std::size_t, 3>{7, 33, 9}, {3, 17, 5}, {7, 0, 9}})
      for (int layout = 0; layout < 7; ++layout) {
        const auto m = shape[0], k = shape[1], n = shape[2];
        std::vector<double> a(std::max<std::size_t>(m*k, 1)), b(std::max<std::size_t>(k*n, 1));
        for (std::size_t i = 0; i < a.size(); ++i) a[i] = double(int(i%5)-2);
        for (std::size_t i = 0; i < b.size(); ++i) b[i] = double(int(i%7)-3);
        std::array<o::Value, 8> args{view(a.data(),m,k,layout&1),view(b.data(),k,n,layout&2)};
        if (k && layout == 4) {
            args[0].data = a.data() + k - 1; args[0].stride[1] = -1;
            args[1].data = b.data() + (k-1)*n; args[1].stride[0] = -std::ptrdiff_t(n);
        }
        if (layout == 5) args[0].stride[0] = 0;
        if (!k && layout == 6) args[0].stride[0] = PTRDIFF_MAX;
        const auto plan = o::prepare(o::lookup("matmul"), args.data(), 2);
        for (const auto isa : {o::Isa::scalar, o::Isa::automatic}) {
            std::vector<double> storage(m*n+2, -993.);
            o::Output out; out.shape = plan.output_shape; out.order = o::MemoryOrder::f;
            out.data = storage.data()+1;
            o::Workspace work; work.ensure(plan.scratch_doubles, plan.scratch_indices);
            o::Audit audit; o::execute(plan,out,work,isa,audit);
            for (std::size_t i=0;i<m;++i) for (std::size_t j=0;j<n;++j) {
                const auto expected = k ? o::dot_value(args[0].row(i),args[1].column(j)) : 0.0;
                check(same(storage[1+j*m+i],expected),"F matmul changed canonical arithmetic");
            }
            check(storage.front()==-993. && storage.back()==-993.,"F matmul crossed output canary");
        }
      }
    const std::size_t m=512,k=256,n=512;
    std::vector<double> a(m*k),b(k*n),storage(m*n+2,-993.);
    for (std::size_t i=0;i<a.size();++i) a[i]=double(int(i%13)-6)/8.;
    for (std::size_t i=0;i<b.size();++i) b[i]=double(int(i%11)-5)/8.;
    std::array<o::Value,8> args{view(a.data(),m,k),view(b.data(),k,n)};
    const auto plan=o::prepare(o::lookup("matmul"),args.data(),2);
    const bool available=o::matmul_accelerate_reserve_doubles(args[0].shape,args[1].shape)!=0;
    const int saved_round=std::fegetround();
    for (const auto order:{o::MemoryOrder::c,o::MemoryOrder::f}) {
        o::Output out;out.shape=plan.output_shape;out.order=order;out.data=storage.data()+1;
        o::Workspace work; o::Audit audit;
        for (const int mode:{FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
            check(std::fesetround(mode)==0,"cannot change signed GEMM rounding");
            std::fill(storage.begin(),storage.end(),-993.);work.ensure(64,8);
            const auto capacity=work.capacity_bytes();audit.isa="untouched";
            check(!o::try_accelerate_matmul_root(plan,out,work,audit),"mixed GEMM admitted non-nearest rounding");
            check(std::fegetround()==mode && work.capacity_bytes()==capacity &&
                  std::strcmp(audit.isa,"untouched")==0,"mixed GEMM rejection mutated caller state");
            for (auto x:storage) check(x==-993.,"mixed GEMM rejection wrote output");
        }
        check(std::fesetround(FE_TONEAREST)==0,"cannot restore nearest rounding");
        check(o::try_accelerate_matmul_root(plan,out,work,audit)==available,"mixed GEMM platform admission");
        if (available) {
            check(std::strcmp(audit.isa,"accelerate-bounded-signed-matmul")==0,"mixed GEMM lost dispatch audit");
            for (std::size_t i=0;i<m;++i) for (std::size_t j=0;j<n;++j) {
                const auto actual=storage[1+(order==o::MemoryOrder::f ? j*m+i : i*n+j)];
                check(same(actual,o::dot_value(args[0].row(i),args[1].column(j))),"mixed GEMM changed exact binary fixture");
            }
            check(storage.front()==-993. && storage.back()==-993.,"mixed GEMM crossed output canary");
        }
        for (auto &x:a) x*=0x1p30;
        std::fill(storage.begin(),storage.end(),-993.);work.ensure(64,8);
        const auto capacity=work.capacity_bytes();audit.isa="untouched";
        check(!o::try_accelerate_matmul_root(plan,out,work,audit),"mixed GEMM exceeded absolute error bound");
        check(work.capacity_bytes()==capacity && std::strcmp(audit.isa,"untouched")==0,"bounded rejection changed state");
        for (auto x:storage) check(x==-993.,"bounded rejection wrote output");
        for (auto &x:a) x*=0x1p-30;
    }
    check(std::fesetround(saved_round)==0,"cannot restore caller rounding");

    // Descriptor rejection must precede any read even when the independent
    // root now permits long axes. Canary storage is deliberately tiny here.
    {
        const auto beyond=std::size_t(INT_MAX)+1;
        check(o::matmul_accelerate_reserve_doubles(o::matrix_shape(beyond,256),
              o::matrix_shape(256,32))==0,"long root admitted non-LP64 rows");
        check(o::matmul_accelerate_reserve_doubles(o::matrix_shape(32,256),
              o::matrix_shape(256,beyond))==0,"long root admitted non-LP64 columns");
        const double dummy=1.; double canary=-993.;
        std::array<o::Value,8> oversized{view(&dummy,8193,256),view(&dummy,256,32)};
        for (const int side:{0,1}) {
            auto inputs=oversized; inputs[side].stride[0]=std::ptrdiff_t(beyond);
            const auto descriptor=o::prepare(o::lookup("matmul"),inputs.data(),2);
            o::Output out;out.shape=descriptor.output_shape;out.data=&canary;
            o::Workspace work;work.ensure(64,8);const auto capacity=work.capacity_bytes();
            o::Audit audit;audit.isa="untouched";
            check(!o::try_accelerate_matmul_root(descriptor,out,work,audit),
                  "long root admitted non-LP64 leading dimension");
            check(canary==-993. && work.capacity_bytes()==capacity &&
                  std::strcmp(audit.isa,"untouched")==0,"LP64 rejection changed caller state");
        }
    }
    // Cancellation in a newly bounded input view must not publish output or
    // mutate a caller's retained numerical workspace.
    if (available) {
        std::vector<double> large(8193*256,.125),small(256*32,.125);
        large[0]=-.125;
        std::array<o::Value,8> inputs{view(large.data(),8193,256),view(small.data(),256,32)};
        const auto descriptor=o::prepare(o::lookup("matmul"),inputs.data(),2);
        double canary=-993.;o::Output out;out.shape=descriptor.output_shape;out.data=&canary;
        o::Workspace work;work.ensure(64,8);const auto capacity=work.capacity_bytes();
        o::Audit audit;audit.isa="untouched";std::size_t checkpoints=0;bool stopped=false;
        try {
            o::try_accelerate_matmul_root(descriptor,out,work,audit,&checkpoints,
                [](void *context){if(++*static_cast<std::size_t*>(context)==2)throw std::runtime_error("test_cancel");});
        } catch (const std::runtime_error &error) {stopped=std::strcmp(error.what(),"test_cancel")==0;}
        check(stopped && checkpoints==2 && canary==-993. && work.capacity_bytes()==capacity &&
              std::strcmp(audit.isa,"untouched")==0,"long bounds cancellation changed caller state");
    }

}
} // namespace matmul_root_native_cases
