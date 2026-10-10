#pragma once
#include "calmetrics_engine/operators.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
namespace matmul_input_bounds_native_cases {
namespace o = calmetrics_engine::ops;
inline void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
inline void run() {
    bool available = false;
#if CALMETRICS_EXPECT_EIGEN_INPUT_SCAN && defined(__APPLE__) && defined(__aarch64__)
    if (__builtin_available(macOS 15.0, *)) available = true;
#endif
    const int rounding = std::fegetround();
    for (const auto shape : {std::array<std::size_t,2>{1,31},{2,32},{65,33},{129,65},{4096,33},{4097,33},{8192,33},{33,8192}})
      for (int layout=0; layout<4; ++layout) {
        const auto rows=shape[0],cols=shape[1];const bool f=layout&1;
        const auto leading=(f ? rows : cols)+((layout&2) ? 7 : 0);
        std::vector<double> data(leading*(f ? cols : rows),std::numeric_limits<double>::quiet_NaN());
        o::Value input;input.shape=o::matrix_shape(rows,cols);input.data=data.data();
        input.stride=f ? std::array<std::ptrdiff_t,3>{1,std::ptrdiff_t(leading),1}
                       : std::array<std::ptrdiff_t,3>{std::ptrdiff_t(leading),1,1};
        const auto offset=[&](std::size_t position) {
            const auto row=position/cols,column=position%cols;
            return f ? column*leading+row : row*leading+column;
        };
        for(std::size_t i=0;i<rows*cols;++i)
            data[offset(i)]=(i%2 ? -1.0 : 1.0)*(.5+double(i%29)/32.0);
        const auto baseline=data;
        for(const int mode:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
            check(std::fesetround(mode)==0,"input bounds rounding setup");
            o::MatmulInputBounds bounds{-993.0,true};
            const auto status=o::library_matmul_input_bounds(input,bounds);
            check(status==(available ? o::MatmulBoundsStatus::valid : o::MatmulBoundsStatus::unavailable),
                  "input bounds optional library coverage");
            check(std::fegetround()==mode,"input bounds changed rounding mode");
            if(available) {
                double maximum=0;bool negative=false;
                for(std::size_t i=0;i<rows*cols;++i) {
                    const double value=data[offset(i)];maximum=std::max(maximum,std::abs(value));
                    negative=negative||std::signbit(value);
                }
                check(bounds.maximum==maximum&&bounds.negative==negative,"input bounds exact independent extrema");
            } else check(bounds.maximum==-993.0&&bounds.negative,"unavailable input bounds changed caller");
            check(std::memcmp(data.data(),baseline.data(),data.size()*sizeof(double))==0,
                  "valid input bounds modified source or padding");
        }
        check(std::fesetround(rounding)==0,"input bounds restore rounding");
        for(const auto position:{std::size_t(0),rows*cols/2,rows*cols-1})
          for(const double fault:{0.0,-0.0,std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),
                0x1p-129,0x1p129,std::nextafter(0x1p-128,0.0),
                std::nextafter(0x1p128,std::numeric_limits<double>::infinity()),0x1p-128,0x1p128}) {
            data=baseline;data[offset(position)]=fault;input.data=data.data();
            const auto snapshot=data;
            o::MatmulInputBounds bounds{-993.0,true};
            const auto expected=!available ? o::MatmulBoundsStatus::unavailable : fault==0.0 ?
                o::MatmulBoundsStatus::scalar_required : (!std::isfinite(fault)||
                std::abs(fault)<0x1p-128||std::abs(fault)>0x1p128) ?
                o::MatmulBoundsStatus::rejected : o::MatmulBoundsStatus::valid;
            check(o::library_matmul_input_bounds(input,bounds)==expected,"input bounds boundary classification");
            if(expected==o::MatmulBoundsStatus::valid) {
                double maximum=0;bool negative=false;
                for(std::size_t i=0;i<rows*cols;++i) {
                    const double value=data[offset(i)];
                    maximum=std::max(maximum,std::abs(value));
                    negative=negative||std::signbit(value);
                }
                check(bounds.maximum==maximum&&bounds.negative,
                      "boundary input bounds exact independent extrema");
            } else
                check(bounds.maximum==-993.0&&bounds.negative,"rejected/deferred input bounds changed caller");
            check(std::memcmp(data.data(),snapshot.data(),data.size()*sizeof(double))==0,
                  "input bounds modified input");
          }
        data=baseline;input.data=data.data();
        for(int mode=0;mode<9;++mode) {
            auto unsupported=input;
            if(mode==0)unsupported.stride[0]=0;
            if(mode==1)unsupported.stride[1]=-1;
            if(mode==2)unsupported.data=nullptr;
            if(mode==3)unsupported.shape=o::matrix_shape(0,cols);
            if(mode==4)unsupported.kind=o::Kind::mask;
            if(mode==5)unsupported.shape=o::tensor_shape(rows,cols,1);
            if(mode==6)unsupported.shape=o::matrix_shape(8193,cols);
            if(mode==8)unsupported.shape=o::matrix_shape(rows,8193);
            if(mode==7)unsupported.stride=f ? std::array<std::ptrdiff_t,3>{1,PTRDIFF_MAX,1}
                                          : std::array<std::ptrdiff_t,3>{PTRDIFF_MAX,1,1};
            o::MatmulInputBounds bounds{-993.0,true};
            check(o::library_matmul_input_bounds(unsupported,bounds)==o::MatmulBoundsStatus::unavailable,
                  "unsupported input bounds geometry must not read");
            check(bounds.maximum==-993.0&&bounds.negative,"unsupported bounds changed caller");
        }
      }
    check(std::fegetround()==rounding,"input bounds leaked rounding state");
}
}
