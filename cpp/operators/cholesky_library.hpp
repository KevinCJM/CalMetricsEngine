#pragma once
#include "calmetrics_engine/operators.hpp"
#include <algorithm>
#include <cmath>

#ifdef CALMETRICS_HAVE_EIGEN_ROTATION
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_MPL2_ONLY
#define EIGEN_FAST_MATH 0
#include <Eigen/Core>
#endif

namespace calmetrics_engine::ops {
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
namespace cholesky_detail {
// Each coefficient retains the canonical ascending-k multiply/subtract order.
// Eigen maps borrow bounded caller-owned storage; lazy expressions own no arrays.
inline void update(double *__restrict dst, const double *__restrict source,
                   std::size_t n, std::size_t count, const double *factor) {
 using Row=Eigen::Array<double,1,Eigen::Dynamic>;
 Eigen::Map<Row> d(dst,static_cast<Eigen::Index>(count));
  const Eigen::Map<const Row> s0(source+0*n,static_cast<Eigen::Index>(count));
  const Eigen::Map<const Row> s1(source+1*n,static_cast<Eigen::Index>(count));
  const Eigen::Map<const Row> s2(source+2*n,static_cast<Eigen::Index>(count));
  const Eigen::Map<const Row> s3(source+3*n,static_cast<Eigen::Index>(count));
  const Eigen::Map<const Row> s4(source+4*n,static_cast<Eigen::Index>(count));
  const Eigen::Map<const Row> s5(source+5*n,static_cast<Eigen::Index>(count));
  const Eigen::Map<const Row> s6(source+6*n,static_cast<Eigen::Index>(count));
  const Eigen::Map<const Row> s7(source+7*n,static_cast<Eigen::Index>(count));
  d=((((((((d-factor[0]*s0)-factor[1]*s1)-factor[2]*s2)-factor[3]*s3)-factor[4]*s4)-factor[5]*s5)-factor[6]*s6)-factor[7]*s7);
}

// Four panel rows are independent after the eight factors have been computed.
// Each fixed array retains one coefficient's ascending-u multiply/subtract order.
// common_count is the shortest row prefix; row r has common_count+r coefficients.
inline void update_group(double *__restrict dst, const double *__restrict source,
                         std::size_t n, std::size_t pitch,
                         std::size_t common_count, const double *factor) {
    constexpr std::size_t rows=4, width=8, steps=8;
    using Tile=Eigen::Array<double,width,1>;
    using Read=Eigen::Map<const Tile,Eigen::Unaligned>;
    std::size_t j=0;
    for (;common_count-j>=width;j+=width) {
        Tile values[rows];
        for (std::size_t r=0;r<rows;++r)
            values[r]=Read(dst+r*pitch+j);
        for (std::size_t u=0;u<steps;++u) {
            // Fixed owning storage materializes the source once for all four rows.
            // This is bounded register/stack storage, never a dynamic Eigen array.
            const Tile shared=Read(source+u*n+j);
            for (std::size_t r=0;r<rows;++r)
                values[r]=values[r]-factor[r*steps+u]*shared;
        }
        for (std::size_t r=0;r<rows;++r)
            Eigen::Map<Tile,Eigen::Unaligned>(dst+r*pitch+j)=values[r];
    }
    // Preserve each row's unequal triangular suffix, including the common tail.
    // The existing ordered expression consumes exactly common_count+r-j values.
    for (std::size_t r=0;r<rows;++r)
        update(dst+r*pitch+j,source+j,n,common_count+r-j,factor+r*steps);
}

// The caller owns validation, final row arithmetic, error replay and upper clear.
template<class FinishRow>
void factor_rows(const Value &a, double *y, Workspace &work, FinishRow &&finish) {
    constexpr std::size_t rows=16, steps=8;
    const auto n=a.shape.dim[0], pitch=n+17;
    auto *panel=work.doubles.data();
    auto *factors=panel+rows*pitch;
    for (std::size_t begin=0;begin<n;begin+=rows) {
        const auto count=std::min(rows,n-begin);
        for (std::size_t r=0;r<count;++r)
            for (std::size_t j=0;j<=begin+r;++j) panel[r*pitch+j]=a.at(begin+r,j);
        // begin is a multiple of rows, so every preceding group has eight columns.
        for (std::size_t k=0;k<begin;k+=steps) {
            for (std::size_t u=0;u<steps;++u) {
                for (std::size_t r=0;r<count;++r) {
                    double value=panel[r*pitch+k+u];
                    for (std::size_t t=0;t<u;++t)
                        value-=factors[r*steps+t]*y[(k+t)*n+k+u];
                    require(std::isfinite(value),"NON_FINITE_RESULT");
                    const double factor=value/y[(k+u)*n+k+u];
                    require(std::isfinite(factor),"NON_FINITE_RESULT");
                    factors[r*steps+u]=factor;
                    panel[r*pitch+k+u]=factor;
                    y[(k+u)*n+begin+r]=factor;
                }
            }
            std::size_t r=0;
            for (;count-r>=4;r+=4)
                update_group(panel+r*pitch+k+steps,y+k*n+k+steps,n,pitch,
                             begin+r-k-steps+1,factors+r*steps);
            for (;r<count;++r)
                update(panel+r*pitch+k+steps,y+k*n+k+steps,n,
                       begin+r-k-steps+1,factors+r*steps);
        }
        for (std::size_t r=0;r<count;++r) {
            const auto i=begin+r;
            std::copy_n(panel+r*pitch,i+1,y+i*n);
            finish(i,begin);
        }
    }
}
} // namespace cholesky_detail
#endif
} // namespace calmetrics_engine::ops
