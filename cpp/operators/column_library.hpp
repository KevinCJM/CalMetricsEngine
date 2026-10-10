#pragma once
#include "calmetrics_engine/operators.hpp"
#include <cmath>

// Ordered independent-column reductions; shared by axis sums and variance.

#ifdef CALMETRICS_HAVE_EIGEN_ROTATION
#ifndef EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_PARALLELIZE
#endif
#ifndef EIGEN_MPL2_ONLY
#define EIGEN_MPL2_ONLY
#endif
#ifndef EIGEN_FAST_MATH
#define EIGEN_FAST_MATH 0
#endif
#include <Eigen/Core>
namespace calmetrics_engine::ops {
namespace {
template<Op Operation>
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
bool eigen_ordered_rows(const Value &x,double divisor,Output &out,Audit &audit) {
  if(x.kind!=Kind::number || x.shape.rank!=2 || !x.contiguous() ||
      x.shape.dim[0]<32 || x.shape.dim[1]<8 || x.shape.dim[1]>256) return false;
  using Row=Eigen::Array<double,Eigen::Dynamic,1>;
  const auto n=x.shape.dim[0],d=x.shape.dim[1];
  const auto *source=static_cast<const double *>(x.data);
  if constexpr(Operation==Op::variance_axis) {
    // Bounded stack maps, never a full input or dynamically allocated Eigen array.
    double mean_storage[256],squared_storage[256];
    Eigen::Map<Row,Eigen::Unaligned> mean(mean_storage,d),squared(squared_storage,d);
    mean.setZero();
    for(std::size_t i=0;i<n;++i)
      mean+=Eigen::Map<const Row,Eigen::Unaligned>(source+i*d,d);
    // Every NaN/Inf input yields a non-finite ordered column sum. Replay the
    // original eight-column path before writes, preserving its failure prefix.
    if(!mean.isFinite().all()) return false;
    mean/=double(n);
    squared.setZero();
    for(std::size_t i=0;i<n;++i) {
      const Eigen::Map<const Row,Eigen::Unaligned> values(source+i*d,d);
      const auto delta=values-mean;
      squared+=delta*delta;
    }
    for(std::size_t j=0;j<d;++j) {
      const double result=squared[j]/divisor;
      require(std::isfinite(result),"NON_FINITE_RESULT");
      out.set(j,result);
    }
  } else {
    Eigen::Map<Row,Eigen::Unaligned> totals(static_cast<double *>(out.data),d);
    totals.setZero();
    for(std::size_t i=0;i<n;++i)
      totals+=Eigen::Map<const Row,Eigen::Unaligned>(source+i*d,d);
    if constexpr(Operation==Op::mean_axis) totals/=divisor;
  }
  audit.isa="eigen-array";
  return true;
}

#if defined(__APPLE__) && defined(__aarch64__)
template<Op Operation, int RowStep>
__attribute__((noinline))
void eigen_ordered_columns_impl(const Value &x, double divisor, Output &out, Audit &audit) {
  constexpr bool Variance = Operation == Op::variance_axis;
  constexpr int width = 8;
  using Block = Eigen::Array<double,width,1>;
  using Stride = Eigen::InnerStride<Eigen::Dynamic>;
  using Map = Eigen::Map<const Block,Eigen::Unaligned,Stride>;
  const auto n=x.shape.dim[0],d=x.shape.dim[1];
  const auto *source=static_cast<const double *>(x.data);
  for (std::size_t first=0;first<d;first+=width) {
    if (d-first>=width) {
      Block sums=Block::Zero();
      Eigen::Array<bool,width,1> valid=Eigen::Array<bool,width,1>::Constant(true);
      const auto *base=source+std::ptrdiff_t(first)*x.stride[1];
      for (std::size_t i=0;i<n;++i) {
        const Map values(base+std::ptrdiff_t(i)*RowStep,Stride(x.stride[1]));
        if constexpr(Variance) valid=valid && values.isFinite();
        sums+=values;
      }
      if constexpr(Variance) {
        const Block means=sums/double(n);
        Block squared=Block::Zero();
        for (std::size_t i=0;i<n;++i) {
          const Map values(base+std::ptrdiff_t(i)*RowStep,Stride(x.stride[1]));
          const auto delta=values-means;
          squared+=delta*delta;
        }
        for (int j=0;j<width;++j) {
          require(valid[j],"NON_FINITE_RESULT");
          const double result=squared[j]/divisor;
          require(std::isfinite(result),"NON_FINITE_RESULT");
          out.set(first+j,result);
        }
      } else for(int j=0;j<width;++j) {
        if constexpr(Operation == Op::sum_axis) out.set(first+j,sums[j]);
        else out.set(first+j,sums[j]/divisor);
      }
    } else {
      for(std::size_t j=first;j<d;++j) {
        const auto *base=source+std::ptrdiff_t(j)*x.stride[1];double sum=0;bool valid=true;
        for(std::size_t i=0;i<n;++i) {const double value=base[std::ptrdiff_t(i)*RowStep];sum+=value;if constexpr(Variance) valid &= std::isfinite(value);}
        if constexpr(Variance) {
          require(valid,"NON_FINITE_RESULT");const double mean=sum/double(n);sum=0;
          for(std::size_t i=0;i<n;++i) {const double delta=base[std::ptrdiff_t(i)*RowStep]-mean;sum+=delta*delta;}
          require(std::isfinite(sum/divisor),"NON_FINITE_RESULT");
        }
        if constexpr(Operation == Op::sum_axis) out.set(j,sum);
        else out.set(j,sum/divisor);
      }
    }
  }
  audit.isa="eigen-array";
}
template<Op Operation>
void eigen_ordered_columns(const Value &x, double divisor, Output &out, Audit &audit) {
  // Dispatch once. The reduction sequence is always logical row 0..n-1,
  // including when the physical column is read backwards.
  if (x.stride[0] == 1) eigen_ordered_columns_impl<Operation,1>(x,divisor,out,audit);
  else eigen_ordered_columns_impl<Operation,-1>(x,divisor,out,audit);
}
#endif
}
}
#endif
