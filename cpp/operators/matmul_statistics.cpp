#include "calmetrics_engine/operators.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include <algorithm>
#include <cfenv>
#include <climits>
#include <cmath>
#include <limits>
#include "axis_primitives.hpp"
#include "numeric_line_sort.hpp"

#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

namespace calmetrics_engine::ops {
std::size_t matmul_statistics_scratch_doubles(const Shape &a, const Shape &b) {
    // Shared geometry admission restricts extents before calling this function.
    const auto n = a.dim[0], k = a.dim[1], t = b.dim[1];
    return t + 2*k + k*k + std::min<std::size_t>(512,t)*k +
           std::min<std::size_t>(1024,n)*k;
}
namespace {
void canonical_row(const Prepared &p, std::size_t row, double *trajectory,
                   double &total, double &squared, std::size_t &replayed,
                   void *context, void (*checkpoint)(void *)) {
    checkpoint(context);
    const auto count=p.args[1].shape.dim[1];
    if (p.args[0].shape.dim[1]==0) {
        total=squared=0.; ++replayed; return;
    }
    auto left=p.args[0];
    left.data=static_cast<const double *>(left.data)+
        static_cast<std::ptrdiff_t>(row)*left.stride[0];
    left.shape.dim[0]=1;
    Workspace work;
    for (std::size_t first=0; first<count; first+=4096) {
        checkpoint(context);
        auto right=p.args[1];
        right.data=static_cast<const double *>(right.data)+
            static_cast<std::ptrdiff_t>(first)*right.stride[1];
        right.shape.dim[1]=std::min<std::size_t>(4096,count-first);
        const Value args[]={left,right};
        const auto plan=prepare(*p.spec,args,2);
        Output out; out.kind=Kind::number; out.shape=plan.output_shape;
        out.data=trajectory+first;
        Audit local;
        execute(plan,out,work,Isa::scalar,local);
    }
    total=0.;
    bool valid=true;
    for (std::size_t i=0;i<count;++i) {
        const auto value=trajectory[i];
        total+=value; valid=valid && std::isfinite(value);
    }
    squared=std::numeric_limits<double>::quiet_NaN();
    if (valid) {
        const auto mean=total/double(count);
        squared=0.;
        for (std::size_t i=0;i<count;++i) {
            const auto delta=trajectory[i]-mean;
            squared+=delta*delta;
        }
    }
    ++replayed;
}
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL)
struct API_AVAILABLE(macos(15.0)) ThreadLimit {
    BLAS_THREADING prior=BLASGetThreading();
    bool active=false;
    ThreadLimit() {
        if (prior!=BLAS_THREADING_SINGLE_THREADED && prior!=BLAS_THREADING_MULTI_THREADED) return;
        active=BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)==0 &&
               BLASGetThreading()==BLAS_THREADING_SINGLE_THREADED;
        if (!active) BLASSetThreading(prior);
    }
    void restore() {
        if (!active) return;
        active=false;
        require(BLASSetThreading(prior)==0 && BLASGetThreading()==prior,
                "NATIVE_LIBRARY_THREAD_RESTORE_FAILED");
    }
    ~ThreadLimit() { if (active) BLASSetThreading(prior); }
};
bool API_AVAILABLE(macos(15.0)) moments(const Prepared &p, double *total,
    double *squared, double *scratch, const MatmulStatisticsAccuracy &accuracy,
    Audit &audit, std::size_t &replayed, void *context, void (*checkpoint)(void *)) {
    const auto &a=p.args[0], &b=p.args[1];
    const auto n=a.shape.dim[0], k=a.shape.dim[1], t=b.shape.dim[1];
    if (!accuracy.allowed || !k || std::fegetround()!=FE_TONEAREST) return false;
    bool column=false;
    std::ptrdiff_t leading=0;
    if (a.stride[1]==1 && a.stride[0]>=std::ptrdiff_t(k)) leading=a.stride[0];
    else if (a.stride[0]==1 && a.stride[1]>=std::ptrdiff_t(n)) {
        column=true; leading=a.stride[1];
    } else return false;
    if (leading>INT_MAX || accuracy.maximum_ddof>=t) return false;
    double maximum=0., weight_sum=0.;
    for (std::size_t i=0;i<k;++i) {
        checkpoint(context);
        for (std::size_t j=0;j<t;++j) {
            const auto value=b.at(i,j);
            if (!std::isfinite(value) || std::abs(value)>1e50) return false;
            maximum=std::max(maximum,std::abs(value));
        }
    }
    for (std::size_t i=0;i<n;++i) {
        if (i%512==0) checkpoint(context);
        double sum=0.;
        for (std::size_t j=0;j<k;++j) {
            const auto value=a.at(i,j);
            if (!std::isfinite(value) || std::abs(value)>1e50) return false;
            sum+=std::abs(value);
        }
        weight_sum=std::max(weight_sum,sum);
    }
    const auto unit=std::numeric_limits<double>::epsilon()*.5;
    const auto count=16.*double(t)+16.*double(k)*double(k)+256.;
    const auto gamma=count*unit/(1.-count*unit);
    const auto magnitude=maximum*weight_sum*(1.+64.*unit);
    const auto tiny=(double(t)+double(k)+1024.)*std::numeric_limits<double>::denorm_min();
    // Conservative complete-dot/centered-product FP64 envelope, including both
    // canonical and library folds. Geometry caps guarantee no overflow here.
    const auto mean_error=64.*gamma*magnitude+tiny;
    const auto squared_error=64.*gamma*double(t)*magnitude*magnitude+tiny;
    const double errors[]={mean_error*double(t),mean_error,
                           squared_error/double(t-accuracy.maximum_ddof)};
    constexpr double tolerance=4e-5; // implementation admission, not output rounding
    for (std::size_t i=0;i<3;++i)
        if (!(accuracy.linear_scale[i]*errors[i]<=tolerance) ||
            !(accuracy.sqrt_scale[i]*std::sqrt(errors[i])<=tolerance)) return false;
    ThreadLimit threads;
    if (!threads.active) return false;
    auto *trajectory=scratch;
    auto *mu=trajectory+t;
    auto *cov=mu+k;
    auto *center=cov+k*k;
    const auto rows=std::min<std::size_t>(512,t);
    auto *product=center+rows*k;
    for (std::size_t j=0;j<k;++j) {
        checkpoint(context);
        double sum=0.;
        for (std::size_t i=0;i<t;++i) sum+=b.at(j,i);
        mu[j]=sum/double(t);
    }
    for (std::size_t first=0;first<t;first+=rows) {
        checkpoint(context);
        const auto length=std::min(rows,t-first);
        for (std::size_t i=0;i<length;++i)
            for (std::size_t j=0;j<k;++j) center[i*k+j]=b.at(j,first+i)-mu[j];
        cblas_dsyrk(CblasRowMajor,CblasUpper,CblasTrans,int(k),int(length),1.,
                    center,int(k),first ? 1. : 0.,cov,int(k));
        ++audit.covariance_syrk_calls;
    }
    for (std::size_t i=0;i<k;++i)
        for (std::size_t j=0;j<i;++j) cov[i*k+j]=cov[j*k+i];
    const auto capacity=std::min<std::size_t>(1024,n);
    const auto *data=static_cast<const double *>(a.data);
    for (std::size_t first=0;first<n;first+=capacity) {
        checkpoint(context);
        const auto length=std::min(capacity,n-first);
        const auto *w=data+std::ptrdiff_t(first)*a.stride[0];
        cblas_dgemm(CblasRowMajor,column ? CblasTrans : CblasNoTrans,CblasNoTrans,
                    int(length),int(k),int(k),1.,w,int(leading),cov,int(k),0.,product,int(k));
        cblas_dgemv(CblasRowMajor,column ? CblasTrans : CblasNoTrans,
                    int(column ? k : length),int(column ? length : k),1.,w,int(leading),
                    mu,1,0.,total+first,1);
        ++audit.accelerate_matmul_calls;
        for (std::size_t row=0;row<length;++row) {
            const auto q=cblas_ddot(int(k),product+row*k,1,
                            w+std::ptrdiff_t(row)*a.stride[0],int(a.stride[1]));
            const auto mean=total[first+row];
            const bool mean_boundary=(accuracy.sqrt_scale[0]>0 || accuracy.sqrt_scale[1]>0) &&
                                     mean<=mean_error && mean>=-mean_error;
            if (!std::isfinite(q) || q<0 || !std::isfinite(mean) || mean_boundary)
                canonical_row(p,first+row,trajectory,total[first+row],squared[first+row],replayed,context,checkpoint);
            else {
                total[first+row]=mean*double(t);
                squared[first+row]=q==0. ? 0. : q;
            }
        }
    }
    threads.restore();
    audit.native_library_workspace_reserve_bytes=1024*1024;
    return true;
}
#endif
} // namespace
std::size_t matmul_statistics(const Prepared &p, double *total, double *squared,
    double *scratch, const MatmulStatisticsAccuracy &accuracy, Audit &audit,
    void *context, void (*checkpoint)(void *)) {
    require(p.spec && p.spec->op==Op::matmul && p.args[0].shape.rank==2 &&
            p.args[1].shape.rank==2,"INVALID_PHYSICAL_PAIR");
    require(checkpoint!=nullptr,"INVALID_PHYSICAL_PAIR");
    checkpoint(context);
    audit={};
    std::size_t replayed=0;
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL)
    if (__builtin_available(macOS 15.0,*))
        if (moments(p,total,squared,scratch,accuracy,audit,replayed,context,checkpoint)) return replayed;
#else
    (void)accuracy;
#endif
    for (std::size_t row=0;row<p.args[0].shape.dim[0];++row)
        canonical_row(p,row,scratch,total[row],squared[row],replayed,context,checkpoint);
    return replayed;
}

std::size_t matmul_quantiles_scratch_doubles(const Shape &a, const Shape &b) {
    // Bound all smaller runtime intervals too: floor(target/T)*T is not
    // monotone in T, so capacity(upper_T)*upper_T alone would underbudget.
    constexpr std::size_t target_cells=163840;
    const auto rows=std::min<std::size_t>(64,a.dim[0]);
    const auto trajectories=std::min(rows*b.dim[1],std::max(target_cells,b.dim[1]));
    return trajectories+rows*a.dim[1];
}
namespace {
void write_quantile(const MatmulQuantileOutput &output,std::size_t index,double value) {
    if (output.result) output.result->set(index,value);
    else output.values[index]=value;
}
void write_quantile_block(const MatmulQuantileOutput &output,const double *lines,
    std::size_t first,std::size_t length,std::size_t t) {
    const auto &q=*output.probabilities;
    if (output.result && output.result->order==MemoryOrder::f) {
        // A full public F result can be large. Write each contiguous candidate
        // segment from the already sorted batch, without transposing/copying it.
        for (std::size_t column=0;column<q.size();++column) {
            const auto probability=q.f(column);
            for (std::size_t row=0;row<length;++row)
                write_quantile(output,(first+row)*q.size()+column,
                    axis_path_detail::sorted_quantile(lines+row*t,t,probability));
        }
    } else {
        for (std::size_t row=0;row<length;++row)
            for (std::size_t column=0;column<q.size();++column)
                write_quantile(output,(first+row)*q.size()+column,
                    axis_path_detail::sorted_quantile(lines+row*t,t,q.f(column)));
    }
}
void line_statistics(const double *line, std::size_t t, double &total, double &squared) {
    total=0.; bool valid=true;
    for (std::size_t i=0;i<t;++i) {
        total+=line[i]; valid=valid && std::isfinite(line[i]);
    }
    squared=std::numeric_limits<double>::quiet_NaN();
    if (valid && t) {
        const auto mean=total/double(t); squared=0.;
        for (std::size_t i=0;i<t;++i) {
            const auto delta=line[i]-mean; squared+=delta*delta;
        }
    }
}
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL)
struct TrajectoryGuard {
    bool allowed=false, transpose=false;
    int leading=0;
    double errors[3]{};
};
TrajectoryGuard trajectory_guard(const Prepared &p,const MatmulStatisticsAccuracy &accuracy,
    double quantile_scale,void *context,void (*checkpoint)(void *)) {
    TrajectoryGuard result;
    const auto &a=p.args[0], &b=p.args[1];
    const auto n=a.shape.dim[0], k=a.shape.dim[1], t=b.shape.dim[1];
    if (!accuracy.allowed || !k || t<=accuracy.maximum_ddof || std::fegetround()!=FE_TONEAREST)
        return result;
    std::ptrdiff_t leading=0;
    if (b.stride[1]==1 && b.stride[0]>=std::ptrdiff_t(t)) leading=b.stride[0];
    else if (b.stride[0]==1 && b.stride[1]>=std::ptrdiff_t(k)) {
        result.transpose=true;leading=b.stride[1];
    } else return result;
    if (leading>INT_MAX) return result;
    result.leading=int(leading);
    double maximum=0.,weight_sum=0.;
    for (std::size_t i=0;i<k;++i) {
        checkpoint(context);
        for (std::size_t j=0;j<t;++j) {
            const auto value=b.at(i,j);
            if (!std::isfinite(value) || std::abs(value)>1e50) return result;
            maximum=std::max(maximum,std::abs(value));
        }
    }
    for (std::size_t row=0;row<n;++row) {
        if (row%512==0) checkpoint(context);
        double sum=0.;
        for (std::size_t column=0;column<k;++column) {
            const auto value=a.at(row,column);
            if (!std::isfinite(value) || std::abs(value)>1e50) return result;
            sum+=std::abs(value);
        }
        weight_sum=std::max(weight_sum,sum);
    }
    const auto u=std::numeric_limits<double>::epsilon()*.5;
    const auto folds=16.*double(t)+16.*double(k)*double(k)+256.;
    const auto gamma=folds*u/(1.-folds*u);
    const auto magnitude=maximum*weight_sum*(1.+64.*u);
    const auto tiny=(double(t)+double(k)+1024.)*std::numeric_limits<double>::denorm_min();
    const auto mean_error=64.*gamma*magnitude+tiny;
    const auto squared_error=64.*gamma*double(t)*magnitude*magnitude+tiny;
    result.errors[0]=mean_error*double(t);
    result.errors[1]=mean_error;
    result.errors[2]=squared_error/double(t-accuracy.maximum_ddof);
    constexpr double tolerance=4e-5;
    for (std::size_t i=0;i<3;++i)
        if (!(accuracy.linear_scale[i]*result.errors[i]<=tolerance)) return result;
    // Sorting and convex interpolation are non-expansive in the maximum norm.
    if (!(quantile_scale*mean_error<=tolerance)) return result;
    result.allowed=true;return result;
}
bool sqrt_consumers_safe(double total,double squared,std::size_t t,
    const MatmulStatisticsAccuracy &accuracy,const TrajectoryGuard &guard) {
    const double values[]={total,total/double(t),
        squared/double(t-accuracy.maximum_ddof)};
    for (std::size_t i=0;i<3;++i) if (accuracy.sqrt_scale[i]>0.) {
        const auto error=guard.errors[i],value=values[i];
        // sqrt is steep near zero. Certify this row rather than rejecting an
        // entire long-history request with the much looser sqrt(error) bound.
        // Unknown sign or an insufficient bound replays the original row.
        if (!(value>error)) return false;
        const auto deviation=error/(std::sqrt(value)+std::sqrt(value-error));
        if (!(accuracy.sqrt_scale[i]*deviation<=4e-5)) return false;
    }
    return true;
}
#endif
} // namespace
std::size_t matmul_quantiles(const Prepared &p,double *total,double *squared,
    double *scratch,const MatmulStatisticsAccuracy &accuracy,double quantile_scale,
    const MatmulQuantileOutput *outputs,std::size_t output_count,Audit &audit,
    void *context,void (*checkpoint)(void *)) {
    require(p.spec && p.spec->op==Op::matmul && p.args[0].shape.rank==2 &&
        p.args[1].shape.rank==2 && checkpoint,"INVALID_PHYSICAL_PAIR");
    audit={};checkpoint(context);
    const auto &a=p.args[0], &b=p.args[1];
    const auto n=a.shape.dim[0], k=a.shape.dim[1], t=b.shape.dim[1];
    if (!n) return 0;
    // Invalid probabilities remain errors at their original logical prepare,
    // after all preceding consumers. Never throw their numerical error here.
    std::vector<bool> valid(output_count,false);
    bool sort_needed=false;
    for (std::size_t output=0;output<output_count;++output) {
        const auto &q=*outputs[output].probabilities;
        bool admissible=q.kind==Kind::number && q.shape.rank==1;
        if (admissible) for (std::size_t i=0;i<q.size();++i) {
            const auto value=q.f(i);
            admissible=admissible && std::isfinite(value) && value>=0. && value<=1.;
        }
        // An invalid raw-native result capacity is checked by its original
        // logical root; no producer write and no early structural exception.
        valid[output]=admissible && (!outputs[output].result || outputs[output].result->data);
        sort_needed=sort_needed || (valid[output] && q.size()!=0);
    }
    // A single mathematical probability only needs its adjacent order
    // statistics. Reuse the canonical mature-library selection helper; never
    // consume the rearranged line for order-sensitive statistics afterwards.
    bool common_probability=true, have_probability=false;
    double probability=0.;
    for (std::size_t output=0;output<output_count;++output) if (valid[output]) {
        const auto &q=*outputs[output].probabilities;
        for (std::size_t column=0;column<q.size();++column) {
            const double current=q.f(column);
            if (!have_probability) { probability=current; have_probability=true; }
            else common_probability=common_probability && current==probability;
        }
    }
    const auto order_line=[&](double *line) {
        if (common_probability && have_probability &&
            axis_path_detail::partition_quantile_line(line,t,probability,Isa::automatic,&audit))
            return;
        axis_path_detail::sort_numeric_line(line,t,Isa::automatic,&audit);
    };
    constexpr std::size_t target_cells=163840;
    const auto capacity=std::min<std::size_t>(64,std::min(n,
        t ? std::max<std::size_t>(1,target_cells/t) : n));
    auto *weights=scratch+capacity*t;
    std::size_t replayed=0;
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL)
    TrajectoryGuard guard;
    if (__builtin_available(macOS 15.0,*)) guard=trajectory_guard(p,accuracy,quantile_scale,context,checkpoint);
    // Current-thread control stays in this scope until all library calls finish.
    if (__builtin_available(macOS 15.0,*)) {
        ThreadLimit threads;
        if (guard.allowed && threads.active) {
            for (std::size_t first=0;first<n;first+=capacity) {
                checkpoint(context);
                const auto length=std::min(capacity,n-first);
                for (std::size_t row=0;row<length;++row)
                    for (std::size_t asset=0;asset<k;++asset)
                        weights[row*k+asset]=a.at(first+row,asset);
                audit.algorithm_copy_bytes+=length*k*sizeof(double);
                cblas_dgemm(CblasRowMajor,CblasNoTrans,guard.transpose ? CblasTrans : CblasNoTrans,
                    int(length),int(t),int(k),1.,weights,int(k),static_cast<const double *>(b.data),
                    guard.leading,0.,scratch,int(t));
                ++audit.accelerate_matmul_calls;
                for (std::size_t row=0;row<length;++row) {
                    auto *line=scratch+row*t;
                    line_statistics(line,t,total[first+row],squared[first+row]);
                    bool zero=false;
                    for (std::size_t i=0;i<t;++i) zero=zero || line[i]==0.;
                    if (zero || !std::isfinite(total[first+row]) || !std::isfinite(squared[first+row]) ||
                        !sqrt_consumers_safe(total[first+row],squared[first+row],t,accuracy,guard))
                        canonical_row(p,first+row,line,total[first+row],squared[first+row],replayed,context,checkpoint);
                    if (sort_needed)
                        order_line(line);
                }
                if (sort_needed)
                    for (std::size_t output=0;output<output_count;++output) if (valid[output])
                        write_quantile_block(outputs[output],scratch,first,length,t);
            }
            threads.restore();
            audit.native_library_workspace_reserve_bytes=std::max<std::size_t>(
                audit.native_library_workspace_reserve_bytes,1024*1024);
            return replayed;
        }
        threads.restore();
    }
#else
    (void)accuracy;(void)quantile_scale;
#endif
    // A rejected layout/domain/library stays bounded. It never restores N*T.
    for (std::size_t row=0;row<n;++row) {
        canonical_row(p,row,scratch,total[row],squared[row],replayed,context,checkpoint);
        // canonical_row's K=0 shortcut computes statistics without touching a line.
        if (!k && t) std::fill_n(scratch,t,0.);
        if (sort_needed && t) {
            order_line(scratch);
            for (std::size_t output=0;output<output_count;++output) if (valid[output]) {
                const auto &q=*outputs[output].probabilities;
                for (std::size_t column=0;column<q.size();++column)
                    write_quantile(outputs[output],row*q.size()+column,
                        axis_path_detail::sorted_quantile(scratch,t,q.f(column)));
            }
        }
    }
    return replayed;
}
} // namespace calmetrics_engine::ops
