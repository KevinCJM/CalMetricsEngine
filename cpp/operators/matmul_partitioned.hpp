#pragma once
// Internal physical strategy. Input-domain and ownership proof stay in the root caller.
#if defined(CALMETRICS_HAVE_ACCELERATE_COVCHOL) && defined(CALMETRICS_HAVE_EIGEN_ROTATION)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_MPL2_ONLY
#define EIGEN_FAST_MATH 0
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cfenv>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace calmetrics_engine::ops::partitioned {
constexpr double budget = 0.000049;
using Row = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Col = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using Read = Eigen::Map<const Row, Eigen::Unaligned>;
using Write = Eigen::Map<Row, Eigen::Unaligned>;
struct Input {
    const double* data;
    std::size_t rows, columns;
    std::ptrdiff_t rs, cs;
    double at(std::size_t i, std::size_t j) const { return data[i*rs+j*cs]; }
};
struct Plan {
    std::size_t chunk=0, pieces=0, depth=0, slots=0, tr=0, tc=0, buffer=0, norm_workspace=0;
    double error=0, ambiguity=0;
};
struct Stats {
    std::size_t calls=0, replay=0, output_copy_bytes=0;
    double returned_error_bound=0;
};
inline double up(double x) { return std::nextafter(x,std::numeric_limits<double>::infinity()); }
inline double gamma(std::size_t count) {
    const double nu=double(count)*0x1p-53;
    return up(nu/std::nextafter(1.-nu,0.));
}
inline std::size_t log2ceil(std::size_t count) {
    std::size_t result=0, size=1;
    while(size<count) { size*=2; ++result; }
    return result;
}
inline bool describe(const Input& x, CBLAS_TRANSPOSE& trans, int& ld) {
    if(x.cs==1 && x.rs>=std::ptrdiff_t(x.columns) && x.rs<=INT_MAX) {
        trans=CblasNoTrans;ld=int(x.rs);return true;
    }
    if(x.rs==1 && x.cs>=std::ptrdiff_t(x.rows) && x.cs<=INT_MAX) {
        trans=CblasTrans;ld=int(x.cs);return true;
    }
    return false;
}
struct NormBounds { double one=0., two=0.; std::size_t temporary=0; };
inline bool norm_bounds(const Input& x,bool by_row,std::size_t capacity,NormBounds& out,
    double* workspace,void* context,void (*checkpoint)(void*)) {
    // Domain/strides/finite checks are completed by the root caller.
    // |a*b| summed along K is bounded by either one-norm times maxabs or
    // Cauchy-Schwarz. Sum/squared-sum reductions stay in mature Eigen.
    const auto lines=by_row?x.rows:x.columns;
    const auto length=by_row?x.columns:x.rows;
    const auto inner_stride=by_row?x.cs:x.rs;
    const auto outer_stride=by_row?x.rs:x.cs;
    using Array=Eigen::Array<double,Eigen::Dynamic,1>;
    using Map=Eigen::Map<const Array,Eigen::Unaligned>;
    double one=0.,squared=0.;
    if(inner_stride==1) {
        for(std::size_t i=0;i<lines;++i) {
            if(checkpoint && i%64==0)checkpoint(context);
            const Map values(x.data+i*outer_stride,length);
            one=std::max(one,values.abs().sum());
            squared=std::max(squared,values.square().sum());
        }
    } else {
        // Physical C/F scan: two bounded vectors, no full transpose or pack.
        if(outer_stride!=1 || capacity<8 || lines>(capacity-8)/2)return false;
        out.temporary=2*lines+8; // include conservative Eigen alignment padding
        Eigen::Map<Array,Eigen::Unaligned> absolute(workspace,lines),energy(workspace+lines,lines);
        absolute.setZero();energy.setZero();
        for(std::size_t j=0;j<length;++j) {
            if(checkpoint && j%64==0)checkpoint(context);
            const Map values(x.data+j*inner_stride,lines);
            absolute+=values.abs();
            energy+=values.square();
        }
        one=absolute.maxCoeff();squared=energy.maxCoeff();
    }
    // All nonzero inputs lie in [2^-128,2^128], length<=8192. Squaring and
    // positive reductions neither underflow nor overflow. gamma(2K+2)
    // covers multiplication and any bounded binary reduction order.
    const double lower=std::nextafter(1.-gamma(2*length+2),0.);
    out.one=up(one/lower);
    out.two=up(std::sqrt(up(squared/lower)));
    return std::isfinite(out.one)&&std::isfinite(out.two);
}
inline double tighter_magnitude(const Input& a,const Input& b,double ma,double mb,
                               double original,std::size_t capacity,std::size_t& peak,double* workspace,
                               void* context,void (*checkpoint)(void*)) {
    NormBounds left,right;
    const bool left_ok=norm_bounds(a,true,capacity,left,workspace,context,checkpoint);
    peak=std::max(peak,left.temporary);
    if(!left_ok)return original;
    const bool right_ok=norm_bounds(b,false,capacity,right,workspace,context,checkpoint);
    peak=std::max(peak,right.temporary);
    if(!right_ok)return original;
    return std::min({original,up(left.one*mb),up(ma*right.one),up(left.two*right.two)});
}

inline bool make_plan(const Input& a,const Input& b,double ma,double mb,Plan& p,
    double* workspace,std::size_t workspace_capacity,void* context,void (*checkpoint)(void*),
    bool output_f=false) {
    if(!a.rows||!a.columns||!b.columns||a.columns!=b.rows ||
       a.rows>INT_MAX||a.columns>8192||b.columns>INT_MAX||std::fegetround()!=FE_TONEAREST) return false;
    CBLAS_TRANSPOSE ta,tb;int la,lb;
    if(!describe(a,ta,la)||!describe(b,tb,lb)) return false;
    // Caller completes the same full finite/magnitude/negative-zero guard.
    if(!std::isfinite(ma)||!std::isfinite(mb)||ma<0||mb<0)return false;
    const auto k=a.columns;
    double magnitude=up(up(double(k)*ma)*mb);
    p.chunk=std::min<std::size_t>(512,k);
    const auto initial_pieces=(k+p.chunk-1)/p.chunk;

    // No extra scan for the already-admitted domain. Tightening never changes
    // the 0.000049 mathematical/replay budget. Explicit merges add at most q-1 rounding steps.
    if(up(gamma(2*p.chunk+2+initial_pieces)*magnitude)>budget)
        magnitude=tighter_magnitude(a,b,ma,mb,magnitude,workspace_capacity,p.norm_workspace,workspace,context,checkpoint);
    bool accepted=false;
    for(;p.chunk>=16;p.chunk/=2) {
        p.pieces=(k+p.chunk-1)/p.chunk;
        p.depth=p.pieces-1;
        p.error=up(gamma(2*p.chunk+2+p.pieces)*magnitude);
        if(p.error<=budget){accepted=true;break;}
    }
    bool balanced=false;
    if(!accepted) {
        // Long K can make q-1 explicit merges too expensive in error depth.
        // Keep a genuinely used balanced reduction for that domain, with the
        // same mature DGEMM pieces and the same unchanged mathematical budget.
        p.chunk=std::min<std::size_t>(512,k);
        for(;p.chunk>=16;p.chunk/=2) {
            p.pieces=(k+p.chunk-1)/p.chunk;
            p.depth=log2ceil(p.pieces);
            p.error=up(gamma(2*p.chunk+2+p.depth)*magnitude);
            if(p.error<=budget){accepted=true;balanced=true;break;}
        }
    }
    if(!accepted) return false;
    // Every ambiguous IEEE zero/sign cell is replayed in the original fold.
    p.ambiguity=up(up(gamma(2*k)*magnitude)+p.error);
    p.slots=balanced?p.depth+2:1;
    // The previously measured narrow tile is restricted to its benefited
    // physical geometry; requested F alone must not change a C-input route.
    const bool narrow_f=output_f && a.rs==1 && b.rs==1 &&
        a.rows>=2048 && b.columns>=2048 && k>=4096;
    p.tr=std::min<std::size_t>(1024,a.rows);
    p.tc=std::min<std::size_t>(narrow_f?256:1024,b.columns);
    const auto available=k*320+72;
    while(p.slots*p.tr*p.tc>available) {
        if(p.tr>=p.tc && p.tr>1)p.tr=(p.tr+1)/2;
        else if(p.tc>1)p.tc=(p.tc+1)/2;
        else return false;
    }
    p.buffer=p.tr*p.tc;
    return true;
}
// A physical line is contiguous in the requested final C/F result. Eigen
// checks every value, without a matrix copy or a heap-backed expression.
// Only lines containing a genuinely ambiguous cell enter the exact fold.
inline bool finish_tile(const Input& a,const Input& b,const Plan& p,double* output,
                       bool f,std::size_t i,std::size_t j,std::size_t rows,std::size_t cols,
                       Stats& stats,void* context,void (*checkpoint)(void*)) {
    using Array=Eigen::Array<double,Eigen::Dynamic,1>;
    const auto lines=f?cols:rows, length=f?rows:cols;
    for(std::size_t line=0;line<lines;++line) {
        if(checkpoint && line%64==0)checkpoint(context);
        double* target=output+(f?(j+line)*a.rows+i:(i+line)*b.columns+j);
        const Eigen::Map<const Array,Eigen::Unaligned> values(target,length);
        if(!values.isFinite().all())return false;
        if(values.abs().minCoeff()>p.ambiguity)continue;
        for(std::size_t inner=0;inner<length;++inner) {
            auto& value=target[inner];
            if(std::abs(value)>p.ambiguity)continue;
            const auto row=i+(f?inner:line), col=j+(f?line:inner);
            double canonical=0.;
            for(std::size_t k=0;k<a.columns;++k) {
                if(checkpoint && k%512==0)checkpoint(context);
                canonical+=a.at(row,k)*b.at(k,col);
            }
            const double certificate=up(p.error+up(std::abs(canonical-value)));
            if(!std::isfinite(canonical)||certificate>budget)return false;
            stats.returned_error_bound=std::max(stats.returned_error_bound,certificate);
            value=canonical;
            ++stats.replay;
        }
    }
    return true;
}

inline bool calculate_tree(const Input& a,const Input& b,const Plan& p,double* output,
                      bool f,std::vector<double>& scratch,Stats& stats,
                      void* context,void (*checkpoint)(void*)) {
    if(scratch.size()<p.buffer*p.slots) throw std::runtime_error("WORKSPACE_TOO_SMALL");
    CBLAS_TRANSPOSE ta,tb;int lda,ldb;
    if(!describe(a,ta,lda)||!describe(b,tb,ldb)) throw std::runtime_error("STRIDE");
    if(p.slots>16) throw std::runtime_error("TREE_DEPTH");
    for(std::size_t i=0;i<a.rows;i+=p.tr) for(std::size_t j=0;j<b.columns;j+=p.tc) {
        const auto rows=std::min(p.tr,a.rows-i),cols=std::min(p.tc,b.columns-j);
        std::array<double*,16> active{};
        std::array<bool,16> in_use{};
        for(std::size_t k=0;k<a.columns;k+=p.chunk) {
            if(checkpoint)checkpoint(context);
            std::size_t free=0;
            while(free<p.slots&&in_use[free])++free;
            if(free==p.slots)throw std::runtime_error("TREE_SLOT_OVERFLOW");
            double* current=scratch.data()+free*p.buffer;
            in_use[free]=true;
            const auto count=std::min(p.chunk,a.columns-k);
            cblas_dgemm(CblasRowMajor,ta,tb,int(rows),int(cols),int(count),1.,
                a.data+i*a.rs+k*a.cs,lda,b.data+k*b.rs+j*b.cs,ldb,
                0.,current,int(cols));
            ++stats.calls;
            if(checkpoint)checkpoint(context);
            std::size_t level=0;
            while(active[level]) {
                double* older=active[level];
                // Dense independent slots; alpha=1 preserves the same elementwise add.
                // The root caller already owns the shared BLAS thread budget.
                cblas_daxpy(int(rows*cols),1.,current,1,older,1);
                in_use[std::size_t(current-scratch.data())/p.buffer]=false;
                current=older;active[level]=nullptr;++level;
            }
            active[level]=current;
        }
        double* current=nullptr;
        for(std::size_t level=0;level<p.slots;++level)if(active[level]) {
            if(current)cblas_daxpy(int(rows*cols),1.,current,1,active[level],1);
            current=active[level];
        }
        if(!current)throw std::runtime_error("EMPTY_TREE");
        // Final tile only, preserving the requested C/F layout and owned result.
        if(f)Eigen::Map<Col,Eigen::Unaligned,Eigen::OuterStride<>>(output+i+j*a.rows,
            rows,cols,Eigen::OuterStride<>(a.rows))=Read(current,rows,cols);
        else Eigen::Map<Row,Eigen::Unaligned,Eigen::OuterStride<>>(output+i*b.columns+j,
            rows,cols,Eigen::OuterStride<>(b.columns))=Read(current,rows,cols);
        stats.output_copy_bytes+=rows*cols*sizeof(double);
        if(!finish_tile(a,b,p,output,f,i,j,rows,cols,stats,context,checkpoint))return false;
    }
    stats.returned_error_bound=std::max(stats.returned_error_bound,p.error);
    return true;
}

inline bool calculate(const Input& a,const Input& b,const Plan& p,double* output,
                      bool f,std::vector<double>& scratch,Stats& stats,
                      void* context,void (*checkpoint)(void*)) {
    if(p.slots>1)return calculate_tree(a,b,p,output,f,scratch,stats,context,checkpoint);
    if(scratch.size()<p.buffer) throw std::runtime_error("WORKSPACE_TOO_SMALL");
    CBLAS_TRANSPOSE ta,tb;int lda,ldb;
    if(!describe(a,ta,lda)||!describe(b,tb,ldb)) throw std::runtime_error("STRIDE");
    // Each DGEMM starts with beta=0. The library cannot carry the previous
    // partial sum through its hidden inner reduction. Only explicit Eigen
    // additions combine pieces, giving the conservative gamma(2b+2+q) bound.
    if(f) {
        ta=ta==CblasNoTrans?CblasTrans:CblasNoTrans;
        tb=tb==CblasNoTrans?CblasTrans:CblasNoTrans;
    }
    for(std::size_t i=0;i<a.rows;i+=p.tr) for(std::size_t j=0;j<b.columns;j+=p.tc) {
        const auto rows=std::min(p.tr,a.rows-i),cols=std::min(p.tc,b.columns-j);
        double* target=output+(f?i+j*a.rows:i*b.columns+j);
        for(std::size_t k=0;k<a.columns;k+=p.chunk) {
            if(checkpoint)checkpoint(context);
            const auto count=std::min(p.chunk,a.columns-k);
            const bool first=k==0;
            double* current=first?target:scratch.data();
            const auto leading=first?(f?a.rows:b.columns):(f?rows:cols);
            cblas_dgemm(f?CblasColMajor:CblasRowMajor,ta,tb,int(rows),int(cols),int(count),1.,
                a.data+i*a.rs+k*a.cs,lda,b.data+k*b.rs+j*b.cs,ldb,
                0.,current,int(leading));
            ++stats.calls;
            if(checkpoint)checkpoint(context);
            if(!first) {
                if(f) Eigen::Map<Col,Eigen::Unaligned,Eigen::OuterStride<>>(target,rows,cols,
                    Eigen::OuterStride<>(a.rows))+=Eigen::Map<const Col,Eigen::Unaligned>(current,rows,cols);
                else Eigen::Map<Row,Eigen::Unaligned,Eigen::OuterStride<>>(target,rows,cols,
                    Eigen::OuterStride<>(b.columns))+=Read(current,rows,cols);
            }
        }
        // Already in the independently owned final layout; no final tile copy.
        // This strategy may decline after writes; the root caller overwrites all
        // private cells through the sole canonical implementation on replay.
        if(!finish_tile(a,b,p,output,f,i,j,rows,cols,stats,context,checkpoint))return false;
    }
    stats.returned_error_bound=std::max(stats.returned_error_bound,p.error);
    return true;
}
} // namespace calmetrics_engine::ops::partitioned


#endif
