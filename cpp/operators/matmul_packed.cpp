// Pinned Eigen pack/gebp adapter; mathematical policy remains ascending-K mul/add.
#include "calmetrics_engine/operators.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include <algorithm>
#include <cstdint>
#include <type_traits>

#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_MPL2_ONLY
#define EIGEN_FAST_MATH 0
#include <Eigen/Core>
static_assert(EIGEN_WORLD_VERSION == 3 && EIGEN_MAJOR_VERSION == 4 &&
              EIGEN_MINOR_VERSION == 0, "Pinned internal Eigen ABI required");
// No ordinary Eigen real-valued expression selects true/true (BlasUtil.h).
// This TU must be the only user of this otherwise-unused internal combination.
// Never specialize the existing false/false NEON traits or change Eigen macros
// in any linked engine TU. See README for the whole-program adoption boundary.
namespace Eigen { namespace internal {
template<>
class gebp_traits<double, double, true, true, Architecture::NEON, GEBPPacketFull>
    : public gebp_traits<double, double, false, false, Architecture::Generic,
                         GEBPPacketFull> {
public:
    using Base = gebp_traits<double, double, false, false, Architecture::Generic,
                             GEBPPacketFull>;
    enum { ConjLhs = true, ConjRhs = true, mr = 6, nr = 4 };
    template<class LP, class RP, class AP, class Lane>
    EIGEN_STRONG_INLINE void madd(const LP& left, const RP& right, AP& acc,
                                  RP&, const Lane&) const {
        // Row-major output is evaluated as B^T * A^T. Multiplication operands
        // are explicitly restored to original A then B, including NaN priority.
        acc = padd(acc, pmul(right, left));
    }
    template<class LP, class AP, class Lane>
    EIGEN_STRONG_INLINE void madd(const LP& left, const RhsPacketx4& right,
                                  AP& acc, RhsPacket& tmp, const Lane& lane) const {
        madd(left, right.get(lane), acc, tmp, lane);
    }
    template<class Packet>
    EIGEN_STRONG_INLINE void acc(const Packet& value, const Packet&, Packet& dest) const {
        // Private call always uses alpha=1, one full-K invocation per output.
        dest = value;
    }
};
}} // Eigen::internal

namespace {
using Index = Eigen::Index;
using Traits = Eigen::internal::gebp_traits<double, double, true, true,
                                          Eigen::Architecture::NEON>;
static_assert(sizeof(Index) == sizeof(std::ptrdiff_t), "Index ABI mismatch");
static_assert(Traits::mr == 6 && Traits::nr == 4 && Traits::LhsProgress == 2 &&
              Traits::RhsProgress == 1 && Traits::ResPacketSize == 2,
              "Only the full 6x4 ascending-K arm64 path is admitted");

// Reads requested by stock gebp are defined zero values, not reads of C.
// Even compile-required scalar/gather branches have no uninitialized reads;
// those branches are excluded at runtime by rows%6==0 and cols%4==0.
struct AssignmentCell {
    double* ptr;
    EIGEN_STRONG_INLINE void operator+=(double value) const { *ptr = value; }
};
template<bool FOutput>
struct AssignmentLinear {
    double* ptr;
    Index stride;
    EIGEN_STRONG_INLINE void prefetch(int) const {}
    template<class Packet>
    EIGEN_STRONG_INLINE Packet loadPacket(Index) const {
        return Eigen::internal::pset1<Packet>(0.0);
    }
    template<class Packet>
    EIGEN_STRONG_INLINE void storePacket(Index i, const Packet& value) const {
        if constexpr (FOutput)
            Eigen::internal::pscatter<double, Packet>(ptr + i * stride, value, stride);
        else
            Eigen::internal::pstoreu<double, Packet>(ptr + i, value);
    }
};
template<bool FOutput>
struct AssignmentMapper {
    using LinearMapper = AssignmentLinear<FOutput>;
    double* ptr;
    Index stride;
    EIGEN_STRONG_INLINE LinearMapper getLinearMapper(Index i, Index j) const {
        if constexpr (FOutput) return {ptr + i * stride + j, stride};
        else return {ptr + i + j * stride, stride};
    }
    EIGEN_STRONG_INLINE AssignmentCell operator()(Index i, Index j) const {
        if constexpr (FOutput) return {ptr + i * stride + j};
        else return {ptr + i + j * stride};
    }
    template<class Packet>
    EIGEN_STRONG_INLINE Packet gatherPacket(Index, Index) const {
        return Eigen::internal::pset1<Packet>(0.0);
    }
    template<class Packet>
    EIGEN_STRONG_INLINE void scatterPacket(Index i, Index j, const Packet& value) const {
        if constexpr (FOutput)
            Eigen::internal::pscatter<double, Packet>(ptr + i * stride + j, value, Index(1));
        else
            Eigen::internal::pscatter<double, Packet>(ptr + i + j * stride, value, stride);
    }
};


struct Plan {
    std::size_t prefix, bulk_cols, mc = 192, nc = 128;
    std::size_t lhs_offset = 0, rhs_offset;
    int at_order, bt_order;
    Index at_stride, bt_stride;
};
// Existing scalar MatMul reference arithmetic: +0; ascending k; separate mul/add.
// Only tails use this loop. No complete GEMM body is duplicated or replaced.
void ordered_region(const double* a, std::ptrdiff_t ar, std::ptrdiff_t ac,
                    const double* b, std::ptrdiff_t br, std::ptrdiff_t bc,
                    std::size_t m, std::size_t n, std::size_t k, double* out, bool f_output,
                    std::size_t row_begin, std::size_t row_end,
                    std::size_t col_begin, std::size_t col_end) {
    for (auto i=row_begin; i<row_end; ++i)
        for (auto j=col_begin; j<col_end; ++j) {
            double total = 0.0;
            for (std::size_t t=0; t<k; ++t)
                total += a[static_cast<Index>(i)*ar+static_cast<Index>(t)*ac] *
                         b[static_cast<Index>(t)*br+static_cast<Index>(j)*bc];
            out[f_output ? j*m+i : i*n+j] = total;
        }
}

template<int BtOrder, int AtOrder, bool FOutput>
void packed_prefix(const double* a, const double* b, double* out,
                   std::size_t k, std::size_t m, std::size_t n, const Plan& p, double* workspace) {
    using namespace Eigen::internal;
    using LhsMap = const_blas_data_mapper<double,Index,BtOrder>;
    using RhsMap = const_blas_data_mapper<double,Index,AtOrder>;
    gemm_pack_lhs<double,Index,LhsMap,6,2,typename Traits::LhsPacket4Packing,
                  BtOrder,false,false> pack_lhs;
    gemm_pack_rhs<double,Index,RhsMap,4,AtOrder,false,false> pack_rhs;
    gebp_kernel<double,double,Index,AssignmentMapper<FOutput>,6,4,true,true> gebp;
    const LhsMap left(b,p.bt_stride);
    const RhsMap right(a,p.at_stride);
    auto* lhs = workspace+p.lhs_offset;
    auto* rhs = workspace+p.rhs_offset;
    for (std::size_t i=0; i<p.prefix; i+=p.nc) {
        const auto cols = std::min(p.nc,p.prefix-i);
        pack_rhs(rhs,right.getSubMapper(0,static_cast<Index>(i)),
                 static_cast<Index>(k),static_cast<Index>(cols));
        for (std::size_t j=0; j<p.bulk_cols; j+=p.mc) {
            const auto rows = std::min(p.mc,p.bulk_cols-j);
            pack_lhs(lhs,left.getSubMapper(static_cast<Index>(j),0),
                     static_cast<Index>(k),static_cast<Index>(rows));
            // cols%4==0 and rows%6==0: never grouped/tail gebp branches.
            // Same B^T*A^T kernel coordinates; only the final owned address changes.
            gebp(AssignmentMapper<FOutput>{out+(FOutput ? j*m+i : i*n+j),
                 static_cast<Index>(FOutput ? m : n)},lhs,rhs,
                 static_cast<Index>(rows),static_cast<Index>(k),
                 static_cast<Index>(cols),1.0);
        }
    }
}

} // namespace
#endif

namespace calmetrics_engine::ops {
std::size_t matmul_scratch_doubles(const Shape &a, const Shape &b) {
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    if (__builtin_available(macOS 15.0, *)) {
        if (a.rank != 2 || b.rank != 2) return 0;
        // Bounded independent row blocks can have a long output history and
        // small inner dimension. Reuse the same ordered Eigen pack/gebp path.
        const bool rectangular = a.dim[0] >= 32 && a.dim[0] <= 128 &&
            a.dim[1] >= 1 && a.dim[1] <= 128 && b.dim[1] >= 128;
        const bool large = a.dim[0] >= 1024 && a.dim[1] >= 256 && b.dim[1] >= 1024;
        if (!rectangular && !large) return 0;
        // 192*K + 128*K; two 256B prefetch guards and 64B base alignment.
        constexpr auto limit = static_cast<std::size_t>(PTRDIFF_MAX) / sizeof(double);
        require(a.dim[1] <= (limit - 72) / 320, "SHAPE_OVERFLOW");
        return a.dim[1] * 320 + 72;
    }
#else
    (void)a; (void)b;
#endif
    return 0;
}

bool matmul_packed_eligible(const Value &a, const Value &b) {
    if (!matmul_scratch_doubles(a.shape, b.shape)) return false;
    const auto dense = [](const Value &v) {
        return (v.stride[1] == 1 && v.stride[0] >= static_cast<std::ptrdiff_t>(v.shape.dim[1])) ||
               (v.stride[0] == 1 && v.stride[1] >= static_cast<std::ptrdiff_t>(v.shape.dim[0]));
    };
    return dense(a) && dense(b);
}

bool library_matmul_packed(const Value &a, const Value &b, double *output,
                           Workspace &work, Audit &audit, bool f_output) {
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    if (!matmul_packed_eligible(a, b)) return false;
    const auto m = a.shape.dim[0], k = a.shape.dim[1], n = b.shape.dim[1];
    const Plan p{m-m%4, n-n%6, 192, 128, 0, k*192+32,
                 a.stride[1] == 1 ? Eigen::ColMajor : Eigen::RowMajor,
                 b.stride[1] == 1 ? Eigen::ColMajor : Eigen::RowMajor,
                 a.stride[1] == 1 ? a.stride[0] : a.stride[1],
                 b.stride[1] == 1 ? b.stride[0] : b.stride[1]};
    require(work.doubles.size() >= matmul_scratch_doubles(a.shape,b.shape), "WORKSPACE_TOO_SMALL");
    const auto checked_mul = [](std::size_t x, std::size_t y) {
        require(!y || x <= SIZE_MAX/y, "SHAPE_OVERFLOW"); return x*y;
    };
    const auto packed_lhs = checked_mul((p.prefix+p.nc-1)/p.nc,p.bulk_cols);
    require(p.prefix <= SIZE_MAX-packed_lhs, "SHAPE_OVERFLOW");
    const auto copied = checked_mul(checked_mul(packed_lhs+p.prefix,k),sizeof(double));
    const auto padding = (64-(reinterpret_cast<std::uintptr_t>(work.doubles.data())&63))&63;
    auto *panels = work.doubles.data()+padding/sizeof(double);
    const auto dispatch = [&](auto order) {
        constexpr bool FOutput = decltype(order)::value;
    if (p.bt_order == Eigen::ColMajor) {
        if (p.at_order == Eigen::ColMajor) packed_prefix<Eigen::ColMajor,Eigen::ColMajor,FOutput>(
            static_cast<const double*>(a.data),static_cast<const double*>(b.data),output,k,m,n,p,panels);
        else packed_prefix<Eigen::ColMajor,Eigen::RowMajor,FOutput>(
            static_cast<const double*>(a.data),static_cast<const double*>(b.data),output,k,m,n,p,panels);
    } else {
        if (p.at_order == Eigen::ColMajor) packed_prefix<Eigen::RowMajor,Eigen::ColMajor,FOutput>(
            static_cast<const double*>(a.data),static_cast<const double*>(b.data),output,k,m,n,p,panels);
        else packed_prefix<Eigen::RowMajor,Eigen::RowMajor,FOutput>(
            static_cast<const double*>(a.data),static_cast<const double*>(b.data),output,k,m,n,p,panels);
    }
    };
    if (f_output) dispatch(std::true_type{});
    else dispatch(std::false_type{});
    const auto *ap = static_cast<const double*>(a.data), *bp = static_cast<const double*>(b.data);
    ordered_region(ap,a.stride[0],a.stride[1],bp,b.stride[0],b.stride[1],m,n,k,output,f_output,0,p.prefix,p.bulk_cols,n);
    ordered_region(ap,a.stride[0],a.stride[1],bp,b.stride[0],b.stride[1],m,n,k,output,f_output,p.prefix,m,0,n);
    audit.isa = f_output ? "eigen-packed-matmul-f-root" : "eigen-packed-matmul";
    audit.vector_elements = p.prefix*p.bulk_cols;
    audit.algorithm_copy_bytes = copied;
    return true;
#else
    (void)a; (void)b; (void)output; (void)work; (void)audit; (void)f_output;
    return false;
#endif
}
} // namespace calmetrics_engine::ops
