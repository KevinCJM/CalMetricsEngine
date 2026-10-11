#include "calmetrics_engine/operators.hpp"
#include "elementwise_math.hpp"
#include <algorithm>
#include <cmath>

#ifdef CALMETRICS_HAVE_EIGEN_ROTATION
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_MPL2_ONLY
#define EIGEN_FAST_MATH 0
#include <Eigen/Core>
#endif
namespace calmetrics_engine::ops {
namespace {
constexpr double pi = 3.141592653589793238462643383279502884;
double normal_ppf(double p) {
    require(!(p <= 0.0 || p >= 1.0), "DOMAIN_ERROR");
    constexpr double a0 = -3.969683028665376e1, a1 = 2.209460984245205e2, a2 = -2.759285104469687e2;
    constexpr double a3 = 1.383577518672690e2, a4 = -3.066479806614716e1, a5 = 2.506628277459239;
    constexpr double b0 = -5.447609879822406e1, b1 = 1.615858368580409e2, b2 = -1.556989798598866e2;
    constexpr double b3 = 6.680131188771972e1, b4 = -1.328068155288572e1;
    constexpr double c0 = -7.784894002430293e-3, c1 = -3.223964580411365e-1,
                     c2 = -2.400758277161838;
    constexpr double c3 = -2.549732539343734, c4 = 4.374664141464968, c5 = 2.938163982698783;
    constexpr double d0 = 7.784695709041462e-3, d1 = 3.224671290700398e-1, d2 = 2.445134137142996,
                     d3 = 3.754408661907416;
    if (p < 0.02425 || p > 1.0 - 0.02425) {
        const double q = std::sqrt(-2.0 * std::log(p < 0.02425 ? p : 1.0 - p));
        const double v = (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) /
                         ((((d0 * q + d1) * q + d2) * q + d3) * q + 1.0);
        return p < 0.02425 ? v : -v;
    }
    const double q = p - 0.5, r = q * q;
    return (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q /
           (((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1.0);
}
bool compare(Op op, double x, double y) {
    switch (op) {
    case Op::equal:
        return x == y;
    case Op::not_equal:
        return x != y;
    case Op::less_than:
        return x < y;
    case Op::less_equal:
        return x <= y;
    case Op::greater_than:
        return x > y;
    case Op::greater_equal:
        return x >= y;
    default:
        throw Error("INVALID_COMPARISON");
    }
}
void where_span(const Prepared &, Output &);
} // namespace

double scalar_math(Op op, double x, double y, double z) {
    switch (op) {
    case Op::add:
        return x + y;
    case Op::subtract:
        return elementwise_detail::subtract_step(x, y);
    case Op::multiply:
        return elementwise_detail::multiply_step(x, y);
    case Op::divide:
        require(y != 0.0, "DIVIDE_BY_ZERO");
        return elementwise_detail::nonzero_divide_step(x, y);
    case Op::power: {
        const double v = std::pow(x, y);
        require(std::isfinite(v), "DOMAIN_ERROR");
        return v;
    }
    case Op::minimum:
        return std::min(x, y);
    case Op::maximum:
        return std::max(x, y);
    case Op::negate:
        return -x;
    case Op::absolute:
        return std::abs(x);
    case Op::sqrt:
        require(!(x < 0.0), "DOMAIN_ERROR");
        return std::sqrt(x);
    case Op::clip:
        require(!(y > z), "INVALID_PARAMETER");
        return std::min(std::max(x, y), z);
    case Op::log:
        require(!(x <= 0.0), "DOMAIN_ERROR");
        return std::log(x);
    case Op::exp: {
        const double v = std::exp(x);
        require(std::isfinite(v), "DOMAIN_ERROR");
        return v;
    }
    case Op::expm1:
        // This primitive has IEEE propagation, unlike the historical exp
        // contract: preserve signed zero, NaN, infinities and finite overflow.
        return std::expm1(x);
    case Op::reciprocal:
        require(x != 0.0, "DIVIDE_BY_ZERO");
        return 1.0 / x;
    case Op::sign:
        return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
    case Op::normal_pdf:
        return std::exp(-0.5 * x * x) / std::sqrt(2.0 * pi);
    case Op::normal_cdf:
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    case Op::floor:
        return std::floor(x);
    case Op::cos:
        return std::cos(x);
    case Op::normal_ppf:
        return normal_ppf(x);
    default:
        throw Error("INVALID_ELEMENTWISE");
    }
}

void elementwise(const Prepared &p, Output &out, Workspace &, Isa requested, Audit &audit) {
    const auto op = p.spec->op;
    const auto &a = p.args;
    const auto n = out.shape.size();
    if ((op == Op::equal || op == Op::not_equal) &&
        (a[0].kind == Kind::integer || a[1].kind == Kind::integer)) {
        for (std::size_t i = 0; i < n; ++i) {
            const auto lhs = a[0].kind == Kind::integer
                ? a[0].i(i) : static_cast<std::int64_t>(a[0].scalar);
            const auto rhs = a[1].kind == Kind::integer
                ? a[1].i(i) : static_cast<std::int64_t>(a[1].scalar);
            out.set_mask(i, op == Op::equal ? lhs == rhs : lhs != rhs);
        }
        return;
    }
    if (op == Op::clip)
        require(!(a[1].scalar > a[2].scalar), "INVALID_PARAMETER");
    if (out.order == MemoryOrder::c && out.shape.rank > 0 && (!a[0].contiguous() || !a[1].contiguous())) {
        switch (op) {
        case Op::add: case Op::subtract: case Op::multiply: case Op::divide:
        case Op::minimum: case Op::maximum: case Op::negate: case Op::absolute:
        case Op::sqrt: case Op::finite_mask:
            elementwise_span(p, out, {0, 0});
            return;
        default: break;
        }
    }
    std::size_t first = 0;
    if (out.order == MemoryOrder::c && simd_eligible(op) && a[0].contiguous() && a[1].contiguous() && out.shape.rank > 0) {
        // Validate throwing domains before a vector instruction; no fast-math assumptions.
        if (op == Op::divide) {
            for (std::size_t i = 0; i < n; ++i)
                require(a[1].f(i) != 0.0, "DIVIDE_BY_ZERO");
        } else if (op == Op::reciprocal || op == Op::sqrt) {
            for (std::size_t i = 0; i < n; ++i) {
                const double x = a[0].f(i);
                require(op == Op::sqrt ? !(x < 0.0) : x != 0.0,
                        op == Op::sqrt ? "DOMAIN_ERROR" : "DIVIDE_BY_ZERO");
            }
        }
        first = simd_transform(op, a[0], a[1], out, n, requested, audit);
    }
    if (first == 0 && n >= 128 && out.order == MemoryOrder::c) {
        // Canonical arithmetic already has span-specialized bodies. Resolve
        // rank/stride once per inner run instead of on each output element.
        switch (op) {
        case Op::where: return where_span(p, out);
        case Op::add: case Op::subtract: case Op::multiply: case Op::divide:
        case Op::sqrt: case Op::minimum: case Op::maximum: case Op::negate:
        case Op::absolute: case Op::equal: case Op::not_equal: case Op::less_than:
        case Op::less_equal: case Op::greater_than: case Op::greater_equal:
        case Op::finite_mask: return elementwise_span(p, out, {0, 0});
        default: break;
        }
    }
    for (std::size_t i = first; i < n; ++i) {
        if (op >= Op::equal && op <= Op::greater_equal)
            out.set_mask(i, compare(op, a[0].f(i), a[1].f(i)));
        else if (op == Op::logical_and)
            out.set_mask(i, a[0].u(i) && a[1].u(i));
        else if (op == Op::logical_or)
            out.set_mask(i, a[0].u(i) || a[1].u(i));
        else if (op == Op::logical_not)
            out.set_mask(i, !a[0].u(i));
        else if (op == Op::finite_mask)
            out.set_mask(i, std::isfinite(a[0].f(i)));
        else if (op == Op::where)
            out.set(i, a[0].u(i) ? a[1].f(i) : a[2].f(i));
        else if (op == Op::divide_or_default) {
            const double x = a[0].f(i), y = a[1].f(i);
            out.set(i, !std::isfinite(x) || !std::isfinite(y)
                           ? std::numeric_limits<double>::quiet_NaN()
                       : std::abs(y) < 1e-12 ? a[2].scalar
                                             : x / y);
        } else
            out.set(i, scalar_math(op, a[0].f(i), a[1].f(i), a[2].scalar));
    }
}

namespace {
void where_span(const Prepared &p, Output &out) {
    const auto n = out.shape.size();
    const auto &mask = p.args[0];
    const auto scalar_mask = static_cast<std::uint8_t>(mask.scalar);
    auto *target = static_cast<double *>(out.data);
    std::array<bool, 3> contiguous{};
    for (std::size_t j = 0; j < 3; ++j) contiguous[j] = p.args[j].contiguous();
    for (std::size_t first = 0; first < n;) {
        auto count = n - first;
        std::array<std::ptrdiff_t, 3> offsets{}, steps{};
        for (std::size_t j = 0; j < 3; ++j) {
            const auto &v = p.args[j];
            if (!v.shape.rank) continue;
            if (!contiguous[j]) count = std::min(count,
                v.shape.dim[v.shape.rank - 1] - first % v.shape.dim[v.shape.rank - 1]);
            offsets[j] = contiguous[j] ? std::ptrdiff_t(first) : v.offset(first);
            steps[j] = contiguous[j] ? 1 : v.stride[v.shape.rank - 1];
        }
        const auto *condition = mask.shape.rank
            ? static_cast<const std::uint8_t *>(mask.data) + offsets[0] : &scalar_mask;
        const auto *left = p.args[1].shape.rank
            ? static_cast<const double *>(p.args[1].data) + offsets[1] : &p.args[1].scalar;
        const auto *right = p.args[2].shape.rank
            ? static_cast<const double *>(p.args[2].data) + offsets[2] : &p.args[2].scalar;
        for (std::size_t j = 0; j < count; ++j)
            target[first + j] = condition[std::ptrdiff_t(j) * steps[0]]
                ? left[std::ptrdiff_t(j) * steps[1]] : right[std::ptrdiff_t(j) * steps[2]];
        first += count;
    }
}
template <Op op> void span_loop(const Prepared &p, Output &out,
                                const std::array<std::size_t, 2> &starts) {
    const auto &a = p.args[0], &b = p.args[1];
    const bool ac = a.contiguous(), bc = b.contiguous();
    const auto n = out.shape.size();
    // Resolve the multidimensional offset once per contiguous/strided inner
    // run, rather than dividing flat indices at every arithmetic operation.
    for (std::size_t first = 0; first < n;) {
        auto count = n - first;
        if (a.shape.rank && !ac) count = std::min(count,
            a.shape.dim[a.shape.rank - 1] - (starts[0] + first) % a.shape.dim[a.shape.rank - 1]);
        if (b.shape.rank && !bc) count = std::min(count,
            b.shape.dim[b.shape.rank - 1] - (starts[1] + first) % b.shape.dim[b.shape.rank - 1]);
        const auto *lhs = a.shape.rank ? static_cast<const double *>(a.data) + a.offset(starts[0] + first) : &a.scalar;
        const auto *rhs = b.shape.rank ? static_cast<const double *>(b.data) + b.offset(starts[1] + first) : &b.scalar;
        const auto sa = !a.shape.rank ? 0 : ac ? 1 : a.stride[a.shape.rank - 1];
        const auto sb = !b.shape.rank ? 0 : bc ? 1 : b.stride[b.shape.rank - 1];
        for (std::size_t j = 0; j < count; ++j) {
            const double x = lhs[static_cast<std::ptrdiff_t>(j) * sa];
            const double y = rhs[static_cast<std::ptrdiff_t>(j) * sb];
            if constexpr (op >= Op::equal && op <= Op::greater_equal)
                static_cast<std::uint8_t *>(out.data)[first+j] = compare(op, x, y);
            else if constexpr (op == Op::finite_mask)
                static_cast<std::uint8_t *>(out.data)[first+j] = std::isfinite(x);
            else static_cast<double *>(out.data)[first+j] = scalar_math(op, x, y);
        }
        first += count;
    }
}
}
void elementwise_span(const Prepared &p, Output &out, const std::array<std::size_t, 2> &starts) {
    switch (p.spec->op) {
#define SPAN(name) case Op::name: return span_loop<Op::name>(p, out, starts);
    SPAN(add) SPAN(subtract) SPAN(multiply) SPAN(divide) SPAN(sqrt) SPAN(minimum) SPAN(maximum)
    SPAN(negate) SPAN(absolute) SPAN(equal) SPAN(not_equal) SPAN(less_than)
    SPAN(less_equal) SPAN(greater_than) SPAN(greater_equal) SPAN(finite_mask)
#undef SPAN
    default: throw Error("INVALID_POINTWISE_SPAN");
    }
}
namespace {
#ifdef CALMETRICS_HAVE_EIGEN_ROTATION
template<Op operation,class Left,class Right>
auto pair_array_math(const Left &left,const Right &right) {
    if constexpr(operation==Op::add) return left+right;
    if constexpr(operation==Op::subtract) return left-right;
    if constexpr(operation==Op::multiply) return left*right;
    if constexpr(operation==Op::divide) return left/right;
}

template<Op first,Op second,bool SourceLeft,bool FirstLeft>
bool eigen_scalar_pair_run(const double *source,double *destination,std::size_t count,
                           double parameter,double other,bool *separated,double error) {
    using Span=Eigen::Array<double,Eigen::Dynamic,1>;
    bool safe=true;
    for (std::size_t first_index=0;first_index<count;first_index+=65536) {
    const auto width=std::min<std::size_t>(65536,count-first_index);
    const Eigen::Map<const Span,Eigen::Unaligned> values(source+first_index,width);
    Eigen::Map<Span,Eigen::Unaligned> result(destination+first_index,width);
    // Lazy coefficient-wise expressions evaluate each old input before its
    // same-position store, including exact in-place graph root ownership.
    // -ffp-contract=off retains both original arithmetic operations.
    if constexpr(SourceLeft) {
        const auto middle=pair_array_math<first>(values,parameter);
        if constexpr(FirstLeft) result=pair_array_math<second>(middle,other);
        else result=pair_array_math<second>(other,middle);
    } else {
        const auto middle=pair_array_math<first>(parameter,values);
        if constexpr(FirstLeft) result=pair_array_math<second>(middle,other);
        else result=pair_array_math<second>(other,middle);
    }
    if (separated && !(result.isFinite().all() && result.abs().minCoeff()>error)) safe=false;
    }
    return safe;
}

// A failed same-order scalar qualification must still reach the existing
// column/tile kernels for F input, pitched columns and reversed F rows.
int eigen_scalar_pair_source(const Prepared &a,const Prepared &b,const Output &out,bool left) {
    if (out.kind!=Kind::number || out.shape.rank!=2 || out.shape.size()<65536)
        return -1;
    const auto geometry=[&](const Value &v) {
        return v.kind==Kind::number && v.shape==out.shape && v.data &&
            (out.order==MemoryOrder::c ? v.contiguous() : v.f_contiguous());
    };
    const auto &other=b.args[left ? 1 : 0];
    if (other.kind!=Kind::number || other.shape.rank) return -1;
    if (b.spec->op==Op::divide && (!left || other.scalar==0.0)) return -1;
    if (geometry(a.args[0]) && a.args[1].kind==Kind::number && !a.args[1].shape.rank) return 0;
    if (geometry(a.args[1]) && a.args[0].kind==Kind::number && !a.args[0].shape.rank) return 1;
    return -1;
}

template<Op first,Op second>
bool eigen_scalar_pair(const Prepared &a,const Prepared &b,Output &out,bool left,
                       bool *separated,double error) {
    const auto source_index=eigen_scalar_pair_source(a,b,out,left);
    if (source_index<0) return false;
    const bool source_left=source_index==0;
    const auto &source=a.args[source_index];
    const auto parameter=a.args[1-source_index].scalar;
    const auto &other=b.args[left ? 1 : 0];
    const auto *data=static_cast<const double *>(source.data);
    auto *target=static_cast<double *>(out.data);
    bool safe=false;
    if (source_left) {
        if (left) safe=eigen_scalar_pair_run<first,second,true,true>(data,target,out.shape.size(),parameter,other.scalar,separated,error);
        else safe=eigen_scalar_pair_run<first,second,true,false>(data,target,out.shape.size(),parameter,other.scalar,separated,error);
    } else {
        if (left) safe=eigen_scalar_pair_run<first,second,false,true>(data,target,out.shape.size(),parameter,other.scalar,separated,error);
        else safe=eigen_scalar_pair_run<first,second,false,false>(data,target,out.shape.size(),parameter,other.scalar,separated,error);
    }
    if (separated) *separated=safe;
    return true;
}

template<Op first,Op second>
bool eigen_row_pair(const Prepared &a,const Prepared &b,Output &out,bool left) {
    if(out.order!=MemoryOrder::c || out.shape.rank!=2 || out.shape.dim[0]<32 || out.shape.dim[1]<8) return false;
    const auto n=out.shape.dim[0],d=out.shape.dim[1];
    const auto source_geometry=[&](const Value &v) {
        return v.kind==Kind::number && v.shape==out.shape && v.stride[1]==1 && v.stride[0]>=std::ptrdiff_t(d);
    };
    const auto parameter_geometry=[&](const Value &v) {
        return v.kind==Kind::number && v.shape==out.shape && v.stride[0]==0 && v.stride[1]==1;
    };
    const bool source_left=source_geometry(a.args[0]) && parameter_geometry(a.args[1]);
    if(!source_left && !(source_geometry(a.args[1]) && parameter_geometry(a.args[0]))) return false;
    const auto &source=a.args[source_left ? 0 : 1],&parameter=a.args[source_left ? 1 : 0],&other=b.args[left ? 1 : 0];
    if(!parameter_geometry(other)) return false;
    if constexpr(second==Op::divide) {
        if(!left) return false;
        for(std::size_t column=0;column<d;++column)
            if(static_cast<const double *>(other.data)[column]==0.) return false;
    }
    using Row=Eigen::Array<double,Eigen::Dynamic,1>;
    const Eigen::Map<const Row,Eigen::Unaligned> p(static_cast<const double *>(parameter.data),d);
    const Eigen::Map<const Row,Eigen::Unaligned> q(static_cast<const double *>(other.data),d);
    const auto *data=static_cast<const double *>(source.data);
    auto *destination=static_cast<double *>(out.data);
    for(std::size_t row=0;row<n;++row) {
        const Eigen::Map<const Row,Eigen::Unaligned> values(data+std::ptrdiff_t(row)*source.stride[0],d);
        // Lazy expressions write one independent row. -ffp-contract=off keeps
        // the two arithmetic stages; there is no dynamically allocated Row.
        const auto middle=source_left ? pair_array_math<first>(values,p) : pair_array_math<first>(p,values);
        Eigen::Map<Row,Eigen::Unaligned> result(destination+row*d,d);
        if(left) result=pair_array_math<second>(middle,q);
        else result=pair_array_math<second>(q,middle);
    }
    return true;
}
#endif

template<Op first, Op second, MemoryOrder order>
void pair_loop_order(const Prepared &a, const Prepared &b, Output &out, bool first_left) {
    const std::array<Value,3> inputs{a.args[0],a.args[1],b.args[first_left ? 1 : 0]};
    const auto n=out.shape.size();
    for (std::size_t offset=0;offset<n;) {
        auto count=n-offset;
        std::array<const double *,3> data{};
        std::array<std::ptrdiff_t,3> step{};
        for (std::size_t j=0;j<3;++j) {
            const auto &v=inputs[j];
            if (v.shape.rank && !v.contiguous())
                count=std::min(count,v.shape.dim[v.shape.rank-1]-offset%v.shape.dim[v.shape.rank-1]);
            data[j]=v.shape.rank ? static_cast<const double *>(v.data)+v.offset(offset) : &v.scalar;
            step[j]=v.shape.rank ? (v.contiguous() ? 1 : v.stride[v.shape.rank-1]) : 0;
        }
        for (std::size_t i=0;i<count;++i) {
            const auto middle=scalar_math(first,data[0][std::ptrdiff_t(i)*step[0]],data[1][std::ptrdiff_t(i)*step[1]]);
            const auto other=data[2][std::ptrdiff_t(i)*step[2]];
            const auto result=first_left ? scalar_math(second,middle,other) : scalar_math(second,other,middle);
            if constexpr (order == MemoryOrder::c) static_cast<double *>(out.data)[offset+i]=result;
            else out.set(offset+i,result);
        }
        offset+=count;
    }
}
template<Op first, Op second>
void pair_loop(const Prepared &a, const Prepared &b, Output &out, bool left,bool *separated,double error) {
#ifdef CALMETRICS_HAVE_EIGEN_ROTATION
    if(eigen_scalar_pair<first,second>(a,b,out,left,separated,error)) return;
    if(eigen_row_pair<first,second>(a,b,out,left)) return;
#endif
    // Resolve storage once per kernel, preserving the original C inner loop.
    if (out.order == MemoryOrder::c) pair_loop_order<first,second,MemoryOrder::c>(a,b,out,left);
    else pair_loop_order<first,second,MemoryOrder::f>(a,b,out,left);
}
template<Op first>
void pair_second(const Prepared &a,const Prepared &b,Output &out,bool left,bool *separated,double error) {
    switch(b.spec->op) {
#define PAIR(name) case Op::name: return pair_loop<first,Op::name>(a,b,out,left,separated,error);
    PAIR(add) PAIR(subtract) PAIR(multiply) PAIR(divide)
#undef PAIR
    default: throw Error("INVALID_POINTWISE_PAIR");
    }
}
}

#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
namespace {
bool column_parameter(const Value &v,const Shape &shape) {
    return !v.shape.rank || (v.shape==shape && v.stride[0]==0);
}
__attribute__((noinline))
bool column_pair_eligible(const Prepared &a,const Prepared &b,Output &out,bool left) {
    if (out.shape.rank!=2 || out.shape.dim[0]<32 || out.shape.dim[1]<8) return false;

    const Value *source=nullptr;
    for (int j=0;j<2;++j) {
        const auto &v=a.args[j];
        if (v.shape==out.shape && (v.stride[0]==1 || (out.order==MemoryOrder::f && v.stride[0]==-1)) && v.stride[1]>=std::ptrdiff_t(v.shape.dim[0]) && column_parameter(a.args[1-j],out.shape)) {
            source=&v;break;
        }
    }
    const auto &other=b.args[left ? 1 : 0];
    if (!source || !column_parameter(other,out.shape)) return false;
    if (__builtin_available(macOS 15.0, *)) {} else return false;
    if (a.spec->op!=Op::add && a.spec->op!=Op::subtract && a.spec->op!=Op::multiply) return false;
    if (b.spec->op!=Op::add && b.spec->op!=Op::subtract && b.spec->op!=Op::multiply && b.spec->op!=Op::divide) return false;
    const auto d=out.shape.dim[1];
    auto param=[](const Value &v,std::size_t c) {return v.shape.rank ? static_cast<const double *>(v.data)[std::ptrdiff_t(c)*v.stride[1]] : v.scalar;};
    // A zero denominator must preserve the old row-major failure prefix.
    // Admission reads only the broadcast parameters; fallback writes nothing here.
    if (b.spec->op==Op::divide) {
        if (!left) return false;
        for (std::size_t c=0;c<d;++c) if(param(other,c)==0.) return false;
    }
    return true;
}
__attribute__((noinline))
void eigen_column_pair_tiles(const Prepared &a,const Prepared &b,Output &out,bool left) {
    const auto n=out.shape.dim[0],d=out.shape.dim[1];
    const bool source_left=a.args[0].shape==out.shape && a.args[0].stride[0]==1 &&
        a.args[0].stride[1]>=std::ptrdiff_t(n);
    const Value *source=&a.args[source_left ? 0 : 1],*parameter=&a.args[source_left ? 1 : 0];
    const auto &other=b.args[left ? 1 : 0];
    // Fixed Eigen matrix tiles change storage order only; each value keeps
    // its original two arithmetic operations and operand order.
    constexpr int rows=4,cols=4;
    using Block=Eigen::Array<double,rows,cols,Eigen::ColMajor>;
    using OutputBlock=Eigen::Array<double,rows,cols,Eigen::RowMajor>;
    using Parameters=Eigen::Array<double,1,cols,Eigen::RowMajor>;
    using Pitch=Eigen::OuterStride<Eigen::Dynamic>;
    using ParamStride=Eigen::InnerStride<Eigen::Dynamic>;
    const auto *data=static_cast<const double *>(source->data);
    auto *target=static_cast<double *>(out.data);
    const auto parameter_pointer=[](const Value &v,std::size_t c) {return v.shape.rank ? static_cast<const double *>(v.data)+std::ptrdiff_t(c)*v.stride[1] : &v.scalar;};
    const auto parameter_stride=[](const Value &v) {return v.shape.rank ? v.stride[1] : 0;};
    const auto full_rows=n-n%rows,full_cols=d-d%cols;
    for(std::size_t row=0;row<full_rows;row+=16) {
        const auto end=std::min(full_rows,row+16);
        std::size_t c=0;
        for(;c<full_cols;c+=cols) {
            const Eigen::Map<const Parameters,Eigen::Unaligned,ParamStride> pv(parameter_pointer(*parameter,c),ParamStride(parameter_stride(*parameter)));
            const Eigen::Map<const Parameters,Eigen::Unaligned,ParamStride> qv(parameter_pointer(other,c),ParamStride(parameter_stride(other)));
            std::size_t i=row;
            for(;i+rows<=end;i+=rows) {
                const Eigen::Map<const Block,Eigen::Unaligned,Pitch> values(data+std::ptrdiff_t(c)*source->stride[1]+i,Pitch(source->stride[1]));
                Block middle;
                switch(a.spec->op) {
                case Op::add: if(source_left) middle=values.rowwise()+pv;else middle=pv.replicate<rows,1>()+values;break;
                case Op::subtract: if(source_left) middle=values.rowwise()-pv;else middle=pv.replicate<rows,1>()-values;break;
                case Op::multiply: if(source_left) middle=values.rowwise()*pv;else middle=pv.replicate<rows,1>()*values;break;
                default:throw Error("INVALID_POINTWISE_PAIR");
                }
                Block result;
                switch(b.spec->op) {
                case Op::add: if(left) result=middle.rowwise()+qv;else result=qv.replicate<rows,1>()+middle;break;
                case Op::subtract: if(left) result=middle.rowwise()-qv;else result=qv.replicate<rows,1>()-middle;break;
                case Op::multiply: if(left) result=middle.rowwise()*qv;else result=qv.replicate<rows,1>()*middle;break;
                case Op::divide: result=middle.rowwise()/qv;break;
                default:throw Error("INVALID_POINTWISE_PAIR");
                }
                Eigen::Map<OutputBlock,Eigen::Unaligned,Pitch>(target+i*d+c,Pitch(d))=result;
            }
        }
    }

}
__attribute__((noinline))
void column_pair_edges(const Prepared &a,const Prepared &b,Output &out,bool left) {
    const auto n=out.shape.dim[0],d=out.shape.dim[1];
    const bool source_left=a.args[0].shape==out.shape && a.args[0].stride[0]==1 && a.args[0].stride[1]>=std::ptrdiff_t(n);
    const auto &source=a.args[source_left ? 0 : 1],&parameter=a.args[source_left ? 1 : 0],&other=b.args[left ? 1 : 0];
    const auto param=[](const Value &v,std::size_t c) {return v.shape.rank ? static_cast<const double *>(v.data)[std::ptrdiff_t(c)*v.stride[1]] : v.scalar;};
    auto *target=static_cast<double *>(out.data);const auto *data=static_cast<const double *>(source.data);
    const auto full_rows=n-n%4,full_cols=d-d%4;
    for(std::size_t i=0;i<n;++i) for(std::size_t c=i<full_rows ? full_cols : 0;c<d;++c) {
        const double v=data[std::ptrdiff_t(c)*source.stride[1]+i],p=param(parameter,c),q=param(other,c);
        const double middle=source_left ? scalar_math(a.spec->op,v,p) : scalar_math(a.spec->op,p,v);
        target[i*d+c]=left ? scalar_math(b.spec->op,middle,q) : scalar_math(b.spec->op,q,middle);
    }
}
template<int RowStep>
__attribute__((noinline))
void eigen_column_pair_f(const Prepared &a,const Prepared &b,Output &out,bool left) {
    const auto n=out.shape.dim[0],d=out.shape.dim[1];
    const bool source_left=a.args[0].shape==out.shape && a.args[0].stride[0]==RowStep &&
        a.args[0].stride[1]>=std::ptrdiff_t(n);
    const Value *source=&a.args[source_left ? 0 : 1],*parameter=&a.args[source_left ? 1 : 0];
    const auto &other=b.args[left ? 1 : 0];
    auto param=[](const Value &v,std::size_t c) {return v.shape.rank ? static_cast<const double *>(v.data)[std::ptrdiff_t(c)*v.stride[1]] : v.scalar;};
    // Two Eigen packets of rows per tile, with independent F-order output.
    // Maps borrow input and evaluated blocks contain only arithmetic results.
    using Block=Eigen::Array<double,8,1>;
    auto *target=static_cast<double *>(out.data);
    const auto *data=static_cast<const double *>(source->data);
    {
        constexpr std::size_t row=0;
        const auto end=n;
        for(std::size_t c=0;c<d;++c) {
            const double p=param(*parameter,c),q=param(other,c);
            const auto *column=data+std::ptrdiff_t(c)*source->stride[1];
            std::size_t i=row;
            for(;i+8<=end;i+=8) {
                // Reverse within a fixed Eigen block, never a packed matrix.
                // Evaluation owns the reversed block; no dangling Map expression.
                const auto values = [&]() {
                    if constexpr (RowStep == 1)
                        return Eigen::Map<const Block,Eigen::Unaligned>(column+i);
                    else
                        return Eigen::Map<const Block,Eigen::Unaligned>(column-std::ptrdiff_t(i)-7).reverse().eval();
                }();
                Block middle;
                switch(a.spec->op) {
                case Op::add: if(source_left) middle=values+p;else middle=p+values;break;
                case Op::subtract: if(source_left) middle=values-p;else middle=p-values;break;
                case Op::multiply: if(source_left) middle=values*p;else middle=p*values;break;
                default:throw Error("INVALID_POINTWISE_PAIR");
                }
                Block result;
                switch(b.spec->op) {
                case Op::add: if(left) result=middle+q;else result=q+middle;break;
                case Op::subtract: if(left) result=middle-q;else result=q-middle;break;
                case Op::multiply: if(left) result=middle*q;else result=q*middle;break;
                case Op::divide: result=middle/q;break;
                default:throw Error("INVALID_POINTWISE_PAIR");
                }
                for(std::size_t j=0;j<8;++j) target[c*n+i+j]=result[j];
            }
            for(;i<end;++i) {
                const double v=column[std::ptrdiff_t(i)*RowStep];const auto middle=source_left ? scalar_math(a.spec->op,v,p) : scalar_math(a.spec->op,p,v);
                target[c*n+i]=left ? scalar_math(b.spec->op,middle,q) : scalar_math(b.spec->op,q,middle);
            }
        }
    }

}
__attribute__((noinline))
void eigen_column_pair(const Prepared &a,const Prepared &b,Output &out,bool left) {
    eigen_column_pair_tiles(a,b,out,left);
    if(out.shape.dim[0]%4 || out.shape.dim[1]%4) column_pair_edges(a,b,out,left);
}
}
#endif

namespace {
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void scalar_elementwise_pair(const Prepared &a,const Prepared &b,Output &out,bool left,bool *separated,double error) {

    switch(a.spec->op) {
    case Op::add: return pair_second<Op::add>(a,b,out,left,separated,error);
    case Op::subtract: return pair_second<Op::subtract>(a,b,out,left,separated,error);
    case Op::multiply: return pair_second<Op::multiply>(a,b,out,left,separated,error);
    default: throw Error("INVALID_POINTWISE_PAIR");
    }
}
}

void elementwise_pair(const Prepared &a,const Prepared &b,Output &out,bool left,
                      bool *separated,double error) {
    if (separated) *separated=false;
#ifdef CALMETRICS_HAVE_EIGEN_ROTATION
    if (eigen_scalar_pair_source(a,b,out,left)>=0) {
        scalar_elementwise_pair(a,b,out,left,separated,error);
        return;
    }
#endif
#if defined(CALMETRICS_HAVE_EIGEN_ROTATION) && defined(__APPLE__) && defined(__aarch64__)
    if (column_pair_eligible(a,b,out,left)) {
        if (out.order == MemoryOrder::f) {
            const bool reversed=(a.args[0].shape==out.shape && a.args[0].stride[0]==-1) ||
                (a.args[1].shape==out.shape && a.args[1].stride[0]==-1);
            if (reversed) eigen_column_pair_f<-1>(a,b,out,left);
            else eigen_column_pair_f<1>(a,b,out,left);
        }
        else eigen_column_pair(a,b,out,left);
        return;
    }
#endif
    scalar_elementwise_pair(a,b,out,left,separated,error);
}

} // namespace calmetrics_engine::ops
