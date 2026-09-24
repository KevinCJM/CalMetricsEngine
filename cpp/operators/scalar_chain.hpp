#pragma once
#include "calmetrics_engine/operators.hpp"
#include <algorithm>
#include <cmath>

namespace calmetrics_engine::ops::detail {
struct ScalarLane {
    static constexpr std::size_t width = 1;
    using Vec = double;
    static Vec load(const double *p) { return *p; }
    static void store(double *p, Vec v) { *p = v; }
    static Vec broadcast(double x) { return x; }
    static Vec add(Vec a, Vec b) { return a+b; }
    static Vec sub(Vec a, Vec b) { return a-b; }
    static Vec mul(Vec a, Vec b) { return a*b; }
    static Vec reverse(Vec a) { return a; }
    static double horizontal_max(Vec a) { return a; }
    static Vec absolute(Vec a) { return std::abs(a); }
    static bool eq(Vec a, Vec b) { return a==b; }
    static bool lt(Vec a, Vec b) { return a<b; }
    static bool le(Vec a, Vec b) { return a<=b; }
    static bool invert(bool a) { return !a; }
    static Vec select(bool m, Vec a, Vec b) { return m ? a : b; }
    static void store_mask(std::uint8_t *p, bool v) { *p = v; }
};

// Four independent hardware vectors expose enough stores and comparisons to
// pack boolean results without extracting every lane into a scalar register.
template <class V> struct Interleaved4 {
    static constexpr std::size_t width = 4*V::width;
    struct Vec { std::array<typename V::Vec,4> parts; };
    using LaneMask = decltype(V::eq(V::broadcast(0),V::broadcast(0)));
    struct Mask { std::array<LaneMask,4> parts; };
    static Vec load(const double *p) { Vec v; for (int i=0;i<4;++i) v.parts[i]=V::load(p+i*V::width); return v; }
    static void store(double *p,Vec v) { for (int i=0;i<4;++i) V::store(p+i*V::width,v.parts[i]); }
    static Vec broadcast(double x) { Vec v; for (auto &p:v.parts) p=V::broadcast(x); return v; }
    static Vec reverse(Vec a) { Vec v; for (int i=0;i<4;++i) v.parts[i]=V::reverse(a.parts[3-i]); return v; }
    static double horizontal_max(Vec a) {
        auto v = a.parts[0];
        for (int i=1;i<4;++i) v=V::select(V::lt(v,a.parts[i]),a.parts[i],v);
        return V::horizontal_max(v);
    }
#define CME_CHAIN_BINARY(name) static Vec name(Vec a,Vec b) { Vec v; for (int i=0;i<4;++i) v.parts[i]=V::name(a.parts[i],b.parts[i]); return v; }
    CME_CHAIN_BINARY(add) CME_CHAIN_BINARY(sub) CME_CHAIN_BINARY(mul)
#undef CME_CHAIN_BINARY
    static Vec absolute(Vec a) { for (auto &p:a.parts) p=V::absolute(p); return a; }
#define CME_CHAIN_COMPARE(name) static Mask name(Vec a,Vec b) { Mask v; for (int i=0;i<4;++i) v.parts[i]=V::name(a.parts[i],b.parts[i]); return v; }
    CME_CHAIN_COMPARE(eq) CME_CHAIN_COMPARE(lt) CME_CHAIN_COMPARE(le)
#undef CME_CHAIN_COMPARE
    static Mask invert(Mask a) { for (auto &p:a.parts) p=V::invert(p); return a; }
    static Vec select(Mask m,Vec a,Vec b) { Vec v; for (int i=0;i<4;++i) v.parts[i]=V::select(m.parts[i],a.parts[i],b.parts[i]); return v; }
    static void store_mask(std::uint8_t *p,Mask m) { V::store_four_masks(p,m.parts[0],m.parts[1],m.parts[2],m.parts[3]); }
};

template <Op op, class V> auto chain_math(typename V::Vec value, typename V::Vec constant, bool reverse) {
    const auto lhs = reverse ? constant : value, rhs = reverse ? value : constant;
    if constexpr (op == Op::add) return V::add(lhs, rhs);
    else if constexpr (op == Op::subtract) return V::sub(lhs, rhs);
    else if constexpr (op == Op::multiply) return V::mul(lhs, rhs);
    else return value;
}
template <Op op, class V> auto chain_predicate(typename V::Vec value, typename V::Vec constant, bool reverse) {
    const auto lhs = reverse ? constant : value, rhs = reverse ? value : constant;
    if constexpr (op == Op::equal) return V::eq(lhs,rhs);
    else if constexpr (op == Op::not_equal) return V::invert(V::eq(lhs,rhs));
    else if constexpr (op == Op::less_than) return V::lt(lhs,rhs);
    else if constexpr (op == Op::less_equal) return V::le(lhs,rhs);
    else if constexpr (op == Op::greater_than) return V::lt(rhs,lhs);
    else if constexpr (op == Op::greater_equal) return V::le(rhs,lhs);
    else return V::lt(V::absolute(value),V::broadcast(std::numeric_limits<double>::infinity()));
}

// Each instantiation fixes the operation sequence, not coefficients or shape.
// Intermediate values stay in registers; the public graph remains unchanged.
template <Op first, Op second, Op predicate, class V>
std::size_t chain_loop(const Value &input, std::size_t offset, std::size_t count, const ScalarChain &chain) {
    const auto c0 = V::broadcast(chain.constants[0]), c1 = V::broadcast(chain.constants[1]);
    const auto threshold = V::broadcast(chain.threshold);
    const auto zero = V::broadcast(0), one = V::broadcast(1);
    const auto infinity = V::broadcast(std::numeric_limits<double>::infinity());
    auto maximum = zero, invalid = zero;
    const auto reverse0 = chain.constant_first[0], reverse1 = chain.constant_first[1], reversep = chain.threshold_first;
    auto *out0 = chain.first_output, *out1 = chain.value_output;
    auto *mask = chain.mask_output;
    const auto *previous = chain.previous;
    const bool contiguous = input.contiguous();
    const auto stride = contiguous ? 1 : input.stride[input.shape.rank-1];
    auto run = input.shape.dim[input.shape.rank-1];
    if (!contiguous) {
        // Coalesce adjacent signed-stride axes without packing. Reversed inner
        // planes are one linear run, even when the outer axis runs forward.
        for (int axis = input.shape.rank-2; axis >= 0; --axis) {
            if (input.shape.dim[axis] == 1) continue;
            if (!run || run > static_cast<std::size_t>(PTRDIFF_MAX)) break;
            const auto length = static_cast<std::ptrdiff_t>(run);
            if (stride > PTRDIFF_MAX/length || stride < PTRDIFF_MIN/length ||
                input.stride[axis] != stride*length) break;
            run *= input.shape.dim[axis];
        }
    }
    std::size_t vector_count = 0;
    for (std::size_t begin = 0; begin < count;) {
        const auto length = contiguous ? count-begin : std::min(count-begin,
            run-(offset+begin)%run);
        const auto *source = static_cast<const double *>(input.data) + input.offset(offset+begin);
        const auto stop = length-length%V::width;
        for (std::size_t i = 0; i < stop; i += V::width) {
            typename V::Vec x;
            if (stride == 1) x = V::load(source+i);
            else if (stride == 0) x = V::broadcast(*source);
            else if (stride == -1) x = V::reverse(V::load(source-static_cast<std::ptrdiff_t>(i+V::width-1)));
            else {
                double lanes[V::width];
                for (std::size_t lane = 0; lane < V::width; ++lane)
                    lanes[lane] = source[static_cast<std::ptrdiff_t>(i+lane)*stride];
                x = V::load(lanes);
            }
            const auto a = chain_math<first,V>(x,c0,reverse0);
            const auto b = chain_math<second,V>(a,c1,reverse1);
            // Scan before stores, including a future proven-safe in-place use.
            if (previous) {
                const auto old = V::load(previous+begin+i);
                const auto delta = V::absolute(V::sub(b,old));
                maximum = V::select(V::lt(maximum,delta),delta,maximum);
                invalid = V::add(invalid,V::select(V::lt(V::absolute(b),infinity),zero,one));
            }
            if (out0) V::store(out0+begin+i,a);
            if (out1) V::store(out1+begin+i,b);
            if constexpr (predicate != Op(0)) V::store_mask(mask+begin+i,chain_predicate<predicate,V>(b,threshold,reversep));
        }
        if constexpr (V::width > 1) vector_count += stop;
        for (std::size_t i = stop; i < length; ++i) {
            const auto x = source[static_cast<std::ptrdiff_t>(i)*stride];
            const auto a = chain_math<first,ScalarLane>(x,chain.constants[0],reverse0);
            const auto b = chain_math<second,ScalarLane>(a,chain.constants[1],reverse1);
            if (previous) {
                const auto delta = std::abs(b-previous[begin+i]);
                if (delta > *chain.residual) *chain.residual = delta;
                *chain.finite = *chain.finite && std::isfinite(b);
            }
            if (out0) out0[begin+i] = a;
            if (out1) out1[begin+i] = b;
            if constexpr (predicate != Op(0)) mask[begin+i] = chain_predicate<predicate,ScalarLane>(b,chain.threshold,reversep);
        }
        begin += length;
    }
    if (previous) {
        // Both accumulators contain nonnegative, non-NaN values. Reduce in
        // registers instead of spilling each lane after every short iteration.
        const auto largest = V::horizontal_max(maximum);
        if (largest > *chain.residual) *chain.residual = largest;
        *chain.finite = *chain.finite && V::horizontal_max(invalid) == 0;
    }
    return vector_count*((first != Op(0))+(second != Op(0))+(predicate != Op(0))+(previous != nullptr));
}

template <class V, Op first, Op second> ScalarChainKernel chain_predicate_dispatch(Op predicate) {
    if (predicate == Op(0)) return chain_loop<first,second,Op(0),V>;
    switch (predicate) {
#define CME_PREDICATE(op) case Op::op: return chain_loop<first,second,Op::op,V>;
    CME_PREDICATE(equal) CME_PREDICATE(not_equal) CME_PREDICATE(less_than)
    CME_PREDICATE(less_equal) CME_PREDICATE(greater_than) CME_PREDICATE(greater_equal) CME_PREDICATE(finite_mask)
#undef CME_PREDICATE
    default: return nullptr;
    }
}
template <class V, Op first> ScalarChainKernel chain_second_dispatch(Op second, Op predicate) {
    if (second == Op(0)) return chain_predicate_dispatch<V,first,Op(0)>(predicate);
    switch (second) {
#define CME_SECOND(op) case Op::op: return chain_predicate_dispatch<V,first,Op::op>(predicate);
    CME_SECOND(add) CME_SECOND(subtract) CME_SECOND(multiply)
#undef CME_SECOND
    default: return nullptr;
    }
}
template <class V> ScalarChainKernel chain_dispatch(Op first, Op second, Op predicate) {
    if (first == Op(0)) return chain_second_dispatch<V,Op(0)>(second,predicate);
    switch (first) {
#define CME_FIRST(op) case Op::op: return chain_second_dispatch<V,Op::op>(second,predicate);
    CME_FIRST(add) CME_FIRST(subtract) CME_FIRST(multiply)
#undef CME_FIRST
    default: return nullptr;
    }
}
} // namespace calmetrics_engine::ops::detail
