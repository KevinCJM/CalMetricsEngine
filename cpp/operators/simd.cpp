#include "calmetrics_engine/operators.hpp"
#include "simd_loop.hpp"

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define CME_NEON 1
#elif defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#define CME_SSE2 1
#ifdef _MSC_VER
#include <intrin.h>
#endif
#endif

namespace calmetrics_engine::ops {
#ifdef CALMETRICS_HAVE_AVX2
std::size_t avx2_transform(Op, const Value &, const Value &, Output &, std::size_t);
std::size_t avx2_matmul(const Value &, const Value &, Output &);
#endif
namespace {
#ifdef CME_SSE2
struct SSE2 {
    static constexpr std::size_t width = 2;
    using Vec = __m128d;
    static Vec load(const double *p) { return _mm_loadu_pd(p); }
    static void store(double *p, Vec v) { _mm_storeu_pd(p, v); }
    static Vec broadcast(double x) { return _mm_set1_pd(x); }
    static Vec add(Vec a, Vec b) { return _mm_add_pd(a, b); }
    static Vec sub(Vec a, Vec b) { return _mm_sub_pd(a, b); }
    static Vec mul(Vec a, Vec b) { return _mm_mul_pd(a, b); }
    static Vec div(Vec a, Vec b) { return _mm_div_pd(a, b); }
    static Vec sqrt(Vec a) { return _mm_sqrt_pd(a); }
    static Vec negate(Vec a) { return _mm_xor_pd(a, _mm_set1_pd(-0.0)); }
    static Vec absolute(Vec a) { return _mm_andnot_pd(_mm_set1_pd(-0.0), a); }
    static Vec eq(Vec a, Vec b) { return _mm_cmpeq_pd(a, b); }
    static Vec lt(Vec a, Vec b) { return _mm_cmplt_pd(a, b); }
    static Vec le(Vec a, Vec b) { return _mm_cmple_pd(a, b); }
    static Vec invert(Vec a) { return _mm_xor_pd(a, _mm_castsi128_pd(_mm_set1_epi32(-1))); }
    static Vec select(Vec m, Vec a, Vec b) {
        return _mm_or_pd(_mm_and_pd(m, a), _mm_andnot_pd(m, b));
    }
    static void store_mask(std::uint8_t *p, Vec m) {
        const auto bits = _mm_movemask_pd(m);
        p[0] = bits & 1;
        p[1] = (bits >> 1) & 1;
    }
};
#endif
#ifdef CME_NEON
struct NEON {
    static constexpr std::size_t width = 2;
    using Vec = float64x2_t;
    using Mask = uint64x2_t;
    static Vec load(const double *p) { return vld1q_f64(p); }
    static void store(double *p, Vec v) { vst1q_f64(p, v); }
    static Vec broadcast(double x) { return vdupq_n_f64(x); }
    static Vec add(Vec a, Vec b) { return vaddq_f64(a, b); }
    static Vec sub(Vec a, Vec b) { return vsubq_f64(a, b); }
    static Vec mul(Vec a, Vec b) { return vmulq_f64(a, b); }
    static Vec div(Vec a, Vec b) { return vdivq_f64(a, b); }
    static Vec sqrt(Vec a) { return vsqrtq_f64(a); }
    static Vec negate(Vec a) { return vnegq_f64(a); }
    static Vec absolute(Vec a) { return vabsq_f64(a); }
    static Mask eq(Vec a, Vec b) { return vceqq_f64(a, b); }
    static Mask lt(Vec a, Vec b) { return vcltq_f64(a, b); }
    static Mask le(Vec a, Vec b) { return vcleq_f64(a, b); }
    static Mask invert(Mask a) { return veorq_u64(a, vdupq_n_u64(~std::uint64_t(0))); }
    static Vec select(Mask m, Vec a, Vec b) { return vbslq_f64(m, a, b); }
    static void store_mask(std::uint8_t *p, Mask m) {
        p[0] = static_cast<std::uint8_t>(vgetq_lane_u64(m, 0) != 0);
        p[1] = static_cast<std::uint8_t>(vgetq_lane_u64(m, 1) != 0);
    }
};
#endif
bool avx2_available() {
#if defined(CALMETRICS_HAVE_AVX2) && defined(CME_SSE2)
#ifdef _MSC_VER
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7)
        return false;
    __cpuid(info, 1);
    if ((info[2] & (1 << 27)) == 0 || (info[2] & (1 << 28)) == 0)
        return false;
    if ((_xgetbv(0) & 6) != 6)
        return false; // OS must save XMM/YMM state.
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
#else
    return __builtin_cpu_supports("avx2");
#endif
#else
    return false;
#endif
}
} // namespace

bool supports_isa(Isa isa) {
    switch (isa) {
    case Isa::automatic:
    case Isa::scalar:
        return true;
    case Isa::avx2: {
        static const bool available = avx2_available();
        return available;
    }
#ifdef CME_SSE2
    case Isa::sse2:
        return true;
#else
    case Isa::sse2:
        return false;
#endif
#ifdef CME_NEON
    case Isa::neon:
        return true;
#else
    case Isa::neon:
        return false;
#endif
    }
    return false;
}
const char *isa_name(Isa isa) {
    switch (isa) {
    case Isa::automatic:
        return "auto";
    case Isa::scalar:
        return "scalar";
    case Isa::sse2:
        return "sse2";
    case Isa::avx2:
        return "avx2";
    case Isa::neon:
        return "neon";
    }
    return "unknown";
}
Isa parse_isa(std::string_view name) {
    for (auto isa : {Isa::automatic, Isa::scalar, Isa::sse2, Isa::avx2, Isa::neon})
        if (name == isa_name(isa))
            return isa;
    throw Error("INVALID_SIMD_POLICY");
}
Isa selected_isa(Isa requested) {
    require(supports_isa(requested), "UNSUPPORTED_ISA");
    if (requested != Isa::automatic)
        return requested;
    for (auto isa : {Isa::avx2, Isa::neon, Isa::sse2})
        if (supports_isa(isa))
            return isa;
    return Isa::scalar;
}
bool simd_eligible(Op op) {
    return (op >= Op::add && op <= Op::divide) || (op >= Op::minimum && op <= Op::sqrt) ||
           op == Op::reciprocal || (op >= Op::equal && op <= Op::greater_equal) ||
           op == Op::finite_mask || op == Op::matmul;
}
std::size_t simd_transform(Op op, const Value &x, const Value &y, Output &out, std::size_t count,
                           Isa requested, Audit &audit) {
    const auto isa = selected_isa(requested);
    std::size_t processed = 0;
#ifdef CME_NEON
    if (isa == Isa::neon)
        processed = detail::vector_dispatch<NEON>(op, x, y, out, count);
#endif
#ifdef CME_SSE2
    if (isa == Isa::sse2)
        processed = detail::vector_dispatch<SSE2>(op, x, y, out, count);
#endif
#ifdef CALMETRICS_HAVE_AVX2
    if (isa == Isa::avx2)
        processed = avx2_transform(op, x, y, out, count);
#endif
    if (processed) {
        audit.isa = isa_name(isa);
        audit.vector_elements += processed;
    }
    return processed;
}
std::size_t simd_matmul(const Value &a, const Value &b, Output &out, Isa requested, Audit &audit) {
    const auto isa = selected_isa(requested);
    std::size_t processed = 0;
#ifdef CME_NEON
    if (isa == Isa::neon)
        processed = detail::matrix_loop<NEON>(a, b, out);
#endif
#ifdef CME_SSE2
    if (isa == Isa::sse2)
        processed = detail::matrix_loop<SSE2>(a, b, out);
#endif
#ifdef CALMETRICS_HAVE_AVX2
    if (isa == Isa::avx2)
        processed = avx2_matmul(a, b, out);
#endif
    if (processed) {
        audit.isa = isa_name(isa);
        audit.vector_elements += processed;
    }
    return processed;
}
} // namespace calmetrics_engine::ops
