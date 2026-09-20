// This translation unit alone is compiled with AVX2. No global constructors.
// The baseline dispatcher calls it only after CPU and OS state checks.
#include "simd_loop.hpp"
#include <immintrin.h>

namespace calmetrics_engine::ops {
namespace {
struct AVX2 {
    static constexpr std::size_t width = 4;
    using Vec = __m256d;
    static Vec load(const double *p) { return _mm256_loadu_pd(p); }
    static void store(double *p, Vec v) { _mm256_storeu_pd(p, v); }
    static Vec broadcast(double x) { return _mm256_set1_pd(x); }
    static Vec add(Vec a, Vec b) { return _mm256_add_pd(a, b); }
    static Vec sub(Vec a, Vec b) { return _mm256_sub_pd(a, b); }
    static Vec mul(Vec a, Vec b) { return _mm256_mul_pd(a, b); }
    static Vec div(Vec a, Vec b) { return _mm256_div_pd(a, b); }
    static Vec sqrt(Vec a) { return _mm256_sqrt_pd(a); }
    static Vec negate(Vec a) { return _mm256_xor_pd(a, _mm256_set1_pd(-0.0)); }
    static Vec absolute(Vec a) { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), a); }
    static Vec eq(Vec a, Vec b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OQ); }
    static Vec lt(Vec a, Vec b) { return _mm256_cmp_pd(a, b, _CMP_LT_OQ); }
    static Vec le(Vec a, Vec b) { return _mm256_cmp_pd(a, b, _CMP_LE_OQ); }
    static Vec invert(Vec a) {
        return _mm256_xor_pd(a, _mm256_castsi256_pd(_mm256_set1_epi32(-1)));
    }
    static Vec select(Vec m, Vec a, Vec b) { return _mm256_blendv_pd(b, a, m); }
    static void store_mask(std::uint8_t *p, Vec m) {
        const auto bits = _mm256_movemask_pd(m);
        for (unsigned i = 0; i < 4; ++i)
            p[i] = static_cast<std::uint8_t>((bits >> i) & 1);
    }
};
} // namespace
std::size_t avx2_transform(Op op, const Value &x, const Value &y, Output &out, std::size_t n) {
    return detail::vector_dispatch<AVX2>(op, x, y, out, n);
}
std::size_t avx2_matmul(const Value &a, const Value &b, Output &out) {
    return detail::matrix_loop<AVX2>(a, b, out);
}
} // namespace calmetrics_engine::ops
