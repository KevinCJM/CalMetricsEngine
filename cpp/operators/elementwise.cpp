#include "calmetrics_engine/operators.hpp"
#include <algorithm>
#include <cmath>

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
} // namespace

double scalar_math(Op op, double x, double y, double z) {
    switch (op) {
    case Op::add:
        return x + y;
    case Op::subtract:
        return x - y;
    case Op::multiply:
        return x * y;
    case Op::divide:
        require(y != 0.0, "DIVIDE_BY_ZERO");
        return x / y;
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
    case Op::reciprocal:
        require(x != 0.0, "DIVIDE_BY_ZERO");
        return 1.0 / x;
    case Op::sign:
        return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
    case Op::normal_pdf:
        return std::exp(-0.5 * x * x) / std::sqrt(2.0 * pi);
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
    if (op == Op::clip)
        require(!(a[1].scalar > a[2].scalar), "INVALID_PARAMETER");
    std::size_t first = 0;
    if (simd_eligible(op) && a[0].contiguous() && a[1].contiguous() && out.shape.rank > 0) {
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
} // namespace calmetrics_engine::ops
