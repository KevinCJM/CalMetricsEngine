#include "calmetrics_engine/operators.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace calmetrics_engine::ops {
namespace {
constexpr double qnan = std::numeric_limits<double>::quiet_NaN();
void rolling_moments(const Prepared &p, Output &out) {
    const auto &x = p.args[0];
    const auto width = static_cast<std::size_t>(p.args[1].scalar);
    const bool deviation = p.spec->op == Op::rolling_std;
    const auto ddof = deviation ? static_cast<std::size_t>(p.args[2].scalar) : 0;
    const auto minimum = static_cast<std::size_t>(p.args[deviation ? 3 : 2].scalar);
    double sum = 0.0, squares = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double current = x.f(i);
        if (std::isfinite(current)) {
            sum += current;
            if (deviation)
                squares += current * current;
            ++count;
        }
        if (i >= width) {
            const double expired = x.f(i - width);
            if (std::isfinite(expired)) {
                sum -= expired;
                if (deviation)
                    squares -= expired * expired;
                --count;
            }
        }
        double result = qnan;
        if (count >= minimum && count > ddof) {
            if (!deviation)
                result = sum / static_cast<double>(count);
            else {
                double centered = squares - sum * sum / static_cast<double>(count);
                if (centered < 0.0 && centered > -1e-12)
                    centered = 0.0;
                if (centered >= 0.0)
                    result = std::sqrt(centered / static_cast<double>(count - ddof));
            }
        }
        out.set(i, result);
    }
}
void rolling_extreme(const Prepared &p, Output &out, Workspace &work) {
    const auto &x = p.args[0];
    const auto width = static_cast<std::size_t>(p.args[1].scalar);
    const auto minimum = static_cast<std::size_t>(p.args[2].scalar);
    const bool maximum = p.spec->op == Op::rolling_max;
    auto *queue = work.indices.data();
    const auto capacity = work.indices.size();
    std::size_t head = 0, used = 0, count = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (i >= width) {
            const auto expired = i - width;
            if (std::isfinite(x.f(expired)))
                --count;
            while (used && queue[head] <= expired) {
                head = (head + 1) % capacity;
                --used;
            }
        }
        const double current = x.f(i);
        if (std::isfinite(current)) {
            ++count;
            while (used) {
                const auto last = (head + used - 1) % capacity;
                const double previous = x.f(queue[last]);
                if (!(maximum ? previous <= current : previous >= current))
                    break;
                --used;
            }
            queue[(head + used) % capacity] = i;
            ++used;
        }
        out.set(i, count >= minimum && used ? x.f(queue[head]) : qnan);
    }
}
void smooth(const Prepared &p, Output &out) {
    const auto &x = p.args[0];
    const double width = p.args[1].scalar;
    double previous = p.args[2].scalar;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double current = x.f(i);
        if (!std::isfinite(current)) {
            out.set(i, qnan);
            continue;
        }
        previous = ((width - 1.0) * previous + current) / width;
        out.set(i, previous);
    }
}
void recursive_filter(const Prepared &p, Output &out) {
    const auto &x = p.args[0];
    const double alpha = p.args[1].scalar;
    const int seed = static_cast<int>(p.args[4].scalar);
    const bool emit_nan = p.args[5].scalar == 1;
    bool initialized = seed != 2;
    double previous = p.args[2].scalar;
    for (std::size_t i = 0; i < x.size(); ++i) {
        // Explicit first-row seeding is independent of the update predicate.
        if (seed == 1 && i == 0) {
            out.set(i, previous);
            continue;
        }
        const double current = x.f(i);
        if (!p.args[3].u(i) || !std::isfinite(current)) {
            out.set(i, initialized && !emit_nan ? previous : qnan);
            continue;
        }
        if (!initialized) {
            previous = current;
            initialized = true;
        } else {
            previous = (1.0 - alpha) * previous + alpha * current;
        }
        out.set(i, previous);
    }
}
} // namespace

// Shared product primitive: an optional add-one input transform and prefix output.
// It is used by product/cumulative_product and their return compositions.
double product_value(const Value &x, bool add_one, Output *prefix) {
    require(x.size() > 0, "INSUFFICIENT_SAMPLE");
    double running = 1.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        running *= add_one ? x.f(i) + 1.0 : x.f(i);
        if (prefix)
            prefix->set(i, add_one ? running - 1.0 : running);
    }
    return running;
}

void sequence(const Prepared &p, Output &out, Workspace &work, Audit &) {
    const auto op = p.spec->op;
    const auto &x = p.args[0];
    const auto n = x.size();
    if (op == Op::aligned_shift) {
        const auto periods = static_cast<std::size_t>(p.args[1].scalar);
        for (std::size_t i = 0; i < n; ++i)
            out.set(i, i < periods ? p.args[2].scalar : x.f(i - periods));
        return;
    }
    if (op == Op::recursive_filter) {
        recursive_filter(p, out);
        return;
    }
    if (op == Op::argsort) {
        auto &indices = work.indices;
        std::iota(indices.begin(), indices.end(), std::size_t{0});
        std::sort(indices.begin(), indices.end(), [&](std::size_t left, std::size_t right) {
            if (x.kind == Kind::integer) {
                const auto lhs = x.i(left), rhs = x.i(right);
                return lhs == rhs ? left < right : lhs < rhs;
            }
            const double lhs = x.f(left), rhs = x.f(right);
            const bool left_nan = std::isnan(lhs), right_nan = std::isnan(rhs);
            if (left_nan != right_nan)
                return !left_nan;
            if (left_nan || lhs == rhs)
                return left < right;
            return lhs < rhs;
        });
        for (std::size_t i = 0; i < n; ++i)
            out.set_integer(i, static_cast<std::int64_t>(indices[i]));
        return;
    }
    if (op == Op::gather) {
        const auto &indices = p.args[1];
        // Validate every index before writing any output.
        for (std::size_t i = 0; i < indices.size(); ++i)
            require(indices.i(i) >= 0 && static_cast<std::uint64_t>(indices.i(i)) < n,
                    "INDEX_OUT_OF_BOUNDS");
        for (std::size_t i = 0; i < indices.size(); ++i) {
            const auto index = static_cast<std::size_t>(indices.i(i));
            if (x.kind == Kind::integer)
                out.set_integer(i, x.i(index));
            else if (x.kind == Kind::mask)
                out.set_mask(i, x.u(index));
            else
                out.set(i, x.f(index));
        }
        return;
    }
    if (op == Op::rolling_mean || op == Op::rolling_std) {
        rolling_moments(p, out);
        return;
    }
    if (op == Op::rolling_min || op == Op::rolling_max) {
        rolling_extreme(p, out, work);
        return;
    }
    if (op == Op::recursive_smooth) {
        smooth(p, out);
        return;
    }
    if (op == Op::length) {
        out.set(0, static_cast<double>(n));
        return;
    }
    require(n > 0, "INSUFFICIENT_SAMPLE");
    if (op == Op::first || op == Op::last) {
        out.set(0, x.f(op == Op::first ? 0 : n - 1));
        return;
    }
    if (op == Op::difference) {
        const auto periods = static_cast<std::size_t>(p.args[1].scalar);
        for (std::size_t i = 0; i < n - periods; ++i)
            out.set(i, x.f(i + periods) - x.f(i));
        return;
    }
    if (op == Op::cumulative_product) {
        product_value(x, false, &out);
        return;
    }
    if (op == Op::drawdown_series || op == Op::new_high_mask) {
        for (std::size_t i = 0; i < n; ++i)
            require(std::isfinite(x.f(i)) && x.f(i) > 0, "DOMAIN_ERROR");
        double peak = x.f(0);
        if (op == Op::drawdown_series)
            out.set(0, 0.0);
        else
            out.set_mask(0, 1);
        for (std::size_t i = 1; i < n; ++i) {
            const double value = x.f(i);
            const bool high = value > peak;
            if (high)
                peak = value;
            if (op == Op::drawdown_series)
                out.set(i, value / peak - 1.0);
            else
                out.set_mask(i, high);
        }
        return;
    }
    double running = op == Op::cumulative_sum ? 0.0 : x.f(0);
    for (std::size_t i = 0; i < n; ++i) {
        if (op == Op::cumulative_sum)
            running += x.f(i);
        else if (op == Op::cumulative_max)
            running = std::max(running, x.f(i));
        else if (op == Op::cumulative_min)
            running = std::min(running, x.f(i));
        else
            throw Error("INVALID_SEQUENCE");
        out.set(i, running);
    }
}

void composite(const Prepared &p, Output &out, Workspace &work, Isa isa, Audit &audit) {
    const auto op = p.spec->op;
    const auto &a = p.args;
    if (op == Op::active_returns) {
        Prepared primitive = p;
        primitive.spec = &lookup("subtract");
        elementwise(primitive, out, work, isa, audit);
        return;
    }
    if (op == Op::portfolio_returns) {
        matvec_value(a[0], a[1], static_cast<double *>(out.data));
        return;
    }
    if (op == Op::quadratic_form) {
        Value transposed = a[1];
        std::swap(transposed.shape.dim[0], transposed.shape.dim[1]);
        std::swap(transposed.stride[0], transposed.stride[1]);
        matvec_value(transposed, a[0], work.doubles.data());
        Value projected;
        projected.shape = a[0].shape;
        projected.data = work.doubles.data();
        out.set(0, dot_value(projected, a[0]));
        return;
    }
    const double product = product_value(a[0], true, op == Op::cumulative_return ? &out : nullptr);
    if (op == Op::cumulative_return)
        return;
    const double total = product - 1.0;
    if (op == Op::total_return) {
        out.set(0, total);
        return;
    }
    require(op == Op::annualized_return, "INVALID_COMPOSITE");
    const double annual = a[1].scalar, base = total + 1.0;
    require(std::isfinite(annual) && annual > 0 && std::isfinite(base) && base >= 0,
            "DOMAIN_ERROR");
    out.set(0, scalar_math(Op::power, base, annual / static_cast<double>(a[0].size())) - 1.0);
}
} // namespace calmetrics_engine::ops
