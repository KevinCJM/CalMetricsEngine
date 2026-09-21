#include "calmetrics_engine/operators.hpp"
#include <cmath>

namespace calmetrics_engine::ops {
namespace {
constexpr double qnan = std::numeric_limits<double>::quiet_NaN();

void adaptive(const Prepared &p, Output &out) {
    const auto &x = p.args[0];
    const auto &alpha = p.args[1];
    const int seed = static_cast<int>(p.args[5].scalar);
    const auto minimum = static_cast<std::size_t>(p.args[6].scalar);
    const bool hold = p.args[7].scalar == 0;
    double previous = p.args[4].scalar;
    bool initialized = seed == 0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (p.args[3].u(i)) {
            previous = p.args[4].scalar;
            initialized = seed == 0;
            count = 0;
        }
        const double current = x.f(i);
        bool produced = false;
        if (std::isfinite(current)) {
            ++count;
            const bool update = p.args[2].u(i) != 0;
            if (!initialized && (seed == 1 || update)) {
                previous = current;
                initialized = true;
                produced = true;
            } else if (initialized && update) {
                // Invalid coefficients on masked/preheat/seed rows are never consumed.
                const double gain = alpha.f(i);
                require(std::isfinite(gain) && gain >= 0 && gain <= 1,
                        "INVALID_PARAMETER");
                previous += gain * (current - previous);
                require(std::isfinite(previous), "DOMAIN_ERROR");
                produced = true;
            }
        }
        out.set(i, initialized && count >= minimum && (produced || hold) ? previous : qnan);
    }
}

void second_order(const Prepared &p, Output &out) {
    const auto &x = p.args[0];
    const double b0 = p.args[1].scalar, b1 = p.args[2].scalar;
    const double a1 = p.args[3].scalar, a2 = p.args[4].scalar;
    const auto minimum = static_cast<std::size_t>(p.args[6].scalar);
    const bool bootstrap = p.args[7].scalar == 1;
    double previous_input = 0, previous = 0, older = 0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (p.args[5].u(i)) {
            previous_input = previous = older = 0;
            count = 0;
        }
        const double current_input = x.f(i);
        if (!std::isfinite(current_input)) {
            out.set(i, qnan);
            continue;
        }
        const double current = bootstrap && count < 2
            ? current_input
            : b0 * current_input + b1 * previous_input + a1 * previous + a2 * older;
        if (!std::isfinite(current)) {
            previous_input = previous = older = 0;
            count = 0;
            out.set(i, qnan);
            continue;
        }
        older = previous;
        previous = current;
        previous_input = current_input;
        ++count;
        out.set(i, count >= minimum ? current : qnan);
    }
}

void kalman(const Prepared &p, Output &out) {
    const auto &x = p.args[0];
    const double process = p.args[1].scalar, measurement = p.args[2].scalar;
    double estimate = 0, variance = p.args[3].scalar;
    bool initialized = false;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double current = x.f(i);
        if (!std::isfinite(current)) {
            out.set(2 * i, qnan);
            out.set(2 * i + 1, qnan);
            continue;
        }
        if (!initialized) {
            estimate = current;
            initialized = true;
        } else {
            variance += process;
            const double denominator = variance + measurement;
            require(std::isfinite(denominator), "DOMAIN_ERROR");
            const double gain = variance / denominator;
            estimate += gain * (current - estimate);
            variance *= 1.0 - gain;
            require(std::isfinite(estimate) && std::isfinite(variance), "DOMAIN_ERROR");
        }
        out.set(2 * i, estimate);
        out.set(2 * i + 1, variance);
    }
}
} // namespace

void recurrence(const Prepared &p, Output &out, Workspace &, Audit &) {
    switch (p.spec->op) {
    case Op::recursive_filter_adaptive:
        adaptive(p, out);
        return;
    case Op::linear_filter2:
        second_order(p, out);
        return;
    case Op::scalar_kalman:
        kalman(p, out);
        return;
    default:
        throw Error("INVALID_RECURRENCE");
    }
}
} // namespace calmetrics_engine::ops
