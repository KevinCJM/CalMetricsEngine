#include "calmetrics_engine/operators.hpp"
#include <cmath>

namespace calmetrics_engine::ops {
std::array<double, 5> fit_value(const Value &x, const Value &y, bool implicit_x) {
    const auto count = y.size();
    require(count >= 2, "INSUFFICIENT_SAMPLE");
    double sum_x = 0.0, sum_y = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double xi = implicit_x ? static_cast<double>(i) : x.f(i), yi = y.f(i);
        require(std::isfinite(xi) && std::isfinite(yi), "DOMAIN_ERROR");
        sum_x += xi;
        sum_y += yi;
    }
    const double n = static_cast<double>(count), mx = sum_x / n, my = sum_y / n;
    double xx = 0.0, xy = 0.0, yy = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double dx = (implicit_x ? static_cast<double>(i) : x.f(i)) - mx, dy = y.f(i) - my;
        xx += dx * dx;
        xy += dx * dy;
        yy += dy * dy;
    }
    require(xx > 0.0 && std::isfinite(xx), "DOMAIN_ERROR");
    const double slope = xy / xx, intercept = my - slope * mx;
    double residual = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const double xi = implicit_x ? static_cast<double>(i) : x.f(i);
        const double error = y.f(i) - (intercept + slope * xi);
        residual += error * error;
    }
    require(std::isfinite(slope) && std::isfinite(intercept) && std::isfinite(residual) &&
                std::isfinite(yy),
            "DOMAIN_ERROR");
    return {slope, intercept, residual, yy, n};
}
void regression(const Prepared &p, Output &out, Workspace &, Audit &) {
    const auto op = p.spec->op;
    const auto fit = fit_value(p.args[0], p.args[p.count == 1 ? 0 : 1], p.count == 1);
    switch (op) {
    case Op::linear_fit:
        out.record = fit;
        return;
    case Op::linear_slope:
        out.set(0, fit[0]);
        return;
    case Op::linear_intercept:
        out.set(0, fit[1]);
        return;
    case Op::linear_r_squared:
        require(std::isfinite(fit[3]) && fit[3] > 0, "DOMAIN_ERROR");
        out.set(0, 1.0 - fit[2] / fit[3]);
        return;
    case Op::regression_standard_error:
        require(fit[4] - 2.0 > 0, "DOMAIN_ERROR");
        out.set(0, std::sqrt(fit[2] / (fit[4] - 2.0)));
        return;
    default:
        throw Error("INVALID_REGRESSION");
    }
}
namespace {
constexpr double qnan = std::numeric_limits<double>::quiet_NaN();
std::array<double, 5> drawdown_interval(const Value &x) {
    const std::array<double, 5> invalid{qnan, qnan, qnan, -1.0, 0.0};
    if (x.size() == 0 || x.f(0) != 0.0)
        return invalid;
    double deepest = 0.0, latest_peak = 0.0, peak = qnan, trough = qnan, recovery = qnan;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double value = x.f(i);
        if (!std::isfinite(value) || value > 0 || value < -1.0)
            return invalid;
        if (value == 0.0) {
            latest_peak = static_cast<double>(i);
            if (std::isfinite(trough) && !std::isfinite(recovery))
                recovery = static_cast<double>(i);
        } else if (value <= deepest) {
            deepest = value;
            peak = latest_peak;
            trough = static_cast<double>(i);
            recovery = qnan;
        }
    }
    return {peak, trough, recovery, deepest < 0 ? 1.0 : 0.0, 0.0};
}
} // namespace
void state(const Prepared &p, Output &out, Workspace &, Audit &) {
    const auto op = p.spec->op;
    const auto &a = p.args;
    if (op == Op::state_select) {
        for (std::size_t i = 0; i < a[0].size(); ++i) {
            if (!a[3].u(i)) {
                out.set_integer(i, -1);
                continue;
            }
            const auto &branch = a[a[0].u(i) ? 1 : 2];
            out.set_integer(i, branch.kind == Kind::integer
                ? branch.i(i) : static_cast<std::int64_t>(branch.scalar));
        }
        return;
    }
    if (op == Op::last_drawdown_interval) {
        out.record = drawdown_interval(a[0]);
        return;
    }
    if (op >= Op::interval_start && op <= Op::interval_recovery) {
        out.set(0,
                a[0].record[static_cast<unsigned>(op) - static_cast<unsigned>(Op::interval_start)]);
        return;
    }
    if (op >= Op::fit_slope && op <= Op::fit_observation_count) {
        out.set(0, a[0].record[static_cast<unsigned>(op) - static_cast<unsigned>(Op::fit_slope)]);
        return;
    }
    if (op == Op::value_at) {
        const double pos = a[1].scalar;
        if (!std::isfinite(pos) || pos < 0 || pos != std::floor(pos) ||
            pos >= static_cast<double>(a[0].size())) {
            out.set(0, qnan);
            return;
        }
        const double value = a[0].f(static_cast<std::size_t>(pos));
        out.set(0, std::isfinite(value) ? value : qnan);
        return;
    }
    const double x = a[0].scalar;
    if (op == Op::days_between) {
        const double y = a[1].scalar;
        out.set(0, !std::isfinite(x) || !std::isfinite(y) || x != std::floor(x) ||
                           y != std::floor(y) || y < x
                       ? qnan
                       : y - x);
        return;
    }
    require(op == Op::require_positive || op == Op::require_nonnegative, "INVALID_STATE_OPERATOR");
    require(std::isfinite(x) && (op == Op::require_positive ? x > 0 : x >= 0), "DOMAIN_ERROR");
    out.set(0, x);
}
} // namespace calmetrics_engine::ops
