#include "calmetrics_engine/operators.hpp"

#include <algorithm>
#include <cmath>

namespace calmetrics_engine::ops {
namespace {
constexpr std::int64_t unknown_state = -1;
constexpr std::int64_t missing_event = -2;
constexpr double qnan = std::numeric_limits<double>::quiet_NaN();

void segment_geometry(const Value &v, std::size_t n) {
    require(v.kind == Kind::integer, "DTYPE_MISMATCH");
    require(v.shape.rank == 2 && v.shape.dim[0] == n && v.shape.dim[1] == 2,
            "SHAPE_MISMATCH");
    std::size_t t = 0;
    while (t < n) {
        const auto left = v.i(t * 2), right = v.i(t * 2 + 1);
        if (left == -1 && right == -1) { ++t; continue; }
        require(left == static_cast<std::int64_t>(t) && right > left &&
                    right < static_cast<std::int64_t>(n), "INVALID_SEGMENT_BOUNDARY");
        for (std::size_t j = t; j < static_cast<std::size_t>(right); ++j)
            require(v.i(j * 2) == left && v.i(j * 2 + 1) == right,
                    "INVALID_SEGMENT_BOUNDARY");
        t = static_cast<std::size_t>(right);
    }
}
void event_codes(const Value &v) {
    for (std::size_t i = 0; i < v.size(); ++i)
        require(v.i(i) >= missing_event && v.i(i) <= 1, "INVALID_EVENT_CODE");
}
bool valid_price(double x) { return std::isfinite(x) && x > 0.0; }
void erase_turn(std::size_t *positions, std::size_t &count, std::size_t at) {
    for (std::size_t j = at; j + 1 < count; ++j)
        positions[j] = positions[j + 1];
    --count;
}
std::size_t alternate(const Value &prices, const Value &events,
                      std::size_t *positions, std::size_t count) {
    std::size_t kept = 0;
    for (std::size_t j = 0; j < count; ++j) {
        const auto t = positions[j];
        const auto kind = events.i(t);
        if (kept && kind == events.i(positions[kept - 1])) {
            if (kind * (prices.f(t) - prices.f(positions[kept - 1])) > 0)
                positions[kept - 1] = t;
        } else if (!kept || kind * (prices.f(t) - prices.f(positions[kept - 1])) > 0) {
            positions[kept++] = t;
        }
    }
    return kept;
}

void hysteresis(const Prepared &p, Output &out) {
    std::int64_t active = 1;
    const auto &a = p.args;
    for (std::size_t i = 0; i < a[0].size(); ++i) {
        const double value = a[0].f(i);
        if (!std::isfinite(value)) {
            out.set_integer(i, unknown_state);
            continue;
        }
        if (active == 0) {
            if (value <= a[3].scalar) active = 2;
            else if (value <= a[2].scalar) active = 1;
        } else if (active == 2) {
            if (value >= a[1].scalar) active = 0;
            else if (value >= a[4].scalar) active = 1;
        } else if (value >= a[1].scalar) active = 0;
        else if (value <= a[3].scalar) active = 2;
        out.set_integer(i, active);
    }
}

void confirm(const Prepared &p, Output &out) {
    const auto &a = p.args;
    const auto confirmation = static_cast<std::size_t>(a[1].scalar);
    const auto min_hold = static_cast<std::size_t>(a[2].scalar);
    std::int64_t active = -1, candidate = -1;
    std::size_t duration = 0, count = 0;
    for (std::size_t t = 0; t < a[0].size(); ++t) {
        if (active >= 0) ++duration;
        const auto observed = a[0].i(t);
        if (observed < 0) {
            candidate = -1;
            count = 0;
            out.set_integer(t, unknown_state);
            continue;
        }
        if (active < 0) {
            count = observed == candidate ? count + 1 : 1;
            candidate = observed;
            if (count >= confirmation) {
                active = candidate;
                duration = 1;
            }
        } else if (observed == active) {
            candidate = -1;
            count = 0;
        } else {
            count = observed == candidate ? count + 1 : 1;
            candidate = observed;
            if (count >= confirmation && duration > min_hold) {
                active = candidate;
                duration = 1;
                candidate = -1;
                count = 0;
            }
        }
        out.set_integer(t, active);
    }
}

void continuous(const Prepared &p, Output &out) {
    const auto &a = p.args;
    const auto confirmation = static_cast<std::int64_t>(a[3].scalar);
    std::int64_t active = -1, proposed = -1, count = 0;
    bool confirmed = false;
    for (std::size_t t = 0; t < a[0].size(); ++t) {
        const auto candidate = a[0].i(t);
        if (t == 0) active = a[1].i(t);
        std::int64_t evidence = confirmed ? 2 : 0;
        if (candidate < 0) {
            proposed = -1;
            count = 0;
        } else if (candidate == active) {
            proposed = -1;
            count = 0;
            confirmed = true;
            evidence = 1;
        } else {
            count = candidate == proposed ? count + 1 : 1;
            proposed = candidate;
            if (count >= confirmation) {
                active = proposed;
                proposed = -1;
                count = 0;
                confirmed = true;
                evidence = 1;
            } else evidence = 3;
        }
        out.set_integer(t * 3, active);
        out.set_integer(t * 3 + 1, evidence);
        out.set_integer(t * 3 + 2, count);
    }
}

void drawdown_cycle(const Prepared &p, Output &out) {
    const auto &a = p.args;
    const double stress = a[3].scalar, rebound = a[4].scalar, exit = a[5].scalar;
    std::int64_t active = 0;
    double trough = qnan;
    for (std::size_t t = 0; t < a[0].size(); ++t) {
        const double price = a[0].f(t), drawdown = a[1].f(t);
        if (!a[2].u(t) || !valid_price(price) || !std::isfinite(drawdown)) {
            active = -1;
            trough = qnan;
            out.set_integer(t, unknown_state);
            continue;
        }
        if (active < 0) {
            if (drawdown <= -stress) { active = 2; trough = price; }
            else if (drawdown >= -exit) { active = 0; trough = price; }
        } else if (active == 0) {
            if (drawdown <= -stress) { active = 2; trough = price; }
        } else if (active == 2) {
            if (!std::isfinite(trough) || price < trough) trough = price;
            else if (price / trough - 1.0 >= rebound) active = 1;
        } else {
            if (price < trough) { trough = price; active = 2; }
            else if (drawdown >= -exit) { active = 0; trough = price; }
        }
        out.set_integer(t, active);
    }
}

void extrema(const Prepared &p, Output &out) {
    const auto &prices = p.args[0];
    const auto n = prices.size();
    const auto left = static_cast<std::size_t>(p.args[1].scalar);
    const auto right = static_cast<std::size_t>(p.args[2].scalar);
    const auto head = std::max(left, static_cast<std::size_t>(p.args[3].scalar));
    const auto tail = std::max(right, static_cast<std::size_t>(p.args[4].scalar));
    for (std::size_t i = 0; i < n; ++i) out.set_integer(i, 0);
    std::size_t start = 0;
    while (start < n) {
        if (!valid_price(prices.f(start))) {
            out.set_integer(start++, missing_event);
            continue;
        }
        std::size_t stop = start + 1;
        while (stop < n && valid_price(prices.f(stop))) ++stop;
        std::size_t previous = n;
        std::int64_t previous_kind = 0;
        if (head < stop - start && tail < stop - start - head) {
            for (std::size_t t = start + head; t < stop - tail; ++t) {
                bool peak = true, trough = true;
                for (std::size_t j = t - left; j < t; ++j) {
                    peak = peak && prices.f(t) > prices.f(j);
                    trough = trough && prices.f(t) < prices.f(j);
                }
                for (std::size_t j = t + 1; j <= t + right; ++j) {
                    peak = peak && prices.f(t) >= prices.f(j);
                    trough = trough && prices.f(t) <= prices.f(j);
                }
                if (!peak && !trough) continue;
                const std::int64_t kind = peak ? 1 : -1;
                if (previous != n) {
                    if (kind * (prices.f(t) - prices.f(previous)) <= 0) continue;
                    if (previous_kind == kind) out.set_integer(previous, 0);
                }
                out.set_integer(t, kind);
                previous = t;
                previous_kind = kind;
            }
        }
        start = stop;
    }
}

void ps_filter(const Prepared &p, Output &out, Workspace &work) {
    const auto &prices = p.args[0], &events = p.args[1];
    const auto n = prices.size();
    const auto min_phase = static_cast<std::size_t>(p.args[2].scalar);
    const auto min_cycle = static_cast<std::size_t>(p.args[3].scalar);
    const double amplitude = p.args[4].scalar;
    auto *positions = work.indices.data();
    for (std::size_t i = 0; i < n; ++i) out.set_integer(i, 0);
    std::size_t start = 0;
    while (start < n) {
        if (!valid_price(prices.f(start)) || events.i(start) == missing_event) {
            out.set_integer(start++, missing_event);
            continue;
        }
        std::size_t stop = start + 1;
        while (stop < n && valid_price(prices.f(stop)) && events.i(stop) != missing_event) ++stop;
        std::size_t count = 0;
        for (std::size_t t = start; t < stop; ++t)
            if (events.i(t) == 1 || events.i(t) == -1) positions[count++] = t;
        count = alternate(prices, events, positions, count);
        bool changed = true;
        // Every changed iteration deletes at least one event; never add/reinsert a turn.
        while (changed && count) {
            changed = false;
            const auto first = positions[0], last = positions[count - 1];
            for (std::size_t j = start; j < first; ++j) {
                if (events.i(first) * (prices.f(j) - prices.f(first)) > 0) {
                    erase_turn(positions, count, 0);
                    changed = true;
                    break;
                }
            }
            if (changed) continue;
            for (std::size_t j = last + 1; j < stop; ++j) {
                if (events.i(last) * (prices.f(j) - prices.f(last)) > 0) {
                    erase_turn(positions, count, count - 1);
                    changed = true;
                    break;
                }
            }
            if (changed) continue;
            for (std::size_t j = 0; j + 2 < count; ++j) {
                if (positions[j + 2] - positions[j] < min_cycle) {
                    const bool later_stronger = events.i(positions[j]) *
                        (prices.f(positions[j + 2]) - prices.f(positions[j])) > 0;
                    const auto at = later_stronger ? j : j + 1;
                    erase_turn(positions, count, at);
                    erase_turn(positions, count, at);
                    changed = true;
                    break;
                }
            }
            if (changed) {
                count = alternate(prices, events, positions, count);
                continue;
            }
            for (std::size_t j = 0; j + 1 < count; ++j) {
                const auto left = positions[j], right = positions[j + 1];
                const double movement = std::abs(prices.f(right) / prices.f(left) - 1.0);
                if (right - left >= min_phase || movement > amplitude) continue;
                std::size_t at = j;
                if (j + 1 == count - 1) at = j + 1;
                else if (j > 0) {
                    const double before = std::abs(std::log(prices.f(left)) -
                                                   std::log(prices.f(positions[j - 1])));
                    const double after = std::abs(std::log(prices.f(positions[j + 2])) -
                                                  std::log(prices.f(right)));
                    if (before >= after) at = j + 1;
                }
                erase_turn(positions, count, at);
                count = alternate(prices, events, positions, count);
                changed = true;
                break;
            }
        }
        for (std::size_t j = 0; j < count; ++j)
            out.set_integer(positions[j], events.i(positions[j]));
        start = stop;
    }
}

void phase_direction(const Prepared &p, Output &out) {
    const auto &events = p.args[0], &segments = p.args[1];
    for (std::size_t t = 0; t < events.size(); ++t) {
        const auto left = segments.i(t * 2), right = segments.i(t * 2 + 1);
        std::int64_t phase = -1;
        if (left >= 0) {
            if (events.i(static_cast<std::size_t>(left)) == -1 &&
                events.i(static_cast<std::size_t>(right)) == 1) phase = 0;
            else if (events.i(static_cast<std::size_t>(left)) == 1 &&
                     events.i(static_cast<std::size_t>(right)) == -1) phase = 1;
        }
        out.set_integer(t, phase);
    }
}

void reference_cycle(const Prepared &p, Output &out) {
    const auto &phases = p.args[0], &changes = p.args[1], &segments = p.args[2];
    const auto n = phases.size();
    const double stress = p.args[3].scalar;
    for (std::size_t i = 0; i < n; ++i) out.set_integer(i, -1);
    auto *result = static_cast<std::int64_t *>(out.data);
    bool prior_stress = false;
    std::size_t t = 0;
    while (t < n) {
        const auto left = segments.i(t * 2), right = segments.i(t * 2 + 1);
        if (left != static_cast<std::int64_t>(t) || right <= left) {
            ++t;
            continue;
        }
        const auto phase = phases.i(t);
        const double change = changes.f(t);
        if ((phase != 0 && phase != 1) || !std::isfinite(change)) {
            prior_stress = false;
            t = static_cast<std::size_t>(right);
            continue;
        }
        if (result[t] < 0) out.set_integer(t, 0);
        for (std::size_t j = t + 1; j <= static_cast<std::size_t>(right); ++j)
            out.set_integer(j, 0);
        if (phase == 1) {
            prior_stress = change <= -stress;
            if (prior_stress)
                for (std::size_t j = t + 1; j <= static_cast<std::size_t>(right); ++j)
                    out.set_integer(j, 2);
        } else {
            if (prior_stress) {
                out.set_integer(t, 2);
                for (std::size_t j = t + 1; j <= static_cast<std::size_t>(right); ++j)
                    out.set_integer(j, 1);
            }
            prior_stress = false;
        }
        t = static_cast<std::size_t>(right);
    }
}

void between(const Prepared &p, Output &out) {
    const auto &events = p.args[0];
    const auto n = events.size();
    for (std::size_t i = 0; i < n * 2; ++i) out.set_integer(i, -1);
    std::size_t previous = n;
    for (std::size_t t = 0; t < n; ++t) {
        const auto kind = events.i(t);
        if (kind == missing_event) previous = n;
        else if (kind == 1 || kind == -1) {
            if (previous != n && kind != events.i(previous)) {
                for (std::size_t j = previous; j < t; ++j) {
                    out.set_integer(j * 2, static_cast<std::int64_t>(previous));
                    out.set_integer(j * 2 + 1, static_cast<std::int64_t>(t));
                }
            }
            previous = t;
        }
    }
}
} // namespace

void prepare_state_events(Prepared &p) {
    const auto op = p.spec->op;
    const auto &a = p.args;
    p.output_kind = Kind::integer;
    const auto vector_arg = [&](std::size_t i, Kind kind) {
        require(a[i].kind == kind, "DTYPE_MISMATCH");
        if (p.geometry(i)) require(a[i].shape.rank == 1, "RANK_MISMATCH");
    };
    const auto equal = [&](std::size_t i, std::size_t j) {
        if (p.geometry(i) && p.geometry(j)) require(a[i].shape == a[j].shape, "SHAPE_MISMATCH");
    };
    const auto configuration = [&](std::size_t i, bool valid) {
        if (!p.payload(i)) return false;
        if (valid) return true;
        if (!p.structure_only) throw Error("INVALID_PARAMETER");
        p.payload_available_mask &= static_cast<std::uint8_t>(~(1u << i));
        return false;
    };
    const auto parameter = [&](std::size_t i) {
        require(a[i].kind == Kind::number, "DTYPE_MISMATCH");
        if (p.geometry(i)) require(a[i].shape.rank == 0, "DTYPE_MISMATCH");
        return p.payload(i) && configuration(i, std::isfinite(a[i].scalar));
    };
    const auto integer_parameter = [&](std::size_t i, std::size_t lo, std::size_t hi) {
        return parameter(i) && configuration(i, a[i].scalar == std::floor(a[i].scalar) &&
            a[i].scalar >= static_cast<double>(lo) && a[i].scalar <= static_cast<double>(hi));
    };
    const auto segments = [&](std::size_t i) {
        require(a[i].kind == Kind::integer, "DTYPE_MISMATCH");
        if (p.geometry(i)) {
            require(a[i].shape.rank == 2 && a[i].shape.dim[1] == 2, "SHAPE_MISMATCH");
            if (p.geometry(0)) require(a[i].shape.dim[0] == a[0].size(), "SHAPE_MISMATCH");
        }
        if (p.payload(i)) segment_geometry(a[i], a[i].shape.dim[0]);
    };
    if (op == Op::continuous_state_values || op == Op::continuous_state_evidence ||
        op == Op::continuous_state_pending || op == Op::segment_starts || op == Op::segment_ends) {
        const bool segment = op == Op::segment_starts || op == Op::segment_ends;
        require(a[0].kind == Kind::integer, "DTYPE_MISMATCH");
        p.output_geometry_known = p.geometry(0);
        if (p.geometry(0)) {
            require(a[0].shape.rank == 2 && a[0].shape.dim[1] == (segment ? 2u : 3u), "SHAPE_MISMATCH");
            p.output_shape = vector_shape(a[0].shape.dim[0]);
            if (p.payload(0)) {
                std::size_t column = 0;
                if (op == Op::continuous_state_evidence || op == Op::segment_ends) column = 1;
                else if (op == Op::continuous_state_pending) column = 2;
                p.view = a[0].column(column);
                p.borrowed = true;
            }
        }
        return;
    }
    const bool integer_input = op == Op::state_confirm || op == Op::state_continuous || op == Op::between_events || op == Op::phase_direction || op == Op::drawdown_cycle_reference;
    vector_arg(0, integer_input ? Kind::integer : Kind::number);
    p.output_geometry_known = p.geometry(0);
    if (p.geometry(0)) p.output_shape = a[0].shape;
    if (op == Op::state_hysteresis) {
        for (std::size_t i = 1; i < 5; ++i) parameter(i);
    } else if (op == Op::state_confirm) {
        integer_parameter(1, 1, static_cast<std::size_t>(INT32_MAX));
        integer_parameter(2, 1, static_cast<std::size_t>(INT32_MAX));
        if (p.payload(0)) for (std::size_t i = 0; i < a[0].size(); ++i) require(a[0].i(i) >= -1, "INVALID_STATE_CODE");
    } else if (op == Op::state_continuous) {
        vector_arg(1, Kind::integer); vector_arg(2, Kind::number);
        equal(0, 1); equal(0, 2); equal(1, 2);
        integer_parameter(3, 1, 252);
        const auto has_states = integer_parameter(4, 2, 12);
        if (p.payload(2)) for (std::size_t i = 0; i < a[2].size(); ++i)
            require(valid_price(a[2].f(i)), "CONTINUOUS_STATE_OBSERVATION_MISSING");
        for (std::size_t j = 0; j < 2; ++j) if (p.payload(j))
            for (std::size_t i = 0; i < a[j].size(); ++i)
                require(a[j].i(i) >= -1 && (!has_states || a[j].i(i) < static_cast<std::int64_t>(a[4].scalar)), "CONTINUOUS_STATE_CODE_INVALID");
        if (p.payload(1)) require(!a[1].size() || a[1].i(0) >= 0, "CONTINUOUS_STATE_INITIAL_REQUIRED");
        if (p.geometry(0)) p.output_shape = matrix_shape(a[0].size(), 3);
    } else if (op == Op::drawdown_cycle_state) {
        vector_arg(1, Kind::number); vector_arg(2, Kind::mask);
        equal(0, 1); equal(0, 2); equal(1, 2);
        const auto stress = parameter(3), rebound = parameter(4), exit = parameter(5);
        if (stress) configuration(3, a[3].scalar > 0 && a[3].scalar < 1);
        if (rebound) configuration(4, a[4].scalar > 0 && a[4].scalar < 1);
        if (exit) configuration(5, a[5].scalar >= 0);
        if (p.payload(3) && p.payload(5)) configuration(5, a[5].scalar < a[3].scalar);
        if (p.payload(2)) for (std::size_t i = 0; i < a[2].size(); ++i) {
            require(a[2].u(i) <= 1, "INVALID_MASK");
            if (p.payload(1) && a[2].u(i) && std::isfinite(a[1].f(i)))
                require(a[1].f(i) >= -1.0 && a[1].f(i) <= 0.0, "INVALID_DRAWDOWN");
        }
    } else if (op == Op::local_extrema) {
        integer_parameter(1, 1, 5000); integer_parameter(2, 1, 5000);
        integer_parameter(3, 0, 5000); integer_parameter(4, 0, 5000);
    } else if (op == Op::ps_filter) {
        vector_arg(1, Kind::integer); equal(0, 1);
        if (p.payload(1)) event_codes(a[1]);
        integer_parameter(2, 1, 10000); integer_parameter(3, 2, 20000);
        if (parameter(4)) configuration(4, a[4].scalar >= 0);
        if (p.geometry(0)) p.scratch_indices = a[0].size();
    } else if (op == Op::between_events) {
        if (p.payload(0)) event_codes(a[0]);
        if (p.geometry(0)) p.output_shape = matrix_shape(a[0].size(), 2);
    } else if (op == Op::phase_direction) {
        if (p.payload(0)) event_codes(a[0]);
        segments(1);
    } else if (op == Op::drawdown_cycle_reference) {
        vector_arg(1, Kind::number); equal(0, 1); segments(2);
        if (p.geometry(1) && p.geometry(2)) require(a[1].size() == a[2].shape.dim[0], "SHAPE_MISMATCH");
        if (parameter(3)) configuration(3, a[3].scalar > 0 && a[3].scalar < 1);
        if (p.payload(0)) for (std::size_t i = 0; i < a[0].size(); ++i)
            require(a[0].i(i) >= -1 && a[0].i(i) <= 1, "INVALID_PHASE_CODE");
    } else throw Error("INVALID_STATE_EVENT_OPERATOR");
}

void state_events(const Prepared &p, Output &out, Workspace &work, Audit &) {
    switch (p.spec->op) {
    case Op::state_hysteresis: hysteresis(p, out); return;
    case Op::state_confirm: confirm(p, out); return;
    case Op::state_continuous: continuous(p, out); return;
    case Op::drawdown_cycle_state: drawdown_cycle(p, out); return;
    case Op::local_extrema: extrema(p, out); return;
    case Op::ps_filter: ps_filter(p, out, work); return;
    case Op::between_events: between(p, out); return;
    case Op::phase_direction: phase_direction(p, out); return;
    case Op::drawdown_cycle_reference: reference_cycle(p, out); return;
    default: throw Error("INVALID_STATE_EVENT_OPERATOR");
    }
}
} // namespace calmetrics_engine::ops
