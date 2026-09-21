#include "calmetrics_engine/operators.hpp"
#include <cmath>
#include <iostream>
#include <numeric>

namespace ops = calmetrics_engine::ops;
namespace {
std::size_t checks = 0;
void check(bool value, const char *message) {
    ++checks;
    if (!value)
        throw std::runtime_error(message);
}
ops::Value vector(const double *data, std::size_t n, std::ptrdiff_t stride = 1) {
    ops::Value v;
    v.data = data;
    v.shape = ops::vector_shape(n);
    v.stride[0] = stride;
    return v;
}
ops::Value matrix(const double *data, std::size_t rows, std::size_t cols) {
    ops::Value v;
    v.data = data;
    v.shape = ops::matrix_shape(rows, cols);
    v.stride = {static_cast<std::ptrdiff_t>(cols), 1};
    return v;
}
ops::Value mask(const std::uint8_t *data, std::size_t n) {
    ops::Value v;
    v.kind = ops::Kind::mask;
    v.data = data;
    v.shape = ops::vector_shape(n);
    return v;
}
struct Computed {
    ops::Kind kind;
    ops::Shape shape;
    std::vector<double> values;
    std::vector<std::uint8_t> masks;
    std::vector<std::int64_t> integers;
    std::array<double, 5> record{};
    ops::Audit audit;
};
Computed run(const ops::Spec &spec, const std::array<ops::Value, 8> &args, std::size_t count,
             ops::Isa isa) {
    const auto plan = ops::prepare(spec, args.data(), count);
    Computed result;
    result.kind = plan.output_kind;
    result.shape = plan.output_shape;
    ops::Output out;
    out.kind = plan.output_kind;
    out.shape = plan.output_shape;
    if (out.kind == ops::Kind::number) {
        result.values.resize(out.shape.size());
        out.data = result.values.data();
    } else if (out.kind == ops::Kind::integer) {
        result.integers.resize(out.shape.size());
        out.data = result.integers.data();
    } else if (out.kind == ops::Kind::mask) {
        result.masks.resize(out.shape.size());
        out.data = result.masks.data();
    }
    ops::Workspace workspace;
    ops::execute(plan, out, workspace, isa, result.audit);
    result.record = out.record;
    return result;
}
void equal(double a, double b) {
    check((std::isnan(a) && std::isnan(b)) || a == b || std::abs(a - b) < 1e-12,
          "scalar/SIMD parity");
    if (a == 0 && b == 0)
        check(std::signbit(a) == std::signbit(b), "signed zero parity");
}
void equal(const Computed &a, const Computed &b) {
    check(a.kind == b.kind && a.shape == b.shape, "output type parity");
    check(a.values.size() == b.values.size() && a.masks == b.masks &&
              a.integers == b.integers, "output storage parity");
    for (std::size_t i = 0; i < a.values.size(); ++i)
        equal(a.values[i], b.values[i]);
    for (std::size_t i = 0; i < a.record.size(); ++i)
        equal(a.record[i], b.record[i]);
}
} // namespace

int main() {
    try {
        double x[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
        double y[]{0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1};
        double m[]{4.0, 1.0, 0.2, 1.0, 3.0, 0.1, 0.2, 0.1, 2.0};
        double weights[]{0.2, 0.3, 0.5};
        double draws[]{0.0, -0.2, 0.0, -0.2, -0.2, 0.0, -0.1, 0.0};
        std::uint8_t flags[]{1, 0, 1, 1, 0, 1, 0, 1};
        const auto vx = vector(x, 8), vy = vector(y, 8), vm = matrix(m, 3, 3),
                   vw = vector(weights, 3), vf = mask(flags, 8);
        ops::Value fit;
        fit.kind = ops::Kind::fit;
        fit.record = {0.2, 1.0, 0.25, 3.0, 8.0};
        ops::Value interval;
        interval.kind = ops::Kind::interval;
        interval.record = {2.0, 4.0, 5.0, 1.0, 0.0};
        const std::int64_t ids[]{1, 2, 2, 3, 4, 4, 5, 0};
        ops::Value vi;
        vi.kind = ops::Kind::integer;
        vi.shape = ops::vector_shape(8);
        vi.data = ids;
        std::size_t covered = 0;
        for (const auto &spec : ops::registry()) {
            check(static_cast<unsigned>(spec.op) == covered + 1, "stable opcode order");
            check(&ops::lookup(spec.name) == &spec, "lookup by name");
            check(&ops::lookup(static_cast<std::uint16_t>(spec.op)) == &spec, "lookup by opcode");
            for (std::size_t arity = spec.min_args; arity <= spec.max_args; ++arity)
                check(ops::parameter_names(spec, arity).size() == arity,
                      "parameter metadata arity");
            std::array<ops::Value, 8> a{vx, vy, ops::Value::number(0.5), ops::Value::number(2.0)};
            const auto op = spec.op;
            if (spec.family == ops::Family::elementwise) {
                if (op == ops::Op::clip) {
                    a[1] = ops::Value::number(0.2);
                    a[2] = ops::Value::number(0.7);
                }
                if (op == ops::Op::logical_and || op == ops::Op::logical_or ||
                    op == ops::Op::logical_not) {
                    a[0] = vf;
                    a[1] = vf;
                }
                if (op == ops::Op::where) {
                    a[0] = vf;
                    a[1] = vx;
                    a[2] = vy;
                }
            } else if (spec.family == ops::Family::reduction) {
                if (op >= ops::Op::sum_where && op <= ops::Op::quantile_where)
                    a[1] = vf;
                if (op == ops::Op::count_true || op == ops::Op::max_consecutive_true)
                    a[0] = vf;
                if (op == ops::Op::quantile)
                    a[1] = ops::Value::number(0.3);
                if (op >= ops::Op::sum_time && op <= ops::Op::max_asset)
                    a[0] = vm;
            } else if (spec.family == ops::Family::rolling)
                a[1] = ops::Value::number(3.0);
            else if (spec.family == ops::Family::matrix) {
                if (op != ops::Op::dot && op != ops::Op::outer && op != ops::Op::diag)
                    a[0] = vm;
                if (op == ops::Op::matmul)
                    a[1] = vm;
                if (op == ops::Op::matvec || op == ops::Op::solve)
                    a[1] = vw;
            } else if (spec.family == ops::Family::state) {
                if (op == ops::Op::last_drawdown_interval)
                    a[0] = vector(draws, 8);
                if (op >= ops::Op::interval_start && op <= ops::Op::interval_recovery)
                    a[0] = interval;
                if (op >= ops::Op::fit_slope && op <= ops::Op::fit_observation_count)
                    a[0] = fit;
                if (op == ops::Op::value_at)
                    a[1] = ops::Value::number(2.0);
                if (op == ops::Op::days_between) {
                    a[0] = ops::Value::number(100);
                    a[1] = ops::Value::number(120);
                }
                if (op == ops::Op::require_positive || op == ops::Op::require_nonnegative)
                    a[0] = ops::Value::number(1);
            } else if (spec.family == ops::Family::composite) {
                if (op == ops::Op::portfolio_returns) {
                    a[0] = vm;
                    a[1] = vw;
                }
                if (op == ops::Op::quadratic_form) {
                    a[0] = vw;
                    a[1] = vm;
                }
                if (op == ops::Op::annualized_return)
                    a[1] = ops::Value::number(12);
            }
            if (op == ops::Op::distinct_count) {
                a[0] = vi;
                a[1] = vf;
            } else if (op == ops::Op::gather) {
                a[1] = vi;
            } else if (op == ops::Op::recursive_filter) {
                a[1] = ops::Value::number(0.4);
                a[2] = ops::Value::number(0.0);
                a[3] = vf;
            }
            try {
                const auto scalar = run(spec, a, spec.min_args, ops::Isa::scalar);
                const auto automatic = run(spec, a, spec.min_args, ops::Isa::automatic);
                equal(scalar, automatic);
            } catch (const std::exception &error) {
                std::cerr << "Failed operator " << spec.name << ": " << error.what() << '\n';
                throw;
            }
            ++covered;
        }
        check(covered == ops::operator_count, "all canonical native operators executed");

        // New primitives have independent contracts beyond scalar/ISA equality.
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double inf = std::numeric_limits<double>::infinity();
        double shifted_source[]{2.0, 4.0, nan, 8.0, 10.0};
        std::uint8_t updates[]{1, 1, 1, 1, 1};
        std::array<ops::Value, 8> extension{
            vector(shifted_source, 5), ops::Value::number(0.4),
            ops::Value::number(0.0), mask(updates, 5),
            ops::Value::number(2.0), ops::Value::number(0.0)};
        auto filtered = run(ops::lookup("recursive_filter"), extension, 6, ops::Isa::scalar);
        const double expected_filter[]{2.0, 2.8, 2.8, 4.88, 6.928};
        for (std::size_t i = 0; i < 5; ++i)
            equal(filtered.values[i], expected_filter[i]);
        extension[1] = ops::Value::number(5);
        auto shifted = run(ops::lookup("aligned_shift"), extension, 2, ops::Isa::scalar);
        for (const auto value : shifted.values)
            check(std::isnan(value), "oversized shift preserves length with missing prefix");

        const std::int64_t exact_ids[]{INT64_MAX, INT64_MIN, 9007199254740993LL,
                                      9007199254740992LL, 9007199254740993LL};
        ops::Value exact;
        exact.kind = ops::Kind::integer;
        exact.shape = ops::vector_shape(5);
        exact.data = exact_ids;
        extension = {exact};
        auto sorted = run(ops::lookup("argsort"), extension, 1, ops::Isa::scalar);
        check(sorted.integers == std::vector<std::int64_t>({1, 3, 2, 4, 0}),
              "int64 stable order retains identifiers above float precision");
        auto unique = run(ops::lookup("distinct_count"), extension, 1, ops::Isa::scalar);
        check(unique.values == std::vector<double>({4}), "int64 exact distinct count");
        auto indices = exact;
        indices.data = sorted.integers.data();
        extension[1] = indices;
        const auto selected = run(ops::lookup("gather"), extension, 2, ops::Isa::scalar);
        check(selected.integers.front() == INT64_MIN && selected.integers.back() == INT64_MAX,
              "int64 selected payload remains exact");
        const std::int64_t invalid_indices[]{0, -1};
        indices.shape = ops::vector_shape(2);
        indices.data = invalid_indices;
        extension[1] = indices;
        const auto invalid_plan = ops::prepare(ops::lookup("gather"), extension.data(), 2);
        std::int64_t unchanged[]{91, 92};
        ops::Output invalid_out;
        invalid_out.kind = ops::Kind::integer;
        invalid_out.shape = ops::vector_shape(2);
        invalid_out.data = unchanged;
        ops::Workspace invalid_work;
        ops::Audit invalid_audit;
        bool rejected_index = false;
        try {
            ops::execute(invalid_plan, invalid_out, invalid_work, ops::Isa::scalar, invalid_audit);
        } catch (const ops::Error &e) {
            rejected_index = std::string(e.what()) == "INDEX_OUT_OF_BOUNDS";
        }
        check(rejected_index && unchanged[0] == 91 && unchanged[1] == 92,
              "invalid gather validates all indices before writing");
        double tails[]{-inf, -10.0, 0.0, 10.0, inf, nan};
        extension = {vector(tails, 6)};
        const auto cdf = run(ops::lookup("normal_cdf"), extension, 1, ops::Isa::scalar);
        check(cdf.values[0] == 0 && cdf.values[1] > 0 && cdf.values[2] == 0.5 &&
                  cdf.values[4] == 1 && std::isnan(cdf.values[5]),
              "normal CDF stable tails and missing value policy");
        extension = {vector(nullptr, 0)};
        check(run(ops::lookup("aligned_shift"), extension, 1, ops::Isa::scalar).values.empty(),
              "empty aligned shift");
        exact.shape = ops::vector_shape(0);
        exact.data = nullptr;
        extension = {exact};
        check(run(ops::lookup("argsort"), extension, 1, ops::Isa::scalar).integers.empty(),
              "empty integer order");
        check(run(ops::lookup("distinct_count"), extension, 1, ops::Isa::scalar).values[0] == 0,
              "empty integer count");

        for (const auto &spec : ops::registry())
            if (ops::simd_eligible(spec.op) && spec.family == ops::Family::elementwise) {
                for (std::size_t n = 0; n < 34; ++n) {
                    // Exact allocations expose any SIMD tail overread to ASan.
                    std::vector<double> a(n), b(n);
                    std::iota(a.begin(), a.end(), 1.0);
                    std::iota(b.begin(), b.end(), 2.0);
                    std::array<ops::Value, 8> args{vector(a.data(), n), vector(b.data(), n)};
                    const auto baseline = run(spec, args, spec.min_args, ops::Isa::scalar);
                    for (auto isa : {ops::Isa::sse2, ops::Isa::avx2, ops::Isa::neon})
                        if (ops::supports_isa(isa)) {
                            const auto result = run(spec, args, spec.min_args, isa);
                            equal(baseline, result);
                            if (n >= 4)
                                check(result.audit.vector_elements > 0,
                                      "real SIMD instructions selected");
                        }
                }
            }
        for (std::size_t rows = 0; rows < 10; ++rows)
            for (std::size_t inner = 0; inner < 8; ++inner)
                for (std::size_t cols = 0; cols < 13; ++cols) {
                    std::vector<double> left(rows * inner), right(inner * cols);
                    std::iota(left.begin(), left.end(), 1.0);
                    std::iota(right.begin(), right.end(), 0.5);
                    std::array<ops::Value, 8> args{matrix(left.data(), rows, inner),
                                                   matrix(right.data(), inner, cols)};
                    const auto baseline = run(ops::lookup("matmul"), args, 2, ops::Isa::scalar);
                    for (auto isa : {ops::Isa::sse2, ops::Isa::avx2, ops::Isa::neon})
                        if (ops::supports_isa(isa)) {
                            const auto result = run(ops::lookup("matmul"), args, 2, isa);
                            equal(baseline, result);
                            if (rows >= 4 && cols >= 8 && inner > 0)
                                check(result.audit.vector_elements > 0, "matrix SIMD selected");
                        }
                }
        std::array<ops::Value, 8> reverse{vector(x + 7, 8, -1)};
        const auto mean = run(ops::lookup("mean"), reverse, 1, ops::Isa::automatic);
        equal(mean.values[0], 0.45);
        auto args = std::array<ops::Value, 8>{vx, ops::Value::number(0.0)};
        bool rejected = false;
        try {
            run(ops::lookup("divide"), args, 2, ops::Isa::automatic);
        } catch (const ops::Error &e) {
            rejected = std::string(e.what()) == "DIVIDE_BY_ZERO";
        }
        check(rejected, "native domain errors");
        for (std::uint16_t id : {std::uint16_t(0), std::uint16_t(ops::operator_count + 1)}) {
            rejected = false;
            try {
                ops::lookup(id);
            } catch (const ops::Error &) {
                rejected = true;
            }
            check(rejected, "unknown opcodes fail closed");
        }
        std::cout << covered << " operators; " << checks << " checks passed; SIMD:";
        for (auto isa : {ops::Isa::scalar, ops::Isa::sse2, ops::Isa::avx2, ops::Isa::neon})
            if (ops::supports_isa(isa))
                std::cout << ' ' << ops::isa_name(isa);
        std::cout << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Native operator test failed: " << error.what() << '\n';
        return 1;
    }
}
