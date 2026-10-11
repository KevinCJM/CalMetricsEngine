#include "calmetrics_engine/operators.hpp"
#include "calmetrics_engine/matrix_backend.hpp"
#include "axis_path_native_cases.hpp"
#include "numeric_line_sort_native_cases.hpp"
#include "matmul_input_bounds_native_cases.hpp"
#include "packed_f_native_cases.hpp"
#include "f_cache_native_cases.hpp"
#include "mask_validation_native_cases.hpp"
#include "partitioned_matmul_native_cases.hpp"
#include "hmm_checkpoint_native_cases.hpp"
#include <algorithm>
#include <cmath>
#include <cfenv>
#include <cstring>
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
ops::Value integer_vector(const std::int64_t *data, std::size_t n) {
    ops::Value v;
    v.kind = ops::Kind::integer;
    v.data = data;
    v.shape = ops::vector_shape(n);
    return v;
}
ops::Value integer_matrix(const std::int64_t *data, std::size_t rows, std::size_t cols) {
    auto v = integer_vector(data, rows * cols);
    v.shape = ops::matrix_shape(rows, cols);
    v.stride = {static_cast<std::ptrdiff_t>(cols), 1};
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
void order_selection_contracts() {
    std::fenv_t environment;
    check(std::fegetenv(&environment)==0,"cannot save order selection floating environment");
    const int original_rounding=std::fegetround(),original_flags=std::fetestexcept(FE_ALL_EXCEPT);
    struct RestoreEnvironment {
        const std::fenv_t *saved;
        ~RestoreEnvironment() { std::fesetenv(saved); }
    } restore{&environment};
    for (std::size_t n : {1024u,1025u}) {
        std::vector<double> input(2*n+2,std::numeric_limits<double>::quiet_NaN());
        std::vector<std::uint8_t> selected(2*n+2,0);
        input.front()=input.back()=-713.;
        for (std::size_t i=0;i<n;++i) {
            input[1+2*i]=double(static_cast<std::ptrdiff_t>((i*37)%n)-std::ptrdiff_t(n/2))/16.;
            selected[1+2*i]=1;
        }
        input[1]=std::numeric_limits<double>::denorm_min();
        input[3]=std::numeric_limits<double>::min();
        input[5]=std::nextafter(1.0,2.0);
        input[7]=std::numeric_limits<double>::max();
        input[9]=-std::numeric_limits<double>::max();
        const auto before=input;
        const auto mask_before=selected;
        for (bool masked : {false,true}) for (double q : {
                 -1.0,std::nextafter(0.0,1.0),0.95,std::nextafter(1.0,0.0)}) {
            const bool quantile=q>=0.0;
            const char *name=masked ? (quantile ? "quantile_where" : "median_where")
                                    : (quantile ? "quantile" : "median");
            std::array<ops::Value,8> args{};
            args[0]=vector(input.data()+1,masked ? 2*n : n,masked ? 1 : 2);
            std::size_t count=1;
            if (masked) args[count++]=mask(selected.data()+1,2*n);
            if (quantile) args[count++]=ops::Value::number(q);
            const auto plan=ops::prepare(ops::lookup(name),args.data(),count);
            ops::Workspace workspace;
            for (int rounding : {FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
                check(std::fesetround(rounding)==0,"cannot set order selection rounding");
                double reference=0.0;
                int reference_flags=0;
                for (auto isa : {ops::Isa::scalar,ops::Isa::automatic}) {
                    check(std::feclearexcept(FE_ALL_EXCEPT)==0,"cannot clear order selection flags");
                    check(std::feraiseexcept(FE_DIVBYZERO)==0,"cannot seed order selection flags");
                    std::array<double,3> storage{-919.,0.,-919.};
                    ops::Output output; output.kind=plan.output_kind;
                    output.shape=plan.output_shape; output.data=storage.data()+1;
                    ops::Audit audit;
                    ops::execute(plan,output,workspace,isa,audit);
                    const int flags=std::fetestexcept(FE_ALL_EXCEPT);
                    check(std::fegetround()==rounding,"order selection changed rounding mode");
                    if (isa==ops::Isa::scalar) {
                        reference=storage[1];
                        reference_flags=flags;
                        check(std::strcmp(audit.isa,"highway_vqselect")!=0,
                              "explicit scalar order selection used Highway");
                    } else check(flags==reference_flags && (flags&FE_DIVBYZERO) &&
                                 std::memcmp(&reference,&storage[1],sizeof(double))==0,
                                 "order selection changed result bits");
                    check(storage.front()==-919. && storage.back()==-919.,
                          "order selection crossed output boundary");
                    check(audit.algorithm_copy_bytes==n*sizeof(double) &&
                          audit.scratch_bytes==n*sizeof(double),
                          "order selection ignored actual mask consumption");
                    check(std::memcmp(input.data(),before.data(),input.size()*sizeof(double))==0 &&
                          selected==mask_before,"order selection mutated input or canaries");
                }
            }
        }
    }
    check(std::fesetenv(&environment)==0 && std::fegetround()==original_rounding &&
          std::fetestexcept(FE_ALL_EXCEPT)==original_flags,
          "order selection test did not restore rounding and flags");
}
void density_state_contracts() {
    constexpr std::size_t n=4096,d=32,k=8;
    std::vector<double> x(n*d),m(k*d),v(k*d,1.0);
    for (std::size_t i=0;i<x.size();++i) x[i]=std::sin(double(i)*0.13);
    for (std::size_t i=0;i<m.size();++i) m[i]=double(static_cast<int>(i%19)-9)/32.;
    auto center=matrix(m.data()+(k-1)*d,k,d);
    center.stride[0]=-static_cast<std::ptrdiff_t>(d);
    const std::array<ops::Value,8> args{matrix(x.data(),n,d),center,matrix(v.data(),k,d)};
    const auto reference=run(ops::lookup("gaussian_log_density"),args,3,ops::Isa::scalar);
    const auto actual=run(ops::lookup("gaussian_log_density"),args,3,ops::Isa::automatic);
    check(std::memcmp(reference.values.data(),actual.values.data(),n*k*sizeof(double))==0,
          "component lanes changed density bits");
    check(reference.audit.eigen_density_calls==0 && actual.audit.eigen_density_calls<=1,
          "completed density call audit");
    check(reference.audit.scratch_bytes==actual.audit.scratch_bytes,
          "component lanes added numeric workspace");
    const auto saved_rounding=std::fegetround();
    for (auto rounding : {FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
        check(std::fesetround(rounding)==0,"density rounding control unavailable");
        const auto scalar=run(ops::lookup("gaussian_log_density"),args,3,ops::Isa::scalar);
        const auto automatic=run(ops::lookup("gaussian_log_density"),args,3,ops::Isa::automatic);
        check(std::memcmp(scalar.values.data(),automatic.values.data(),n*k*sizeof(double))==0,
              "component lanes changed directed-rounding bits");
    }
    check(std::fesetround(saved_rounding)==0,"density rounding mode not restored");
    // Arithmetic failure must write exactly the original logical prefix.
    for (auto row : {0u,63u,4095u}) for (auto component : {0u,3u,4u,7u}) {
        std::fill(x.begin(),x.end(),0.0);
        std::fill(m.begin(),m.end(),0.0);
        std::fill(v.begin(),v.end(),1e308);
        x[row*d]=1e154; v[component*d]=1e-12;
        const std::array<ops::Value,8> failure_args{matrix(x.data(),n,d),matrix(m.data(),k,d),matrix(v.data(),k,d)};
        const auto prepared=ops::prepare(ops::lookup("gaussian_log_density"),failure_args.data(),3);
        std::vector<double> scalar(n*k+2,771.),automatic(scalar);
        for (auto isa : {ops::Isa::scalar,ops::Isa::automatic}) {
            auto &storage=isa==ops::Isa::scalar ? scalar : automatic;
            ops::Output output; output.kind=prepared.output_kind;
            output.shape=prepared.output_shape; output.data=storage.data()+1;
            ops::Workspace workspace; ops::Audit audit;
            bool failed=false;
            try {ops::execute(prepared,output,workspace,isa,audit);}
            catch (const std::exception &e) {failed=std::string(e.what()).find("NON_FINITE_RESULT")!=std::string::npos;}
            check(failed,"density arithmetic overflow accepted");
            check(storage.front()==771. && storage.back()==771.,"density crossed result boundary");
        }
        check(std::memcmp(scalar.data(),automatic.data(),scalar.size()*sizeof(double))==0,
              "component lanes changed failure prefix");
    }
}
void hmm_workspace_tile_canaries() {
    constexpr std::size_t guard = 16;
    constexpr double scratch_canary = -917.0, output_canary = -719.0;
    const auto &spec = ops::lookup("gaussian_hmm_em");
    for (std::size_t n : {128u, 129u, 257u, 258u})
        for (std::size_t k : {3u, 12u}) {
            // One feature isolates XI tiling from existing feature-library paths.
            std::vector<double> x(n), initial(k, 1.0 / double(k)),
                transition(k * k), means(k), variances(k, 4.0);
            for (std::size_t i = 0; i < n; ++i)
                x[i] = (double(i % 17) - 8.0) * 0.125 + double(i % k) * 2.0;
            for (std::size_t s = 0; s < k; ++s) {
                means[s] = x[s];
                for (std::size_t t = 0; t < k; ++t)
                    transition[s * k + t] = s == t ? 0.9 : 0.1 / double(k - 1);
            }
            const auto before_x = x, before_initial = initial,
                before_transition = transition, before_means = means,
                before_variances = variances;
            const std::array<ops::Value, 8> args{
                matrix(x.data(), n, 1), vector(initial.data(), k),
                matrix(transition.data(), k, k), matrix(means.data(), k, 1),
                matrix(variances.data(), k, 1), ops::Value::number(3),
                ops::Value::number(0)};
            const auto plan = ops::prepare(spec, args.data(), 7);
            const auto needed = 3 * n * k + 2 * k * k + 4 * k +
                std::min(n - 1, std::size_t{128});
            check(plan.scratch_doubles == needed && plan.scratch_indices == 0,
                  "HMM tile scratch geometry");
            const auto reference = run(spec, args, 7, ops::Isa::scalar);
            for (auto isa : {ops::Isa::scalar, ops::Isa::automatic}) {
                ops::Workspace work;
                // Keep all guard doubles alive: reading merely reserved vector
                // capacity after execute() shrinks its size would be invalid.
                // Test the actual kernel directly; normal entry/admission is
                // covered separately by the public operator and sanitizer tests.
                work.ensure(needed + guard, 0);
                std::fill(work.doubles.begin(), work.doubles.end(), scratch_canary);
                const auto *address = work.doubles.data();
                const auto capacity = work.capacity_bytes();
                std::vector<double> storage(plan.output_shape.size() + 2, output_canary);
                ops::Output out;
                out.kind = plan.output_kind;
                out.shape = plan.output_shape;
                out.data = storage.data() + 1;
                ops::Audit audit;
                ops::latent_execute(plan, out, work, isa, audit);
                check(work.doubles.size() == needed + guard &&
                          work.doubles.data() == address && work.capacity_bytes() == capacity,
                      "HMM kernel resized its prepared workspace");
                check(std::all_of(work.doubles.begin() + needed, work.doubles.end(),
                                  [&](double value) { return value == scratch_canary; }),
                      "HMM tile crossed its scratch boundary");
                check(storage.front() == output_canary && storage.back() == output_canary,
                      "HMM tile crossed its result boundary");
                check(std::memcmp(storage.data() + 1, reference.values.data(),
                                  reference.values.size() * sizeof(double)) == 0,
                      "HMM tile changed scalar model bits");
            }
            check(x == before_x && initial == before_initial &&
                      transition == before_transition && means == before_means &&
                      variances == before_variances,
                  "HMM tile changed its input arrays");
        }
}
} // namespace

int main() {
    try {
        matmul_input_bounds_native_cases::run();
        packed_f_native_cases::run();
        f_cache_native_cases::run();
    mask_validation_native_cases::run();
        partitioned_matmul_native_cases::run();
        test_axis_path_contracts();
        test_numeric_line_sort_contracts();
        density_state_contracts();
        order_selection_contracts();
        hmm_workspace_tile_canaries();
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
        const double prices[]{100, 120, 105, 130, 90, 125, 110, 115};
        const double changes[]{0.2, -0.125, 130.0 / 105 - 1, 90.0 / 130 - 1,
                               125.0 / 90 - 1, 110.0 / 125 - 1, 0, 0};
        const std::int64_t events[]{-1, 1, -1, 1, -1, 1, -1, 0};
        const std::uint8_t no_reset[]{0, 0, 0, 0, 0, 0, 0, 0};
        const std::uint8_t all_valid[]{1, 1, 1, 1, 1, 1, 1, 1};
        const auto vp = vector(prices, 8), ve = integer_vector(events, 8),
                   vr = mask(no_reset, 8), valid = mask(all_valid, 8);

        // Projection inputs are real typed outputs, held alive for the full registry sweep.
        const auto kalman_fixture = run(ops::lookup("scalar_kalman"),
            {vx, ops::Value::number(0.01), ops::Value::number(0.1)}, 3, ops::Isa::scalar);
        const auto vk = matrix(kalman_fixture.values.data(), 8, 2);
        const auto continuous_fixture = run(ops::lookup("state_continuous"),
            {vi, vi, vp, ops::Value::number(2), ops::Value::number(6)}, 5, ops::Isa::scalar);
        const auto vc = integer_matrix(continuous_fixture.integers.data(), 8, 3);
        const auto segments_fixture = run(ops::lookup("between_events"), {ve}, 1, ops::Isa::scalar);
        const auto segments = integer_matrix(segments_fixture.integers.data(), 8, 2);
        const auto phases_fixture = run(ops::lookup("phase_direction"), {ve, segments},
                                       2, ops::Isa::scalar);
        const auto phases = integer_vector(phases_fixture.integers.data(), 8);
        const auto eigen_fixture = run(ops::lookup("symmetric_eigh"), {vm}, 1, ops::Isa::scalar);
        const auto eigen = matrix(eigen_fixture.values.data(), 3, 4);
        const auto gm_fixture = run(ops::lookup("gaussian_mixture_em"),
            {vm,vw,vm,vm,ops::Value::number(4)},5,ops::Isa::scalar);
        const auto gm=matrix(gm_fixture.values.data(),3,10);
        const auto hm_fixture = run(ops::lookup("gaussian_hmm_em"),
            {vm,vw,vm,vm,vm,ops::Value::number(4)},6,ops::Isa::scalar);
        const auto hm=matrix(hm_fixture.values.data(),3,13);
        const std::int64_t permutation[]{2,0,1};
        const std::int64_t selected_rows[]{2, 0};
        const double probabilities[]{0.05, 0.5, 0.95};
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
            } else if (op == ops::Op::recursive_filter_adaptive) {
                a = {vx, vy, vf, vr};
            } else if (op == ops::Op::linear_filter2) {
                a = {vx, ops::Value::number(0.2), ops::Value::number(0.2),
                     ops::Value::number(0.7), ops::Value::number(-0.1), vr};
            } else if (op == ops::Op::scalar_kalman) {
                a = {vx, ops::Value::number(0.01), ops::Value::number(0.1)};
            } else if (op == ops::Op::state_estimate || op == ops::Op::state_variance) {
                a = {vk};
            } else if (op == ops::Op::state_hysteresis) {
                a = {vx, ops::Value::number(0.6), ops::Value::number(0.4),
                     ops::Value::number(0.2), ops::Value::number(0.3)};
            } else if (op == ops::Op::state_confirm) {
                a = {vi, ops::Value::number(2), ops::Value::number(2)};
            } else if (op == ops::Op::state_continuous) {
                a = {vi, vi, vp, ops::Value::number(2), ops::Value::number(6)};
            } else if (op == ops::Op::continuous_state_values ||
                       op == ops::Op::continuous_state_evidence ||
                       op == ops::Op::continuous_state_pending) {
                a = {vc};
            } else if (op == ops::Op::drawdown_cycle_state) {
                a = {vp, vector(draws, 8), vf, ops::Value::number(0.15),
                     ops::Value::number(0.1), ops::Value::number(0.05)};
            } else if (op == ops::Op::local_extrema) {
                a = {vp, ops::Value::number(1), ops::Value::number(1),
                     ops::Value::number(0), ops::Value::number(0)};
            } else if (op == ops::Op::ps_filter) {
                a = {vp, ve, ops::Value::number(1), ops::Value::number(2),
                     ops::Value::number(0.05)};
            } else if (op == ops::Op::between_events) {
                a = {ve};
            } else if (op == ops::Op::segment_starts || op == ops::Op::segment_ends) {
                a = {segments};
            } else if (op == ops::Op::phase_direction) {
                a = {ve, segments};
            } else if (op == ops::Op::drawdown_cycle_reference) {
                a = {phases, vector(changes, 8), segments, ops::Value::number(0.2)};
            } else if (op == ops::Op::state_select) {
                a = {vf, vi, ops::Value::number(0), valid};
            } else if (op == ops::Op::solve_multi || op == ops::Op::triangular_solve) {
                a = {vm, vm};
            } else if (op == ops::Op::eigenvalues || op == ops::Op::eigenvectors) {
                a = {eigen};
            } else if (op == ops::Op::take_axis) {
                a = {vm, integer_vector(selected_rows, 2), ops::Value::number(0)};
            } else if (op == ops::Op::slice_axis) {
                a = {vm, ops::Value::number(1), ops::Value::number(0),
                     ops::Value::number(3), ops::Value::number(2)};
            } else if (op == ops::Op::concatenate || op == ops::Op::stack) {
                a = {vm, vm, ops::Value::number(1)};
            } else if (op == ops::Op::broadcast_like) {
                a = {vw, vm};
            } else if (op == ops::Op::sum_axis || op == ops::Op::mean_axis || op == ops::Op::swap_axes) {
                a = {vm, ops::Value::number(0), ops::Value::number(1)};
            }
            if(op>=ops::Op::variance_axis && op<=ops::Op::softmax_axis)a={vm,ops::Value::number(0)};
            else if(op==ops::Op::gaussian_log_density)a={vm,vm,vm};
            else if(op==ops::Op::gaussian_mixture_em)a={vm,vw,vm,vm};
            else if(op==ops::Op::gaussian_hmm_em)a={vm,vw,vm,vm,vm};
            else if(op>=ops::Op::mixture_weights && op<=ops::Op::em_converged)a={gm};
            else if(op>=ops::Op::hmm_initial && op<=ops::Op::hmm_variances)a={hm};
            else if(op==ops::Op::hmm_filter || op==ops::Op::hmm_smooth)a={vm,vw,vm};
            else if(op==ops::Op::transition_counts || op==ops::Op::one_hot)a={vi,ops::Value::number(6)};
            else if(op==ops::Op::partition_mean)a={vm,integer_vector(permutation,3),ops::Value::number(2)};
            else if(op==ops::Op::sample_unique_indices)a={vm,ops::Value::number(2)};
            else if(op==ops::Op::scatter_rows)a={vm,integer_vector(permutation,3),ops::Value::number(5)};
            else if(op==ops::Op::affine_contract)a={vm,vm,vw};
            else if(op==ops::Op::cumulative_product_axis || op==ops::Op::cumulative_max_axis)
                a={vm,ops::Value::number(0),ops::Value::number(1)};
            else if(op==ops::Op::quantiles_axis)
                a={vm,vector(probabilities, 3),ops::Value::number(0)};
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
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j) {
                double reconstructed = 0;
                for (std::size_t k = 0; k < 3; ++k)
                    reconstructed += eigen_fixture.values[k*4] *
                        eigen_fixture.values[k*4+i+1] * eigen_fixture.values[k*4+j+1];
                check(std::abs(reconstructed-m[i*3+j]) < 1e-10, "eigen reconstruction");
            }
        // Geometry validation must still reject an independent RHS mismatch
        // when the coefficient payload has failed upstream.
        bool independent_shape_rejected = false;
        ops::Value mismatched[]{vm, matrix(nullptr, 4, 2)};
        try {
            ops::validate_structure(ops::lookup("solve_multi"), mismatched, 2, 3, 0);
        } catch (const ops::Error &e) {
            independent_shape_rejected = std::string(e.what()) == "SHAPE_MISMATCH";
        }
        check(independent_shape_rejected, "failed matrix does not suppress RHS validation");

        // Independent latent shapes cannot disappear with a failed initial
        // value or unknown means geometry (the earlier isolate regressions).
        const auto latent_shape_error = [&](const char *name,
            const std::array<ops::Value,8> &values,std::size_t count,std::uint32_t known) {
            bool rejected=false;
            try { ops::validate_structure(ops::lookup(name),values.data(),count,known,0); }
            catch(const ops::Error &e){rejected=std::string(e.what())=="SHAPE_MISMATCH";}
            check(rejected,"independent latent geometry is always validated");
        };
        latent_shape_error("gaussian_mixture_em",
            {matrix(nullptr,7,2),vector(nullptr,2),matrix(nullptr,2,2),matrix(nullptr,2,3)},4,11);
        latent_shape_error("gaussian_hmm_em",
            {matrix(nullptr,7,2),vector(nullptr,3),matrix(nullptr,2,2),matrix(nullptr,3,2),matrix(nullptr,3,2)},5,29);
        latent_shape_error("hmm_filter",
            {matrix(nullptr,7,2),vector(nullptr,2),matrix(nullptr,3,3)},3,5);

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
        for (std::size_t n : {128u,256u}) {
            std::vector<double> a(n*n,0.0);
            for (std::size_t i=0;i<n;++i) a[i*n+i]=double(n-i);
            const auto original=a;
            std::array<ops::Value,8> args{matrix(a.data(),n,n)};
            const auto plan=ops::prepare(ops::lookup("symmetric_eigh"),args.data(),1);
            check(plan.scratch_indices==n,"stable eigenpair index budget");
            const auto full=run(ops::lookup("symmetric_eigh"),args,1,ops::Isa::scalar);
            for (std::size_t i=0;i<n;++i) {
                check(full.values[i*(n+1)]==double(i+1),"descending eigenpair sort");
                for (std::size_t j=0;j<n;++j)
                    check(full.values[i*(n+1)+j+1]==double(j==n-1-i),"eigenpair row permutation");
            }
            std::vector<double> guarded(n+2,-993.0);
            ops::Output out; out.shape=ops::vector_shape(n);out.data=guarded.data()+1;
            ops::Workspace workspace;ops::Audit audit;
            ops::symmetric_eigh_values_pair(plan,out,workspace,audit);
            check(guarded.front()==-993.0 && guarded.back()==-993.0,"projection output guards");
            for (std::size_t i=0;i<n;++i)
                check(std::memcmp(guarded.data()+1+i,full.values.data()+i*(n+1),8)==0,"bitwise eigenvalue projection");
            check(a==original,"projection mutated separate input");
            // Graph liveness can reuse the dead source arena for the smaller result.
            out.data=a.data();
            ops::symmetric_eigh_values_pair(plan,out,workspace,audit);
            for (std::size_t i=0;i<n;++i)
                check(std::memcmp(a.data()+i,guarded.data()+1+i,8)==0,"projection copied aliased source before write");
        }
        for (std::size_t n : {255u,256u,257u,384u,512u}) {
            std::vector<double> a(n*n,0.0);
            for (std::size_t i=0;i<n;++i) a[i*n+i]=1.0;
            a[1]=a[n]=0.25;
            std::array<ops::Value,8> args{matrix(a.data(),n,n)};
            const auto plan=ops::prepare(ops::lookup("symmetric_eigh"),args.data(),1);
            bool packed=false;
#if defined(__APPLE__) && defined(__aarch64__)
            if (__builtin_available(macOS 15.0, *)) packed=n>=256 && n%128==0;
#endif
            check(plan.scratch_doubles==(packed ? n*(n+1)/2 : n*n),"Jacobi workspace geometry");
            const auto result=run(ops::lookup("symmetric_eigh"),args,1,ops::Isa::automatic);
            equal(result.values[0],0.75);
            equal(result.values[(n-1)*(n+1)],1.25);
            for (std::size_t column=0;column<n;++column) {
                const auto base=column*(n+1)+1;
                const auto eigenvalue=result.values[base-1];
                for (std::size_t row=0;row<n;++row) {
                    const auto v=result.values[base+row];
                    const auto av=v+(row<2 ? .25*result.values[base+1-row] : 0.0);
                    equal(av,eigenvalue*v);
                }
            }
            check(a[1]==.25 && a[n]==.25,"Jacobi mutated input");
            if (!packed) check(result.audit.eigen_rotation_calls==0,"small/odd matrix used library");
        }
        for (std::size_t d : {0u,1u,8u,9u}) {
            std::vector<double> x(65*d);
            for (std::size_t i=0;i<x.size();++i) x[i]=std::sin(double(i));
            for (bool transpose : {false,true}) {
                auto input=matrix(x.data(),65,d);
                if(transpose) input.stride={1,65,1};
                std::array<ops::Value,8> args{input};
                for (const char *name : {"covariance","correlation"}) {
                    const auto &spec=ops::lookup(name);
                    check(ops::prepare(spec,args.data(),1).scratch_doubles==0,"covariance scratch");
                    const auto baseline=run(spec,args,1,ops::Isa::scalar);
                    const auto result=run(spec,args,1,ops::Isa::automatic);
                    equal(baseline,result);
                }
            }
        }
        for (std::size_t n : {127u,128u,129u,256u}) {
            std::vector<double> a(n*n,0.0), b(n,0.5);
            for (std::size_t i=0;i<n;++i) a[i*n+i]=2.0;
            a[1]=a[n]=0.25;
            std::array<ops::Value,8> args{matrix(a.data(),n,n),vector(b.data(),n)};
            for (const char *name : {"cholesky","matvec"}) {
                const auto count=std::string(name)=="cholesky" ? 1u : 2u;
                const auto &spec=ops::lookup(name);
                const auto reference=run(spec,args,count,ops::Isa::scalar);
                const auto actual=run(spec,args,count,ops::Isa::automatic);
                equal(reference,actual);
                check(reference.audit.eigen_matvec_calls==0,"scalar used Eigen matvec");
                check(actual.audit.scratch_bytes==0,"library introduced numeric scratch");
            }
            a[(n-1)*n+n-1]=1e-20;
            const auto &chol=ops::lookup("cholesky");
            equal(run(chol,args,1,ops::Isa::scalar),run(chol,args,1,ops::Isa::automatic));
            check(a[1]==.25 && a[n]==.25,"library mutated input");
        }
        {
            const std::size_t rows=4096, cols=32;
            std::vector<double> a(rows*cols,0.5), b(cols,1.0);
            auto column_major=matrix(a.data(),rows,cols);
            column_major.stride={1,static_cast<std::ptrdiff_t>(rows),1};
            std::array<ops::Value,8> args{column_major,vector(b.data(),cols)};
            const auto &spec=ops::lookup("matvec");
            equal(run(spec,args,2,ops::Isa::scalar),run(spec,args,2,ops::Isa::automatic));
            for (std::size_t j=0;j<cols;++j) a[j*rows]=0;
            a[0]=1e16; a[rows]=1; a[2*rows]=-1e16;
            const auto result=run(spec,args,2,ops::Isa::automatic);
            equal(result.values[0],0.0);
            equal(run(spec,args,2,ops::Isa::scalar),result);
            check(result.audit.scratch_bytes==0,"Eigen matvec introduced scratch");
        }
        {
            const std::size_t n=129;
            std::vector<double> a(n*n,0.0), x(n,1.0);
            a[0]=1e16; a[1]=1.0; a[2]=-1e16; a[n]=1.0;
            a[(n-1)*n+n-1]=7.0;
            x[n-1]=2.0;
            const auto original_a=a, original_x=x;
            for (bool transpose : {false,true}) {
                auto input=matrix(a.data(),n,n);
                if (transpose) std::swap(input.stride[0],input.stride[1]);
                std::array<ops::Value,8> args{vector(x.data(),n),input};
                const auto plan=ops::prepare(ops::lookup("quadratic_form"),args.data(),2);
                check(plan.scratch_doubles==n && plan.scratch_indices==0,
                      "quadratic projection workspace geometry");
                ops::Workspace workspace;
                workspace.ensure(plan.scratch_doubles,plan.scratch_indices);
                const auto *scratch=workspace.doubles.data();
                const auto capacity=workspace.capacity_bytes();
                ops::Output reference, actual;
                reference.kind=actual.kind=plan.output_kind;
                reference.shape=actual.shape=plan.output_shape;
                ops::Audit scalar_audit, automatic_audit;
                ops::execute(plan,reference,workspace,ops::Isa::scalar,scalar_audit);
                check(workspace.doubles.data()==scratch && workspace.capacity_bytes()==capacity,
                      "scalar quadratic changed workspace capacity");
                ops::execute(plan,actual,workspace,ops::Isa::automatic,automatic_audit);
                check(std::memcmp(&reference.scalar,&actual.scalar,sizeof(double))==0,
                      "quadratic library changed accumulation order");
                // A^T x then dot gives 28; the transposed view gives 29. The
                // final nonzero component also exercises the 129-element tail.
                check(actual.scalar==(transpose ? 29.0 : 28.0),
                      "quadratic changed projection orientation or tail");
                check(scalar_audit.eigen_matvec_calls==0,"scalar quadratic used Eigen");
                check(scalar_audit.scratch_bytes==n*sizeof(double) &&
                          automatic_audit.scratch_bytes==n*sizeof(double),
                      "quadratic library introduced numeric scratch");
                check(workspace.doubles.data()==scratch && workspace.capacity_bytes()==capacity,
                      "quadratic library changed workspace capacity");
                auto projection=input;
                std::swap(projection.stride[0],projection.stride[1]);
                const std::array<ops::Value,8> projection_args{projection,vector(x.data(),n)};
                const auto projected=run(ops::lookup("matvec"),projection_args,2,
                                         ops::Isa::automatic);
                // Compare actual backend availability, including Eigen-OFF builds.
                check(automatic_audit.eigen_matvec_calls==projected.audit.eigen_matvec_calls,
                      "quadratic failed to reuse eligible matvec backend");
                check(a==original_a && x==original_x,"quadratic library mutated input");
            }
        }
        // Runtime control also works for Eigen-OFF and non-Apple native builds.
        bool matmul_library_available=false;
        {
            // Padded F right operand and negative left row stride exercise the
            // library tile plus both tails. A scaled coordinate selection is
            // an independent oracle without another matrix multiplication.
            constexpr std::size_t rows=33, inner=35, cols=37, pitch=inner+3;
            std::vector<double> a(rows*inner), b(pitch*cols,-713.0);
            for (std::size_t i=0;i<a.size();++i) a[i]=static_cast<int>(i%11)-5;
            for (std::size_t j=0;j<cols;++j) {
                std::fill_n(b.data()+j*pitch,inner,0.0);
                b[j*pitch+j%inner]=static_cast<double>(j+1);
            }
            const auto original_a=a, original_b=b;
            auto left=matrix(a.data()+(rows-1)*inner,rows,inner);
            left.stride[0]=-static_cast<std::ptrdiff_t>(inner);
            auto right=matrix(b.data(),inner,cols);
            right.stride={1,static_cast<std::ptrdiff_t>(pitch)};
            const std::array<ops::Value,8> args{left,right};
            const auto plan=ops::prepare(ops::lookup("matmul"),args.data(),2);
            check(plan.scratch_doubles==0 && plan.scratch_indices==0,
                  "F matmul introduced planned scratch");
            ops::Workspace workspace;
            for (auto isa : {ops::Isa::scalar,ops::Isa::automatic}) {
                std::vector<double> storage(rows*cols+2,-917.0);
                ops::Output out;
                out.kind=plan.output_kind; out.shape=plan.output_shape;
                out.data=storage.data()+1;
                ops::Audit audit;
                ops::execute(plan,out,workspace,isa,audit);
                if (isa==ops::Isa::automatic)
                    matmul_library_available=std::strcmp(audit.isa,"eigen-array-matmul")==0;
                for (std::size_t i=0;i<rows;++i) for (std::size_t j=0;j<cols;++j)
                    check(storage[1+i*cols+j]==a[(rows-1-i)*inner+j%inner]*(j+1),
                          "F matmul failed scaled-coordinate oracle");
                check(storage.front()==-917.0 && storage.back()==-917.0,
                      "F matmul wrote outside independent result");
                check(a==original_a && b==original_b,"F matmul mutated input or padding");
                check(workspace.capacity_bytes()==0 && audit.scratch_bytes==0 &&
                          audit.algorithm_copy_bytes==0,
                      "F matmul introduced workspace or algorithm copies");
            }
        }
        {
            // Retain the rectangular fallback/zero-row cases, and exercise the
            // square C library with both C/F left views. Coordinate selection
            // independently covers reversed and broadcast left views too.
            const std::array<std::size_t,3> dimensions[]{
                {33,35,37},{33,35,40},{33,35,41},{40,40,40}};
            for (const auto &shape : dimensions) {
                const auto rows=shape[0], inner=shape[1], cols=shape[2];
                const bool square=rows==inner && inner==cols;
                for (int layout=0;layout<3;++layout) {
                    const std::size_t pitch=layout==2 ? inner+3 : cols+(layout==1 ? 3 : 0);
                    std::vector<double> a(rows*inner), b(layout==2 ? pitch*cols : inner*pitch,-713.0);
                    for (std::size_t i=0;i<a.size();++i)
                        a[i]=static_cast<int>(i%11)-5;
                    for (std::size_t k=0;k<inner;++k)
                        for (std::size_t j=0;j<cols;++j)
                            b[layout==2 ? j*pitch+k : k*pitch+j]=
                                k==j%inner ? static_cast<double>(j+1) : 0.0;
                    const auto original_a=a, original_b=b;
                    auto right=matrix(b.data(),inner,cols);
                    right.stride=layout==2 ? std::array<std::ptrdiff_t,3>{1,static_cast<std::ptrdiff_t>(pitch),1}
                                          : std::array<std::ptrdiff_t,3>{static_cast<std::ptrdiff_t>(pitch),1,1};
                    for (int left_layout=0;left_layout<(square ? 6 : 5);++left_layout) {
                        const bool reversed_rows=left_layout==1;
                        const bool reversed_columns=left_layout==2 || left_layout==4;
                        const bool repeated_rows=left_layout==3 || left_layout==4;
                        const bool fortran_left=left_layout==5;
                        const auto first_row=reversed_rows ? rows-1 : (repeated_rows ? rows/2 : 0);
                        auto left=matrix(a.data()+first_row*inner+(reversed_columns ? inner-1 : 0),rows,inner);
                        left.stride[0]=repeated_rows ? 0 :
                            (reversed_rows ? -static_cast<std::ptrdiff_t>(inner) : static_cast<std::ptrdiff_t>(inner));
                        left.stride[1]=reversed_columns ? -1 : 1;
                        if (fortran_left)
                            left.stride={1,static_cast<std::ptrdiff_t>(rows),1};
                        const std::array<ops::Value,8> args{left,right};
                        const auto plan=ops::prepare(ops::lookup("matmul"),args.data(),2);
                        check(plan.scratch_doubles==0 && plan.scratch_indices==0,
                              "wide matmul introduced planned scratch");
                        ops::Workspace workspace;
                        for (auto isa : {ops::Isa::scalar,ops::Isa::automatic}) {
                            std::vector<double> storage(rows*cols+2,-917.0);
                            ops::Output out;
                            out.kind=plan.output_kind; out.shape=plan.output_shape;
                            out.data=storage.data()+1;
                            ops::Audit audit;
                            ops::execute(plan,out,workspace,isa,audit);
                            for (std::size_t i=0;i<rows;++i)
                                for (std::size_t j=0;j<cols;++j) {
                                    const auto source_row=repeated_rows ? rows/2 : (reversed_rows ? rows-1-i : i);
                                    const auto source_column=reversed_columns ? inner-1-j%inner : j%inner;
                                    const auto source_index=fortran_left ?
                                        source_column*rows+source_row : source_row*inner+source_column;
                                    check(storage[1+i*cols+j]==a[source_index]*(j+1),
                                          "wide matmul failed scaled-coordinate oracle");
                                }
                            check(storage.front()==-917.0 && storage.back()==-917.0,
                                  "wide matmul wrote outside independent result");
                            check(a==original_a && b==original_b,
                                  "wide matmul mutated input or padding");
                            check(workspace.capacity_bytes()==0 && audit.scratch_bytes==0,
                                  "wide matmul introduced numeric workspace");
                            const bool library_used=std::strcmp(audit.isa,"eigen-array-matmul")==0;
                            if (matmul_library_available && isa==ops::Isa::automatic &&
                                square && layout==0 && !repeated_rows)
                                check(library_used,"square C/FC matmul bypassed available Eigen kernel");
                            if (layout!=2 && !repeated_rows &&
                                (!square || layout==1 || isa==ops::Isa::scalar))
                                check(!library_used,"unadmitted C matmul used Eigen kernel");
                            const bool copied_rows=std::strcmp(audit.isa,"matmul-repeated-rows")==0;
                            check(!copied_rows || (repeated_rows && isa==ops::Isa::automatic),
                                  "matmul reused rows without an exact dependency identity");
                            check(audit.algorithm_copy_bytes==(copied_rows ? (rows-1)*cols*sizeof(double) : 0),
                                  "matmul repeated-row copy accounting changed");
                        }
                    }
                }
            }
        }
        {
            // Exercise the fixed working-set gate without assuming Eigen was
            // built in. The existing F control above detects the actual backend.
            // layout: C, reverse rows, reverse columns, sliced, F, zero columns,
            // repeated rows, and a dense (non-coordinate) arithmetic fixture.
            struct Fixture { std::size_t rows, inner, cols; int layout; bool padded; };
            const Fixture fixtures[]{
                {32,682,32,0,false},{32,683,32,0,false},{32,684,32,0,false},
                {31,683,32,0,false},{32,683,31,0,false},
                {33,683,41,0,true},{33,683,41,1,true},{33,683,41,2,true},
                {33,683,41,3,true},{33,683,41,4,true},{33,683,41,5,true},
                {33,683,41,6,true},{33,683,41,7,true}};
            for (const auto &fixture : fixtures) {
                const auto rows=fixture.rows, inner=fixture.inner, cols=fixture.cols;
                const auto apitch=fixture.layout==3 ? inner*2+3 : inner;
                const auto bpitch=cols+(fixture.padded ? 3 : 0);
                std::vector<double> a(rows*apitch), b(inner*bpitch,-713.0);
                for (std::size_t i=0;i<a.size();++i)
                    a[i]=(static_cast<int>(i%19)-9)*0.125;
                for (std::size_t k=0;k<inner;++k)
                    for (std::size_t j=0;j<cols;++j)
                        b[k*bpitch+j]=fixture.layout==7 ?
                            (static_cast<int>((k*7+j*3)%17)-8)*0.01 :
                            (k==(j*71+inner-1)%inner ? (j+1)*0.125 : 0.0);
                const auto original_a=a, original_b=b;
                auto left=matrix(a.data(),rows,inner);
                left.stride={static_cast<std::ptrdiff_t>(apitch),1,1};
                if (fixture.layout==1) {
                    left.data=a.data()+(rows-1)*apitch;
                    left.stride[0]=-static_cast<std::ptrdiff_t>(apitch);
                } else if (fixture.layout==2) {
                    left.data=a.data()+inner-1;
                    left.stride[1]=-1;
                } else if (fixture.layout==3) left.stride[1]=2;
                else if (fixture.layout==4)
                    left.stride={1,static_cast<std::ptrdiff_t>(rows),1};
                else if (fixture.layout==5) left.stride[1]=0;
                else if (fixture.layout==6) {
                    left.data=a.data()+(rows/2)*apitch;
                    left.stride[0]=0;
                }
                auto right=matrix(b.data(),inner,cols);
                right.stride[0]=static_cast<std::ptrdiff_t>(bpitch);
                const std::array<ops::Value,8> args{left,right};
                const auto plan=ops::prepare(ops::lookup("matmul"),args.data(),2);
                check(plan.scratch_doubles==0 && plan.scratch_indices==0,
                      "blocked matmul introduced planned scratch");
                ops::Workspace workspace;
                std::vector<double> scalar;
                for (auto isa : {ops::Isa::scalar,ops::Isa::automatic}) {
                    std::vector<double> storage(rows*cols+2,-917.0);
                    ops::Output out;
                    out.kind=plan.output_kind; out.shape=plan.output_shape;
                    out.data=storage.data()+1;
                    ops::Audit audit;
                    ops::execute(plan,out,workspace,isa,audit);
                    if (isa==ops::Isa::scalar)
                        scalar.assign(storage.begin()+1,storage.end()-1);
                    else
                        check(std::memcmp(scalar.data(),storage.data()+1,scalar.size()*sizeof(double))==0,
                              "blocked matmul changed the scalar accumulation order");
                    if (fixture.layout!=7)
                        for (std::size_t i=0;i<rows;++i)
                            for (std::size_t j=0;j<cols;++j) {
                                const auto k=(j*71+inner-1)%inner;
                                const auto source_row=fixture.layout==1 ? rows-1-i :
                                    (fixture.layout==6 ? rows/2 : i);
                                const auto source_col=fixture.layout==2 ? inner-1-k :
                                    (fixture.layout==5 ? 0 : k);
                                const auto index=fixture.layout==4 ? source_col*rows+source_row :
                                    source_row*apitch+source_col*(fixture.layout==3 ? 2 : 1);
                                check(storage[1+i*cols+j]==a[index]*(j+1)*0.125,
                                      "blocked matmul failed scaled-coordinate oracle");
                            }
                    const bool admitted=matmul_library_available && isa==ops::Isa::automatic &&
                        rows>=32 && cols>=32;
                    const bool repeated=admitted && fixture.layout==6;
                    const bool blocked=admitted && !repeated && inner>=683;
                    check((std::strcmp(audit.isa,"eigen-block-matmul")==0)==blocked,
                          "blocked matmul working-set, shape, or scalar admission changed");
                    check((std::strcmp(audit.isa,"matmul-repeated-rows")==0)==repeated,
                          "blocked matmul bypassed repeated-row priority");
                    check(storage.front()==-917.0 && storage.back()==-917.0,
                          "blocked matmul wrote outside its independent output");
                    check(a==original_a && b==original_b,
                          "blocked matmul mutated input, slices, or padding");
                    check(workspace.capacity_bytes()==0 && audit.scratch_bytes==0 &&
                              audit.algorithm_copy_bytes==(repeated ? (rows-1)*cols*sizeof(double) : 0),
                          "blocked matmul introduced workspace or unaccounted copies");
                }
            }
        }
        {
            // Products straddle both the first and second K=64 boundaries.
            // Incorrect partial-sum addition, FMA, or zero-skipping can agree
            // on ordinary matrices but change these results and their consumers.
            constexpr std::size_t rows=33, inner=683, cols=41, pitch=cols+3;
            for (int fixture=0;fixture<8;++fixture) {
                std::vector<double> a(rows*inner,fixture==6 ? -0.0 : 0.0);
                std::vector<double> b(inner*pitch,-713.0);
                for (std::size_t k=0;k<inner;++k)
                    std::fill_n(b.data()+k*pitch,cols,1.0);
                for (std::size_t row=0;row<rows;++row) {
                    auto *x=a.data()+row*inner;
                    if (fixture==0 || fixture==1) {
                        const auto first=fixture==0 ? 63 : 127;
                        x[first]=1e16; x[first+1]=-1e16; x[first+2]=1.0;
                    } else if (fixture==2) {
                        x[63]=std::numeric_limits<double>::max();
                        x[64]=std::numeric_limits<double>::max();
                        x[65]=-std::numeric_limits<double>::max();
                    } else if (fixture==3) {
                        x[63]=-1.0; x[64]=1.0+std::ldexp(1.0,-27);
                    } else if (fixture==5) {
                        x[63]=std::numeric_limits<double>::denorm_min();
                        x[64]=std::numeric_limits<double>::denorm_min();
                        x[65]=-std::numeric_limits<double>::denorm_min();
                    } else if (fixture==7) x[63]=std::numeric_limits<double>::quiet_NaN();
                }
                for (std::size_t j=0;j<cols;++j) {
                    if (fixture==3) b[64*pitch+j]=1.0-std::ldexp(1.0,-27);
                    if (fixture==4) b[63*pitch+j]=std::numeric_limits<double>::infinity();
                    if (fixture==5) {
                        b[63*pitch+j]=0.5;
                        b[65*pitch+j]=0.5;
                    }
                }
                const auto original_a=a, original_b=b;
                auto right=matrix(b.data(),inner,cols);
                right.stride[0]=pitch;
                const std::array<ops::Value,8> args{matrix(a.data(),rows,inner),right};
                const auto plan=ops::prepare(ops::lookup("matmul"),args.data(),2);
                ops::Workspace workspace;
                std::vector<double> scalar;
                for (auto isa : {ops::Isa::scalar,ops::Isa::automatic}) {
                    std::vector<double> storage(rows*cols+2,-917.0);
                    ops::Output out;
                    out.kind=plan.output_kind; out.shape=plan.output_shape;
                    out.data=storage.data()+1;
                    ops::Audit audit;
                    ops::execute(plan,out,workspace,isa,audit);
                    if (isa==ops::Isa::scalar)
                        scalar.assign(storage.begin()+1,storage.end()-1);
                    for (std::size_t i=0;i<rows*cols;++i) {
                        const double value=storage[i+1];
                        if (fixture==0 || fixture==1) check(value==1.0,"matmul regrouped panel sums");
                        else if (fixture==2) check(std::isinf(value) && value>0.0,
                                                 "matmul reordered sequential overflow");
                        else if (fixture==3 || fixture==6)
                            check(value==0.0 && !std::signbit(value),"matmul changed separate multiply/add or +0 seed");
                        else if (fixture==5) check(value==std::numeric_limits<double>::denorm_min(),
                                                  "matmul changed subnormal products across a panel");
                        else check(std::isnan(value),"matmul skipped a NaN or zero-times-infinity product");
                        if (std::isfinite(scalar[i]))
                            check(std::memcmp(&scalar[i],&value,sizeof(double))==0,
                                  "blocked matmul changed a finite scalar bit pattern");
                        else
                            check((std::isnan(scalar[i]) && std::isnan(value)) ||
                                      (std::isinf(scalar[i]) && std::isinf(value) &&
                                       std::signbit(scalar[i])==std::signbit(value)),
                                  "blocked matmul changed scalar IEEE classification");
                    }
                    check((std::strcmp(audit.isa,"eigen-block-matmul")==0)==
                              (matmul_library_available && isa==ops::Isa::automatic),
                          "IEEE matmul fixture did not exercise the admitted backend");
                    check(storage.front()==-917.0 && storage.back()==-917.0,
                          "blocked IEEE matmul wrote outside output");
                    check(std::memcmp(a.data(),original_a.data(),a.size()*sizeof(double))==0 &&
                              std::memcmp(b.data(),original_b.data(),b.size()*sizeof(double))==0,
                          "blocked IEEE matmul mutated an input or NaN payload");
                    check(workspace.capacity_bytes()==0 && audit.scratch_bytes==0 &&
                              audit.algorithm_copy_bytes==0,
                          "blocked IEEE matmul introduced scratch or copies");
                }
            }
        }
        for (const std::size_t n : {std::size_t{129}, std::size_t{1025}})
        for (const int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
            const int prior_rounding = std::fegetround();
            struct RestoreRounding {
                int prior;
                ~RestoreRounding() { std::fesetround(prior); }
            } restore_rounding{prior_rounding};
            check(std::fesetround(rounding)==0,"solve rounding control");
            // The second panel must swap its saved column-32 multiplier when
            // pivot 33 moves to row 65. n=129 also exercises a partial panel and
            // the fixed-width Eigen tail; 1025 enters four-row grouped updates.
            constexpr std::size_t columns=3;
            std::vector<double> a(n*n,0.0), expected(n*columns), b(n*columns,0.0);
            for (std::size_t i=0;i<n;++i) a[i*n+i]=2.0;
            const std::size_t indices[]{32,33,65};
            const double block[]{4,1,2,1,1,7,2,5,3};
            for (std::size_t i=0;i<3;++i) for (std::size_t j=0;j<3;++j)
                a[indices[i]*n+indices[j]]=block[i*3+j];
            for (std::size_t i=0;i<n;++i) for (std::size_t j=0;j<columns;++j)
                expected[i*columns+j]=double((i+3*j)%7+1);
            // Independently construct A*X for a known integer X; no second
            // elimination implementation is used as the correctness oracle.
            for (std::size_t i=0;i<n;++i) for (std::size_t k=0;k<n;++k)
                for (std::size_t j=0;j<columns;++j)
                    b[i*columns+j]+=a[i*n+k]*expected[k*columns+j];
            const auto readonly_a=a, readonly_b=b;
            const std::array<ops::Value,8> args{
                matrix(readonly_a.data(),n,n),matrix(readonly_b.data(),n,columns)};
            const auto plan=ops::prepare(ops::lookup("solve_multi"),args.data(),2);
            check(plan.scratch_doubles==n*n && plan.scratch_indices==0,
                  "large solve_multi workspace geometry");
            ops::Workspace workspace;
            workspace.ensure(plan.scratch_doubles,plan.scratch_indices);
            const auto *scratch=workspace.doubles.data();
            const auto capacity=workspace.capacity_bytes();
            std::vector<double> scalar(n*columns,-913.25), automatic(n*columns,-913.25);
            ops::Output out;
            out.kind=plan.output_kind; out.shape=plan.output_shape;
            ops::Audit scalar_audit, automatic_audit;
            out.data=scalar.data();
            ops::execute(plan,out,workspace,ops::Isa::scalar,scalar_audit);
            out.data=automatic.data();
            ops::execute(plan,out,workspace,ops::Isa::automatic,automatic_audit);
            check(std::memcmp(scalar.data(),automatic.data(),scalar.size()*sizeof(double))==0,
                  "large solve_multi changed ordered arithmetic or pivot factors");
            for (std::size_t i=0;i<expected.size();++i)
                check(std::abs(automatic[i]-expected[i])<1e-12,
                      "large solve_multi failed independent known solution");
            check(readonly_a==a && readonly_b==b,"large solve_multi mutated readonly inputs");
            check(workspace.doubles.data()==scratch && workspace.capacity_bytes()==capacity,
                  "large solve_multi changed workspace allocation");
            for (const auto *audit : {&scalar_audit,&automatic_audit}) {
                check(audit->scratch_bytes==n*n*sizeof(double),
                      "large solve_multi introduced numeric scratch");
                check(audit->algorithm_copy_bytes==(n*n+n*columns)*sizeof(double),
                      "large solve_multi changed algorithm copy accounting");
            }
            check(std::strcmp(scalar_audit.isa,"scalar")==0,
                  "explicit scalar solve_multi used the library");
            // This existing backend has the same Eigen/platform admission,
            // avoiding a test-only compile macro that misses Eigen-OFF builds.
            const auto backend=run(ops::lookup("matvec"),
                {matrix(readonly_a.data(),n,n),vector(expected.data(),n,columns)},
                2,ops::Isa::automatic);
            check(std::strcmp(automatic_audit.isa,backend.audit.eigen_matvec_calls ?
                              "eigen-array-blocked-lu" : "scalar")==0,
                  "large solve_multi did not use the available backend");
            check(std::fegetround()==rounding,"solve changed caller rounding mode");
        }
        {
            // Single solve permits NaN pivots and must leave out untouched on
            // elimination failure, unlike the multi-RHS finite-check contract.
            constexpr std::size_t n=129;
            for (int fault=0;fault<3;++fault) {
                std::vector<double> a(n*n,0.),b(n,2.),scalar(n+2,-913.25),automatic(n+2,-913.25);
                for (std::size_t i=0;i<n;++i) a[i*n+i]=2.;
                if (fault==1) a[64*n+64]=std::numeric_limits<double>::quiet_NaN();
                if (fault==2) a[(n-1)*n+n-1]=0.;
                const auto before_a=a,before_b=b;
                std::array<ops::Value,8> args{matrix(a.data(),n,n),vector(b.data(),n)};
                const auto plan=ops::prepare(ops::lookup("solve"),args.data(),2);
                ops::Workspace workspace;workspace.ensure(plan.scratch_doubles,plan.scratch_indices);
                for (auto isa:{ops::Isa::scalar,ops::Isa::automatic}) {
                    auto &storage=isa==ops::Isa::scalar ? scalar : automatic;
                    ops::Output out;out.shape=plan.output_shape;out.data=storage.data()+1;
                    ops::Audit audit;bool failed=false;
                    try {ops::execute(plan,out,workspace,isa,audit);}
                    catch (const ops::Error &error) {failed=std::strcmp(error.what(),"SINGULAR_MATRIX")==0;}
                    check(failed==(fault==2),"single solve changed pivot failure contract");
                    check(storage.front()==-913.25 && storage.back()==-913.25,"single solve crossed output");
                    if (fault==2)
                        check(std::all_of(storage.begin(),storage.end(),[](double x){return x==-913.25;}),
                              "single solve wrote output before elimination failure");
                    else for (std::size_t i=0;i<n;++i)
                        check(fault==1 ? std::isnan(storage[i+1]) : storage[i+1]==1.,
                              "single solve changed NaN propagation or known solution");
                }
                check(std::memcmp(a.data(),before_a.data(),a.size()*8)==0 && b==before_b,
                      "single solve changed inputs");
            }
        }
        {
            // Empty RHS/output buffers may be null, but A still has a solve
            // contract. Complete finite validation precedes pivot rejection.
            constexpr std::size_t n=129;
            std::vector<double> a(n*n,0.0);
            const std::array<ops::Value,8> args{matrix(a.data(),n,n),matrix(nullptr,n,0)};
            const auto plan=ops::prepare(ops::lookup("solve_multi"),args.data(),2);
            check(plan.output_shape==ops::matrix_shape(n,0) && plan.scratch_doubles==n*n,
                  "empty RHS lost matrix validation geometry");
            ops::Workspace workspace;
            workspace.ensure(plan.scratch_doubles,plan.scratch_indices);
            const auto *scratch=workspace.doubles.data();
            const auto capacity=workspace.capacity_bytes();
            ops::Output out;
            out.kind=plan.output_kind; out.shape=plan.output_shape; out.data=nullptr;
            for (int fixture=0;fixture<4;++fixture) {
                std::fill(a.begin(),a.end(),0.0);
                for (std::size_t i=0;i<n;++i) a[i*n+i]=2.0;
                if (fixture==1) a[(n-1)*n+n-1]=0.0;
                if (fixture>=2) {
                    a[0]=0.0;
                    a[(n-1)*n+n-1]=fixture==2 ? std::numeric_limits<double>::quiet_NaN()
                                                            : std::numeric_limits<double>::infinity();
                }
                const auto original=a;
                for (auto isa : {ops::Isa::scalar,ops::Isa::automatic}) {
                    ops::Audit audit;
                    std::string error;
                    try { ops::execute(plan,out,workspace,isa,audit); }
                    catch (const ops::Error &e) { error=e.what(); }
                    check(error==(fixture==0 ? "" : fixture==1 ? "SINGULAR_MATRIX" : "NON_FINITE_RESULT"),
                          "empty RHS suppressed or reordered coefficient validation");
                    check(out.data==nullptr && out.shape.size()==0,
                          "empty RHS materialized an output buffer");
                    check(std::memcmp(a.data(),original.data(),a.size()*sizeof(double))==0,
                          "empty RHS solve_multi mutated input");
                    check(workspace.doubles.data()==scratch && workspace.capacity_bytes()==capacity,
                          "empty RHS solve_multi changed workspace allocation");
                }
            }
        }
        {
            // Sanitizer builds exercise every RHS block/tail and both triangle
            // directions. Power-of-two coefficients give an exact known solve.
            constexpr std::size_t n=65;
            for (auto columns : {8u,15u,17u,31u,33u})
            for (bool lower : {false,true}) for (bool unit : {false,true})
            for (bool column_major : {false,true}) {
                std::vector<double> a(n*n),b(n*columns),expected(n*columns);
                const auto index=[&](std::size_t i,std::size_t j) {
                    return column_major ? i+j*n : i*n+j;
                };
                for (std::size_t i=0;i<n;++i) for (std::size_t j=0;j<columns;++j)
                    expected[i*columns+j]=double((i+j)%5);
                for (std::size_t i=0;i<n;++i) {
                    for (std::size_t k=0;k<n;++k) {
                        const bool consumed=lower ? k<i : k>i;
                        a[index(i,k)]=i==k ? (unit ? NAN : 2.) : consumed ?
                            double(int((i+3*k)%7)-3)/128. : NAN;
                    }
                    for (std::size_t j=0;j<columns;++j) {
                        b[i*columns+j]=(unit ? 1. : 2.)*expected[i*columns+j];
                        for (std::size_t k=lower ? 0 : i+1;k<(lower ? i : n);++k)
                            b[i*columns+j]+=a[index(i,k)]*expected[k*columns+j];
                    }
                }
                auto av=matrix(a.data(),n,n);
                if (column_major) av.stride={1,static_cast<std::ptrdiff_t>(n),1};
                const std::array<ops::Value,8> args{av,matrix(b.data(),n,columns),
                    ops::Value::number(lower),ops::Value::number(unit)};
                for (auto isa : {ops::Isa::scalar,ops::Isa::automatic}) {
                    const auto result=run(ops::lookup("triangular_solve"),args,4,isa);
                    check(result.values==expected,"triangular RHS blocks lost exact known solution");
                    check(result.audit.scratch_bytes==0,"triangular solve allocated numerical scratch");
                    check(result.audit.algorithm_copy_bytes==b.size()*sizeof(double),
                          "triangular solve changed RHS copy accounting");
                }
            }
        }
        {
            ops::Workspace work;
            work.ensure(1024,0);
            work.ensure(1536,0);
            check(work.doubles.capacity()==1536,"scratch growth retained geometric spare capacity");
            const auto *address=work.doubles.data();
            work.ensure(1024,0);
            check(work.doubles.data()==address,"smaller scratch failed to reuse storage");
            constexpr std::size_t m=1025,k=257,n=1027;
            const auto needed=ops::matmul_scratch_doubles(ops::matrix_shape(m,k),ops::matrix_shape(k,n));
            check(needed==0 || needed==320*k+72,"packed scratch shape bound");
            check(ops::matmul_scratch_doubles(ops::matrix_shape(1023,k),ops::matrix_shape(k,n))==0,
                  "small matrix unexpectedly reserved panels");
            for (int layout=0;layout<4;++layout) {
                std::vector<double> a(m*k),b(k*n),storage(m*n+2,-718.0);
                auto av=matrix(a.data(),m,k),bv=matrix(b.data(),k,n);
                if (layout&1) av.stride={1,static_cast<std::ptrdiff_t>(m)};
                if (layout&2) bv.stride={1,static_cast<std::ptrdiff_t>(k)};
                for (std::size_t i=0;i<m;++i) for(std::size_t t=0;t<k;++t)
                    a[i*av.stride[0]+t*av.stride[1]]=static_cast<int>(i%5)-2;
                for (std::size_t t=0;t<k;++t) for(std::size_t j=0;j<n;++j)
                    b[t*bv.stride[0]+j*bv.stride[1]]=static_cast<int>(j%7)-3;
                const auto saved_a=a,saved_b=b;
                ops::Value args[]{av,bv};
                const auto prepared=ops::prepare(ops::lookup("matmul"),args,2);
                ops::Output out; out.shape=ops::matrix_shape(m,n);out.data=storage.data()+1;
                ops::Audit audit;ops::execute(prepared,out,work,ops::Isa::automatic,audit);
                check(std::strcmp(audit.isa,"eigen-packed-matmul")==0 || needed==0,
                      "admitted matrix bypassed packed Eigen");
                check(audit.scratch_bytes==needed*sizeof(double),"panel scratch audit mismatch");
                for(std::size_t i=0;i<m;++i) for(std::size_t j=0;j<n;++j)
                    check(storage[1+i*n+j]==static_cast<double>(k)*(static_cast<int>(i%5)-2)*(static_cast<int>(j%7)-3),
                          "packed matmul changed exact product or tail");
                check(storage.front()==-718.0 && storage.back()==-718.0,"packed matmul crossed output extent");
                check(a==saved_a && b==saved_b,"packed matmul mutated borrowed input");
                if (needed) {
                    const auto prefix=m-m%4,bulk=n-n%6;
                    check(audit.algorithm_copy_bytes==(prefix+((prefix+127)/128)*bulk)*k*8,
                          "panel copy audit mismatch");
                }
            }
        }
        {
            constexpr std::size_t n=1025;
            const auto needed=ops::cholesky_scratch_doubles(n);
            check(needed==0 || needed==16*(n+17)+128,"Cholesky panel shape bound");
            check(ops::cholesky_scratch_doubles(1023)==0,"small Cholesky reserved panels");
            for (bool column_major : {false,true}) {
                std::vector<double> a(n*n),storage(n*n+2,-772.0);
                for (std::size_t i=0;i<n;++i) for (std::size_t j=0;j<n;++j)
                    a[i*n+j]=i==j ? n+1. : double(static_cast<int>((i+j)%11)-5)/16.;
                const auto saved=a;
                auto av=matrix(a.data(),n,n);
                if (column_major) av.stride={1,static_cast<std::ptrdiff_t>(n)};
                const std::array<ops::Value,8> args{av};
                const auto expected=run(ops::lookup("cholesky"),args,1,ops::Isa::scalar);
                const auto prepared=ops::prepare(ops::lookup("cholesky"),args.data(),1);
                ops::Workspace work;
                ops::Output out;out.shape=ops::matrix_shape(n,n);out.data=storage.data()+1;
                ops::Audit audit;ops::execute(prepared,out,work,ops::Isa::automatic,audit);
                check(std::memcmp(storage.data()+1,expected.values.data(),n*n*sizeof(double))==0,
                      "Cholesky panel arithmetic or final row changed");
                check(a==saved,"Cholesky panel changed input");
                check(storage.front()==-772. && storage.back()==-772.,"Cholesky crossed output extent");
                check(audit.scratch_bytes==needed*sizeof(double),"Cholesky panel scratch audit");
                check(audit.algorithm_copy_bytes==n*(n+1)/2*8*(needed ? 2 : 1),"Cholesky panel copy audit");
                check(!needed || std::strcmp(audit.isa,"eigen-array-panel-cholesky")==0,
                      "eligible Cholesky did not use Eigen panels");
            }
        }
        {
            constexpr std::size_t n=33,d=16,k=2;
            std::vector<double> x(n*d),means(k*d),vars(k*d,1.);
            double weights[k]{0.5,0.5};
            for (std::size_t j=0;j<d;++j) for (std::size_t i=0;i<n;++i)
                x[i+j*n]=double(i%3)+double(j%5)/8.;
            auto xv=matrix(x.data(),n,d);xv.stride={1,static_cast<std::ptrdiff_t>(n)};
            std::array<ops::Value,8> args{xv,vector(weights,k),matrix(means.data(),k,d),
                matrix(vars.data(),k,d),ops::Value::number(5),ops::Value::number(0)};
            const auto expected=run(ops::lookup("gaussian_mixture_em"),args,6,ops::Isa::scalar);
            const auto plan=ops::prepare(ops::lookup("gaussian_mixture_em"),args.data(),6);
            std::vector<double> storage(plan.output_shape.size()+2,-883.);
            ops::Output out;out.shape=plan.output_shape;out.data=storage.data()+1;
            ops::Workspace work;ops::Audit audit;
            ops::execute(plan,out,work,ops::Isa::automatic,audit);
            check(std::memcmp(storage.data()+1,expected.values.data(),expected.values.size()*8)==0,
                  "column EM changed model parameters or stopping state");
            check(storage.front()==-883. && storage.back()==-883.,"column EM crossed output bounds");
            check(audit.scratch_bytes==expected.audit.scratch_bytes,"column EM added heap scratch");
            check(audit.algorithm_copy_bytes==expected.audit.algorithm_copy_bytes,"column EM added input packing");
        }
        {
            for (const auto d : {std::size_t(8),std::size_t(1024)}) {
                const auto rows=d+1;
                std::vector<double> x(rows*d),storage(d*d+2,-993.);
                for (std::size_t i=0;i<d;++i) x[i*d+i]=1.;
                const auto saved=x;
                const std::array<ops::Value,8> args{matrix(x.data(),rows,d)};
                const auto covariance=run(ops::lookup("covariance"),args,1,ops::Isa::automatic);
                const std::array<ops::Value,8> factor_args{matrix(covariance.values.data(),d,d)};
                const auto expected=run(ops::lookup("cholesky"),factor_args,1,ops::Isa::automatic);
                const auto first=ops::prepare(ops::lookup("covariance"),args.data(),1);
                const auto second=ops::prepare(ops::lookup("cholesky"),factor_args.data(),1);
                std::vector<double> failed_c;
                for (bool column_output : {false,true}) {
                    std::fill(storage.begin(),storage.end(),-993.);
                    ops::Output out;out.shape=ops::matrix_shape(d,d);out.data=storage.data()+1;
                    out.order=column_output ? ops::MemoryOrder::f : ops::MemoryOrder::c;
                    ops::Workspace work;ops::Audit audit;
                    const auto verify=[&] {
                        for (std::size_t i=0;i<d;++i) for (std::size_t j=0;j<d;++j)
                            check(std::memcmp(storage.data()+1+(column_output ? j*d+i : i*d+j),
                                              expected.values.data()+i*d+j,sizeof(double))==0,
                                  "consumed covariance changed Cholesky arithmetic or output order");
                        check(x==saved,"consumed covariance modified public source");
                        check(storage.front()==-993. && storage.back()==-993.,"consumed covariance crossed output");
                        check(out.order==(column_output ? ops::MemoryOrder::f : ops::MemoryOrder::c),
                              "consumed covariance changed output descriptor");
                    };
                    ops::covariance_cholesky_pair(first,second,out,work,audit,column_output);
                    verify();
                    check(audit.eigen_covariance_calls==covariance.audit.eigen_covariance_calls,
                          "consumed covariance double-counted library work");
                    check(audit.algorithm_copy_bytes==(second.scratch_doubles ? d*(d+1)*8 : 0)+
                          (column_output ? (d*d-d)*8 : 0),
                          "consumed covariance layout copy accounting");
                    x[(d-1)*d+d-1]=0.;
                    const auto bad_source=x;
                    bool failed=false;
                    try {ops::covariance_cholesky_pair(first,second,out,work,audit,column_output);}
                    catch (const ops::Error &error) {failed=std::strcmp(error.what(),"DOMAIN_ERROR")==0;}
                    check(failed,"consumed covariance failed to preserve singular pivot error");
                    check(x==bad_source && storage.front()==-993. && storage.back()==-993.,
                          "failed consumed covariance changed input or output canary");
                    if (!column_output) failed_c.assign(storage.begin()+1,storage.end()-1);
                    else check(std::memcmp(storage.data()+1,failed_c.data(),d*d*8)==0,
                               "failed F covariance pair transposed canonical replay contents");
                    x[(d-1)*d+d-1]=1.;
                    ops::covariance_cholesky_pair(first,second,out,work,audit,column_output);
                    verify();
                }
            }
        }
        {
            const std::size_t m=513,k=257,n=515;
            std::vector<double> a(m*k),b(k*n),storage(m*n+2,-993.);
            for (std::size_t i=0;i<a.size();++i) a[i]=1.+double(i%17)/16.;
            for (std::size_t i=0;i<b.size();++i) b[i]=1.+double(i%13)/16.;
            auto right=matrix(b.data(),k,n);right.stride={1,static_cast<std::ptrdiff_t>(k)};
            const std::array<ops::Value,8> args{matrix(a.data(),m,k),right};
            const auto plan=ops::prepare(ops::lookup("matmul"),args.data(),2);
            ops::Output out;out.shape=plan.output_shape;out.data=storage.data()+1;
            ops::Workspace work;work.ensure(64,8);ops::Audit audit;
            const auto reserve=ops::matmul_accelerate_reserve_doubles(args[0].shape,args[1].shape);
            const bool accepted=ops::try_accelerate_matmul_root(plan,out,work,audit);
            check(accepted==(reserve!=0),"guarded matmul platform admission mismatch");
            if (accepted) {
                const auto expected=run(ops::lookup("matmul"),args,2,ops::Isa::automatic);
                check(std::memcmp(storage.data()+1,expected.values.data(),m*n*8)==0,
                      "guarded matmul changed exact binary fixture");
                check(audit.accelerate_matmul_calls==1 && audit.matmul_zero_calls==0,
                      "guarded matmul lost successful library audit");
                const auto vendor_reserve=(1024*1024+512*k)*8;
                check(audit.native_library_workspace_reserve_bytes==vendor_reserve && reserve*8>=vendor_reserve,
                      "guarded matmul reserve audit mismatch");
                check(work.capacity_bytes()==0,"guarded matmul retained old panels");
                for (std::size_t i=0;i<m;i+=2)
                    for (std::size_t j=0;j<k;++j) a[i*k+j]=-a[i*k+j];
                for (std::size_t j=0;j<n;j+=2)
                    for (std::size_t i=0;i<k;++i) b[j*k+i]=-b[j*k+i];
                const int previous_round=std::fegetround();
                for (const int mode:{FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
                    check(std::fesetround(mode)==0,"cannot set GEMM rejection rounding");
                    std::fill(storage.begin()+1,storage.end()-1,-993.);
                    work.ensure(64,8);const auto capacity=work.capacity_bytes();
                    audit.isa="untouched";
                    check(!ops::try_accelerate_matmul_root(plan,out,work,audit),
                          "signed GEMM accepted non-nearest rounding");
                    check(std::fegetround()==mode && work.capacity_bytes()==capacity &&
                          std::strcmp(audit.isa,"untouched")==0,
                          "signed GEMM rejection changed caller state");
                    check(std::all_of(storage.begin()+1,storage.end()-1,[](double x){return x==-993.;}),
                          "signed GEMM rounding rejection wrote output");
                }
                check(std::fesetround(FE_TONEAREST)==0,"cannot set GEMM nearest rounding");
                check(ops::try_accelerate_matmul_root(plan,out,work,audit),
                      "signed GEMM rejected cancellation-free inputs");
                const auto signed_expected=run(ops::lookup("matmul"),args,2,ops::Isa::automatic);
                check(std::memcmp(storage.data()+1,signed_expected.values.data(),m*n*8)==0,
                      "signed GEMM changed exact binary fixture");
                check(std::fesetround(previous_round)==0,"cannot restore GEMM rounding");
                std::fill(storage.begin()+1,storage.end()-1,-993.);
                a[0]=-0.;
                check(!ops::try_accelerate_matmul_root(plan,out,work,audit),
                      "guarded matmul accepted negative zero");
                check(std::all_of(storage.begin()+1,storage.end()-1,[](double x){return x==-993.;}),
                      "guarded matmul rejection wrote output");
                std::fill(a.begin(),a.end(),0.);work.ensure(64,8);
                check(ops::try_accelerate_matmul_root(plan,out,work,audit),
                      "guarded matmul rejected finite all-zero domain");
                check(audit.matmul_zero_calls==1 && audit.accelerate_matmul_calls==0,
                      "zero matmul was mislabeled as BLAS");
                check(work.capacity_bytes()==0,"zero matmul retained old panels");
                check(std::all_of(storage.begin()+1,storage.end()-1,[](double x){return x==0. && !std::signbit(x);}),
                      "zero matmul changed zero sign");
            } else {
                check(std::all_of(storage.begin()+1,storage.end()-1,[](double x){return x==-993.;}),
                      "disabled library wrote output");
            }
            check(storage.front()==-993. && storage.back()==-993.,"guarded matmul crossed output bounds");
        }
        {
            const std::size_t n=65,d=256;
            std::vector<double> x(n*d),storage(d*d+2,-993.);
            std::uint64_t seed=91021;
            for (auto &v:x) { seed=seed*6364136223846793005ull+1; v=double((seed>>32)%1024)/1024.-.5; }
            const auto before=x;
            const std::array<ops::Value,8> args{matrix(x.data(),n,d)};
            const auto plan=ops::prepare(ops::lookup("covariance"),args.data(),1);
            ops::Output out;out.shape=plan.output_shape;out.data=storage.data()+1;
            ops::Workspace work;ops::Audit audit;
            const auto reserve=ops::covariance_accelerate_reserve_doubles(args[0].shape);
            const bool handled=ops::try_accelerate_covariance(plan,out,work,audit);
            check(handled==(reserve!=0),"covariance platform admission mismatch");
            if (handled) {
                const auto expected=run(ops::lookup("covariance"),args,1,ops::Isa::automatic);
                check(audit.accelerate_covariance_calls==1 && audit.covariance_syrk_calls==1,
                      "covariance successful DSYRK audit missing");
                for (std::size_t j=0;j<d;++j) {
                    check(std::memcmp(storage.data()+1+j*d+j,expected.values.data()+j*d+j,8)==0,
                          "covariance diagonal bits changed");
                    for (std::size_t k=0;k<d;++k) {
                        const auto a=storage[1+j*d+k],b=expected.values[j*d+k];
                        check(std::abs(a-b)<=1e-12+1e-12*std::abs(b),"covariance tolerance changed");
                        check((a>0)==(b>0) && (a==0)==(b==0) && std::signbit(a)==std::signbit(b),
                              "covariance discrete classes changed");
                    }
                }
                check(audit.covariance_panel_materialization_bytes==n*d*8,"covariance panel audit missing");
                check(audit.native_library_workspace_reserve_bytes==reserve*8,"covariance reserve missing");
                std::fill(x.begin(),x.end(),0.);
                check(ops::try_accelerate_covariance(plan,out,work,audit),"zero covariance not handled");
                check(audit.covariance_zero_calls==1 && audit.accelerate_covariance_attempts==0 &&
                      audit.accelerate_covariance_calls==0 && audit.covariance_panel_materialization_bytes==0,
                      "zero covariance mislabeled as BLAS");
                check(std::all_of(storage.begin()+1,storage.end()-1,[](double v){return v==0. && !std::signbit(v);}),
                      "zero covariance changed exact zero sign");
                check(work.capacity_bytes()==0,"zero covariance retained dead panel");
                x[0]=-0.;
                std::fill(storage.begin()+1,storage.end()-1,-993.);
                check(!ops::try_accelerate_covariance(plan,out,work,audit),"signed-zero covariance bypassed canonical");
                check(std::all_of(storage.begin()+1,storage.end()-1,[](double v){return v==-993.;}),
                      "signed-zero rejection changed output");
            }
            std::copy(before.begin(),before.end(),x.begin());
            std::fill(storage.begin()+1,storage.end()-1,-993.);
            const auto rounding=std::fegetround();
            check(std::fesetround(FE_UPWARD)==0,"cannot set rounding mode for covariance control");
            const bool rejected=!ops::try_accelerate_covariance(plan,out,work,audit);
            const auto restored=std::fesetround(rounding);
            check(restored==0 && rejected,"covariance accepted non-nearest rounding mode");
            check(std::all_of(storage.begin(),storage.end(),[](double v){return v==-993.;}),
                  "covariance rejected environment wrote output");
            check(x==before,"covariance changed borrowed input");
        }
        for (bool reverse_f : {false, true}) {
            constexpr std::size_t n=1031,d=259;
            std::vector<double> x(n*d),storage(d*d+2,-993.);
            std::uint64_t seed=91021;
            for (auto &v:x) { seed=seed*6364136223846793005ull+1; v=double((seed>>32)%1024)/128.-4.; }
            for (std::size_t i=0;i<n;++i) x[reverse_f ? i : i*d]=(i%2 ? -12. : 12.);
            auto input=matrix(x.data(),n,d);
            if (reverse_f) { input.data=x.data()+n-1;input.stride={-1,static_cast<std::ptrdiff_t>(n),1}; }
            const std::array<ops::Value,8> args{input};
            const auto plan=ops::prepare(ops::lookup("covariance"),args.data(),1);
            ops::Output out;out.shape=plan.output_shape;out.data=storage.data()+1;
            ops::Workspace work;ops::Audit audit;
            const auto before=x;
            const bool handled=ops::try_accelerate_covariance(plan,out,work,audit,true);
            check(handled==(ops::covariance_accelerate_reserve_doubles(input.shape)!=0),
                  "short covariance platform admission mismatch");
            if (handled) {
                const auto expected=run(ops::lookup("covariance"),args,1,ops::Isa::automatic);
                check(audit.accelerate_covariance_calls==1 && audit.covariance_syrk_calls==9 &&
                      audit.covariance_panel_materialization_bytes==(1024+n)*d*8,
                      "short covariance retry or materialization audit missing");
                for (std::size_t i=0;i<d*d;++i) {
                    const auto a=storage[i+1],b=expected.values[i];
                    check(std::abs(a-b)<=1e-12+1e-12*std::abs(b),"short covariance tolerance changed");
                    check((a==0)==(b==0) && std::signbit(a)==std::signbit(b),"short covariance classes changed");
                    if (i/d==i%d) check(std::memcmp(&storage[i+1],&expected.values[i],8)==0,
                                        "short covariance diagonal changed");
                }
                check(x==before,"short covariance changed borrowed input");
                x[reverse_f ? 0 : (n-1)*d]=64.;
                check(ops::try_accelerate_covariance(plan,out,work,audit,true),"late short covariance not handled");
                const auto replay=run(ops::lookup("covariance"),args,1,ops::Isa::automatic);
                check(audit.covariance_fallbacks==1 && audit.covariance_syrk_calls==8,
                      "late short covariance lost attempted blocks");
                check(std::memcmp(storage.data()+1,replay.values.data(),d*d*8)==0,
                      "late short covariance did not restore canonical output");
            }
            check(storage.front()==-993. && storage.back()==-993.,"short covariance crossed output bounds");
        }
        for (int exponent : {-202, 195, 393}) {
            constexpr std::size_t n=65,d=256;
            std::vector<double> x(n*d),storage(d*d+2,-993.);
            for (std::size_t i=0;i<n;++i) for (std::size_t j=0;j<d;++j)
                x[i*d+j]=std::ldexp(double((i*17+j*13)%71)-35.,exponent);
            const auto before=x;
            const std::array<ops::Value,8> args{matrix(x.data(),n,d)};
            const auto plan=ops::prepare(ops::lookup("correlation"),args.data(),1);
            ops::Output out;out.shape=plan.output_shape;out.data=storage.data()+1;
            ops::Workspace work;ops::Audit audit;
            const bool handled=ops::try_accelerate_covariance(plan,out,work,audit);
            check(handled==(ops::covariance_accelerate_reserve_doubles(args[0].shape)!=0),
                  "correlation platform admission mismatch");
            if (handled) {
                const auto expected=run(ops::lookup("correlation"),args,1,ops::Isa::automatic);
                for (std::size_t i=0;i<d*d;++i) {
                    const auto a=storage[i+1],b=expected.values[i];
                    check(std::abs(a-b)<=1e-12+1e-12*std::abs(b),"correlation normalized tolerance changed");
                    check((a>0)==(b>0) && (a==0)==(b==0) && std::signbit(a)==std::signbit(b),
                          "correlation sign or zero changed");
                }
                for (std::size_t i=0;i<d;++i)
                    check(std::memcmp(storage.data()+1+i*d+i,expected.values.data()+i*d+i,8)==0,
                          "correlation diagonal bits changed");
                check(x==before,"correlation changed borrowed input");
                const auto rounding=std::fegetround();
                check(std::fesetround(FE_UPWARD)==0,"cannot set correlation rounding");
                check(!ops::try_accelerate_covariance(plan,out,work,audit),"correlation accepted non-nearest rounding");
                check(std::fegetround()==FE_UPWARD && std::fesetround(rounding)==0,"correlation changed rounding");
                std::fill(x.begin(),x.end(),0.);
                check(!ops::try_accelerate_covariance(plan,out,work,audit),"zero correlation bypassed canonical error");
            }
            check(storage.front()==-993. && storage.back()==-993.,"correlation crossed output bounds");
        }
        for (std::size_t n=32;n<36;++n) {
            constexpr std::size_t d=256;
            const auto leading=n+3;
            std::vector<double> x(leading*d,-993.),storage(d*d+2,-993.);
            for (std::size_t j=0;j<d;++j)
                for (std::size_t i=0;i<n;++i)
                    x[j*leading+i]=double((i*17+j*13)%71)-35.;
            const auto before=x;
            auto value=matrix(x.data(),n,d);
            value.stride={1,static_cast<std::ptrdiff_t>(leading)};
            const std::array<ops::Value,8> args{value};
            const auto plan=ops::prepare(ops::lookup("covariance"),args.data(),1);
            ops::Output out;out.shape=plan.output_shape;out.data=storage.data()+1;
            ops::Workspace work;ops::Audit audit;
            const auto reserve=ops::covariance_accelerate_reserve_doubles(value.shape);
            const bool handled=ops::try_accelerate_covariance(plan,out,work,audit);
            check(handled==(reserve!=0),"centered covariance platform admission mismatch");
            if (handled) {
                const auto expected=run(ops::lookup("covariance"),args,1,ops::Isa::automatic);
                check(std::memcmp(storage.data()+1,expected.values.data(),d*d*8)==0,
                      "centered covariance changed packet or tail bits");
                check(audit.eigen_covariance_calls==1 && audit.covariance_fallbacks==1 &&
                      audit.covariance_syrk_calls==0 && audit.accelerate_covariance_calls==0,
                      "centered covariance lost replay audit");
                check(audit.scratch_bytes==(n+3)*d*8 && work.capacity_bytes()==0,
                      "centered covariance lost peak or retained dead panel");
            }
            check(x==before,"centered covariance changed borrowed input");
            check(storage.front()==-993. && storage.back()==-993.,
                  "centered covariance crossed output bounds");
        }
        {
            constexpr std::size_t n=512;
            const auto capacity=ops::symmetric_eigh_split_scratch_doubles(n);
            if (capacity) {
                std::vector<double> input(n*n,0),values(n+2,-993.),vectors(n*n+2,-993.);
                for (std::size_t i=0;i<n;++i) input[i*n+i]=double(n-i);
                input[1]=input[n]=.25;
                const auto before=input;
                const std::array<ops::Value,8> args{matrix(input.data(),n,n)};
                const auto plan=ops::prepare(ops::lookup("symmetric_eigh"),args.data(),1);
                const auto reference=run(ops::lookup("symmetric_eigh"),args,1,ops::Isa::scalar);
                ops::Output w,v;
                w.shape=ops::vector_shape(n);w.data=values.data()+1;
                v.shape=ops::matrix_shape(n,n);v.data=vectors.data()+1;v.order=ops::MemoryOrder::f;
                ops::Workspace work;ops::Audit audit;
                ops::symmetric_eigh_split_pair(plan,w,v,work,audit);
                for (std::size_t i=0;i<n;++i) {
                    check(std::memcmp(values.data()+i+1,reference.values.data()+i*(n+1),8)==0,
                          "split Jacobi value changed");
                    check(std::memcmp(vectors.data()+i*n+1,reference.values.data()+i*(n+1)+1,n*8)==0,
                          "split Jacobi vector changed");
                }
                check(input==before && values.front()==-993. && values.back()==-993. &&
                      vectors.front()==-993. && vectors.back()==-993.,"split Jacobi ownership or bounds");
                check(capacity==n*(n+1) && audit.scratch_bytes==work.capacity_bytes(),"split Jacobi scratch budget");
                check(audit.eigen_rotation_calls>0 && audit.algorithm_copy_bytes>plan.scratch_doubles*8,
                      "split Jacobi library/copy audit");
                check(ops::symmetric_eigh_split_scratch_doubles(n+1)==0,"split Jacobi odd shape admission");
            }
        }
        {
            // Independent graph roots may replay one bad library block. Direct
            // API attempts still decline before any observable output write.
            constexpr std::size_t n=131073;
            for (auto op : {ops::Op::exp,ops::Op::log}) {
                std::vector<double> input(n,.5),storage(n+2,-993.),expected(n);
                input.back()=op==ops::Op::exp ? 701. : std::numeric_limits<double>::denorm_min();
                const auto before=input;
                const auto argument=vector(input.data(),n);
                const auto plan=ops::prepare(ops::lookup(op==ops::Op::exp ? "exp" : "log"),&argument,1);
                ops::Output out;out.shape=plan.output_shape;out.data=storage.data()+1;
                ops::Output oracle;oracle.shape=plan.output_shape;oracle.data=expected.data();
                ops::Workspace work;ops::Audit audit;
                ops::execute(plan,oracle,work,ops::Isa::scalar,audit);
                check(!ops::try_accelerate_transcendental(plan,out,work,audit),"direct outlier accepted");
                check(std::all_of(storage.begin(),storage.end(),[](double v){return v==-993.;}),
                      "direct outlier changed its output before canonical replay");
                const bool handled=ops::try_accelerate_transcendental(plan,out,work,audit,true);
                if (handled) {
                    check(audit.accelerate_vforce_calls==2 && audit.vector_elements==n-1,
                          "root block replay lost actual library element counts");
                    for (std::size_t i=0;i<n;++i)
                        check(std::abs(storage[i+1]-expected[i])<=2e-14+2e-12*std::abs(expected[i]),
                              "root block replay changed unary mathematics");
                    check(std::memcmp(storage.data()+n,expected.data()+n-1,8)==0,
                          "outlier did not retain exact scalar mathematics");
                    check(work.capacity_bytes()==0 && audit.algorithm_copy_bytes==0,
                          "root replay allocated a transaction buffer");
                    input.back()=op==ops::Op::exp ? 710. : 0.;
                    bool failed=false;
                    try {ops::try_accelerate_transcendental(plan,out,work,audit,true);}
                    catch(const ops::Error &error) {failed=std::strcmp(error.what(),"DOMAIN_ERROR")==0;}
                    check(failed,"root replay lost canonical domain error");
                    input.back()=before.back();
                    const int rounding=std::fegetround();
                    check(std::fesetround(FE_UPWARD)==0,"cannot set replay rounding control");
                    const bool rejected=!ops::try_accelerate_transcendental(plan,out,work,audit,true);
                    const int restored=std::fesetround(rounding);
                    check(restored==0 && rejected,"root replay admitted non-nearest rounding");
                }
                check(input==before && storage.front()==-993. && storage.back()==-993.,
                      "root replay mutated input or canary storage");
            }
        }
        hmm_checkpoint_native_cases::run(check);
        hmm_checkpoint_native_cases::gaussian_stream(check);
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
