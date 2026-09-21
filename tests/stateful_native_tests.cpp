#include "calmetrics_engine/compiler.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace c = calmetrics_engine::compiler;
namespace g = calmetrics_engine::graph;
namespace o = calmetrics_engine::ops;
namespace t = calmetrics_engine::typed;

namespace {
void check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
o::Value numbers(const std::vector<double> &data) {
    o::Value value;
    value.shape = o::vector_shape(data.size());
    value.data = data.data();
    return value;
}
o::Value integers(const std::vector<std::int64_t> &data) {
    o::Value value;
    value.kind = o::Kind::integer;
    value.shape = o::vector_shape(data.size());
    value.data = data.data();
    return value;
}
o::Value mask(const std::vector<std::uint8_t> &data) {
    o::Value value;
    value.kind = o::Kind::mask;
    value.shape = o::vector_shape(data.size());
    value.data = data.data();
    return value;
}
struct Result {
    o::Shape shape;
    o::Kind kind;
    std::vector<double> floating;
    std::vector<std::int64_t> integer;
    o::Value view() const {
        o::Value value;
        value.shape = shape;
        value.kind = kind;
        value.data = kind == o::Kind::integer ? static_cast<const void *>(integer.data())
                                            : static_cast<const void *>(floating.data());
        value.stride = {shape.rank == 2 ? static_cast<std::ptrdiff_t>(shape.dim[1]) : 1, 1};
        return value;
    }
};
Result call(const char *name, const std::vector<o::Value> &args) {
    const auto plan = o::prepare(o::lookup(name), args.data(), args.size());
    Result result{plan.output_shape, plan.output_kind, {}, {}};
    o::Output output;
    output.shape = plan.output_shape;
    output.kind = plan.output_kind;
    if (output.kind == o::Kind::integer) {
        result.integer.resize(output.shape.size());
        output.data = result.integer.data();
    } else {
        result.floating.resize(output.shape.size());
        output.data = result.floating.data();
    }
    o::Workspace workspace;
    o::Audit audit;
    o::execute(plan, output, workspace, o::Isa::scalar, audit);
    return result;
}
o::Value scalar(double value) { return o::Value::number(value); }
t::ValueType integer_series(const char *semantic = "category") {
    auto type = t::ValueType::series();
    type.dtype = t::DType::int64;
    type.semantic_dimension = semantic;
    return type;
}
} // namespace

int main() {
    try {
        const std::vector<double> values{0, 2, NAN, 1.5, 1, -2, -1.5, -1};
        const auto states = call("state_hysteresis", {numbers(values), scalar(2), scalar(1), scalar(-2), scalar(-1)});
        check(states.integer == std::vector<std::int64_t>({1, 0, -1, 0, 1, 2, 2, 1}), "hysteresis state/gaps");

        const std::vector<std::int64_t> candidate{-1, 1, -1, 1, 1, -1};
        const std::vector<std::int64_t> initial{0, -1, -1, -1, -1, -1};
        const std::vector<double> observed(candidate.size(), 100);
        const auto bundle = call("state_continuous", {integers(candidate), integers(initial), numbers(observed), scalar(2), scalar(3)});
        check(bundle.shape == o::matrix_shape(6, 3), "state shared bundle shape");
        check(bundle.integer == std::vector<std::int64_t>({0,0,0, 0,3,1, 0,0,0, 0,3,1, 1,1,0, 1,2,0}), "state shared fields");
        const auto bundle_view = bundle.view();
        const auto projection = o::prepare(o::lookup("continuous_state_pending"), &bundle_view, 1);
        check(projection.borrowed && projection.view.stride[0] == 3 &&
              projection.view.data == bundle.integer.data() + 2, "borrowed int64 field");
        check(call("continuous_state_values", {bundle.view()}).integer ==
              std::vector<std::int64_t>({0,0,0,0,1,1}), "integer projection explicit copy-out");

        const std::vector<double> prices{1, 3, 1, 4, 2, 5, 2, 6};
        const auto events = call("local_extrema", {numbers(prices), scalar(1), scalar(1), scalar(0), scalar(0)});
        check(events.integer == std::vector<std::int64_t>({0,1,-1,1,-1,1,-1,0}), "peak/trough events");
        const auto selected = call("ps_filter", {numbers(prices), events.view(), scalar(1), scalar(2), scalar(.2)});
        const auto segments = call("between_events", {selected.view()});
        check(segments.shape == o::matrix_shape(8, 2), "segment output capacity");
        const auto phases = call("phase_direction", {selected.view(), segments.view()});
        check(phases.integer == std::vector<std::int64_t>({-1,1,0,1,0,1,-1,-1}), "segment phase half-open");
        auto malformed = segments.integer;
        malformed[3] = 100;
        auto malformed_view = integers(malformed);
        malformed_view.shape = segments.shape;
        malformed_view.stride = {2, 1};
        bool rejected = false;
        try { call("phase_direction", {selected.view(), malformed_view}); }
        catch (const o::Error &error) { rejected = std::string(error.what()) == "INVALID_SEGMENT_BOUNDARY"; }
        check(rejected, "segment geometry rejects unsafe endpoint");
        for (const auto *name : {"segment_starts", "segment_ends"}) {
            const auto &spec = o::lookup(name);
            for (const bool structure_only : {false, true}) {
                rejected = false;
                try {
                    if (structure_only) o::validate_structure(spec, &malformed_view, 1, 1, 1);
                    else o::prepare(spec, &malformed_view, 1);
                } catch (const o::Error &error) {
                    rejected = std::string(error.what()) == "INVALID_SEGMENT_BOUNDARY";
                }
                check(rejected, "segment projections validate available boundary payload");
            }
            auto unavailable = malformed_view;
            unavailable.data = nullptr;
            const auto known = o::validate_structure(spec, &unavailable, 1, 1, 0);
            check(known.geometry_known && known.output_shape == o::vector_shape(8),
                  "unavailable boundary payload retains known output geometry without access");
            const auto unknown = o::validate_structure(spec, &unavailable, 1, 0, 0);
            check(!unknown.geometry_known, "unknown boundary geometry is not invented");
        }

        const std::vector<double> filter_values{1, 3, 5, 7};
        const std::vector<double> alpha{.5, .5, .25, 1};
        const std::vector<std::uint8_t> yes{1,1,1,1}, no{0,0,0,0};
        const auto adaptive = call("recursive_filter_adaptive", {numbers(filter_values), numbers(alpha), mask(yes), mask(no), scalar(0), scalar(2), scalar(1), scalar(0)});
        check(adaptive.floating == std::vector<double>({1,2,2.75,7}), "adaptive coefficient recurrence");
        const auto kalman = call("scalar_kalman", {numbers(filter_values), scalar(.1), scalar(.2), scalar(1)});
        check(kalman.shape == o::matrix_shape(4, 2), "Kalman bundle allocation");
        const auto kalman_view = kalman.view();
        const auto variance_plan = o::prepare(o::lookup("state_variance"), &kalman_view, 1);
        check(variance_plan.borrowed && variance_plan.view.stride[0] == 2 &&
              variance_plan.view.data == kalman.floating.data() + 1, "Kalman borrowed variance");
        check(std::abs(call("state_estimate", {kalman.view()}).floating[1] - (1 + 1.1 / 1.3 * 2)) < 1e-12, "Kalman shared solve projection");

        auto graph = c::compile({"state_hysteresis(x,2,1,-2,-1)"},
                                std::vector<t::Variable>{{"x", t::ValueType::series()}});
        auto decoded = c::decode_program(c::encode_program(graph->program));
        check(decoded.output_dtype == g::OutputDType::int64, "typed output survives worker serialization");
        const std::int64_t start = 0, end = values.size();
        std::vector<std::int64_t> integer_output(values.size());
        g::execute(decoded, {numbers(values)}, nullptr, 0, &start, &end, 1, integer_output.data(), 1);
        check(integer_output == states.integer, "typed graph state output");

        auto continuous_graph = c::compile({
            "continuous_state_values(state_continuous(candidate,initial,price,2,3))",
            "continuous_state_evidence(state_continuous(candidate,initial,price,2,3))",
            "continuous_state_pending(state_continuous(candidate,initial,price,2,3))"},
            std::vector<t::Variable>{{"candidate", integer_series()}, {"initial", integer_series()}, {"price", t::ValueType::series()}});
        const std::int64_t state_end = candidate.size();
        std::vector<std::int64_t> fields(candidate.size() * 3);
        g::execute(continuous_graph->program, {integers(candidate), integers(initial), numbers(observed)}, nullptr, 0, &start, &state_end, 1, fields.data(), 3);
        check(fields == bundle.integer, "CSE bundle views survive arena liveness");

        auto segmented = c::compile({
            "segment_apply(last(price)/first(price)-1,between_events(local_extrema(price,1,1,0,0)))"},
            std::vector<t::Variable>{{"price", t::ValueType::series()}});
        const std::int64_t price_end = prices.size();
        std::vector<double> changes(prices.size());
        g::execute(c::decode_program(c::encode_program(segmented->program)), {numbers(prices)}, nullptr, 0,
                   &start, &price_end, 1, changes.data(), 1);
        check(std::isnan(changes[0]) && std::abs(changes[1] + 2.0/3.0) < 1e-12 &&
              changes[2] == 3 && changes[3] == -.5 && changes[4] == 1.5 &&
              std::isnan(changes[6]), "generic segment endpoint composition");
        // A constant segment body has zero captures, yet its nested block
        // output needs the full endpoint-inclusive interval capacity. ASan
        // previously caught a write past an eight-byte child arena here.
        auto uncaptured = c::compile({
            "segment_apply(sum(block_apply(1,1)),between_events(events))"},
            std::vector<t::Variable>{{"events", integer_series("event")}});
        const std::vector<std::int64_t> uncaptured_events{-1,0,0,0,1};
        const std::int64_t uncaptured_end = 5;
        std::vector<double> uncaptured_output(5);
        g::execute(c::decode_program(c::encode_program(uncaptured->program)),
                   {integers(uncaptured_events)}, nullptr, 0, &start, &uncaptured_end,
                   1, uncaptured_output.data(), 1);
        check(uncaptured_output[0] == 5 && uncaptured_output[3] == 5 &&
              std::isnan(uncaptured_output[4]), "zero-capture nested scope capacity");

        // One execution interval contains both a zero-variance failing segment
        // and a valid segment. Comparisons and classification must not turn a
        // failed segment's NaN into a successful False or ordinary state code.
        const std::vector<double> mixed_segments{2, 2, 2, NAN, 2, 3, 4, 5};
        const std::vector<std::int64_t> mixed_events{-1, 0, 1, -2, -1, 0, 1, 0};
        const std::vector<t::Variable> mixed_variables{
            {"x", t::ValueType::series()}, {"e", integer_series("event")}};
        const std::int64_t mixed_end = mixed_segments.size();
        const std::string risky_segment = "segment_apply(1/std(x,0),between_events(e))";
        auto comparison_graph = c::compile({risky_segment + ">0", "finite_mask(x)"},
                                           mixed_variables, true);
        auto comparison_program = c::decode_program(c::encode_program(comparison_graph->program));
        check(comparison_program.isolate_errors &&
              comparison_program.output_dtype == g::OutputDType::boolean,
              "segment comparison preserves isolate and boolean output contracts");
        std::vector<std::uint8_t> compared(mixed_segments.size() * 2);
        const auto comparison_audit = g::execute(comparison_program,
            {numbers(mixed_segments), integers(mixed_events)}, nullptr, 0,
            &start, &mixed_end, 1, compared.data(), 2);
        check(comparison_audit.statuses.size() == compared.size(),
              "segment comparison status geometry");
        for (const auto row : {std::size_t{0}, std::size_t{1}})
            check(comparison_audit.statuses[row * 2] != 0,
                  "failed segment comparison retains numerical error provenance");
        for (const auto row : {std::size_t{4}, std::size_t{5}})
            check(comparison_audit.statuses[row * 2] == 0 && compared[row * 2] == 1,
                  "valid segment comparison remains successful in same interval");
        for (std::size_t row = 0; row < mixed_segments.size(); ++row)
            check(comparison_audit.statuses[row * 2 + 1] == 0 &&
                  compared[row * 2 + 1] == static_cast<std::uint8_t>(row != 3),
                  "independent boolean root remains valid including legitimate false");

        auto classification_graph = c::compile({
            "state_select(" + risky_segment + ">0,1,2,finite_mask(x))",
            "state_select(x>0,7,8,finite_mask(x))"}, mixed_variables, true);
        auto classification_program = c::decode_program(c::encode_program(classification_graph->program));
        check(classification_program.isolate_errors &&
              classification_program.output_dtype == g::OutputDType::int64,
              "segment classification preserves isolate and exact state output contracts");
        std::vector<std::int64_t> classified(mixed_segments.size() * 2);
        const auto classification_audit = g::execute(classification_program,
            {numbers(mixed_segments), integers(mixed_events)}, nullptr, 0,
            &start, &mixed_end, 1, classified.data(), 2);
        check(classification_audit.statuses.size() == classified.size(),
              "segment classification status geometry");
        for (const auto row : {std::size_t{0}, std::size_t{1}})
            check(classification_audit.statuses[row * 2] != 0,
                  "failed segment state selection retains numerical error provenance");
        for (const auto row : {std::size_t{4}, std::size_t{5}})
            check(classification_audit.statuses[row * 2] == 0 && classified[row * 2] == 1,
                  "valid segment state selection remains successful in same interval");
        for (std::size_t row = 0; row < mixed_segments.size(); ++row)
            check(classification_audit.statuses[row * 2 + 1] == 0 &&
                  classified[row * 2 + 1] == (row == 3 ? -1 : 7),
                  "independent integer root remains valid including explicit unknown");

        // Rolling/group failures originate provenance even without any segment
        // scope. Serialization must reconstruct the same tracing metadata.
        const std::vector<double> ordinary_values{2, 2, 3, 4};
        const std::vector<std::int64_t> ordinary_keys{0, 0, 1, 1};
        const std::vector<t::Variable> ordinary_variables{
            {"x", t::ValueType::series()}, {"key", integer_series("category")}};
        const std::int64_t ordinary_end = 4;
        for (const bool rolling : {true, false}) {
            const auto expression = rolling ? "rolling_apply(1/std(x,0),2)>0"
                                            : "group_apply(1/std(x,0),key)>0";
            const auto compiled = c::compile({expression, "x>0"}, ordinary_variables, true);
            const auto decoded = c::decode_program(c::encode_program(compiled->program));
            std::vector<std::uint8_t> output(8);
            const auto audit = g::execute(decoded, {numbers(ordinary_values), integers(ordinary_keys)},
                nullptr, 0, &start, &ordinary_end, 1, output.data(), 2);
            for (std::size_t row = 0; row < 4; ++row) {
                const bool failed = rolling ? row == 1 : row < 2;
                check((audit.statuses[row * 2] != 0) == failed,
                      "ordinary window/group failure retains exact membership");
                check(audit.statuses[row * 2 + 1] == 0 && output[row * 2 + 1] == 1,
                      "ordinary scope failure preserves independent root");
            }
        }

        // Structural contracts remain fatal even when an upstream numerical
        // failure prevents evaluation. Use one interval so a later healthy
        // interval cannot accidentally be the one that discovers the error.
        const std::vector<double> healthy_segments{1, 2, 3, 4, 5, 6, 7, 8};
        const std::vector<std::uint8_t> valid_mask(mixed_segments.size(), 1);
        auto invalid_mask = valid_mask;
        invalid_mask[1] = 2; // The malformed byte overlaps a failed segment.
        const std::vector<std::int64_t> valid_indices(mixed_segments.size(), 0);
        auto negative_indices = valid_indices;
        negative_indices[1] = -1;
        auto oversized_indices = valid_indices;
        oversized_indices[1] = mixed_end;
        const std::vector<t::Variable> structural_variables{
            {"x", t::ValueType::series()}, {"e", integer_series("event")},
            {"valid", t::ValueType::mask({"time"}, {"T"})},
            {"indices", integer_series("index")}};
        const auto expect_structural_error = [&](
                const std::string &expression, const char *expected,
                const std::vector<std::uint8_t> &checked_mask,
                const std::vector<std::int64_t> &checked_indices,
                const std::map<std::string, std::string> &bindings = {}) {
            const auto compiled = c::compile({expression}, structural_variables, true, {bindings});
            const auto program = c::decode_program(c::encode_program(compiled->program));
            check(program.isolate_errors && program.output_dtype == g::OutputDType::float64,
                  "structural regression uses isolated numeric graph");
            for (const bool source_failure : {false, true}) {
                const auto &source = source_failure ? mixed_segments : healthy_segments;
                std::vector<double> output(source.size());
                std::string actual;
                try {
                    g::execute(program,
                        {numbers(source), integers(mixed_events), mask(checked_mask),
                         integers(checked_indices)}, nullptr, 0, &start, &mixed_end,
                        1, output.data(), 1);
                } catch (const std::exception &error) {
                    actual = error.what();
                }
                if (actual != expected)
                    throw std::runtime_error(std::string(source_failure ? "failed" : "healthy") +
                        " source must throw " + expected + "; received " +
                        (actual.empty() ? "no error" : actual) + " for " + expression);
            }
        };
        const std::string first_recurrence =
            "recursive_filter(" + risky_segment + ",.2,0,finite_mask(x))";
        const std::string second_recurrence =
            "recursive_filter(" + first_recurrence + ",.2,0,finite_mask(x))";
        for (const auto &source : {risky_segment, second_recurrence})
            expect_structural_error("recursive_filter(" + source + ",.2,0,valid)",
                                    "INVALID_MASK", invalid_mask, valid_indices);
        for (const auto &source : {risky_segment, first_recurrence}) {
            expect_structural_error("sum(recursive_filter(lag(" + source +
                                    ",1),.2,0,finite_mask(lag(x,2))))",
                                    "SHAPE_MISMATCH", valid_mask, valid_indices);
            for (const auto *indices : {&negative_indices, &oversized_indices})
                expect_structural_error("sum(gather(" + source + ",indices))",
                                        "INDEX_OUT_OF_BOUNDS", valid_mask, *indices);
        }
        for (const auto &source : {risky_segment, second_recurrence}) {
            const std::map<std::string, std::string> bindings{{"v", source}};
            for (const std::string body : {
                    "solve_x-mean_where(v,valid)",
                    "solve_x-filter_apply(mean_where(v,valid),finite_mask(x))"})
                expect_structural_error("bisect(" + body + ",0,10,1e-12,100)",
                                        "INVALID_MASK", invalid_mask, valid_indices, bindings);
            expect_structural_error(
                "bisect(solve_x-sum(recursive_filter(lag(v,1),.2,0,finite_mask(lag(x,2)))),"
                "0,10,1e-12,100)", "SHAPE_MISMATCH", valid_mask, valid_indices, bindings);
            const auto compiled = c::compile({
                "bisect(solve_x-(2+0*mean_where(v,logical_and(valid,finite_mask(v)))),0,10,1e-12,100)",
                "count_true(finite_mask(x))"},
                structural_variables, true, {bindings, {}});
            const auto program = c::decode_program(c::encode_program(compiled->program));
            for (const bool source_failure : {true, false, true, false}) {
                const auto &input = source_failure ? mixed_segments : healthy_segments;
                double output[2]{};
                const auto audit = g::execute(program,
                    {numbers(input), integers(mixed_events), mask(valid_mask), integers(valid_indices)},
                    nullptr, 0, &start, &mixed_end, 1, output, 2);
                check(audit.statuses[1] == 0 && std::isfinite(output[1]),
                      "bisect failure preserves independent root");
                check(source_failure ? audit.statuses[0] != 0 && std::isnan(output[0])
                                     : audit.statuses[0] == 0 && std::abs(output[0] - 2) < 1e-12,
                      "bisect child status resets between failed and healthy runs");
            }
        }
        const std::map<std::string, std::string> partial_bindings{
            {"n", "1+0*count_true(" + risky_segment + ">0)"}, {"v", "lag(x,n)"}};
        for (const std::string expression : {
                "bisect(solve_x-n-count_true(valid),0,100,1e-8,100)",
                "bisect(solve_x-sum(v)-count_true(valid),0,100,1e-8,100)",
                "bisect(solve_x-count_true(valid),n,100,1e-8,100)",
                "sum(block_apply(count_true(valid)-n,2))",
                "filter_apply(count_true(valid)-n,finite_mask(x))",
                "sum(group_apply(count_true(valid)-n,indices))",
                "sum(segment_apply(count_true(valid)-n,between_events(e)))"})
            expect_structural_error(expression, "INVALID_MASK", invalid_mask, valid_indices, partial_bindings);
        for (const std::string expression : {
                "sum(block_apply(count_true(state_select(v>0,-2,1,finite_mask(v))==1),2))",
                "sum(block_apply(count_true(state_select(x>0,-2,1,finite_mask(x))==1),n))",
                "rolling_apply(count_true(state_select(x>0,-2,1,finite_mask(x))==1),n)"})
            expect_structural_error(expression, "INVALID_STATE_CODE", valid_mask, valid_indices, partial_bindings);
        std::cout << "native state/event/recurrence contracts passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
