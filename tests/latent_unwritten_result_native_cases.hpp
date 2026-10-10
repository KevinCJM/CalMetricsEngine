#pragma once

// Caller-owned native buffers must agree even when later roots never execute.
inline void latent_unwritten_result_native_contracts() {
  constexpr std::size_t rows = 350000, dimensions = 4, states = 3;
  constexpr std::uint64_t sentinel = 0x5a7f31be19d04c62ULL;
  for (bool hmm : {false, true}) for (int scenario = 0; scenario < 4; ++scenario) {
    std::vector<double> x(rows * dimensions), bad(rows * states, 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::sin(double(i) * 0.17);
    if (scenario == 1 || scenario == 3) bad[0] = std::numeric_limits<double>::quiet_NaN();
    if (scenario == 2) x.back() = std::numeric_limits<double>::quiet_NaN();
    const auto original_x = x, original_bad = bad;
    std::vector<double> weights(states, 1.0 / states), means(states * dimensions, 0.1),
        variances(states * dimensions, 1.0), transition(states * states, 1.0 / states);
    std::vector<std::int64_t> positions(rows);
    for (std::size_t i = 0; i < rows; ++i) positions[i] = i;
    const auto value = [](const double *data, o::Shape shape) {
      o::Value result; result.kind = o::Kind::number; result.data = data;
      result.shape = shape; result.set_contiguous_strides(); return result;
    };
    o::Value index; index.kind = o::Kind::integer; index.data = positions.data();
    index.shape = o::vector_shape(rows); index.set_contiguous_strides();
    const std::vector<o::Value> inputs{value(x.data(), o::matrix_shape(rows, dimensions)),
        value(weights.data(), o::vector_shape(states)), value(means.data(), o::matrix_shape(states, dimensions)),
        value(variances.data(), o::matrix_shape(states, dimensions)), index,
        value(bad.data(), o::matrix_shape(rows, states)), value(transition.data(), o::matrix_shape(states, states))};
    const std::string prefix = hmm ? "hmm" : "mixture";
    const std::string fit = hmm ? "gaussian_hmm_em(x,weights,transition,means,variances,2,0)" :
                                 "gaussian_mixture_em(x,weights,means,variances,2,0)";
    const std::string density = "gaussian_log_density(x," + prefix + "_means(" + fit + ")," + prefix + "_variances(" + fit + "))";
    const std::string posterior = hmm ? "hmm_filter(" + density + ",hmm_initial(" + fit + "),hmm_transition(" + fit + "))" :
        "softmax_axis(" + density + "+broadcast_like(log(maximum(mixture_weights(" + fit + "),1e-12))," + density + "),1)";
    const std::string future = "scatter_rows(" + posterior + ",positions," + std::to_string(rows) + ",0)";
    const auto roots = scenario == 3 ? std::vector<std::string>{prefix + "_means(" + fit + ")", future, "argmax_axis(bad,1)"} :
                                      std::vector<std::string>{prefix + "_means(" + fit + ")", "argmax_axis(bad,1)", future};
    auto weights_type = t::ValueType::vector("3"); weights_type.axes = {"component"};
    auto index_type = t::ValueType::series("T", "index"); index_type.dtype = t::DType::int64;
    const auto compiled = c::compile(roots, std::vector<t::Variable>{
        {"x",t::ValueType::matrix({"time","factor"},{"T","4"})}, {"weights",weights_type},
        {"means",t::ValueType::matrix({"component","factor"},{"3","4"})},
        {"variances",t::ValueType::matrix({"component","factor"},{"3","4"})}, {"positions",index_type},
        {"bad",t::ValueType::matrix({"time","component"},{"T","3"})},
        {"transition",t::ValueType::matrix({"component","component"},{"3","3"})}},
        false, {}, {}, 0, 100000000, true);
    const std::int64_t start = 0, end = rows;
    const auto layout = g::result_layout(compiled->program, inputs, &start, &end, 1);
    std::vector<std::uint64_t> expected((layout.bytes()+7)/8+2,sentinel), actual(expected.size(),sentinel);
    std::string canonical_error, loan_error;
    g::Audit canonical_audit, loan_audit;
    try { canonical_audit = g::execute(compiled->program, inputs, nullptr, 0, &start, &end, 1,
                                      expected.data()+1, 3, &layout, 0, n::Deadline::max(), nullptr, false); }
    catch (const std::exception &error) { canonical_error = error.what(); }
    try { loan_audit = g::execute(compiled->program, inputs, nullptr, 0, &start, &end, 1,
                                 actual.data()+1, 3, &layout, 0, n::Deadline::max(), nullptr, true); }
    catch (const std::exception &error) { loan_error = error.what(); }
    check(canonical_error == loan_error, "latent loan changed its original error");
    check(actual == expected, "latent loan exposed workspace in an unfinished root or erased a finished root");
    check(actual.front() == sentinel && actual.back() == sentinel, "latent loan overwrote a canary");
    check(std::memcmp(x.data(),original_x.data(),x.size()*sizeof(double)) == 0 &&
          std::memcmp(bad.data(),original_bad.data(),bad.size()*sizeof(double)) == 0,
          "latent loan changed an input");
    if (scenario == 0) {
      check(canonical_error.empty(), "healthy latent loan unexpectedly failed");
      check(loan_audit.operator_workspace_capacity_bytes < canonical_audit.operator_workspace_capacity_bytes,
            "latent loan did not reduce actual workspace reservation");
    } else check(!canonical_error.empty(), "latent failure fixture did not reach an error");
  }
}
