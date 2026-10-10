#pragma once

// Real native result buffers expose unpublished-root bytes on failure.
inline void state_root_loan_native_contracts() {
  constexpr std::size_t rows = 262145, columns = 3;
  constexpr std::uint64_t sentinel = 0x5a7f31be19d04c62ULL;
  const auto expression = [](bool bad_state, bool bad_threshold) {
    std::string result = "-1";
    for (std::size_t state = 0; state < columns; ++state) {
      const auto threshold = bad_threshold && state == 1 ? "1.5" : std::to_string(state);
      const auto code = bad_state && state == 1 ? "-2" : std::to_string((state + 2) % columns);
      result = "state_select(argmax_axis(matrix,1)==" + threshold + "," + code + "," + result + ",valid)";
    }
    return result;
  };
  for (int scenario = 0; scenario < 8; ++scenario) {
    std::vector<double> storage(rows * columns, 0.0);
    o::Value matrix; matrix.kind = o::Kind::number; matrix.shape = o::matrix_shape(rows, columns);
    matrix.data = storage.data(); matrix.set_contiguous_strides(scenario == 1 ? o::MemoryOrder::f : o::MemoryOrder::c);
    if (scenario == 2) { matrix.data = storage.data() + (rows - 1) * columns; matrix.stride[0] = -static_cast<std::ptrdiff_t>(columns); }
    auto *data = const_cast<double *>(static_cast<const double *>(matrix.data));
    for (std::size_t row = 0; row < rows; ++row) data[matrix.offset(row * columns + 2)] = 1.0;
    if (scenario >= 3 && scenario <= 5) {
      const auto row = scenario == 3 ? 0 : scenario == 4 ? 4096 : rows - 1;
      data[matrix.offset(row * columns + 1)] = scenario == 5 ?
          std::numeric_limits<double>::infinity() : std::numeric_limits<double>::quiet_NaN();
    }
    const auto original = storage;
    std::vector<std::uint8_t> validity(rows, 1);
    o::Value valid; valid.kind = o::Kind::mask; valid.shape = o::vector_shape(rows);
    valid.data = validity.data(); valid.set_contiguous_strides();
    const bool bad_state = scenario == 6, cancelled = scenario == 7;
    std::vector<std::string> roots;
    if (bad_state) roots.push_back("sum_axis(matrix,0)");
    roots.push_back(expression(bad_state, false));
    auto compiled = c::compile(roots, std::vector<t::Variable>{
        {"matrix", t::ValueType::matrix()}, {"valid", t::ValueType::mask({"time"}, {"T"})}},
        false, {}, {}, 0, 100000000, true);
    const std::int64_t start = 0, end = rows;
    const std::vector<o::Value> inputs{matrix, valid};
    const auto layout = g::result_layout(compiled->program, inputs, &start, &end, 1);
    std::vector<std::uint64_t> expected((layout.bytes() + 7) / 8 + 2, sentinel), actual(expected.size(), sentinel);
    std::string canonical_error, loan_error;
    std::atomic<bool> stop{cancelled};
    try { g::execute(compiled->program, inputs, nullptr, 0, &start, &end, 1, expected.data() + 1,
                     roots.size(), &layout, 0, n::Deadline::max(), &stop, false); }
    catch (const std::exception &error) { canonical_error = error.what(); }
    g::Audit audit;
    try { audit = g::execute(compiled->program, inputs, nullptr, 0, &start, &end, 1, actual.data() + 1,
                            roots.size(), &layout, 0, n::Deadline::max(), &stop, true); }
    catch (const std::exception &error) { loan_error = error.what(); }
    check(canonical_error == loan_error, "state root loan changed original error");
    check(actual == expected, "state root loan changed native failure bytes or result");
    check(actual.front() == sentinel && actual.back() == sentinel, "state root loan overwrote canary");
    check(std::memcmp(storage.data(), original.data(), storage.size() * sizeof(double)) == 0,
          "state root loan mutated original input");
    if (scenario < 3) {
      check(canonical_error.empty() && audit.numeric_arena_bytes == 0 && audit.algorithm_copy_bytes == rows * 8,
            "state root loan did not omit private labels");
      const auto state_root = roots.size() - 1;
      const auto *values = reinterpret_cast<const std::int64_t *>(
          reinterpret_cast<const std::uint8_t *>(actual.data() + 1) + layout.slots[state_root].byte_offset);
      check(std::all_of(values, values + rows, [](auto state) { return state == 1; }),
            "state root loan reread overwritten labels");
    } else check(!canonical_error.empty(), "state root loan failure case did not fail");
  }
}
