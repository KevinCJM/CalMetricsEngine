#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "my_ctools/kernels.hpp"
#include <regex>

namespace py = pybind11;
namespace mc = my_ctools;

namespace {
template <class T>
const T* checked_array(const py::array& array, int dimensions, const char* name) {
    if (!array.dtype().equal(py::dtype::of<T>())) {
        throw py::type_error(std::string(name) + " has an incorrect dtype or byte order");
    }
    if (array.ndim() != dimensions) {
        throw py::value_error(std::string(name) + " has an incorrect number of dimensions");
    }
    if (!(array.flags() & py::array::c_style)) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
    if (reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T) != 0) {
        throw py::value_error(std::string(name) + " must be aligned");
    }
    return static_cast<const T*>(array.data());
}
mc::MatrixView matrix(const py::array& array) {
    const auto* data = checked_array<double>(array, 2, "values");
    return {data, static_cast<std::size_t>(array.shape(0)),
                  static_cast<std::size_t>(array.shape(1))};
}
unsigned checked_threads(int count) {
    if (count < 0 || count > 256) throw py::value_error("n_threads must be between 0 and 256");
    return static_cast<unsigned>(count);
}
const std::int64_t* dates(const py::array& array, std::size_t rows, bool sorted = false) {
    const auto* data = checked_array<std::int64_t>(array, 1, "dates");
    if (static_cast<std::size_t>(array.size()) != rows) {
        throw py::value_error("dates length must equal the number of rows");
    }
    for (std::size_t row = 0; row < rows; ++row) {
        if (data[row] == std::numeric_limits<std::int64_t>::min()) {
            throw py::value_error("dates must not contain NaT");
        }
        if (sorted && row && data[row] < data[row - 1]) {
            throw py::value_error("dates must be sorted in nondecreasing order");
        }
    }
    return data;
}
py::array_t<double> standard_deviation(const py::array& values, int n_threads) {
    const auto view = matrix(values);
    const auto threads = checked_threads(n_threads);
    py::array_t<double> output(static_cast<py::ssize_t>(view.cols));
    auto* result = output.mutable_data();
    { py::gil_scoped_release release; mc::column_statistics(view, result, nullptr, threads); }
    return output;
}
py::array_t<double> mean_standard_deviation(const py::array& values, int n_threads) {
    const auto view = matrix(values);
    const auto threads = checked_threads(n_threads);
    py::array_t<double> output({py::ssize_t(2), static_cast<py::ssize_t>(view.cols)});
    auto* result = output.mutable_data();
    { py::gil_scoped_release release;
      mc::column_statistics(view, result + view.cols, result, threads); }
    return output;
}
py::array_t<double> cpr(const py::array& types, const py::array& values, int n_threads) {
    const auto view = matrix(values);
    const auto* groups = checked_array<std::int32_t>(types, 1, "f_type");
    if (static_cast<std::size_t>(types.size()) != view.cols) {
        throw py::value_error("f_type length must equal the number of columns");
    }
    const auto threads = checked_threads(n_threads);
    py::array_t<double> output(static_cast<py::ssize_t>(view.cols));
    auto* result = output.mutable_data();
    { py::gil_scoped_release release; mc::cpr(view, groups, result, threads); }
    return output;
}
py::array_t<std::int64_t> longest_recovery(const py::array& values, int n_threads) {
    const auto view = matrix(values);
    const auto threads = checked_threads(n_threads);
    py::array_t<std::int64_t> output(static_cast<py::ssize_t>(view.cols));
    auto* result = output.mutable_data();
    { py::gil_scoped_release release; mc::longest_recovery(view, result, threads); }
    return output;
}
py::tuple max_drawdown(const py::array& values, const py::array& date_array, int n_threads) {
    const auto view = matrix(values);
    const auto* days = dates(date_array, view.rows);
    const auto threads = checked_threads(n_threads);
    py::array_t<double> output(static_cast<py::ssize_t>(view.cols));
    py::array_t<std::int64_t> recovery(static_cast<py::ssize_t>(view.cols));
    std::vector<std::string> output_dates(view.cols);
    auto* result = output.mutable_data();
    auto* recovery_data = recovery.mutable_data();
    { py::gil_scoped_release release;
      mc::max_drawdown(view, days, result, output_dates, recovery_data, threads); }
    return py::make_tuple(output, py::cast(output_dates), recovery);
}
py::object streak(const py::array& values, const py::array& date_array,
                  const std::string& code, int n_threads, bool longest) {
    const auto view = matrix(values);
    const auto* days = dates(date_array, view.rows);
    const auto threads = checked_threads(n_threads);
    if (code != "positive" && code != "negative") {
        throw py::value_error("i_code must be 'positive' or 'negative'");
    }
    py::array_t<double> output(static_cast<py::ssize_t>(view.cols));
    py::array_t<std::int64_t> periods(static_cast<py::ssize_t>(view.cols));
    std::vector<std::string> starts(view.cols, "nan"), ends(view.cols, "nan");
    auto* result = output.mutable_data();
    auto* period_data = periods.mutable_data();
    { py::gil_scoped_release release;
      const auto kernel = longest ? mc::longest_streak : mc::largest_streak;
      kernel(view, days, code == "positive", result, period_data, starts, ends, threads); }
    if (longest) return py::make_tuple(output, py::cast(starts), py::cast(ends), periods);
    py::dict result_dict;
    result_dict["r"] = output;
    result_dict["p"] = periods;
    result_dict["s"] = py::cast(starts);
    result_dict["l"] = py::cast(ends);
    return result_dict;
}
int parse_months(const std::string& code) {
    static const std::regex pattern(R"(([0-9]+)([MY]))");
    std::smatch match;
    if (!std::regex_search(code, match, pattern)) {
        throw py::value_error("i_code must contain a positive period such as 1M, 3M or 1Y");
    }
    std::int64_t number;
    try { number = std::stoll(match[1].str()); }
    catch (const std::exception&) { throw py::value_error("i_code period is too large"); }
    const int multiplier = match[2].str() == "Y" ? 12 : 1;
    if (number < 1 || number > std::numeric_limits<int>::max() / multiplier) {
        throw py::value_error("i_code period is out of range");
    }
    return static_cast<int>(number) * multiplier;
}
py::tuple rolling(const std::string& code, const py::array& values,
                  const py::array& start_array, const py::array& end_array,
                  const py::array& date_array, int n_threads) {
    const auto view = matrix(values);
    const auto* days = dates(date_array, view.rows, true);
    const auto* starts = checked_array<std::int64_t>(start_array, 1, "start_idx");
    const auto* ends = checked_array<std::int64_t>(end_array, 1, "end_idx");
    if (static_cast<std::size_t>(start_array.size()) != view.cols ||
        static_cast<std::size_t>(end_array.size()) != view.cols) {
        throw py::value_error("start_idx and end_idx lengths must equal the number of columns");
    }
    for (std::size_t col = 0; col < view.cols; ++col) {
        const bool inactive = starts[col] == -1 && ends[col] == -1;
        const bool empty = view.rows == 0 && starts[col] == 0 && ends[col] == -1;
        if (!inactive && !empty && (starts[col] < 0 || ends[col] < starts[col]
            || ends[col] >= static_cast<std::int64_t>(view.rows))) {
            throw py::value_error("indices must satisfy 0 <= start <= end < rows, or both be -1");
        }
    }
    const auto threads = checked_threads(n_threads);
    const auto months = parse_months(code);
    std::array<py::array_t<double>, 9> outputs;
    std::array<double*, 9> pointers{};
    for (std::size_t k = 0; k < outputs.size(); ++k) {
        outputs[k] = py::array_t<double>(static_cast<py::ssize_t>(view.cols));
        pointers[k] = outputs[k].mutable_data();
    }
    { py::gil_scoped_release release;
      mc::rolling_gain(view, starts, ends, days, months, pointers, threads); }
    py::tuple result(outputs.size());
    for (std::size_t k = 0; k < outputs.size(); ++k) result[k] = outputs[k];
    return result;
}
}  // namespace

PYBIND11_MODULE(_core, module) {
    module.doc() = "Portable AOT numerical kernels; strict native NumPy array inputs.";
    module.attr("__version__") = MY_CTOOLS_VERSION;
    module.def("build_info", [] {
        py::dict info;
        info["version"] = MY_CTOOLS_VERSION;
        info["compiler"] = MY_CTOOLS_COMPILER;
        info["architecture"] = MY_CTOOLS_ARCH;
        info["cpu_policy"] = "baseline";
        info["thread_backend"] = "std::thread";
        info["cxx_standard"] = 17;
        return info;
    });
    module.def("cal_std_mean", &standard_deviation,
               py::arg("input").noconvert(), py::kw_only(), py::arg("n_threads") = 1);
    module.def("cal_std_mean_simd", &mean_standard_deviation,
               py::arg("input").noconvert(), py::kw_only(), py::arg("n_threads") = 1);
    module.def("cal_cpr", &cpr, py::arg("f_type").noconvert(),
               py::arg("funds_value").noconvert(), py::kw_only(), py::arg("n_threads") = 1);
    module.def("cal_longest_dd_recover", &longest_recovery,
               py::arg("funds_val").noconvert(), py::kw_only(), py::arg("n_threads") = 1);
    module.def("cal_max_dd", &max_drawdown, py::arg("funds_val").noconvert(),
               py::arg("day_arr").noconvert(), py::kw_only(), py::arg("n_threads") = 1);
    module.def("cal_all_largest_indicators", [](const py::array& v, const py::array& d,
               const std::string& mode, int threads) { return streak(v, d, mode, threads, false); },
               py::arg("array_value").noconvert(), py::arg("dates").noconvert(),
               py::arg("i_code") = "positive", py::kw_only(), py::arg("n_threads") = 1);
    module.def("cal_all_longest_indicators", [](const py::array& v, const py::array& d,
               const std::string& mode, int threads) { return streak(v, d, mode, threads, true); },
               py::arg("a_value").noconvert(), py::arg("dates").noconvert(),
               py::arg("i_code") = "positive", py::kw_only(), py::arg("n_threads") = 1);
    module.def("cal_rolling_gain_loss", &rolling, py::arg("i_code"),
               py::arg("funds_val").noconvert(), py::arg("start_idx").noconvert(),
               py::arg("end_idx").noconvert(), py::arg("day_arr").noconvert(),
               py::kw_only(), py::arg("n_threads") = 1);
}
