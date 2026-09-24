#include "calmetrics_engine/excel.hpp"
#include "graph_binding_utils.hpp"
#include <cmath>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace e = calmetrics_engine::excel;
namespace o = calmetrics_engine::ops;
namespace c = calmetrics_engine::compiler;
namespace b = calmetrics_engine::binding;
namespace {
e::Snapshot snapshot(py::handle item, std::size_t &remaining,
                     std::size_t &items, o::Kind record = o::Kind::number) {
  auto consume = [&](std::size_t n, std::size_t bytes) {
    if (n > items || bytes > remaining)
      throw e::Error("OVER_BUDGET: input cells/snapshot before allocation");
    items -= n;
    remaining -= bytes;
  };
  e::Snapshot s;
  if (record == o::Kind::fit || record == o::Kind::interval) {
    auto r = py::cast<std::vector<double>>(item);
    if (r.size() != (record == o::Kind::fit ? 5u : 4u))
      throw py::value_error("record field count mismatch");
    consume(r.size(), r.size() * 8);
    s.value.kind = record;
    std::copy(r.begin(), r.end(), s.value.record.begin());
    return s;
  }
  const auto numpy = py::module_::import("numpy");
  if (py::isinstance<py::bool_>(item) ||
      py::isinstance(item, numpy.attr("bool_")) ||
      py::isinstance(item, numpy.attr("uint8"))) {
    consume(1, 1);
    s.value.kind = o::Kind::mask;
    s.value.scalar = py::cast<double>(item);
    if (s.value.scalar != 0 && s.value.scalar != 1)
      throw py::value_error("INVALID_MASK");
    return s;
  }
  if (py::isinstance<py::float_>(item) || py::isinstance<py::int_>(item)) {
    consume(1, 8);
    auto v = py::cast<double>(item);
    if (py::isinstance<py::int_>(item) && std::abs(v) > 999999999999999.0)
      throw e::Error("UNREPRESENTABLE_VALUE: integer scalar");
    s.value = o::Value::number(v);
    return s;
  }
  auto array = b::require_array(item);
  b::validate_owner(array);
  b::bounds(array);
  if (array.ndim() > 3)
    throw py::value_error("input rank must be 0..3");
  auto &v = s.value;
  v.shape.rank = static_cast<int>(array.ndim());
  for (int j = 0; j < v.shape.rank; ++j) {
    v.shape.dim[j] = static_cast<std::size_t>(array.shape(j));
    if (array.strides(j) % array.itemsize())
      throw py::value_error("unaligned stride");
    v.stride[j] = array.strides(j) / array.itemsize();
  }
  if (array.dtype().equal(py::dtype::of<double>()))
    v.kind = o::Kind::number;
  else if (array.dtype().equal(py::dtype::of<std::int64_t>()))
    v.kind = o::Kind::integer;
  else if (array.dtype().equal(py::dtype::of<bool>()) ||
           array.dtype().equal(py::dtype::of<std::uint8_t>()))
    v.kind = o::Kind::mask;
  else
    throw py::type_error(
        "Excel inputs require exact float64, int64, bool or uint8 dtype");
  if (reinterpret_cast<std::uintptr_t>(array.data()) % array.itemsize())
    throw py::value_error("unaligned input");
  const auto size = v.size(),
             itemsize = static_cast<std::size_t>(array.itemsize());
  if (size > remaining / itemsize)
    throw e::Error("OVER_BUDGET: input snapshot before allocation");
  consume(size, size * itemsize);
  if (v.kind == o::Kind::number)
    s.numbers.reserve(size);
  else if (v.kind == o::Kind::integer)
    s.integers.reserve(size);
  else
    s.masks.reserve(size);
  v.data = array.data();
  if (!v.shape.rank) {
    if (v.kind == o::Kind::number)
      v.scalar = *static_cast<const double *>(v.data);
    else if (v.kind == o::Kind::integer)
      v.integer = *static_cast<const std::int64_t *>(v.data);
    else
      v.scalar = *static_cast<const std::uint8_t *>(v.data);
  } else
    for (std::size_t i = 0; i < size; ++i) {
      if (v.kind == o::Kind::number)
        s.numbers.push_back(v.f(i));
      else if (v.kind == o::Kind::integer)
        s.integers.push_back(v.i(i));
      else {
        auto mask = v.u(i);
        if (mask > 1)
          throw py::value_error("INVALID_MASK");
        s.masks.push_back(mask);
      }
    }
  if (v.kind == o::Kind::mask && !v.shape.rank && v.scalar != 0 &&
      v.scalar != 1)
    throw py::value_error("INVALID_MASK");
  v.data = nullptr;
  return s;
}
py::list shape(const o::Shape &s) {
  py::list r;
  for (int i = 0; i < s.rank; ++i)
    r.append(s.dim[i]);
  return r;
}
} // namespace
void register_excel(py::module_ &native) {
  auto m =
      native.def_submodule("excel", "On-demand native Excel formula compiler");
  py::register_exception<e::Error>(m, "ExcelExportError", PyExc_ValueError);
  py::class_<e::Plan, std::shared_ptr<e::Plan>>(m, "ExcelPlan")
      .def_property_readonly("identity", &e::Plan::identity)
      .def("cancel", &e::Plan::cancel)
      .def("check_cancelled", &e::Plan::check_cancelled)
      .def("metadata",
           [](const e::Plan &p) {
             py::dict d;
             auto &r = p.report();
             d["identity"] = p.identity();
             d["engine_build_id"] = p.build_identity();
             d["formula_version"] = e::version;
             d["status"] = "READY";
             d["verification"] = "NOT_RECALCULATED";
             d["cells"] = r.cells;
             d["formula_characters"] = r.formula_characters;
             d["snapshot_bytes"] = r.snapshot_bytes;
             d["estimated_work"] = r.estimated_work;
             const auto &bound = p.budget();
             py::dict budget;
             budget["method"] = "symbolic-shape-upper-bound-v1";
             budget["cells"] = bound.cells;
             budget["formula_cells"] = bound.formula_cells;
             budget["formula_characters"] = bound.formula_characters;
             budget["snapshot_bytes"] = bound.snapshot_bytes;
             budget["work_units"] = bound.work_units;
             py::list largest;
             for (const auto &node : bound.largest_nodes) {
               py::dict item;
               item["node"] = node.node;
               item["kind"] = node.kind;
               item["cells"] = node.cells;
               largest.append(item);
             }
             budget["largest_nodes"] = largest;
             d["upper_bound"] = budget;
             d["timeout_seconds"] = p.limits().timeout_seconds;
             d["atol"] = p.limits().atol;
             d["rtol"] = p.limits().rtol;
             d["max_stream_bytes"] = 512 * 1024 * 1024;
             d["sheets"] = r.sheets;
             py::list outputs;
             for (auto &out : r.outputs) {
               py::dict v;
               v["name"] = out.name;
               v["dtype"] = out.dtype;
               v["shape"] = shape(out.shape);
               v["addresses"] = out.addresses;
               v["axes"] = out.axes;
               v["fields"] = out.fields;
               outputs.append(v);
             }
             d["outputs"] = outputs;
             return d;
           })
      .def("write_cells", &e::Plan::write, py::arg("path"),
           py::arg("sheet_prefix") = "",
           py::call_guard<py::gil_scoped_release>())
      .def("reference", [](const e::Plan &p) {
        std::vector<e::Reference> result;
        {
          py::gil_scoped_release release;
          result = p.reference();
        }
        py::list out;
        for (auto &r : result) {
          py::dict d;
          d["shape"] = shape(r.shape);
          d["error"] = r.error;
          d["statuses"] = r.statuses;
          d["values"] = r.kind == o::Kind::mask || r.kind == o::Kind::integer
                            ? py::cast(r.integers)
                            : py::cast(r.numbers);
          out.append(d);
        }
        return out;
      });
  m.def(
      "plan",
      [](const c::CompiledGraph &g, const py::dict &inputs,
         const py::dict &params, std::size_t cells, std::size_t chars,
         std::size_t bytes, double atol, double rtol, double timeout) {
        e::Limits limits{cells, chars, bytes};
        limits.atol = atol;
        limits.rtol = rtol;
        limits.timeout_seconds = timeout;
        std::size_t remaining = bytes, items = cells / 4;
        if (cells == 0 || cells > 2000000)
          throw e::Error("INVALID_INPUT: max_cells must be 1..2000000");
        std::vector<e::Snapshot> in;
        std::vector<double> p;
        if (inputs.size() != g.input_names.size() ||
            params.size() != g.parameter_names.size())
          throw py::value_error(
              "graph input/parameter names must match exactly");
        std::map<std::string, std::size_t> dimensions;
        for (auto &name : g.input_names) {
          if (!inputs.contains(py::str(name)))
            throw py::value_error("missing input " + name);
          auto declared =
              std::find_if(g.variable_types.begin(), g.variable_types.end(),
                           [&](const auto &v) { return v.name == name; });
          if (declared == g.variable_types.end())
            throw py::value_error("missing typed declaration");
          auto array = b::require_array(inputs[py::str(name)]);
          const auto &type = declared->type;
          if (array.ndim() != static_cast<int>(type.rank()))
            throw py::value_error("input rank differs from typed declaration");
          bool exact =
              type.dtype == calmetrics_engine::typed::DType::float64
                  ? array.dtype().equal(py::dtype::of<double>())
              : type.dtype == calmetrics_engine::typed::DType::int64
                  ? array.dtype().equal(py::dtype::of<std::int64_t>())
                  : array.dtype().equal(py::dtype::of<bool>()) ||
                        array.dtype().equal(py::dtype::of<std::uint8_t>());
          if (!exact)
            throw py::type_error("input dtype differs from typed declaration");
          for (int axis = 0; axis < array.ndim(); ++axis) {
            const auto &symbol = type.shape.at(axis);
            auto size = static_cast<std::size_t>(array.shape(axis));
            if (!symbol.empty() &&
                std::all_of(symbol.begin(), symbol.end(),
                            [](char c) { return c >= '0' && c <= '9'; })) {
              if (std::stoull(symbol) != size)
                throw py::value_error("fixed dimension mismatch");
            } else {
              auto [it, inserted] = dimensions.emplace(symbol, size);
              if (!inserted && it->second != size)
                throw py::value_error("symbolic dimension mismatch");
            }
          }
          in.push_back(snapshot(array, remaining, items));
        }
        for (auto &name : g.parameter_names) {
          if (!params.contains(py::str(name)))
            throw py::value_error("missing parameter " + name);
          p.push_back(py::cast<double>(params[py::str(name)]));
        }
        py::gil_scoped_release release;
        return std::make_shared<e::Plan>(g, std::move(in), std::move(p), limits,
                                         CALMETRICS_ENGINE_BUILD_ID);
      },
      py::arg("graph"), py::arg("inputs"), py::arg("parameters") = py::dict(),
      py::kw_only(), py::arg("max_cells") = 2000000,
      py::arg("max_formula_characters") = 128 * 1024 * 1024,
      py::arg("max_snapshot_bytes") = 64 * 1024 * 1024, py::arg("atol") = 1e-12,
      py::arg("rtol") = 1e-10, py::arg("timeout_seconds") = 180.0);
  m.def(
      "plan_operator",
      [](const std::string &name, const py::sequence &args, std::size_t cells,
         double atol, double rtol, double timeout) {
        auto &spec = o::lookup(name);
        e::Limits limits;
        limits.cells = cells;
        limits.atol = atol;
        limits.rtol = rtol;
        limits.timeout_seconds = timeout;
        std::size_t remaining = limits.snapshot_bytes, items = cells / 4;
        if (cells == 0 || cells > 2000000)
          throw e::Error("INVALID_INPUT: max_cells must be 1..2000000");
        std::vector<e::Snapshot> in;
        for (std::size_t i = 0; i < args.size(); ++i) {
          auto record = i == 0 && spec.op >= o::Op::fit_slope &&
                                spec.op <= o::Op::fit_observation_count
                            ? o::Kind::fit
                        : i == 0 && spec.op >= o::Op::interval_start &&
                                spec.op <= o::Op::interval_recovery
                            ? o::Kind::interval
                            : o::Kind::number;
          in.push_back(snapshot(args[i], remaining, items, record));
        }
        py::gil_scoped_release release;
        return std::make_shared<e::Plan>(spec.op, std::move(in), limits,
                                         CALMETRICS_ENGINE_BUILD_ID);
      },
      py::arg("name"), py::arg("arguments"), py::kw_only(),
      py::arg("max_cells") = 2000000, py::arg("atol") = 1e-12,
      py::arg("rtol") = 1e-10, py::arg("timeout_seconds") = 180.0);
  m.def("coverage", &e::coverage);
}
