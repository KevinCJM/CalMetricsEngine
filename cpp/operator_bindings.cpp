#include "calmetrics_engine/operators.hpp"
#include "calmetrics_engine/scheduler.hpp"
#include "graph_binding_utils.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>

namespace py = pybind11;
namespace op = calmetrics_engine::ops;
namespace binding = calmetrics_engine::binding;
namespace {
struct Handle {
  const op::Spec *spec;
};
op::Value array_value(const py::array &a) {
  op::Value v;
  if (a.dtype().equal(py::dtype::of<double>()))
    v.kind = op::Kind::number;
  else if (a.dtype().equal(py::dtype::of<std::int64_t>()))
    v.kind = op::Kind::integer;
  else if (a.dtype().equal(py::dtype::of<std::uint8_t>()) ||
           a.dtype().equal(py::dtype::of<bool>()))
    v.kind = op::Kind::mask;
  else
    throw py::type_error("Exact native float64, int64 or bool/uint8 ndarray required; "
                         "no implicit conversion");
  op::require(a.ndim() >= 0 && a.ndim() <= 2, "RANK_MISMATCH");
  const auto item =
      v.kind == op::Kind::mask ? sizeof(std::uint8_t) : sizeof(double);
  op::require(reinterpret_cast<std::uintptr_t>(a.data()) % item == 0,
              "UNALIGNED_INPUT");
  v.shape.rank = static_cast<int>(a.ndim());
  v.data = a.data();
  for (int axis = 0; axis < v.shape.rank; ++axis) {
    op::require(a.strides(axis) % static_cast<py::ssize_t>(item) == 0,
                "UNSUPPORTED_BYTE_STRIDE");
    v.shape.dim[axis] = static_cast<std::size_t>(a.shape(axis));
    v.stride[axis] = a.strides(axis) / static_cast<py::ssize_t>(item);
  }
  binding::bounds(a); // Check arithmetic before constructing native views.
  binding::validate_owner(a);
  if (v.shape.rank == 0 && v.kind == op::Kind::integer)
    v.integer = *static_cast<const std::int64_t *>(a.data());
  else if (v.shape.rank == 0)
    v.scalar = v.kind == op::Kind::number
                   ? *static_cast<const double *>(a.data())
                   : *static_cast<const std::uint8_t *>(a.data());
  return v;
}
op::Value parse_value(py::handle object) {
  if (py::isinstance<py::array>(object))
    return array_value(py::reinterpret_borrow<py::array>(object));
  op::Value v;
  if (py::isinstance<py::tuple>(object)) {
    const auto tuple = py::reinterpret_borrow<py::tuple>(object);
    op::require(tuple.size() == 4 || tuple.size() == 5, "RECORD_TYPE_MISMATCH");
    v.kind = tuple.size() == 4 ? op::Kind::interval : op::Kind::fit;
    for (std::size_t i = 0; i < tuple.size(); ++i)
      v.record[i] = py::cast<double>(tuple[i]);
    return v;
  }
  if (py::isinstance<py::bool_>(object)) {
    v.kind = op::Kind::mask;
    v.scalar = py::cast<bool>(object);
    return v;
  }
  if (py::isinstance<py::int_>(object)) {
    const double value = py::cast<double>(object);
    if (std::abs(value) > 9007199254740992.0)
      throw py::value_error(
          "Integer scalar is not guaranteed exact in float64");
    v.scalar = value;
    return v;
  }
  if (py::isinstance<py::float_>(object)) {
    v.scalar = py::cast<double>(object);
    return v;
  }
  // Exact NumPy scalar support without ndarray materialization or Python loops.
  const auto numpy = py::module_::import("numpy");
  if (py::isinstance(object, numpy.attr("float64"))) {
    v.scalar = py::cast<double>(object);
    return v;
  }
  if (py::isinstance(object, numpy.attr("bool_")) ||
      py::isinstance(object, numpy.attr("uint8"))) {
    v.kind = op::Kind::mask;
    v.scalar = py::cast<double>(object);
    return v;
  }
  if (py::isinstance(object, numpy.attr("integer"))) {
    v.scalar = py::cast<double>(object);
    op::require(std::abs(v.scalar) <= 9007199254740992.0,
                "INEXACT_INTEGER_SCALAR");
    return v;
  }
  throw py::type_error("Expected exact-dtype ndarray, numeric scalar, mask or "
                       "typed state tuple; "
                       "lists are not converted");
}

struct Bound {
  std::array<py::object, 8> owners;
  std::array<op::Value, 8> args;
  std::array<bool, 8> provided{};
  std::size_t count = 0;
  py::object out = py::none(), workspace = py::none();
  op::Isa isa = op::Isa::automatic;
  bool audit = false;
};
Bound bind_args(const op::Spec &spec, py::args args, py::kwargs kwargs) {
  Bound b;
  for (auto &owner : b.owners)
    owner = py::none();
  op::require(args.size() <= spec.max_args, "ARITY_MISMATCH");
  for (std::size_t i = 0; i < args.size(); ++i) {
    b.owners[i] = py::reinterpret_borrow<py::object>(args[i]);
    b.provided[i] = true;
  }
  b.count = args.size();
  auto names = op::parameter_names(spec, spec.max_args);
  // One-input fit/covariance overloads have different public argument names.
  if (spec.min_args == 1 && spec.max_args == 2 &&
      (spec.family == op::Family::regression || spec.op == op::Op::covariance ||
       spec.op == op::Op::correlation)) {
    const auto unary_names = op::parameter_names(spec, 1);
    if (kwargs.contains(py::str(unary_names[0])))
      names = unary_names;
  }
  for (auto item : kwargs) {
    const auto name = py::cast<std::string>(item.first);
    const auto value = py::reinterpret_borrow<py::object>(item.second);
    if (name == "out")
      b.out = value;
    else if (name == "workspace")
      b.workspace = value;
    else if (name == "simd")
      b.isa = op::parse_isa(py::cast<std::string>(value));
    else if (name == "audit") {
      if (!py::isinstance<py::bool_>(value))
        throw py::type_error("audit must be bool");
      b.audit = py::cast<bool>(value);
    } else {
      const auto found = std::find(names.begin(), names.end(), name);
      if (found == names.end())
        throw py::type_error("Unknown operator parameter: " + name);
      const auto index = static_cast<std::size_t>(found - names.begin());
      if (b.provided[index])
        throw py::type_error("Multiple values for parameter: " + name);
      b.owners[index] = value;
      b.provided[index] = true;
      b.count = std::max(b.count, index + 1);
    }
  }
  for (std::size_t i = 0; i < b.count; ++i) {
    if (!b.provided[i]) {
      if ((spec.op == op::Op::rolling_std && i == 2) ||
          (spec.op == op::Op::recursive_filter && i == 4))
        b.owners[i] = py::float_(0.0);
      else if (spec.op == op::Op::aligned_shift && i == 1)
        b.owners[i] = py::float_(1.0);
      else if (spec.op == op::Op::recursive_filter_adaptive && i >= 4)
        b.owners[i] = py::float_(i == 4 ? 0.0 : 1.0);
      else if ((spec.op == op::Op::linear_filter2 && i >= 6) ||
               (spec.op == op::Op::scalar_kalman && i == 3))
        b.owners[i] = py::float_(1.0);
      else
        throw py::type_error("Missing operator parameter: " + names[i]);
    }
    b.args[i] = parse_value(b.owners[i]);
    if (b.args[i].kind == op::Kind::integer && spec.op != op::Op::argsort &&
        spec.op != op::Op::gather && spec.op != op::Op::distinct_count &&
        spec.op != op::Op::equal && spec.op != op::Op::not_equal &&
        spec.op != op::Op::state_confirm && spec.op != op::Op::state_continuous &&
        spec.op != op::Op::continuous_state_values && spec.op != op::Op::continuous_state_evidence &&
        spec.op != op::Op::continuous_state_pending && spec.op != op::Op::ps_filter &&
        spec.op != op::Op::between_events && spec.op != op::Op::segment_starts &&
        spec.op != op::Op::segment_ends && spec.op != op::Op::phase_direction &&
        spec.op != op::Op::drawdown_cycle_reference && spec.op != op::Op::state_select)
      throw py::type_error("operator requires exact float64; int64 is reserved for declared index/category inputs");
  }
  return b;
}

std::vector<py::ssize_t> dimensions(const op::Shape &shape) {
  std::vector<py::ssize_t> result;
  for (int i = 0; i < shape.rank; ++i)
    result.push_back(static_cast<py::ssize_t>(shape.dim[i]));
  return result;
}
py::dtype dtype(op::Kind k) {
  if (k == op::Kind::integer) return py::dtype::of<std::int64_t>();
  return k == op::Kind::mask ? py::dtype::of<std::uint8_t>()
                             : py::dtype::of<double>();
}
py::array allocate_output(const op::Prepared &p) {
  return py::array(dtype(p.output_kind), dimensions(p.output_shape));
}
py::array checked_output(const Bound &b, const op::Prepared &p) {
  if (!py::isinstance<py::array>(b.out))
    throw py::type_error("out must be ndarray");
  auto a = py::reinterpret_borrow<py::array>(b.out);
  const auto value = array_value(a);
  op::require(value.kind == p.output_kind && value.shape == p.output_shape,
              "OUTPUT_MISMATCH");
  op::require((a.flags() & py::array::c_style) && a.writeable(),
              "OUTPUT_NOT_WRITABLE_CONTIGUOUS");
  const auto target = binding::bounds(a);
  for (std::size_t i = 0; i < b.count; ++i)
    if (py::isinstance<py::array>(b.owners[i])) {
      const auto input =
          binding::bounds(py::reinterpret_borrow<py::array>(b.owners[i]));
      op::require(!binding::overlaps(target, input), "OUTPUT_ALIASES_INPUT");
    }
  return a;
}
py::object borrowed_output(const Bound &b, const op::Prepared &p) {
  std::vector<py::ssize_t> strides;
  for (int i = 0; i < p.view.shape.rank; ++i)
    strides.push_back(p.view.stride[i] *
                      static_cast<py::ssize_t>(sizeof(double)));
  py::array result(dtype(p.output_kind), dimensions(p.output_shape), strides,
                   p.view.data, b.owners[0]);
  result.attr("setflags")(false);
  return std::move(result);
}
py::dict execution_audit(const Bound &b, const op::Prepared &p,
                         const op::Audit &audit,
                         const op::Workspace &workspace) {
  py::dict d;
  d["operator_id"] = p.spec->name;
  d["opcode"] = static_cast<std::uint16_t>(p.spec->op);
  d["execution_backend"] = "pybind11_aot";
  d["scheduler_backend"] = "process_wide_cpp_cpu_admission";
  d["cpu_tokens"] = p.borrowed ? 0 : 1;
  d["python_fallback"] = 0;
  d["isa"] = p.borrowed && b.out.is_none() ? "view" : audit.isa;
  d["vector_elements"] = audit.vector_elements;
  d["input_copy_bytes"] = 0;
  d["algorithm_copy_bytes"] = audit.algorithm_copy_bytes;
  d["workspace_bytes"] = audit.scratch_bytes;
  d["workspace_capacity_bytes"] = workspace.capacity_bytes();
  d["output_is_view"] = p.borrowed && b.out.is_none();
  py::list addresses;
  for (std::size_t i = 0; i < b.count; ++i)
    if (py::isinstance<py::array>(b.owners[i]))
      addresses.append(
          py::int_(reinterpret_cast<std::uintptr_t>(b.args[i].data)));
  d["input_addresses"] = addresses;
  return d;
}

py::object invoke(const op::Spec &spec, py::args args, py::kwargs kwargs) {
  auto b = bind_args(spec, args, kwargs);
  const auto p = op::prepare(spec, b.args.data(), b.count);
  op::require(op::supports_isa(b.isa), "UNSUPPORTED_ISA");
  op::Workspace local;
  if (!b.workspace.is_none() && !py::isinstance<op::Workspace>(b.workspace))
    throw py::type_error("workspace must be Workspace");
  auto &workspace =
      b.workspace.is_none() ? local : b.workspace.cast<op::Workspace &>();
  std::unique_lock<std::mutex> lease(workspace.mutex, std::try_to_lock);
  op::require(lease.owns_lock(), "WORKSPACE_BUSY");
  op::Audit audit;
  py::object result;
  if (p.borrowed && b.out.is_none())
    result = borrowed_output(b, p);
  else {
    op::Output out;
    out.kind = p.output_kind;
    out.shape = p.output_shape;
    py::array storage;
    if (out.kind == op::Kind::fit || out.kind == op::Kind::interval) {
      if (!b.out.is_none())
        throw py::type_error("record outputs do not support out");
    } else if (!b.out.is_none() || out.shape.rank > 0) {
      storage = b.out.is_none() ? allocate_output(p) : checked_output(b, p);
      out.data = storage.mutable_data();
    }
    {
      py::gil_scoped_release release;
      auto &cpu =
          calmetrics_engine::native::NativeScheduler::instance().budget();
      if (!cpu.acquire(1))
        throw std::runtime_error("native operator CPU admission failed");
      struct CpuRelease {
        calmetrics_engine::native::CpuBudget &budget;
        ~CpuRelease() { budget.release(1); }
      } cpu_release{cpu};
      op::execute(p, out, workspace, b.isa, audit);
    }
    if (out.kind == op::Kind::fit || out.kind == op::Kind::interval) {
      const auto size = out.kind == op::Kind::fit ? 5u : 4u;
      py::tuple record(size);
      for (unsigned i = 0; i < size; ++i)
        record[i] = out.record[i];
      result = std::move(record);
    } else if (!b.out.is_none() || out.shape.rank > 0)
      result = std::move(storage);
    else if (out.kind == op::Kind::mask)
      result = py::bool_(out.scalar != 0);
    else if (out.kind == op::Kind::integer)
      result = py::int_(out.integer);
    else
      result = py::float_(out.scalar);
  }
  if (b.audit)
    return py::make_tuple(result, execution_audit(b, p, audit, workspace));
  return result;
}
py::dict metadata(const op::Spec &spec) {
  py::dict d;
  d["id"] = spec.name;
  d["opcode"] = static_cast<std::uint16_t>(spec.op);
  d["registry_version"] = op::registry_version;
  d["operator_version"] = "1.0.0";
  d["family"] = op::family_name(spec.family);
  d["min_args"] = spec.min_args;
  d["max_args"] = spec.max_args;
  d["parameters"] = op::parameter_names(spec, spec.max_args);
  d["defaults"] = op::default_rule(spec);
  py::list signatures;
  for (std::size_t arity = spec.min_args; arity <= spec.max_args; ++arity) {
    py::dict signature;
    signature["arity"] = arity;
    signature["parameters"] = op::parameter_names(spec, arity);
    signature["shape_rule"] = op::shape_rule(spec);
    signatures.append(signature);
  }
  d["signatures"] = signatures;
  d["shape_rule"] = op::shape_rule(spec);
  d["missing_policy"] = op::missing_policy(spec);
  d["composition"] = op::composition(spec);
  d["granularity"] = *op::composition(spec)          ? "composition"
                     : spec.op == op::Op::linear_fit ? "coupled_fit"
                     : spec.op == op::Op::recursive_filter ? "coupled_recurrence"
                                                     : "primitive";
  if (static_cast<std::uint16_t>(spec.op) >= 126)
    d["granularity"] = op::granularity(spec);
  d["temporal_dependency"] = op::temporal_dependency(spec);
  d["simd_eligible"] = op::simd_eligible(spec.op);
  d["parallel_policy"] = "caller_scheduled_no_internal_threads";
  d["input_policy"] = "exact_native_dtype_readonly_strided_no_copy";
  d["status_contract"] = "stable_error_code_or_explicit_interval_status";
  d["axis_semantics"] =
      "nominal axes/units/knowledge time validated by caller Typed DAG";
  d["execution_backend"] = "pybind11_aot";
  return d;
}
py::dict requirements(const op::Spec &spec, py::args args, py::kwargs kwargs) {
  auto b = bind_args(spec, args, kwargs);
  const auto p = op::prepare(spec, b.args.data(), b.count);
  py::dict d;
  d["shape"] = dimensions(p.output_shape);
  d["kind"] = p.output_kind == op::Kind::fit        ? "fit"
              : p.output_kind == op::Kind::interval ? "interval"
              : p.output_kind == op::Kind::mask     ? "uint8"
              : p.output_kind == op::Kind::integer  ? "int64"
                                                    : "float64";
  d["scratch_doubles"] = p.scratch_doubles;
  d["scratch_indices"] = p.scratch_indices;
  d["workspace_bytes"] = p.scratch_doubles * sizeof(double) +
                         p.scratch_indices * sizeof(std::size_t);
  d["borrowed_output"] = p.borrowed;
  return d;
}
} // namespace

void register_operators(py::module_ &parent) {
  auto m = parent.def_submodule(
      "operators", "Canonical AOT operators; no Python numeric fallback.");
  py::register_exception<op::Error>(m, "OperatorError", PyExc_ValueError);
  m.attr("REGISTRY_VERSION") = op::registry_version;
  py::class_<op::Workspace>(m, "Workspace")
      .def(py::init<>())
      .def(
          "reserve",
          [](op::Workspace &w, std::size_t doubles, std::size_t indices) {
            std::unique_lock<std::mutex> lock(w.mutex, std::try_to_lock);
            op::require(lock.owns_lock(), "WORKSPACE_BUSY");
            op::require(doubles <= static_cast<std::size_t>(PTRDIFF_MAX) /
                                       sizeof(double) &&
                            indices <= static_cast<std::size_t>(PTRDIFF_MAX) /
                                           sizeof(std::size_t),
                        "SHAPE_OVERFLOW");
            w.doubles.reserve(doubles);
            w.indices.reserve(indices);
          },
          py::arg("doubles") = 0, py::arg("indices") = 0)
      .def_property_readonly("capacity_bytes", [](op::Workspace &w) {
        std::unique_lock<std::mutex> lock(w.mutex, std::try_to_lock);
        op::require(lock.owns_lock(), "WORKSPACE_BUSY");
        return w.capacity_bytes();
      });
  py::class_<Handle>(m, "Operator")
      .def_property_readonly("name",
                             [](const Handle &h) { return h.spec->name; })
      .def_property_readonly("spec",
                             [](const Handle &h) { return metadata(*h.spec); })
      .def("__call__", [](const Handle &h, py::args a,
                          py::kwargs k) { return invoke(*h.spec, a, k); })
      .def("requirements", [](const Handle &h, py::args a, py::kwargs k) {
        return requirements(*h.spec, a, k);
      });
  m.def("get", [](const std::string &id) { return Handle{&op::lookup(id)}; });
  m.def("get_by_opcode",
        [](std::uint16_t id) { return Handle{&op::lookup(id)}; });
  m.def("catalog", [] {
    py::tuple result(op::operator_count);
    std::size_t i = 0;
    for (const auto &spec : op::registry())
      result[i++] = metadata(spec);
    return result;
  });
  m.def("available_simd", [] {
    py::list result;
    for (auto isa :
         {op::Isa::scalar, op::Isa::sse2, op::Isa::avx2, op::Isa::neon})
      if (op::supports_isa(isa))
        result.append(op::isa_name(isa));
    return result;
  });
  m.def(
      "call",
      [](const std::string &name, py::args a, py::kwargs k) {
        return invoke(op::lookup(name), a, k);
      },
      py::arg("name"));
  for (const auto &spec : op::registry()) {
    const auto *stable = &spec;
    m.def(
        spec.name,
        [stable](py::args a, py::kwargs k) { return invoke(*stable, a, k); },
        op::shape_rule(spec));
  }
}
