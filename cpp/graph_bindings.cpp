#include "calmetrics_engine/compiler.hpp"
#include "calmetrics_engine/planner.hpp"
#include "calmetrics_engine/scheduler.hpp"
#include "graph_binding_utils.hpp"
#include <pybind11/stl.h>

namespace py = pybind11;
namespace c = calmetrics_engine::compiler;
namespace g = calmetrics_engine::graph;
namespace p = calmetrics_engine::planner;
namespace b = calmetrics_engine::binding;
namespace t = calmetrics_engine::typed;

namespace calmetrics_engine::binding {
py::dict graph_audit(const g::Audit &a) {
  py::dict d;
  d["rows"] = a.rows;
  d["nodes"] = a.nodes;
  d["max_window"] = a.max_window;
  d["numeric_arena_bytes"] = a.numeric_arena_bytes;
  d["mask_arena_bytes"] = a.mask_arena_bytes;
  d["operator_workspace_capacity_bytes"] = a.operator_workspace_capacity_bytes;
  d["order_scratch_capacity_bytes"] = a.order_scratch_capacity_bytes;
  d["input_copy_bytes"] = a.input_copy_bytes;
  d["algorithm_copy_bytes"] = a.algorithm_copy_bytes;
  d["fused_scalar_calls"] = a.fused_scalar_calls;
  d["summary_source_scans"] = a.summary_source_scans;
  d["order_stat_sorts"] = a.order_stat_sorts;
  d["python_operator_calls"] = 0;
  d["execution_backend"] = "native_graph_interpreter";
  return d;
}
py::dict plan_metadata(const p::Plan &x) {
  py::dict d;
  d["graph_fingerprint"] = x.graph->fingerprint;
#define FIELD(name) d[#name] = x.name
  FIELD(lane);
  FIELD(process_count);
  FIELD(thread_count);
  FIELD(threads_per_process);
  FIELD(use_shared_memory);
  FIELD(async_orchestration);
  FIELD(hard_stop);
  FIELD(estimated_work_units);
  FIELD(estimated_logical_work_units);
  FIELD(estimated_typed_array_work_units);
  FIELD(estimated_input_bytes);
  FIELD(estimated_output_bytes);
  FIELD(estimated_status_bytes);
  FIELD(estimated_worker_scratch_bytes);
  FIELD(estimated_total_memory_bytes);
  FIELD(row_count);
  FIELD(interval_observations);
  FIELD(product_count);
  FIELD(max_intervals_per_product);
  FIELD(parallel_dimension);
  FIELD(max_window);
  FIELD(cpu_budget);
  FIELD(memory_budget_bytes);
  FIELD(simd_nodes);
  FIELD(reason_codes);
#undef FIELD
  d["branch_task_count"] = x.branch_tasks.size();
  py::list chunks;
  for (const auto &chunk : x.chunks)
    chunks.append(py::make_tuple(chunk.begin, chunk.end));
  d["chunks"] = std::move(chunks);
  py::list branch_tasks;
  for (const auto &task : x.branch_tasks)
    branch_tasks.append(py::make_tuple(task.row, task.branch_index));
  d["branch_tasks"] = std::move(branch_tasks);
  d["planner_backend"] = "cpp";
  d["memory_estimate_scope"] =
      "input/output/transport/arena/scratch; not process RSS";
  return d;
}
} // namespace calmetrics_engine::binding
namespace {
py::dict value_type_dict(const t::ValueType &value) {
  py::dict d;
  d["kind"] = t::kind_name(value.kind);
  d["dtype"] = t::dtype_name(value.dtype);
  d["axes"] = value.axes;
  d["shape"] = value.shape;
  d["semantic_dimension"] = value.semantic_dimension;
  d["price_basis"] = value.price_basis.empty() ? py::none() : py::cast(value.price_basis);
  if (!value.record_tag.empty()) {
    d["record_tag"] = value.record_tag;
    d["fields"] = value.fields;
  }
  d["display"] = t::display(value);
  return d;
}

t::ValueType parse_value_type(py::handle item) {
  if (py::isinstance<py::str>(item)) {
    const auto kind = py::cast<std::string>(item);
    if (kind == "series")
      return t::ValueType::series();
    if (kind == "scalar")
      return t::ValueType::scalar();
    throw c::CompileError("legacy variable type must be series or scalar");
  }
  if (!py::isinstance<py::dict>(item))
    throw c::CompileError("typed variable declaration must be a string or mapping");
  auto d = py::reinterpret_borrow<py::dict>(item);
  if (!d.contains("kind"))
    throw c::CompileError("typed variable declaration requires kind");
  t::ValueType value;
  const auto kind = py::cast<std::string>(d["kind"]);
  if (kind == "scalar")
    value.kind = t::ValueKind::scalar;
  else if (kind == "series")
    value.kind = t::ValueKind::series;
  else if (kind == "vector")
    value.kind = t::ValueKind::vector;
  else if (kind == "matrix")
    value.kind = t::ValueKind::matrix;
  else if (kind == "window")
    value.kind = t::ValueKind::window;
  else if (kind == "record")
    value.kind = t::ValueKind::record;
  else
    throw c::CompileError("unsupported typed variable kind: " + kind);
  const auto dtype = d.contains("dtype") ? py::cast<std::string>(d["dtype"])
                                         : std::string("float64");
  if (dtype == "float64")
    value.dtype = t::DType::float64;
  else if (dtype == "int64")
    value.dtype = t::DType::int64;
  else if (dtype == "bool")
    value.dtype = t::DType::boolean;
  else
    throw c::CompileError("unsupported typed variable dtype: " + dtype);
  if (d.contains("axes"))
    value.axes = py::cast<std::vector<std::string>>(d["axes"]);
  else if (value.kind == t::ValueKind::series)
    value.axes = {"time"};
  else if (value.kind == t::ValueKind::vector)
    value.axes = {"asset"};
  else if (value.kind == t::ValueKind::matrix)
    value.axes = {"time", "asset"};
  else if (value.kind == t::ValueKind::window)
    value.axes = {"time", "window"};
  if (d.contains("shape")) {
    for (auto shape : py::reinterpret_borrow<py::sequence>(d["shape"]))
      value.shape.push_back(py::cast<std::string>(py::str(shape)));
  } else if (value.kind == t::ValueKind::series)
    value.shape = {"T"};
  else if (value.kind == t::ValueKind::vector)
    value.shape = {"N"};
  else if (value.kind == t::ValueKind::matrix)
    value.shape = {"T", "N"};
  else if (value.kind == t::ValueKind::window)
    value.shape = {"T", "W"};
  value.semantic_dimension =
      d.contains("semantic_dimension")
          ? py::cast<std::string>(d["semantic_dimension"])
          : std::string(value.dtype == t::DType::boolean ? "mask" : "dimensionless");
  if (d.contains("price_basis") && !d["price_basis"].is_none())
    value.price_basis = py::cast<std::string>(d["price_basis"]);
  if (value.kind == t::ValueKind::record) {
    if (d.contains("record_tag"))
      value.record_tag = py::cast<std::string>(d["record_tag"]);
    if (d.contains("fields"))
      value.fields = py::cast<std::vector<std::string>>(d["fields"]);
  }
  try {
    value.validate();
  } catch (const t::Error &error) {
    throw c::CompileError(error.what());
  }
  return value;
}

template <class T> py::tuple tuple(const std::vector<T> &v) {
  py::tuple t(v.size());
  for (std::size_t i = 0; i < v.size(); ++i)
    t[i] = py::cast(v[i]);
  return t;
}
py::list raw_nodes(const c::CompiledGraph &graph) {
  py::list result;
  for (const auto &n : graph.program.nodes) {
    py::dict d;
    d["kind"] = c::kind_name(n.kind);
    d["storage"] = c::storage_name(n.storage);
    d["slot"] = n.slot;
    if (n.kind == g::NodeKind::operation ||
        n.kind == g::NodeKind::rolling_scope || n.kind == g::NodeKind::apply_scope ||
        n.kind == g::NodeKind::interval_tail) {
      if (n.kind == g::NodeKind::operation)
        d["opcode"] = n.opcode;
      else if (n.kind == g::NodeKind::rolling_scope || n.kind == g::NodeKind::apply_scope)
        d["scope_index"] = n.input_index;
      d["parents"] = std::vector<std::uint32_t>(
          n.parents.begin(), n.parents.begin() + n.parent_count);
    } else if (n.kind == g::NodeKind::constant)
      d["constant"] = n.constant;
    else
      d["input_index"] = n.input_index;
    result.append(d);
  }
  return result;
}
py::dict graph_metadata(const c::CompiledGraph &graph) {
  py::dict d;
  d["fingerprint"] = graph.fingerprint;
  d["error_policy"] = graph.program.isolate_errors ? "isolate" : "raise";
  d["status_contract"] = "indicator-status-1";
  d["source_contracts"] = graph.source_contracts;
  d["minimum_observations"] = graph.program.minimum_observations;
  d["scope_work_budget"] = graph.program.scope_work_budget;
  d["expression_count"] = graph.expressions.size();
  d["node_count"] = graph.nodes.size();
  d["raw_node_count"] = graph.raw_node_count;
  d["cse_eliminated_nodes"] = graph.raw_node_count - graph.nodes.size();
  d["root_count"] = graph.program.roots.size();
  d["input_names"] = graph.input_names;
  d["parameter_names"] = graph.parameter_names;
  d["numeric_slots"] = graph.program.numeric_slots;
  d["mask_slots"] = graph.program.mask_slots;
  d["integer_slots"] = graph.program.integer_slots;
  d["typed_ir_version"] = "cpp-typed-ir-1";
  d["output_kind"] = graph.program.output_kind == g::OutputKind::series ? "series" : "scalar";
  d["output_dtype"] = g::output_dtype_name(graph.program.output_dtype);
  d["rolling_scope_count"] = graph.program.rolling_scopes.size();
  d["apply_scope_count"] = graph.program.apply_scopes.size();
  py::dict variable_types;
  for (const auto &variable : graph.variable_types)
    variable_types[py::str(variable.name)] = value_type_dict(variable.type);
  d["variable_types"] = std::move(variable_types);
  py::list root_types;
  for (auto root : graph.program.roots)
    root_types.append(value_type_dict(graph.nodes[root].inferred_type));
  d["root_types"] = std::move(root_types);
  std::size_t count = 0;
  for (const auto &n : graph.nodes)
    count += n.simd_eligible;
  d["simd_node_count"] = count;
  d["branch_count"] = graph.branches.size();
  py::dict physical;
  physical["constant_per_row"] = graph.physical_cost.constant_per_row;
  physical["linear_per_observation"] =
      graph.physical_cost.linear_per_observation;
  physical["sort_nlogn"] = graph.physical_cost.sort_nlogn;
  d["physical_cost"] = std::move(physical);
  py::list branches;
  py::list branch_details;
  for (const auto &branch : graph.branches) {
    branches.append(branch.root_indices);
    py::dict detail;
    detail["roots"] = branch.root_indices;
    detail["node_count"] = branch.program.nodes.size();
    detail["numeric_slots"] = branch.program.numeric_slots;
    detail["mask_slots"] = branch.program.mask_slots;
    detail["integer_slots"] = branch.program.integer_slots;
    detail["source_nodes"] = branch.source_nodes;
    py::dict cost;
    cost["constant_per_row"] = branch.cost.constant_per_row;
    cost["linear_per_observation"] = branch.cost.linear_per_observation;
    cost["sort_nlogn"] = branch.cost.sort_nlogn;
    detail["physical_cost"] = std::move(cost);
    branch_details.append(std::move(detail));
  }
  d["branch_roots"] = std::move(branches);
  d["branch_details"] = std::move(branch_details);
  d["compiler_backend"] = "cpp";
  d["liveness_policy"] = "borrowed_view_aware";
  return d;
}
class CompilerAPI {
public:
  std::vector<t::Variable> variables;
  explicit CompilerAPI(const py::dict &mapping) {
    for (auto item : mapping) {
      if (!py::isinstance<py::str>(item.first))
        throw c::CompileError("variable names must be strings");
      if (!PyUnicode_IsIdentifier(item.first.ptr()))
        throw c::CompileError("invalid variable name");
      variables.push_back(
          {py::cast<std::string>(item.first), parse_value_type(item.second)});
    }
    if (variables.empty())
      throw c::CompileError("at least one variable is required");
  }
  std::shared_ptr<c::CompiledGraph>
  compile(const py::object &expressions, const std::string &error_policy,
          const std::vector<std::map<std::string, std::string>> &root_bindings,
          const std::vector<std::string> &source_contracts, std::uint32_t minimum_observations,
          std::uint64_t scope_work_budget) const {
    if (error_policy != "raise" && error_policy != "isolate")
      throw c::CompileError("error_policy must be raise or isolate");
    std::vector<std::string> sources;
    if (py::isinstance<py::str>(expressions))
      sources.push_back(py::cast<std::string>(expressions));
    else {
      if (!py::isinstance<py::sequence>(expressions))
        throw c::CompileError("expressions must be strings");
      for (auto item : py::reinterpret_borrow<py::sequence>(expressions)) {
        if (!py::isinstance<py::str>(item))
          throw c::CompileError("expressions must be strings");
        sources.push_back(py::cast<std::string>(item));
      }
    }
    py::gil_scoped_release release;
    return c::compile(sources, variables, error_policy == "isolate", root_bindings, source_contracts, minimum_observations, scope_work_budget);
  }
};
g::Program raw_program(const py::list &raw,
                       const std::vector<std::uint32_t> &roots,
                       std::size_t inputs, std::size_t parameters,
                       std::size_t numeric_slots, std::size_t mask_slots,
                       std::size_t integer_slots, const std::vector<std::uint8_t> &input_axes) {
  if (raw.size() > 65536)
    throw py::value_error("native graph node limit exceeded");
  g::Program program;
  program.roots = roots;
  program.input_count = inputs;
  program.parameter_count = parameters;
  program.numeric_slots = numeric_slots;
  program.mask_slots = mask_slots;
  program.integer_slots = integer_slots;
  program.input_axes = input_axes;
  for (auto item : raw) {
    auto d = py::cast<py::dict>(item);
    g::Node n;
    auto kind = py::cast<std::string>(d["kind"]);
    if (kind == "input")
      n.kind = g::NodeKind::input;
    else if (kind == "parameter")
      n.kind = g::NodeKind::parameter;
    else if (kind == "constant")
      n.kind = g::NodeKind::constant;
    else if (kind == "operation")
      n.kind = g::NodeKind::operation;
    else
      throw py::value_error("invalid graph node kind");
    if (d.contains("opcode"))
      n.opcode = py::cast<std::uint16_t>(d["opcode"]);
    if (d.contains("input_index"))
      n.input_index = py::cast<std::uint16_t>(d["input_index"]);
    if (d.contains("constant"))
      n.constant = py::cast<double>(d["constant"]);
    if (d.contains("parents")) {
      auto parents = py::cast<std::vector<std::uint32_t>>(d["parents"]);
      if (parents.size() > 8)
        throw py::value_error("too many parents");
      n.parent_count = static_cast<std::uint8_t>(parents.size());
      std::copy(parents.begin(), parents.end(), n.parents.begin());
    }
    if (d.contains("storage")) {
      auto storage = py::cast<std::string>(d["storage"]);
      if (storage == "numeric")
        n.storage = g::StorageKind::numeric;
      else if (storage == "mask")
        n.storage = g::StorageKind::mask;
      else if (storage == "integer")
        n.storage = g::StorageKind::integer;
      else if (storage != "inline")
        throw py::value_error("invalid storage kind");
    }
    if (d.contains("slot"))
      n.slot = py::cast<std::uint32_t>(d["slot"]);
    program.nodes.push_back(n);
  }
  program.finalize();
  return program;
}
py::dict raw_execute(const g::Program &program, const py::tuple &input_arrays,
                     const py::array &starts, const py::array &ends,
                     py::array output, const py::object &parameters) {
  if (input_arrays.size() != program.input_count)
    throw py::value_error("input array count mismatch");
  std::vector<calmetrics_engine::ops::Value> inputs;
  b::exact<std::int64_t>(starts, 1, "starts");
  b::exact<std::int64_t>(ends, 1, "ends");
  switch (program.output_dtype) {
  case g::OutputDType::float64: b::exact<double>(output, 2, "out", true); break;
  case g::OutputDType::boolean: b::exact<bool>(output, 2, "out", true); break;
  case g::OutputDType::int64: b::exact<std::int64_t>(output, 2, "out", true); break;
  }
  if (starts.size() != ends.size())
    throw py::value_error("batch shape mismatch");
  py::ssize_t expected_output_rows = starts.size();
  if (program.output_kind == g::OutputKind::series) {
    expected_output_rows = 0;
    const auto *begin_ptr = static_cast<const std::int64_t *>(starts.data());
    const auto *end_ptr = static_cast<const std::int64_t *>(ends.data());
    for (py::ssize_t row = 0; row < starts.size(); ++row) {
      if (begin_ptr[row] < 0 || end_ptr[row] < begin_ptr[row])
        throw py::value_error("invalid interval bounds");
      const auto length = end_ptr[row] - begin_ptr[row];
      if (length > PY_SSIZE_T_MAX - expected_output_rows)
        throw py::value_error("series output shape overflow");
      expected_output_rows += static_cast<py::ssize_t>(length);
    }
  }
  if (output.shape(0) != expected_output_rows ||
      output.shape(1) != static_cast<py::ssize_t>(program.roots.size()))
    throw py::value_error("batch shape mismatch");
  const auto output_bounds = b::bounds(output);
  for (auto item : input_arrays) {
    auto a = b::require_array(item);
    if (a.ndim() < 1 || a.ndim() > 2) throw py::value_error("input must be rank 1 or 2");
    const bool integer = a.dtype().equal(py::dtype::of<std::int64_t>());
    const bool mask = a.dtype().equal(py::dtype::of<bool>()) || a.dtype().equal(py::dtype::of<std::uint8_t>());
    if (!integer && !mask && !a.dtype().equal(py::dtype::of<double>()))
      throw py::type_error("input requires exact float64, int64 or bool/uint8 dtype");
    const auto input_index = inputs.size();
    const bool temporal = program.input_axes.empty() || program.input_axes.at(input_index) == 0;
    if (!integer && !mask && temporal) b::exact<double>(a, 1, "input");
    const auto itemsize = mask ? 1u : 8u;
    if (reinterpret_cast<std::uintptr_t>(a.data()) % itemsize)
      throw py::value_error("input must be aligned");
    b::validate_owner(a);
    if (b::overlaps(output_bounds, b::bounds(a)))
      throw py::value_error("output aliases an input");
    calmetrics_engine::ops::Value value;
    value.kind = integer ? calmetrics_engine::ops::Kind::integer :
        (mask ? calmetrics_engine::ops::Kind::mask : calmetrics_engine::ops::Kind::number);
    value.shape.rank = static_cast<int>(a.ndim());
    value.data = a.data();
    for (int axis = 0; axis < value.shape.rank; ++axis) {
      if (a.strides(axis) % itemsize) throw py::value_error("unsupported byte stride");
      value.shape.dim[axis] = a.shape(axis);
      value.stride[axis] = a.strides(axis) / static_cast<py::ssize_t>(itemsize);
    }
    inputs.push_back(value);
  }
  if (b::overlaps(output_bounds, b::bounds(starts)) ||
      b::overlaps(output_bounds, b::bounds(ends)))
    throw py::value_error("output aliases interval metadata");
  py::array param;
  const double *ptr = nullptr;
  std::size_t count = 0;
  if (!parameters.is_none()) {
    param = b::require_array(parameters);
    b::exact<double>(param, 1, "parameters");
    ptr = static_cast<const double *>(param.data());
    count = param.size();
    if (b::overlaps(output_bounds, b::bounds(param)))
      throw py::value_error("output aliases parameters");
  }
  auto *out = output.mutable_data();
  g::Audit audit;
  {
    py::gil_scoped_release release;
    auto &cpu = calmetrics_engine::native::NativeScheduler::instance().budget();
    if (!cpu.acquire(1))
      throw std::runtime_error("native program CPU admission failed");
    struct CpuRelease {
      calmetrics_engine::native::CpuBudget &budget;
      ~CpuRelease() { budget.release(1); }
    } cpu_release{cpu};
    audit = g::execute(program, inputs, ptr, count,
                       static_cast<const std::int64_t *>(starts.data()),
                       static_cast<const std::int64_t *>(ends.data()),
                       starts.size(), out, program.roots.size());
  }
  auto result = b::graph_audit(audit);
  result["output_dtype"] = g::output_dtype_name(program.output_dtype);
  if (program.isolate_errors) {
    if (audit.statuses.size() != static_cast<std::size_t>(output.size()))
      throw std::runtime_error("invalid native statuses");
    py::array_t<std::int16_t> statuses({output.shape(0), output.shape(1)});
    std::copy(audit.statuses.begin(), audit.statuses.end(), statuses.mutable_data());
    statuses.attr("setflags")(false);
    result["statuses"] = std::move(statuses);
  }
  result["scheduler_backend"] = "process_wide_cpp_cpu_admission";
  result["cpu_tokens"] = 1;
  return result;
}
} // namespace
void register_native_api(py::module_ &module);
void register_graph(py::module_ &parent) {
  auto module = parent.def_submodule(
      "graph", "C++ compiler, planner, scheduler and numerical graph runtime");
  py::register_exception<c::CompileError>(module, "GraphCompileError",
                                          PyExc_ValueError);
  py::class_<g::Program>(module, "Program")
      .def(py::init(&raw_program), py::arg("nodes"), py::arg("roots"),
           py::arg("input_count"), py::arg("parameter_count") = 0,
           py::arg("numeric_slots") = 0, py::arg("mask_slots") = 0,
           py::arg("integer_slots") = 0, py::arg("input_axes") = std::vector<std::uint8_t>{})
      .def("execute", &raw_execute, py::arg("inputs"),
           py::arg("starts").noconvert(), py::arg("ends").noconvert(),
           py::arg("out").noconvert(), py::kw_only(),
           py::arg("parameters") = py::none())
      .def_property_readonly("metadata", [](const g::Program &p) {
        py::dict d;
        d["node_count"] = p.nodes.size();
        d["roots"] = p.roots;
        d["input_count"] = p.input_count;
        d["parameter_count"] = p.parameter_count;
        d["numeric_slots"] = p.numeric_slots;
        d["mask_slots"] = p.mask_slots;
        d["integer_slots"] = p.integer_slots;
        d["input_axes"] = p.input_axes;
        d["output_kind"] =
            p.output_kind == g::OutputKind::series ? "series" : "scalar";
        d["output_dtype"] = g::output_dtype_name(p.output_dtype);
        d["python_operator_calls"] = 0;
        d["execution_backend"] = "native_graph_interpreter";
        return d;
      });
  py::class_<c::NodeInfo>(module, "GraphNode")
      .def_readonly("node_id", &c::NodeInfo::node_id)
      .def_property_readonly(
          "kind",
          [](const c::NodeInfo &n) { return c::kind_name(n.node.kind); })
      .def_property_readonly(
          "value_class",
          [](const c::NodeInfo &n) { return c::class_name(n.value_class); })
      .def_property_readonly("opcode",
                             [](const c::NodeInfo &n) { return n.node.opcode; })
      .def_property_readonly(
          "operator_id",
          [](const c::NodeInfo &n) -> py::object {
            return n.node.kind == g::NodeKind::operation
                       ? py::cast(std::string(
                             calmetrics_engine::ops::lookup(n.node.opcode)
                                 .name))
                       : py::none();
          })
      .def_property_readonly(
          "parents",
          [](const c::NodeInfo &n) {
            return tuple(std::vector<std::uint32_t>(n.node.parents.begin(),
                                                    n.node.parents.begin() +
                                                        n.node.parent_count));
          })
      .def_property_readonly(
          "constant", [](const c::NodeInfo &n) { return n.node.constant; })
      .def_property_readonly(
          "binding_index",
          [](const c::NodeInfo &n) { return n.node.input_index; })
      .def_property_readonly(
          "storage",
          [](const c::NodeInfo &n) { return c::storage_name(n.node.storage); })
      .def_property_readonly("slot",
                             [](const c::NodeInfo &n) { return n.node.slot; })
      .def_property_readonly("is_array",
                             [](const c::NodeInfo &n) {
                               return n.value_class == c::ValueClass::series ||
                                      n.value_class == c::ValueClass::mask_series ||
                                      n.value_class == c::ValueClass::integer_series;
                             })
      .def_property_readonly(
          "inferred_type",
          [](const c::NodeInfo &n) { return value_type_dict(n.inferred_type); })
      .def_readonly("last_use", &c::NodeInfo::last_use)
      .def_readonly("cost_model", &c::NodeInfo::cost_model)
      .def_readonly("simd_eligible", &c::NodeInfo::simd_eligible);
  py::class_<c::CompiledGraph, std::shared_ptr<c::CompiledGraph>>(
      module, "CompiledGraph")
      .def("metadata", &graph_metadata)
      .def("native_nodes", &raw_nodes)
      .def_property_readonly(
          "expressions",
          [](const c::CompiledGraph &g) { return tuple(g.expressions); })
      .def_property_readonly(
          "nodes", [](const c::CompiledGraph &g) { return tuple(g.nodes); })
      .def_property_readonly(
          "roots",
          [](const c::CompiledGraph &g) { return tuple(g.program.roots); })
      .def_property_readonly(
          "input_names",
          [](const c::CompiledGraph &g) { return tuple(g.input_names); })
      .def_property_readonly(
          "parameter_names",
          [](const c::CompiledGraph &g) { return tuple(g.parameter_names); })
      .def_property_readonly(
          "numeric_slots",
          [](const c::CompiledGraph &g) { return g.program.numeric_slots; })
      .def_property_readonly(
          "mask_slots",
          [](const c::CompiledGraph &g) { return g.program.mask_slots; })
      .def_property_readonly("cse_eliminated_nodes",
                             [](const c::CompiledGraph &g) {
                               return g.raw_node_count - g.nodes.size();
                             })
      .def_property_readonly("simd_nodes",
                             [](const c::CompiledGraph &g) {
                               std::vector<std::uint32_t> nodes;
                               for (const auto &n : g.nodes)
                                 if (n.simd_eligible)
                                   nodes.push_back(n.node_id);
                               return tuple(nodes);
                             })
      .def_property_readonly(
          "_program", [](const c::CompiledGraph &g) { return g.program; })
      .def_readonly("fingerprint", &c::CompiledGraph::fingerprint)
      .def_readonly("raw_node_count", &c::CompiledGraph::raw_node_count)
      .def(py::pickle(
          [](const c::CompiledGraph &g) {
            py::dict variables;
            for (const auto &variable : g.variable_types)
              variables[py::str(variable.name)] = value_type_dict(variable.type);
            return py::make_tuple(2, g.expressions, variables,
                g.program.isolate_errors, g.root_bindings, g.source_contracts,
                g.program.minimum_observations, g.program.scope_work_budget);
          },
          [](py::tuple state) {
            // Saved legacy definitions retain their original scalar/series
            // admission; all new states preserve explicit typed contracts.
            if (state.size() == 2)
              return c::compile(
                  py::cast<std::vector<std::string>>(state[0]),
                  py::cast<std::vector<std::pair<std::string, std::string>>>(state[1]));
            if (state.size() != 8 || py::cast<int>(state[0]) != 2)
              throw py::value_error("invalid compiled graph state");
            std::vector<t::Variable> variables;
            for (auto item : py::cast<py::dict>(state[2]))
              variables.push_back({py::cast<std::string>(item.first), parse_value_type(item.second)});
            return c::compile(py::cast<std::vector<std::string>>(state[1]), variables,
                py::cast<bool>(state[3]),
                py::cast<std::vector<std::map<std::string, std::string>>>(state[4]),
                py::cast<std::vector<std::string>>(state[5]),
                py::cast<std::uint32_t>(state[6]), py::cast<std::uint64_t>(state[7]));
          }));
  py::class_<CompilerAPI>(module, "GraphCompiler")
      .def(py::init<const py::dict &>(), py::arg("variables"))
      .def("compile", &CompilerAPI::compile, py::arg("expressions"),
           py::kw_only(), py::arg("error_policy") = "raise",
           py::arg("root_bindings") = std::vector<std::map<std::string, std::string>>{},
           py::arg("source_contracts") = std::vector<std::string>{},
           py::arg("minimum_observations") = 0,
           py::arg("scope_work_budget") = 100000000ull);
  py::class_<p::Config>(module, "PlannerConfig")
      .def(py::init([](double thread, double process, std::size_t input,
                       std::size_t shared, std::size_t rows, std::size_t simd,
                       std::size_t processes, std::size_t jobs,
                       double dag_branch) {
             p::Config c;
             c.thread_work_units = thread;
             c.process_work_units = process;
             c.process_input_threshold_bytes = input;
             c.shared_memory_threshold_bytes = shared;
             c.min_rows_per_worker = rows;
             c.simd_min_elements = simd;
             c.max_processes = processes;
             c.max_async_jobs = jobs;
             c.dag_branch_work_units = dag_branch;
             c.validate();
             return c;
           }),
           py::arg("thread_work_units") = 250000.0,
           py::arg("process_work_units") = 4000000000.0,
           py::arg("process_input_threshold_bytes") = 256 * 1024 * 1024,
           py::arg("shared_memory_threshold_bytes") = 8 * 1024 * 1024,
           py::arg("min_rows_per_worker") = 16,
           py::arg("simd_min_elements") = 128, py::arg("max_processes") = 8,
           py::arg("max_async_jobs") = 8,
           py::arg("dag_branch_work_units") = 5000000.0)
#define FIELD(name) .def_readonly(#name, &p::Config::name)
          FIELD(thread_work_units) FIELD(dag_branch_work_units)
              FIELD(process_work_units) FIELD(process_input_threshold_bytes)
                  FIELD(shared_memory_threshold_bytes)
                      FIELD(min_rows_per_worker) FIELD(simd_min_elements)
                          FIELD(max_processes) FIELD(max_async_jobs)
#undef FIELD
      ;
  py::class_<p::Plan, std::shared_ptr<p::Plan>>(module, "ExecutionPlan")
      .def("metadata", &b::plan_metadata)
      .def_readonly("graph", &p::Plan::graph)
#define FIELD(name) .def_readonly(#name, &p::Plan::name)
          FIELD(lane) FIELD(process_count) FIELD(thread_count)
              FIELD(threads_per_process) FIELD(use_shared_memory)
                  FIELD(async_orchestration) FIELD(hard_stop) FIELD(
                      estimated_work_units) FIELD(estimated_logical_work_units) FIELD(estimated_typed_array_work_units)
                      FIELD(estimated_input_bytes) FIELD(estimated_output_bytes)
                          FIELD(estimated_worker_scratch_bytes) FIELD(
                              estimated_total_memory_bytes) FIELD(row_count)
                              FIELD(interval_observations) FIELD(product_count)
                                  FIELD(max_intervals_per_product)
                                      FIELD(parallel_dimension)
                                          FIELD(max_window) FIELD(cpu_budget)
                                              FIELD(memory_budget_bytes)
#undef FIELD
      .def_property_readonly(
          "simd_nodes", [](const p::Plan &p) { return tuple(p.simd_nodes); })
      .def_property_readonly("reason_codes", [](const p::Plan &p) {
        return tuple(p.reason_codes);
      });
  register_native_api(module);
}
