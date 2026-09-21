#include "calmetrics_engine/native_runtime.hpp"
#include "graph_binding_utils.hpp"
#include <cstring>
#include <filesystem>
#include <map>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace n = calmetrics_engine::native;
namespace c = calmetrics_engine::compiler;
namespace p = calmetrics_engine::planner;
namespace b = calmetrics_engine::binding;
namespace o = calmetrics_engine::ops;

namespace {
std::size_t default_cpu(std::optional<std::size_t> cpu) {
  return cpu ? *cpu : std::max(1u, std::thread::hardware_concurrency());
}
std::size_t byte_size(const std::vector<py::ssize_t> &shape,
                      const py::dtype &dtype) {
  if (py::cast<bool>(dtype.attr("hasobject")) || dtype.itemsize() <= 0)
    throw py::type_error(
        "shared arrays require a non-object fixed-width dtype");
  std::size_t count = 1;
  for (auto dimension : shape) {
    if (dimension < 0)
      throw py::value_error("negative shared dimension");
    count = p::checked_mul(count, static_cast<std::size_t>(dimension));
  }
  return p::checked_mul(count, static_cast<std::size_t>(dtype.itemsize()));
}
struct SharedArrayDescriptor {
  std::string name;
  std::vector<py::ssize_t> shape;
  std::string dtype;
  bool readonly = false;
  std::size_t nbytes() const { return byte_size(shape, py::dtype(dtype)); }
};
class SharedArrayOwner {
public:
  std::shared_ptr<n::SharedRegion> region;
  std::vector<py::ssize_t> shape;
  py::dtype dtype;
  bool readonly = false;
  SharedArrayOwner(std::shared_ptr<n::SharedRegion> r,
                   std::vector<py::ssize_t> s, py::dtype d, bool ro = false)
      : region(std::move(r)), shape(std::move(s)), dtype(std::move(d)),
        readonly(ro) {}
  void ensure() const {
    if (!region)
      throw std::runtime_error("shared array owner is released");
  }
  static std::shared_ptr<SharedArrayOwner> from_array(py::handle item) {
    auto a = b::require_array(item);
    if (!(a.flags() & py::array::c_style))
      throw py::value_error("shared input must be C-contiguous");
    b::validate_owner(a);
    std::vector<py::ssize_t> shape(a.shape(), a.shape() + a.ndim());
    const auto bytes = byte_size(shape, a.dtype());
    auto region = n::SharedRegion::create(bytes);
    if (bytes)
      std::memcpy(region->data(), a.data(), bytes);
    return std::make_shared<SharedArrayOwner>(region, shape, a.dtype());
  }
  static std::shared_ptr<SharedArrayOwner>
  empty(const std::vector<py::ssize_t> &shape, const py::object &dtype) {
    py::dtype type = py::dtype::from_args(dtype);
    return std::make_shared<SharedArrayOwner>(
        n::SharedRegion::create(byte_size(shape, type)), shape, type);
  }
  py::array array() const {
    ensure();
    py::array a(dtype, shape, {}, region->data(), py::cast(region));
    if (readonly)
      a.attr("setflags")(py::arg("write") = false);
    return a;
  }
  SharedArrayDescriptor descriptor() const {
    ensure();
    return {region->name(), shape, py::cast<std::string>(dtype.attr("str")),
            readonly || !region->writable()};
  }
  std::size_t nbytes() const {
    ensure();
    return byte_size(shape, dtype);
  }
  void close() { region.reset(); }
  void unlink() {
    if (region)
      region->unlink();
  }
  void release() { region.reset(); }
};
class SharedInputBundle {
public:
  std::map<std::string, std::shared_ptr<SharedArrayOwner>> owners;
  std::size_t boundary_copy_bytes = 0;
  bool released = false;
  static std::shared_ptr<SharedInputBundle>
  from_inputs(const py::dict &inputs) {
    auto bundle = std::make_shared<SharedInputBundle>();
    for (auto item : inputs) {
      auto owner = SharedArrayOwner::from_array(item.second);
      owner->region->make_readonly();
      owner->readonly = true;
      bundle->boundary_copy_bytes =
          p::checked_add(bundle->boundary_copy_bytes, owner->nbytes());
      bundle->owners.emplace(py::cast<std::string>(item.first),
                             std::move(owner));
    }
    return bundle;
  }
  void ensure() const {
    if (released)
      throw std::runtime_error("shared input bundle is released");
  }
  py::dict arrays() const {
    ensure();
    py::dict d;
    for (const auto &item : owners)
      d[py::str(item.first)] = item.second->array();
    return d;
  }
  py::dict descriptors() const {
    ensure();
    py::dict d;
    for (const auto &item : owners)
      d[py::str(item.first)] = py::cast(item.second->descriptor());
    return d;
  }
  std::size_t nbytes() const {
    ensure();
    std::size_t bytes = 0;
    for (const auto &item : owners)
      bytes = p::checked_add(bytes, item.second->nbytes());
    return bytes;
  }
  void release() {
    if (released)
      return;
    owners.clear();
    released = true;
  }
};

struct ArrayPin {
  py::array owner;
  py::dtype dtype;
  const void *data = nullptr;
  py::ssize_t size = 0, stride = 0;
  std::vector<py::ssize_t> shape, strides;
  py::object allocation_owner = py::none();
  const void *allocation_data = nullptr;
  py::ssize_t allocation_bytes = 0;
  static ArrayPin capture(const py::array &a) {
    ArrayPin pin{a, a.dtype(), a.data(), a.size(), a.ndim() ? a.strides(0) : 0,
                 {a.shape(), a.shape() + a.ndim()},
                 {a.strides(), a.strides() + a.ndim()}};
    py::object current = a;
    for (unsigned depth = 0; depth < 32; ++depth) {
      if (py::isinstance<py::array>(current)) {
        auto array = py::reinterpret_borrow<py::array>(current);
        if (array.owndata()) {
          pin.allocation_owner = current;
          pin.allocation_data = array.data();
          pin.allocation_bytes = array.nbytes();
          break;
        }
      }
      auto base = py::getattr(current, "base", py::none());
      if (base.is_none()) break;
      current = base;
    }
    return pin;
  }
  template <class T> static ArrayPin bind(py::handle item, const char *name) {
    auto a = b::require_array(item);
    b::exact<T>(a, 1, name);
    return capture(a);
  }
  bool unchanged() const {
    if (!allocation_owner.is_none()) {
      const auto allocation = py::reinterpret_borrow<py::array>(allocation_owner);
      if (allocation.data() != allocation_data || allocation.nbytes() != allocation_bytes)
        return false;
    }
    return owner.data() == data && owner.size() == size &&
           owner.ndim() == static_cast<py::ssize_t>(shape.size()) &&
           std::equal(shape.begin(), shape.end(), owner.shape()) &&
           std::equal(strides.begin(), strides.end(), owner.strides()) &&
           owner.dtype().equal(dtype);
  }
  bool same(py::handle item) const { return item.is(owner) && unchanged(); }
};

// Typed binding is an exact-dtype view. Shape/owner checks run before any GIL
// release; native workers retain only the resulting pointers and descriptors.
o::Value bind_graph_array(const py::array &a, const calmetrics_engine::typed::ValueType &type,
                          std::map<std::string, std::size_t> &dimensions) {
  namespace t = calmetrics_engine::typed;
  const bool dtype_ok = type.dtype == t::DType::float64
      ? a.dtype().equal(py::dtype::of<double>())
      : type.dtype == t::DType::int64
          ? a.dtype().equal(py::dtype::of<std::int64_t>())
          : (a.dtype().equal(py::dtype::of<bool>()) ||
             a.dtype().equal(py::dtype::of<std::uint8_t>()));
  if (!dtype_ok) throw py::type_error("graph input must use its declared exact native dtype");
  if (a.ndim() != static_cast<py::ssize_t>(type.rank()) || a.ndim() < 1 || a.ndim() > 2)
    throw py::value_error("graph input rank does not match typed declaration");
  // Existing temporal float64 batches retain the documented contiguous
  // contract. Newly typed vector/matrix/category bindings carry explicit strides.
  if (type.kind == t::ValueKind::series && type.dtype == t::DType::float64 &&
      !(a.flags() & py::array::c_style))
    throw py::value_error("graph float64 series must be C-contiguous");
  const auto itemsize = type.dtype == t::DType::boolean ? 1u : 8u;
  if (reinterpret_cast<std::uintptr_t>(a.data()) % itemsize)
    throw py::value_error("graph input must be aligned");
  b::bounds(a);
  b::validate_owner(a);
  o::Value value;
  value.kind = type.dtype == t::DType::float64 ? o::Kind::number
      : (type.dtype == t::DType::int64 ? o::Kind::integer : o::Kind::mask);
  value.shape.rank = static_cast<int>(a.ndim());
  value.data = a.data();
  for (int axis = 0; axis < value.shape.rank; ++axis) {
    if (a.strides(axis) % itemsize)
      throw py::value_error("graph input has unsupported byte stride");
    const auto size = static_cast<std::size_t>(a.shape(axis));
    value.shape.dim[axis] = size;
    value.stride[axis] = a.strides(axis) / static_cast<py::ssize_t>(itemsize);
    const auto &symbol = type.shape[axis];
    if (!symbol.empty() && std::all_of(symbol.begin(), symbol.end(), [](char c) { return c >= '0' && c <= '9'; })) {
      if (std::stoull(symbol) != size) throw py::value_error("graph input fixed dimension mismatch");
    } else {
      const auto found = dimensions.emplace(symbol, size);
      if (!found.second && found.first->second != size)
        throw py::value_error("graph inputs must share declared symbolic dimensions");
    }
  }
  return value;
}
py::dict mapping(py::handle item, const char *name) {
  if (!PyMapping_Check(item.ptr()))
    throw py::type_error(std::string(name) + " must be a mapping");
  return py::dict(py::reinterpret_borrow<py::object>(item));
}
bool same_double(double a, double b) {
  std::uint64_t x = 0, y = 0;
  std::memcpy(&x, &a, 8);
  std::memcpy(&y, &b, 8);
  return x == y;
}

// All Python owners remain on the binding/caller thread. Native tasks see only
// Batch.
class BoundData {
public:
  std::shared_ptr<c::CompiledGraph> graph;
  std::vector<py::str> input_keys, parameter_keys;
  std::vector<ArrayPin> pins;
  ArrayPin starts, ends;
  std::optional<ArrayPin> product_ids, parameter_pin;
  std::vector<double> parameter_values;
  std::shared_ptr<SharedInputBundle> bundle;
  n::Batch native;

  BoundData(std::shared_ptr<c::CompiledGraph> g, const py::object &inputs,
            const py::array &a, const py::array &z,
            const py::object &parameters, const py::object &products,
            bool bind_parameters)
      : graph(std::move(g)), starts(ArrayPin::bind<std::int64_t>(a, "starts")),
        ends(ArrayPin::bind<std::int64_t>(z, "ends")) {
    for (const auto &name : graph->input_names) input_keys.emplace_back(name);
    for (const auto &name : graph->parameter_names) parameter_keys.emplace_back(name);
    if (starts.size != ends.size)
      throw py::value_error("starts and ends lengths must match");
    native.starts = static_cast<const std::int64_t *>(starts.data);
    native.ends = static_cast<const std::int64_t *>(ends.data);
    native.rows = starts.size;
    if (!products.is_none()) {
      product_ids = ArrayPin::bind<std::int64_t>(products, "product_ids");
      if (product_ids->size != starts.size)
        throw py::value_error("product_ids must match interval rows");
      native.product_ids = static_cast<const std::int64_t *>(product_ids->data);
    }
    std::map<std::string, std::size_t> dimensions;
    const auto input_type = [&](const std::string &name) -> const calmetrics_engine::typed::ValueType & {
      for (const auto &variable : graph->variable_types)
        if (variable.name == name) return variable.type;
      throw py::value_error("missing graph input type: " + name);
    };
    if (py::isinstance<SharedInputBundle>(inputs)) {
      bundle = py::cast<std::shared_ptr<SharedInputBundle>>(inputs);
      bundle->ensure();
      if (bundle->owners.size() != graph->input_names.size())
        throw py::value_error("graph input keys mismatch");
      for (const auto &name : graph->input_names) {
        auto found = bundle->owners.find(name);
        if (found == bundle->owners.end())
          throw py::value_error("missing graph input: " + name);
        const auto &owner = found->second;
        owner->ensure();
        auto array = owner->array();
        native.inputs.push_back(bind_graph_array(array, input_type(name), dimensions));
        native.shared_owners.push_back(owner->region);
        native.shared_inputs.push_back(
            {owner->region->name(), owner->region->size(), 0});
      }
    } else {
      auto d = mapping(inputs, "inputs");
      if (d.size() != graph->input_names.size())
        throw py::value_error("graph input keys mismatch");
      for (const auto &name : graph->input_names) {
        if (!d.contains(py::str(name)))
          throw py::value_error("missing graph input: " + name);
        auto array = b::require_array(d[py::str(name)]);
        native.inputs.push_back(bind_graph_array(array, input_type(name), dimensions));
        pins.push_back(ArrayPin::capture(array));
      }
    }
    if (bind_parameters) {
      if (py::isinstance<py::array>(parameters)) {
        parameter_pin = ArrayPin::bind<double>(parameters, "parameters");
        if (static_cast<std::size_t>(parameter_pin->size) !=
            graph->parameter_names.size())
          throw py::value_error("parameter count mismatch");
        native.parameters = static_cast<const double *>(parameter_pin->data);
        native.parameter_count = parameter_pin->size;
      } else {
        py::dict d = parameters.is_none() ? py::dict()
                                          : mapping(parameters, "parameters");
        if (d.size() != graph->parameter_names.size())
          throw py::value_error("graph parameter keys mismatch");
        for (const auto &name : graph->parameter_names) {
          if (!d.contains(py::str(name)))
            throw py::value_error("missing graph parameter: " + name);
          parameter_values.push_back(py::cast<double>(d[py::str(name)]));
        }
        native.parameters = parameter_values.data();
        native.parameter_count = parameter_values.size();
      }
    }
  }
  bool unchanged() const {
    if (!starts.unchanged() || !ends.unchanged())
      return false;
    if (product_ids && !product_ids->unchanged())
      return false;
    if (parameter_pin && !parameter_pin->unchanged())
      return false;
    if (bundle)
      bundle->ensure();
    for (const auto &pin : pins)
      if (!pin.unchanged())
        return false;
    return true;
  }
  bool matches(const std::shared_ptr<c::CompiledGraph> &g,
               const py::object &inputs, const py::array &a, const py::array &z,
               const py::object &parameters, const py::object &products) const {
    if (g.get() != graph.get() || !starts.same(a) || !ends.same(z))
      return false;
    if (products.is_none() != !product_ids)
      return false;
    if (product_ids && !product_ids->same(products))
      return false;
    if (bundle) {
      if (!py::isinstance<SharedInputBundle>(inputs) ||
          py::cast<std::shared_ptr<SharedInputBundle>>(inputs).get() !=
              bundle.get())
        return false;
      bundle->ensure();
    } else {
      if (!py::isinstance<py::dict>(inputs))
        return false;
      auto d = py::reinterpret_borrow<py::dict>(inputs);
      if (d.size() != pins.size())
        return false;
      for (std::size_t i = 0; i < pins.size(); ++i) {
        const auto &key = input_keys[i];
        if (PyDict_CheckExact(d.ptr())) {
          auto *value = PyDict_GetItemWithError(d.ptr(), key.ptr());
          if (!value && PyErr_Occurred()) throw py::error_already_set();
          if (!value || !pins[i].same(py::handle(value))) return false;
        } else if (!d.contains(key) || !pins[i].same(d[key])) return false;
      }
    }
    if (parameter_pin)
      return py::isinstance<py::array>(parameters) &&
             parameter_pin->same(parameters);
    if (parameters.is_none())
      return parameter_values.empty();
    if (!py::isinstance<py::dict>(parameters))
      return false;
    auto d = py::reinterpret_borrow<py::dict>(parameters);
    if (d.size() != parameter_values.size())
      return false;
    for (std::size_t i = 0; i < parameter_values.size(); ++i) {
      const auto &key = parameter_keys[i];
      if (PyDict_CheckExact(d.ptr())) {
        auto *value = PyDict_GetItemWithError(d.ptr(), key.ptr());
        if (!value && PyErr_Occurred()) throw py::error_already_set();
        if (!value || !same_double(parameter_values[i], py::cast<double>(py::handle(value)))) return false;
      } else if (!d.contains(key) || !same_double(parameter_values[i], py::cast<double>(d[key]))) return false;
    }
    return true;
  }
};

struct Result {
  py::array values;
  py::object offsets;
  std::shared_ptr<p::Plan> plan;
  n::ExecutionAudit native;
  bool prepared = false, cache_hit = false;
  bool snapshot = false;
  py::object statuses() const {
    if (!plan->graph->program.isolate_errors) return py::none();
    py::array_t<std::int16_t> result({values.shape(0), values.shape(1)});
    auto *dst = result.mutable_data();
    const auto columns = static_cast<std::size_t>(values.shape(1));
    if (plan->parallel_dimension == "dag_branch") {
      for (std::size_t i = 0; i < native.chunks.size(); ++i) {
        const auto task = plan->branch_tasks.at(i);
        const auto &roots = plan->graph->branches.at(task.branch_index).root_indices;
        const auto &source = native.chunks[i].statuses;
        if (source.size() != roots.size()) throw std::runtime_error("invalid native statuses");
        for (std::size_t j = 0; j < roots.size(); ++j)
          dst[task.row * columns + roots[j]] = source[j];
      }
    } else {
      std::size_t written = 0;
      for (const auto &chunk : native.chunks) {
        if (chunk.statuses.size() > static_cast<std::size_t>(result.size()) - written)
          throw std::runtime_error("invalid native statuses");
        std::copy(chunk.statuses.begin(), chunk.statuses.end(), dst + written);
        written += chunk.statuses.size();
      }
      if (written != static_cast<std::size_t>(result.size()))
        throw std::runtime_error("incomplete native statuses");
    }
    result.attr("setflags")(false);
    return result;
  }
  py::dict audit() const {
    py::dict d;
    d["execution_backend"] = "cpp_aot";
    d["audit_schema"] = "cpp-aot-execution-1";
    d["engine"] = "calmetrics_engine";
    d["engine_version"] = CALMETRICS_ENGINE_VERSION;
    d["engine_build_id"] = CALMETRICS_ENGINE_BUILD_ID;
    d["operator_registry_version"] = o::registry_version;
    d["typed_ir_version"] = "cpp-typed-ir-1";
    d["plan_fingerprint"] = plan->graph->fingerprint;
    d["source_contracts"] = plan->graph->source_contracts;
    std::size_t status_bytes = 0;
    for (const auto &chunk : native.chunks) status_bytes += chunk.statuses.size() * 2;
    d["status_result_copy_bytes"] = status_bytes;
    d["status_transport_bytes"] = plan->lane == "process" ? status_bytes : 0;
    d["native_aot"] = true;
    d["python_fallback"] = 0;
    d["python_operator_calls"] = 0;
    d["request_time_compilation"] = 0;
    py::dict input_dtypes;
    std::string common_dtype;
    for (const auto &name : plan->graph->input_names) {
      for (const auto &variable : plan->graph->variable_types) {
        if (variable.name != name) continue;
        const std::string dtype = calmetrics_engine::typed::dtype_name(variable.type.dtype);
        input_dtypes[py::str(name)] = dtype;
        common_dtype = common_dtype.empty() ? dtype : (common_dtype == dtype ? dtype : "mixed");
      }
    }
    d["input_dtype"] = common_dtype.empty() ? "float64" : common_dtype;
    d["input_dtypes"] = std::move(input_dtypes);
    d["output_dtype"] = calmetrics_engine::graph::output_dtype_name(plan->graph->program.output_dtype);
    d["error_policy"] = plan->graph->program.isolate_errors ? "isolate" : "raise";
    d["status_contract"] = "indicator-status-1";
    d["result_lifetime"] = prepared && !snapshot ? "borrowed_until_next_run" : "independent";
    d["lane"] = plan->lane;
    d["elapsed_ms"] = native.elapsed_ms;
    d["queue_wait_ms"] = native.queue_wait_ms;
    d["compute_ms"] = native.compute_ms;
    d["cpu_tokens"] = native.cpu_tokens;
    d["cpu_budget"] = native.cpu_budget;
    d["process_count"] = plan->process_count;
    d["thread_count"] = plan->thread_count;
    d["threads_per_process"] = plan->threads_per_process;
    d["async_orchestration"] = plan->async_orchestration;
    d["hard_stop"] = plan->hard_stop;
    d["use_shared_memory"] = plan->use_shared_memory;
    d["reason_codes"] = plan->reason_codes;
    d["parallel_dimension"] = plan->parallel_dimension;
    d["product_count"] = plan->product_count;
    d["estimated_work_units"] = plan->estimated_work_units;
    d["prepared"] = prepared;
    d["prepared_cached"] = cache_hit;
    d["shared_memory_bytes"] = native.shared_memory_bytes;
    d["boundary_copy_bytes"] = native.boundary_copy_bytes;
    d["output_copy_bytes"] = native.output_copy_bytes;
    d["scheduler_backend"] = "cpp";
    d["native_threads"] = native.native_threads;
    d["native_processes"] = native.native_processes;
    d["python_worker_callbacks"] = 0;
    d["python_native_transitions"] = 1;
    d["output_ownership"] =
        native.output_owner ? "native_shared_region" : "numpy_owned";
    d["output_kind"] =
        plan->graph->program.output_kind == calmetrics_engine::graph::OutputKind::series
            ? "series"
            : "scalar";
    py::list chunks;
    for (const auto &a : native.chunks)
      chunks.append(b::graph_audit(a));
    d["native_chunks"] = chunks;
    return d;
  }
};
py::dtype output_dtype(const calmetrics_engine::graph::Program &program) {
  switch (program.output_dtype) {
  case calmetrics_engine::graph::OutputDType::float64: return py::dtype::of<double>();
  case calmetrics_engine::graph::OutputDType::boolean: return py::dtype::of<bool>();
  case calmetrics_engine::graph::OutputDType::int64: return py::dtype::of<std::int64_t>();
  }
  throw py::value_error("GRAPH_OUTPUT_DTYPE");
}
py::array new_output(std::size_t rows, std::size_t columns,
                     const calmetrics_engine::graph::Program &program) {
  p::checked_mul(p::checked_mul(rows, columns),
                 calmetrics_engine::graph::output_itemsize(program.output_dtype));
  return py::array(output_dtype(program),
      {static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(columns)});
}
std::size_t output_rows(const c::CompiledGraph &graph, const n::Batch &batch) {
  if (graph.program.output_kind == calmetrics_engine::graph::OutputKind::scalar)
    return batch.rows;
  std::size_t rows = 0;
  for (std::size_t i = 0; i < batch.rows; ++i) {
    if (batch.starts[i] < 0 || batch.ends[i] < batch.starts[i])
      throw py::value_error("invalid interval bounds");
    rows = p::checked_add(
        rows, static_cast<std::size_t>(batch.ends[i] - batch.starts[i]));
  }
  return rows;
}
py::object output_offsets(const c::CompiledGraph &graph, const n::Batch &batch) {
  if (graph.program.output_kind != calmetrics_engine::graph::OutputKind::series)
    return py::none();
  py::array_t<std::int64_t> offsets(batch.rows + 1);
  auto *data = offsets.mutable_data();
  data[0] = 0;
  for (std::size_t i = 0; i < batch.rows; ++i)
    data[i + 1] = data[i] + (batch.ends[i] - batch.starts[i]);
  return offsets;
}
void same_graph(const std::shared_ptr<c::CompiledGraph> &g,
                const std::shared_ptr<p::Plan> &plan) {
  if (plan->graph.get() == g.get())
    return;
  if (plan->graph->fingerprint != g->fingerprint ||
      c::encode_program(plan->graph->program) != c::encode_program(g->program))
    throw py::value_error("execution plan belongs to a different graph");
}
class PreparedExecution {
public:
  std::shared_ptr<n::Engine> engine;
  std::shared_ptr<BoundData> bound;
  std::shared_ptr<p::Plan> plan;
  py::array output;
  py::object offsets;
  const void *output_address;
  std::mutex mutex;
  PreparedExecution(std::shared_ptr<n::Engine> e, std::shared_ptr<BoundData> b,
                    std::shared_ptr<p::Plan> p)
      : engine(std::move(e)), bound(std::move(b)), plan(std::move(p)),
        output(new_output(output_rows(*plan->graph, bound->native),
                          plan->graph->program.roots.size(), plan->graph->program)),
        offsets(output_offsets(*plan->graph, bound->native)),
        output_address(output.data()) {}
  n::ExecutionAudit execute(void *destination = nullptr) {
    if (!bound->unchanged())
      throw py::value_error(
          "PREPARED_INPUT_CHANGED: rebind resized or retyped arrays");
    if (output.ndim() != 2 ||
        output.shape(0) != static_cast<py::ssize_t>(
                               output_rows(*plan->graph, bound->native)) ||
        output.shape(1) !=
            static_cast<py::ssize_t>(plan->graph->program.roots.size()) ||
        output.data() != output_address || !output.writeable() ||
        !output.dtype().equal(output_dtype(plan->graph->program)) ||
        !(output.flags() & py::array::c_style))
      throw py::value_error("PREPARED_OUTPUT_CHANGED");
    auto *data = output.mutable_data();
    py::gil_scoped_release release;
    std::unique_lock<std::mutex> guard(mutex, std::try_to_lock);
    if (!guard.owns_lock())
      throw std::runtime_error("PREPARED_BATCH_BUSY");
    return engine->execute(*plan, bound->native, destination ? destination : data);
  }
  py::array run() {
    const auto audit = execute();
    if (plan->graph->program.output_dtype != calmetrics_engine::graph::OutputDType::float64)
      for (const auto &chunk : audit.chunks)
        if (std::any_of(chunk.statuses.begin(), chunk.statuses.end(),
                        [](auto status) { return status != 0; }))
          throw py::value_error(
              "TYPED_OUTPUT_STATUS_REQUIRED: use run_audit or run_snapshot to retain validity statuses");
    return output;
  }
  Result run_audit() {
    auto a = execute();
    return {output, offsets, plan, std::move(a), true, true};
  }
  Result run_snapshot() {
    // Write directly into independent output while holding the execution lock.
    // Copying the borrowed output after releasing that lock would race another run.
    auto owned = new_output(output_rows(*plan->graph, bound->native),
                            plan->graph->program.roots.size(), plan->graph->program);
    auto a = execute(owned.mutable_data());
    owned.attr("setflags")(false);
    return {owned, output_offsets(*plan->graph, bound->native), plan,
            std::move(a), true, true, true};
  }
};
std::string worker_path(const py::object &path) {
  if (!path.is_none())
    return py::cast<std::string>(py::str(path));
  auto extension = py::cast<std::string>(
      py::module_::import("calmetrics_engine._native").attr("__file__"));
#ifdef _WIN32
  return (std::filesystem::path(extension).parent_path() /
          "calmetrics_worker.exe")
      .string();
#else
  return (std::filesystem::path(extension).parent_path() / "calmetrics_worker")
      .string();
#endif
}
class EngineAPI {
public:
  std::shared_ptr<n::Engine> engine;
  std::shared_ptr<BoundData> cached;
  EngineAPI(std::optional<std::size_t> cpu, std::optional<p::Config> config,
            const py::object &path)
      : engine(std::make_shared<n::Engine>(default_cpu(cpu),
                                           config.value_or(p::Config{}),
                                           worker_path(path))) {}
  ~EngineAPI() { engine->close(); }
  std::shared_ptr<p::Plan> plan(std::shared_ptr<c::CompiledGraph> graph,
                                const py::object &inputs,
                                const py::array &starts, const py::array &ends,
                                std::optional<std::size_t> memory,
                                bool hard_stop, bool async_io,
                                const py::object &products) const {
    BoundData bound(graph, inputs, starts, ends, py::none(), products, false);
    py::gil_scoped_release release;
    return engine->plan(std::move(graph), bound.native, memory, hard_stop,
                        async_io);
  }
  std::shared_ptr<BoundData>
  bind(std::shared_ptr<c::CompiledGraph> graph, const py::object &inputs,
       const py::array &starts, const py::array &ends, const py::object &params,
       const py::object &products, bool &hit) {
    hit = cached &&
          cached->matches(graph, inputs, starts, ends, params, products);
    if (hit)
      return cached;
    auto bound = std::make_shared<BoundData>(std::move(graph), inputs, starts,
                                             ends, params, products, true);
    // Shared bundles own explicit lifetimes; do not retain them in an implicit
    // ordinary-call cache.
    if (!bound->bundle)
      cached = bound;
    return bound;
  }
  Result execute(std::shared_ptr<c::CompiledGraph> graph,
                 const py::object &inputs, const py::array &starts,
                 const py::array &ends, const py::object &params,
                 std::shared_ptr<p::Plan> plan,
                 std::optional<std::size_t> memory, bool hard_stop,
                 std::optional<double> timeout, const py::object &products,
                 bool async_io) {
    bool hit = false;
    auto bound = bind(graph, inputs, starts, ends, params, products, hit);
    if (plan) {
      same_graph(graph, plan);
      if (hard_stop && !plan->hard_stop)
        throw py::value_error("provided plan does not satisfy hard_stop");
      if (memory && plan->estimated_total_memory_bytes > *memory)
        throw std::bad_alloc();
      if (async_io && !plan->async_orchestration) {
        plan = std::make_shared<p::Plan>(*plan);
        plan->async_orchestration = true;
        plan->reason_codes.push_back(
            "coroutine_orchestration_for_async_boundary");
      }
    } else {
      py::gil_scoped_release release;
      plan = engine->plan(graph, bound->native, memory, hard_stop, async_io);
    }
    const bool shared_output =
        plan->lane == "process" && plan->use_shared_memory;
    py::array output;
    void *data = nullptr;
    if (!shared_output) {
      output = new_output(output_rows(*graph, bound->native),
                          graph->program.roots.size(), graph->program);
      data = output.mutable_data();
    }
    n::ExecutionAudit audit;
    {
      py::gil_scoped_release release;
      audit =
          engine->execute(*plan, bound->native, data, timeout, shared_output);
    }
    if (shared_output) {
      if (!audit.output_owner)
        throw std::runtime_error("native shared output owner missing");
      const std::vector<py::ssize_t> shape{
          static_cast<py::ssize_t>(output_rows(*graph, bound->native)),
          static_cast<py::ssize_t>(graph->program.roots.size())};
      output =
          py::array(output_dtype(graph->program), shape, {},
                    audit.output_owner->data(), py::cast(audit.output_owner));
    }
    return {std::move(output), output_offsets(*graph, bound->native),
            std::move(plan), std::move(audit), false, hit};
  }
  std::shared_ptr<PreparedExecution>
  prepare(std::shared_ptr<c::CompiledGraph> graph, const py::object &inputs,
          const py::array &starts, const py::array &ends,
          const py::object &params, std::shared_ptr<p::Plan> plan,
          const py::object &products) {
    if (plan && plan->lane != "single")
      throw py::value_error(
          "prepared execution currently requires a single-lane plan");
    auto bound = std::make_shared<BoundData>(graph, inputs, starts, ends,
                                             params, products, true);
    if (plan)
      same_graph(graph, plan);
    else {
      py::gil_scoped_release release;
      plan = engine->plan(graph, bound->native);
    }
    if (plan->lane != "single")
      throw py::value_error(
          "prepared execution currently requires a single-lane plan");
    {
      py::gil_scoped_release release;
      p::validate_plan(*plan, bound->native.input_sizes(), bound->native.starts,
                       bound->native.ends, bound->native.rows,
                       bound->native.product_ids, engine->cpu(), &bound->native.inputs);
    }
    return std::make_shared<PreparedExecution>(engine, bound, plan);
  }
  void close() {
    {
      py::gil_scoped_release release;
      engine->close();
    }
    cached.reset();
  }
};
class PlannerAPI {
public:
  p::Config config;
  explicit PlannerAPI(std::optional<p::Config> c)
      : config(c.value_or(p::Config{})) {
    config.validate();
  }
  std::shared_ptr<p::Plan> plan(std::shared_ptr<c::CompiledGraph> graph,
                                const py::object &inputs,
                                const py::array &starts, const py::array &ends,
                                std::optional<std::size_t> cpu,
                                std::optional<std::size_t> memory,
                                bool hard_stop, bool async_io, bool shared,
                                const py::object &products) const {
    BoundData bound(graph, inputs, starts, ends, py::none(), products, false);
    py::gil_scoped_release release;
    return p::make_plan(graph, config, bound.native.input_sizes(),
                        bound.native.starts, bound.native.ends,
                        bound.native.rows, bound.native.product_ids,
                        default_cpu(cpu), memory, hard_stop, async_io,
                        shared || !bound.native.shared_inputs.empty(), &bound.native.inputs);
  }
};
} // namespace
void register_native_api(py::module_ &module) {
  py::register_exception<n::Timeout>(module, "ExecutionTimeout",
                                     PyExc_TimeoutError);
  py::class_<n::SharedRegion, std::shared_ptr<n::SharedRegion>>(
      module, "_SharedRegion", py::buffer_protocol())
      .def_buffer([](n::SharedRegion &r) {
        return py::buffer_info(
            r.data(), 1, py::format_descriptor<std::uint8_t>::format(), 1,
            {static_cast<py::ssize_t>(r.size())}, {1}, !r.writable());
      });
  py::class_<SharedArrayDescriptor>(module, "SharedArrayDescriptor")
      .def(py::init([](std::string name, std::vector<py::ssize_t> shape,
                       std::string dtype, bool readonly) {
             SharedArrayDescriptor d{std::move(name), std::move(shape),
                                     std::move(dtype), readonly};
             d.nbytes();
             return d;
           }),
           py::arg("name"), py::arg("shape"), py::arg("dtype"),
           py::arg("readonly") = false)
      .def_readonly("name", &SharedArrayDescriptor::name)
      .def_readonly("dtype", &SharedArrayDescriptor::dtype)
      .def_readonly("readonly", &SharedArrayDescriptor::readonly)
      .def_property_readonly("shape",
                             [](const SharedArrayDescriptor &d) {
                               return py::tuple(py::cast(d.shape));
                             })
      .def_property_readonly("nbytes", &SharedArrayDescriptor::nbytes);
  py::class_<SharedArrayOwner, std::shared_ptr<SharedArrayOwner>>(
      module, "SharedArrayOwner")
      .def_static("from_array", &SharedArrayOwner::from_array, py::arg("value"))
      .def_static("empty", &SharedArrayOwner::empty, py::arg("shape"),
                  py::arg("dtype"))
      .def_property_readonly("array", &SharedArrayOwner::array)
      .def_property_readonly("descriptor", &SharedArrayOwner::descriptor)
      .def_property_readonly("nbytes", &SharedArrayOwner::nbytes)
      .def("close", &SharedArrayOwner::close)
      .def("unlink", &SharedArrayOwner::unlink)
      .def("release", &SharedArrayOwner::release)
      .def("__enter__",
           [](std::shared_ptr<SharedArrayOwner> self) {
             self->ensure();
             return self;
           })
      .def("__exit__", [](SharedArrayOwner &self, py::object, py::object,
                          py::object) { self.release(); });
  py::class_<SharedInputBundle, std::shared_ptr<SharedInputBundle>>(
      module, "SharedInputBundle")
      .def_static("from_inputs", &SharedInputBundle::from_inputs,
                  py::arg("inputs"))
      .def_property_readonly("arrays", &SharedInputBundle::arrays)
      .def_property_readonly("descriptors", &SharedInputBundle::descriptors)
      .def_property_readonly("nbytes", &SharedInputBundle::nbytes)
      .def_readonly("boundary_copy_bytes",
                    &SharedInputBundle::boundary_copy_bytes)
      .def("release", &SharedInputBundle::release)
      .def("__enter__",
           [](std::shared_ptr<SharedInputBundle> self) {
             self->ensure();
             return self;
           })
      .def("__exit__", [](SharedInputBundle &self, py::object, py::object,
                          py::object) { self.release(); });
  module.def(
      "attach_shared_array",
      [](const SharedArrayDescriptor &d, bool readonly) {
        const bool effective_readonly = readonly || d.readonly;
        auto owner = std::make_shared<SharedArrayOwner>(
            n::SharedRegion::attach(d.name, d.nbytes(), !effective_readonly),
            d.shape, py::dtype(d.dtype), effective_readonly);
        return py::make_tuple(owner, owner->array());
      },
      py::arg("descriptor"), py::kw_only(), py::arg("readonly") = true);
  py::class_<Result>(module, "GraphExecutionResult")
      .def_readonly("values", &Result::values)
      .def_readonly("offsets", &Result::offsets)
      .def_property_readonly("statuses", &Result::statuses)
      .def_property_readonly(
          "output_kind",
          [](const Result &r) {
            return r.plan->graph->program.output_kind ==
                           calmetrics_engine::graph::OutputKind::series
                       ? "series"
                       : "scalar";
          })
      .def_readonly("plan", &Result::plan)
      .def_property_readonly("audit", &Result::audit);
  py::class_<PreparedExecution, std::shared_ptr<PreparedExecution>>(
      module, "PreparedGraphExecution")
      .def_readonly("plan", &PreparedExecution::plan)
      .def_readonly("output", &PreparedExecution::output)
      .def_readonly("offsets", &PreparedExecution::offsets)
      .def("run", &PreparedExecution::run)
      .def("run_audit", &PreparedExecution::run_audit)
      .def("run_snapshot", &PreparedExecution::run_snapshot);
  py::class_<PlannerAPI>(module, "AdaptivePlanner")
      .def(py::init<std::optional<p::Config>>(), py::arg("config") = py::none())
      .def_readonly("config", &PlannerAPI::config)
      .def("plan", &PlannerAPI::plan, py::arg("graph"), py::arg("inputs"),
           py::arg("starts").noconvert(), py::arg("ends").noconvert(),
           py::kw_only(), py::arg("cpu_budget") = py::none(),
           py::arg("memory_budget_bytes") = py::none(),
           py::arg("hard_stop") = false, py::arg("async_io") = false,
           py::arg("inputs_already_shared") = false,
           py::arg("product_ids") = py::none());
  py::class_<EngineAPI, std::shared_ptr<EngineAPI>>(module, "AdaptiveScheduler")
      .def(py::init<std::optional<std::size_t>, std::optional<p::Config>,
                    const py::object &>(),
           py::kw_only(), py::arg("cpu_budget") = py::none(),
           py::arg("config") = py::none(), py::arg("worker_path") = py::none())
      .def_property_readonly("cpu_budget",
                             [](const EngineAPI &e) { return e.engine->cpu(); })
      .def_property_readonly(
          "config", [](const EngineAPI &e) { return e.engine->config(); })
      .def_property_readonly(
          "planner",
          [](const EngineAPI &e) { return PlannerAPI(e.engine->config()); })
      .def_property_readonly(
          "peak_active_cpu_tokens",
          [](const EngineAPI &e) { return e.engine->peak_active(); })
      .def("plan", &EngineAPI::plan, py::arg("graph"), py::arg("inputs"),
           py::arg("starts").noconvert(), py::arg("ends").noconvert(),
           py::kw_only(), py::arg("memory_budget_bytes") = py::none(),
           py::arg("hard_stop") = false, py::arg("async_io") = false,
           py::arg("product_ids") = py::none())
      .def("execute", &EngineAPI::execute, py::arg("graph"), py::arg("inputs"),
           py::arg("starts").noconvert(), py::arg("ends").noconvert(),
           py::kw_only(), py::arg("parameters") = py::none(),
           py::arg("plan") = py::none(),
           py::arg("memory_budget_bytes") = py::none(),
           py::arg("hard_stop") = false, py::arg("timeout") = py::none(),
           py::arg("product_ids") = py::none(), py::arg("async_io") = false)
      .def("prepare_execution", &EngineAPI::prepare, py::arg("graph"),
           py::arg("inputs"), py::arg("starts").noconvert(),
           py::arg("ends").noconvert(), py::kw_only(),
           py::arg("parameters") = py::none(), py::arg("plan") = py::none(),
           py::arg("product_ids") = py::none())
      .def("close", &EngineAPI::close)
      .def("__enter__",
           [](std::shared_ptr<EngineAPI> self) {
             if (self->engine->closed())
               throw std::runtime_error("native engine is closed");
             return self;
           })
      .def("__exit__", [](EngineAPI &self, py::object, py::object, py::object) {
        self.close();
      });
  module.def(
      "product_chunks",
      [](const py::array &products, const py::array &starts,
         const py::array &ends, std::size_t workers) {
        b::exact<std::int64_t>(products, 1, "product_ids");
        b::exact<std::int64_t>(starts, 1, "starts");
        b::exact<std::int64_t>(ends, 1, "ends");
        if (products.size() != starts.size() || starts.size() != ends.size() ||
            workers == 0)
          throw py::value_error("partition shape mismatch");
        std::vector<p::Chunk> chunks;
        {
          py::gil_scoped_release release;
          auto geometry = p::inspect(
              {static_cast<std::size_t>(PTRDIFF_MAX)},
              static_cast<const std::int64_t *>(starts.data()),
              static_cast<const std::int64_t *>(ends.data()), starts.size(),
              static_cast<const std::int64_t *>(products.data()));
          chunks = p::partition(geometry, workers, true);
        }
        py::list result;
        for (const auto &chunk : chunks)
          result.append(py::make_tuple(chunk.begin, chunk.end));
        return result;
      },
      py::arg("product_ids"), py::arg("starts"), py::arg("ends"),
      py::arg("workers"));
}
