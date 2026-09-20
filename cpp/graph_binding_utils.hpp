#pragma once
#include "calmetrics_engine/operators.hpp"
#include "calmetrics_engine/planner.hpp"
#include "calmetrics_engine/shared_memory.hpp"
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace calmetrics_engine::binding {
namespace py = pybind11;
py::dict graph_audit(const graph::Audit &audit);
py::dict plan_metadata(const planner::Plan &plan);
struct Bounds {
  std::uintptr_t lo = 0, hi = 0;
};
inline Bounds bounds(const py::array &array) {
  const auto address = reinterpret_cast<std::uintptr_t>(array.data());
  if (!array.size())
    return {address, address};
  std::size_t below = 0, above = 0;
  for (py::ssize_t axis = 0; axis < array.ndim(); ++axis) {
    if (array.shape(axis) <= 1)
      continue;
    const auto step = array.strides(axis);
    const auto magnitude = step < 0 ? static_cast<std::size_t>(-(step + 1)) + 1
                                    : static_cast<std::size_t>(step);
    const auto extent = static_cast<std::size_t>(array.shape(axis) - 1);
    ops::require(!magnitude || extent <= static_cast<std::size_t>(PTRDIFF_MAX) /
                                             magnitude,
                 "STRIDE_OVERFLOW");
    auto &total = step < 0 ? below : above;
    const auto span = extent * magnitude;
    ops::require(total <= static_cast<std::size_t>(PTRDIFF_MAX) - span,
                 "STRIDE_OVERFLOW");
    total += span;
  }
  ops::require(address >= below && above <= UINTPTR_MAX - address &&
                   static_cast<std::size_t>(array.itemsize()) <=
                       UINTPTR_MAX - address - above,
               "STRIDE_OVERFLOW");
  return {address - below,
          address + above + static_cast<std::size_t>(array.itemsize())};
}
inline void validate_owner(const py::array &array) {
  if (!array.size())
    return;
  const auto view = bounds(array);
  py::object owner = py::reinterpret_borrow<py::object>(array);
  for (unsigned depth = 0; depth < 32; ++depth) {
    if (py::isinstance<py::array>(owner)) {
      auto current = py::reinterpret_borrow<py::array>(owner);
      if (current.owndata()) {
        auto allocation = bounds(current);
        ops::require(view.lo >= allocation.lo && view.hi <= allocation.hi,
                     "INPUT_OUT_OF_BOUNDS");
        return;
      }
    }
    if (py::isinstance<native::SharedRegion>(owner)) {
      auto region = py::cast<std::shared_ptr<native::SharedRegion>>(owner);
      const auto lo = reinterpret_cast<std::uintptr_t>(region->data());
      ops::require(view.lo >= lo && view.hi <= lo + region->size(),
                   "INPUT_OUT_OF_BOUNDS");
      return;
    }
    auto base = py::getattr(owner, "base", py::none());
    if (!base.is_none()) {
      ops::require(!base.is(owner), "INVALID_ARRAY_OWNER");
      owner = base;
      continue;
    }
    Py_buffer buffer{};
    if (PyObject_GetBuffer(owner.ptr(), &buffer, PyBUF_SIMPLE) == 0) {
      auto lo = reinterpret_cast<std::uintptr_t>(buffer.buf);
      auto size = static_cast<std::size_t>(buffer.len);
      bool valid =
          size <= UINTPTR_MAX - lo && view.lo >= lo && view.hi <= lo + size;
      PyBuffer_Release(&buffer);
      ops::require(valid, "INPUT_OUT_OF_BOUNDS");
    } else
      PyErr_Clear();
    return;
  }
  throw ops::Error("ARRAY_OWNER_CHAIN_TOO_DEEP");
}
template <class T>
inline void exact(const py::array &array, int ndim, const char *name,
                  bool writable = false) {
  if (!array.dtype().equal(py::dtype::of<T>()))
    throw py::type_error(std::string(name) + " must use exact native dtype");
  if (array.ndim() != ndim || !(array.flags() & py::array::c_style))
    throw py::value_error(std::string(name) +
                          " must be C-contiguous with expected ndim");
  if (reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T))
    throw py::value_error(std::string(name) + " must be aligned");
  if (writable && !array.writeable())
    throw py::value_error(std::string(name) + " must be writable");
  validate_owner(array);
}
inline py::array require_array(py::handle value) {
  if (!py::isinstance<py::array>(value))
    throw py::type_error(
        "expected NumPy ndarray; implicit copies are forbidden");
  return py::reinterpret_borrow<py::array>(value);
}
inline bool overlaps(const Bounds &a, const Bounds &b) {
  return a.lo < a.hi && b.lo < b.hi && a.lo < b.hi && b.lo < a.hi;
}
} // namespace calmetrics_engine::binding
