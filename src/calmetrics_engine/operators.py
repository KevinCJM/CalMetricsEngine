"""Canonical C++ operators and their immutable native registry.

No numerical implementation lives in Python. Names and metadata are exported
from the single AOT backend; callers can resolve an Operator handle once for
repeated calls. Formula parsing and DAG scheduling are not implemented here.
"""

from ._native import operators as _native_operators

REGISTRY_VERSION = _native_operators.REGISTRY_VERSION
OperatorError = _native_operators.OperatorError
Operator = _native_operators.Operator
Workspace = _native_operators.Workspace
catalog = _native_operators.catalog
get = _native_operators.get
get_by_opcode = _native_operators.get_by_opcode
call = _native_operators.call
available_simd = _native_operators.available_simd

_CANONICAL_NAMES = tuple(spec["id"] for spec in catalog())
for _name in _CANONICAL_NAMES:
    globals()[_name] = getattr(_native_operators, _name)
del _name

__all__ = [
    "REGISTRY_VERSION",
    "OperatorError",
    "Operator",
    "Workspace",
    "catalog",
    "get",
    "get_by_opcode",
    "call",
    "available_simd",
    *_CANONICAL_NAMES,
]
