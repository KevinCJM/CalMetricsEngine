"""Python-facing names for C++ RAII shared mappings and array owners.

Creator unlink and mapping lifetimes are native. A live NumPy view retains its
mapping after an owner or bundle is released; no Python resource tracker is used.
"""

from ._native.graph import (
    SharedArrayDescriptor,
    SharedArrayOwner,
    SharedInputBundle,
    attach_shared_array,
)

__all__ = [
    "SharedArrayDescriptor",
    "SharedArrayOwner",
    "SharedInputBundle",
    "attach_shared_array",
]
