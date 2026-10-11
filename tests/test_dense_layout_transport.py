"""Physical C/F storage survives shared and model input transport."""

import gc
import struct

import numpy as np
import pytest

from calmetrics_engine import (
    AdaptiveScheduler,
    GraphCompiler,
    ModelPayload,
    SharedArrayDescriptor,
    SharedArrayOwner,
    SharedInputBundle,
)
from calmetrics_engine.shared import attach_shared_array
from tests.test_model_payload import META
from tests.test_tensor_protocol import configuration


def source(shape, dtype, order):
    value = np.arange(np.prod(shape, dtype=int)).reshape(shape)
    if dtype == "bool":
        value = value % 3 == 1
    elif dtype == "int64":
        value = value + 2**53 + 1
    return np.array(value, dtype=dtype, order=order)


def field_type(dtype, shape):
    return {
        "kind": ["value", "vector", "matrix", "tensor"][len(shape)],
        "dtype": dtype,
        "axes": ["scenario", "path", "asset"][: len(shape)],
        "shape": list(map(str, shape)),
    }


@pytest.mark.parametrize("shape", [(), (7,), (5, 3), (4, 3, 2), (2, 3, 2, 2), (0, 3, 2), (1, 3, 1)])
@pytest.mark.parametrize("dtype", ["float64", "int64", "bool"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_shared_layout_bytes_geometry_and_lifetime(shape, dtype, order):
    x = source(shape, dtype, order)
    owner = SharedArrayOwner.from_array(x)
    view = owner.array
    descriptor = owner.descriptor
    assert descriptor.order == ("C" if x.flags.c_contiguous else "F")
    assert descriptor.strides == view.strides
    assert view.shape == shape and view.dtype == x.dtype
    assert view.flags.c_contiguous == x.flags.c_contiguous
    assert view.flags.f_contiguous == x.flags.f_contiguous
    assert view.tobytes(order="A") == x.tobytes(order="A")
    assert not np.shares_memory(x, view)
    np.testing.assert_array_equal(view, x)
    attached_owner, attached = attach_shared_array(descriptor)
    assert attached.strides == view.strides
    assert not attached.flags.writeable
    with pytest.raises(ValueError):
        attached.setflags(write=True)
    owner.release()
    attached_owner.close()
    gc.collect()
    np.testing.assert_array_equal(attached, x)
    np.testing.assert_array_equal(view, x)


@pytest.mark.parametrize("order", ["C", "F"])
def test_shared_empty_explicit_order_and_backward_compatible_descriptor(order):
    with SharedArrayOwner.empty((5, 3), np.int64, order=order) as owner:
        view = owner.array
        view[:] = np.arange(15).reshape(5, 3)
        assert owner.descriptor.order == order
        assert view.strides == np.empty((5, 3), dtype=np.int64, order=order).strides
        attached_owner, attached = attach_shared_array(owner.descriptor, readonly=False)
        attached[2, 1] = 71
        assert view[2, 1] == 71
        attached_owner.close()
    with SharedArrayOwner.empty((5, 3), np.float64) as owner:
        old = SharedArrayDescriptor(owner.descriptor.name, (5, 3), "<f8")
        assert old.order == "C" and old.strides == (24, 8)


@pytest.mark.parametrize("order", ["A", "K", "f", "", "CF"])
def test_unknown_shared_order_rejected_before_allocating_or_attaching(order):
    with pytest.raises(ValueError, match="order"):
        SharedArrayOwner.empty((0, 2), np.float64, order=order)
    with pytest.raises(ValueError, match="order"):
        SharedArrayDescriptor("nonexistent", (0, 2), "<f8", order=order)


def test_shared_rejects_non_dense_and_overflow():
    x = np.arange(30.0).reshape(6, 5)
    for view in (x[::2], x[::-1], np.broadcast_to(x[:1], x.shape)):
        with pytest.raises(ValueError, match="contiguous"):
            SharedArrayOwner.from_array(view)
    with pytest.raises((ValueError, OverflowError)):
        SharedArrayOwner.empty((2**62, 4), np.uint8, order="F")
    with SharedArrayOwner.empty((3, 2), np.float64, order="F") as owner:
        descriptor = SharedArrayDescriptor(owner.descriptor.name, (10000, 2), "<f8", order="F")
        with pytest.raises((ValueError, RuntimeError)):
            attach_shared_array(descriptor)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("kind", ["dict", "bundle", "model"])
@pytest.mark.parametrize("dtype", ["float64", "int64", "bool"])
@pytest.mark.parametrize("shape", [(5, 3), (4, 3, 2)])
def test_fortran_inputs_across_lanes_and_retained_results(lane, kind, dtype, shape):
    x = source(shape, dtype, "F")
    types = {"x": field_type(dtype, shape)}
    inputs = {"x": x}
    if kind == "bundle":
        inputs = SharedInputBundle.from_inputs(inputs)
        assert inputs.descriptors["x"].order == "F"
        assert inputs.arrays["x"].strides == x.strides
        assert inputs.boundary_copy_bytes == x.nbytes
    elif kind == "model":
        inputs = ModelPayload.from_fields(types, inputs, META)
        assert inputs.fields["x"].strides == x.strides
    graph = GraphCompiler(types).compile({"x": "x"}, error_policy="isolate")
    starts, ends = np.array([0, 0], np.int64), np.array([1, 1], np.int64)
    expected = x.copy()
    with AdaptiveScheduler(cpu_budget=2, config=configuration(lane)) as engine:
        first = engine.execute(graph, inputs, starts, ends)
        assert first.plan.lane == ("process" if lane.startswith("process") else lane)
        if lane.startswith("process"):
            assert first.plan.use_shared_memory == (lane == "process_shared" or kind != "dict")
            if kind != "dict":
                assert first.audit["boundary_copy_bytes"] == starts.nbytes + ends.nbytes
        if kind == "dict":
            x[...] = 0
        engine.execute(graph, inputs, starts, ends)
    if kind == "bundle":
        inputs.release()
    for value in first.outputs[0].values:
        np.testing.assert_array_equal(value, expected)
        assert not np.shares_memory(value, x)


@pytest.mark.parametrize("dtype", ["float64", "int64", "bool"])
@pytest.mark.parametrize("shape", [(5, 3), (4, 3, 2)])
def test_model_layout_version_and_identity_roundtrip(dtype, shape):
    x = source(shape, dtype, "F")
    types = {"x": field_type(dtype, shape)}
    model = ModelPayload.from_fields(types, {"x": x}, META)
    encoded = model.to_bytes()
    assert struct.unpack_from("<Q", encoded)[0] == 0x434D454D00000002
    assert model.identity.startswith("model-2-")
    restored = ModelPayload.from_bytes(encoded)
    assert restored.identity == model.identity
    assert restored.to_bytes() == encoded
    assert restored.fields["x"].strides == x.strides
    assert not restored.fields["x"].flags.writeable
    np.testing.assert_array_equal(restored.fields["x"], x)
    # C payloads keep the historical version and identity rules.
    c_model = ModelPayload.from_fields(types, {"x": x.copy(order="C")}, META)
    assert struct.unpack_from("<Q", c_model.to_bytes())[0] == 0x434D454D00000001
    assert c_model.identity.startswith("model-1-")
    assert c_model.identity != model.identity
    assert ModelPayload.from_bytes(c_model.to_bytes()).identity == c_model.identity
    # A one-field v2 layout word precedes the payload byte count and aligned data.
    offset = len(encoded) - model.nbytes - 16
    for order in (2, 255, 2**63):
        corrupt = bytearray(encoded)
        struct.pack_into("<Q", corrupt, offset, order)
        with pytest.raises((ValueError, RuntimeError), match="MODEL_MEMORY_ORDER"):
            ModelPayload.from_bytes(bytes(corrupt))
    corrupt = bytearray(encoded)
    struct.pack_into("<Q", corrupt, offset, 0)
    with pytest.raises((ValueError, RuntimeError), match="MODEL_NONCANONICAL_VERSION"):
        ModelPayload.from_bytes(bytes(corrupt))
    for bad in (encoded[:-1], encoded + b"x", encoded[: offset + 4]):
        with pytest.raises((ValueError, RuntimeError)):
            ModelPayload.from_bytes(bad)


def test_mixed_model_fields_keep_each_order_alignment_and_scalar():
    fields = {
        "f": source((5, 3), "float64", "F"),
        "c": source((2, 3, 2), "int64", "C"),
        "mask": source((5, 3), "bool", "F"),
        "scalar": np.array(2**53 + 9, np.int64),
    }
    types = {name: field_type(str(value.dtype), value.shape) for name, value in fields.items()}
    model = ModelPayload.from_fields(types, fields, META)
    restored = ModelPayload.from_bytes(model.to_bytes())
    assert restored.identity == model.identity
    for name, expected in fields.items():
        actual = restored.fields[name]
        assert actual.ctypes.data % 8 == 0
        assert actual.flags.c_contiguous == expected.flags.c_contiguous
        assert actual.flags.f_contiguous == expected.flags.f_contiguous
        np.testing.assert_array_equal(actual, expected)
    views = restored.fields
    del model, restored
    gc.collect()
    for name, expected in fields.items():
        np.testing.assert_array_equal(views[name], expected)


def test_v1_identity_and_exact_bytes_match_pre_layout_build():
    # Frozen with pre-layout build a58228d2: includes negative strides, exact
    # int64, signed zero and a NaN payload. Round-tripping only a newly encoded
    # model would not establish compatibility with historical v1 bytes.
    import hashlib

    from tests.test_model_payload import fixture

    fields, types = fixture()
    bits = fields["weights"].view(np.uint64)
    bits.flat[0] = 0x8000000000000000
    bits.flat[1] = 0x7FF8000000000007
    model = ModelPayload.from_fields(types, fields, META)
    encoded = model.to_bytes()
    assert model.identity == "model-1-395eefb26e0b82a10ffbdbf54c067432"
    assert (
        hashlib.sha256(encoded).hexdigest()
        == "79c46ed1f7653029efb35226974569964298f09c2b61ea146b45331e1bcb3c06"
    )
    assert ModelPayload.from_bytes(encoded).to_bytes() == encoded
