"""Ordinary binding caches release borrowed owners; prepared bindings retain them.

These are lifecycle and public-boundary tests, not allocator timing tests.
Concurrent GIL-release pinning needs a separate deterministic native barrier.
"""

from __future__ import annotations

import gc
import weakref
from collections.abc import Mapping

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler


def _graph(*, failure=False):
    shifted = "sqrt(x)+k" if failure else "x+k"
    return GraphCompiler({"x": "series", "k": "scalar"}).compile(
        {"average": "mean(x)+k", "shifted": shifted}
    )


def _payload():
    # Every view has a separately observable ndarray that owns its allocation.
    x_base = np.zeros(16, dtype=np.float64)
    x_base[4:12] = np.arange(8, dtype=np.float64) * 2.0
    starts_base = np.array([91, 0, 4, 92], dtype=np.int64)
    ends_base = np.array([93, 4, 8, 94], dtype=np.int64)
    products_base = np.array([95, 0, 1, 96], dtype=np.int64)
    parameters_base = np.array([-20.0, 2.0, -30.0])
    return {
        "x_base": x_base,
        "x": x_base[4:12],
        "starts_base": starts_base,
        "starts": starts_base[1:3],
        "ends_base": ends_base,
        "ends": ends_base[1:3],
        "products_base": products_base,
        "products": products_base[1:3],
        "parameters_base": parameters_base,
        "parameters": parameters_base[1:2],
    }


def _execute(scheduler, graph, payload):
    return scheduler.execute(
        graph,
        {"x": payload["x"]},
        payload["starts"],
        payload["ends"],
        parameters=payload["parameters"],
        product_ids=payload["products"],
    )


def _refs(payload):
    return {name: weakref.ref(value) for name, value in payload.items()}


def _assert_collected(references):
    gc.collect()
    retained = [name for name, reference in references.items() if reference() is not None]
    assert retained == [], f"borrowed owners still retained: {retained}"


def _assert_values(result, *, offset=0.0, parameter=2.0):
    columns = result.named_outputs
    for row, average in enumerate((3.0, 11.0)):
        scalar = columns["average"].values[row]
        shifted = columns["shifted"].values[row]
        np.testing.assert_array_equal(scalar, offset + parameter + average)
        np.testing.assert_array_equal(
            shifted,
            np.array([offset + parameter + 2 * index for index in range(4 * row, 4 * row + 4)]),
        )
        assert scalar.shape == () and shifted.shape == (4,)
        assert scalar.dtype == shifted.dtype == np.dtype(np.float64)
        assert not scalar.flags.writeable and not shifted.flags.writeable
        assert columns["average"].root_statuses[row] == 0
        assert columns["shifted"].root_statuses[row] == 0


def _assert_no_input_alias(result, payload):
    for column in result.outputs:
        for value in column.values:
            assert all(not np.shares_memory(value, array) for array in payload.values())


def _assert_independent_results(first, second):
    for left in first.outputs:
        for right in second.outputs:
            for a in left.values:
                for b in right.values:
                    assert not np.shares_memory(a, b)


def _execute_then_forget(scheduler, graph, *, failure=False):
    payload = _payload()
    if failure:
        payload["x"][0] = -1.0
    references = _refs(payload)
    try:
        result = _execute(scheduler, graph, payload)
    except (ValueError, RuntimeError) as error:
        assert failure and "DOMAIN_ERROR" in str(error)
        result = None
    else:
        assert not failure, "the deliberately invalid sqrt request succeeded"
        _assert_values(result)
        _assert_no_input_alias(result, payload)
    # Do not retain exception tracebacks or the input dictionary in the caller.
    return result, references


@pytest.mark.parametrize("failure", [False, True], ids=["success", "raise"])
def test_ordinary_completion_releases_views_and_allocation_owners(failure):
    graph = _graph(failure=failure)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result, references = _execute_then_forget(scheduler, graph, failure=failure)
        _assert_collected(references)
        if result is not None:
            _assert_values(result)
        # A numerical error must not prevent using the engine with fresh owners.
        replacement, replacement_refs = _execute_then_forget(scheduler, _graph())
        _assert_collected(replacement_refs)
        _assert_values(replacement)
        if result is not None:
            _assert_values(result)
            _assert_independent_results(result, replacement)


def _prepare_then_forget(scheduler, graph):
    payload = _payload()
    references = _refs(payload)
    prepared = scheduler.prepare_execution(
        graph,
        {"x": payload["x"]},
        payload["starts"],
        payload["ends"],
        parameters=payload["parameters"],
        product_ids=payload["products"],
    )
    return prepared, references


def test_prepared_keeps_owners_until_destroyed_and_snapshots_remain_independent():
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared, references = _prepare_then_forget(scheduler, _graph())
        gc.collect()
        assert all(reference() is not None for reference in references.values())
        first = prepared.run_snapshot()
        _assert_values(first)
        references["x_base"]()[:] += 100.0
        references["parameters"]()[0] = 7.0
        second = prepared.run_snapshot()
        _assert_values(second, offset=100.0, parameter=7.0)
        _assert_values(first)
        _assert_independent_results(first, second)
        del prepared
        _assert_collected(references)
        _assert_values(first)
        _assert_values(second, offset=100.0, parameter=7.0)


def test_live_ordinary_owners_still_hit_cache_and_updated_payload_is_read():
    graph, payload = _graph(), _payload()
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        first = _execute(scheduler, graph, payload)
        cached = _execute(scheduler, graph, payload)
        assert not first.audit["prepared_cached"]
        assert cached.audit["prepared_cached"]
        _assert_no_input_alias(cached, payload)
        payload["x_base"][:] += 100.0
        payload["parameters"][0] = 7.0
        updated = _execute(scheduler, graph, payload)
        assert updated.audit["prepared_cached"]
        _assert_values(first)
        _assert_values(cached)
        _assert_values(updated, offset=100.0, parameter=7.0)
        _assert_independent_results(first, cached)
        _assert_independent_results(cached, updated)


def _execute_metadata_input(scheduler, location, dtype):
    # A tiny supported input can indirectly retain an arbitrarily large allocation.
    auxiliary = np.zeros(1_000_000, dtype=np.float64)
    metadata_dtype = np.dtype(dtype, metadata={"auxiliary": auxiliary})
    if location == "input":
        source = np.arange(8) % 2 if dtype == "uint8" else np.arange(8)
        value = source.astype(dtype).view(metadata_dtype)
        expected = source.astype("bool" if dtype == "uint8" else dtype)
        graph = GraphCompiler(
            {"x": {"kind": "vector", "dtype": "bool" if dtype == "uint8" else dtype}}
        ).compile("x")
        inputs, starts, ends = {"x": value}, np.array([0], np.int64), np.array([8], np.int64)
        first = scheduler.execute(graph, inputs, starts, ends)
        second = scheduler.execute(graph, inputs, starts, ends)
        for result in (first, second):
            actual = result.outputs[0].values[0]
            np.testing.assert_array_equal(actual, expected)
            assert not np.shares_memory(actual, value)
            assert not actual.flags.writeable
    else:
        graph, payload = _graph(), _payload()
        payload[location] = payload[location].view(metadata_dtype)
        value = payload[location]
        first, second = _execute(scheduler, graph, payload), _execute(scheduler, graph, payload)
        _assert_values(first)
        _assert_values(second)
    assert second.audit["prepared_cached"]
    assert value.dtype.metadata["auxiliary"] is auxiliary
    assert value.dtype == np.dtype(dtype)
    _assert_independent_results(first, second)
    return first, second, {"input": weakref.ref(value), "dtype_metadata": weakref.ref(auxiliary)}


@pytest.mark.parametrize(
    ("location", "dtype"),
    [("input", dtype) for dtype in ("float64", "int64", "bool", "uint8")]
    + [(location, dtype) for location, dtype in (
        ("starts", "int64"), ("ends", "int64"),
        ("products", "int64"), ("parameters", "float64"),
    )],
)
def test_ordinary_cache_does_not_retain_data_through_dtype_metadata(location, dtype):
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        first, second, references = _execute_metadata_input(scheduler, location, dtype)
        _assert_collected(references)
        # Kept outputs must survive collection of both the input and metadata owner.
        _assert_independent_results(first, second)


def test_expired_input_cannot_hit_cache_for_new_same_geometry_data():
    graph, payload = _graph(), _payload()
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        first = _execute(scheduler, graph, payload)
        expired = {name: weakref.ref(payload[name]) for name in ("x", "x_base")}
        del payload["x"], payload["x_base"]
        _assert_collected(expired)
        # Pointer reuse is allocator-dependent; current geometry is revalidated.
        payload["x_base"] = np.zeros(16, dtype=np.float64)
        payload["x_base"][4:12] = np.arange(8, dtype=np.float64) * 2.0 + 100.0
        payload["x"] = payload["x_base"][4:12]
        replacement = _execute(scheduler, graph, payload)
        # Allocator pointer reuse can admit identical, freshly revalidated
        # metadata. It must never supply stale numeric values or input owners.
        _assert_values(replacement, offset=100.0)
        _assert_values(first)
        cached = _execute(scheduler, graph, payload)
        assert cached.audit["prepared_cached"]
        _assert_independent_results(first, replacement)


@pytest.mark.parametrize("entry", ["ordinary", "prepared"])
@pytest.mark.parametrize(
    "mutation",
    [
        "input_dtype",
        "input_shape",
        "parameter_dtype",
        "parameter_shape",
        "interval_dtype",
        "interval_shape",
        "product_dtype",
        "product_shape",
    ],
)
@pytest.mark.filterwarnings(
    "ignore:Setting the (dtype|shape) on a NumPy array has been deprecated:DeprecationWarning"
)
def test_mutated_live_descriptors_retain_input_validation(entry, mutation):
    graph, payload = _graph(), _payload()
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        if entry == "prepared":
            prepared = scheduler.prepare_execution(
                graph,
                {"x": payload["x"]},
                payload["starts"],
                payload["ends"],
                parameters=payload["parameters"],
                product_ids=payload["products"],
            )
            first = prepared.run_snapshot()
        else:
            first = _execute(scheduler, graph, payload)
        if mutation == "input_dtype":
            payload["x"].dtype = np.int64
        elif mutation == "input_shape":
            payload["x"].shape = (2, 4)
        elif mutation == "parameter_dtype":
            payload["parameters"].dtype = np.int64
        elif mutation == "parameter_shape":
            payload["parameters"].shape = (1, 1)
        elif mutation == "interval_dtype":
            payload["ends"].dtype = np.float64
        elif mutation == "interval_shape":
            payload["starts"].shape = (1, 2)
        elif mutation == "product_dtype":
            payload["products"].dtype = np.float64
        else:
            payload["products"].shape = (2, 1)
        if entry == "prepared":
            with pytest.raises(ValueError, match="PREPARED_INPUT_CHANGED"):
                prepared.run_snapshot()
        else:
            with pytest.raises((ValueError, TypeError)):
                _execute(scheduler, graph, payload)
        _assert_values(first)


def test_ordinary_owned_array_resize_rebinds_and_preserves_old_result():
    graph = GraphCompiler({"x": "series", "k": "scalar"}).compile("mean(x)+k")
    x = np.arange(8, dtype=np.float64)
    starts, ends = np.array([0], np.int64), np.array([8], np.int64)
    parameters = np.array([2.0])
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        first = scheduler.execute(graph, {"x": x}, starts, ends, parameters=parameters)
        x.resize(12, refcheck=False)
        x[:] = np.arange(12, dtype=np.float64)
        ends[0] = 12
        resized = scheduler.execute(graph, {"x": x}, starts, ends, parameters=parameters)
        assert not resized.audit["prepared_cached"]
        np.testing.assert_array_equal(first.values, [[5.5]])
        np.testing.assert_array_equal(resized.values, [[7.5]])
        cached = scheduler.execute(graph, {"x": x}, starts, ends, parameters=parameters)
        assert cached.audit["prepared_cached"]
        assert not np.shares_memory(first.values, resized.values)


@pytest.mark.parametrize("view", [False, True], ids=["owned_array", "allocation_under_view"])
def test_prepared_rejects_resized_allocation_before_using_old_pointer(view):
    graph = GraphCompiler({"x": "series"}).compile("mean(x)")
    base = np.arange(16, dtype=np.float64)
    x = base[4:12] if view else base
    starts, ends = np.array([0], np.int64), np.array([x.size], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"x": x}, starts, ends)
        prepared.run_snapshot()
        # The view may now be dangling. The guard must reject before numerical work.
        base.resize(32, refcheck=False)
        with pytest.raises(ValueError, match="PREPARED_INPUT_CHANGED"):
            prepared.run_snapshot()


@pytest.mark.parametrize("dictionary", [False, True], ids=["mapping", "dict_subclass"])
def test_mapping_miss_leases_the_array_from_its_single_fetch(dictionary):
    class FreshValues:
        calls = 0

        def __getitem__(self, key):
            assert key == "x"
            self.calls += 1
            # Distinct sizes make a cache miss deterministic even if CPython
            # reuses an expired object's address and NumPy reuses its buffer.
            value = np.arange(8 + self.calls, dtype=np.float64) + 100 * self.calls
            self.last_owner = weakref.ref(value)
            return value

    class FreshMapping(FreshValues, Mapping):
        def __iter__(self):
            return iter(("x",))

        def __len__(self):
            return 1

    class FreshDictionary(FreshValues, dict):
        pass

    inputs = FreshDictionary(x=None) if dictionary else FreshMapping()
    graph = GraphCompiler({"x": "series"}).compile("sum(x)")
    starts, ends = np.array([0], np.int64), np.array([8], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        first = scheduler.execute(graph, inputs, starts, ends)
        assert inputs.calls == 1
        np.testing.assert_array_equal(first.values, [[828.0]])
        assert inputs.last_owner() is None
        second = scheduler.execute(graph, inputs, starts, ends)
        # A dict subclass is consulted for cache matching, then for a new bind
        # when that freshly generated array does not match the cached identity.
        expected_calls = 3 if dictionary else 2
        assert inputs.calls == expected_calls
        np.testing.assert_array_equal(second.values, [[28.0 + 800 * expected_calls]])
        np.testing.assert_array_equal(first.values, [[828.0]])
        assert inputs.last_owner() is None


def test_dict_subclass_cache_hit_leases_the_fetched_array_before_replacement():
    class ReplacingDictionary(dict):
        calls = 0

        def __getitem__(self, key):
            self.calls += 1
            value = super().__getitem__(key)
            self[key] = np.arange(8, dtype=np.float64) + 100 * self.calls
            return value

    graph = GraphCompiler({"x": "series"}).compile("sum(x)")
    starts, ends = np.array([0], np.int64), np.array([8], np.int64)
    inputs = ReplacingDictionary(x=np.arange(8, dtype=np.float64))
    old_owner = weakref.ref(dict.__getitem__(inputs, "x"))
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        # Warm the cache without invoking the subclass getter.
        first = scheduler.execute(graph, dict(inputs), starts, ends)
        hit = scheduler.execute(graph, inputs, starts, ends)
        assert hit.audit["prepared_cached"]
        assert inputs.calls == 1
        np.testing.assert_array_equal(hit.values, [[28.0]])
        assert old_owner() is None
        replacement = scheduler.execute(graph, inputs, starts, ends)
        assert not replacement.audit["prepared_cached"]
        # Cache matching fetches the new owner once; a miss binds it anew once.
        assert inputs.calls == 3
        np.testing.assert_array_equal(replacement.values, [[1628.0]])
        np.testing.assert_array_equal(first.values, [[28.0]])
        np.testing.assert_array_equal(hit.values, [[28.0]])


@pytest.mark.parametrize("callback_kind", ["drop_mapping", "replace_cache"])
def test_cache_validation_holds_current_owner_and_binding_across_python_callbacks(callback_kind):
    class CallbackView(np.ndarray):
        _callback = None

        @property
        def base(self):
            callback, self._callback = self._callback, None
            if callback is not None:
                callback()
            return np.ndarray.base.__get__(self)

    graph = GraphCompiler({"x": "series"}).compile("sum(x)")
    other_graph = GraphCompiler({"y": "series"}).compile("mean(y)")
    starts, ends = np.array([0], np.int64), np.array([8], np.int64)
    inputs = {"x": np.arange(8, dtype=np.float64).view(CallbackView)}
    owner = weakref.ref(inputs["x"])
    callbacks = []
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        first = scheduler.execute(graph, inputs, starts, ends)

        def callback():
            inputs.clear()
            gc.collect()
            assert owner() is not None, "validation lost the fetched array"
            if callback_kind == "replace_cache":
                nested = scheduler.execute(other_graph, {"y": np.full(8, 19.0)}, starts, ends)
                np.testing.assert_array_equal(nested.values, [[19.0]])
            callbacks.append(callback_kind)

        inputs["x"]._callback = callback
        current = scheduler.execute(graph, inputs, starts, ends)
        assert callbacks == [callback_kind]
        assert current.audit["prepared_cached"]
        assert owner() is None
        np.testing.assert_array_equal(current.values, [[28.0]])
        np.testing.assert_array_equal(first.values, [[28.0]])
        assert not np.shares_memory(first.values, current.values)
