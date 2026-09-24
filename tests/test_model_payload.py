"""Immutable native models and heterogeneous named results."""
import gc
import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, ModelPayload
from tests.test_tensor_protocol import configuration, tensor

META = {"schema": "test-model-1", "algorithm": "fixture", "algorithm_version": "1",
        "source_identity": "fixed-fixture", "training_options": "{}", "random_version": "none"}


def fixture():
    fields = {"weights": np.arange(24.).reshape(2, 3, 4)[:, ::-1],
              "labels": np.array([2**53+1, 2**53+3], np.int64),
              "active": np.array([True, False]), "iteration": np.array(2**53+7, np.int64)}
    types = {"weights": tensor(temporal=False),
             "labels": {"kind": "vector", "axes": ["model"], "shape": ["M"], "dtype": "int64"},
             "active": {"kind": "vector", "axes": ["model"], "shape": ["M"], "dtype": "bool"},
             "iteration": {"kind": "value", "dtype": "int64"}}
    return fields, types


def test_model_identity_exact_bytes_immutable_snapshot_and_lifetime():
    fields, types = fixture()
    model = ModelPayload.from_fields(types, fields, META)
    restored = ModelPayload.from_bytes(model.to_bytes())
    assert restored.identity == model.identity
    assert restored.metadata == META
    for name, expected in fields.items():
        view = model.fields[name]
        assert not view.flags.writeable
        with pytest.raises(ValueError):
            view.setflags(write=True)
        assert not np.shares_memory(view, expected)
        np.testing.assert_array_equal(view, expected)
        np.testing.assert_array_equal(restored.fields[name], expected)
    retained = model.fields["weights"]
    expected = retained.copy()
    fields["weights"][:] = 0
    assert ModelPayload.from_fields(types, fields, META).identity != model.identity
    changed = dict(META, algorithm_version="2")
    assert ModelPayload.from_fields(restored.types, restored.fields, changed).identity != model.identity
    del model, restored
    gc.collect()
    np.testing.assert_array_equal(retained, expected)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_model_input_named_output_and_export_across_lanes(lane):
    fields, types = fixture()
    model = ModelPayload.from_fields(types, fields, META)
    graph = GraphCompiler(model.types).compile({name: name for name in fields})
    graph = pickle.loads(pickle.dumps(graph))
    with AdaptiveScheduler(cpu_budget=2, config=configuration(lane)) as engine:
        result = engine.execute(graph, model, np.array([0, 0], np.int64), np.array([1, 1], np.int64))
        assert result.plan.metadata()["model_identity"] == model.identity
        assert result.plan.estimated_input_bytes >= model.nbytes
        if result.plan.lane == "process":
            assert result.plan.use_shared_memory
            assert result.audit["shared_memory_bytes"] == model.nbytes + 32 + result.plan.estimated_output_bytes
        exported = result.to_model_payload(META)
    assert exported.identity == model.identity
    for name, expected in fields.items():
        np.testing.assert_array_equal(result.named_outputs[name].values[1], expected)


@pytest.mark.parametrize("mutation", ["missing_metadata", "wrong_dtype", "wrong_shape", "extra_field", "temporal"])
def test_invalid_models_fail_closed(mutation):
    fields, types = fixture()
    metadata = dict(META)
    if mutation == "missing_metadata":
        del metadata["source_identity"]
    elif mutation == "wrong_dtype":
        fields["labels"] = fields["labels"].astype(float)
    elif mutation == "wrong_shape":
        fields["active"] = np.ones(3, bool)
    elif mutation == "extra_field":
        fields["unused"] = np.array(1.)
    else:
        types["weights"] = tensor(temporal=True)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        ModelPayload.from_fields(types, fields, metadata)


def test_binary_protocol_rejects_truncation_version_and_trailing_bytes():
    fields, types = fixture()
    encoded = ModelPayload.from_fields(types, fields, META).to_bytes()
    for blob in [b"", encoded[:7], encoded[:-1], encoded + b"x", b"\xff" + encoded[1:]]:
        with pytest.raises((ValueError, RuntimeError)):
            ModelPayload.from_bytes(blob)


def test_failed_and_temporal_results_cannot_be_exported_as_models():
    with AdaptiveScheduler(cpu_budget=1) as engine:
        failed = engine.execute(GraphCompiler({"unused": "series"}).compile({"v": "1/0"}, error_policy="isolate"),
                                {"unused": np.ones(1)}, np.array([0], np.int64), np.array([1], np.int64))
        with pytest.raises(ValueError, match="failed"):
            failed.to_model_payload(META)
        temporal = engine.execute(GraphCompiler({"x": "series"}).compile({"v": "x"}),
                                  {"x": np.ones(2)}, np.array([0], np.int64), np.array([2], np.int64))
        with pytest.raises((ValueError, RuntimeError), match="STATIC"):
            temporal.to_model_payload(META)


@pytest.mark.parametrize("dtype,first,second", [("float64", 1.5, 3.5),
    ("int64", 2**53+1, 2**53+3), ("bool", True, False)])
def test_exact_scalar_array_stays_live_in_cached_and_prepared_execution(dtype, first, second):
    value = np.array(first, dtype=dtype)
    graph = GraphCompiler({"x": {"kind": "value", "dtype": dtype}}).compile({"x": "x"})
    starts, ends = np.array([0], np.int64), np.array([1], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as engine:
        prepared = engine.prepare_execution(graph, {"x": value}, starts, ends)
        old = prepared.run_snapshot()
        ordinary = engine.execute(graph, {"x": value}, starts, ends)
        value[...] = second
        assert prepared.run().named_outputs["x"].values[0] == second
        assert engine.execute(graph, {"x": value}, starts, ends).named_outputs["x"].values[0] == second
        assert old.named_outputs["x"].values[0] == first
        assert ordinary.named_outputs["x"].values[0] == first


def test_model_identity_prevents_plan_reuse_for_another_model():
    fields, types = fixture()
    first = ModelPayload.from_fields(types, fields, META)
    fields["weights"] = fields["weights"] + 1
    second = ModelPayload.from_fields(types, fields, META)
    graph = GraphCompiler(first.types).compile({name: name for name in fields})
    starts, ends = np.array([0], np.int64), np.array([1], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as engine:
        plan = engine.plan(graph, first, starts, ends)
        with pytest.raises((ValueError, RuntimeError), match="model|MODEL"):
            engine.execute(graph, second, starts, ends, plan=plan)


def test_unnamed_int64_scalar_preserves_precision():
    value = np.array(2**53+5, np.int64)
    graph = GraphCompiler({"x": {"kind": "value", "dtype": "int64"}}).compile("x")
    with AdaptiveScheduler(cpu_budget=1) as engine:
        result = engine.execute(graph, {"x": value}, np.array([0], np.int64), np.array([1], np.int64))
    assert result.outputs[0].values[0].dtype == np.int64
    assert result.outputs[0].values[0] == value
