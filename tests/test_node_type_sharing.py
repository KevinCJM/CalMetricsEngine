"""Public type values and lifetime when compiler descriptions share owners."""
import gc
import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from tests.test_typed_results import config


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("policy", ["raise", "isolate"])
def test_group_and_iteration_types_do_not_modify_shared_predecessors(lane, policy):
    declarations = {"x": "series", "key": {"kind": "series", "dtype": "int64"},
                    "initial": "scalar"}
    iteration = "iterate(iterate_x*0.5,initial,1e-6,100)"
    compiler = GraphCompiler(declarations)
    graph = compiler.compile({
        "groups": "group_apply(mean(x),key)", "keys": "key",
        "positive": "group_apply(mean(x),key)>0", "value": iteration,
        "count": f"iteration_count({iteration})",
        "status": f"iteration_status({iteration})",
        "residual": f"iteration_residual({iteration})",
    }, error_policy=policy)
    metadata, native = graph.metadata(), graph.native_nodes()
    assert [t["dtype"] for t in metadata["root_types"]] == [
        "float64", "int64", "bool", "float64", "int64", "int64", "float64"]
    nodes = graph.nodes
    descriptions = [node.inferred_type for node in nodes]
    for node in nodes:
        external = node.inferred_type
        external["axes"].append("user_edit")
        external["shape"].append("user_edit")
        external["dtype"] = "user_edit"
    assert [node.inferred_type for node in nodes] == descriptions
    clone = pickle.loads(pickle.dumps(graph))
    assert clone.metadata() == metadata
    assert clone.native_nodes() == native
    del graph, compiler, declarations
    gc.collect()
    assert [node.inferred_type for node in nodes] == descriptions
    x = np.array([1., 9., 3., 7., 4., 16., 12., 8.])
    keys = np.array([2**60, 2**60 + 1, 2**60, 2**60 + 1] * 2, np.int64)
    x.flags.writeable = keys.flags.writeable = False
    starts, ends = np.array([0, 4], np.int64), np.array([4, 8], np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        results = [scheduler.execute(clone, {"x": x, "key": keys}, starts, ends,
                                     parameters={"initial": initial})
                   for initial in (8., 16.)]
        assert all(r.plan.lane == ("process" if lane.startswith("process") else lane)
                   for r in results)
    del clone, nodes
    gc.collect()
    for result in results:
        for row, expected in enumerate(([2., 8., 2., 8.], [8., 12., 8., 12.])):
            np.testing.assert_array_equal(result.named_outputs["groups"].values[row], expected)
            np.testing.assert_array_equal(result.named_outputs["keys"].values[row],
                                          keys[row * 4:row * 4 + 4])
            np.testing.assert_array_equal(result.named_outputs["positive"].values[row],
                                          np.ones(4, np.bool_))
        for name in ("count", "status"):
            assert result.named_outputs[name].dtype == "int64"
        for name in ("value", "residual"):
            assert result.named_outputs[name].dtype == "float64"
        assert result.audit["python_operator_calls"] == 0
    assert not np.shares_memory(results[0].named_outputs["groups"].values[0],
                                results[1].named_outputs["groups"].values[0])


@pytest.mark.parametrize("changed", ["dtype", "axes", "shape"])
def test_distinct_declared_types_survive_graph_copy_and_external_edits(changed):
    first = {"kind": "matrix", "dtype": "float64", "axes": ["factor", "asset"],
             "shape": ["F", "A"]}
    second = {**first, "axes": list(first["axes"]), "shape": list(first["shape"])}
    second[changed] = {"dtype": "int64", "axes": ["model", "asset"],
                       "shape": ["M", "A"]}[changed]
    graph = GraphCompiler({"a": first, "b": second}).compile({"a": "a", "b": "b"})
    roots = graph.metadata()["root_types"]
    assert roots[0][changed] != roots[1][changed]
    first["axes"].append("user_edit")
    second["shape"].append("user_edit")
    clone = pickle.loads(pickle.dumps(graph))
    snapshot = clone.nodes
    del graph, clone
    gc.collect()
    assert [node.inferred_type for node in snapshot] == roots
