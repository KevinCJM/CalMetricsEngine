"""Native feedback state, diagnostics, isolation and worker protocol."""
import pickle

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, GraphCompileError
from tests.test_tensor_protocol import configuration, tensor


def outputs(expression):
    return {"value": expression, "status": f"iteration_status({expression})",
            "count": f"iteration_count({expression})", "residual": f"iteration_residual({expression})"}


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("shape", [(), (7,), (3, 4), (2, 3, 4), (2, 0, 4)])
def test_native_iteration_shapes_lanes_and_cse(lane, shape):
    spec = {"kind": "value"} if not shape else (
        {"kind": "vector", "axes": ["asset"], "shape": ["N"]} if len(shape) == 1 else
        {"kind": "matrix", "axes": ["model", "asset"], "shape": ["M", "N"]} if len(shape) == 2 else tensor(temporal=False))
    initial = np.full(shape, 8.)
    expression = "iterate(iterate_x*0.5,x,1e-6,100)"
    graph = GraphCompiler({"x": spec}).compile(outputs(expression))
    assert sum(n.kind == "apply_scope" for n in graph.nodes) == 1
    graph = pickle.loads(pickle.dumps(graph))
    with AdaptiveScheduler(cpu_budget=2, config=configuration(lane)) as engine:
        result = engine.execute(graph, {"x": initial}, np.array([0, 0], np.int64), np.array([1, 1], np.int64))
    fields = result.named_outputs
    expected_count = 23 if initial.size else 1
    for row in range(2):
        np.testing.assert_allclose(fields["value"].values[row], initial * 0.5**expected_count, rtol=0, atol=0)
        assert fields["status"].values[row] == 0
        assert fields["status"].values[row].dtype == np.int64
        assert fields["count"].values[row] == expected_count
        assert fields["residual"].values[row] <= 1e-6
    np.testing.assert_array_equal(initial, np.full(shape, 8.))


@pytest.mark.parametrize("expression,status,count,value", [
    ("iterate(iterate_x+1,0,0,3)", 1, 3, 3),
    ("iterate(1/iterate_x,0,1e-5,4)", 5, 1, 0),
    ("iterate(2,0,0,4)", 0, 2, 2),
    ("iterate(iterate_x,7,0,4)", 0, 1, 7),
])
def test_iteration_stopping_is_separate_from_execution_error(expression, status, count, value):
    graph = GraphCompiler({"unused": "series"}).compile(outputs(expression), error_policy="isolate")
    with AdaptiveScheduler(cpu_budget=1) as engine:
        result = engine.execute(graph, {"unused": np.ones(1)}, np.array([0], np.int64), np.array([1], np.int64))
    fields = result.named_outputs
    assert fields["status"].values[0] == status
    assert fields["count"].values[0] == count
    assert fields["value"].values[0] == value


def test_iteration_more_than_eight_captures_and_nested_scope():
    variables = {f"x{i}": "series" for i in range(13)}
    data = {key: np.ones(5) for key in variables}
    expression = "iterate(iterate_x*0.5+" + "+".join(f"sum({key})*0.5" for key in variables) + ",0,1e-8,100)"
    graph = GraphCompiler(variables).compile(outputs(expression))
    with AdaptiveScheduler(cpu_budget=1) as engine:
        result = engine.execute(graph, data, np.array([0], np.int64), np.array([5], np.int64))
    assert result.named_outputs["value"].values[0] == pytest.approx(65, abs=1e-8)
    assert result.named_outputs["status"].values[0] == 0


@pytest.mark.parametrize("suffix", ["-1,3", "1e-8,0", "1e-8,2.5", "1e-8,10001"])
def test_iteration_invalid_limits_fail_even_with_failed_initial(suffix):
    graph = GraphCompiler({"x": "series"}).compile(f"iterate(iterate_x,1/std(x,0),{suffix})", error_policy="isolate")
    with AdaptiveScheduler(cpu_budget=1) as engine:
        with pytest.raises((ValueError, RuntimeError), match="INVALID_PARAMETER"):
            engine.execute(graph, {"x": np.ones(4)}, np.array([0], np.int64), np.array([4], np.int64))


def test_iteration_rejects_shape_change_and_non_native_callback():
    compiler = GraphCompiler({"x": "series"})
    for expression in ["iterate(sum(iterate_x),x,1e-6,20)", "iterate(callback(iterate_x),x,1e-6,20)"]:
        with pytest.raises(GraphCompileError):
            compiler.compile(expression)


def test_iteration_budget_and_deadline():
    compiler = GraphCompiler({"x": "series"})
    graph = compiler.compile("iterate(iterate_x+1,x,0,10000)", scope_work_budget=1)
    data = {"x": np.zeros(1000)}
    with AdaptiveScheduler(cpu_budget=1) as engine:
        with pytest.raises((ValueError, RuntimeError), match="SCOPE_COMPUTE_BUDGET_EXCEEDED"):
            engine.execute(graph, data, np.array([0], np.int64), np.array([1000], np.int64))


@pytest.mark.parametrize("policy", ["raise", "isolate"])
@pytest.mark.parametrize("lane", ["single", "process_inline", "process_shared"])
@pytest.mark.parametrize("initial,body", [
    (np.nan, "iterate_x+sum(gather(y,idx))"),
    (1., "iterate_x/0+sum(gather(y,idx))"),
])
def test_numerical_stop_preserves_independent_body_structure_validation(policy, lane, initial, body):
    compiler = GraphCompiler({"x": "series", "y": "series", "idx": {"kind": "series", "dtype": "int64"}})
    graph = compiler.compile(outputs(f"iterate({body},x,0,3)"), error_policy=policy)
    with AdaptiveScheduler(cpu_budget=2, config=configuration(lane)) as engine:
        with pytest.raises((ValueError, RuntimeError), match="INDEX_OUT_OF_BOUNDS"):
            engine.execute(graph, {"x": np.array([initial]), "y": np.ones(1), "idx": np.array([7], np.int64)},
                           np.array([0, 0], np.int64), np.array([1, 1], np.int64))


@pytest.mark.parametrize("policy", ["raise", "isolate"])
def test_nonfinite_initial_state_returns_numerical_failure_without_invalid_payload_use(policy):
    expression = "iterate(iterate_x+1,x,0,3)"
    graph = GraphCompiler({"x": "series"}).compile(outputs(expression), error_policy=policy)
    with AdaptiveScheduler(cpu_budget=1) as engine:
        result = engine.execute(graph, {"x": np.array([np.nan])}, np.array([0], np.int64), np.array([1], np.int64))
    assert result.named_outputs["status"].values[0] == 5
    assert result.named_outputs["count"].values[0] == 0


@pytest.mark.parametrize('factor', [.5, 1.01])
def test_iteration_workspace_is_released_when_scope_is_no_longer_used(factor):
    compiler = GraphCompiler({"x": "series"})
    iterative = compiler.compile(outputs(f"iterate(iterate_x*{factor},x,1e-6,100)"))
    ordinary = compiler.compile("sum(x)")
    data = {"x": np.full(4096, 8.)}
    starts, ends = np.array([0], np.int64), np.array([4096], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as engine:
        before = engine.execute(ordinary, data, starts, ends)
        middle = engine.execute(iterative, data, starts, ends)
        smaller = engine.execute(iterative, data, starts, np.array([4], np.int64))
        after = engine.execute(ordinary, data, starts, ends)
    def workspace(result):
        return result.audit["native_chunks"][0]["operator_workspace_capacity_bytes"]
    # Only a proven finite-preserving chain can eliminate the candidate.
    if factor <= 1:
        assert workspace(middle) < data['x'].nbytes
    else:
        assert data["x"].nbytes <= workspace(middle) < 2 * data["x"].nbytes
    assert workspace(after) == workspace(before)
    assert middle.plan.estimated_worker_scratch_bytes >= workspace(middle)
    assert smaller.plan.estimated_worker_scratch_bytes >= workspace(smaller)
    assert workspace(smaller) <= workspace(middle)
