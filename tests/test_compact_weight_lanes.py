"""Contract checks for the isolated combination-lane candidate.

The reference is the same native editable DAG with a public intermediate.
No numerical tolerance or public behavior is changed. These tests have only
been parsed statically; no import or execution is implied by their presence.
Set CME_EXPECT_WEIGHT_LANES=0 for Eigen-OFF or EIGEN_DONT_VECTORIZE builds.
"""

import copy
import os

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler

from .test_affine_projection_library import DARWIN_ARM64_ROUNDS, rounding
from .test_compact_path_execution import (
    assert_canonical,
    assert_private,
    canonical_spec,
    check_ownership,
    chunk_sum,
    compare_selected,
    execute,
    saved_outputs,
    selected_spec,
    used_inputs,
)
from .test_mc_compact_graph import (
    compact_spec,
    compile_compact,
    ieee_bits,
    small_tape,
)

EXPECT_PACKETS = os.environ.get("CME_EXPECT_WEIGHT_LANES", "1") == "1"
SHAPES = {1: (32, 128), 4: (32, 256), 5: (12, 683),
          6: (12, 683), 8: (32, 128), 9: (24, 192)}
PARAMETERS = {"n0": 17.3, "target": .03}


def lane_fixture(combinations=8, layout="C"):
    steps, paths = SHAPES[combinations]
    spec = compact_spec(steps, paths, 32, combinations)
    inputs = small_tape(steps, paths, 4, 32, combinations, layout)
    starts = np.array([0], np.int64)
    ends = np.array([steps], np.int64)
    return spec, inputs, starts, ends


def sparse_fence(spec):
    # One absent asset capture rejects only the new dense packet path.
    # All existing fan/stats/sample mathematics remains the same compact DAG.
    return selected_spec(spec, [name for name in spec["expressions"]
                                if name != "contribution_0_31"])


def packet_lanes(combinations):
    return {1: 0, 4: 4, 5: 4, 6: 4, 8: 8, 9: 8}[combinations]


@pytest.mark.parametrize("combinations", tuple(SHAPES))
@pytest.mark.parametrize("layout,order", [("C", "F"), ("F", "C")])
@pytest.mark.parametrize("mode", ["ordinary", "prepared"])
def test_dense_weight_lanes_all_roots_tail_batches_and_two_owners(combinations, layout, order, mode):
    spec, inputs, starts, ends = lane_fixture(combinations, layout)
    graph = compile_compact(spec, order=order)
    canonical = compile_compact(canonical_spec(spec), order=order)
    fenced = compile_compact(sparse_fence(spec), order=order)
    originals = {name: value.copy() for name, value in inputs.items()}
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        reference = execute(scheduler, canonical, inputs, starts, ends, PARAMETERS)
        assert_canonical(reference)
        scalar = execute(scheduler, fenced, inputs, starts, ends, PARAMETERS)
        compare_selected(scalar, reference)
        assert_private(scalar)
        if mode == "prepared":
            prepared = scheduler.prepare_execution(
                graph, used_inputs(graph, inputs), starts, ends, parameters=PARAMETERS)
            first = prepared.run_snapshot()
            saved = saved_outputs(first)
            later = prepared.run_snapshot()
        else:
            first = execute(scheduler, graph, inputs, starts, ends, PARAMETERS)
            saved = saved_outputs(first)
            later = execute(scheduler, graph, inputs, starts, ends, PARAMETERS)
        expected = spec["steps"] * inputs["x"].shape[1] * spec["assets"] * packet_lanes(combinations)
        for result in (first, later):
            compare_selected(result, reference)
            assert_private(result)
            assert result.audit["result_lifetime"] == "independent"
            # Old affine SIMD is present on both sides; only weight packets
            # explain this additional work, including scalar tail batches.
            assert chunk_sum(result, "vector_elements") - chunk_sum(scalar, "vector_elements") == (
                expected if EXPECT_PACKETS else 0)
            for chunk in result.audit["native_chunks"]:
                assert chunk["operator_workspace_capacity_bytes"] <= result.plan.estimated_worker_scratch_bytes
            for output in result.outputs:
                for value in output.values:
                    if value.ndim == 2:
                        assert value.flags.f_contiguous if order == "F" else value.flags.c_contiguous
        check_ownership(first, later, saved, inputs)
    for name, value in inputs.items():
        ieee_bits(value, originals[name])
    # Scheduler and scratch destruction cannot change the retained outputs.
    for name, snapshots in saved.items():
        for value, snapshot in zip(first.named_outputs[name].values, snapshots, strict=True):
            ieee_bits(value, snapshot)


@pytest.mark.parametrize("layout", ["strided", "negative"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_logical_weight_views_and_permuted_root_group_order(layout, order):
    spec, inputs, starts, ends = lane_fixture(layout=layout)
    items = list(zip(spec["expressions"].items(), spec["root_bindings"], strict=True))
    items = items[47:] + items[:47]
    spec["expressions"] = dict(item for item, _ in items)
    spec["root_bindings"] = [binding for _, binding in items]
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        actual = execute(scheduler, compile_compact(spec, order=order), inputs, starts, ends, PARAMETERS)
        reference = execute(scheduler, compile_compact(canonical_spec(spec), order=order),
                            inputs, starts, ends, PARAMETERS)
    compare_selected(actual, reference)
    # This root permutation declined compact admission in actual79 too. Keep
    # mathematical compatibility and the existing canonical route explicit.
    assert_canonical(actual)


@pytest.mark.parametrize("layout", ["strided", "negative"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_logical_weight_views_keep_packet_admission_and_owners(layout, order):
    spec, inputs, starts, ends = lane_fixture(layout=layout)
    originals = {name: value.copy() for name, value in inputs.items()}
    graph = compile_compact(spec, order=order)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        reference = execute(scheduler, compile_compact(canonical_spec(spec), order=order),
                            inputs, starts, ends, PARAMETERS)
        scalar = execute(scheduler, compile_compact(sparse_fence(spec), order=order),
                         inputs, starts, ends, PARAMETERS)
        first = execute(scheduler, graph, inputs, starts, ends, PARAMETERS)
        saved = saved_outputs(first)
        later = execute(scheduler, graph, inputs, starts, ends, PARAMETERS)
        expected = spec["steps"] * inputs["x"].shape[1] * spec["assets"] * 8
        for actual in (first, later):
            compare_selected(actual, reference)
            assert_private(actual)
            assert chunk_sum(actual, "vector_elements") - chunk_sum(scalar, "vector_elements") == (
                expected if EXPECT_PACKETS else 0)
        check_ownership(first, later, saved, inputs)
    for name, value in inputs.items():
        ieee_bits(value, originals[name])


@pytest.mark.parametrize("order", ["C", "F"])
def test_duplicate_public_roots_are_independent_and_use_one_contribution_capture(order):
    spec, inputs, starts, ends = lane_fixture()
    bindings = dict(zip(spec["expressions"], spec["root_bindings"], strict=True))
    for original, alias in (("fan_0", "duplicate_fan"), ("contribution_0_17", "duplicate_contribution")):
        spec["expressions"][alias] = spec["expressions"][original]
        spec["root_bindings"].append(copy.deepcopy(bindings[original]))
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        actual = execute(scheduler, compile_compact(spec, order=order), inputs, starts, ends, PARAMETERS)
        reference = execute(scheduler, compile_compact(canonical_spec(spec), order=order),
                            inputs, starts, ends, PARAMETERS)
    compare_selected(actual, reference)
    assert_private(actual)
    for original, alias in (("fan_0", "duplicate_fan"), ("contribution_0_17", "duplicate_contribution")):
        left, right = (actual.named_outputs[name].values[0] for name in (original, alias))
        ieee_bits(left, right)
        assert not np.shares_memory(left, right)


@pytest.mark.parametrize("value", [0., -0., np.finfo(np.float64).smallest_subnormal])
def test_weight_zero_and_subnormal_keep_original_bits_and_guard(value):
    spec, frozen, starts, ends = lane_fixture()
    inputs = dict(frozen)
    inputs["w0"] = frozen["w0"].copy()
    inputs["w0"][0] = value
    inputs["w0"].flags.writeable = False
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        actual = execute(scheduler, compile_compact(spec), inputs, starts, ends, PARAMETERS)
        scalar = execute(scheduler, compile_compact(sparse_fence(spec)), inputs, starts, ends, PARAMETERS)
        reference = execute(scheduler, compile_compact(canonical_spec(spec)), inputs, starts, ends, PARAMETERS)
    compare_selected(actual, reference)
    compare_selected(scalar, reference)
    eligible = EXPECT_PACKETS and value == 0. and not np.signbit(value)
    expected = spec["steps"] * inputs["x"].shape[1] * 32 * 8 if eligible else 0
    assert chunk_sum(actual, "vector_elements") - chunk_sum(scalar, "vector_elements") == expected


@pytest.mark.parametrize("fault", ["nan_weight", "inf_weight", "nonfinite_producer",
                                    "q5_zero_nav", "qreturn_zero_nav",
                                    "nan_and_bad_weight_shape", "nan_and_bad_q_shape"])
def test_nonfinite_values_and_original_node_error_order_recover(fault):
    spec, frozen, starts, ends = lane_fixture()
    bad = {name: value.copy() for name, value in frozen.items()}
    parameters = dict(PARAMETERS)
    if fault == "nan_weight":
        bad["w0"][-1] = np.nan
    elif fault == "inf_weight":
        bad["w7"][-1] = np.inf
    elif fault == "nonfinite_producer":
        bad["x"][-1, -1, -1] = np.inf
    elif fault == "nan_and_bad_weight_shape":
        bad["w0"][0] = np.nan
        bad["w7"] = bad["w7"][:-1].copy()
    elif fault == "nan_and_bad_q_shape":
        bad["w0"][0] = np.nan
        bad["q5"] = bad["q5"][:-1].copy()
    else:
        bad["q5" if fault == "q5_zero_nav" else "qreturn"][0] = 1.1
        parameters["n0"] = 0.
    graph, canonical = compile_compact(spec), compile_compact(canonical_spec(spec))
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        failures = []
        for compiled in (canonical, graph):
            with pytest.raises((ValueError, RuntimeError)) as error:
                execute(scheduler, compiled, bad, starts, ends, parameters)
            failures.append((type(error.value), str(error.value)))
        assert failures[0] == failures[1]
        # These are independent input-binding shape checks, before NaN math;
        # they do not claim coverage of downstream multi-hop scope validation.
        if fault == "nan_and_bad_weight_shape":
            assert "share declared symbolic dimensions" in failures[0][1]
        elif fault == "nan_and_bad_q_shape":
            assert "fixed dimension mismatch" in failures[0][1]
        recovered = execute(scheduler, graph, frozen, starts, ends, PARAMETERS)
        reference = execute(scheduler, canonical, frozen, starts, ends, PARAMETERS)
    compare_selected(recovered, reference)
    assert_private(recovered)


@pytest.mark.parametrize("mode", ["up", "down", "zero"])
def test_non_nearest_keeps_scalar_weight_fold_and_environment(mode):
    spec, inputs, starts, ends = lane_fixture()
    graph = compile_compact(spec)
    canonical = compile_compact(canonical_spec(spec))
    fenced = compile_compact(sparse_fence(spec))
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with rounding(DARWIN_ARM64_ROUNDS[mode]):
            actual = execute(scheduler, graph, inputs, starts, ends, PARAMETERS)
            reference = execute(scheduler, canonical, inputs, starts, ends, PARAMETERS)
            scalar = execute(scheduler, fenced, inputs, starts, ends, PARAMETERS)
            compare_selected(actual, reference)
            assert chunk_sum(actual, "vector_elements") == chunk_sum(scalar, "vector_elements")


def test_prepared_repacks_same_address_weights_and_preserves_old_owner():
    spec, frozen, starts, ends = lane_fixture()
    inputs = {name: value.copy() for name, value in frozen.items()}
    address = inputs["w1"].__array_interface__["data"][0]
    graph, canonical = compile_compact(spec), compile_compact(canonical_spec(spec))
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, used_inputs(graph, inputs), starts, ends,
                                               parameters=PARAMETERS)
        first = prepared.run_snapshot()
        saved = saved_outputs(first)
        inputs["w1"][:] *= .75
        later = prepared.run_snapshot()
        reference = execute(scheduler, canonical, inputs, starts, ends, PARAMETERS)
        compare_selected(later, reference)
        check_ownership(first, later, saved, inputs)
        assert inputs["w1"].__array_interface__["data"][0] == address
        assert not np.array_equal(first.named_outputs["fan_1"].values[0], later.named_outputs["fan_1"].values[0])


def test_packet_reserve_is_admitted_at_exact_memory_budget():
    spec, inputs, starts, ends = lane_fixture()
    graph, canonical = compile_compact(spec), compile_compact(canonical_spec(spec))
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(graph, inputs, starts, ends)
        with pytest.raises(MemoryError):
            execute(scheduler, graph, inputs, starts, ends, PARAMETERS,
                    memory_budget_bytes=plan.estimated_total_memory_bytes - 1)
        actual = execute(scheduler, graph, inputs, starts, ends, PARAMETERS,
                         memory_budget_bytes=plan.estimated_total_memory_bytes)
        reference = execute(scheduler, canonical, inputs, starts, ends, PARAMETERS)
    compare_selected(actual, reference)
    assert_private(actual)
    for chunk in actual.audit["native_chunks"]:
        assert chunk["operator_workspace_capacity_bytes"] <= plan.estimated_worker_scratch_bytes
