"""Private compact execution versus the unchanged editable canonical DAG.

The oracle publishes a full intermediate root to reject private admission.
No Python numerical implementation substitutes for a native node. Work-space
and audit assertions are structural checks, not time or heap/RSS acceptance.
"""

import copy
import gc

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, PlannerConfig, SharedInputBundle

from .test_mc_compact_graph import (
    compact_spec,
    compile_compact,
    ieee_bits,
    layout_view,
    probability_inputs,
    small_tape,
)


def canonical_spec(spec, public="full_nav"):
    out = copy.deepcopy(spec)
    name = f"canonical_public_intermediate_{len(out['expressions'])}"
    out["expressions"][name] = public
    out["root_bindings"].append(out["chains"][0].copy())
    return out


def selected_spec(spec, names):
    out = copy.deepcopy(spec)
    bound = dict(zip(spec["expressions"], spec["root_bindings"], strict=True))
    out["expressions"] = {name: spec["expressions"][name] for name in names}
    out["root_bindings"] = [bound[name].copy() for name in names]
    return out


def used_inputs(graph, inputs):
    return {name: inputs[name] for name in graph.metadata()["input_names"]}


def chunk_sum(result, key):
    return sum(chunk[key] for chunk in result.audit["native_chunks"])


def assert_private(result):
    # Dedicated counters cover direct-period and tape paths alike; a direct
    # period chain need not execute an affine pointwise tile at all.
    assert chunk_sum(result, "compact_path_groups") > 0
    assert chunk_sum(result, "compact_path_batches") > 0
    assert chunk_sum(result, "compact_path_steps") > 0
    assert result.audit["python_fallback"] == 0
    assert result.audit["python_operator_calls"] == 0
    assert result.audit["request_time_compilation"] == 0


def assert_canonical(result):
    assert chunk_sum(result, "compact_path_groups") == 0
    assert chunk_sum(result, "compact_path_batches") == 0
    assert chunk_sum(result, "compact_path_steps") == 0


def compare_selected(actual, reference):
    for name, out in actual.named_outputs.items():
        target = reference.named_outputs[name]
        assert out.axes == target.axes and out.dtype == target.dtype
        assert out.order == target.order
        np.testing.assert_array_equal(out.root_statuses, target.root_statuses)
        np.testing.assert_array_equal(out.shape_known, target.shape_known)
        assert len(out.values) == len(target.values)
        for value, expected in zip(out.values, target.values, strict=True):
            ieee_bits(value, expected)
            assert not value.flags.writeable


def saved_outputs(result):
    return {name: [value.copy() for value in out.values]
            for name, out in result.named_outputs.items()}


def check_ownership(first, later, saved, inputs):
    for name, snapshots in saved.items():
        old = first.named_outputs[name]
        new = later.named_outputs[name]
        for value, current, snapshot in zip(old.values, new.values, snapshots, strict=True):
            ieee_bits(value, snapshot)
            assert not np.shares_memory(value, current)
            assert all(not np.shares_memory(value, source) for source in inputs.values())
            assert all(not np.shares_memory(current, source) for source in inputs.values())
            assert not value.flags.writeable and not current.flags.writeable


def fixture(source, *, steps=4, paths=7, factors=2, assets=3,
            combinations=2, layout="C"):
    spec = compact_spec(steps, paths, assets if source == "tape" else 0,
                        combinations, source=source)
    if source == "tape":
        inputs = small_tape(steps, paths, factors, assets, combinations, layout)
    else:
        inputs = probability_inputs()
        base = (np.arange(steps * paths, dtype=np.float64).reshape(steps, paths) % 11 - 5) / 64
        for combination in range(combinations):
            inputs[f"period{combination}"] = layout_view(base * (combination + 1), layout)
    return spec, inputs


def execute(scheduler, graph, inputs, starts, ends, parameters, **kwargs):
    if isinstance(parameters, dict):
        parameters = {name: parameters[name] for name in graph.metadata()["parameter_names"]}
    return scheduler.execute(graph, used_inputs(graph, inputs), starts, ends,
                             parameters=parameters, **kwargs)


def execute_pair(scheduler, spec, inputs, starts, ends, parameters, *, order="C"):
    graph = compile_compact(spec, order=order)
    generic = compile_compact(canonical_spec(spec), order=order)
    actual = execute(scheduler, graph, inputs, starts, ends, parameters)
    reference = execute(scheduler, generic, inputs, starts, ends, parameters)
    compare_selected(actual, reference)
    assert_canonical(reference)
    return graph, generic, actual, reference


@pytest.mark.parametrize("source", ["period", "tape"])
@pytest.mark.parametrize("layout", ["C", "F", "negative", "strided"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("mode", ["ordinary", "prepared"])
def test_private_complete_chain_matches_canonical_layouts_and_two_owners(source, layout, order, mode):
    spec, inputs = fixture(source, layout=layout)
    originals = {name: value.copy() for name, value in inputs.items()}
    starts, ends = np.array([0], np.int64), np.array([4], np.int64)
    parameters = {"n0": 17.3, "target": .03}
    graph = compile_compact(spec, order=order)
    generic = compile_compact(canonical_spec(spec), order=order)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        reference = execute(scheduler, generic, inputs, starts, ends, parameters)
        assert_canonical(reference)
        if mode == "prepared":
            prepared = scheduler.prepare_execution(
                graph, used_inputs(graph, inputs), starts, ends, parameters=parameters)
            first = prepared.run_snapshot()
            later = prepared.run_snapshot()
        else:
            first = execute(scheduler, graph, inputs, starts, ends, parameters)
            later = execute(scheduler, graph, inputs, starts, ends, parameters)
        saved = saved_outputs(first)
        for result in (first, later):
            compare_selected(result, reference)
            assert_private(result)
            assert result.audit["result_lifetime"] == "independent"
            for out in result.outputs:
                for value in out.values:
                    if value.ndim >= 2:
                        assert value.flags.f_contiguous if order == "F" else value.flags.c_contiguous
        check_ownership(first, later, saved, inputs)
    for name, original in originals.items():
        ieee_bits(inputs[name], original)
    del inputs, reference, later, graph, generic
    gc.collect()
    for name, snapshots in saved.items():
        for value, snapshot in zip(first.named_outputs[name].values, snapshots, strict=True):
            ieee_bits(value, snapshot)


@pytest.mark.parametrize("factors,assets", [(0, 0), (0, 3), (2, 0), (4, 17)])
def test_private_model_empty_axes_and_packet_width_keep_the_complete_math(factors, assets):
    spec, inputs = fixture("tape", steps=3, paths=5, factors=factors, assets=assets)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        _, _, actual, _ = execute_pair(
            scheduler, spec, inputs, np.array([0], np.int64), np.array([3], np.int64),
            {"n0": .1, "target": 0.})
    assert_private(actual)


@pytest.mark.parametrize("source", ["period", "tape"])
@pytest.mark.parametrize("n0", [0., -0., np.nan, np.inf, -np.inf, -17.3])
def test_private_seed_ieee_and_errors_follow_canonical_without_financial_gates(source, n0):
    spec, inputs = fixture(source, combinations=1)
    starts, ends = np.array([0], np.int64), np.array([4], np.int64)
    parameters = {"n0": float(n0), "target": .1}
    graph, generic = compile_compact(spec), compile_compact(canonical_spec(spec))
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        if n0 == 0.:
            with pytest.raises((ValueError, RuntimeError), match="DIVIDE_BY_ZERO") as expected:
                execute(scheduler, generic, inputs, starts, ends, parameters)
            with pytest.raises(type(expected.value)) as actual:
                execute(scheduler, graph, inputs, starts, ends, parameters)
            assert str(actual.value) == str(expected.value)
        else:
            # max_axis and later scalar consumers retain their existing finite
            # gates. A nonfinite seed is legal for the new scan itself but is
            # not a promise that every downstream canonical node will succeed.
            try:
                reference = execute(scheduler, generic, inputs, starts, ends, parameters)
            except (ValueError, RuntimeError) as expected:
                with pytest.raises(type(expected)) as actual:
                    execute(scheduler, graph, inputs, starts, ends, parameters)
                assert str(actual.value) == str(expected)
            else:
                actual = execute(scheduler, graph, inputs, starts, ends, parameters)
                compare_selected(actual, reference)
                assert_private(actual)
                assert_canonical(reference)


@pytest.mark.parametrize("source", ["period", "tape"])
@pytest.mark.parametrize("probability", [np.nan, np.inf, -.1, 1.1])
@pytest.mark.parametrize("qname,n0,code", [
    ("q5", 0., "INVALID_PARAMETER"),
    ("qreturn", 0., "DIVIDE_BY_ZERO"),
    ("qdd", 0., "DIVIDE_BY_ZERO"),
    ("qreturn", 1., "INVALID_PARAMETER"),
])
def test_deferred_probability_failure_keeps_the_original_node_error_priority(source, probability, qname, n0, code):
    spec, inputs = fixture(source, combinations=2)
    inputs[qname] = inputs[qname].copy()
    inputs[qname][0] = probability
    starts, ends = np.array([0], np.int64), np.array([4], np.int64)
    parameters = {"n0": n0, "target": .1}
    graph, generic = compile_compact(spec), compile_compact(canonical_spec(spec))
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises((ValueError, RuntimeError), match=code) as expected:
            execute(scheduler, generic, inputs, starts, ends, parameters)
        with pytest.raises(type(expected.value)) as actual:
            execute(scheduler, graph, inputs, starts, ends, parameters)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("source", ["period", "tape"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_nonfinite_numeric_payload_does_not_override_a_real_probability_error(source, nonfinite):
    spec, inputs = fixture(source, combinations=2)
    key = "x" if source == "tape" else "period0"
    inputs[key] = inputs[key].copy()
    inputs[key].flat[-1] = nonfinite
    inputs["qreturn"] = np.array([1.1, .5, .95])
    graph, generic = compile_compact(spec), compile_compact(canonical_spec(spec))
    starts, ends = np.array([0], np.int64), np.array([4], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        with pytest.raises((ValueError, RuntimeError), match="INVALID_PARAMETER") as expected:
            execute(scheduler, generic, inputs, starts, ends, {"n0": 1., "target": .1})
        with pytest.raises(type(expected.value)) as actual:
            execute(scheduler, graph, inputs, starts, ends, {"n0": 1., "target": .1})
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("order", ["C", "F"])
def test_duplicate_fan_roots_remain_independent_outputs(order):
    spec, inputs = fixture("tape")
    spec["expressions"]["duplicate_fan"] = spec["expressions"]["fan_0"]
    spec["root_bindings"].append(spec["root_bindings"][0].copy())
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        _, _, actual, _ = execute_pair(
            scheduler, spec, inputs, np.array([0], np.int64), np.array([4], np.int64),
            {"n0": 13.2, "target": .1}, order=order)
    assert_private(actual)
    first = actual.named_outputs["fan_0"].values[0]
    duplicate = actual.named_outputs["duplicate_fan"].values[0]
    ieee_bits(first, duplicate)
    assert not np.shares_memory(first, duplicate)


@pytest.mark.parametrize("public", ["full_nav", "period", "simple"])
def test_public_intermediates_force_complete_canonical_budget_and_execution(public):
    spec, inputs = fixture("tape")
    visible = canonical_spec(spec, public)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        graph = compile_compact(visible)
        reference = compile_compact(canonical_spec(visible))
        starts, ends = np.array([0], np.int64), np.array([4], np.int64)
        actual = execute(scheduler, graph, inputs, starts, ends, {"n0": 1., "target": .1})
        expected = execute(scheduler, reference, inputs, starts, ends, {"n0": 1., "target": .1})
    compare_selected(actual, expected)
    assert_canonical(actual)


@pytest.mark.parametrize("selection", ["fan", "mixed", "extra"])
def test_selected_consumers_and_extra_independent_root_obey_exact_closure(selection):
    spec, inputs = fixture("tape")
    names = ["fan_1"] if selection == "fan" else ["fan_1", "stat_1_es_95", "contribution_1_1"]
    spec = selected_spec(spec, names)
    if selection == "extra":
        spec["expressions"]["independent"] = "mean(x)"
        spec["root_bindings"].append({})
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        _, _, actual, _ = execute_pair(
            scheduler, spec, inputs, np.array([0], np.int64), np.array([4], np.int64),
            {"n0": 1., "target": .1})
    # The first private proof requires the complete fan/sample/terminal/maxDD
    # consumer closure. Selected-only graphs preserve all public mathematics
    # through the canonical route and its full worker budget.
    assert tuple(actual.named_outputs) == tuple(spec["expressions"])
    assert_canonical(actual)
    assert chunk_sum(actual, "numeric_arena_bytes") > 0


@pytest.mark.parametrize("source", ["period", "tape"])
def test_isolate_uses_canonical_statuses_and_preserves_independent_healthy_roots(source):
    spec, inputs = fixture(source, combinations=1)
    inputs["qreturn"] = np.array([1.5, .5, .95])
    graph = compile_compact(spec, error_policy="isolate")
    generic = compile_compact(canonical_spec(spec), error_policy="isolate")
    starts, ends = np.array([0], np.int64), np.array([4], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        actual = execute(scheduler, graph, inputs, starts, ends, {"n0": 1., "target": .1})
        reference = execute(scheduler, generic, inputs, starts, ends, {"n0": 1., "target": .1})
    compare_selected(actual, reference)
    assert_canonical(actual)
    assert actual.named_outputs["fan_0"].root_statuses[0] == 0
    assert actual.named_outputs["stat_0_terminal_return_p05"].root_statuses[0] == 4


@pytest.mark.parametrize("order", ["C", "F"])
def test_prepared_reads_same_address_mutable_tape_model_parameters_and_probabilities(order):
    spec, frozen = fixture("tape", factors=4, assets=17, combinations=2)
    inputs = {name: value.copy() for name, value in frozen.items()}
    addresses = {name: value.__array_interface__["data"][0] for name, value in inputs.items()}
    graph, generic = compile_compact(spec, order=order), compile_compact(canonical_spec(spec), order=order)
    parameter_values = {"n0": 1., "target": .1}
    parameter_names = graph.metadata()["parameter_names"]
    parameters = np.array([parameter_values[name] for name in parameter_names], np.float64)
    positions = {name: index for index, name in enumerate(parameter_names)}
    starts, ends = np.array([0], np.int64), np.array([4], np.int64)
    retained = []
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, used_inputs(graph, inputs), starts, ends,
                                               parameters=parameters)
        retained.append(prepared.run_snapshot())
        snapshots = [saved_outputs(retained[0])]
        mutations = ("beta", "x", "initial", "w1", "n0", "target", "q5", "qreturn", "qdd")
        for key in mutations:
            if key in {"beta", "x", "initial", "w1"}:
                inputs[key][:] += .001
            elif key == "n0":
                parameters[positions[key]] = 17.3
            elif key == "target":
                parameters[positions[key]] = -.5
            elif key == "q5":
                inputs[key][:] = [.95, .75, .5, .25, .05]
            elif key == "qreturn":
                inputs[key][:] = [.1, .6, .9]
            else:
                inputs[key][:] = [.5]
            current = prepared.run_snapshot()
            numeric_parameters = {name: float(parameters[index]) for name, index in positions.items()}
            expected = execute(scheduler, generic, inputs, starts, ends, numeric_parameters)
            compare_selected(current, expected)
            assert_private(current)
            for old, saved in zip(retained, snapshots, strict=True):
                check_ownership(old, current, saved, inputs)
            retained.append(current)
            snapshots.append(saved_outputs(current))
        assert all(value.__array_interface__["data"][0] == addresses[name]
                   for name, value in inputs.items())
        inputs["q5"][0] = 1.1
        with pytest.raises((ValueError, RuntimeError), match="INVALID_PARAMETER"):
            prepared.run_snapshot()
        inputs["q5"][0] = .95
        recovered = prepared.run_snapshot()
        expected = execute(scheduler, generic, inputs, starts, ends,
                           {name: float(parameters[index]) for name, index in positions.items()})
        compare_selected(recovered, expected)
        assert_private(recovered)
        for old, saved in zip(retained, snapshots, strict=True):
            check_ownership(old, recovered, saved, inputs)
    del inputs, expected, recovered, prepared, graph, generic
    gc.collect()
    for result, saved in zip(retained, snapshots, strict=True):
        for name, values in saved.items():
            for value, snapshot in zip(result.named_outputs[name].values, values, strict=True):
                ieee_bits(value, snapshot)


def test_compact_workspace_is_bounded_and_admitted_at_exact_total_budget():
    # A small fixture with a long enough editable graph to distinguish bounded
    # state from canonical [T,P,A] intermediates. No timing is measured here.
    spec, inputs = fixture("tape", steps=33, paths=31, factors=4, assets=17)
    graph, generic = compile_compact(spec), compile_compact(canonical_spec(spec))
    starts, ends = np.array([0], np.int64), np.array([33], np.int64)
    parameters = {"n0": 17.3, "target": .1}
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(graph, inputs, starts, ends)
        canonical_plan = scheduler.plan(generic, inputs, starts, ends)
        assert plan.estimated_worker_scratch_bytes * 2 < canonical_plan.estimated_worker_scratch_bytes
        with pytest.raises(MemoryError):
            execute(scheduler, graph, inputs, starts, ends, parameters,
                    memory_budget_bytes=plan.estimated_total_memory_bytes - 1)
        actual = execute(scheduler, graph, inputs, starts, ends, parameters,
                         memory_budget_bytes=plan.estimated_total_memory_bytes)
        later = execute(scheduler, graph, inputs, starts, ends, parameters,
                        memory_budget_bytes=plan.estimated_total_memory_bytes)
        expected = execute(scheduler, generic, inputs, starts, ends, parameters)
        compare_selected(actual, expected)
        compare_selected(later, expected)
        assert_private(actual)
        assert chunk_sum(actual, "order_stat_sorts") > 0
        # A shared P-sort serves all q values; never Q full copies/sorts per step.
        assert chunk_sum(actual, "order_stat_sorts") <= spec["combinations"] * (spec["steps"] + 3)
        for chunk in actual.audit["native_chunks"]:
            assert chunk["operator_workspace_capacity_bytes"] <= plan.estimated_worker_scratch_bytes
        check_ownership(actual, later, saved_outputs(actual), inputs)


def test_prepared_large_then_small_child_capacity_matches_fresh_current_plan():
    # samples[12,T+1] keeps the child maximum-array bound unchanged while P
    # shrinks. The child has to release its old retained workspace anyway.
    spec, large_inputs = fixture("tape", steps=33, paths=300, factors=4,
                                 assets=17, combinations=1)
    _, small_inputs = fixture("tape", steps=33, paths=16, factors=4,
                              assets=17, combinations=1)
    graph, generic = compile_compact(spec), compile_compact(canonical_spec(spec))
    starts, ends = np.array([0], np.int64), np.array([33], np.int64)
    parameters = {"n0": 17.3, "target": .1}
    capacity_fields = ("numeric_arena_bytes", "mask_arena_bytes",
                       "order_scratch_capacity_bytes",
                       "pointwise_workspace_capacity_bytes",
                       "operator_workspace_capacity_bytes")
    with AdaptiveScheduler(cpu_budget=1) as fresh_scheduler:
        fresh = execute(fresh_scheduler, graph, small_inputs, starts, ends, parameters)
        expected = execute(fresh_scheduler, generic, small_inputs, starts, ends, parameters)
    assert_private(fresh)
    compare_selected(fresh, expected)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        large = scheduler.prepare_execution(graph, large_inputs, starts, ends, parameters=parameters)
        small_plan = scheduler.plan(graph, small_inputs, starts, ends)
        small = scheduler.prepare_execution(graph, small_inputs, starts, ends,
                                            parameters=parameters, plan=small_plan)
        before = large.run_snapshot()
        saved_large = saved_outputs(before)
        first, later = small.run_snapshot(), small.run_snapshot()
        for result in (first, later):
            compare_selected(result, expected)
            assert_private(result)
            assert result.plan.estimated_worker_scratch_bytes == fresh.plan.estimated_worker_scratch_bytes
            for key in capacity_fields:
                assert chunk_sum(result, key) == chunk_sum(fresh, key)
            for chunk in result.audit["native_chunks"]:
                assert chunk["operator_workspace_capacity_bytes"] <= small_plan.estimated_worker_scratch_bytes
        check_ownership(first, later, saved_outputs(first), small_inputs)
        # Returning to the large prepared binding may grow its own workspace,
        # while neither retained small nor old large output can be overwritten.
        restored = large.run_snapshot()
        check_ownership(before, restored, saved_large, large_inputs)
        for name, snapshots in saved_large.items():
            for value, snapshot in zip(restored.named_outputs[name].values, snapshots, strict=True):
                ieee_bits(value, snapshot)


@pytest.mark.parametrize("source", ["period", "tape"])
def test_heterogeneous_interval_lengths_decline_before_writes_and_use_full_budget(source):
    spec, inputs = fixture(source, steps=5)
    # The terminal slice is explicitly tied to T=2. A three-step row has a
    # different consumer geometry and cannot be silently admitted as that T.
    spec = compact_spec(2, 7, 3 if source == "tape" else 0, 2, source=source)
    graph, generic = compile_compact(spec), compile_compact(canonical_spec(spec))
    starts, ends = np.array([0, 3], np.int64), np.array([3, 5], np.int64)
    parameters = {"n0": 1., "target": .1}
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(graph, inputs, starts, ends)
        with pytest.raises(MemoryError):
            execute(scheduler, graph, inputs, starts, ends, parameters,
                    memory_budget_bytes=plan.estimated_total_memory_bytes - 1)
        actual = execute(scheduler, graph, inputs, starts, ends, parameters,
                         memory_budget_bytes=plan.estimated_total_memory_bytes)
        expected = execute(scheduler, generic, inputs, starts, ends, parameters)
    compare_selected(actual, expected)
    assert_canonical(actual)
    assert actual.named_outputs["fan_0"].values[0].shape == (5, 4)
    assert actual.named_outputs["fan_0"].values[1].shape == (5, 3)


def process_config(lane):
    return PlannerConfig(
        thread_work_units=1e100,
        process_work_units=1e100 if lane == "hard_stop" else 1,
        shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
        min_rows_per_worker=1,
        max_processes=2,
    )


@pytest.mark.parametrize("lane", ["process_inline", "process_shared", "hard_stop"])
@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_complete_private_roots_process_transport_and_shared_lifetime(lane, layout, order):
    _, inputs = fixture("tape", steps=8, paths=17, factors=4, assets=17,
                        combinations=2, layout=layout)
    spec = compact_spec(4, 17, 17, 2)
    graph, generic = compile_compact(spec, order=order), compile_compact(canonical_spec(spec), order=order)
    starts, ends = np.array([0, 4], np.int64), np.array([4, 8], np.int64)
    parameters = {"n0": 17.3, "target": .1}
    originals = {name: value.copy() for name, value in inputs.items()}
    with AdaptiveScheduler(cpu_budget=1) as single:
        reference = execute(single, generic, inputs, starts, ends, parameters)
    bundle = SharedInputBundle.from_inputs(inputs) if lane == "process_shared" else None
    transported = bundle.arrays if bundle is not None else inputs
    try:
        with AdaptiveScheduler(cpu_budget=2, config=process_config(lane)) as scheduler:
            options = {"hard_stop": lane == "hard_stop", "timeout": 20.}
            first = execute(scheduler, graph, transported, starts, ends, parameters, **options)
            saved = saved_outputs(first)
            later = execute(scheduler, graph, transported, starts, ends, parameters, **options)
            for result in (first, later):
                compare_selected(result, reference)
                assert_private(result)
                assert result.plan.lane == "process"
                assert result.plan.process_count <= 2
                assert result.audit["cpu_tokens"] <= 2
                assert result.audit["python_worker_callbacks"] == 0
                assert result.audit["native_processes"] > 0
                assert result.audit["result_lifetime"] == "independent"
                if lane != "process_inline":
                    assert result.plan.use_shared_memory
                    assert result.audit["use_shared_memory"]
                    assert result.audit["output_ownership"] == "native_shared_region"
                    assert result.audit["output_copy_bytes"] == 0
                else:
                    assert not result.plan.use_shared_memory
                if lane == "hard_stop":
                    assert result.plan.hard_stop
                    assert "hard_stop_uses_shared_process_transport" in result.plan.reason_codes
                for out in result.outputs:
                    np.testing.assert_array_equal(out.root_statuses, np.zeros(2, np.int16))
                    for value in out.values:
                        if value.ndim >= 2:
                            assert value.flags.f_contiguous if order == "F" else value.flags.c_contiguous
            check_ownership(first, later, saved, transported)
    finally:
        if bundle is not None:
            bundle.release()
    for name, original in originals.items():
        ieee_bits(inputs[name], original)
    del inputs, transported, later, reference, bundle, graph, generic
    gc.collect()
    # Every fan/statistic/sample/contribution still has its independent owner
    # after the source bundle, native workers and other generation are gone.
    for name, snapshots in saved.items():
        for value, snapshot in zip(first.named_outputs[name].values, snapshots, strict=True):
            ieee_bits(value, snapshot)


@pytest.mark.parametrize("lane", ["process_inline", "process_shared", "hard_stop"])
@pytest.mark.parametrize("qname,n0,code", [
    ("q5", 0., "INVALID_PARAMETER"),
    ("qreturn", 0., "DIVIDE_BY_ZERO"),
])
def test_private_process_errors_match_canonical_priority_and_next_call_recovers(lane, qname, n0, code):
    spec, frozen = fixture("tape", combinations=2)
    inputs = {name: value.copy() for name, value in frozen.items()}
    graph, generic = compile_compact(spec), compile_compact(canonical_spec(spec))
    starts, ends = np.array([0], np.int64), np.array([4], np.int64)
    good_parameters = {"n0": 1., "target": .1}
    options = {"hard_stop": lane == "hard_stop", "timeout": 20.}
    with AdaptiveScheduler(cpu_budget=2, config=process_config(lane)) as scheduler:
        before = execute(scheduler, graph, inputs, starts, ends, good_parameters, **options)
        saved = saved_outputs(before)
        original = float(inputs[qname][0])
        inputs[qname][0] = 1.1
        with pytest.raises((ValueError, RuntimeError), match=code) as expected:
            execute(scheduler, generic, inputs, starts, ends,
                    {"n0": n0, "target": .1}, **options)
        with pytest.raises(type(expected.value)) as actual:
            execute(scheduler, graph, inputs, starts, ends,
                    {"n0": n0, "target": .1}, **options)
        assert str(actual.value) == str(expected.value)
        inputs[qname][0] = original
        recovered = execute(scheduler, graph, inputs, starts, ends, good_parameters, **options)
        reference = execute(scheduler, generic, inputs, starts, ends, good_parameters, **options)
        compare_selected(recovered, reference)
        assert_private(recovered)
        assert_canonical(reference)
        check_ownership(before, recovered, saved, inputs)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("provided_plan", [False, True])
def test_process_prepared_remains_explicitly_unsupported_by_existing_api(order, provided_plan):
    _, inputs = fixture("tape", steps=8)
    spec = compact_spec(4, 7, 3, 2)
    graph = compile_compact(spec, order=order)
    starts, ends = np.array([0, 4], np.int64), np.array([4, 8], np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=process_config("process_shared")) as scheduler:
        plan = scheduler.plan(graph, inputs, starts, ends)
        assert plan.lane == "process"
        with pytest.raises(ValueError, match="prepared execution currently requires a single-lane plan"):
            scheduler.prepare_execution(graph, inputs, starts, ends,
                                        parameters={"n0": 1., "target": .1},
                                        plan=plan if provided_plan else None)


@pytest.mark.parametrize("mode", ["ordinary", "prepared"])
@pytest.mark.parametrize("matrix_operand", [False, True])
def test_canonical_post_arithmetic_pair_reserves_the_unfused_child_arena(mode, matrix_operand):
    spec, inputs = fixture("period", steps=4, paths=257, combinations=1)
    if matrix_operand:
        spec["variables"]["v"] = dict(kind="matrix", dtype="float64",
                                      axes=["scenario", "path"], shape=["64", "P"],
                                      semantic_dimension="dimensionless")
        inputs["v"] = (np.arange(64 * 257, dtype=np.float64).reshape(64, 257) % 13 - 6) / 32
        expression = "sum(add(multiply(broadcast_like(max_dd,v),v),n0))"
    else:
        spec["variables"]["v"] = dict(kind="vector", dtype="float64",
                                      axes=["path"], shape=["P"],
                                      semantic_dimension="dimensionless")
        inputs["v"] = np.linspace(-.5, .5, 257)
        expression = "sum(add(multiply(max_dd,v),n0))"
    inputs["v"].flags.writeable = False
    # A valid scalar post consumer can contain an otherwise fusible array pair.
    # Deferred original-node checkpoints deliberately prohibit that pair in the
    # post executor, so the shared child estimate must reserve both temporaries.
    spec["expressions"]["custom_post"] = expression
    spec["root_bindings"].append(spec["chains"][0].copy())
    graph, generic = compile_compact(spec), compile_compact(canonical_spec(spec))
    starts, ends = np.array([0], np.int64), np.array([4], np.int64)
    parameters = {"n0": 17.3, "target": .1}
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(graph, inputs, starts, ends)
        with pytest.raises(MemoryError):
            execute(scheduler, graph, inputs, starts, ends, parameters,
                    memory_budget_bytes=plan.estimated_total_memory_bytes - 1)
        if mode == "prepared":
            prepared = scheduler.prepare_execution(graph, inputs, starts, ends,
                                                   parameters=parameters, plan=plan)
            first, later = prepared.run_snapshot(), prepared.run_snapshot()
        else:
            first = execute(scheduler, graph, inputs, starts, ends, parameters,
                            memory_budget_bytes=plan.estimated_total_memory_bytes)
            later = execute(scheduler, graph, inputs, starts, ends, parameters,
                            memory_budget_bytes=plan.estimated_total_memory_bytes)
        reference = execute(scheduler, generic, inputs, starts, ends, parameters)
        for result in (first, later):
            compare_selected(result, reference)
            assert_private(result)
            for chunk in result.audit["native_chunks"]:
                assert chunk["numeric_arena_bytes"] + chunk["mask_arena_bytes"] <= plan.estimated_worker_scratch_bytes
                assert chunk["operator_workspace_capacity_bytes"] <= plan.estimated_worker_scratch_bytes
        if matrix_operand:
            # These two live rank-two temporaries cannot borrow or overwrite
            # each other. The larger operand prevents old P-sized slot bounds
            # elsewhere in the post DAG from accidentally hiding one omission.
            assert chunk_sum(first, "numeric_arena_bytes") >= 2 * inputs["v"].nbytes
        check_ownership(first, later, saved_outputs(first), inputs)


@pytest.mark.parametrize("mode", ["ordinary", "prepared"])
def test_duplicate_scalar_post_roots_keep_independent_shapes_and_budgeted_child_audit(mode):
    spec, inputs = fixture("period", steps=3, paths=17, combinations=1)
    name = "stat_0_average_max_drawdown"
    position = list(spec["expressions"]).index(name)
    binding = spec["root_bindings"][position]
    for index in range(512):
        spec["expressions"][f"duplicate_scalar_{index}"] = spec["expressions"][name]
        spec["root_bindings"].append(binding.copy())
    graph, generic = compile_compact(spec, order="F"), compile_compact(canonical_spec(spec), order="F")
    starts, ends = np.array([0], np.int64), np.array([3], np.int64)
    parameters = {"n0": 17.3, "target": .1}
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(graph, inputs, starts, ends)
        with pytest.raises(MemoryError):
            execute(scheduler, graph, inputs, starts, ends, parameters,
                    memory_budget_bytes=plan.estimated_total_memory_bytes - 1)
        if mode == "prepared":
            prepared = scheduler.prepare_execution(graph, inputs, starts, ends,
                                                   parameters=parameters, plan=plan)
            first, later = prepared.run_snapshot(), prepared.run_snapshot()
        else:
            first = execute(scheduler, graph, inputs, starts, ends, parameters,
                            memory_budget_bytes=plan.estimated_total_memory_bytes)
            later = execute(scheduler, graph, inputs, starts, ends, parameters,
                            memory_budget_bytes=plan.estimated_total_memory_bytes)
        reference = execute(scheduler, generic, inputs, starts, ends, parameters)
        compare_selected(first, reference)
        compare_selected(later, reference)
        assert_private(first)
        check_ownership(first, later, saved_outputs(first), inputs)
    original = first.named_outputs[name].values[0]
    for index in range(512):
        out = first.named_outputs[f"duplicate_scalar_{index}"]
        assert out.values[0].shape == ()
        np.testing.assert_array_equal(out.root_statuses, [0])
        np.testing.assert_array_equal(out.shape_known, [True])
        ieee_bits(out.values[0], original)
        assert not np.shares_memory(out.values[0], original)
