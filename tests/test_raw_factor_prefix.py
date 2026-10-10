"""Raw fixed factor tapes through the editable complete consumer graph.

The small independent oracle covers finite draws, finite loadings/means and
positive chi values. It does not generate random streams, construct covariance
roots or certify FIRP's NaN/negative-chi or stage-ordered first-fault behavior.
Private execution is also compared with the same public canonical AST. No
helper in this file participates in production numerical execution.
"""

import copy
import gc
import math

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, SharedInputBundle

from .test_compact_path_execution import (
    assert_canonical,
    assert_private,
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
    compare_named,
    compile_compact,
    declaration,
    expected_named,
    ieee_bits,
    layout_view,
    probability_inputs,
)
from .test_typed_results import config


def raw_factor_spec(steps, paths, assets, combinations, student=False):
    """Replace the transformed input with an inspectable mathematical prefix.

    F (factors) and I (innovations) are independent input dimensions. Every old
    output and its reachable bindings remain present. Scalar df/n0/target are
    passed through GraphCompiler's existing parameter interface. The innovation
    dimension uses the existing public axis name component, not new vocabulary.
    """
    spec = compact_spec(steps, paths, assets, combinations)
    del spec["variables"]["x"]
    spec["variables"].update({
        "raw": declaration("tensor", ["time", "path", "component"], ["T", "P", "I"]),
        "root": declaration("matrix", ["factor", "component"], ["F", "I"]),
        "means": declaration("vector", ["factor"], ["F"]),
    })
    prefix = {"rotated": "affine_contract(raw,transpose(root),0)"}
    if student:
        spec["variables"].update({
            "df": declaration("scalar"),
            # A unit axis must still have the same axis identity as the
            # broadcast destination; component would be rejected here.
            "chi3": declaration("tensor", ["time", "path", "factor"], ["T", "P", "1"]),
        })
        prefix["scale"] = (
            "divide(sqrt(divide(subtract(df,2),df)),sqrt(divide(chi3,df)))"
        )
        prefix["x"] = (
            "add(broadcast_like(means,rotated),"
            "multiply(rotated,broadcast_like(scale,rotated)))"
        )
    else:
        # broadcast_like does not accept a scalar. The explicit scale=1
        # multiplication retains the normal path's original arithmetic tree.
        prefix["x"] = "add(broadcast_like(means,rotated),multiply(rotated,1))"
    spec["root_bindings"] = [{**prefix, **binding} for binding in spec["root_bindings"]]
    spec["chains"] = [{**prefix, **chain} for chain in spec["chains"]]
    spec["student"] = bool(student)
    return spec


def small_raw_tape(steps=9, paths=7, factors=3, innovations=4, assets=9,
                   combinations=2, student=False, layout="C"):
    """Deterministic, bounded successful fixtures; no RNG or engine oracle."""
    raw = (np.arange(steps * paths * innovations, dtype=np.float64)
           .reshape(steps, paths, innovations) % 19 - 9) / 64
    root = (np.arange(factors * innovations, dtype=np.float64)
            .reshape(factors, innovations) % 7 - 3) / 16
    means = (np.arange(factors, dtype=np.float64) - 1) / 2048
    beta = (np.arange(assets * factors, dtype=np.float64)
            .reshape(assets, factors) % 11 - 5) / 64
    inputs = dict(raw=raw, root=root, means=means, beta=beta,
                  initial=np.arange(assets, dtype=np.float64) / 4096)
    for combination in range(combinations):
        inputs[f"w{combination}"] = (
            (np.arange(assets, dtype=np.float64) + 1)
            * (-1 if combination % 2 else 1) / max(assets, 1)
        )
    if student:
        inputs["chi3"] = (
            2 + np.arange(steps * paths, dtype=np.float64).reshape(steps, paths, 1) % 9
        )
    inputs = {name: layout_view(value, layout) for name, value in inputs.items()}
    inputs.update(probability_inputs())
    return inputs


def reference_raw_transform(inputs, *, student=False, df=5.):
    """Initial-zero ordered fold, then multiply scale, then add the mean.

    Only small finite successful fixtures call this oracle. In particular it
    does not replace native canonical errors with NumPy's IEEE defaults.
    """
    raw, root, means = inputs["raw"], inputs["root"], inputs["means"]
    steps, paths, innovations = raw.shape
    transformed = np.empty((steps, paths, len(means)), np.float64)
    constant = math.sqrt((df - 2.) / df) if student else 1.
    for step in range(steps):
        for path in range(paths):
            scale = constant / math.sqrt(float(inputs["chi3"][step, path, 0]) / df) if student else 1.
            for factor in range(len(means)):
                value = np.float64(0.)
                for innovation in range(innovations):
                    # All operands are finite in this oracle. Multiplication
                    # order also agrees with canonical affine_contract.
                    product = np.float64(root[factor, innovation] * raw[step, path, innovation])
                    value = np.float64(value + product)
                scaled = np.float64(value * np.float64(scale))
                transformed[step, path, factor] = np.float64(means[factor] + scaled)
    return transformed


def expected_raw_named(spec, inputs, *, n0=17.3, target=.02, df=5.):
    transformed = reference_raw_transform(inputs, student=spec["student"], df=df)
    return expected_named(spec, {**inputs, "x": transformed}, n0=n0, target=target)


def public_factor_spec(spec):
    """Publishing x rejects private admission, preserving the canonical DAG."""
    out = copy.deepcopy(spec)
    out["expressions"]["public_factors"] = "x"
    out["root_bindings"].append(spec["chains"][0].copy())
    return out


def raw_parameters(spec, *, n0=17.3, target=.02, df=5.):
    values = {"n0": float(n0), "target": float(target)}
    if spec["student"]:
        values["df"] = float(df)
    return values


def assert_raw_private(result, *, steps, paths, groups, rows=1, student=False):
    assert_private(result)
    assert "bounded_compact_scan_consumer_closure" in result.plan.reason_codes
    assert chunk_sum(result, "compact_path_raw_rows") == rows * steps * paths
    assert chunk_sum(result, "compact_path_raw_scale_elements") == (rows * steps * paths if student else 0)
    assert chunk_sum(result, "compact_path_raw_pack_bytes") % 8 == 0
    assert chunk_sum(result, "compact_path_groups") == rows * groups
    assert chunk_sum(result, "compact_path_batches") == rows
    assert chunk_sum(result, "compact_path_steps") == rows * steps
    for chunk in result.audit["native_chunks"]:
        assert chunk["operator_workspace_capacity_bytes"] <= result.plan.estimated_worker_scratch_bytes
    assert result.audit["result_lifetime"] == "independent"


@pytest.mark.parametrize("student", [False, True])
@pytest.mark.parametrize("layout", ["C", "F", "negative", "strided"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_raw_factor_complete_roots_match_independent_oracle_and_public_ast(student, layout, order):
    spec = raw_factor_spec(9, 7, 9, 2, student)
    inputs = small_raw_tape(student=student, layout=layout)
    originals = {name: value.copy() for name, value in inputs.items()}
    graph = compile_compact(spec, order=order)
    canonical = compile_compact(public_factor_spec(spec), order=order)
    starts, ends = np.array([0], np.int64), np.array([9], np.int64)
    parameters = raw_parameters(spec)
    expected = expected_raw_named(spec, inputs)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        reference = execute(scheduler, canonical, inputs, starts, ends, parameters)
        assert_canonical(reference)
        first = execute(scheduler, graph, inputs, starts, ends, parameters)
        saved = saved_outputs(first)
        later = execute(scheduler, graph, inputs, starts, ends, parameters)
        for result in (first, later):
            compare_named(result, expected)
            compare_selected(result, reference)
            assert_raw_private(result, steps=9, paths=7, groups=2, student=student)
            for out in result.outputs:
                for value in out.values:
                    if value.ndim >= 2:
                        assert value.flags.f_contiguous if order == "F" else value.flags.c_contiguous
        check_ownership(first, later, saved, inputs)
    for name, original in originals.items():
        ieee_bits(inputs[name], original)
        assert not inputs[name].flags.writeable
    del scheduler, graph, canonical, inputs, reference, later
    gc.collect()
    for name, values in saved.items():
        for value, snapshot in zip(first.named_outputs[name].values, values, strict=True):
            ieee_bits(value, snapshot)


@pytest.mark.parametrize("student", [False, True])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_raw_prefix_two_intervals_transport_counters_and_result_owners(student, lane):
    spec = raw_factor_spec(9, 7, 9, 2, student)
    inputs = small_raw_tape(steps=18, student=student, layout="F")
    originals = {name: value.copy() for name, value in inputs.items()}
    graph = compile_compact(spec, order="F")
    starts, ends = np.array([0, 9], np.int64), np.array([9, 18], np.int64)
    parameters = raw_parameters(spec)
    expected = []
    for start, stop in zip(starts, ends, strict=True):
        window = {**inputs, "raw": inputs["raw"][start:stop]}
        if student:
            window["chi3"] = inputs["chi3"][start:stop]
        expected.append(expected_raw_named(spec, window))
    bundle = SharedInputBundle.from_inputs(inputs) if lane == "process_shared" else None
    transported = bundle.arrays if bundle is not None else inputs
    try:
        with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
            first = execute(scheduler, graph, transported, starts, ends, parameters, timeout=20.)
            saved = saved_outputs(first)
            later = execute(scheduler, graph, transported, starts, ends, parameters, timeout=20.)
            for result in (first, later):
                assert result.plan.lane == ("process" if lane.startswith("process") else lane)
                assert result.audit["cpu_tokens"] <= 2
                assert_raw_private(result, steps=9, paths=7, groups=2, rows=2, student=student)
                for row, target in enumerate(expected):
                    compare_named(result, target, row=row)
                if lane.startswith("process"):
                    assert result.audit["native_processes"] > 0
                    assert result.audit["python_worker_callbacks"] == 0
                    assert result.plan.use_shared_memory == (lane == "process_shared")
                if lane == "process_shared":
                    assert result.audit["output_copy_bytes"] == 0
                    assert result.audit["output_ownership"] == "native_shared_region"
            check_ownership(first, later, saved, transported)
    finally:
        if bundle is not None:
            bundle.release()
    for name, original in originals.items():
        ieee_bits(inputs[name], original)
    del graph, inputs, transported, later, bundle
    gc.collect()
    for name, values in saved.items():
        for value, snapshot in zip(first.named_outputs[name].values, values, strict=True):
            ieee_bits(value, snapshot)


@pytest.mark.parametrize("student", [False, True])
@pytest.mark.parametrize("order", ["C", "F"])
def test_prepared_raw_prefix_reads_current_same_address_inputs_and_parameters(student, order):
    spec = raw_factor_spec(9, 7, 9, 2, student)
    owners = {name: value.copy(order="K") for name, value in small_raw_tape(student=student).items()}
    inputs = {name: value.view() for name, value in owners.items()}
    for value in inputs.values():
        value.flags.writeable = False
    addresses = {name: value.__array_interface__["data"][0] for name, value in inputs.items()}
    graph = compile_compact(spec, order=order)
    canonical = compile_compact(public_factor_spec(spec), order=order)
    names = graph.metadata()["parameter_names"]
    initial_parameters = raw_parameters(spec)
    parameters = np.array([initial_parameters[name] for name in names], np.float64)
    positions = {name: index for index, name in enumerate(names)}
    starts, ends = np.array([0], np.int64), np.array([9], np.int64)
    retained, snapshots = [], []
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(
            graph, used_inputs(graph, inputs), starts, ends, parameters=parameters)
        mutations = [None, "raw", "root", "means", "beta", "initial", "w1", "n0", "target"]
        if student:
            mutations.extend(("chi3", "df"))
        for key in mutations:
            if key in owners:
                owners[key][:] += .001
            elif key is not None:
                parameters[positions[key]] = {"n0": 13.2, "target": -.02, "df": 7.}[key]
            values = {name: float(parameters[index]) for name, index in positions.items()}
            current = prepared.run_snapshot()
            expected = expected_raw_named(spec, inputs, **values)
            reference = execute(scheduler, canonical, inputs, starts, ends, values)
            compare_named(current, expected)
            compare_selected(current, reference)
            assert_raw_private(current, steps=9, paths=7, groups=2, student=student)
            for old, saved in zip(retained, snapshots, strict=True):
                check_ownership(old, current, saved, inputs)
            retained.append(current)
            snapshots.append(saved_outputs(current))
        assert all(value.__array_interface__["data"][0] == addresses[name] for name, value in inputs.items())
        assert all(not value.flags.writeable for value in inputs.values())
    del prepared, graph, canonical, inputs, owners, current, reference
    gc.collect()
    for result, saved in zip(retained, snapshots, strict=True):
        for name, values in saved.items():
            for value, snapshot in zip(result.named_outputs[name].values, values, strict=True):
                ieee_bits(value, snapshot)


@pytest.mark.parametrize("student", [False, True])
def test_public_transformed_factor_root_preserves_the_full_canonical_budget(student):
    spec = public_factor_spec(raw_factor_spec(9, 7, 9, 2, student))
    inputs = small_raw_tape(student=student)
    graph = compile_compact(spec)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        actual = execute(scheduler, graph, inputs, np.array([0], np.int64), np.array([9], np.int64),
                         raw_parameters(spec))
    assert_canonical(actual)
    assert chunk_sum(actual, "compact_path_raw_rows") == 0
    assert chunk_sum(actual, "compact_path_raw_scale_elements") == 0
    assert chunk_sum(actual, "numeric_arena_bytes") > 0
    ieee_bits(actual.named_outputs["public_factors"].values[0],
              reference_raw_transform(inputs, student=student))


@pytest.mark.parametrize("student", [False, True])
def test_short_many_combinations_decline_prefix_fusion_before_full_canonical_execution(student):
    spec = raw_factor_spec(5, 3, 3, 32, student)
    inputs = small_raw_tape(steps=5, paths=3, factors=2, innovations=3,
                            assets=3, combinations=32, student=student)
    graph = compile_compact(spec)
    canonical = compile_compact(public_factor_spec(spec))
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        starts, ends = np.array([0], np.int64), np.array([5], np.int64)
        actual = execute(scheduler, graph, inputs, starts, ends, raw_parameters(spec))
        expected = execute(scheduler, canonical, inputs, starts, ends, raw_parameters(spec))
    compare_selected(actual, expected)
    compare_named(actual, expected_raw_named(spec, inputs))
    assert_canonical(actual)
    assert chunk_sum(actual, "compact_path_raw_rows") == 0


@pytest.mark.parametrize("student,order,lane", [
    (False, "C", "single"),
    (True, "F", "single"),
    (False, "F", "thread"),
    (True, "C", "thread"),
    (False, "C", "process_inline"),
    (True, "F", "process_inline"),
    (False, "F", "process_shared"),
    (True, "C", "process_shared"),
])
def test_short_wide_factor_prefix_full_batch_reuses_rotation_across_all_combinations(student, order, lane):
    # Keep the small-F refusal fixture above: full-batch admission must depend
    # on proved shape/cost, not merely on a short period or many combinations.
    # Include contribution accumulators: 31*(3*17+32)=2573 <= 5*17*32=2720.
    spec = raw_factor_spec(5, 17, 32, 32, student)
    inputs = small_raw_tape(steps=5, paths=17, factors=32, innovations=4,
                            assets=32, combinations=32, student=student, layout=order)
    originals = {name: value.copy() for name, value in inputs.items()}
    graph = compile_compact(spec, order=order)
    canonical = compile_compact(public_factor_spec(spec), order=order)
    parameters = raw_parameters(spec)
    expected = expected_raw_named(spec, inputs)
    # The planner needs two intervals for the real thread/process lanes. Two
    # identical [0,5) requests keep the tape itself at the requested T=5 and
    # prove one rotation per T*P and one full batch independently per interval.
    rows = 1 if lane == "single" else 2
    starts, ends = np.zeros(rows, np.int64), np.full(rows, 5, np.int64)
    bundle = SharedInputBundle.from_inputs(inputs) if lane == "process_shared" else None
    transported = bundle.arrays if bundle is not None else inputs
    try:
        with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
            reference = execute(scheduler, canonical, transported, starts, ends, parameters,
                                timeout=20.)
            assert_canonical(reference)
            first = execute(scheduler, graph, transported, starts, ends, parameters, timeout=20.)
            saved = saved_outputs(first)
            later = execute(scheduler, graph, transported, starts, ends, parameters, timeout=20.)
            for result in (first, later):
                assert result.plan.lane == ("process" if lane.startswith("process") else lane)
                assert result.audit["cpu_tokens"] <= 2
                assert_raw_private(result, steps=5, paths=17, groups=32, rows=rows, student=student)
                compare_selected(result, reference)
                for row in range(rows):
                    compare_named(result, expected, row=row)
                for out in result.outputs:
                    for value in out.values:
                        if value.ndim >= 2:
                            assert value.flags.f_contiguous if order == "F" else value.flags.c_contiguous
                if lane.startswith("process"):
                    assert result.audit["native_processes"] > 0
                    assert result.audit["python_worker_callbacks"] == 0
                    assert result.plan.use_shared_memory == (lane == "process_shared")
                if lane == "process_shared":
                    assert result.audit["output_copy_bytes"] == 0
                    assert result.audit["output_ownership"] == "native_shared_region"
            check_ownership(first, later, saved, transported)
            for name, original in originals.items():
                ieee_bits(transported[name], original)
                assert not transported[name].flags.writeable
    finally:
        if bundle is not None:
            bundle.release()
    for name, original in originals.items():
        ieee_bits(inputs[name], original)
        assert not inputs[name].flags.writeable
    del scheduler, graph, canonical, inputs, transported, reference, later, bundle
    gc.collect()
    for name, values in saved.items():
        for value, snapshot in zip(first.named_outputs[name].values, values, strict=True):
            ieee_bits(value, snapshot)


@pytest.mark.parametrize("student", [False, True])
def test_short_wide_prefix_full_batch_cost_counts_asset_contribution_state(student):
    # A=32 is real per-combination accumulator state: 31*(3*7+32)=1643
    # exceeds 5*7*32=1120. Ignoring A would incorrectly admit this shape.
    spec = raw_factor_spec(5, 7, 32, 32, student)
    inputs = small_raw_tape(steps=5, paths=7, factors=32, innovations=4,
                            assets=32, combinations=32, student=student)
    originals = {name: value.copy() for name, value in inputs.items()}
    graph = compile_compact(spec)
    canonical = compile_compact(public_factor_spec(spec))
    starts, ends = np.array([0], np.int64), np.array([5], np.int64)
    parameters = raw_parameters(spec)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        actual = execute(scheduler, graph, inputs, starts, ends, parameters)
        reference = execute(scheduler, canonical, inputs, starts, ends, parameters)
    compare_selected(actual, reference)
    compare_named(actual, expected_raw_named(spec, inputs))
    assert_canonical(actual)
    assert chunk_sum(actual, "compact_path_raw_rows") == 0
    assert chunk_sum(actual, "compact_path_raw_scale_elements") == 0
    assert chunk_sum(actual, "compact_path_raw_pack_bytes") == 0
    for name, original in originals.items():
        ieee_bits(inputs[name], original)
        assert not inputs[name].flags.writeable


@pytest.mark.parametrize("student", [False, True])
@pytest.mark.parametrize("selection", ["fan", "mixed", "complete_first"])
def test_selected_raw_consumers_keep_only_requested_roots_and_prove_actual_prefix_use(student, selection):
    full = raw_factor_spec(9, 7, 9, 2, student)
    if selection == "complete_first":
        names = [name for name in full["expressions"] if name.startswith(("fan_0", "samples_0", "stat_0_", "contribution_0_"))]
    else:
        names = ["fan_1"] if selection == "fan" else ["fan_1", "stat_1_es_95", "contribution_1_1"]
    spec = selected_spec(full, names)
    inputs = small_raw_tape(student=student)
    graph = compile_compact(spec)
    expected = expected_raw_named(full, inputs)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        actual = execute(scheduler, graph, inputs, np.array([0], np.int64), np.array([9], np.int64),
                         raw_parameters(spec))
    compare_named(actual, {name: expected[name] for name in names})
    if selection == "complete_first":
        assert_raw_private(actual, steps=9, paths=7, groups=1, student=student)
    else:
        assert_canonical(actual)
        assert chunk_sum(actual, "compact_path_raw_rows") == 0


@pytest.mark.parametrize("chi,code", [(0., "DIVIDE_BY_ZERO"), (-1., "DOMAIN_ERROR")])
@pytest.mark.parametrize("policy", ["raise", "isolate"])
def test_student_zero_and_negative_chi_follow_cme_canonical_errors(chi, code, policy):
    # These are CME sqrt/divide tests, not a claim of FIRP's IEEE chi policy.
    spec = raw_factor_spec(9, 7, 9, 2, True)
    inputs = small_raw_tape(student=True)
    changed = inputs["chi3"].copy()
    changed[3, 2, 0] = chi
    changed.flags.writeable = False
    inputs["chi3"] = changed
    graph = compile_compact(spec, error_policy=policy)
    canonical = compile_compact(public_factor_spec(spec), error_policy=policy)
    starts, ends = np.array([0], np.int64), np.array([9], np.int64)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        if policy == "raise":
            with pytest.raises((ValueError, RuntimeError), match=code) as expected:
                execute(scheduler, canonical, inputs, starts, ends, raw_parameters(spec))
            with pytest.raises(type(expected.value)) as actual:
                execute(scheduler, graph, inputs, starts, ends, raw_parameters(spec))
            assert str(actual.value) == str(expected.value)
        else:
            expected = execute(scheduler, canonical, inputs, starts, ends, raw_parameters(spec))
            actual = execute(scheduler, graph, inputs, starts, ends, raw_parameters(spec))
            compare_selected(actual, expected)
            assert_canonical(actual)
            assert all(out.root_statuses[0] != 0 for out in actual.outputs)


@pytest.mark.parametrize("chi_case", ["min_subnormal", "nan", "positive_inf", "negative_inf"])
@pytest.mark.parametrize("policy", ["raise", "isolate"])
def test_student_extreme_chi_matches_public_canonical_errors_and_ieee_results(chi_case, policy):
    # A strictly positive chi can still make chi/df underflow to zero. NaN and
    # infinities follow CME's public operators; FIRP's abnormal-input semantics
    # are deliberately not certified by this comparison.
    chi = {
        "min_subnormal": float.fromhex("0x0.0000000000001p-1022"),
        "nan": float("nan"),
        "positive_inf": float("inf"),
        "negative_inf": -float("inf"),
    }[chi_case]
    required_error = {"min_subnormal": "DIVIDE_BY_ZERO", "negative_inf": "DOMAIN_ERROR"}.get(chi_case)
    spec = raw_factor_spec(9, 7, 9, 2, True)
    inputs = small_raw_tape(student=True)
    changed = inputs["chi3"].copy()
    changed[3, 2, 0] = chi
    changed.flags.writeable = False
    inputs["chi3"] = changed
    original = changed.copy()
    graph = compile_compact(spec, error_policy=policy)
    canonical = compile_compact(public_factor_spec(spec), error_policy=policy)
    starts, ends = np.array([0], np.int64), np.array([9], np.int64)
    parameters = raw_parameters(spec)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        try:
            reference = execute(scheduler, canonical, inputs, starts, ends, parameters)
        except (ValueError, RuntimeError) as expected:
            assert policy == "raise"
            if required_error is not None:
                assert required_error in str(expected)
            with pytest.raises(type(expected)) as actual:
                execute(scheduler, graph, inputs, starts, ends, parameters)
            assert str(actual.value) == str(expected)
        else:
            if policy == "raise":
                assert required_error is None
            actual = execute(scheduler, graph, inputs, starts, ends, parameters)
            compare_selected(actual, reference)
            assert_canonical(reference)
            if policy == "isolate":
                assert_canonical(actual)
                if required_error is not None:
                    assert all(out.root_statuses[0] != 0 for out in actual.outputs)
            else:
                assert_raw_private(actual, steps=9, paths=7, groups=2, student=True)
    ieee_bits(inputs["chi3"], original)
    assert not inputs["chi3"].flags.writeable


@pytest.mark.parametrize("df,code", [(0., "DIVIDE_BY_ZERO"), (1., "DOMAIN_ERROR"), (2., None)])
def test_student_scalar_configuration_keeps_canonical_math_without_business_df_gate(df, code):
    spec = raw_factor_spec(9, 7, 9, 2, True)
    inputs = small_raw_tape(student=True)
    graph, canonical = compile_compact(spec), compile_compact(public_factor_spec(spec))
    starts, ends = np.array([0], np.int64), np.array([9], np.int64)
    parameters = raw_parameters(spec, df=df)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        if code is None:
            expected = execute(scheduler, canonical, inputs, starts, ends, parameters)
            actual = execute(scheduler, graph, inputs, starts, ends, parameters)
            compare_selected(actual, expected)
            assert_raw_private(actual, steps=9, paths=7, groups=2, student=True)
        else:
            with pytest.raises((ValueError, RuntimeError), match=code) as expected:
                execute(scheduler, canonical, inputs, starts, ends, parameters)
            with pytest.raises(type(expected.value)) as actual:
                execute(scheduler, graph, inputs, starts, ends, parameters)
            assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("student", [False, True])
def test_raw_prefix_exact_budget_rejects_one_byte_less_and_preserves_two_results(student):
    spec = raw_factor_spec(33, 31, 17, 2, student)
    inputs = small_raw_tape(steps=33, paths=31, assets=17, student=student)
    graph = compile_compact(spec)
    canonical = compile_compact(public_factor_spec(spec))
    starts, ends = np.array([0], np.int64), np.array([33], np.int64)
    parameters = raw_parameters(spec)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        plan = scheduler.plan(graph, inputs, starts, ends)
        assert "bounded_compact_scan_consumer_closure" in plan.reason_codes
        with pytest.raises(MemoryError):
            execute(scheduler, graph, inputs, starts, ends, parameters,
                    memory_budget_bytes=plan.estimated_total_memory_bytes - 1)
        first = execute(scheduler, graph, inputs, starts, ends, parameters,
                        memory_budget_bytes=plan.estimated_total_memory_bytes)
        saved = saved_outputs(first)
        later = execute(scheduler, graph, inputs, starts, ends, parameters,
                        memory_budget_bytes=plan.estimated_total_memory_bytes)
        reference = execute(scheduler, canonical, inputs, starts, ends, parameters)
        for result in (first, later):
            compare_selected(result, reference)
            compare_named(result, expected_raw_named(spec, inputs))
            assert_raw_private(result, steps=33, paths=31, groups=2, student=student)
        check_ownership(first, later, saved, inputs)
