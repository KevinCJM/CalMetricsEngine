"""Exact order statistics, exceptional ties and native result ownership."""

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, get_ident

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, PlannerConfig
from calmetrics_engine import operators as op


def assert_bits(actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def finite_values(size):
    # Repeated, exactly representable values and a single zero sign.
    return (((np.arange(size) * 37) % 257 - 128) / 16.0)[::-1].copy()


def sorted_value(values, probability=None):
    ordered = np.sort(np.asarray(values).reshape(-1))
    if probability is None:
        middle = len(ordered) // 2
        return (ordered[middle] if len(ordered) % 2 else
                (ordered[middle - 1] + ordered[middle]) * 0.5)
    position = float(len(ordered) - 1) * probability
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    fraction = position - float(lower)
    # Keep the canonical expression, including its zero times infinity behavior.
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cached_sort_values(values, probability):
    # Two consumers deliberately exercise the unchanged native full-sort cache.
    # NumPy does not promise the same NaN/zero tie order as C++ std::sort.
    values = np.asarray(values).reshape(-1).copy()
    values.setflags(write=False)
    probability = float(probability)
    graph = GraphCompiler({"x": "series"}).compile(
        ["median(x)", f"quantile(x,{probability!r})"], result_format="typed")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"x": values},
                                   np.array([0], np.int64), np.array([len(values)], np.int64))
    assert sum(chunk["order_stat_sorts"] for chunk in result.audit["native_chunks"]) == 1
    return tuple(field.values[0] for field in result.outputs)


@pytest.fixture(scope="module")
def eligible_auto_isa():
    # CI can require ON/OFF explicitly; local runs still check every result.
    expected = os.environ.get("CME_EXPECT_HIGHWAY")
    assert expected in (None, "0", "1")
    actual, audit = op.call("median", finite_values(1024), audit=True)
    assert_bits(actual, sorted_value(finite_values(1024)))
    assert audit["isa"] in ("highway_vqselect", "scalar")
    if expected is not None:
        assert audit["isa"] == ("highway_vqselect" if expected == "1" else "scalar")
    return audit["isa"]


@pytest.mark.parametrize("size", [1, 2, 31, 1023, 1024, 1025, 4096, 4097])
@pytest.mark.parametrize("masked", [False, True])
def test_selection_threshold_odd_even_and_exact_interpolation(size, masked, eligible_auto_isa):
    chosen = finite_values(size)
    values = np.full(2 * size, np.nan) if masked else chosen.copy()
    mask = np.zeros(values.shape, bool)
    if masked:
        values[::2] = chosen
        mask[::2] = True
    values.setflags(write=False)
    mask.setflags(write=False)
    before = values.copy()
    workspace = op.Workspace()
    median = "median_where" if masked else "median"
    quantile = "quantile_where" if masked else "quantile"
    arguments = (values, mask) if masked else (values,)
    for simd in ("scalar", "auto"):
        assert_bits(op.call(median, *arguments, workspace=workspace, simd=simd),
                    sorted_value(chosen))
        for probability in (np.nextafter(0., 1.), .25, .5, .95, np.nextafter(1., 0.)):
            with np.errstate(over="ignore", invalid="ignore"):
                expected = sorted_value(chosen, probability)
            owner = np.full(3, 912.)
            actual, audit = op.call(quantile, *arguments, probability,
                                   out=owner[1:2].reshape(()), workspace=workspace,
                                   simd=simd, audit=True)
            assert_bits(actual, expected)
            assert owner[0] == owner[-1] == 912.
            assert audit["algorithm_copy_bytes"] == size * 8
            assert audit["input_copy_bytes"] == 0
            assert audit["isa"] == (eligible_auto_isa if simd == "auto" and size >= 1024
                                    else "scalar")
    assert_bits(values, before)


@pytest.mark.parametrize("pattern", ["sorted", "reverse", "equal", "negative_zero",
                                      "positive_infinity", "both_infinities", "overflow"])
def test_selection_repeated_values_infinities_and_original_arithmetic(pattern):
    values = finite_values(4097)
    if pattern == "sorted":
        values.sort()
    elif pattern == "reverse":
        values.sort()
        values = values[::-1]
    elif pattern == "equal":
        values.fill(3.25)
    elif pattern == "negative_zero":
        values.fill(-0.)
    elif pattern == "positive_infinity":
        values.fill(np.inf)
    elif pattern == "both_infinities":
        values[::2], values[1::2] = -np.inf, np.inf
    elif pattern == "overflow":
        values.fill(np.finfo(float).max)
    values.setflags(write=False)
    with np.errstate(over="ignore", invalid="ignore"):
        assert_bits(op.median(values), sorted_value(values))
        for probability in (.25, .5, .75, .95, np.nextafter(1., 0.)):
            expected = (cached_sort_values(values, probability)[1] if np.isinf(values).any()
                        else sorted_value(values, probability))
            assert_bits(op.quantile(values, probability), expected)
        # Even median must still overflow on (maximum + maximum) * 0.5.
        expected = (cached_sort_values(values[:-1], .5)[0] if np.isinf(values).any()
                    else sorted_value(values[:-1]))
        assert_bits(op.median(values[:-1]), expected)


@pytest.mark.parametrize("size", [1023, 1024, 1025, 4096])
@pytest.mark.parametrize("masked", [False, True])
def test_neighbor_ranks_at_half_and_last_elements(size, masked):
    # Unique ranks ensure a wrong neighboring item cannot pass through ties.
    chosen = np.random.default_rng(31).permutation(size).astype(np.float64) / 16.
    values = np.full(3 * size, np.nan) if masked else chosen.copy()
    mask = np.zeros(values.shape, bool)
    if masked:
        values[::3], mask[::3] = chosen, True
    values.setflags(write=False)
    mask.setflags(write=False)
    before = values.copy()
    arguments = (values, mask) if masked else (values,)
    name = "quantile_where" if masked else "quantile"
    # Fractional ranks cross the midpoint and leave tails of only one/two items.
    for rank in (size // 2 - .25, size // 2 + .25, size - 2.25, size - 1.25,
                 size - 2.):
        probability = rank / (size - 1)
        owner = np.full(3, 912.)
        result, audit = op.call(name, *arguments, probability,
                                out=owner[1:2].reshape(()), audit=True)
        assert_bits(result, sorted_value(chosen, probability))
        assert audit["algorithm_copy_bytes"] == chosen.nbytes
        assert audit["input_copy_bytes"] == 0
        assert owner[0] == owner[-1] == 912.
    assert_bits(values, before)


@pytest.mark.parametrize("pattern", ["mixed_zero", "nan_payload", "nan_and_infinity"])
@pytest.mark.parametrize("masked", [False, True])
def test_exceptional_ties_keep_the_native_full_sort_result(pattern, masked):
    chosen = finite_values(4097)
    if pattern == "mixed_zero":
        chosen[::2], chosen[1::2] = -0., 0.
    else:
        nan = np.array([0x7FF8000000000001, 0xFFF8000000000013], np.uint64).view(np.float64)
        chosen[::11] = nan[0]
        chosen[::13] = nan[1]
        if pattern == "nan_and_infinity":
            chosen[1:3] = [-np.inf, np.inf]
    values = np.full(2 * len(chosen), 777.) if masked else chosen.copy()
    mask = np.zeros(values.shape, bool)
    if masked:
        values[::2], mask[::2] = chosen, True
    values.setflags(write=False)
    mask.setflags(write=False)
    arguments = (values, mask) if masked else (values,)
    median = "median_where" if masked else "median"
    quantile = "quantile_where" if masked else "quantile"
    # At an integral rank both arithmetic operands are the same payload.
    # With two distinct NaNs, IEEE addition need not select the same payload
    # across the cache and direct functions, even in the unchanged old wheel.
    for probability in (.5, (len(chosen) - 2) / (len(chosen) - 1)):
        expected_median, expected_quantile = cached_sort_values(chosen, probability)
        assert_bits(op.call(median, *arguments), expected_median)
        assert_bits(op.call(quantile, *arguments, probability), expected_quantile)


@pytest.mark.parametrize("layout", ["F", "padded", "negative", "tensor", "broadcast"])
@pytest.mark.parametrize("masked", [False, True])
def test_logical_flattening_strides_and_read_only_input(layout, masked):
    dense = finite_values(4096).reshape(64, 64)
    if layout == "F":
        values = np.array(dense, order="F")
    elif layout == "padded":
        owner = np.full((128, 128), np.nan)
        owner[::2, ::2] = dense
        values = owner[::2, ::2]
    elif layout == "negative":
        values = dense[::-1, ::-1]
    elif layout == "tensor":
        values = np.asfortranarray(dense.reshape(16, 16, 16))[::-1, :, ::-1]
    else:
        values = np.broadcast_to(np.array([-0.]), (64, 64))
    mask = (np.arange(values.size).reshape(values.shape) % 3 != 0)[:, ::-1].copy()
    values.setflags(write=False)
    mask.setflags(write=False)
    before = values.copy()
    selected = values.reshape(-1)[mask.reshape(-1)] if masked else values.reshape(-1)
    arguments = (values, mask) if masked else (values,)
    assert_bits(op.call("median_where" if masked else "median", *arguments),
                sorted_value(selected))
    assert_bits(op.call("quantile_where" if masked else "quantile", *arguments, .95),
                sorted_value(selected, .95))
    assert_bits(values, before)


def test_unconsumed_nan_and_opposite_zero_do_not_change_selected_results():
    values = np.tile([np.nan, -0., 0., 4.], 2048)
    mask = np.tile([False, False, True, True], 2048)
    values.setflags(write=False)
    mask.setflags(write=False)
    assert_bits(op.median_where(values, mask), 2.)
    assert_bits(op.quantile_where(values, mask, .25), 0.)


@pytest.mark.parametrize("probability", [0., 1., -.1, 1.1, np.nan, np.inf])
@pytest.mark.parametrize("masked", [False, True])
def test_invalid_probability_is_rejected_before_output_write(probability, masked):
    values = finite_values(4097)
    mask = np.ones(values.shape, bool)
    arguments = (values, mask) if masked else (values,)
    owner = np.full(3, 912.)
    with pytest.raises(ValueError, match="INVALID_PARAMETER"):
        op.call("quantile_where" if masked else "quantile", *arguments, probability,
                out=owner[1:2].reshape(()))
    np.testing.assert_array_equal(owner, 912.)


@pytest.mark.parametrize("name", ["median", "quantile", "median_where", "quantile_where"])
def test_empty_selection_keeps_error_and_output(name):
    masked = name.endswith("_where")
    values = finite_values(4097) if masked else np.empty(0)
    arguments = (values, np.zeros(values.shape, bool)) if masked else (values,)
    if name.startswith("quantile"):
        arguments += (.5,)
    owner = np.full(3, 912.)
    with pytest.raises(ValueError, match="INSUFFICIENT_SAMPLE"):
        op.call(name, *arguments, out=owner[1:2].reshape(()))
    np.testing.assert_array_equal(owner, 912.)


def lane_config(lane):
    return PlannerConfig(thread_work_units=1e100 if lane == "single" else 1,
                         process_work_units=1 if lane.startswith("process") else 1e100,
                         shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
                         min_rows_per_worker=1, max_processes=2)


@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
@pytest.mark.parametrize("roots", ["single", "shared_sort", "masked"])
def test_single_multiple_roots_and_process_result_ownership(lane, roots):
    values = finite_values(8194)
    mask = np.arange(len(values)) % 3 != 0
    values.setflags(write=False)
    mask.setflags(write=False)
    declarations, inputs = {"x": "series"}, {"x": values}
    if roots == "single":
        formulas = ["quantile(x,.95)"]
    elif roots == "shared_sort":
        formulas = ["median(x)", "quantile(x,.95)"]
    else:
        declarations["m"] = dict(kind="series", dtype="bool", semantic_dimension="mask")
        inputs["m"] = mask
        formulas = ["median_where(x,m)", "quantile_where(x,m,.95)"]
    graph = GraphCompiler(declarations).compile(formulas, result_format="typed")
    starts, ends = np.array([0, 4097], np.int64), np.array([4097, 8194], np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=lane_config(lane)) as scheduler:
        first = scheduler.execute(graph, inputs, starts, ends)
        second = scheduler.execute(graph, inputs, starts, ends)
    assert first.plan.lane == ("process" if lane.startswith("process") else lane)
    for row, (start, end) in enumerate(zip(starts, ends, strict=True)):
        selected = values[start:end]
        if roots == "masked":
            selected = selected[mask[start:end]]
        expected = ([sorted_value(selected, .95)] if roots == "single" else
                    [sorted_value(selected), sorted_value(selected, .95)])
        for field, other, wanted in zip(first.outputs, second.outputs, expected, strict=True):
            assert_bits(field.values[row], wanted)
            assert not field.values[row].flags.writeable
            assert not np.shares_memory(field.values[row], values)
            assert not np.shares_memory(field.values[row], other.values[row])


def test_prepared_rereads_selection_admission_and_keeps_snapshots_independent():
    values = finite_values(4097)
    values.setflags(write=False)
    graph = GraphCompiler({"x": "series"}).compile("quantile(x,.5)", result_format="typed")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        prepared = scheduler.prepare_execution(graph, {"x": values},
                                               np.array([0], np.int64), np.array([len(values)], np.int64))
        saved = prepared.run_snapshot()
        preserved = saved.outputs[0].values[0].copy()
        values.setflags(write=True)
        values.view(np.uint64)[:] = 0x7FF8000000000013
        values.setflags(write=False)
        nan_fallback = prepared.run_snapshot()
        _, expected_nan = cached_sort_values(values, .5)
        assert_bits(nan_fallback.outputs[0].values[0], expected_nan)
        values.setflags(write=True)
        values[::2], values[1::2] = -0., 0.
        values.setflags(write=False)
        fallback = prepared.run_snapshot()
        _, expected = cached_sort_values(values, .5)
        assert_bits(fallback.outputs[0].values[0], expected)
        values.setflags(write=True)
        values[:] = finite_values(len(values)) + 5.
        values.setflags(write=False)
        borrowed = prepared.run()
        assert_bits(borrowed.outputs[0].values[0], sorted_value(values, .5))
        assert_bits(saved.outputs[0].values[0], preserved)
        assert_bits(nan_fallback.outputs[0].values[0], expected_nan)
        assert not np.shares_memory(nan_fallback.outputs[0].values[0], borrowed.outputs[0].values[0])
        assert not np.shares_memory(saved.outputs[0].values[0], borrowed.outputs[0].values[0])
        assert not np.shares_memory(fallback.outputs[0].values[0], borrowed.outputs[0].values[0])


@pytest.mark.parametrize("mixed_zero", [False, True])
def test_order_stat_used_by_discrete_consumer_keeps_exact_decision(mixed_zero):
    values = np.tile([-1., 0., 1.], 2048)
    if mixed_zero:
        values[1::6] = -0.
    values.setflags(write=False)
    _, expected = cached_sort_values(values, .5)
    graph = GraphCompiler({"x": "series"}).compile("quantile(x,.5)>0", result_format="typed")
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(graph, {"x": values},
                                   np.array([0], np.int64), np.array([len(values)], np.int64))
    np.testing.assert_array_equal(result.outputs[0].values[0], expected > 0)


def test_order_stat_output_alias_is_rejected_before_input_change():
    values = finite_values(4097)
    before = values.copy()
    with pytest.raises(ValueError, match="OUTPUT_ALIASES_INPUT"):
        op.quantile(values, .95, out=values[:1].reshape(()))
    assert_bits(values, before)


@pytest.mark.parametrize("size", [1024, 1025])
@pytest.mark.parametrize("pattern", ["wide_exponents", "tiny", "infinities"])
@pytest.mark.parametrize("masked", [False, True])
def test_highway_exact_extreme_values_and_explicit_isa(size, pattern, masked, eligible_auto_isa):
    if pattern == "wide_exponents":
        powers = np.resize(np.arange(-1022, 1024, 17), size)
        chosen = np.ldexp(np.ones(size), powers)
        chosen[1::2] *= -1.
        chosen[:2] = [-np.finfo(float).max, np.finfo(float).max]
    elif pattern == "tiny":
        chosen = np.resize(np.array([1, 2, 3, 0x000FFFFFFFFFFFFF,
                                     0x0010000000000000, 0x8000000000000001,
                                     0x8000000000000002, 0x800FFFFFFFFFFFFF],
                                    np.uint64), size).view(np.float64)
    else:
        chosen = finite_values(size)
        chosen[::7], chosen[::13] = -np.inf, np.inf
    # Ineligible values outside the mask must not disable selected VQSelect.
    values = np.full(size * 2, np.nan) if masked else chosen.copy()
    mask = np.zeros(values.shape, bool)
    if masked:
        values[::2], mask[::2] = chosen, True
    values.setflags(write=False)
    mask.setflags(write=False)
    before = values.copy()
    arguments = (values, mask) if masked else (values,)
    names = ("median_where", "quantile_where") if masked else ("median", "quantile")
    workspace = op.Workspace()
    explicit = [isa for isa in op.available_simd() if isa not in ("auto", "scalar")]
    for probability in (.25, .5, .95, np.nextafter(1., 0.)):
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            # Native cached full sort also fixes NaN results from Inf arithmetic.
            expected = (cached_sort_values(chosen, probability) if pattern == "infinities"
                        else (sorted_value(chosen), sorted_value(chosen, probability)))
        for name, wanted in zip(names, expected, strict=True):
            extra = () if name.startswith("median") else (probability,)
            for isa in ("auto", "scalar", *explicit):
                owner = np.full(3, 912.)
                actual, audit = op.call(name, *arguments, *extra,
                                       out=owner[1:2].reshape(()), workspace=workspace,
                                       simd=isa, audit=True)
                assert_bits(actual, wanted)
                assert audit["isa"] == (eligible_auto_isa if isa == "auto" else "scalar")
                assert audit["algorithm_copy_bytes"] == chosen.nbytes
                assert audit["input_copy_bytes"] == 0
                assert owner[0] == owner[-1] == 912.
    assert_bits(values, before)


@pytest.mark.parametrize("masked", [False, True])
def test_highway_actual_threads_keep_workspaces_and_results_independent(masked, eligible_auto_isa):
    workers = 4
    barrier = Barrier(workers)
    workspaces = [op.Workspace() for _ in range(workers)]

    def run(index):
        chosen = finite_values(65537) + index
        values = np.full(chosen.size * 2, np.nan) if masked else chosen.copy()
        mask = np.zeros(values.shape, bool)
        if masked:
            values[::2], mask[::2] = chosen, True
        values.setflags(write=False)
        mask.setflags(write=False)
        before = values.copy()
        arguments = (values, mask) if masked else (values,)
        name = "quantile_where" if masked else "quantile"
        barrier.wait(timeout=30)
        first, first_audit = op.call(name, *arguments, .25, out=np.empty(()),
                                    workspace=workspaces[index], audit=True)
        saved = first.copy()
        barrier.wait(timeout=30)
        second, second_audit = op.call(name, *arguments, .95, out=np.empty(()),
                                      workspace=workspaces[index], audit=True)
        assert_bits(first, sorted_value(chosen, .25))
        assert_bits(second, sorted_value(chosen, .95))
        assert_bits(first, saved)
        assert_bits(values, before)
        assert not np.shares_memory(first, second)
        assert not np.shares_memory(first, values)
        assert not np.shares_memory(second, values)
        for audit in (first_audit, second_audit):
            assert audit["isa"] == eligible_auto_isa
            assert audit["algorithm_copy_bytes"] == chosen.nbytes
        return get_ident(), first, second

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, range(workers)))
    assert len({identity for identity, _, _ in results}) == workers
    for i, (_, first, second) in enumerate(results):
        for _, other_first, other_second in results[i + 1:]:
            assert not any(np.shares_memory(a, b) for a in (first, second)
                           for b in (other_first, other_second))
