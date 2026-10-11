"""Independent-axis graph math, error order, transport and output ownership."""

import math

import numpy as np

from calmetrics_engine import AdaptiveScheduler, GraphCompiler, PlannerConfig
from calmetrics_engine import operators as op

TYPES = dict(
    w=dict(kind="matrix", axes=["asset", "factor"], shape=["N", "K"]),
    r=dict(kind="matrix", axes=["time", "factor"], shape=["T", "K"]),
    p=dict(kind="vector", axes=["component"], shape=["1"]),
)
Y = "matmul(w,transpose(r))"
QT = f"quantiles_axis({Y},p,1)"
# Reduce a declared singleton probability dimension before broadcasting on N.
THRESHOLD = f"transpose(broadcast_like(sum_axis({QT},1),transpose({Y})))"
TAIL = f"less_equal({Y},{THRESHOLD})"
TAIL_COUNT = f"sum_axis(where({TAIL},1,0),1)"
CVAR = f"-divide(sum_axis(where({TAIL},{Y},0),1),maximum({TAIL_COUNT},1))"
NEGATIVE = f"less_than({Y},0)"
NEGATIVE_COUNT = f"sum_axis(where({NEGATIVE},1,0),1)"
NEGATIVE_MEAN = f"divide(sum_axis(where({NEGATIVE},{Y},0),1),maximum({NEGATIVE_COUNT},1))"
CENTER = f"transpose(broadcast_like({NEGATIVE_MEAN},transpose({Y})))"
DELTA = f"subtract({Y},{CENTER})"
SQUARED = f"sum_axis(where({NEGATIVE},multiply({DELTA},{DELTA}),0),1)"
DOWNSIDE = f"where(greater_than({NEGATIVE_COUNT},1),sqrt(divide({SQUARED},maximum(subtract({NEGATIVE_COUNT},1),1))),0)"
NAV = f"cumulative_product_axis(add(1,{Y}),1,1)"
PEAK = f"cumulative_max_axis({NAV},1,1)"
DRAWDOWN = f"max_axis(subtract(1,divide({NAV},{PEAK})),1)"
ROOTS = [CVAR, DOWNSIDE, DRAWDOWN]


def independent(data, actual):
    errors = []
    for row in range(min(37, data["w"].shape[0])):
        # Separate scalar definition, including inclusive ties and ordered state.
        values = [
            sum(
                float(data["w"][row, j]) * float(data["r"][i, j]) for j in range(data["w"].shape[1])
            )
            for i in range(data["r"].shape[0])
        ]
        ordered = sorted(values)
        position = (len(values) - 1) * float(data["p"][0])
        lo = math.floor(position)
        hi = math.ceil(position)
        fraction = position - lo
        threshold = (
            ordered[lo] if lo == hi else ordered[lo] * (1 - fraction) + ordered[hi] * fraction
        )
        tail = [v for v in values if v <= threshold]
        negative = [v for v in values if v < 0]
        center = sum(negative) / len(negative) if negative else 0.0
        deviation = (
            math.sqrt(sum((v - center) ** 2 for v in negative) / (len(negative) - 1))
            if len(negative) > 1
            else 0.0
        )
        nav = peak = 1.0
        maximum = 0.0
        for v in values:
            nav *= 1 + v
            peak = max(peak, nav)
            maximum = max(maximum, 1 - nav / peak)
        wants = [-sum(tail) / len(tail) if tail else 0.0, deviation, maximum]
        errors.append([abs(float(a[row]) - want) for a, want in zip(actual, wants, strict=True)])
    maximum = np.max(errors, axis=0).tolist()
    assert max(maximum) < 4e-5, maximum
    assert all(np.isfinite(a).all() for a in actual)
    assert np.all(actual[1] >= 0) and np.all(actual[2] >= 0)
    return maximum


def config(lane):
    return PlannerConfig(
        thread_work_units=1e100 if lane == "single" else 1,
        process_work_units=1 if lane.startswith("process") else 1e100,
        shared_memory_threshold_bytes=1 if lane == "process_shared" else 1 << 30,
        min_rows_per_worker=1,
        max_processes=2,
    )


def graph(roots, layout="C"):
    return GraphCompiler(TYPES).compile(roots, result_format="typed", output_order=layout)


def run(g, data, lane="single", starts=(0,), ends=(257,)):
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        return scheduler.execute(g, data, np.asarray(starts, np.int64), np.asarray(ends, np.int64))


def arena(result):
    return max(
        c["numeric_arena_bytes"] + c["mask_arena_bytes"] for c in result.audit["native_chunks"]
    )


def test_closed_graph_partition_math_errors_transports_and_ownership():
    records = []
    for layout in ["C", "F"]:
        for mode in ["normal", "positive", "negative", "ties"]:
            for probability in [0.0, 0.05, 1.0]:
                rng = np.random.default_rng(7)
                w = rng.uniform(0.1, 1.0, (193, 7))
                w /= w.sum(axis=1, keepdims=True)
                r = rng.normal(scale=0.01, size=(257, 7))
                if mode == "positive":
                    r = np.abs(r) + 0.001
                if mode == "negative":
                    r = -np.abs(r) - 0.001
                if mode == "ties":
                    r = np.repeat(np.array([-0.02, 0.0, 0.02])[np.arange(257) % 3, None], 7, axis=1)
                data = dict(
                    w=np.array(w, order=layout),
                    r=np.array(r, order=layout),
                    p=np.array([probability]),
                )
                saved = {name: a.copy() for name, a in data.items()}
                for a in data.values():
                    a.flags.writeable = False
                roots = ROOTS + [TAIL_COUNT, NEGATIVE_COUNT]
                g = graph(roots, layout)
                control = graph(roots + [Y], layout)
                actual = run(g, data)
                canonical = run(control, data)
                for a, b in zip(actual.outputs, canonical.outputs[:5], strict=True):
                    np.testing.assert_allclose(a.values[0], b.values[0], atol=4e-5, rtol=1e-12)
                    assert not a.values[0].flags.writeable and a.values[0].shape == (193,)
                    assert a.root_statuses[0] == 0
                for a, b in zip(actual.outputs[3:], canonical.outputs[3:5], strict=True):
                    np.testing.assert_array_equal(a.values[0], b.values[0])
                maximum = independent(data, tuple(a.values[0] for a in actual.outputs[:3]))
                snapshots = [a.values[0].copy() for a in actual.outputs]
                current = run(g, data)
                for a, b, snapshot in zip(actual.outputs, current.outputs, snapshots, strict=True):
                    np.testing.assert_array_equal(a.values[0], snapshot)
                    assert not np.shares_memory(a.values[0], b.values[0])
                    assert all(not np.shares_memory(b.values[0], x) for x in data.values())
                for name in data:
                    np.testing.assert_array_equal(data[name], saved[name])
                assert arena(actual) < arena(canonical) / 2, (arena(actual), arena(canonical))
                records.append(
                    dict(
                        layout=layout,
                        mode=mode,
                        probability=probability,
                        maximum_absolute_errors=maximum,
                        bounded_arena=arena(actual),
                        canonical_arena=arena(canonical),
                    )
                )

    # A later batch fails in an earlier node than the first batch's DOMAIN_ERROR.
    w = np.full((193, 1), 0.1)
    w[130, 0] = 1.0
    data = dict(w=w, r=np.ones((257, 1)), p=np.array([0.05]))
    for roots in [
        [f"mean_axis(divide(1,subtract({Y},1)),1)", f"mean_axis(sqrt(subtract({Y},2)),1)"],
        [f"mean_axis(sqrt(subtract({Y},2)),1)", f"mean_axis(divide(1,subtract({Y},1)),1)"],
    ]:
        errors = []
        for expressions in [roots, roots + [Y]]:
            try:
                run(graph(expressions), data)
            except ValueError as error:
                errors.append(str(error))
            else:
                raise AssertionError("expected numerical failure")
        assert errors[0] == errors[1], errors
        records.append(dict(error_priority=errors[0], later_batch=130))

    for lane in ["single", "thread", "process_inline", "process_shared"]:
        rng = np.random.default_rng(12)
        data = dict(
            w=rng.uniform(0.01, 0.1, (193, 7)),
            r=rng.normal(scale=0.01, size=(257, 7)),
            p=np.array([0.05]),
        )
        for a in data.values():
            a.flags.writeable = False
        actual = run(graph(ROOTS), data, lane, starts=(0, 11), ends=(257, 257))
        canonical = run(graph(ROOTS + [Y]), data, lane, starts=(0, 11), ends=(257, 257))
        assert actual.plan.lane == ("process" if lane.startswith("process") else lane)
        for a, b in zip(actual.outputs, canonical.outputs[:3], strict=True):
            for row in range(2):
                np.testing.assert_allclose(a.values[row], b.values[row], atol=4e-5, rtol=1e-12)
        records.append(dict(lane=lane, two_intervals=True, bounded_arena=arena(actual)))
    assert len(records) == 30


def test_shorter_interval_and_wide_consumer_remain_within_planner_budget():
    rng = np.random.default_rng(9)
    w = rng.uniform(0.1, 1.0, (128, 7))
    w /= w.sum(axis=1, keepdims=True)
    data = dict(w=w, r=rng.normal(scale=1e-6, size=(81921, 7)), p=np.array([0.05]))
    for a in data.values():
        a.flags.writeable = False
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(
            graph(ROOTS), data, np.zeros(2, np.int64), np.array([81921, 81920], np.int64)
        )
        observed = max(
            c["numeric_arena_bytes"]
            + c["mask_arena_bytes"]
            + c["operator_workspace_capacity_bytes"]
            for c in result.audit["native_chunks"]
        )
        assert result.plan.metadata()["estimated_worker_scratch_bytes"] >= observed
        assert [c["max_window"] for c in result.audit["native_chunks"]] == [81921]
    data = dict(w=w, r=data["r"][:257], p=data["p"], q=np.linspace(0.0, 1.0, 8193))
    for a in data.values():
        a.flags.writeable = False
    types = dict(TYPES, q=dict(kind="vector", axes=["component"], shape=["Q"]))
    roots = ROOTS + [f"mean_axis(quantiles_axis({Y},q,1),1)"]
    compiler = GraphCompiler(types)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(
            compiler.compile(roots, result_format="typed"),
            data,
            np.array([0], np.int64),
            np.array([257], np.int64),
        )
        control = scheduler.execute(
            compiler.compile(roots + [Y], result_format="typed"),
            data,
            np.array([0], np.int64),
            np.array([257], np.int64),
        )
        for a, b in zip(result.outputs, control.outputs[:4], strict=True):
            np.testing.assert_array_equal(a.values[0], b.values[0])
        chunk = result.audit["native_chunks"][0]
        assert chunk["numeric_arena_bytes"] < 4_000_000
        assert result.plan.metadata()["estimated_worker_scratch_bytes"] >= (
            chunk["numeric_arena_bytes"]
            + chunk["mask_arena_bytes"]
            + chunk["operator_workspace_capacity_bytes"]
        )


def test_ordered_skinny_library_preserves_scalar_bits_and_ieee_classes():
    rng = np.random.default_rng(9123)
    for layout in ["C", "F"]:
        for m in [4, 7, 64]:
            for k in [1, 7, 30, 32, 128]:
                for n in [128, 129, 257]:
                    a = np.array(rng.normal(size=(m, k)), order=layout)
                    b = np.array(rng.normal(size=(k, n)), order=layout)
                    a.flags.writeable = b.flags.writeable = False
                    expected = op.matmul(a, b, simd="scalar")
                    actual, audit = op.matmul(a, b, audit=True)
                    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
                    assert audit["input_copy_bytes"] == 0
                    if audit["isa"] == "eigen-packed-matmul":
                        assert (
                            audit["workspace_bytes"]
                            == audit["workspace_capacity_bytes"]
                            == (320 * k + 72) * 8
                        )
                        prefix = m // 4 * 4
                        bulk = n // 6 * 6
                        assert (
                            audit["algorithm_copy_bytes"]
                            == (prefix + ((prefix + 127) // 128) * bulk) * k * 8
                        )
                    else:
                        assert (
                            audit["algorithm_copy_bytes"] == audit["workspace_capacity_bytes"] == 0
                        )
        a = np.zeros((7, 7))
        b = np.ones((7, 129), order=layout)
        a[0, :3] = [1e308, 1e308, -1e308]
        a[1, :2] = [np.finfo(float).max, -np.finfo(float).max]
        a[2, 0] = np.nextafter(0.0, 1.0)
        a[3, 0] = -np.nextafter(0.0, 1.0)
        a[4, 0] = np.inf
        a[5, 0] = -np.inf
        a[6, 0] = np.nan
        expected = op.matmul(a, b, simd="scalar")
        actual = op.matmul(a, b)
        for function in [np.isnan, np.isposinf, np.isneginf, np.signbit]:
            np.testing.assert_array_equal(function(actual), function(expected))
        finite = np.isfinite(expected)
        np.testing.assert_array_equal(
            actual[finite].view(np.uint64), expected[finite].view(np.uint64)
        )


def test_span_arithmetic_masks_selection_and_strided_broadcast():
    rng = np.random.default_rng(917)
    base = rng.normal(size=(7, 257))
    base.flat[0:5] = [np.nan, np.inf, -np.inf, 0.0, -0.0]
    for layout in ["C", "F", "negative", "padded", "broadcast"]:
        x = np.array(base, order="F" if layout == "F" else "C")
        y = np.array(rng.uniform(0.1, 1.0, size=x.shape), order="F" if layout == "F" else "C")
        if layout == "negative":
            x = x[::-1, ::-1]
            y = y[::-1, ::-1]
        if layout == "padded":
            owner = np.zeros((7, 514))
            owner[:, ::2] = x
            x = owner[:, ::2]
            other = np.zeros((7, 514))
            other[:, ::2] = y
            y = other[:, ::2]
        if layout == "broadcast":
            y = np.broadcast_to(y[:, 0:1], x.shape)
        for a in [x, y]:
            a.flags.writeable = False
        with np.errstate(all="ignore"):
            cases = [
                ("add", x + y),
                ("subtract", x - y),
                ("multiply", x * y),
                ("divide", x / y),
                ("minimum", np.where(y < x, y, x)),
                ("maximum", np.where(x < y, y, x)),
                ("negate", -x),
                ("absolute", np.abs(x)),
                ("equal", x == y),
                ("not_equal", x != y),
                ("less_than", x < y),
                ("less_equal", x <= y),
                ("greater_than", x > y),
                ("greater_equal", x >= y),
                ("finite_mask", np.isfinite(x)),
            ]
        for name, expected in cases:
            unary = name in ["negate", "absolute", "finite_mask"]
            actual = getattr(op, name)(x) if unary else getattr(op, name)(x, y)
            np.testing.assert_array_equal(actual, expected)
            if actual.dtype == np.float64:
                np.testing.assert_array_equal(
                    np.signbit(actual[actual == 0]), np.signbit(expected[expected == 0])
                )
        mask = np.array(np.isfinite(x), order="F" if layout == "F" else "C")
        np.testing.assert_array_equal(op.where(mask, x, y), np.where(mask, x, y))
        np.testing.assert_array_equal(op.where(mask, x, 0.0), np.where(mask, x, 0.0))


def test_strided_domain_failure_retains_ascending_output_prefix():
    import pytest

    # Avoid the pre-existing contiguous SIMD prevalidation route; the span
    # must retain the canonical partial write order for this raw out contract.
    backing = np.ones((7, 514))
    backing[:, ::2] = 2.0
    x = backing[:, ::2]
    denominator = np.ones_like(backing)
    denominator[:, ::2] = 1.0
    y = denominator[:, ::2]
    y.flat[17] = 0.0
    target = np.full(x.shape, -777.0)
    with pytest.raises(ValueError, match="DIVIDE_BY_ZERO"):
        op.divide(x, y, out=target)
    np.testing.assert_array_equal(target.flat[:17], 2.0)
    np.testing.assert_array_equal(target.flat[17:], -777.0)
