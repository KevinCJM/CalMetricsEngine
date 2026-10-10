"""Root admission boundaries and unchanged C/F output/error/ownership contracts."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

TYPES = {
    "a": dict(kind="matrix", axes=["asset", "asset"], shape=["M", "K"]),
    "b": dict(kind="matrix", axes=["asset", "asset"], shape=["K", "N"]),
}


def layout_copy(value, layout):
    if layout in ["CC", "FF"]:
        return np.array(value, order=layout[0])
    if layout == "paddedC":
        result = np.full((value.shape[0], value.shape[1] + 3), np.nan)
        result[:, : value.shape[1]] = value
        return result[:, : value.shape[1]]
    result = np.full((value.shape[0] + 3, value.shape[1]), np.nan, order="F")
    result[: value.shape[0]] = value
    return result[: value.shape[0]]


def execute(graph, a, b, scheduler):
    return scheduler.execute(graph, {"a": a, "b": b}, np.zeros(1, np.int64), np.zeros(1, np.int64))


def calls(result):
    return sum(chunk["accelerate_matmul_calls"] for chunk in result.audit["native_chunks"])


@pytest.mark.parametrize("layout", ["CC", "FF", "paddedC", "paddedF"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize(
    "pattern",
    [
        "random",
        "consistent",
        "consistent_high",
        "sparse",
        "late_zero",
        "late_negative_zero",
        "late_nan",
        "late_inf",
        "late_tiny",
        "late_huge",
    ],
)
def test_root_bounds_full_payload_and_layout(layout, order, pattern):
    rng = np.random.default_rng(20261005)
    a = rng.uniform(-1, 1, (512, 256))
    b = rng.uniform(-1, 1, (256, 512))
    if pattern.startswith("consistent"):
        a = np.abs(a) + 0.125
        b = np.abs(b) + 0.125
        a[1::2] *= -1
        b[:, 1::2] *= -1
        if pattern == "consistent_high":
            a *= 1024
            b *= 1024
    if pattern == "sparse":
        a[:, 1::2] = 0
        b[1::2] = 0
    faults = {
        "late_zero": 0.0,
        "late_negative_zero": -0.0,
        "late_nan": np.nan,
        "late_inf": np.inf,
        "late_tiny": 2.0**-129,
        "late_huge": 2.0**129,
    }
    if pattern in faults:
        a[-1, -1] = faults[pattern]
    a = layout_copy(a, layout)
    b = layout_copy(b, layout)
    before = [v.copy() for v in [a, b]]
    for v in [a, b]:
        v.flags.writeable = False
    graph = GraphCompiler(TYPES).compile("matmul(a,b)", output_order=order)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        first = execute(graph, a, b, scheduler)
        saved = first.outputs[0].values[0].copy()
        second = execute(graph, a, b, scheduler)
        actual = second.outputs[0].values[0]
        np.testing.assert_array_equal(first.outputs[0].values[0], saved)
        assert not actual.flags.writeable and not np.shares_memory(
            actual, first.outputs[0].values[0]
        )
        assert not np.shares_memory(actual, a) and not np.shares_memory(actual, b)
        assert actual.flags.c_contiguous if order == "C" else actual.flags.f_contiguous
        for row in [0, 256, 511]:
            for column in [0, 256, 511]:
                expected = 0.0
                for k in range(256):
                    expected += float(a[row, k]) * float(b[k, column])
                if np.isnan(expected):
                    assert np.isnan(actual[row, column])
                elif np.isinf(expected):
                    assert actual[row, column] == expected
                elif pattern in ["late_tiny", "late_huge", "late_negative_zero"]:
                    assert actual[row, column] == expected
                else:
                    assert abs(actual[row, column] - expected) < 0.000049
        if pattern in ["late_nan", "late_inf"]:
            # Existing MatMul returns IEEE classes. Only the last A row is faulty.
            assert np.isfinite(actual[:-1]).all()
            if pattern == "late_nan":
                assert np.isnan(actual[-1]).all()
            else:
                assert np.isinf(actual[-1]).all()
                np.testing.assert_array_equal(np.signbit(actual[-1]), np.signbit(b[-1]))
        if pattern in ["late_tiny", "late_huge", "late_negative_zero", "late_nan", "late_inf"]:
            assert calls(second) == 0
        assert (
            second.audit["python_operator_calls"] == 0
            and second.audit["request_time_compilation"] == 0
        )
    for v, old in zip([a, b], before, strict=True):
        np.testing.assert_array_equal(v, old)


@pytest.mark.parametrize("layout", ["CC", "FF"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_opposite_sign_after_probe_is_still_classified(layout, order):
    a = np.ones((512, 256))
    b = np.ones((256, 512))
    a[-1, -1] = -1.0
    a = layout_copy(a, layout)
    b = layout_copy(b, layout)
    graph = GraphCompiler(TYPES).compile("matmul(a,b)", output_order=order)
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = execute(graph, a, b, scheduler)
        expected = np.full((512, 512), 256.0)
        expected[-1] = 254.0
        np.testing.assert_array_equal(result.outputs[0].values[0], expected)
