"""Nested concatenation moves whole IEEE payloads across offset F rectangles."""
import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler

from .test_typed_results import config


@pytest.mark.parametrize("rows,columns", [(31, 33), (32, 32), (33, 65), (65, 33)])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_offset_f_rectangles_preserve_all_bits_and_generations(rows, columns, order, lane):
    patterns = np.array([0, 0x8000000000000000, 0x7ff0000000000000,
                         0xfff0000000000000, 0x7ff0000000000001,
                         0xfff8000000000042, 1, 0x3ff0000000000000], dtype=np.uint64)
    inputs = {}
    for index, name in enumerate(("x", "y", "z", "w")):
        base = np.empty((rows + 7, columns + 11), dtype=np.float64, order="F")
        base.view(np.uint64)[:] = patterns[(np.arange(base.size).reshape(base.shape) + index) % len(patterns)]
        value = base[1:rows + 1, 2:columns + 2]
        value.flags.writeable = False
        inputs[name] = value
    saved = {name: value.view(np.uint64).copy() for name, value in inputs.items()}
    expected = np.concatenate((np.concatenate((inputs["x"], inputs["y"]), axis=1),
                               np.concatenate((inputs["z"], inputs["w"]), axis=1)), axis=0)
    graph = GraphCompiler({name: dict(kind="matrix", axes=["asset", "asset"], shape=["N", "K"])
                           for name in inputs}).compile(
        "concatenate(concatenate(x,y,1),concatenate(z,w,1),0)",
        result_format="typed", output_order=order)
    bounds = np.zeros(2, dtype=np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=config(lane)) as scheduler:
        first = scheduler.execute(graph, inputs, bounds, bounds)
        second = scheduler.execute(graph, inputs, bounds, bounds)
    retained = [output.values[row] for result in (first, second)
                for output in result.outputs for row in range(2)]
    for index, actual in enumerate(retained):
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        assert actual.flags.c_contiguous if order == "C" else actual.flags.f_contiguous
        assert not actual.flags.writeable
        assert all(not np.shares_memory(actual, v) for v in inputs.values())
        assert all(not np.shares_memory(actual, v) for v in retained[index + 1:])
    for name, value in inputs.items():
        np.testing.assert_array_equal(value.view(np.uint64), saved[name])
