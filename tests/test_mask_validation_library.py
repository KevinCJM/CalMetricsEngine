"""Mask-format and failure contracts through direct and complete native entries."""

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler
from calmetrics_engine import operators as op

from .test_matmul_library import settings


@pytest.mark.parametrize("n", [0, 1, 15, 16, 17, 127, 128, 129, 255, 256, 257, 4097, 65537])
@pytest.mark.parametrize("pattern", ["zero", "one", "mixed"])
def test_mask_format_counts_and_selected_order_statistics(n, pattern):
    mask = np.zeros(n, np.uint8) if pattern == "zero" else np.ones(n, np.uint8)
    if pattern == "mixed":
        mask[1::3] = 0
    data = (np.arange(n) % 17 - 8).astype(np.float64)
    before = mask.copy()
    mask.flags.writeable = data.flags.writeable = False
    selected = data[mask == 1]
    assert op.count_true(mask) == len(selected)
    if len(selected):
        assert op.sum_where(data, mask) == selected.sum()
        assert op.median_where(data, mask) == np.median(selected)
        assert op.quantile_where(data, mask, 0.5) == np.quantile(selected, 0.5)
    else:
        for function, args in [
            (op.sum_where, (data, mask)),
            (op.median_where, (data, mask)),
            (op.quantile_where, (data, mask, 0.5)),
        ]:
            with pytest.raises(ValueError, match="^INSUFFICIENT_SAMPLE$"):
                function(*args)
    np.testing.assert_array_equal(mask, before)


@pytest.mark.parametrize("operator", ["count_true", "sum_where", "quantile_where"])
@pytest.mark.parametrize("bad", [2, 3, 127, 128, 255])
@pytest.mark.parametrize("position", [0, 15, 16, 127, 128, 256])
def test_invalid_mask_bytes_keep_the_existing_error(operator, bad, position):
    mask = np.ones(257, np.uint8)
    mask[position] = bad
    before = mask.copy()
    mask.flags.writeable = False
    args = (mask,) if operator == "count_true" else (np.ones(257), mask)
    if operator == "quantile_where":
        args += (0.5,)
    with pytest.raises(ValueError, match="^INVALID_MASK$"):
        op.get(operator)(*args)
    np.testing.assert_array_equal(mask, before)


@pytest.mark.parametrize("geometry", ["strided", "reversed", "f_matrix", "f_tensor", "broadcast"])
@pytest.mark.parametrize("bad", [1, 2, 255])
def test_mask_views_borrow_only_their_logical_bytes(geometry, bad):
    if geometry == "strided":
        owner = np.full(520, 255, np.uint8)
        mask = owner[3:517:2]
        mask[:] = 1
        mask[128] = bad
    elif geometry == "reversed":
        owner = np.ones(257, np.uint8)
        owner[128] = bad
        mask = owner[::-1]
    elif geometry == "broadcast":
        owner = np.array([bad], np.uint8)
        mask = np.broadcast_to(owner, (257,))
    else:
        shape = (17, 19) if geometry == "f_matrix" else (7, 9, 11)
        owner = np.ones(shape, np.uint8, order="F")
        owner.flat[128] = bad
        mask = owner
    before = owner.copy()
    owner.flags.writeable = mask.flags.writeable = False
    if bad == 1:
        np.testing.assert_array_equal(op.logical_not(mask), np.zeros(mask.shape, bool))
    else:
        with pytest.raises(ValueError, match="^INVALID_MASK$"):
            op.logical_not(mask)
    np.testing.assert_array_equal(owner, before)


@pytest.mark.parametrize(
    "expression",
    [
        "mean_where(log(x),m)",
        "filter_apply(mean_where(log(x),m),finite_mask(x))",
        "bisect(solve_x-mean_where(log(x),m),0,10,1e-12,100)",
    ],
)
@pytest.mark.parametrize("source_failed", [False, True])
@pytest.mark.parametrize("lane", ["single", "thread", "process_inline", "process_shared"])
def test_large_mask_structure_survives_failed_sources_and_scopes(expression, source_failed, lane):
    x = np.full(257, -1.0 if source_failed else 1.0)
    mask = np.ones(257, np.uint8)
    mask[256] = 2
    x.flags.writeable = mask.flags.writeable = False
    graph = GraphCompiler({"x": "series", "m": {"kind": "series", "dtype": "bool"}}).compile(
        expression, error_policy="isolate"
    )
    starts = np.zeros(2, np.int64)
    ends = np.full(2, len(x), np.int64)
    with AdaptiveScheduler(cpu_budget=2, config=settings(lane)) as scheduler:
        error = RuntimeError if lane.startswith("process") else ValueError
        with pytest.raises(error, match="INVALID_MASK"):
            scheduler.execute(graph, {"x": x, "m": mask}, starts, ends)


def test_same_mask_address_does_not_cache_validity():
    mask = np.ones(257, np.uint8)
    assert op.count_true(mask) == 257
    mask[256] = 2
    with pytest.raises(ValueError, match="^INVALID_MASK$"):
        op.count_true(mask)
    mask[256] = 0
    assert op.count_true(mask) == 256
