"""Independent metric references; production formulas execute entirely in C++.

These are explicit graphs exercising missing mathematical capabilities, not a
second production MetricsFactory implementation or a dependency on its checkout.
"""

import math

import numpy as np
import pytest

from calmetrics_engine import AdaptiveScheduler, GraphCompiler


def run(expressions, inputs, *, bindings=None, types=None, starts=None, ends=None, error_policy="raise"):
    options = {"error_policy": error_policy}
    if bindings is not None:
        options["root_bindings"] = bindings
    graph = GraphCompiler(types or {name: "series" for name in inputs}).compile(expressions, **options)
    n = len(next(iter(inputs.values())))
    with AdaptiveScheduler(cpu_budget=1) as scheduler:
        result = scheduler.execute(
            graph,
            inputs,
            np.asarray([0] if starts is None else starts, dtype=np.int64),
            np.asarray([n] if ends is None else ends, dtype=np.int64),
        )
    return result


def reference_ema(x, span):
    state = np.nan
    out = []
    alpha = 2 / (span + 1)
    for value in x:
        if np.isfinite(value):
            state = value if np.isnan(state) else alpha * value + (1 - alpha) * state
        out.append(state)
    return np.asarray(out)


@pytest.mark.parametrize("span", [2, 4, 5, 12])
def test_ema_even_odd_spans_nan_hold_and_interval_reset(span):
    x = np.array([np.nan, 2.0, 5.0, np.nan, 8.0, 3.0, np.nan, 7.0, 4.0])
    x.flags.writeable = False
    expr = f"recursive_filter(x,2/({span}+1),0,finite_mask(x),2,0)"
    result = run([expr], {"x": x}, starts=[0, 5], ends=[5, 9])
    expected = np.concatenate([reference_ema(x[:5], span), reference_ema(x[5:], span)])
    np.testing.assert_allclose(result.values[:, 0], expected, equal_nan=True)
    np.testing.assert_array_equal(result.offsets, [0, 5, 9])


def test_kdj_initial_row_and_shared_rsv_update_mask():
    rsv = np.array([80.0, 60.0, np.nan, 20.0, 75.0, np.nan])
    expected_k, expected_d = [], []
    k = d = 50.0
    for index, value in enumerate(rsv):
        if index and np.isfinite(value):
            k = value / 3 + 2 * k / 3
            d = k / 3 + 2 * d / 3
        expected_k.append(k)
        expected_d.append(d)
    k_expr = "recursive_filter(rsv,1/3,50,finite_mask(rsv),1,0)"
    d_expr = f"recursive_filter({k_expr},1/3,50,finite_mask(rsv),1,0)"
    result = run([k_expr, d_expr, f"3*({k_expr})-2*({d_expr})"], {"rsv": rsv})
    expected = np.column_stack(
        [expected_k, expected_d, 3 * np.asarray(expected_k) - 2 * np.asarray(expected_d)]
    )
    np.testing.assert_allclose(result.values, expected, rtol=1e-13)


def test_trix_three_ema_stages_and_derived_series_shift():
    x = np.array([10.0, 11.0, 9.0, 12.0, 13.0, 12.0, 15.0])
    bindings = {"e1": "recursive_filter(x,0.4,0,finite_mask(x),2,0)"}
    bindings["e2"] = "recursive_filter(e1,0.4,0,finite_mask(e1),2,0)"
    bindings["e3"] = "recursive_filter(e2,0.4,0,finite_mask(e2),2,0)"
    result = run(["100*(e3/aligned_shift(e3)-1)"], {"x": x}, bindings=[bindings])
    e3 = reference_ema(reference_ema(reference_ema(x, 4), 4), 4)
    expected = np.concatenate([[np.nan], 100 * (e3[1:] / e3[:-1] - 1)])
    np.testing.assert_allclose(result.values[:, 0], expected, equal_nan=True, atol=1e-13)


def test_adxr_combines_derived_rolling_series_without_resetting_history():
    dx = np.array([15.0, 30.0, 21.0, 33.0, 18.0, 12.0, 36.0, 42.0])
    result = run(
        ["(adx+aligned_shift(adx,2))/2"],
        {"dx": dx},
        bindings=[{"adx": "rolling_mean(dx,3)"}],
    )
    adx = np.array([np.nan, np.nan] + [dx[i - 2 : i + 1].mean() for i in range(2, len(dx))])
    expected = (adx + np.concatenate([[np.nan, np.nan], adx[:-2]])) / 2
    np.testing.assert_allclose(result.values[:, 0], expected, equal_nan=True)


def reference_hurst(x, rs):
    x = x[~np.isnan(x)]
    if len(x) < 64:
        return np.nan
    scales, values = [], []
    for m in (1, 2, 4, 8, 16, 32):
        width = len(x) // m
        if rs and width < 8:
            continue
        segments = []
        for j in range(m):
            part = x[j * width : (j + 1) * width]
            sd = np.std(part)
            if rs:
                if not np.isfinite(sd) or sd <= 0:
                    continue
                path = np.concatenate([[0.0], np.cumsum(part - np.mean(part))])
                value = np.ptp(path) / sd
                if not np.isfinite(value) or value <= 0:
                    continue
            else:
                value = np.ptp(part - np.mean(part)) / (sd if sd else 0.00001)
            segments.append(value)
        if segments:
            scales.append(width)
            values.append(np.mean(segments))
    if (rs and len(scales) < 4) or any(value <= 0 for value in values):
        return np.nan
    lx, ly = np.log(scales), np.log(values)
    return np.dot(lx - lx.mean(), ly - ly.mean()) / np.sum((lx - lx.mean()) ** 2)


def hurst_expression(rs):
    # Missing scalar is an explicit empty-set result, never divide-by-zero.
    nan = "filter_apply(sum(r),not_equal(r,r))"
    sd = "std(r,0)"
    safe_sd = f"where({sd}>0,{sd},0.00001)"
    if rs:
        path = "cumulative_sum(r-mean(r))"
        spread = f"maximum(max_value({path}),0)-minimum(min_value({path}),0)"
        segment = f"where({sd}>0,({spread})/({safe_sd}),{nan})"
    else:
        segment = f"(max_value(r)-min_value(r))/({safe_sd})"
    terms = []
    for m in (1, 2, 4, 8, 16, 32):
        width = f"floor(length(r)/{m})"
        blocks = f"block_apply({segment},{width})"
        if rs:
            mean_rs = f"sum(where(finite_mask({blocks}),{blocks},0))/maximum(count_true(finite_mask({blocks})),1)"
            valid = f"logical_and({width}>=8,({mean_rs})>0)"
        else:
            mean_rs = f"mean({blocks})"
            valid = f"({mean_rs})>0"
        # Explicitly divide a count by one observation before taking its log.
        unit_count = "(length(r)-length(r)+1)"
        terms.append((f"log(({width})/{unit_count})", f"log(maximum(({mean_rs}),1e-300))", valid))
    count = "+".join(f"where({v},1,0)" for _, _, v in terms)
    sx = "+".join(f"where({v},{x},0)" for x, _, v in terms)
    sy = "+".join(f"where({v},{y},0)" for _, y, v in terms)
    sxy = "+".join(f"where({v},({x})*({y}),0)" for x, y, v in terms)
    sxx = "+".join(f"where({v},({x})**2,0)" for x, _, v in terms)
    safe_count = f"maximum(({count}),1)"
    numerator = f"({sxy})-({sx})*({sy})/{safe_count}"
    denominator = f"({sxx})-({sx})**2/{safe_count}"
    value = f"({numerator})/maximum(({denominator}),1e-30)"
    minimum_scales = 4 if rs else 6
    value = f"where(({count})>={minimum_scales},{value},{nan})"
    valid = "logical_not(not_equal(r,r))"
    return f"filter_apply({value},where({valid},count_true({valid}),0)>=64)"


@pytest.mark.parametrize("rs", [False, True])
@pytest.mark.parametrize("size", [40, 64, 257])
@pytest.mark.parametrize("error_policy", ["raise", "isolate"])
def test_hurst_multiscale_compaction_and_tail_semantics(rs, size, error_policy):
    rng = np.random.default_rng(217)
    r = rng.normal(0.0001, 0.01, size=size)
    r = np.insert(r, [3, 7, 21], np.nan)
    result = run([hurst_expression(rs)], {"r": r}, error_policy=error_policy)
    actual = result.values[0, 0]
    np.testing.assert_allclose(actual, reference_hurst(r, rs), rtol=1e-10, atol=1e-10)
    if error_policy == "isolate":
        assert result.statuses[0, 0] == (4 if size < 64 else 0)


@pytest.mark.parametrize("rs", [False, True])
def test_hurst_degenerate_samples_are_explicit_missing_values(rs):
    for values in [np.ones(260), np.full(260, np.nan), np.array([], dtype=np.float64)]:
        actual = run([hurst_expression(rs)], {"r": values}).values[0, 0]
        assert np.isnan(actual)


def cpr_body(name):
    prev = f"aligned_shift({name})"
    valid = f"finite_mask({prev})"
    counts = {}
    for label, yesterday, today in [("ww", ">=0", ">=0"), ("ll", "<0", "<0"), ("wl", ">=0", "<0"), ("lw", "<0", ">=0")]:
        counts[label] = f"count_true(logical_and({valid},logical_and({prev}{yesterday},{name}{today})))"
    numerator = f"({counts['ww']})*({counts['ll']})"
    denominator = f"({counts['wl']})*({counts['lw']})"
    nan = f"filter_apply(sum({name}),not_equal({name},{name}))"
    return f"where(({denominator})>0,({numerator})/maximum(({denominator}),1),{nan})"


@pytest.mark.parametrize("width", [1, 5, 10])
def test_cross_product_ratio_original_axis_blocks_before_dropna(width):
    rng = np.random.default_rng(23)
    r = rng.normal(size=237)
    r[10:20] = np.nan
    r[57] = np.nan
    block = "filter_apply(sum(r),logical_not(not_equal(r,r)))"
    bindings = {"b": "r" if width == 1 else f"block_apply({block},{width})"}
    formula = f"filter_apply({cpr_body('b')},finite_mask(b))"
    actual = run([formula], {"r": r}, bindings=[bindings]).values[0, 0]
    if width == 1:
        b = r
    else:
        parts = r[: len(r) // width * width].reshape(-1, width)
        b = np.nansum(parts, axis=1)
        b[np.all(np.isnan(parts), axis=1)] = np.nan
    b = b[~np.isnan(b)]
    previous, current = b[:-1] >= 0, b[1:] >= 0
    ww, ll = np.sum(previous & current), np.sum(~previous & ~current)
    wl, lw = np.sum(previous & ~current), np.sum(~previous & current)
    expected = ww * ll / (wl * lw) if wl * lw else np.nan
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def typed_ids():
    return {"kind": "series", "dtype": "int64", "axes": ["time"], "shape": ["T"], "semantic_dimension": "category"}


def test_group_hhi_and_exact_large_holder_ids():
    amount = np.array([2.0, 3.0, 7.0, 5.0, 3.0, 2.0])
    group = np.array([1, 2, 1, 2, 2, 1], dtype=np.int64)
    ids = np.array([2**60, 2**60 + 1, 2**60 + 1, 2**60 + 1, 2**60 + 2, 2**60], dtype=np.int64)
    inputs = {"amount": amount, "group": group, "ids": ids}
    formulas = ["group_apply(sum((amount/sum(amount))**2),group)", "group_apply(distinct_count(ids,finite_mask(amount)),group)"]
    result = run(formulas, inputs, types={"amount": "series", "group": typed_ids(), "ids": typed_ids()})
    expected = np.empty((len(group), 2))
    for key in np.unique(group):
        mask = group == key
        expected[mask, 0] = np.sum((amount[mask] / np.sum(amount[mask])) ** 2)
        expected[mask, 1] = len(np.unique(ids[mask]))
    np.testing.assert_allclose(result.values, expected)


def bs_call(s, k, r, t, sigma):
    def cdf(x):
        return 0.5 * math.erfc(-x / math.sqrt(2))
    d1 = (math.log(s / k) + (r + sigma**2 / 2) * t) / (sigma * math.sqrt(t))
    return s * cdf(d1) - k * math.exp(-r * t) * cdf(d1 - sigma * math.sqrt(t))


def test_black_scholes_visible_formula_to_native_iv_solver():
    s = np.array([100.0, 105.0, 100.0, 88.0])
    k = np.array([100.0, 99.0, 110.0, 90.0])
    t = np.array([1.0, 0.25, 2.0, 0.7])
    sigma = np.array([0.2, 0.45, 0.31, 0.18])
    rates = np.full(4, 0.03)
    prices = np.array([bs_call(*args) for args in zip(s, k, rates, t, sigma, strict=True)])
    d1 = "((log(first(s)/first(k))+(first(r)+solve_x**2/2)*first(t))/(solve_x*sqrt(first(t))))"
    residual = f"first(s)*normal_cdf({d1})-first(k)*exp(-first(r)*first(t))*normal_cdf({d1}-solve_x*sqrt(first(t)))-first(price)"
    expression = f"group_apply(bisect({residual},0.0001,5,1e-11,100),contract)"
    inputs = {"s": s, "k": k, "r": rates, "t": t, "price": prices, "contract": np.arange(4, dtype=np.int64)}
    types = {name: "series" for name in inputs}
    types["contract"] = typed_ids()
    result = run([expression], inputs, types=types)
    np.testing.assert_allclose(result.values[:, 0], sigma, rtol=0, atol=1e-10)
