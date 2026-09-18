# My C Tools

Precompiled C++17 numerical kernels for Python financial analytics. One native
extension, no runtime JIT compilation, and no mandatory AVX instruction set.

**Version 0.2.0 is prepared in this repository; building a wheel does not publish
it to PyPI.** See [publishing](docs/publishing.md) before creating a public release.
The repository currently does not specify a redistribution license; the owner
must confirm the license before public distribution.

## Install

After the corresponding version has been published:

```bash
python -m pip install --only-binary=:all: "my-ctools==0.2.0"
```

`--only-binary` fails clearly when no compatible wheel exists, rather than trying
to compile on the application server. The distribution name is `my_ctools`
(`my-ctools` is the normalized pip name); the import name remains `my_ctools`.

To install the current checkout before publication, with a C++17 compiler:

```bash
python -m pip install .
```

Build dependencies are isolated automatically. Binary wheel users need Python
and NumPy, not CMake, a C++ compiler, Numba or a startup warmup step.

## Platform matrix

The configured CI builds and tests these targets. A target is release-ready only
when its CI lane passes; configuration alone is not evidence of a successful run.
Local verification results are in [the verification report](docs/verification-2026-09-18.md).

| Operating system | CPU architecture | Binary baseline |
| --- | --- | --- |
| Linux glibc | x86_64, aarch64 | manylinux2014 / glibc 2.17+ |
| Linux musl | x86_64, aarch64 | musllinux 1.2 |
| macOS 11+ | Intel x86_64, Apple Silicon arm64 | Separate architecture wheels |
| Windows | AMD64 | 64-bit MSVC build |

CPython **3.10–3.14**, ordinary GIL-enabled builds; NumPy **1.26–2.x** subject to
its Python-version compatibility. Python 3.8/3.9, PyPy, free-threaded Python,
32-bit CPUs, Windows ARM64 and GPU acceleration are not in this release's matrix.
Other platforms may compile from source but are not claimed as tested targets.

Wheels are specific to OS, CPU and Python ABI: there is not one binary that runs
on every machine. The build never adds `-march=native`, `-mavx*` or `/arch:AVX*`.
The `_simd` suffix on a legacy function is retained for API compatibility, not as
a guarantee of a hand-written SIMD implementation.

## Use

```python
import numpy as np
import my_ctools as mc

returns = np.array([[0.01, 0.02], [-0.03, 0.01], [0.02, -0.01]], dtype=np.float64)
dates = np.array(["2026-01-01", "2026-01-02", "2026-01-03"], dtype="datetime64[ns]")

std = mc.cal_std_mean(returns)  # (2,), sample std, not a mean/std tuple
mean_std = mc.cal_std_mean_simd(returns)  # (2, 2): means first, sample stds second
drawdown, drawdown_dates, recovery = mc.cal_max_dd(returns, dates)
print(mc.build_info())

# Explicit per-call parallelism; 0 selects automatically, default is 1.
std_parallel = mc.cal_std_mean(returns, n_threads=2)

# Historical submodule imports remain valid.
from my_ctools.cal_max_dd import cal_max_dd
```

| Function | Return contract |
| --- | --- |
| `cal_std_mean(input)` | Column sample standard deviations, `(N,)`, ddof=1 |
| `cal_std_mean_simd(input)` | Means and sample standard deviations, `(2, N)` |
| `cal_cpr(f_type, funds_value)` | Per-column persistence ratios, `(N,)` |
| `cal_max_dd(funds_val, day_arr)` | `(drawdown, YYYYMMDD_dates, recovery_periods)` |
| `cal_longest_dd_recover(funds_val)` | Longest drawdown-recovery periods, `(N,)` |
| `cal_all_largest_indicators(array_value, dates, i_code="positive")` | Dict with `r`, `p`, `s`, `l` |
| `cal_all_longest_indicators(a_value, dates, i_code="positive")` | `(return, starts, ends, periods)` |
| `cal_rolling_gain_loss(i_code, funds_val, start_idx, end_idx, day_arr)` | Nine arrays: mean, median, win rate, three gain buckets, three loss buckets |

All eight functions also accept keyword-only `n_threads=1`; valid values are
0–256. Default single-thread execution avoids multiplying native threads across
web-server workers. `n_threads=0` is bounded by detected hardware and work size.
The GIL is released only while the numerical kernel runs. There is no global
thread pool; independent calls can execute concurrently.

### Array and memory contract

Values are 2-D `(observations, columns)`. Already aligned, native-endian,
C-contiguous `float64` arrays are borrowed without copying. Other real numeric
arrays are normalized once; slices, Fortran-order arrays, `float32` or unaligned
buffers may therefore need a copy. Read-only inputs are supported. **Do not
modify an input concurrently with a running calculation.** Outputs own their
NumPy memory and remain valid after inputs are deleted.

Group identifiers are `int32`; dates and indices are `int64`. Integer narrowing
is range-checked. Dates may be integer nanoseconds or exact `datetime64[ns]`
arrays; NaT and other datetime units are rejected. Rolling dates must be sorted,
and start/end indices are inclusive. `(-1, -1)` denotes an inactive column.
Private `_core` functions reject incompatible arrays instead of converting them;
the `_core` interface is not a stable public API.

Empty columns return empty outputs. For zero observations, statistics, CPR and
rolling outputs are NaN; longest periods are 0; maximum drawdown returns NaN,
empty dates and the historical not-recovered sentinel `1000000`. For existing
nonempty data, legacy NaN, tie-breaking and return layouts are retained.

### Deliberately retained legacy behavior

This is an architecture release, not a financial-methodology rewrite. In
particular, both `positive` and `negative` streak modes select positive input
segments in the old algorithm; the negative mode changes accumulation. The
longest-streak implementation also retains its last-row denominator when the
selected segment starts at row zero. Maximum drawdown retains its original
backward-fill semantics. Rolling statistics retain the original `tail + 1`
exclusion, with its unsafe negative array index corrected.

The historical variance formula uses sums and squared sums; it may suffer
cancellation for large-offset or almost-constant data. This release does not
silently replace that formula. See [the design](docs/architecture-upgrade-2026-09-18.md)
for the precise compatibility boundary and regression methodology.

## Develop and test

Python 3.12 is a convenient development interpreter. Use an isolated environment:

```bash
python -m venv .venv
# Activate .venv using the command appropriate to your shell.
python -m pip install ".[dev]"
python -m build
python tools/check_dist.py
python -m twine check --strict dist/*
python -I -m pytest tests -q --import-mode=importlib
ruff check src tests tools
```

`python -m build` first builds an sdist, then reconstructs the wheel from it.
Install that wheel in a clean environment to test the actual artifact rather
than only the checkout. If local pip configuration is unrelated to this project,
`python -m build --installer uv` is supported with uv installed; do not commit
local indexes, proxy settings or credentials into the repository.

For standalone C++ tests, without Python headers or pybind11:

```bash
cmake -S . -B .build-native -DMY_CTOOLS_BUILD_PYTHON=OFF -DMY_CTOOLS_BUILD_TESTS=ON
cmake --build .build-native --config Release
ctest --test-dir .build-native -C Release --output-on-failure
```

GCC/Clang builds can add `-DMY_CTOOLS_SANITIZE=ON` for address and undefined-behavior
sanitizers. Golden Python fixtures come from the actual historical Git C++
implementations, not from the rewritten kernels. Regeneration is an explicit
developer operation: `python tools/legacy_reference.py --write`.

## Architecture

```text
src/my_ctools/          Public API, normalization, legacy import paths
        ↓
cpp/bindings.cpp        Array checks, NumPy allocation, GIL and bindings
        ↓
cpp/kernels/           Pure C++ algorithms
        ↓
cpp/include/my_ctools/ Shared array view, calendar and bounded parallelism
```

The package produces exactly one `my_ctools._core` extension. New kernels should
reuse the same validation, calendar and threading facilities, and ship with
numerical references and boundary tests. Do not introduce Numba, Python imports
inside kernels, unconditional architecture-specific instructions, or financial
business orchestration from downstream applications.

Build and release instructions: [publishing](docs/publishing.md).
