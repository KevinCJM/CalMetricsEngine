# CalMetricsEngine

AOT calculation engine for quantitative finance and fund research.

CalMetricsEngine is being built as the reusable execution layer behind the fund
investment research platform:

```text
DSL / AST
    ↓
Typed DAG
    ↓
Operator Lowering
    ↓
Native Execution Plan
    ↓
PyBind11 / C++17
```

The current 0.3.0 code line establishes the native foundation: portable C++ kernels,
a strict zero-copy NumPy boundary, cross-platform wheels, and one native extension.
The production DSL / Typed DAG currently living in FundInvestmentResearchPlatform
will be extracted only after contract-equivalence tests are in place.

## Install

After publication:

```bash
python -m pip install --only-binary=:all: "calmetrics-engine==0.3.0"
```

From the current checkout:

```bash
python -m pip install .
```

Import:

```python
import calmetrics_engine as engine
```

## Why this engine

- No runtime JIT compilation or service-start warmup.
- PyBind11/C++ AOT backend.
- Exact-dtype, strided, zero-copy NumPy inputs.
- C-contiguous, Fortran-order, sliced and readonly arrays are supported directly.
- GIL released during native numerical work.
- Portable baseline wheels instead of mandatory AVX.
- One numerical implementation per reusable operator.
- Designed for future single-call Typed-DAG execution and workspace reuse.

## Zero-copy input contract

CalMetricsEngine does not silently normalize calculation inputs.

```python
import numpy as np
import calmetrics_engine as engine

values = np.arange(40, dtype=np.float64).reshape(10, 4)
view = values[:, ::2]                 # non-contiguous view, no copy

result = engine.cal_std_mean(view)    # C++ reads the original strides directly
assert np.shares_memory(view, values)
```

Current input rules:

- Values: NumPy `float64` ndarray.
- Group ids: NumPy `int32` ndarray.
- Dates / indices: NumPy `int64` ndarray.
- Dates may also be exact `datetime64[ns]`; they are viewed as int64 without copying.
- Arrays must be aligned.
- Readonly arrays are supported.
- Wrong dtype, Python lists, or incompatible objects fail instead of being copied.

Outputs and required scratch/workspace memory may be allocated. The contract is
**zero input copies and zero unnecessary intermediate copies**, not “no allocation”.

## High-performance data preparation

The generic native API can read strided NumPy views without copying. The future
high-throughput SIMD batch path has a stricter preferred layout: each product's
observations should be stored contiguously with unit stride.

Recommended product-major representation:

```text
values  = [ product_0 ][ product_1 ][ product_2 ] ... [ product_n ]
dates   = [ product_0 ][ product_1 ][ product_2 ] ... [ product_n ]
offsets = [0, p0_end, p1_end, ..., total_observations]
```

C++ locates one product by pointer arithmetic:

```text
begin = offsets[product]
end   = offsets[product + 1]

product_values = values + begin
length         = end - begin
```

Different intervals of the same product should be represented as
`product_id + start_offset + end_offset`, not materialized as new NumPy arrays.

For multiple fields, prefer Structure of Arrays (SoA):

```text
returns[total_observations]
close[total_observations]
volume[total_observations]
dates[total_observations]
offsets[product_count + 1]
```

The calling application should perform dtype normalization, sorting/alignment,
and any unavoidable data compaction once at the ingestion boundary. After data
enters CalMetricsEngine, hot-path code must not silently call
`astype`, `copy`, `np.ascontiguousarray`, advanced indexing, or equivalent
operations that materialize another input array.

### NaN and SIMD

NaN does not disable SIMD by itself, but per-element missing-value branches and
masks can reduce SIMD throughput. Missing-data semantics must never be changed
only for speed.

The intended execution lanes are:

1. **Dense SIMD lane** — metadata proves the block contains no NaN.
2. **Masked SIMD lane** — vector masks preserve the operator's NaN contract.
3. **Valid-span lane** — use contiguous valid spans when the operator semantics allow it.
4. **Scalar fallback** — irregular sparse missingness where SIMD is not beneficial.
5. **Reject** — operators whose contract forbids NaN fail explicitly.

Do not delete observations or fill NaN unless the operator's documented financial
semantics explicitly require that behavior.

## Parallel execution model

Heavy batch execution will use one scheduling authority rather than letting each
business module create independent pools.

The Scheduler will consider:

```text
products × intervals × DAG cost × observations
+ scenario count
+ input/workspace bytes
+ CPU budget
+ memory budget
+ hard-stop requirements
```

Preferred hierarchy:

- small job: single thread + SIMD
- medium CPU job: one process + native thread pool + SIMD
- large independent product batches: process pool + shared memory + per-process thread budget + SIMD
- hard-stop/fault-isolated jobs: separate process

When a process pool reads the same large input dataset, workers should attach to
shared memory or mmap and receive only descriptors/offsets. Large NumPy arrays
should not be pickled into every worker.

Process count and native thread count must share one CPU budget. For example,
`4 processes × 4 threads` may be valid on a 16-core machine; `8 × 16` is
oversubscription and is not acceptable by default.

Metric-level parallelism is not the first choice because many indicators share
upstream DAG work. Product blocks, interval blocks and scenario blocks are
usually better scheduling dimensions.

The detailed implementation rules for future changes are in [AGENTS.md](AGENTS.md).

## Current finance kernels

| Function | Result |
| --- | --- |
| `cal_std_mean` | Column sample standard deviation |
| `cal_std_mean_simd` | Column means and sample standard deviations |
| `cal_cpr` | Persistence ratio by peer group |
| `cal_max_dd` | Maximum drawdown, date and recovery period |
| `cal_longest_dd_recover` | Longest drawdown-recovery duration |
| `cal_all_largest_indicators` | Largest streak statistics |
| `cal_all_longest_indicators` | Longest streak statistics |
| `cal_rolling_gain_loss` | Rolling return distribution statistics |

These kernels retain their existing financial calculation semantics. The rename
and memory architecture change do not silently redefine formulas.

## Architecture

```text
src/calmetrics_engine/
    Python public boundary
          ↓
cpp/bindings.cpp
    dtype / ndim / stride validation
          ↓
cpp/include/calmetrics_engine/
    ArrayView / threading / numeric / calendar contracts
          ↓
cpp/finance/
    pure C++ finance kernels
```

The Python package contains one native extension:

```text
calmetrics_engine._native
```

The target architecture adds the compiler/runtime layers without creating a
second DSL:

```text
FundInvestmentResearchPlatform
        │
        │ current production Typed DSL / DAG contracts
        ▼
CalMetricsEngine
├── compiler       DSL / AST / types / Typed DAG
├── operators      versioned operator contracts
├── runtime        lowering / liveness / memory plan
└── native         C++ operator execution
```

See [docs/architecture.md](docs/architecture.md).

## Platform matrix

Configured CI targets:

| OS | Architectures |
| --- | --- |
| Linux glibc | x86_64, aarch64 |
| Linux musl | x86_64, aarch64 |
| macOS 11+ | x86_64, arm64 |
| Windows | AMD64 |

CPython 3.10–3.14 and NumPy 1.26–2.x are covered by the configured matrix.

Published wheels must not use `-march=native`, mandatory `-mavx*`, or
host-only CPU assumptions. Architecture-specific optimization may be added later
only through tested runtime dispatch or separate safe wheel policy.

## Development

```bash
python -m pip install ".[dev]"
python -m build
python tools/check_dist.py
python -m twine check --strict dist/*
python -I -m pytest tests -q --import-mode=importlib
ruff check src tests tools
```

Native-only tests:

```bash
cmake -S . -B .build-native \
  -DCALMETRICS_ENGINE_BUILD_PYTHON=OFF \
  -DCALMETRICS_ENGINE_BUILD_TESTS=ON
cmake --build .build-native --config Release
ctest --test-dir .build-native -C Release --output-on-failure
```

Sanitizers on GCC/Clang:

```bash
cmake -S . -B .build-sanitized \
  -DCALMETRICS_ENGINE_BUILD_PYTHON=OFF \
  -DCALMETRICS_ENGINE_BUILD_TESTS=ON \
  -DCALMETRICS_ENGINE_SANITIZE=ON
```

## Design rules

1. Python decides **what to calculate**; C++ performs **how it is calculated**.
2. Reusable mathematics belongs in CalMetricsEngine, not duplicated across business centers.
3. Primitive DAG execution should cross Python/C++ once per plan, not once per node.
4. Native kernels accept views and explicit strides; contiguity is an optimization, not a prerequisite.
5. No runtime Numba/JIT dependency in CalMetricsEngine.
6. Business-specific orchestration remains in the research platform.
7. Coupled black-box kernels are allowed only for genuinely inseparable recursive,
   fitting or jointly constrained algorithms.

Build and release details: [docs/publishing.md](docs/publishing.md).
