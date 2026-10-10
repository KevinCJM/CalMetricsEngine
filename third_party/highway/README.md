# Highway 1.3.0

Unmodified upstream archive: https://github.com/google/highway/archive/refs/tags/1.3.0.tar.gz

SHA256: `07b3c1ba2c1096878a85a31a5b9b3757427af963b1141ca904db2f9f4afe0bc2`.

CMake verifies/extracts the archive for offline AOT builds and statically links upstream `hwy` plus double ascending selection sources (`vqsort.cc`, `vqsort_f64a.cc`). No installed Homebrew library is required. Upstream runtime CPU dispatch stays enabled; no private numerical thread pool.

The adapter uses VQSelect with existing numerical/tie guards. `CALMETRICS_ENGINE_USE_HIGHWAY=OFF` preserves standard-library selection. Initially enabled only on Apple arm64; other targets preserve existing behavior until separately measured.

Keep upstream LICENSE (Apache-2.0), LICENSE-BSD3, and per-file notices. VQSort uses Apache-2.0; other files retain their upstream license options.
