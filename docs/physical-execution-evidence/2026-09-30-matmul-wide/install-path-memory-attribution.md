# Installation-path allocation attribution

**The six-child controlled comparison removed the observed +8 B retained / +72 B cumulative heap difference.** It supports attribution to unequal install paths for these 512³ C/C cases; it is not a full final-build memory acceptance or a per-allocation stack trace.

Evidence: `equal-install-attribution/manifest.json` has `status=completed-attribution-only`; all six raw stdout/stderr files and `memory.json` are retained. The same immutable 1aec/16d2 bytes were copied into real `equal-install-old` / `equal-install-new` directories, each 53 path bytes and 89 worker-path bytes. Full copied file hashes matched the source installs. Three entries (ordinary, prepared, direct) × old/new used the unchanged heap child and archived scalar reference.

| Entry | Original setup/cold/warm cumulative allocated difference | Original retained live difference | Equal-path allocated/live/peak/count differences |
|---|---:|---:|---|
| Ordinary | +72 B | +8 B | All 0 |
| Prepared | +72 B | +8 B | All 0 |
| Direct | 0 B | 0 B | All 0 |

The original ordinary/prepared setup peak difference was 0, while cold/warm peak differences were +8 B. The complete original 114-case timing and 51-case/306-child memory campaign remains under `candidate-16d2`; its measured increases have not been erased or relabelled zero.

The source mechanism is specific. `cpp/native_api_bindings.cpp:826–838` constructs the worker path from `_native.__file__`; the `EngineAPI` constructor invokes this even for single-lane work (`:844–848`). `cpp/native_runtime.cpp:50–51` stores it in `worker_path_`; process-pool creation remains lazy. Original worker paths were 74 bytes for 1aec and 85 for 16d2. The SDK libc++ `string:2096–2107` uses alignment 8 and `align_up(n+1,8)-1`; its initializer (`:2367–2382`) requests recommendation+1. Thus these retained strings request 80 and 88 bytes, explaining the +8 B capacity difference. The header is `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1/string`.

`tools/heap_probe.cpp:1–4,98–100` counts requested malloc-family bytes, not allocator usable size or RSS. The +72 B already appeared during setup and remained unchanged through cold/warm work; equal-length path control removes it. We have not traced every transient allocation contributing to that total and do not claim such a trace.

Final aligned-admission acceptance will use another equal-length real pair, `final-install-old` / `final-install-new`, preserving the attribution binaries. All original 90 timing and 31 memory cases plus the predeclared additions remain; final totals stay 114 timing and 51 memory cases. RSS and stack conclusions remain separate from this heap attribution.
