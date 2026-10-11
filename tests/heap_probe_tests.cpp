// macOS benchmark-instrumentation regression; run with DYLD_INSERT_LIBRARIES.
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>

struct Counters {
  std::uint64_t allocations, frees, allocated_bytes, live_bytes, peak_bytes;
  std::uint64_t untracked_frees, table_overflows;
};

int main() {
  auto begin = reinterpret_cast<void (*)()>(dlsym(RTLD_DEFAULT, "cme_heap_begin"));
  auto end = reinterpret_cast<void (*)(Counters *)>(dlsym(RTLD_DEFAULT, "cme_heap_end"));
  auto snapshot = reinterpret_cast<void (*)(Counters *)>(dlsym(RTLD_DEFAULT, "cme_heap_snapshot"));
  auto reset = reinterpret_cast<void (*)()>(dlsym(RTLD_DEFAULT, "cme_heap_reset_peak"));
  auto allocate = reinterpret_cast<void *(*)(std::size_t)>(dlsym(RTLD_DEFAULT, "cme_malloc"));
  auto release = reinterpret_cast<void (*)(void *)>(dlsym(RTLD_DEFAULT, "cme_free"));
  if (!begin || !end || !snapshot || !reset || !allocate || !release)
    return 2;
  Counters setup{}, cold{}, retained{}, warm{}, finished{};
  begin();
  void *workspace = allocate(4096);
  snapshot(&setup);
  void *first_result = allocate(8192);
  snapshot(&cold);
  release(first_result);
  snapshot(&retained);
  reset();
  void *second_result = allocate(8192);
  snapshot(&warm);
  release(second_result);
  release(workspace);
  end(&finished);
  // The freed first result is absent from warm peak; retained workspace remains.
  if (setup.live_bytes != 4096 || cold.peak_bytes != 12288 ||
      retained.live_bytes != 4096 || warm.peak_bytes != 12288 ||
      finished.live_bytes != 0 || finished.allocations != 3 ||
      finished.frees != 3 || finished.table_overflows != 0)
    return 1;
  std::puts("PASS: setup retained, released cold result excluded, warm peak exact");
}
