// Test-only malloc-family audit. Build as a dylib and inject into an isolated
// benchmark process; this is never linked into the engine or shipped wheel.
// Counters are requested bytes, not allocator usable size or RSS. CPython's
// suballocations within an existing arena are not separate malloc calls.
#if !defined(__APPLE__)
#error "This audit adapter currently requires macOS dyld interposition"
#endif
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <pthread.h>
#include <algorithm>

namespace {
constexpr std::size_t capacity = 1u << 20;
struct Entry { void *pointer; std::size_t bytes; std::uint64_t epoch, ticket; };
Entry entries[capacity]{};
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
std::uint64_t epoch = 0, next_ticket = 0;
bool active = false;
struct Counters {
    std::uint64_t allocations = 0, frees = 0, allocated_bytes = 0;
    std::uint64_t live_bytes = 0, peak_bytes = 0, untracked_frees = 0, table_overflows = 0;
} counters;
void track(void *pointer, std::size_t bytes) {
    if (!pointer) return;
    pthread_mutex_lock(&mutex);
    if (active) {
        auto index = (reinterpret_cast<std::uintptr_t>(pointer) >> 4) * 11400714819323198485ull;
        bool found = false;
        for (std::size_t i = 0; i < capacity; ++i) {
            auto &entry = entries[(index + i) & (capacity - 1)];
            if (!entry.pointer || entry.pointer == pointer || entry.epoch != epoch) {
                // A concurrent realloc may release this address before its
                // wrapper can retire the old record. Account that retirement
                // here; the old ticket can no longer remove the new object.
                if (entry.pointer == pointer && entry.epoch == epoch) {
                    counters.live_bytes -= entry.bytes; ++counters.frees;
                }
                entry = {pointer, bytes, epoch, ++next_ticket}; found = true; break;
            }
        }
        if (!found) ++counters.table_overflows;
        ++counters.allocations; counters.allocated_bytes += bytes;
        counters.live_bytes += bytes;
        counters.peak_bytes = std::max(counters.peak_bytes, counters.live_bytes);
    }
    pthread_mutex_unlock(&mutex);
}
void forget(void *pointer, std::uint64_t ticket = 0) {
    if (!pointer) return;
    pthread_mutex_lock(&mutex);
    if (active) {
        auto index = (reinterpret_cast<std::uintptr_t>(pointer) >> 4) * 11400714819323198485ull;
        bool found = false;
        for (std::size_t i = 0; i < capacity; ++i) {
            auto &entry = entries[(index + i) & (capacity - 1)];
            if (entry.epoch != epoch) break;
            if (entry.pointer == pointer && (!ticket || entry.ticket == ticket)) {
                counters.live_bytes -= entry.bytes; entry.pointer = nullptr;
                found = true; ++counters.frees; break;
            }
        }
        if (!found && !ticket) ++counters.untracked_frees;
    }
    pthread_mutex_unlock(&mutex);
}
std::uint64_t ticket_for(void *pointer) {
    pthread_mutex_lock(&mutex);
    std::uint64_t result = 0;
    if (active && pointer) {
        const auto index = (reinterpret_cast<std::uintptr_t>(pointer) >> 4) * 11400714819323198485ull;
        for (std::size_t i = 0; i < capacity; ++i) {
            const auto &entry = entries[(index+i) & (capacity-1)];
            if (entry.epoch != epoch) break;
            if (entry.pointer == pointer) { result = entry.ticket; break; }
        }
    }
    pthread_mutex_unlock(&mutex);
    return result;
}
}
extern "C" {
__attribute__((visibility("default"))) void cme_heap_begin() {
    pthread_mutex_lock(&mutex); ++epoch; counters = {}; active = true; pthread_mutex_unlock(&mutex);
}
__attribute__((visibility("default"))) void cme_heap_end(Counters *output) {
    pthread_mutex_lock(&mutex); active = false; *output = counters; pthread_mutex_unlock(&mutex);
}
void *cme_malloc(std::size_t size) { auto *p = std::malloc(size); track(p, size); return p; }
void *cme_calloc(std::size_t n, std::size_t size) {
    auto *p = std::calloc(n, size); if (p) track(p, n * size); return p;
}
void cme_free(void *p) { forget(p); std::free(p); }
void *cme_realloc(void *p, std::size_t size) {
    // A failed realloc retains the old allocation. Its tracking record must
    // therefore be removed only after successful reallocation (or free).
    const auto ticket = ticket_for(p);
    auto *next = std::realloc(p, size);
    if (next || !size) {
        if (ticket) forget(p, ticket);
        track(next, size);
    }
    return next;
}
int cme_posix_memalign(void **pointer, std::size_t alignment, std::size_t size) {
    const auto result = posix_memalign(pointer, alignment, size);
    if (!result) track(*pointer, size); return result;
}
void *cme_aligned_alloc(std::size_t alignment, std::size_t size) {
    auto *p = aligned_alloc(alignment, size); track(p, size); return p;
}
}
#define INTERPOSE(replacement, original) \
    __attribute__((used)) static const struct { const void *replace; const void *with; } \
    interpose_##original __attribute__((section("__DATA,__interpose"))) = \
    {reinterpret_cast<const void *>(&replacement), reinterpret_cast<const void *>(&original)};
INTERPOSE(cme_malloc, malloc)
INTERPOSE(cme_calloc, calloc)
INTERPOSE(cme_free, free)
INTERPOSE(cme_realloc, realloc)
INTERPOSE(cme_posix_memalign, posix_memalign)
INTERPOSE(cme_aligned_alloc, aligned_alloc)
