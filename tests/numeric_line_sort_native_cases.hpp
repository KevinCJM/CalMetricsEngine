#pragma once

#include "../cpp/operators/axis_primitives.hpp"
#include "../cpp/operators/numeric_line_sort.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// This must describe the linked operator library, including HIGHWAY=OFF builds.
// Otherwise an always-standard helper could make the library tests pass silently.
#ifndef CALMETRICS_EXPECT_HIGHWAY_ORDER
#error "operator tests require CALMETRICS_EXPECT_HIGHWAY_ORDER=0 or 1"
#endif

namespace numeric_line_sort_tests {
namespace o = calmetrics_engine::ops;
namespace d = calmetrics_engine::ops::axis_path_detail;

constexpr std::uint64_t canary = UINT64_C(0x40f865a800000000);
constexpr bool have_highway = CALMETRICS_EXPECT_HIGHWAY_ORDER != 0;

inline void check(bool condition, const char *message, const std::string &name) {
    if (!condition) throw std::runtime_error(name + ": " + message);
}

inline double from_bits(std::uint64_t bits) {
    double result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline std::uint64_t to_bits(const double &number) {
    std::uint64_t result;
    std::memcpy(&result, &number, sizeof(result));
    return result;
}

inline bool exact(const double *left, const double *right, std::size_t count) {
    return count == 0 || std::memcmp(left, right, count * sizeof(double)) == 0;
}

// Integer classification is independent of rounding and flush settings. This
// states the expected library domain; the test still calls the real helper.
inline bool library_domain(const std::vector<double> &input) {
    for (const auto &number : input) {
        const auto bits = to_bits(number);
        const auto magnitude = bits & UINT64_C(0x7fffffffffffffff);
        if (bits == UINT64_C(0x8000000000000000) ||
            magnitude > UINT64_C(0x7ff0000000000000) ||
            (magnitude != 0 && magnitude < UINT64_C(0x0010000000000000)))
            return false;
    }
    return true;
}

// Only the test changes FPCR. Production must not modify the caller's floating
// environment. Non-ARM builds still execute all standard fenv/guard cases.
inline std::uint64_t fpcr() noexcept {
#if (defined(__aarch64__) || defined(__arm64__)) && (defined(__clang__) || defined(__GNUC__))
    std::uint64_t result;
    __asm__ volatile("mrs %0, fpcr" : "=r"(result));
    return result;
#else
    return 0;
#endif
}

inline void set_fpcr(std::uint64_t value) noexcept {
#if (defined(__aarch64__) || defined(__arm64__)) && (defined(__clang__) || defined(__GNUC__))
    __asm__ volatile("msr fpcr, %0\n\tisb" : : "r"(value) : "memory");
#else
    (void)value;
#endif
}

struct FloatingEnvironment {
    std::fenv_t saved{};
    std::uint64_t saved_fpcr = fpcr();
    bool active = true;
    FloatingEnvironment() {
        check(std::fegetenv(&saved) == 0, "cannot save floating environment", "line sort");
    }
    ~FloatingEnvironment() {
        if (active) {
            (void)std::fesetenv(&saved);
            set_fpcr(saved_fpcr);
        }
    }
    void restore() {
        check(std::fesetenv(&saved) == 0, "cannot restore floating environment", "line sort");
        set_fpcr(saved_fpcr);
        active = false;
    }
};

inline void seed_flags(const std::string &name) {
    check(std::feclearexcept(FE_ALL_EXCEPT) == 0 &&
              std::feraiseexcept(FE_DIVBYZERO) == 0,
          "cannot seed floating flags", name);
}

struct Counts {
    std::size_t cases = 0, library = 0, standard = 0;
};

inline void compare(const std::vector<double> &input, const std::string &name,
                    Counts &counts) {
    FloatingEnvironment restore;
    const int rounding = std::fegetround();
    const auto control = fpcr();
    const auto count = input.size();
    const auto snapshot = input;
    auto expected = input;
    seed_flags(name);
    if (count > 1) std::sort(expected.begin(), expected.end(), d::NaNLast{});
    const int expected_flags = std::fetestexcept(FE_ALL_EXCEPT);
    for (const auto isa : {o::Isa::automatic, o::Isa::scalar, o::Isa::sse2,
                          o::Isa::avx2, o::Isa::neon}) {
        std::vector<double> storage(count + 4, from_bits(canary));
        std::copy(input.begin(), input.end(), storage.begin() + 2);
        auto *keys = count ? storage.data() + 2 : nullptr;
        o::Audit audit;
        constexpr std::size_t previous_calls = 3;
        audit.highway_vqsort_calls = previous_calls;
        seed_flags(name);
        const bool used = d::sort_numeric_line(keys, count, isa, &audit);
        const bool expected_library = have_highway && isa == o::Isa::automatic &&
            count >= 1024 && library_domain(input);
        check(used == expected_library, "actual library guard differs", name);
        check(audit.highway_vqsort_calls == previous_calls + std::size_t(used),
              "actual library call counter differs", name);
        if (used) check(std::strcmp(audit.isa, "highway_vqsort") == 0,
                        "actual library ISA audit absent", name);
        check(exact(keys, expected.data(), count), "complete sorted bits differ", name);
        const auto actual_control = fpcr();
        const auto actual_rounding = std::fegetround();
        const auto actual_flags = std::fetestexcept(FE_ALL_EXCEPT);
        check(actual_rounding == rounding && actual_control == control &&
                  actual_flags == expected_flags,
              "sort changed floating control or accumulated flags",
              name + ":isa=" + std::to_string(unsigned(isa)) +
              ":round=" + std::to_string(actual_rounding) + "/" + std::to_string(rounding) +
              ":fpcr=" + std::to_string(actual_control) + "/" + std::to_string(control) +
              ":flags=" + std::to_string(actual_flags) + "/" + std::to_string(expected_flags));
        for (const auto index : {std::size_t{0}, std::size_t{1}, count + 2, count + 3})
            check(to_bits(storage[index]) == canary, "sort overwrote canary", name);
        check(exact(input.data(), snapshot.data(), count), "source input changed", name);
        if (count) {
            const std::array<double, 9> probabilities{{0., 1., .05, .5, .95, .5,
                .125, .875, count > 1 ? double(count / 2) / double(count - 1) : .25}};
            for (double probability : probabilities) {
                // Compare the one canonical interpolation body, not a separate
                // reimplementation or approximate/allclose reference.
                const double reference = d::sorted_quantile(expected.data(), count, probability);
                const double actual = d::sorted_quantile(keys, count, probability);
                check(to_bits(actual) == to_bits(reference), "canonical quantile bits differ", name);
            }
            check(std::fegetround() == rounding && fpcr() == control,
                  "interpolation changed floating control", name);
        }
        ++counts.cases;
        if (used) ++counts.library;
        else ++counts.standard;
    }
}

enum class Pattern {
    near_one, ascending, descending, equal, few, positive_zero, negative_zero,
    mixed_zero, infinities, extremes, subnormal, nan_payloads
};

inline std::vector<double> fixture(std::size_t count, Pattern pattern) {
    std::uint64_t state = UINT64_C(0x8dabe761359e04f1);
    std::vector<double> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        result[i] = 1. + double(state & UINT64_C(0xfffff)) / 1048576.;
        switch (pattern) {
        case Pattern::near_one: break;
        case Pattern::ascending: result[i] = double(i) / 16.; break;
        case Pattern::descending: result[i] = double(count - i) / 16.; break;
        case Pattern::equal: result[i] = 1.25; break;
        case Pattern::few: result[i] = double(i % 7) / 8.; break;
        case Pattern::positive_zero: if (i % 3 == 0) result[i] = 0.; break;
        case Pattern::negative_zero: result[i] = i % 3 == 0 ? -0. : -result[i]; break;
        case Pattern::mixed_zero: result[i] = i % 2 ? -0. : 0.; break;
        case Pattern::infinities:
            if (i % 3 == 0) result[i] = -std::numeric_limits<double>::infinity();
            if (i % 3 == 1) result[i] = std::numeric_limits<double>::infinity();
            break;
        case Pattern::extremes:
            if (i % 3 == 0) result[i] = std::numeric_limits<double>::max();
            if (i % 3 == 1) result[i] = -std::numeric_limits<double>::max();
            break;
        case Pattern::subnormal:
            result[i] = from_bits((i % 2 ? UINT64_C(0x8000000000000000) : 0) | (i % 1000 + 1));
            break;
        case Pattern::nan_payloads: {
            constexpr std::array<std::uint64_t, 6> words{{UINT64_C(0x7ff8000000000001),
                UINT64_C(0xfff8000000000247), UINT64_C(0x7ff0000000000073),
                UINT64_C(0xfff0000000000011), UINT64_C(0x3ff0000000000000),
                UINT64_C(0x7ff0000000000000)}};
            result[i] = from_bits(words[i % words.size()]);
            break;
        }
        }
    }
    return result;
}

inline void subnormal_boundaries(Counts &counts) {
    const auto original_flush = fpcr() & (UINT64_C(1) << 24);
    for (const std::uint64_t magnitude : {UINT64_C(0x0000000000000001),
             UINT64_C(0x000fffffffffffff), UINT64_C(0x0010000000000000)}) {
        for (const std::uint64_t sign : {UINT64_C(0), UINT64_C(0x8000000000000000)}) {
            auto input = fixture(1025, Pattern::near_one);
            // A single boundary value at the tail catches incomplete scans.
            input.back() = from_bits(magnitude | sign);
            for (const int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
                check(std::fesetround(rounding) == 0, "cannot set rounding", "subnormal boundary");
                check((fpcr() & (UINT64_C(1) << 24)) == original_flush,
                      "rounding setup changed flush mode", "subnormal boundary");
                compare(input, "boundary:" + std::to_string(magnitude) + ":" + std::to_string(sign), counts);
            }
        }
    }
}

} // namespace numeric_line_sort_tests

inline void test_numeric_line_sort_contracts() {
    using namespace numeric_line_sort_tests;
    FloatingEnvironment restore;
    const int original_rounding = std::fegetround();
    const int original_flags = std::fetestexcept(FE_ALL_EXCEPT);
    const auto original_fpcr = fpcr();
    check(d::numeric_line_sort_available() == have_highway,
          "linked library differs from native test build", "line sort");
    Counts counts;
    constexpr std::array<std::size_t, 18> lengths{{0, 1, 2, 15, 16, 17, 31, 32, 33,
        63, 64, 65, 1023, 1024, 1025, 10000, 50000, 131071}};
    constexpr std::array<Pattern, 12> patterns{{Pattern::near_one, Pattern::ascending,
        Pattern::descending, Pattern::equal, Pattern::few, Pattern::positive_zero,
        Pattern::negative_zero, Pattern::mixed_zero, Pattern::infinities,
        Pattern::extremes, Pattern::subnormal, Pattern::nan_payloads}};
    for (auto count : lengths) for (auto pattern : patterns) {
        const auto input = fixture(count, pattern);
        const auto name = "line:" + std::to_string(count) + ":" + std::to_string(unsigned(pattern));
        for (const int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
            check(std::fesetround(rounding) == 0, "cannot set rounding", name);
            compare(input, name, counts);
        }
    }
    check(counts.cases == 4320 && counts.library == (have_highway ? 160u : 0u) &&
              counts.standard == counts.cases - counts.library,
          "frozen boundary matrix did not exercise its actual backends", "line sort");
    subnormal_boundaries(counts);
#if (defined(__aarch64__) || defined(__arm64__)) && (defined(__clang__) || defined(__GNUC__))
    // FZ is bit 24. Keep rounding, exception masks and all other FPCR bits.
    // Reference and candidate both run under this same setting; comparing with
    // an FZ-off reference would test a different mathematical environment.
    constexpr std::uint64_t flush = UINT64_C(1) << 24;
    for (bool enabled : {false, true}) {
        set_fpcr(enabled ? original_fpcr | flush : original_fpcr & ~flush);
        check((fpcr() & flush) == (enabled ? flush : 0),
              "cannot set requested flush mode", "flush guard");
        subnormal_boundaries(counts);
        for (auto pattern : {Pattern::near_one, Pattern::positive_zero,
                 Pattern::negative_zero, Pattern::mixed_zero, Pattern::subnormal,
                 Pattern::nan_payloads, Pattern::infinities, Pattern::extremes}) {
            const auto input = fixture(1025, pattern);
            for (const int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
                check(std::fesetround(rounding) == 0, "cannot set rounding", "flush guard");
                check((fpcr() & flush) == (enabled ? flush : 0),
                      "rounding setup changed flush mode", "flush guard");
                compare(input, "flush:" + std::to_string(enabled) + ":" +
                    std::to_string(unsigned(pattern)), counts);
            }
        }
    }
#endif
    restore.restore();
    check(std::fegetround() == original_rounding &&
              std::fetestexcept(FE_ALL_EXCEPT) == original_flags && fpcr() == original_fpcr,
          "test did not restore caller floating environment", "line sort");
}
