#include "calmetrics_engine/operators.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

extern "C" int covariance_syrk_screen(const double *, int, int, int,
                                     std::ptrdiff_t, int, double *);
extern "C" void covariance_syrk_counters(std::uint64_t *);
extern "C" int covariance_syrk_threading();

namespace ops = calmetrics_engine::ops;
int main() {
    constexpr double sentinel = -937.25;
    std::size_t checked = 0, accepted = 0, fallbacks = 0, sparse_replays = 0;
    for (auto shape : {std::pair{129, 17}, std::pair{513, 65},
                       std::pair{4101, 17}, std::pair{1024, 128},
                       std::pair{513, 257}}) {
        const auto [n, d] = shape;
        std::mt19937_64 generator(9117);
        std::normal_distribution<double> normal;
        std::vector<double> source(static_cast<std::size_t>(n) * d);
        for (auto &value : source) value = normal(generator);
        for (int pattern = 0; pattern < 8; ++pattern) {
            for (int storage = 0; storage < 4; ++storage) {
                const bool f = storage % 2;
                const int leading = (f ? n : d) + (storage >= 2 ? 3 : 0);
                std::vector<double> input(static_cast<std::size_t>(leading) * (f ? d : n), sentinel);
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < d; ++j) {
                        double value = source[static_cast<std::size_t>(i) * d + j];
                        if (pattern == 1 && j == d - 1) value = 7.0;
                        if (pattern == 2) value *= 1e3;
                        if (pattern == 3) value *= 0x1p-450;
                        if (pattern == 4 && j < 2)
                            value = j ? (i % 2 ? -1.0 : 1.0) : (i % 4 < 2 ? 1.0 : -1.0);
                        if (pattern == 5 && j == d - 1) value = source[static_cast<std::size_t>(i) * d];
                        if (pattern == 6 && i == n - 2 && j == d - 1)
                            value = std::numeric_limits<double>::quiet_NaN();
                        if (pattern == 7 && i == n - 2 && j == d - 1)
                            value = std::numeric_limits<double>::infinity();
                        input[static_cast<std::size_t>(f ? j : i) * leading + (f ? i : j)] = value;
                    }
                const auto before = input;
                ops::Value value;
                value.data = input.data();
                value.shape = ops::matrix_shape(n, d);
                value.stride = f ? std::array<std::ptrdiff_t, 3>{1, leading, 1} :
                                   std::array<std::ptrdiff_t, 3>{leading, 1, 1};
                const auto plan = ops::prepare(ops::lookup("covariance"), &value, 1);
                std::vector<double> expected(static_cast<std::size_t>(d) * d);
                ops::Output output;
                output.shape = plan.output_shape;
                output.data = expected.data();
                ops::Workspace workspace;
                ops::Audit audit;
                ops::execute(plan, output, workspace, ops::Isa::automatic, audit);
                for (int block : {1, 127, 512}) {
                    // Medium-panel admission overrides the requested block.
                    // Verify that branch once, including its non-power-of-two tail.
                    if (d >= 256 && n <= 2048 && block != 512) continue;
                    std::vector<double> guarded(expected.size() + 16, sentinel);
                    auto *actual = guarded.data() + 8;
                    const int threading = covariance_syrk_threading();
                    if (covariance_syrk_screen(input.data(), n, d, f, leading, block, actual) != 0)
                        return 1;
                    if (threading != covariance_syrk_threading()) return 2;
                    if (std::memcmp(before.data(), input.data(), input.size() * sizeof(double))) return 3;
                    for (int i = 0; i < 8; ++i)
                        if (guarded[i] != sentinel || guarded[guarded.size() - i - 1] != sentinel) return 4;
                    std::uint64_t counters[5]{};
                    covariance_syrk_counters(counters);
                    fallbacks += counters[2] != 0;
                    accepted += counters[2] == 0;
                    sparse_replays += counters[2] == 0 && counters[1] > static_cast<std::uint64_t>(f ? d : 0);
                    if (counters[2] && std::memcmp(actual, expected.data(), expected.size() * sizeof(double))) return 5;
                    for (int j = 0; j < d; ++j) {
                        if (std::memcmp(actual + static_cast<std::size_t>(j) * d + j,
                                        expected.data() + static_cast<std::size_t>(j) * d + j,
                                        sizeof(double))) return 6;
                        for (int k = 0; k < d; ++k) {
                            const auto index = static_cast<std::size_t>(j) * d + k;
                            const double a = actual[index], e = expected[index];
                            if (std::isnan(a) != std::isnan(e) || std::isinf(a) != std::isinf(e)) return 7;
                            if (!std::isnan(e) && (std::signbit(a) != std::signbit(e) || (a == 0) != (e == 0))) return 8;
                            if (std::isfinite(e) && std::abs(a - e) > 1e-12 + 1e-12 * std::abs(e)) return 9;
                            if (std::memcmp(actual + index, actual + static_cast<std::size_t>(k) * d + j, sizeof(double))) return 10;
                        }
                    }
                    ++checked;
                }
            }
        }
    }
    if (!accepted || !fallbacks || !sparse_replays) return 11;
    std::cout << "{\"calls\":" << checked << ",\"accepted\":" << accepted
              << ",\"fallbacks\":" << fallbacks << ",\"sparse_replays\":"
              << sparse_replays << ",\"passed\":true}\n";
}
