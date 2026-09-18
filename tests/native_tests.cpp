#include "calmetrics_engine/calendar.hpp"
#include "calmetrics_engine/finance.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    namespace mc = calmetrics_engine;
    std::size_t checks = 0;
    auto require = [&](bool condition, const char* message) {
        ++checks;
        if (!condition) throw std::runtime_error(message);
    };
    try {
        require(mc::date_string(0) == "19700101", "epoch date");
        require(mc::date_string(-1) == "19691231", "negative fractional timestamp");
        require(mc::date_string(std::numeric_limits<std::int64_t>::max()) == "22620411", "upper date");
        require(mc::date_string(std::numeric_limits<std::int64_t>::min() + 1) == "16770921", "lower date");
        require(mc::add_months_ns(1'580'428'800'000'000'000, 1) == 1'582'934'400'000'000'000,
                "January 31 must clamp to February 29 in 2020");
        require(!mc::leap_year(1900) && mc::leap_year(2000), "Gregorian leap years");
        require(mc::add_months_ns(0, std::numeric_limits<int>::max()) ==
                std::numeric_limits<std::int64_t>::max(), "large period saturation");
        for (std::int64_t day = -106752; day <= 106751; ++day) {
            require(mc::epoch_day(mc::civil_date(day)) == day, "calendar round trip");
        }
        bool invoked = false;
        mc::parallel_columns(0, 0, [&](std::size_t, std::size_t) { invoked = true; });
        require(!invoked, "zero work must not spawn workers");
        bool propagated = false;
        try {
            mc::parallel_columns(4, 2, [](std::size_t first, std::size_t) {
                if (first == 1) throw std::runtime_error("worker failure");
            });
        } catch (const std::runtime_error&) { propagated = true; }
        require(propagated, "worker exception must propagate after join");

        std::int64_t starts[] = {0, 0}, ends[] = {-1, -1}, day[] = {0};
        double storage[9][4];
        std::array<double*, 9> outputs{};
        for (std::size_t k = 0; k < outputs.size(); ++k) {
            std::fill_n(storage[k], 4, 123.0);
            outputs[k] = storage[k] + 1;
        }
        mc::rolling_gain(
            {nullptr, 0, 2, 2, 1},
            {starts, 2, 1},
            {ends, 2, 1},
            {nullptr, 0, 1},
            1,
            outputs,
            2
        );
        for (std::size_t k = 0; k < outputs.size(); ++k) {
            require(storage[k][0] == 123.0 && storage[k][3] == 123.0, "empty output bounds");
            require(std::isnan(storage[k][1]) && std::isnan(storage[k][2]), "empty rolling values");
        }
        double one_row[] = {0.01, -0.01};
        ends[0] = ends[1] = 0;
        mc::rolling_gain(
            {one_row, 1, 2, 2, 1},
            {starts, 2, 1},
            {ends, 2, 1},
            {day, 1, 1},
            1,
            outputs,
            2
        );
        for (std::size_t k = 0; k < outputs.size(); ++k) {
            require(storage[k][0] == 123.0 && storage[k][3] == 123.0, "short window output bounds");
            require(std::isnan(storage[k][1]) && std::isnan(storage[k][2]), "short window values");
        }
        double result[2];
        std::int64_t periods[2];
        std::vector<std::string> dates(2);
        mc::max_drawdown(
            {nullptr, 0, 2, 2, 1},
            {nullptr, 0, 1},
            result,
            dates,
            periods,
            2
        );
        require(std::isnan(result[0]) && periods[0] == mc::not_recovered, "empty max drawdown");
        mc::longest_recovery({nullptr, 0, 2, 2, 1}, periods, 2);
        require(periods[0] == 0 && periods[1] == 0, "empty recovery");
        mc::column_statistics({nullptr, 0, 2, 2, 1}, result, nullptr, 2);
        require(std::isnan(result[0]) && std::isnan(result[1]), "empty standard deviation");

        double strided_storage[] = {1.0, 99.0, 2.0, 99.0, 3.0, 99.0};
        double strided_std[1], strided_mean[1];
        mc::column_statistics({strided_storage, 3, 1, 2, 1}, strided_std, strided_mean, 1);
        require(strided_mean[0] == 2.0, "strided matrix mean");
        require(strided_std[0] == 1.0, "strided matrix std");
        std::cout << checks << " native checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << "Native test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
