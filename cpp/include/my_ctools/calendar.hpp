#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

namespace my_ctools {
inline constexpr std::int64_t ns_per_second = 1'000'000'000;
inline constexpr std::int64_t seconds_per_day = 86'400;
inline constexpr std::int64_t ns_per_day = ns_per_second * seconds_per_day;

inline std::int64_t floor_div(std::int64_t value, std::int64_t divisor) {
    return value / divisor - (value % divisor < 0 ? 1 : 0);
}
inline bool leap_year(std::int64_t year) {
    return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}
inline int month_days(std::int64_t year, int month) {
    constexpr int lengths[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    return lengths[month - 1] + (month == 2 && leap_year(year) ? 1 : 0);
}
inline std::int64_t days_before_year(std::int64_t year) {
    const auto previous = year - 1;
    return 365 * (year - 1970) + previous / 4 - previous / 100 + previous / 400
           - (1969 / 4 - 1969 / 100 + 1969 / 400);
}
struct CivilDate { std::int64_t year; int month; int day; };
inline CivilDate civil_date(std::int64_t days) {
    auto year = 1970 + days / 365;
    while (days < days_before_year(year)) --year;
    while (days >= days_before_year(year + 1)) ++year;
    auto remaining = days - days_before_year(year);
    int month = 1;
    while (remaining >= month_days(year, month)) {
        remaining -= month_days(year, month++);
    }
    return {year, month, static_cast<int>(remaining) + 1};
}
inline std::int64_t epoch_day(const CivilDate& date) {
    auto days = days_before_year(date.year) + date.day - 1;
    for (int month = 1; month < date.month; ++month) days += month_days(date.year, month);
    return days;
}
inline std::string date_string(std::int64_t ns) {
    const auto date = civil_date(floor_div(ns, ns_per_day));
    return std::to_string(date.year) + (date.month < 10 ? "0" : "")
         + std::to_string(date.month) + (date.day < 10 ? "0" : "")
         + std::to_string(date.day);
}

// The rolling algorithm historically uses second precision and clamps month ends.
// Targets past datetime64[ns]'s upper bound saturate instead of overflowing.
inline std::int64_t add_months_ns(std::int64_t ns, int months) {
    const auto day = floor_div(ns, ns_per_day);
    auto date = civil_date(day);
    const auto total_months = date.year * 12 + date.month - 1 + months;
    date.year = total_months / 12;
    date.month = static_cast<int>(total_months % 12) + 1;
    date.day = std::min(date.day, month_days(date.year, date.month));
    const auto seconds = epoch_day(date) * seconds_per_day
                       + (floor_div(ns, ns_per_second) - day * seconds_per_day);
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    if (seconds > maximum / ns_per_second) return maximum;
    return seconds * ns_per_second;
}
}  // namespace my_ctools
