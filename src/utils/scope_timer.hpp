#pragma once

#include <chrono>
#include <source_location>

#include <utils/utils_export.h>

namespace utils
{

class UTILS_EXPORT ScopeTimer
{
public:
    using Clock = std::chrono::high_resolution_clock;

    explicit ScopeTimer(
        std::string_view name,
        const std::source_location & sourceLocation = std::source_location::current());
    ~ScopeTimer();

private:
    const std::string name;
    const std::source_location sourceLocation;
    const Clock::time_point start = Clock::now();
};

}  // namespace utils
