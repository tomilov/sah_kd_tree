#include <utils/scope_timer.hpp>

#include <fmt/chrono.h>
#include <spdlog/spdlog.h>

namespace utils
{

ScopeTimer::ScopeTimer(
    utils::Name nameIn,
    const std::source_location & sourceLocationIn)
    : name{std::move(nameIn)}
    , sourceLocation{sourceLocationIn}
{}

ScopeTimer::~ScopeTimer()
{
    auto dt = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start);
    spdlog::source_loc srcLoc{sourceLocation.file_name(), static_cast<int>(sourceLocation.line()), sourceLocation.function_name()};
    spdlog::log(srcLoc, spdlog::level::info, "{}: {}", name, dt);
}

}  // namespace utils
