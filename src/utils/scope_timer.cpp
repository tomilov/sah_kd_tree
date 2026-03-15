#include <utils/scope_timer.hpp>

#include <fmt/chrono.h>
#include <spdlog/spdlog.h>

namespace utils
{

ScopeTimer::ScopeTimer(
    std::string_view nameIn,
    const std::source_location & sourceLocationIn)
    : name{nameIn}
    , sourceLocation{sourceLocationIn}
{}

ScopeTimer::~ScopeTimer()
{
    auto dt = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start);
    SPDLOG_TRACE("{}:{}:{} in {}: {} {}", sourceLocation.file_name(), sourceLocation.line(), sourceLocation.column(), sourceLocation.function_name(), name, dt);
}

}  // namespace utils
