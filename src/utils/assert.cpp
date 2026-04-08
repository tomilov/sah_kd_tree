#include <utils/assert.hpp>
#include <utils/exception.hpp>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <string>
#include <string_view>

#include <cstdlib>

using namespace std::string_view_literals;

namespace utils
{

void vAssertFailed(
    bool assert,
    const char * expression,
    std::source_location sourceLocation,
    fmt::string_view format,
    fmt::format_args args)
{
    fmt::memory_buffer errorMessageBuffer;
    fmt::format_to(fmt::appender(errorMessageBuffer), "Invariant ({}) violation", expression);
    if (std::size(format) != 0) {
        errorMessageBuffer.append(": "sv);
        fmt::vformat_to(fmt::appender(errorMessageBuffer), format, args);
    }
    spdlog::source_loc srcLoc{sourceLocation.file_name(), static_cast<int>(sourceLocation.line()), sourceLocation.function_name()};
    spdlog::log(srcLoc, spdlog::level::critical, "{}", std::string_view{errorMessageBuffer.data(), errorMessageBuffer.size()});
    if (assert) {
        std::abort();
    } else {
        throw InvariantError{std::string{errorMessageBuffer.data(), errorMessageBuffer.size()}};
    }
}

}  // namespace utils
