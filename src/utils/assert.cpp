#include <utils/assert.hpp>
#include <utils/exception.hpp>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <iterator>
#include <string>

#include <cstdlib>

namespace utils
{

void vAssertFailed(bool assert, const char * expression, std::source_location sourceLocation, fmt::string_view format, fmt::format_args args)
{
    std::string errorMessage;
    if (std::size(format) == 0) {
        errorMessage = fmt::format(FMT_STRING("Invariant ({}) violation"), expression);
    } else {
        errorMessage = fmt::format(FMT_STRING("Invariant ({}) violation: {}"), expression, fmt::vformat(format, args));
    }
    spdlog::source_loc srcLoc{sourceLocation.file_name(), static_cast<int>(sourceLocation.line()), sourceLocation.function_name()};
    spdlog::log(srcLoc, spdlog::level::critical, "{}", errorMessage);
    if (assert) {
        std::abort();
    } else {
        throw InvariantError{errorMessage};
    }
}

}  // namespace utils
