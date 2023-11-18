#include <utils/assert.hpp>
#include <utils/exception.hpp>

#include <fmt/compile.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <string_view>

#include <cstdlib>

using namespace std::string_view_literals;

namespace utils
{

void vAssertFailed(const char * expression, spdlog::source_loc sourceLoc, fmt::string_view format, fmt::format_args args)
{
    if (std::size(format) == 0) {
        spdlog::log(sourceLoc, spdlog::level::critical, FMT_STRING("Assertion ({}) failed"), expression);
    } else {
        spdlog::log(sourceLoc, spdlog::level::critical, FMT_STRING("Assertion ({}) failed: {}"), expression, fmt::vformat(format, args));
    }
    std::abort();
}

void vThrowInvariantError(const char * expression, spdlog::source_loc sourceLoc, fmt::string_view format, fmt::format_args args)
{
    std::string errorMessage;
    if (std::size(format) == 0) {
        errorMessage = fmt::format(FMT_STRING("Invariant ({}) violation"), expression);
    } else {
        errorMessage = fmt::format(FMT_STRING("Invariant ({}) violation: {}"), expression, fmt::vformat(format, args));
    }
    spdlog::log(sourceLoc, spdlog::level::critical, "{}", errorMessage);
    if constexpr (sah_kd_tree::kIsDebugBuild) {
        std::abort();
    } else {
        throw InvariantError{errorMessage};
    }
}

}  // namespace utils
