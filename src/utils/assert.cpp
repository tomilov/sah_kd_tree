#include <utils/assert.hpp>
#include <utils/exception.hpp>

#include <fmt/compile.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <string_view>

#include <cstdlib>

using namespace std::string_view_literals;

namespace utils::impl
{

void assertFailed(const char * expression, spdlog::source_loc sourceLoc, std::string_view message)
{
    spdlog::log(sourceLoc, spdlog::level::critical, FMT_STRING("Assertion ({}) failed{}{}"), expression, (std::empty(message) ? ""sv : ": "sv), message);
    std::abort();
}

void throwInvariantError(const char * expression, spdlog::source_loc sourceLoc, std::string_view message)
{
    auto errorMessage = fmt::format(FMT_STRING("Invariant ({}) violation{}{}"), expression, (std::empty(message) ? ""sv : ": "sv), message);
    spdlog::log(sourceLoc, spdlog::level::critical, "{}", errorMessage);
    if constexpr (sah_kd_tree::kIsDebugBuild) {
        std::abort();
    } else {
        throw InvariantError{errorMessage};
    }
}

}  // namespace utils::impl
