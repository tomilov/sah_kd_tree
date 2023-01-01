#include <utils/assert.hpp>
#include <utils/exception.hpp>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <cstdlib>

namespace utils::impl
{

void AssertFailed(const char * expression, const char * file, unsigned int line, const char * function, std::string_view message)
{
    SPDLOG_CRITICAL("ERROR at {}:{}{}{}. Assertion '{}' failed{}{}", file, line, (function ? ":" : ""), (function ? function : ""), expression, (!std::empty(message) ? ": " : ""), message);
    std::abort();
}

void ThrowInvariantError(const char * expression, std::string_view message)
{
    auto errorMessage = fmt::format("Invariant ({}) violation: {}", expression, message);
    SPDLOG_CRITICAL("{}", errorMessage);
    throw InvariantError{errorMessage};
}

}  // namespace utils::impl
