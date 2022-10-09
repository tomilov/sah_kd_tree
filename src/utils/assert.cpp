#include <utils/assert.hpp>
#include <utils/exception.hpp>

#include <fmt/compile.h>
#include <fmt/format.h>

#include <iostream>

#include <cstdlib>

namespace utils::impl
{

void AssertFailed(const char * expression, const char * file, unsigned int line, const char * function, const char * message)
{
    auto errorMessage = fmt::format(FMT_COMPILE("ERROR at {}:{}{}{}. Assertion '{}' failed{}{}\n"), file, line, (function ? ":" : ""), (function ? function : ""), expression, (message ? ": " : ""), (message ? message : ""));

    std::cerr << errorMessage;

    // flush log

    std::abort();
}

void ThrowInvariantError(const char * expression, const char * message)
{
    auto errorMessage = fmt::format(FMT_COMPILE("Invariant ({}) violation: {}"), expression, message);

    // log

    throw InvariantError{errorMessage};
}

}  // namespace utils::impl