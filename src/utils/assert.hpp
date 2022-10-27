#pragma once

#include <utils/utils_export.h>

#include <string_view>

namespace utils
{

#ifdef NDEBUG
inline constexpr bool kEnableAssert = false;
#else
inline constexpr bool kEnableAssert = true;
#endif

namespace impl
{

void AssertFailed [[noreturn]] (const char * expression, const char * file, unsigned int line, const char * function, std::string_view message) UTILS_EXPORT;
void ThrowInvariantError [[noreturn]] (const char * expression, std::string_view message) UTILS_EXPORT;

}  // namespace impl

}  // namespace utils

// clang-format off
#define ASSERT_MSG(condition, message) do if constexpr (utils::kEnableAssert) if (!(condition)) utils::impl::AssertFailed(#condition, __FILE__, __LINE__, __PRETTY_FUNCTION__, message); while (false)
#define ASSERT(condition) ASSERT_MSG(condition, "")
#define INVARIANT(condition, message) do if (!(condition)) { if constexpr (utils::kEnableAssert) utils::impl::AssertFailed(#condition, __FILE__, __LINE__, __PRETTY_FUNCTION__, message); else utils::impl::ThrowInvariantError(#condition, message); } while (false)
// clang-format on
