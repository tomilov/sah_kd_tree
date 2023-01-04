#pragma once

#include <fmt/format.h>

#include <string_view>

#include <utils/utils_export.h>

namespace utils
{

#ifdef NDEBUG
inline constexpr bool kEnableAssert = false;
#else
inline constexpr bool kEnableAssert = true;
#endif

namespace impl
{

void assertFailed [[noreturn]] (const char * expression, const char * file, int line, const char * function, std::string_view message) UTILS_EXPORT;
void throwInvariantError [[noreturn]] (const char * expression, const char * file, int line, const char * function, std::string_view message) UTILS_EXPORT;

}  // namespace impl

}  // namespace utils

// clang-format off
#define ASSERT_MSG(condition, format, ...) do if constexpr (utils::kEnableAssert) if (!(condition)) utils::impl::assertFailed(#condition, __FILE__, __LINE__, static_cast<const char *>(__FUNCTION__), fmt::vformat(FMT_STRING(format), fmt::make_format_args(__VA_ARGS__))); while (false)
#define ASSERT(condition) ASSERT_MSG(condition, "")
#define INVARIANT(condition, format,  ...) do if (!(condition)) { utils::impl::throwInvariantError(#condition, __FILE__, __LINE__, static_cast<const char *>(__FUNCTION__), fmt::vformat(FMT_STRING(format), fmt::make_format_args(__VA_ARGS__))); } while (false)
// clang-format on
