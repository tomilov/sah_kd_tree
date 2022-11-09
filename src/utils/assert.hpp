#pragma once

#include <utils/utils_export.h>

#include <fmt/format.h>

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
#define ASSERT_MSG(condition, format, ...) do if constexpr (utils::kEnableAssert) if (!(condition)) utils::impl::AssertFailed(#condition, __FILE__, __LINE__, __PRETTY_FUNCTION__, fmt::vformat(FMT_STRING(format), fmt::make_format_args(__VA_ARGS__))); while (false)
#define ASSERT(condition) ASSERT_MSG(condition, "")
#define INVARIANT(condition, format,  ...) do if (!(condition)) { auto message = fmt::vformat(FMT_STRING(format), fmt::make_format_args(__VA_ARGS__)); if constexpr (utils::kEnableAssert) utils::impl::AssertFailed(#condition, __FILE__, __LINE__, __PRETTY_FUNCTION__, message); else utils::impl::ThrowInvariantError(#condition, message); } while (false)
// clang-format on
