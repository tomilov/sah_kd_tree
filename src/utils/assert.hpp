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
#define ASSERT_MSG(condition, ...) do if constexpr (utils::kEnableAssert) if (!(condition)) utils::impl::AssertFailed(#condition, __FILE__, __LINE__, __PRETTY_FUNCTION__, fmt::format(__VA_ARGS__)); while (false)
#define ASSERT(condition) ASSERT_MSG(condition, "")
#define INVARIANT(condition, ...) do if (!(condition)) { if constexpr (utils::kEnableAssert) utils::impl::AssertFailed(#condition, __FILE__, __LINE__, __PRETTY_FUNCTION__, fmt::format(__VA_ARGS__)); else utils::impl::ThrowInvariantError(#condition, fmt::format(__VA_ARGS__)); } while (false)
// clang-format on
