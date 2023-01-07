#pragma once

#include <common/config.hpp>

#include <fmt/format.h>
#include <spdlog/common.h>

#include <string_view>

#include <utils/utils_export.h>

namespace utils
{

namespace impl
{

void assertFailed [[noreturn]] (const char * expression, spdlog::source_loc source_loc, std::string_view message) UTILS_EXPORT;
void throwInvariantError [[noreturn]] (const char * expression, spdlog::source_loc source_loc, std::string_view message) UTILS_EXPORT;

}  // namespace impl

}  // namespace utils

// clang-format off
#define ASSERT_MSG(condition, format, ...) do if constexpr (sah_kd_tree::kIsDebugBuild) if (!(condition)) utils::impl::assertFailed(#condition, {__FILE__, __LINE__, static_cast<const char *>(__FUNCTION__)}, fmt::vformat(FMT_STRING(format), fmt::make_format_args(__VA_ARGS__))); while (false)
#define ASSERT(condition) ASSERT_MSG(condition, "")
#define INVARIANT(condition, format,  ...) do if (!(condition)) { utils::impl::throwInvariantError(#condition, {__FILE__, __LINE__, static_cast<const char *>(__FUNCTION__)}, fmt::vformat(FMT_STRING(format), fmt::make_format_args(__VA_ARGS__))); } while (false)
// clang-format on
