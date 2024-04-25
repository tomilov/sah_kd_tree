#pragma once

#include <common/config.hpp>

#include <fmt/format.h>
#include <spdlog/common.h>

#include <utils/utils_export.h>

namespace utils
{

void vAssertFailed [[noreturn]] (bool assert, const char * expression, spdlog::source_loc sourceLoc, fmt::string_view format, fmt::format_args args) UTILS_EXPORT;

template<typename... Args>
void assertFailed [[noreturn]] (bool assert, const char * expression, spdlog::source_loc sourceLoc, fmt::format_string<Args...> format, Args &&... args)
{
    vAssertFailed(assert, expression, sourceLoc, format, fmt::make_format_args(args...));
}

}  // namespace utils

#define ASSERT_MSG(condition, format, ...)                                                                                      \
    do {                                                                                                                        \
        if constexpr (sah_kd_tree::kIsDebugBuild) {                                                                             \
            if (!(condition)) {                                                                                                 \
                ::utils::assertFailed(true, #condition, {__FILE__, __LINE__, __FUNCTION__}, FMT_STRING(format), ##__VA_ARGS__); \
            }                                                                                                                   \
        }                                                                                                                       \
    } while (false)

#define ASSERT(condition) ASSERT_MSG(condition, "")

#define INVARIANT(condition, format, ...)                                                                                                         \
    do {                                                                                                                                          \
        if (!(condition)) {                                                                                                                       \
            ::utils::assertFailed(sah_kd_tree::kIsDebugBuild, #condition, {__FILE__, __LINE__, __FUNCTION__}, FMT_STRING(format), ##__VA_ARGS__); \
        }                                                                                                                                         \
    } while (false)
