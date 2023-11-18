#pragma once

#include <common/config.hpp>

#include <fmt/format.h>
#include <spdlog/common.h>

#include <string_view>

#include <utils/utils_export.h>

namespace utils
{

void vAssertFailed [[noreturn]] (const char * expression, spdlog::source_loc sourceLoc, fmt::string_view format, fmt::format_args args) UTILS_EXPORT;
void vThrowInvariantError [[noreturn]] (const char * expression, spdlog::source_loc sourceLoc, fmt::string_view format, fmt::format_args args) UTILS_EXPORT;

template<typename... T>
void assertFailed [[noreturn]] (const char * expression, spdlog::source_loc sourceLoc, fmt::format_string<T...> format, T &&... args)
{
    vAssertFailed(expression, sourceLoc, format, fmt::make_format_args(args...));
}

template<typename... T>
void throwInvariantError [[noreturn]] (const char * expression, spdlog::source_loc sourceLoc, fmt::format_string<T...> format, T &&... args)
{
    vThrowInvariantError(expression, sourceLoc, format, fmt::make_format_args(args...));
}

}  // namespace utils

#define ASSERT_MSG(condition, format, ...)                                                                              \
    do {                                                                                                                \
        if constexpr (sah_kd_tree::kIsDebugBuild) {                                                                     \
            if (!(condition)) {                                                                                         \
                utils::assertFailed(#condition, {__FILE__, __LINE__, __FUNCTION__}, FMT_STRING(format), ##__VA_ARGS__); \
            }                                                                                                           \
        }                                                                                                               \
    } while (false)

#define ASSERT(condition) ASSERT_MSG(condition, "")

#define INVARIANT(condition, format, ...)                                                                                  \
    do {                                                                                                                   \
        if (!(condition)) {                                                                                                \
            utils::throwInvariantError(#condition, {__FILE__, __LINE__, __FUNCTION__}, FMT_STRING(format), ##__VA_ARGS__); \
        }                                                                                                                  \
    } while (false)
