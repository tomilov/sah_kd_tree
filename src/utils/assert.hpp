#pragma once

#include <common/config.hpp>

#include <fmt/format.h>

#include <source_location>

#include <utils/utils_export.h>

namespace utils
{

void vAssertFailed [[noreturn]] (bool assert, const char * expression, std::source_location sourceLocation, fmt::string_view format, fmt::format_args args) UTILS_EXPORT;

template<typename... Args>
void assertFailed [[noreturn]] (bool assert, const char * expression, std::source_location sourceLocation, fmt::format_string<Args...> format, Args &&... args)
{
    vAssertFailed(assert, expression, sourceLocation, format, fmt::make_format_args(args...));
}

}  // namespace utils

#define ASSERT_MSG_SRCLOC(condition, srcLoc, format, ...)                                           \
    do {                                                                                            \
        if constexpr (sah_kd_tree::kIsDebugBuild) {                                                 \
            if (condition) {                                                                        \
            } else {                                                                                \
                ::utils::assertFailed(true, #condition, srcLoc, FMT_STRING(format), ##__VA_ARGS__); \
            }                                                                                       \
        }                                                                                           \
    } while (false)

#define ASSERT_MSG(condition, format, ...) ASSERT_MSG_SRCLOC(condition, std::source_location::current(), format, ##__VA_ARGS__)

#define ASSERT_SRCLOC(condition, srcLoc) ASSERT_MSG_SRCLOC(condition, srcLoc, "")
#define ASSERT(condition) ASSERT_MSG(condition, "")

#define INVARIANT_SRCLOC(condition, srcLoc, format, ...)                                                              \
    do {                                                                                                              \
        if (condition) {                                                                                              \
        } else {                                                                                                      \
            ::utils::assertFailed(sah_kd_tree::kIsDebugBuild, #condition, srcLoc, FMT_STRING(format), ##__VA_ARGS__); \
        }                                                                                                             \
    } while (false)

#define INVARIANT(condition, format, ...) INVARIANT_SRCLOC(condition, std::source_location::current(), format, ##__VA_ARGS__)
