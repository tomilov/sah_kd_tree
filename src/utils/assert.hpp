#pragma once

#include <common/config.hpp>

#include <fmt/format.h>

#include <source_location>

#include <utils/utils_export.h>

namespace utils
{

void vAssertFailed [[noreturn]] (
    bool assert,
    const char * expression,
    std::source_location sourceLocation,
    fmt::string_view format,
    fmt::format_args args) UTILS_EXPORT;

template<typename... Args>
void assertFailed [[noreturn]] (
    bool assert,
    const char * expression,
    std::source_location sourceLocation,
    fmt::format_string<Args...> format,
    Args &&... args)
{
    vAssertFailed(assert, expression, sourceLocation, format, fmt::make_format_args(args...));
}

}  // namespace utils

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif

#define SKT_ASSERT_MSG_SRCLOC(condition, srcLoc, format, ...)                                       \
    do {                                                                                            \
        if constexpr (sah_kd_tree::kIsDebugBuild) {                                                 \
            if (!(condition)) [[unlikely]] {                                                        \
                ::utils::assertFailed(true, #condition, srcLoc, FMT_STRING(format), ##__VA_ARGS__); \
            }                                                                                       \
        }                                                                                           \
    } while (false)

#define SKT_ASSERT_MSG(condition, format, ...) SKT_ASSERT_MSG_SRCLOC(condition, std::source_location::current(), format, ##__VA_ARGS__)
#define SKT_ASSERT_SRCLOC(condition, srcLoc) SKT_ASSERT_MSG_SRCLOC(condition, srcLoc, "")
#define SKT_ASSERT(condition) SKT_ASSERT_MSG(condition, "")

#define SKT_INVARIANT_SRCLOC(condition, srcLoc, format, ...)                                                          \
    do {                                                                                                              \
        if (!(condition)) [[unlikely]] {                                                                              \
            ::utils::assertFailed(sah_kd_tree::kIsDebugBuild, #condition, srcLoc, FMT_STRING(format), ##__VA_ARGS__); \
        }                                                                                                             \
    } while (false)

#define SKT_INVARIANT(condition, format, ...) SKT_INVARIANT_SRCLOC(condition, std::source_location::current(), format, ##__VA_ARGS__)

#ifdef __clang__
#pragma clang diagnostic pop
#endif
