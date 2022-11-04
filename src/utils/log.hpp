#pragma once

#include <utils/fast_pimpl.hpp>
#include <utils/utils_export.h>

#include <fmt/format.h>

#include <string_view>

namespace utils
{
class UTILS_EXPORT LoggableBase
{
public:
    enum class LogLevel
    {
        Debug,
        Info,
        Warning,
        Critical,
    };

    LoggableBase();
    ~LoggableBase();

    void setLogLevel(LogLevel logLevel);
    bool checkLogLevel(LogLevel logLevel) const;

    void log(std::string_view message, LogLevel logLevel) const;

    template<typename... Args>
    void log(LogLevel logLevel, fmt::string_view format, const Args &... args) const
    {
        log(fmt::vformat(format, fmt::make_format_args(args...)), logLevel);
    }

private:
    struct Impl;

    static constexpr std::size_t kSize = 4;
    static constexpr std::size_t kAlignment = 4;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;
};

}  // namespace utils
