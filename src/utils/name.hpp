#pragma once

#include <utils/noncopyable.hpp>

#include <fmt/format.h>

#include <string>
#include <string_view>
#include <utility>

namespace utils
{

class Name : public OneTime<Name>
{
public:
    template<typename... Args>
    explicit Name(
        fmt::format_string<Args...> format,
        Args &&... args)
    {
        fmt::vformat_to(fmt::appender(memoryBuffer), format, fmt::make_format_args(args...));
        memoryBuffer.push_back('\0');
    }

    [[nodiscard]] std::string_view toStdStringView() const
    {
        return {memoryBuffer.data(), memoryBuffer.size()};
    }

    [[nodiscard]] fmt::string_view toFmtStringView() const
    {
        return {memoryBuffer.data(), memoryBuffer.size()};
    }

    [[nodiscard]] std::string toStdString() const
    {
        return {memoryBuffer.data(), memoryBuffer.size()};
    }

    [[nodiscard]] const char * toCStr() const &
    {
        return memoryBuffer.data();
    }

    [[nodiscard]] Name clone() const
    {
        fmt::memory_buffer memoryBufferCopy;
        memoryBufferCopy.append(memoryBuffer);
        return Name{std::move(memoryBufferCopy)};
    }

    [[nodiscard]] bool isEmpty() const
    {
        return memoryBuffer.size() == 0;
    }

private:
    fmt::memory_buffer memoryBuffer;

    explicit Name(fmt::memory_buffer memoryBuffer)
        : memoryBuffer{std::move(memoryBuffer)}
    {}
};

}  // namespace utils

template<>
struct fmt::formatter<utils::Name> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(
        const utils::Name & name,
        FormatContext & ctx) const
    {
        return fmt::formatter<fmt::string_view>::format(name.toFmtStringView(), ctx);
    }
};
