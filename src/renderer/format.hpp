#pragma once

#include <fmt/color.h>
#include <fmt/format.h>
#include <vulkan/vulkan.hpp>

#include <limits>
#include <type_traits>
#include <utility>

#include <cstdint>

template<typename FlagBitsType>
std::size_t getFlagBitsMaxNameLength()
{
    using MaskType = typename vk::Flags<FlagBitsType>::MaskType;
    auto mask = MaskType(vk::FlagTraits<FlagBitsType>::allFlags);
    std::size_t maxLength = 0;
    while (mask != 0) {
        auto bit = (mask & (mask - 1)) ^ mask;
        std::size_t length = fmt::formatted_size("{}", FlagBitsType{bit});
        if (maxLength < length) {
            maxLength = length;
        }
        mask &= mask - 1;
    }
    return maxLength;
}

template<typename T>
struct fmt::formatter<T, char, std::void_t<decltype(vk::to_string(std::declval<const T &>()))>> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const T & value, FormatContext & ctx) const
    {
        return fmt::formatter<fmt::string_view>::format(vk::to_string(value), ctx);
    }
};

template<typename FlagBitsType>
struct fmt::formatter<vk::Flags<FlagBitsType>, char> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const vk::Flags<FlagBitsType> & flags, FormatContext & ctx) const
    {
        using FlagTraits = vk::FlagTraits<FlagBitsType>;
        static_assert(FlagTraits::isBitmask);
        auto out = ctx.out();
        if (!flags) {
            return out;
        }
        using MaskType = typename vk::Flags<FlagBitsType>::MaskType;
        constexpr auto allFlags = static_cast<MaskType>(FlagTraits::allFlags);
        auto mask = static_cast<MaskType>(flags);
        while (mask != 0) {
            const auto bit = (mask & (mask - 1)) ^ mask;
            if ((~allFlags & bit) == 0) {
                out = fmt::format_to(out, "{}", FlagBitsType{bit});
            } else {
                out = fmt::format_to(out, "{:#x}", bit);
            }
            mask &= mask - 1;
            if (mask != 0) {
                *out++ = '|';
            }
        }
        return out;
    }
};

template<>
struct fmt::formatter<vk::DebugUtilsLabelEXT> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const vk::DebugUtilsLabelEXT & debugUtilsLabel, FormatContext & ctx) const
    {
        auto color = debugUtilsLabel.color;
        auto clamp = [](float color) -> uint8_t { return std::floor(std::numeric_limits<uint8_t>::max() * std::clamp(color, 0.0f, 1.0f)); };
        auto rgb = fmt::rgb(clamp(color[0]), clamp(color[1]), clamp(color[2]));
        auto styledLabelName = fmt::styled<fmt::string_view>(debugUtilsLabel.pLabelName, fmt::fg(rgb));
        return fmt::format_to(ctx.out(), "'{}'", styledLabelName);
    }
};

template<>
struct fmt::formatter<vk::DebugUtilsObjectNameInfoEXT> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const vk::DebugUtilsObjectNameInfoEXT & debugUtilsObjectNameInfo, FormatContext & ctx) const
    {
        auto objectName = debugUtilsObjectNameInfo.pObjectName;
        return fmt::format_to(ctx.out(), "{{ handle = {:#x}, type = {}, name = '{}' }}", debugUtilsObjectNameInfo.objectHandle, debugUtilsObjectNameInfo.objectType, objectName ? objectName : "");
    }
};
