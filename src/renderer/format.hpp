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
    auto messageSeverityMask = MaskType(vk::FlagTraits<FlagBitsType>::allFlags);
    std::size_t messageSeverityMaxLength = 0;
    while (messageSeverityMask != 0) {
        auto bit = (messageSeverityMask & (messageSeverityMask - 1)) ^ messageSeverityMask;
        std::size_t messageSeverityLength = fmt::formatted_size("{}", FlagBitsType{bit});
        if (messageSeverityMaxLength < messageSeverityLength) {
            messageSeverityMaxLength = messageSeverityLength;
        }
        messageSeverityMask &= messageSeverityMask - 1;
    }
    return messageSeverityMaxLength;
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
        auto rgb = fmt::rgb(255 * color[0], 255 * color[1], 255 * color[2]);
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
