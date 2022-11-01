#pragma once

#include <fmt/color.h>
#include <fmt/format.h>
#include <vulkan/vulkan.hpp>

#include <type_traits>
#include <utility>

#include <cstdint>

template<typename T>
struct fmt::formatter<T, char, std::void_t<decltype(vk::to_string(std::declval<const T &>()))>> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const T & value, FormatContext & ctx) const
    {
        return fmt::formatter<fmt::string_view>::format(vk::to_string(value), ctx);
    }
};

template<>
struct fmt::formatter<vk::DebugUtilsLabelEXT> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const vk::DebugUtilsLabelEXT & debugUtilsLabel, FormatContext & ctx) const
    {
        auto out = ctx.out();
        *out++ = '"';
        auto color = fmt::rgb(256 * debugUtilsLabel.color[0], 256 * debugUtilsLabel.color[1], 256 * debugUtilsLabel.color[2]);
        auto styled = fmt::styled<fmt::string_view>(debugUtilsLabel.pLabelName, fmt::fg(color));
        out = fmt::formatter<decltype(styled)>{}.format(styled, ctx);
        *out++ = '"';
        return out;
    }
};

template<>
struct fmt::formatter<vk::DebugUtilsObjectNameInfoEXT> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const vk::DebugUtilsObjectNameInfoEXT & debugUtilsObjectNameInfo, FormatContext & ctx) const
    {
        fmt::formatter<fmt::string_view>::format("object #", ctx);
        fmt::formatter<std::uint64_t>{}.format(debugUtilsObjectNameInfo.objectHandle, ctx);
        fmt::formatter<fmt::string_view>::format(" (type: ", ctx);
        fmt::formatter<vk::ObjectType>{}.format(debugUtilsObjectNameInfo.objectType, ctx);
        fmt::formatter<fmt::string_view>::format(")", ctx);
        if (debugUtilsObjectNameInfo.pObjectName) {
            fmt::formatter<fmt::string_view>::format(" name: \"", ctx);
            fmt::formatter<fmt::string_view>::format(debugUtilsObjectNameInfo.pObjectName, ctx);
            fmt::formatter<fmt::string_view>::format("\"", ctx);
        }
        return ctx.out();
    }
};
