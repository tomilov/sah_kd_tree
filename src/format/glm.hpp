#pragma once

#include <fmt/base.h>
#include <glm/fwd.hpp>
#include <glm/gtx/string_cast.hpp>

template<>
struct fmt::formatter<glm::mat4> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const glm::mat4 & m, FormatContext & ctx) const
    {
        return fmt::formatter<fmt::string_view>::format(glm::to_string(m), ctx);
    }
};

template<>
struct fmt::formatter<glm::quat> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const glm::quat & q, FormatContext & ctx) const
    {
        return fmt::formatter<fmt::string_view>::format(glm::to_string(q), ctx);
    }
};

template<>
struct fmt::formatter<glm::vec3> : fmt::formatter<fmt::string_view>
{
    template<typename FormatContext>
    auto format(const glm::vec3 & v, FormatContext & ctx) const
    {
        return fmt::formatter<fmt::string_view>::format(glm::to_string(v), ctx);
    }
};
