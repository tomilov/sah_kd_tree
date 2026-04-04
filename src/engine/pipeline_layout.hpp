#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <string>
#include <string_view>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT PipelineLayout final : utils::OneTime<PipelineLayout>
{
    PipelineLayout(
        std::string_view name,
        const Context & context,
        const ShaderStages & shaderStages);

    [[nodiscard]] const ShaderStages & getShaderStages() const &
    {
        return shaderStages;
    }

    [[nodiscard]] vk::PipelineLayout getHandle() const &
    {
        SKT_ASSERT(pipelineLayout);
        return *pipelineLayout;
    }

    [[nodiscard]] operator vk::PipelineLayout() const &  // NOLINT: google-explicit-constructor
    {
        return getHandle();
    }

private:
    std::string name;
    const Context & context;
    const ShaderStages & shaderStages;

    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
    vk::UniquePipelineLayout pipelineLayout;

    void init();
};

}  // namespace engine
