#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/name.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT PipelineLayout final : utils::OneTime<PipelineLayout>
{
    PipelineLayout(
        utils::Name name,
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
    utils::Name name;
    const Context & context;
    const ShaderStages & shaderStages;

    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
    vk::UniquePipelineLayout pipelineLayout;

    void init();
};

}  // namespace engine
