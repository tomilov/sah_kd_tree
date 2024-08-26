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

struct ENGINE_EXPORT ComputePipeline final : utils::OneTime<ComputePipeline>
{
    vk::ComputePipelineCreateInfo computePipelineCreateInfo;

    ComputePipeline(std::string_view name, const Context & context, vk::PipelineCache pipelineCache, bool descriptorBufferEnabled, const PipelineLayout & pipelineLayout);

    void create();

    [[nodiscard]] bool getUseDescriptorBuffer() const
    {
        return descriptorBufferEnabled;
    }

    [[nodiscard]] vk::Pipeline getPipeline() const &
    {
        ASSERT(pipeline);
        return *pipeline;
    }

    [[nodiscard]] operator vk::Pipeline() const &  // NOLINT: google-explicit-constructor
    {
        return getPipeline();
    }

private:
    std::string name;
    const Context & context;
    const vk::PipelineCache pipelineCache;
    const bool descriptorBufferEnabled;

    vk::UniquePipeline pipeline;

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

}  // namespace engine
