#pragma once

#include <engine/fwd.hpp>
#include <engine/specialization_info.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
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

    ComputePipeline(std::string_view name, const Context & context, vk::PipelineCache pipelineCache, bool descriptorBufferEnabled, const PipelineLayout & pipelineLayout, SpecializationInfos && specializationInfos);

    void create();

    [[nodiscard]] bool getUseDescriptorBuffer() const
    {
        return descriptorBufferEnabled;
    }

    [[nodiscard]] vk::Pipeline getHandle() const &
    {
        ASSERT(pipeline);
        return *pipeline;
    }

    [[nodiscard]] operator vk::Pipeline() const &  // NOLINT: google-explicit-constructor
    {
        return getHandle();
    }

private:
    std::string name;
    const Context & context;
    const vk::PipelineCache pipelineCache;
    const bool descriptorBufferEnabled;

    SpecializationInfos specializationInfos;
    vk::UniquePipeline pipeline;

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

}  // namespace engine
