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

    ComputePipeline(
        std::string_view name,
        const Context & context,
        vk::PipelineCache pipelineCache,
        DescriptorManagementKind descriptorManagementKind,
        const PipelineLayout & pipelineLayout,
        SpecializationInfos && specializationInfos);

    void create();

    [[nodiscard]] DescriptorManagementKind getDescriptorManagementKind() const
    {
        return descriptorManagementKind;
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
    const DescriptorManagementKind descriptorManagementKind;

    SpecializationInfos specializationInfos;
    vk::UniquePipeline pipeline;
};

}  // namespace engine

template struct utils::OneTime<engine::ComputePipeline>::CheckTraits;
