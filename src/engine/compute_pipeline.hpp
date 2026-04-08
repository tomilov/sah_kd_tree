#pragma once

#include <engine/fwd.hpp>
#include <engine/specialization_info.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/name.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT ComputePipeline final : utils::OneTime<ComputePipeline>
{
    vk::StructureChain<vk::ComputePipelineCreateInfo, vk::PipelineCreateFlags2CreateInfo> computePipelineCreateInfoChain;

    ComputePipeline(
        utils::Name name,
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
        SKT_ASSERT(pipeline);
        return *pipeline;
    }

    [[nodiscard]] operator vk::Pipeline() const &  // NOLINT: google-explicit-constructor
    {
        return getHandle();
    }

private:
    utils::Name name;
    const Context & context;
    const vk::PipelineCache pipelineCache;
    const DescriptorManagementKind descriptorManagementKind;

    std::vector<vk::PipelineShaderStageCreateInfo> pipelineShaderStageCreateInfos;
    SpecializationInfos specializationInfos;
    vk::UniquePipeline pipeline;
};

}  // namespace engine
