#include <engine/compute_pipeline.hpp>
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/pipeline_layout.hpp>
#include <engine/shader_module.hpp>
#include <engine/specialization_info.hpp>
#include <format/vulkan.hpp>

#include <iterator>
#include <utility>

template struct utils::OneTime<engine::ComputePipeline>::CheckTraits;

namespace engine
{

ComputePipeline::ComputePipeline(
    std::string_view nameIn,
    const Context & contextIn,
    vk::PipelineCache pipelineCacheIn,
    DescriptorManagementKind descriptorManagementKindIn,
    const PipelineLayout & pipelineLayout,
    SpecializationInfos && specializationInfosIn)
    : name{nameIn}
    , context{contextIn}
    , pipelineCache{pipelineCacheIn}
    , descriptorManagementKind{descriptorManagementKindIn}
    , specializationInfos{std::move(specializationInfosIn)}
{
    {
        auto & pipelineCreateFlags2CreateInfo = computePipelineCreateInfoChain.get<vk::PipelineCreateFlags2CreateInfo>();
        switch (descriptorManagementKind) {
        case DescriptorManagementKind::Sets: {
            break;
        }
        case DescriptorManagementKind::Buffer: {
            pipelineCreateFlags2CreateInfo.flags |= vk::PipelineCreateFlagBits2::eDescriptorBufferEXT;
            break;
        }
        case DescriptorManagementKind::Heap: {
            pipelineCreateFlags2CreateInfo.flags |= vk::PipelineCreateFlagBits2::eDescriptorHeapEXT;
            break;
        }
        }
    }
    auto & computePipelineCreateInfo = computePipelineCreateInfoChain.get<vk::ComputePipelineCreateInfo>();
    const ShaderStages & shaderStages = pipelineLayout.getShaderStages();
    SKT_INVARIANT(std::size(shaderStages.pipelineShaderStageCreateInfos) == 1, "{}", std::size(shaderStages.pipelineShaderStageCreateInfos));
    if (descriptorManagementKind != DescriptorManagementKind::Heap) {
        computePipelineCreateInfo.layout = pipelineLayout;
    }
    computePipelineCreateInfo.stage = shaderStages.pipelineShaderStageCreateInfos.at(0);
    SKT_INVARIANT(computePipelineCreateInfo.stage.stage == vk::ShaderStageFlagBits::eCompute, "{}", computePipelineCreateInfo.stage.stage);
    if (!std::empty(specializationInfos)) {
        computePipelineCreateInfo.stage.setPSpecializationInfo(&specializationInfos.at(vk::ShaderStageFlagBits::eCompute).getSpecializationInfo());
    }
}

void ComputePipeline::create()
{
    auto & computePipelineCreateInfo = computePipelineCreateInfoChain.get<vk::ComputePipelineCreateInfo>();
    auto result = context.getDevice().getHandle().createComputePipelineUnique(pipelineCache, computePipelineCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());
    SKT_INVARIANT(result.result == vk::Result::eSuccess, "Failed to create compute pipeline {}", name);
    pipeline = std::move(result.value);
    context.getDevice().setDebugUtilsObjectName(*pipeline, name);
}

}  // namespace engine
