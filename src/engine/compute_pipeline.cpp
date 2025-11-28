#include <engine/compute_pipeline.hpp>
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/pipeline_layout.hpp>
#include <engine/shader_module.hpp>
#include <engine/specialization_info.hpp>
#include <format/vulkan.hpp>

#include <iterator>
#include <utility>

namespace engine
{

ComputePipeline::ComputePipeline(std::string_view name, const Context & context, vk::PipelineCache pipelineCache, bool descriptorBufferEnabled, const PipelineLayout & pipelineLayout, SpecializationInfos && specializationInfosIn)
    : name{name}
    , context{context}
    , pipelineCache{pipelineCache}
    , descriptorBufferEnabled{descriptorBufferEnabled}
    , specializationInfos{std::move(specializationInfosIn)}
{
    computePipelineCreateInfo.flags = {};  // TODO: eDispatchBase?
    if (descriptorBufferEnabled) {
        computePipelineCreateInfo.flags |= vk::PipelineCreateFlagBits::eDescriptorBufferEXT;
    }
    const ShaderStages & shaderStages = pipelineLayout.getShaderStages();
    computePipelineCreateInfo.layout = pipelineLayout;
    INVARIANT(std::size(shaderStages.pipelineShaderStageCreateInfos) == 1, "{}", std::size(shaderStages.pipelineShaderStageCreateInfos));
    computePipelineCreateInfo.stage = shaderStages.pipelineShaderStageCreateInfos.at(0);
    INVARIANT(computePipelineCreateInfo.stage.stage == vk::ShaderStageFlagBits::eCompute, "{}", computePipelineCreateInfo.stage.stage);
    if (!std::empty(specializationInfos)) {
        computePipelineCreateInfo.stage.setPSpecializationInfo(&specializationInfos.at(vk::ShaderStageFlagBits::eCompute).getSpecializationInfo());
    }
}

void ComputePipeline::create()
{
    auto result = context.getDevice().getDevice().createComputePipelineUnique(pipelineCache, computePipelineCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());
    INVARIANT(result.result == vk::Result::eSuccess, "Failed to create compute pipeline {}", name);
    pipeline = std::move(result.value);
    context.getDevice().setDebugUtilsObjectName(*pipeline, name);
}

}  // namespace engine
