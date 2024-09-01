#include <engine/compute_pipeline.hpp>
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/pipeline_layout.hpp>
#include <engine/shader_module.hpp>

#include <iterator>
#include <utility>

namespace engine
{

ComputePipeline::ComputePipeline(std::string_view name, const Context & context, vk::PipelineCache pipelineCache, bool descriptorBufferEnabled, const PipelineLayout & pipelineLayout)
    : name{name}
    , context{context}
    , pipelineCache{pipelineCache}
    , descriptorBufferEnabled{descriptorBufferEnabled}
{
    const ShaderStages & shaderStages = pipelineLayout.getShaderStages();
    computePipelineCreateInfo.flags = {};  // TODO: eDispatchBase?
    if (descriptorBufferEnabled) {
        computePipelineCreateInfo.flags |= vk::PipelineCreateFlagBits::eDescriptorBufferEXT;
    }
    computePipelineCreateInfo.layout = pipelineLayout;
    INVARIANT(std::size(shaderStages.pipelineShaderStageCreateInfos) == 1, "{}", std::size(shaderStages.pipelineShaderStageCreateInfos));
    computePipelineCreateInfo.stage = shaderStages.pipelineShaderStageCreateInfos.at(0);
    computePipelineCreateInfo.stage.setPSpecializationInfo(&specializationInfo);
}

void ComputePipeline::create()
{
    auto result = context.getDevice().getDevice().createComputePipelineUnique(pipelineCache, computePipelineCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());
    INVARIANT(result.result == vk::Result::eSuccess, "Failed to create compute pipeline {}", name);
    pipeline = std::move(result.value);
    context.getDevice().setDebugUtilsObjectName(*pipeline, name);
}

}  // namespace engine
