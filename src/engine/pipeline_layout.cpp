#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/pipeline_layout.hpp>
#include <engine/shaders.hpp>

template struct utils::OneTime<engine::PipelineLayout>::CheckTraits;

namespace engine
{

PipelineLayout::PipelineLayout(
    utils::Name nameIn,
    const Context & contextIn,
    const ShaderStages & shaderStagesIn)
    : name{std::move(nameIn)}
    , context{contextIn}
    , shaderStages{shaderStagesIn}
{
    init();
}

void PipelineLayout::init()
{
    SKT_ASSERT(!name.isEmpty());

    pipelineLayoutCreateInfo.flags = {};
    pipelineLayoutCreateInfo.setSetLayouts(shaderStages.descriptorSetLayouts);
    pipelineLayoutCreateInfo.setPushConstantRanges(shaderStages.pushConstantRanges);

    pipelineLayout = context.getDevice().getHandle().createPipelineLayoutUnique(pipelineLayoutCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());
    context.getDevice().setDebugUtilsObjectName(*pipelineLayout, name.toCStr());
}

}  // namespace engine
