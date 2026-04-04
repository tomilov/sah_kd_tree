#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/pipeline_layout.hpp>
#include <engine/shader_module.hpp>

#include <iterator>

template struct utils::OneTime<engine::PipelineLayout>::CheckTraits;

namespace engine
{

PipelineLayout::PipelineLayout(
    std::string_view nameIn,
    const Context & contextIn,
    const ShaderStages & shaderStagesIn)
    : name{nameIn}
    , context{contextIn}
    , shaderStages{shaderStagesIn}
{
    init();
}

void PipelineLayout::init()
{
    SKT_ASSERT(!std::empty(name));

    pipelineLayoutCreateInfo.flags = {};
    pipelineLayoutCreateInfo.setSetLayouts(shaderStages.descriptorSetLayouts);
    pipelineLayoutCreateInfo.setPushConstantRanges(shaderStages.pushConstantRanges);

    pipelineLayout = context.getDevice().getHandle().createPipelineLayoutUnique(pipelineLayoutCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());
    context.getDevice().setDebugUtilsObjectName(*pipelineLayout, name);
}

}  // namespace engine
