#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/pipeline_layout.hpp>
#include <engine/shader_module.hpp>

#include <iterator>

namespace engine
{

PipelineLayout::PipelineLayout(std::string_view name, const Context & context, const ShaderStages & shaderStages)
    : name{name}
    , context{context}
    , shaderStages{shaderStages}
{
    init();
}

void PipelineLayout::init()
{
    ASSERT(!std::empty(name));

    pipelineLayoutCreateInfo.flags = {};
    pipelineLayoutCreateInfo.setSetLayouts(shaderStages.descriptorSetLayouts);
    pipelineLayoutCreateInfo.setPushConstantRanges(shaderStages.pushConstantRanges);

    pipelineLayout = context.getDevice().getHandle().createPipelineLayoutUnique(pipelineLayoutCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());
    context.getDevice().setDebugUtilsObjectName(*pipelineLayout, name);
}

}  // namespace engine
