#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/shader_module.hpp>
#include <utils/assert.hpp>

#include <fmt/ranges.h>

#include <optional>
#include <utility>

namespace engine
{

GraphicsPipelineLayout::GraphicsPipelineLayout(std::string_view name, const Context & context, const ShaderStages & shaderStages)
    : name{name}
    , context{context}
    , shaderStages{std::move(shaderStages)}
{
    init();
}

void GraphicsPipelineLayout::init()
{
    ASSERT(!std::empty(name));

    pipelineLayoutCreateInfo.flags = {};
    pipelineLayoutCreateInfo.setSetLayouts(shaderStages.descriptorSetLayouts);
    pipelineLayoutCreateInfo.setPushConstantRanges(shaderStages.pushConstantRanges);

    pipelineLayout = context.getDevice().getDevice().createPipelineLayoutUnique(pipelineLayoutCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());
    context.getDevice().setDebugUtilsObjectName(*pipelineLayout, name);
}

GraphicsPipeline::GraphicsPipeline(std::string_view name, const Context & context, vk::PipelineCache pipelineCache, bool descriptorBufferEnabled, const GraphicsPipelineLayout & graphicsPipelineLayout, vk::RenderPass renderPass)
    : name{name}
    , context{context}
    , pipelineCache{pipelineCache}
    , descriptorBufferEnabled{descriptorBufferEnabled}
    , renderPass{renderPass}
{
    ASSERT(renderPass);

    pipelineInputAssemblyStateCreateInfo.flags = {};
    pipelineInputAssemblyStateCreateInfo.setPrimitiveRestartEnable(VK_FALSE);
    pipelineInputAssemblyStateCreateInfo.setTopology(vk::PrimitiveTopology::eTriangleList);

    pipelineViewportStateCreateInfo.flags = {};
    pipelineViewportStateCreateInfo.setViewportCount(1);
    pipelineViewportStateCreateInfo.setScissorCount(1);

    pipelineRasterizationStateCreateInfo = {
        .flags = {},
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable = VK_FALSE,
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = 0.0f,
        .lineWidth = 1.0f,
    };

    pipelineColorBlendAttachmentState = {
        .blendEnable = VK_TRUE,
        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };

    pipelineMultisampleStateCreateInfo = {
        .flags = {},
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 0.0f,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE,
    };

    pipelineDepthStencilStateCreateInfo = {
        .flags = {},
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = vk::CompareOp::eLess,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
        .front = {},
        .back = {},
        .minDepthBounds = engine::kMinDepth,
        .maxDepthBounds = 1.0f,
    };

    pipelineColorBlendStateCreateInfo = {
        .flags = {},
        .logicOpEnable = VK_FALSE,
        .logicOp = vk::LogicOp::eCopy,
        .blendConstants = {{0.0f, 0.0f, 0.0f, 0.0f}},
    };
    pipelineColorBlendStateCreateInfo.setAttachments(pipelineColorBlendAttachmentState);

    dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };
    pipelineDynamicStateCreateInfo.setDynamicStates(dynamicStates);

    const ShaderStages & shaderStages = graphicsPipelineLayout.getShaderStages();
    graphicsPipelineCreateInfo.flags = {};
    if (descriptorBufferEnabled) {
        graphicsPipelineCreateInfo.flags |= vk::PipelineCreateFlagBits::eDescriptorBufferEXT;
    }
    graphicsPipelineCreateInfo.setStages(shaderStages.pipelineShaderStageCreateInfos);
    if (shaderStages.vertexInputState) {
        graphicsPipelineCreateInfo.pVertexInputState = &shaderStages.vertexInputState.value().pipelineVertexInputStateCreateInfo;
    }
    graphicsPipelineCreateInfo.pInputAssemblyState = &pipelineInputAssemblyStateCreateInfo;
    graphicsPipelineCreateInfo.pTessellationState = nullptr;
    graphicsPipelineCreateInfo.pViewportState = &pipelineViewportStateCreateInfo;
    graphicsPipelineCreateInfo.pRasterizationState = &pipelineRasterizationStateCreateInfo;
    graphicsPipelineCreateInfo.pMultisampleState = &pipelineMultisampleStateCreateInfo;
    graphicsPipelineCreateInfo.pDepthStencilState = &pipelineDepthStencilStateCreateInfo;
    graphicsPipelineCreateInfo.pColorBlendState = &pipelineColorBlendStateCreateInfo;
    graphicsPipelineCreateInfo.pDynamicState = &pipelineDynamicStateCreateInfo;
    graphicsPipelineCreateInfo.layout = graphicsPipelineLayout;
    graphicsPipelineCreateInfo.renderPass = renderPass;
    graphicsPipelineCreateInfo.subpass = 0;
    graphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
    graphicsPipelineCreateInfo.basePipelineIndex = 0;
}

void GraphicsPipeline::create()
{
    auto result = context.getDevice().getDevice().createGraphicsPipelinesUnique(pipelineCache, graphicsPipelineCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());
    INVARIANT(result.result == vk::Result::eSuccess, "Failed to create graphics pipeline {}", name);
    pipeline = std::move(result.value.at(0));
    context.getDevice().setDebugUtilsObjectName(*pipeline, name);
}

}  // namespace engine
