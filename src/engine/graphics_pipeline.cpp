#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/pipeline_layout.hpp>
#include <engine/shader_module.hpp>
#include <utils/assert.hpp>

#include <fmt/ranges.h>

#include <iterator>
#include <utility>

namespace engine
{

GraphicsPipeline::GraphicsPipeline(std::string_view name, const Context & context, vk::PipelineCache pipelineCache, bool descriptorBufferEnabled, const PipelineLayout & pipelineLayout, vk::RenderPass renderPass,
                                   SpecializationInfos && specializationInfosIn)
    : name{name}
    , context{context}
    , pipelineCache{pipelineCache}
    , descriptorBufferEnabled{descriptorBufferEnabled}
    , renderPass{renderPass}
    , specializationInfos{std::move(specializationInfosIn)}
{
    ASSERT(renderPass);

    pipelineInputAssemblyStateCreateInfo.flags = {};
    pipelineInputAssemblyStateCreateInfo.setPrimitiveRestartEnable(vk::False);
    pipelineInputAssemblyStateCreateInfo.setTopology(vk::PrimitiveTopology::eTriangleList);

    pipelineViewportStateCreateInfo.flags = {};
    pipelineViewportStateCreateInfo.setViewportCount(1);
    pipelineViewportStateCreateInfo.setScissorCount(1);

    pipelineRasterizationStateCreateInfo = {
        .flags = {},
        .depthClampEnable = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable = vk::False,
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = 0.0f,
        .lineWidth = 1.0f,
    };

    pipelineColorBlendAttachmentState = {
        .blendEnable = vk::True,
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
        .sampleShadingEnable = vk::False,
        .minSampleShading = 0.0f,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = vk::False,
        .alphaToOneEnable = vk::False,
    };

    pipelineDepthStencilStateCreateInfo = {
        .flags = {},
        .depthTestEnable = vk::True,
        .depthWriteEnable = vk::True,
        .depthCompareOp = vk::CompareOp::eLess,
        .depthBoundsTestEnable = vk::False,
        .stencilTestEnable = vk::False,
        .front = {},
        .back = {},
        .minDepthBounds = engine::kMinDepth,
        .maxDepthBounds = 1.0f,
    };

    pipelineColorBlendStateCreateInfo = {
        .flags = {},
        .logicOpEnable = vk::False,
        .logicOp = vk::LogicOp::eCopy,
        .blendConstants = {{0.0f, 0.0f, 0.0f, 0.0f}},
    };
    pipelineColorBlendStateCreateInfo.setAttachments(pipelineColorBlendAttachmentState);

    dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };
    pipelineDynamicStateCreateInfo.setDynamicStates(dynamicStates);

    graphicsPipelineCreateInfo.flags = {};
    if (descriptorBufferEnabled) {
        graphicsPipelineCreateInfo.flags |= vk::PipelineCreateFlagBits::eDescriptorBufferEXT;
    }
    const ShaderStages & shaderStages = pipelineLayout.getShaderStages();
    pipelineShaderStageCreateInfos = shaderStages.pipelineShaderStageCreateInfos;
    INVARIANT(std::size(specializationInfos) <= std::size(pipelineShaderStageCreateInfos), "");
    for (vk::PipelineShaderStageCreateInfo & pipelineShaderStageCreateInfo : pipelineShaderStageCreateInfos) {
        auto specializationInfo = specializationInfos.find(pipelineShaderStageCreateInfo.stage);
        if (specializationInfo != std::cend(specializationInfos)) {
            pipelineShaderStageCreateInfo.setPSpecializationInfo(&specializationInfo->second.getSpecializationInfo());
        }
    }
    graphicsPipelineCreateInfo.setStages(pipelineShaderStageCreateInfos);
    if (shaderStages.vertexInputState) {
        graphicsPipelineCreateInfo.pVertexInputState = &shaderStages.vertexInputState->pipelineVertexInputStateCreateInfo;
    }
    graphicsPipelineCreateInfo.pInputAssemblyState = &pipelineInputAssemblyStateCreateInfo;
    graphicsPipelineCreateInfo.pTessellationState = nullptr;
    graphicsPipelineCreateInfo.pViewportState = &pipelineViewportStateCreateInfo;
    graphicsPipelineCreateInfo.pRasterizationState = &pipelineRasterizationStateCreateInfo;
    graphicsPipelineCreateInfo.pMultisampleState = &pipelineMultisampleStateCreateInfo;
    graphicsPipelineCreateInfo.pDepthStencilState = &pipelineDepthStencilStateCreateInfo;
    graphicsPipelineCreateInfo.pColorBlendState = &pipelineColorBlendStateCreateInfo;
    graphicsPipelineCreateInfo.pDynamicState = &pipelineDynamicStateCreateInfo;
    graphicsPipelineCreateInfo.layout = pipelineLayout;
    graphicsPipelineCreateInfo.renderPass = renderPass;
    graphicsPipelineCreateInfo.subpass = 0;
    graphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
    graphicsPipelineCreateInfo.basePipelineIndex = 0;
}

void GraphicsPipeline::create()
{
    auto result = context.getDevice().getDevice().createGraphicsPipelineUnique(pipelineCache, graphicsPipelineCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());
    INVARIANT(result.result == vk::Result::eSuccess, "Failed to create graphics pipeline {}", name);
    pipeline = std::move(result.value);
    context.getDevice().setDebugUtilsObjectName(*pipeline, name);
}

}  // namespace engine
