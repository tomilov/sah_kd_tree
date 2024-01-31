#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/library.hpp>
#include <engine/shader_module.hpp>
#include <utils/auto_cast.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <utility>

namespace engine
{

GraphicsPipelineLayout::GraphicsPipelineLayout(std::string_view name, const Context & context, const ShaderStages & shaderStages, vk::RenderPass renderPass)
    : name{name}, context{context}, library{context.getLibrary()}, device{context.getDevice()}, shaderStages{shaderStages}, renderPass{renderPass}
{
    init();
}

void GraphicsPipelineLayout::fill(std::string & name, vk::GraphicsPipelineCreateInfo & graphicsPipelineCreateInfo, bool useDescriptorBuffer) const
{
    name = this->name;

    graphicsPipelineCreateInfo.flags = {};
    if (useDescriptorBuffer) {
        graphicsPipelineCreateInfo.flags = vk::PipelineCreateFlagBits::eDescriptorBufferEXT;
    }

    graphicsPipelineCreateInfo.setStages(shaderStages.shaderStages.ref());
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
    graphicsPipelineCreateInfo.layout = pipelineLayout;
    graphicsPipelineCreateInfo.renderPass = renderPass;
    graphicsPipelineCreateInfo.subpass = 0;
    graphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
    graphicsPipelineCreateInfo.basePipelineIndex = 0;
}

void GraphicsPipelineLayout::init()
{
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
        .minDepthBounds = 0.0f,
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

    pipelineLayoutCreateInfo.flags = {};
    pipelineLayoutCreateInfo.setSetLayouts(shaderStages.descriptorSetLayouts);
    pipelineLayoutCreateInfo.setPushConstantRanges(shaderStages.pushConstantRanges);

    pipelineLayoutHolder = device.device.createPipelineLayoutUnique(pipelineLayoutCreateInfo, library.allocationCallbacks, library.dispatcher);
    pipelineLayout = *pipelineLayoutHolder;
    device.setDebugUtilsObjectName(pipelineLayout, name);
}

GraphicsPipelines::GraphicsPipelines(const Context & context, vk::PipelineCache pipelineCache) : context{context}, library{context.getLibrary()}, device{context.getDevice()}, pipelineCache{pipelineCache}
{}

void GraphicsPipelines::add(const GraphicsPipelineLayout & graphicsPipelineLayout, bool useDescriptorBuffer)
{
    graphicsPipelineLayout.fill(names.emplace_back(), graphicsPipelineCreateInfos.emplace_back(), useDescriptorBuffer);
}

void GraphicsPipelines::create()
{
    auto result = device.device.createGraphicsPipelinesUnique(pipelineCache, graphicsPipelineCreateInfos, library.allocationCallbacks, library.dispatcher);
    INVARIANT(result.result == vk::Result::eSuccess, "Failed to create graphics pipelines {}", fmt::join(names, ", "));
    pipelineHolders = std::move(result.value);
    pipelines.reserve(std::size(pipelineHolders));
    size_t i = 0;
    for (const auto & pipelineHolder : pipelineHolders) {
        pipelines.push_back(*pipelineHolder);
        device.setDebugUtilsObjectName(pipelines.back(), names.at(i));
        ++i;
    }
}

}  // namespace engine
