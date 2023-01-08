#include <engine/device.hpp>
#include <engine/engine.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/library.hpp>
#include <engine/shader_module.hpp>
#include <utils/auto_cast.hpp>

#include <fmt/format.h>

#include <utility>

namespace engine
{

GraphicsPipelineLayout::GraphicsPipelineLayout(std::string_view name, const Engine & engine, const PipelineVertexInputState & pipelineVertexInputState, const ShaderStages & shaderStages, vk::RenderPass renderPass,
                                               const std::vector<vk::DescriptorSetLayout> & descriptorSetLayouts, const std::vector<vk::PushConstantRange> & pushConstantRanges)
    : name{name}
    , engine{engine}
    , library{*engine.library}
    , device{*engine.device}
    , pipelineVertexInputState{pipelineVertexInputState}
    , shaderStages{shaderStages}
    , renderPass{renderPass}
    , descriptorSetLayouts{descriptorSetLayouts}
    , pushConstantRanges{pushConstantRanges}
{
    init();
}

void GraphicsPipelineLayout::fill(std::string & name, vk::GraphicsPipelineCreateInfo & graphicsPipelineCreateInfo) const
{
    name = this->name;

    graphicsPipelineCreateInfo.flags = {};
    graphicsPipelineCreateInfo.setStages(shaderStages.shaderStages.ref());
    graphicsPipelineCreateInfo.pVertexInputState = &pipelineVertexInputState.pipelineVertexInputStateCreateInfo.value();
    graphicsPipelineCreateInfo.pInputAssemblyState = &pipelineInputAssemblyStateCreateInfo;
    graphicsPipelineCreateInfo.pTessellationState = nullptr;
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
    pipelineInputAssemblyStateCreateInfo.setTopology(vk::PrimitiveTopology::eTriangleStrip);

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
        .dstColorBlendFactor = vk::BlendFactor::eOne,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstAlphaBlendFactor = vk::BlendFactor::eOne,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };

    pipelineMultisampleStateCreateInfo = {
        .flags = {},
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.0f,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE,
    };

    pipelineDepthStencilStateCreateInfo = {
        .flags = {},
        .depthTestEnable = VK_FALSE,
        .depthWriteEnable = VK_FALSE,
        .depthCompareOp = vk::CompareOp::eNever,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
        .front = {},
        .back = {},
        .minDepthBounds = 0.0f,
        .maxDepthBounds = 0.0f,
    };

    pipelineColorBlendStateCreateInfo.flags = {};
    pipelineColorBlendStateCreateInfo.logicOpEnable = VK_FALSE;
    pipelineColorBlendStateCreateInfo.logicOp = vk::LogicOp::eCopy;
    pipelineColorBlendStateCreateInfo.setAttachments(pipelineColorBlendAttachmentState);
    pipelineColorBlendStateCreateInfo.blendConstants = {{0.0f, 0.0f, 0.0f, 0.0f}};

    dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };
    pipelineDynamicStateCreateInfo.setDynamicStates(dynamicStates);

    pipelineLayoutCreateInfo.flags = {};
    pipelineLayoutCreateInfo.setSetLayouts(descriptorSetLayouts);
    pipelineLayoutCreateInfo.setPushConstantRanges(pushConstantRanges);

    pipelineLayoutHolder = device.device.createPipelineLayoutUnique(pipelineLayoutCreateInfo, library.allocationCallbacks, library.dispatcher);
    pipelineLayout = *pipelineLayoutHolder;
    device.setDebugUtilsObjectName(pipelineLayout, name);
}

GraphicsPipelines::GraphicsPipelines(const Engine & engine, vk::PipelineCache pipelineCache) : engine{engine}, library{*engine.library}, device{*engine.device}, pipelineCache{pipelineCache}
{}

void GraphicsPipelines::add(const GraphicsPipelineLayout & graphicsPipelineLayout)
{
    graphicsPipelineLayout.fill(names.emplace_back(), graphicsPipelineCreateInfos.emplace_back());
}

void GraphicsPipelines::create()
{
    auto result = device.device.createGraphicsPipelinesUnique(pipelineCache, graphicsPipelineCreateInfos, library.allocationCallbacks, library.dispatcher);
    INVARIANT(result.result == vk::Result::eSuccess, "Failed to create graphics pipelines {}", fmt::join(names, ","));
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
