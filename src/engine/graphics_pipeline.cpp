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

GraphicsPipelines::GraphicsPipelines(std::string_view name, const Engine & engine, const ShaderStages & shaderStages, vk::RenderPass renderPass, vk::PipelineCache pipelineCache, const std::vector<vk::DescriptorSetLayout> & descriptorSetLayouts,
                                     const std::vector<vk::PushConstantRange> & pushConstantRange, vk::Extent2D extent)
    : name{name}
    , engine{engine}
    , library{*engine.library}
    , device{*engine.device}
    , shaderStages{shaderStages}
    , renderPass{renderPass}
    , pipelineCache{pipelineCache}
    , descriptorSetLayouts{descriptorSetLayouts}
    , pushConstantRange{pushConstantRange}
    , extent{extent}
{
    load();
}

void GraphicsPipelines::load()
{
    pipelineVertexInputStateCreateInfo.flags = {};
    pipelineVertexInputStateCreateInfo.setVertexBindingDescriptions(nullptr);
    pipelineVertexInputStateCreateInfo.setVertexAttributeDescriptions(nullptr);

    pipelineInputAssemblyStateCreateInfo.flags = {};
    pipelineInputAssemblyStateCreateInfo.setPrimitiveRestartEnable(VK_FALSE);
    pipelineInputAssemblyStateCreateInfo.setTopology(vk::PrimitiveTopology::eTriangleList);

    viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = utils::autoCast(extent.width),
        .height = utils::autoCast(extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    scissor = {
        .offset = {.x = 0, .y = 0},
        .extent = extent,
    };

    pipelineViewportStateCreateInfo.flags = {};
    pipelineViewportStateCreateInfo.setViewports(viewport);
    pipelineViewportStateCreateInfo.setScissors(scissor);

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
        .blendEnable = VK_FALSE,
        .srcColorBlendFactor = vk::BlendFactor::eZero,
        .dstColorBlendFactor = vk::BlendFactor::eZero,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eZero,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
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

    pipelineLayoutCreateInfo.flags = {};
    pipelineLayoutCreateInfo.setSetLayouts(descriptorSetLayouts);
    pipelineLayoutCreateInfo.setPushConstantRanges(nullptr);

    pipelineLayoutHolder = device.device.createPipelineLayoutUnique(pipelineLayoutCreateInfo, library.allocationCallbacks, library.dispatcher);
    pipelineLayout = *pipelineLayoutHolder;

    auto & graphicsPipelineCreateInfo = graphicsPipelineCreateInfos.emplace_back();
    graphicsPipelineCreateInfo.flags = {};
    graphicsPipelineCreateInfo.setStages(shaderStages.shaderStages.ref());
    graphicsPipelineCreateInfo.pVertexInputState = &pipelineVertexInputStateCreateInfo;
    graphicsPipelineCreateInfo.pInputAssemblyState = &pipelineInputAssemblyStateCreateInfo;
    graphicsPipelineCreateInfo.pTessellationState = nullptr;
    graphicsPipelineCreateInfo.pViewportState = &pipelineViewportStateCreateInfo;
    graphicsPipelineCreateInfo.pRasterizationState = &pipelineRasterizationStateCreateInfo;
    graphicsPipelineCreateInfo.pMultisampleState = &pipelineMultisampleStateCreateInfo;
    graphicsPipelineCreateInfo.pDepthStencilState = &pipelineDepthStencilStateCreateInfo;
    graphicsPipelineCreateInfo.pColorBlendState = &pipelineColorBlendStateCreateInfo;
    graphicsPipelineCreateInfo.pDynamicState = nullptr;
    graphicsPipelineCreateInfo.layout = pipelineLayout;
    graphicsPipelineCreateInfo.renderPass = renderPass;
    graphicsPipelineCreateInfo.subpass = 0;
    graphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
    graphicsPipelineCreateInfo.basePipelineIndex = 0;

    auto result = device.device.createGraphicsPipelinesUnique(pipelineCache, graphicsPipelineCreateInfos, library.allocationCallbacks, library.dispatcher);
    vk::resultCheck(result.result, fmt::format("Failed to create graphics pipeline '{}'", name).c_str());
    pipelineHolders = std::move(result.value);
    pipelines.reserve(std::size(pipelineHolders));
    for (const auto & pipelineHolder : pipelineHolders) {
        pipelines.push_back(*pipelineHolder);
    }
}

}  // namespace engine
