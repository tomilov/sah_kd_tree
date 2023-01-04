#include <engine/device.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/library.hpp>
#include <engine/pipeline_cache.hpp>
#include <engine/render_pass.hpp>
#include <engine/utils.hpp>
#include <utils/auto_cast.hpp>

#include <fmt/format.h>

#include <utility>

namespace engine
{

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
        .width = utils::autoCast(width),
        .height = utils::autoCast(height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    scissor = {
        .offset = {.x = 0, .y = 0},
        .extent = {.width = width, .height = height},
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

    pipelineColorBlendStateCreateInfo.flags = {};
    pipelineColorBlendStateCreateInfo.logicOpEnable = VK_FALSE;
    pipelineColorBlendStateCreateInfo.logicOp = vk::LogicOp::eCopy;
    pipelineColorBlendStateCreateInfo.setAttachments(pipelineColorBlendAttachmentState);
    pipelineColorBlendStateCreateInfo.blendConstants = {{0.0f, 0.0f, 0.0f, 0.0f}};

    pipelineMultisampleStateCreateInfo = {
        .flags = {},
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.0f,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE,
    };

    pipelineLayoutCreateInfo.flags = {};
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.setSetLayouts(nullptr);
    pipelineLayoutCreateInfo.setPushConstantRanges(nullptr);

    pipelineLayoutHolder = device.device.createPipelineLayoutUnique(pipelineLayoutCreateInfo, library.allocationCallbacks, library.dispatcher);
    pipelineLayout = *pipelineLayoutHolder;

    shaderStagesHeads = toChainHeads(shaderStages.shaderStages);
    auto & graphicsPipelineCreateInfo = graphicsPipelineCreateInfos.emplace_back();
    graphicsPipelineCreateInfo.flags = {};
    graphicsPipelineCreateInfo.setStages(shaderStagesHeads);
    graphicsPipelineCreateInfo.pVertexInputState = &pipelineVertexInputStateCreateInfo;
    graphicsPipelineCreateInfo.pInputAssemblyState = &pipelineInputAssemblyStateCreateInfo;
    graphicsPipelineCreateInfo.pTessellationState = nullptr;
    graphicsPipelineCreateInfo.pViewportState = &pipelineViewportStateCreateInfo;
    graphicsPipelineCreateInfo.pRasterizationState = &pipelineRasterizationStateCreateInfo;
    graphicsPipelineCreateInfo.pMultisampleState = &pipelineMultisampleStateCreateInfo;
    graphicsPipelineCreateInfo.pDepthStencilState = nullptr;
    graphicsPipelineCreateInfo.pColorBlendState = &pipelineColorBlendStateCreateInfo;
    graphicsPipelineCreateInfo.pDynamicState = nullptr;
    graphicsPipelineCreateInfo.layout = pipelineLayout;
    graphicsPipelineCreateInfo.renderPass = renderPass.renderPass;
    graphicsPipelineCreateInfo.subpass = 0;
    graphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
    graphicsPipelineCreateInfo.basePipelineIndex = 0;

    auto result = device.device.createGraphicsPipelinesUnique(pipelineCache.pipelineCache, graphicsPipelineCreateInfos, library.allocationCallbacks, library.dispatcher);
    vk::resultCheck(result.result, fmt::format("Failed to create graphics pipeline '{}'", name).c_str());
    pipelineHolders = std::move(result.value);
    pipelines.reserve(std::size(pipelineHolders));
    for (const auto & pipelineHolder : pipelineHolders) {
        pipelines.push_back(*pipelineHolder);
    }
}

}  // namespace engine
