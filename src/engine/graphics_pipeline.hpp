#pragma once

#include <engine/fwd.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <string>
#include <string_view>
#include <vector>

#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{

struct GraphicsPipelines;

struct ENGINE_EXPORT GraphicsPipelineLayout final : utils::NonCopyable
{
    const std::string name;

    const Engine & engine;
    const Library & library;
    const Device & device;
    const ShaderStages & shaderStages;
    const vk::RenderPass renderPass;

    vk::PipelineInputAssemblyStateCreateInfo pipelineInputAssemblyStateCreateInfo;
    vk::PipelineViewportStateCreateInfo pipelineViewportStateCreateInfo;
    vk::PipelineRasterizationStateCreateInfo pipelineRasterizationStateCreateInfo;
    vk::PipelineColorBlendAttachmentState pipelineColorBlendAttachmentState;  // single attachment
    vk::PipelineColorBlendStateCreateInfo pipelineColorBlendStateCreateInfo;
    std::vector<vk::DynamicState> dynamicStates;
    vk::PipelineDynamicStateCreateInfo pipelineDynamicStateCreateInfo;
    vk::PipelineMultisampleStateCreateInfo pipelineMultisampleStateCreateInfo;
    vk::PipelineDepthStencilStateCreateInfo pipelineDepthStencilStateCreateInfo;

    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;

    vk::UniquePipelineLayout pipelineLayoutHolder;
    vk::PipelineLayout pipelineLayout;

    GraphicsPipelineLayout(std::string_view name, const Engine & engine, const ShaderStages & shaderStages, vk::RenderPass renderPass);

private:
    friend GraphicsPipelines;

    void init();
    void fill(std::string & name, vk::GraphicsPipelineCreateInfo & graphicsPipelineCreateInfo) const;
};

struct ENGINE_EXPORT GraphicsPipelines final : utils::NonCopyable
{
    const Engine & engine;
    const Library & library;
    const Device & device;
    const vk::PipelineCache pipelineCache;

    std::vector<std::string> names;
    std::vector<vk::GraphicsPipelineCreateInfo> graphicsPipelineCreateInfos;

    std::vector<vk::UniquePipeline> pipelineHolders;
    std::vector<vk::Pipeline> pipelines;

    GraphicsPipelines(const Engine & engine, vk::PipelineCache pipelineCache);

    void add(const GraphicsPipelineLayout & graphicsPipelineLayout);
    void create();
};

}  // namespace engine