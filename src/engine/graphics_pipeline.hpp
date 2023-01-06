#pragma once

#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <string>
#include <string_view>
#include <vector>

#include <cstdint>

#include <engine/engine_export.h>

namespace engine
{
class Engine;
struct Library;
struct Device;
struct ShaderStages;

struct ENGINE_EXPORT GraphicsPipelines final : utils::NonCopyable
{
    const std::string name;

    const Engine & engine;
    const Library & library;
    const Device & device;
    const ShaderStages & shaderStages;
    const vk::RenderPass renderPass;
    const vk::PipelineCache pipelineCache;
    const std::vector<vk::DescriptorSetLayout> & descriptorSetLayouts;
    const std::vector<vk::PushConstantRange> & pushConstantRange;
    const vk::Extent2D extent;

    vk::PipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo;
    vk::PipelineInputAssemblyStateCreateInfo pipelineInputAssemblyStateCreateInfo;
    vk::Viewport viewport;
    vk::Rect2D scissor;
    vk::PipelineViewportStateCreateInfo pipelineViewportStateCreateInfo;
    vk::PipelineRasterizationStateCreateInfo pipelineRasterizationStateCreateInfo;
    vk::PipelineColorBlendAttachmentState pipelineColorBlendAttachmentState;
    vk::PipelineColorBlendStateCreateInfo pipelineColorBlendStateCreateInfo;
    vk::PipelineMultisampleStateCreateInfo pipelineMultisampleStateCreateInfo;
    vk::PipelineDepthStencilStateCreateInfo pipelineDepthStencilStateCreateInfo;

    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;

    vk::UniquePipelineLayout pipelineLayoutHolder;
    vk::PipelineLayout pipelineLayout;

    std::vector<vk::GraphicsPipelineCreateInfo> graphicsPipelineCreateInfos;

    std::vector<vk::UniquePipeline> pipelineHolders;
    std::vector<vk::Pipeline> pipelines;

    GraphicsPipelines(std::string_view name, const Engine & engine, const ShaderStages & shaderStages, vk::RenderPass renderPass, vk::PipelineCache pipelineCache, const std::vector<vk::DescriptorSetLayout> & descriptorSetLayouts,
                      const std::vector<vk::PushConstantRange> & pushConstantRange, vk::Extent2D extent);

private:
    void load();
};

}  // namespace engine
