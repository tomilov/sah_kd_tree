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
struct PhysicalDevice;
struct Device;
struct ShaderStages;
struct RenderPass;
struct PipelineCache;

struct ENGINE_EXPORT GraphicsPipelines final : utils::NonCopyable
{
    const std::string name;

    Engine & engine;
    Library & library;
    PhysicalDevice & physicalDevice;
    Device & device;
    ShaderStages & shaderStages;
    RenderPass & renderPass;
    PipelineCache & pipelineCache;

    const uint32_t width;
    const uint32_t height;

    vk::PipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo;
    vk::PipelineInputAssemblyStateCreateInfo pipelineInputAssemblyStateCreateInfo;
    vk::Viewport viewport;
    vk::Rect2D scissor;
    vk::PipelineViewportStateCreateInfo pipelineViewportStateCreateInfo;
    vk::PipelineRasterizationStateCreateInfo pipelineRasterizationStateCreateInfo;
    vk::PipelineColorBlendAttachmentState pipelineColorBlendAttachmentState;
    vk::PipelineColorBlendStateCreateInfo pipelineColorBlendStateCreateInfo;
    vk::PipelineMultisampleStateCreateInfo pipelineMultisampleStateCreateInfo;

    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;

    vk::UniquePipelineLayout pipelineLayoutHolder;
    vk::PipelineLayout pipelineLayout;

    std::vector<vk::PipelineShaderStageCreateInfo> shaderStagesHeads;
    std::vector<vk::GraphicsPipelineCreateInfo> graphicsPipelineCreateInfos;

    std::vector<vk::UniquePipeline> pipelineHolders;
    std::vector<vk::Pipeline> pipelines;

    GraphicsPipelines(std::string_view name, Engine & engine, Library & library, PhysicalDevice & physicalDevice, Device & device, ShaderStages & shaderStages, RenderPass & renderPass, PipelineCache & pipelineCache, uint32_t width, uint32_t height)
        : name{name}, engine{engine}, library{library}, physicalDevice{physicalDevice}, device{device}, shaderStages{shaderStages}, renderPass{renderPass}, pipelineCache{pipelineCache}, width{width}, height{height}
    {
        load();
    }

private:
    void load();
};

}  // namespace engine
