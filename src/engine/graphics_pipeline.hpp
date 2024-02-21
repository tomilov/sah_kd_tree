#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
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

struct ENGINE_EXPORT GraphicsPipelineLayout final : utils::OneTime<GraphicsPipelineLayout>
{
    GraphicsPipelineLayout(std::string_view name, const Context & context, const ShaderStages & shaderStages, vk::RenderPass renderPass);

    [[nodiscard]] vk::RenderPass getAssociatedRenderPass() const &;
    [[nodiscard]] vk::PipelineLayout getPipelineLayout() const &;

private:
    friend GraphicsPipelines;

    std::string name;

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

    void fill(std::string & name, vk::GraphicsPipelineCreateInfo & graphicsPipelineCreateInfo, bool useDescriptorBuffer) const;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

struct ENGINE_EXPORT GraphicsPipelines final : utils::OneTime<GraphicsPipelines>
{
    GraphicsPipelines(const Context & context, vk::PipelineCache pipelineCache);

    void add(const GraphicsPipelineLayout & graphicsPipelineLayout, bool useDescriptorBuffer);
    void create();

    [[nodiscard]] const std::vector<vk::Pipeline> & getPipelines() const &;

private:
    const Context & context;
    const vk::PipelineCache pipelineCache;

    std::vector<std::string> names;
    std::vector<vk::GraphicsPipelineCreateInfo> graphicsPipelineCreateInfos;

    std::vector<vk::UniquePipeline> pipelineHolders;
    std::vector<vk::Pipeline> pipelines;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

}  // namespace engine
