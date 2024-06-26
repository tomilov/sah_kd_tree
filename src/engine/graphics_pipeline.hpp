#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <string>
#include <string_view>
#include <vector>

#include <engine/engine_export.h>

namespace engine
{

struct ENGINE_EXPORT GraphicsPipelineLayout final : utils::OneTime<GraphicsPipelineLayout>
{
    GraphicsPipelineLayout(std::string_view name, const Context & context, const ShaderStages & shaderStages);

    [[nodiscard]] const ShaderStages & getShaderStages() const &
    {
        return shaderStages;
    }

    [[nodiscard]] vk::PipelineLayout getPipelineLayout() const &
    {
        ASSERT(pipelineLayout);
        return *pipelineLayout;
    }

    [[nodiscard]] operator vk::PipelineLayout() const &  // NOLINT: google-explicit-constructor
    {
        return getPipelineLayout();
    }

private:
    std::string name;
    const Context & context;
    const ShaderStages & shaderStages;

    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
    vk::UniquePipelineLayout pipelineLayout;

    void init();

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

struct ENGINE_EXPORT GraphicsPipeline final : utils::OneTime<GraphicsPipeline>
{
    GraphicsPipeline(std::string_view name, const Context & context, vk::PipelineCache pipelineCache, bool descriptorBufferEnabled, const GraphicsPipelineLayout & graphicsPipelineLayout, vk::RenderPass renderPass);

    [[nodiscard]] bool getUseDescriptorBuffer() const
    {
        return descriptorBufferEnabled;
    }

    [[nodiscard]] vk::RenderPass getRenderPass() const
    {
        return renderPass;
    }

    [[nodiscard]] vk::Pipeline getPipeline() const &
    {
        ASSERT(pipeline);
        return *pipeline;
    }
    [[nodiscard]] operator vk::Pipeline() const &  // NOLINT: google-explicit-constructor
    {
        return getPipeline();
    }

private:
    std::string name;
    const bool descriptorBufferEnabled;
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
    vk::GraphicsPipelineCreateInfo graphicsPipelineCreateInfo;
    vk::UniquePipeline pipeline;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

}  // namespace engine
