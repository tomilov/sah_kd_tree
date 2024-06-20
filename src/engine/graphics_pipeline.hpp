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

struct GraphicsPipeline;

struct ENGINE_EXPORT GraphicsPipelineLayout final : utils::OneTime<GraphicsPipelineLayout>
{
    GraphicsPipelineLayout(std::string_view name, const Context & context, std::shared_ptr<const ShaderStages> shaderStages);

    [[nodiscard]] std::shared_ptr<const ShaderStages> getShaderStages() const;

    [[nodiscard]] vk::PipelineLayout getPipelineLayout() const &;
    [[nodiscard]] operator vk::PipelineLayout() const &;  // NOLINT: google-explicit-constructor

private:
    friend GraphicsPipeline;

    std::string name;

    std::shared_ptr<const ShaderStages> shaderStages;

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

    vk::UniquePipelineLayout pipelineLayout;

    void fill(vk::GraphicsPipelineCreateInfo & graphicsPipelineCreateInfo, bool useDescriptorBuffer, vk::RenderPass renderPass) const;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

struct ENGINE_EXPORT GraphicsPipeline final : utils::OneTime<GraphicsPipeline>
{
    GraphicsPipeline(std::string_view name, const Context & context, bool useDescriptorBuffer, vk::PipelineCache pipelineCache, const GraphicsPipelineLayout & graphicsPipelineLayout, vk::RenderPass renderPass);

    [[nodiscard]] bool getUseDescriptorBuffer() const;
    [[nodiscard]] vk::RenderPass getRenderPass() const;

    [[nodiscard]] vk::Pipeline getPipeline() const &;
    [[nodiscard]] operator vk::Pipeline() const &;  // NOLINT: google-explicit-constructor

private:
    std::string name;
    const bool useDescriptorBuffer;
    const vk::RenderPass renderPass;

    vk::GraphicsPipelineCreateInfo graphicsPipelineCreateInfo;
    vk::UniquePipeline pipeline;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

}  // namespace engine
