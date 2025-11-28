#pragma once

#include <engine/fwd.hpp>
#include <engine/specialization_info.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <string>
#include <string_view>

#include <engine/engine_export.h>

namespace engine
{

#if GLM_FORCE_DEPTH_ZERO_TO_ONE
inline constexpr float kMinDepth = 0.0f;
#else
inline constexpr float kMinDepth = -1.0f;
#endif

struct ENGINE_EXPORT GraphicsPipeline final : utils::NonCopyable
{
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

    GraphicsPipeline(std::string_view name, const Context & context, vk::PipelineCache pipelineCache, bool descriptorBufferEnabled, const PipelineLayout & pipelineLayout, vk::RenderPass renderPass, SpecializationInfos && specializationInfos);

    void create();

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
    const Context & context;
    const vk::PipelineCache pipelineCache;
    const bool descriptorBufferEnabled;
    const vk::RenderPass renderPass;

    SpecializationInfos specializationInfos;
    std::vector<vk::PipelineShaderStageCreateInfo> pipelineShaderStageCreateInfos;
    vk::UniquePipeline pipeline;
};

}  // namespace engine
