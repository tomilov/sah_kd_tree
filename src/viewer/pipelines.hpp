#pragma once

#include <engine/context.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/pipeline_cache.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <memory>
#include <string_view>

namespace viewer
{

class FileIo;
class Shaders;

struct GraphicsPipeline : utils::OneTime<GraphicsPipeline>
{
    engine::GraphicsPipelineLayout pipelineLayout;
    engine::GraphicsPipeline pipeline;

    GraphicsPipeline(std::string_view name, const engine::Context & context, bool useDescriptorBuffer, vk::PipelineCache pipelineCache, std::shared_ptr<const engine::ShaderStages> shaderStages, vk::RenderPass renderPass);

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

class Pipelines : utils::OneTime<Pipelines>
{
public:
    explicit Pipelines(const engine::Context & context, bool descriptorBufferEnabled);
    Pipelines(Pipelines && rhs) noexcept = default;
    ~Pipelines();

    [[nodiscard]] GraphicsPipeline createDisplayGraphicsPipeline(std::string_view name, vk::RenderPass renderPass) const &;
    [[nodiscard]] GraphicsPipeline createSceneGraphicsPipeline(std::string_view name, vk::RenderPass renderPass) const &;

private:
    const engine::Context & context;
    const bool descriptorBufferEnabled;

    std::unique_ptr<FileIo> fileIo;
    engine::PipelineCache pipelineCache;
    std::shared_ptr<Shaders> sceneShaders;
    std::shared_ptr<Shaders> displayShaders;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

}  // namespace viewer
