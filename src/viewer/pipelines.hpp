#pragma once

#include <engine/context.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/pipeline_cache.hpp>
#include <engine/shader_module.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <cstdint>

namespace viewer
{

class FileIo;

struct ShaderModule final : utils::OneTime<ShaderModule>
{
    engine::ShaderModule shaderModule;
    engine::ShaderModuleReflection shaderReflection;

    ShaderModule(const engine::Context & context, const FileIo & fileIo, std::string_view shaderName, std::string_view entryPoint);

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

class Shaders
    : utils::NonCopyable
    , public std::enable_shared_from_this<Shaders>
{
    struct Private
    {
        explicit Private() = default;
    };

public:
    Shaders(Private, std::string_view name, const engine::Context & context, const FileIo & fileIo, bool descriptorBufferEnabled);

    [[nodiscard]] static std::shared_ptr<Shaders> make(std::string_view name, const engine::Context & context, const FileIo & fileIo, bool descriptorBufferEnabled)
    {
        return std::make_shared<Shaders>(Private{}, name, context, fileIo, descriptorBufferEnabled);
    }

    void addShader(std::string_view shaderName, std::string_view entryPoint);
    void create();

    [[nodiscard]] bool getDescriptorBufferEnabled() const
    {
        return descriptorBufferEnabled;
    }

    [[nodiscard]] const std::vector<ShaderModule> & getShaderModules() const &
    {
        return shaderModules;
    }

    [[nodiscard]] std::shared_ptr<const engine::ShaderStages> getShaderStages() const &
    {
        return {shared_from_this(), &shaderStages};
    }

private:
    static constexpr uint32_t kVertexBufferBinding = 0;

    std::string name;
    const engine::Context & context;
    const FileIo & fileIo;
    const bool descriptorBufferEnabled;

    std::vector<ShaderModule> shaderModules;
    engine::ShaderStages shaderStages;
};

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
