#pragma once

#include <engine/context.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/pipeline_cache.hpp>
#include <engine/shader_module.hpp>
#include <utils/assert.hpp>
#include <utils/noncopyable.hpp>
#include <viewer/file_io.hpp>

#include <vulkan/vulkan.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <cstdint>

namespace viewer
{

struct ShaderModule final : utils::OneTime<ShaderModule>
{
    engine::ShaderModule shaderModule;
    engine::ShaderModuleReflection shaderReflection;

    ShaderModule(const engine::Context & context, const FileIo & fileIo, std::string_view shaderName, std::string_view entryPoint)
        : shaderModule{context, fileIo, shaderName}
        , shaderReflection{context, shaderModule, entryPoint}
    {}

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

class Shaders final
    : utils::NonCopyable
    , public std::enable_shared_from_this<Shaders>
{
    struct Private
    {
        explicit Private() = default;
    };

public:
    static const std::string_view kDefaultEntryPoint;

    Shaders(Private, std::string_view name, const engine::Context & context, const FileIo & fileIo, bool descriptorBufferEnabled);

    [[nodiscard]] static std::shared_ptr<Shaders> make(std::string_view name, const engine::Context & context, const FileIo & fileIo, bool descriptorBufferEnabled)
    {
        return std::make_shared<Shaders>(Private{}, name, context, fileIo, descriptorBufferEnabled);
    }

    void addShader(std::string_view shaderName, std::string_view entryPoint = kDefaultEntryPoint);
    void create();

    [[nodiscard]] bool getDescriptorBufferEnabled() const
    {
        return descriptorBufferEnabled;
    }

    [[nodiscard]] const std::vector<ShaderModule> & getShaderModules() const &
    {
        return shaderModules;
    }

    [[nodiscard]] const engine::ShaderStages & getShaderStages() const &
    {
        return shaderStages;
    }

    [[nodiscard]] std::shared_ptr<const engine::ShaderStages> getShaderStagesPtr() const
    {
        return {shared_from_this(), &getShaderStages()};
    }

    [[nodiscard]] const engine::GraphicsPipelineLayout & getGraphicsPipelineLayout() const &
    {
        ASSERT(pipelineLayout);
        return pipelineLayout.value();
    }

    [[nodiscard]] std::shared_ptr<const engine::GraphicsPipelineLayout> getGraphicsPipelineLayoutPtr() const
    {
        return {shared_from_this(), &getGraphicsPipelineLayout()};
    }

private:
    static constexpr uint32_t kVertexBufferBinding = 0;

    std::string name;
    const engine::Context & context;
    const FileIo & fileIo;
    const bool descriptorBufferEnabled;

    std::vector<ShaderModule> shaderModules;
    engine::ShaderStages shaderStages;
    std::optional<engine::GraphicsPipelineLayout> pipelineLayout;
};

struct GraphicsPipeline : utils::OneTime<GraphicsPipeline>
{
    std::shared_ptr<const Shaders> shaders;
    std::optional<engine::GraphicsPipeline> pipeline = {};

    explicit GraphicsPipeline(std::shared_ptr<const Shaders> shaders)
        : shaders{std::move(shaders)}
    {
        ASSERT(this->shaders);
    }

    void initPipeline(std::string_view name, const engine::Context & context, vk::PipelineCache pipelineCache, bool descriptorBufferEnabled, vk::RenderPass renderPass);

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

    [[nodiscard]] const engine::PipelineCache & getPipelineCache() const &
    {
        return pipelineCache;
    }

    [[nodiscard]] const std::shared_ptr<const Shaders> & getSceneShaders() const &
    {
        return sceneShaders;
    }

    [[nodiscard]] const std::shared_ptr<const Shaders> & getDisplayShaders() const &
    {
        return displayShaders;
    }

private:
    const engine::Context & context;
    const bool descriptorBufferEnabled;

    std::unique_ptr<FileIo> fileIo;
    engine::PipelineCache pipelineCache;
    std::shared_ptr<const Shaders> sceneShaders;
    std::shared_ptr<const Shaders> displayShaders;

    void init();

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

}  // namespace viewer
