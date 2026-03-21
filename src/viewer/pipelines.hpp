#pragma once

#include <engine/compute_pipeline.hpp>
#include <engine/context.hpp>
#include <engine/file_io.hpp>
#include <engine/fwd.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/pipeline_cache.hpp>
#include <engine/pipeline_layout.hpp>
#include <engine/shader_module.hpp>
#include <engine/specialization_info.hpp>
#include <utils/assert.hpp>
#include <utils/hash.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>

#include <memory>
#include <optional>
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

    ShaderModule(
        const engine::Context & context,
        const engine::FileIo & fileIo,
        std::string_view shaderName,
        std::string_view entryPoint)
        : shaderModule{context,
              fileIo,
              shaderName}
        , shaderReflection{context,
              shaderModule,
              entryPoint}
    {}
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

    Shaders(
        Private,
        std::string_view name,
        const engine::Context & context,
        std::shared_ptr<const engine::FileIo> fileIo,
        engine::DescriptorManagementKind descriptorManagementKind);

    [[nodiscard]] static std::shared_ptr<Shaders> make(
        std::string_view name,
        const engine::Context & context,
        std::shared_ptr<const engine::FileIo> fileIo,
        engine::DescriptorManagementKind descriptorManagementKind)
    {
        return std::make_shared<Shaders>(Private{}, name, context, fileIo, descriptorManagementKind);
    }

    void addShader(
        std::string_view shaderName,
        std::string_view entryPoint = kDefaultEntryPoint);
    void create();

    [[nodiscard]] engine::DescriptorManagementKind getDescriptorManagementKind() const
    {
        return descriptorManagementKind;
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

    [[nodiscard]] const engine::PipelineLayout & getPipelineLayout() const &
    {
        ASSERT(pipelineLayout);
        return pipelineLayout.value();
    }

    [[nodiscard]] std::shared_ptr<const engine::PipelineLayout> getPipelineLayoutPtr() const
    {
        return {shared_from_this(), &getPipelineLayout()};
    }

private:
    static constexpr uint32_t kVertexBufferBinding = 0;

    std::string name;
    const engine::Context & context;
    std::shared_ptr<const engine::FileIo> fileIo;
    const engine::DescriptorManagementKind descriptorManagementKind;

    std::vector<ShaderModule> shaderModules;
    engine::ShaderStages shaderStages;
    std::optional<engine::PipelineLayout> pipelineLayout;
};

struct GraphicsPipeline : utils::OneTime<GraphicsPipeline>
{
    static constexpr vk::PipelineBindPoint kPipelineBindPoint = vk::PipelineBindPoint::eGraphics;

    std::shared_ptr<const Shaders> shaders;
    std::unique_ptr<engine::GraphicsPipeline> pipeline;

    explicit GraphicsPipeline(std::shared_ptr<const Shaders> shaders);

    [[nodiscard]] engine::GraphicsPipeline & initPipeline(
        std::string_view name,
        const engine::Context & context,
        vk::PipelineCache pipelineCache,
        engine::DescriptorManagementKind descriptorManagementKind,
        vk::RenderPass renderPass,
        engine::SpecializationInfos && specializationInfos);
};

struct ComputePipeline : utils::OneTime<ComputePipeline>
{
    static constexpr vk::PipelineBindPoint kPipelineBindPoint = vk::PipelineBindPoint::eCompute;

    std::shared_ptr<const Shaders> shaders;
    std::unique_ptr<engine::ComputePipeline> pipeline;

    explicit ComputePipeline(std::shared_ptr<const Shaders> shaders);

    [[nodiscard]] engine::ComputePipeline & initPipeline(
        std::string_view name,
        const engine::Context & context,
        vk::PipelineCache pipelineCache,
        engine::DescriptorManagementKind descriptorManagementKind,
        engine::SpecializationInfos && specializationInfos);
};

class Pipelines : utils::OneTime<Pipelines>
{
public:
    explicit Pipelines(
        const engine::Context & context,
        engine::DescriptorManagementKind descriptorManagementKind);
    Pipelines(Pipelines && rhs) noexcept = default;
    ~Pipelines();

    [[nodiscard]] const engine::PipelineCache & getPipelineCache() const &
    {
        return pipelineCache;
    }

    [[nodiscard]] std::shared_ptr<const Shaders> getSceneShaders() const;
    [[nodiscard]] std::shared_ptr<const Shaders> getDisplayShaders() const;
    [[nodiscard]] std::shared_ptr<const Shaders> getTraceSahKdTreeShaders() const;

private:
    const engine::Context & context;
    const engine::DescriptorManagementKind descriptorManagementKind;

    std::shared_ptr<engine::FileIo> fileIo;
    engine::PipelineCache pipelineCache;
    mutable std::weak_ptr<Shaders> sceneShaders;
    mutable std::weak_ptr<Shaders> displayShaders;
    mutable std::weak_ptr<Shaders> traceSahKdTreeShaders;
};

}  // namespace viewer

template struct utils::OneTime<viewer::ShaderModule>::CheckTraits;
template struct utils::OneTime<viewer::GraphicsPipeline>::CheckTraits;
template struct utils::OneTime<viewer::ComputePipeline>::CheckTraits;
template struct utils::OneTime<viewer::Pipelines>::CheckTraits;
