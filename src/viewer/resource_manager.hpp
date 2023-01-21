#pragma once

#include <engine/descriptors.hpp>
#include <engine/fwd.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/pipeline_cache.hpp>
#include <engine/shader_module.hpp>
#include <engine/vma.hpp>
#include <scene/scene.hpp>
#include <utils/noncopyable.hpp>
#include <viewer/file_io.hpp>

#include <glm/mat4x4.hpp>
#include <vulkan/vulkan.hpp>

#include <QtCore/QChar>

#include <memory>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace Qt::StringLiterals;

namespace viewer
{

using VertexType = glm::vec2;

inline const VertexType kVertices[] = {
    {-1.0f, -1.0f},
    {1.0f, -1.0f},
    {-1.0f, 1.0f},
    {1.0f, 1.0f},
};

#pragma pack(push, 1)
struct UniformBuffer
{
    float alpha = 0.0f;

    float t = 0.0f;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct PushConstants
{
    glm::mat4x4 viewTransform{1.0};
};
#pragma pack(pop)

class Resources
    : utils::NonCopyable
    , public std::enable_shared_from_this<Resources>
{
public:
    struct Descriptors : utils::NonCopyable
    {
        vk::DeviceSize uniformBufferPerFrameSize = 0;
        engine::Buffer uniformBuffer;
        engine::Buffer vertexBuffer;

        std::optional<engine::DescriptorPool> descriptorPool;
        std::optional<engine::DescriptorSets> descriptorSets;

        std::vector<vk::PushConstantRange> pushConstantRanges;

        Descriptors(const engine::Engine & engine, uint32_t framesInFlight, const engine::ShaderStages & shaderStages, const std::vector<vk::DescriptorPoolSize> & descriptorPoolSizes);
    };

    struct GraphicsPipeline : utils::NonCopyable
    {
        engine::GraphicsPipelineLayout pipelineLayout;
        engine::GraphicsPipelines pipelines;

        GraphicsPipeline(std::string_view name, const engine::Engine & engine, vk::PipelineCache pipelineCache, const engine::ShaderStages & shaderStages, vk::RenderPass renderPass);
    };

    [[nodiscard]] uint32_t getFramesInFlight() const
    {
        return framesInFlight;
    }

    [[nodiscard]] static std::shared_ptr<Resources> make(const engine::Engine & engine, const FileIo & fileIo, vk::PipelineCache pipelineCache, uint32_t framesInFlight)
    {
        return std::shared_ptr<Resources>{new Resources{engine, fileIo, pipelineCache, framesInFlight}};
    }

    [[nodiscard]] std::unique_ptr<const Descriptors> getDescriptors() const;
    [[nodiscard]] std::unique_ptr<const GraphicsPipeline> createGraphicsPipeline(vk::RenderPass renderPass) const;

private:
    const engine::Engine & engine;
    const FileIo & fileIo;
    const vk::PipelineCache pipelineCache;
    const uint32_t framesInFlight;

    engine::ShaderModule vertexShader;
    engine::ShaderModuleReflection vertexShaderReflection;

    engine::ShaderModule fragmentShader;
    engine::ShaderModuleReflection fragmentShaderReflection;

    static constexpr uint32_t vertexBufferBinding = 0;
    engine::ShaderStages shaderStages;
    std::vector<vk::DescriptorPoolSize> descriptorPoolSizes;

    Resources(const engine::Engine & engine, const FileIo & fileIo, vk::PipelineCache pipelineCache, uint32_t framesInFlight);

    void init();
};

class ResourceManager
{
public:
    ResourceManager(const engine::Engine & engine);

    [[nodiscard]] std::shared_ptr<const Resources> getOrCreateResources(uint32_t framesInFlight) const;

private:
    const engine::Engine & engine;
    const FileIo fileIo{u"shaders:"_s};

    mutable std::mutex mutex;
    mutable std::unordered_map<uint32_t /*framesInFlight*/, std::weak_ptr<const Resources>> resources;
    mutable std::unique_ptr<const engine::PipelineCache> pipelineCache;
};

}  // namespace viewer
