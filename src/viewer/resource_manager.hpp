#pragma once

#include <engine/fwd.hpp>
#include <engine/pipeline_cache.hpp>
#include <engine/shader_module.hpp>
#include <engine/vma.hpp>
#include <viewer/file_io.hpp>

#include <vulkan/vulkan.hpp>

#include <QtCore/QStringLiteral>

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

using namespace Qt::StringLiterals;

namespace viewer
{

#pragma pack(push, 1)
struct UniformBuffer
{
    float t = 0.0f;
};
#pragma pack(pop)

class Resources : public std::enable_shared_from_this<Resources>
{
public:
    struct GraphicsPipeline
    {
        std::unique_ptr<const engine::GraphicsPipelineLayout> pipelineLayout;
        std::unique_ptr<const engine::GraphicsPipelines> pipeline;
    };

    std::shared_ptr<Resources> get()
    {
        return shared_from_this();
    }

    std::shared_ptr<const Resources> get() const
    {
        return shared_from_this();
    }

    [[nodiscard]] static std::shared_ptr<Resources> make(const engine::Engine & engine, const FileIo & fileIo, uint32_t framesInFlight)
    {
        return std::shared_ptr<Resources>{new Resources{engine, fileIo, framesInFlight}};
    }

    GraphicsPipeline createGraphicsPipeline(vk::RenderPass renderPass, vk::Extent2D extent) const;

private:
    const engine::Engine & engine;
    const FileIo & fileIo;
    const uint32_t framesInFlight;

    engine::ShaderStages shaderStages;

    engine::ShaderModule vertexShader;
    engine::ShaderModuleReflection vertexShaderReflection;

    engine::ShaderModule fragmentShader;
    engine::ShaderModuleReflection fragmentShaderReflection;

    engine::Buffer uniformBuffer;
    engine::Buffer vertexBuffer;

    engine::PipelineVertexInputState pipelineVertexInputState;
    std::vector<vk::UniqueDescriptorSetLayout> descriptorSetLayoutHolders;
    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;

    std::vector<vk::PushConstantRange> pushConstantRanges;

    std::unique_ptr<const engine::PipelineCache> pipelineCache;

    Resources(const engine::Engine & engine, const FileIo & fileIo, uint32_t framesInFlight);

    void init();
};

class ResourceManager
{
public:
    ResourceManager(engine::Engine & engine);

    std::shared_ptr<const Resources> getOrCreateResources(uint32_t framesInFlight);

private:
    engine::Engine & engine;
    const FileIo fileIo{QStringLiteral("shaders:")};

    mutable std::mutex mutex;
    std::unordered_map<uint32_t /*framesInFlight*/, std::weak_ptr<const Resources>> resources;
};

}  // namespace viewer
