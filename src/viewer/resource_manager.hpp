#pragma once

#include <engine/fwd.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/pipeline_cache.hpp>
#include <engine/shader_module.hpp>
#include <engine/vma.hpp>
#include <scene/scene.hpp>
#include <viewer/file_io.hpp>

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include <QtCore/QStringLiteral>

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
    float t = 0.0f;
};
#pragma pack(pop)

class Resources : public std::enable_shared_from_this<Resources>
{
public:
    struct GraphicsPipeline
    {
        engine::GraphicsPipelineLayout pipelineLayout;
        engine::GraphicsPipelines pipelines;

        GraphicsPipeline(std::string_view name, const engine::Engine & engine, vk::PipelineCache pipelineCache, const engine::PipelineVertexInputState & pipelineVertexInputState, const engine::ShaderStages & shaderStages, vk::RenderPass renderPass,
                         const std::vector<vk::DescriptorSetLayout> & descriptorSetLayouts, const std::vector<vk::PushConstantRange> & pushConstantRanges);
    };

    [[nodiscard]] uint32_t getFramesInFlight() const
    {
        return framesInFlight;
    }

    [[nodiscard]] static std::shared_ptr<Resources> make(const engine::Engine & engine, const FileIo & fileIo, uint32_t framesInFlight)
    {
        return std::shared_ptr<Resources>{new Resources{engine, fileIo, framesInFlight}};
    }

    [[nodiscard]] std::unique_ptr<const GraphicsPipeline> createGraphicsPipeline(vk::RenderPass renderPass) const;

    [[nodiscard]] vk::DeviceSize getUniformBufferPerFrameSize() const
    {
        return uniformBufferPerFrameSize;
    }

    [[nodiscard]] const engine::Buffer & getUniformBuffer() const
    {
        return uniformBuffer;
    };

    [[nodiscard]] const engine::Buffer & getVertexBuffer() const
    {
        return vertexBuffer;
    };

    [[nodiscard]] const std::vector<vk::DescriptorSet> & getDescriptorSets() const
    {
        return descriptorSets;
    }

private:
    const engine::Engine & engine;
    const FileIo & fileIo;
    const uint32_t framesInFlight;

    engine::ShaderStages shaderStages;

    engine::ShaderModule vertexShader;
    engine::ShaderModuleReflection vertexShaderReflection;

    engine::ShaderModule fragmentShader;
    engine::ShaderModuleReflection fragmentShaderReflection;

    vk::DeviceSize uniformBufferPerFrameSize = 0;
    engine::Buffer uniformBuffer;
    engine::Buffer vertexBuffer;

    engine::PipelineVertexInputState pipelineVertexInputState;
    std::vector<vk::UniqueDescriptorSetLayout> descriptorSetLayoutHolders;
    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;

    vk::UniqueDescriptorPool descriptorPoolHolder;
    std::vector<vk::UniqueDescriptorSet> descriptorSetHolders;
    std::vector<vk::DescriptorSet> descriptorSets;

    std::vector<vk::PushConstantRange> pushConstantRanges;

    std::unique_ptr<const engine::PipelineCache> pipelineCache;

    Resources(const engine::Engine & engine, const FileIo & fileIo, uint32_t framesInFlight);

    void init();
};

class ResourceManager
{
public:
    ResourceManager(engine::Engine & engine);

    [[nodiscard]] std::shared_ptr<const Resources> getOrCreateResources(uint32_t framesInFlight);

private:
    engine::Engine & engine;
    const FileIo fileIo{QStringLiteral("shaders:")};

    mutable std::mutex mutex;
    std::unordered_map<uint32_t /*framesInFlight*/, std::weak_ptr<const Resources>> resources;
};

}  // namespace viewer
