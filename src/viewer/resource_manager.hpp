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

class Resources
{
public:
    Resources(const engine::Engine & engine, const FileIo & fileIo, uint32_t framesInFlight);

    std::unique_ptr<engine::GraphicsPipelines> createGraphicsPipelines(vk::RenderPass renderPass, vk::Extent2D extent) const;

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

    vk::UniqueDescriptorSetLayout descriptorSetLayoutHolder;
    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;

    std::vector<vk::PushConstantRange> pushConstantRange;

    std::unique_ptr<const engine::PipelineCache> pipelineCache;

    void init();
};

class ResourceManager
{
public:
    ResourceManager(engine::Engine & engine) : engine{engine}
    {}

    std::shared_ptr<const Resources> getOrCreateResources(uint32_t framesInFlight);

private:
    engine::Engine & engine;
    const FileIo fileIo{QStringLiteral("shaders:")};

    mutable std::mutex mutex;
    std::unordered_map<uint32_t /*framesInFlight*/, std::weak_ptr<const Resources>> resources;
};

}  // namespace viewer
