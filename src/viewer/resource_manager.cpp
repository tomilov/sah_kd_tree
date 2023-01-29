#include <engine/device.hpp>
#include <engine/engine.hpp>
#include <engine/format.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/pipeline_cache.hpp>
#include <engine/vma.hpp>
#include <utils/assert.hpp>
#include <viewer/resource_manager.hpp>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <bitset>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>
#include <utility>

#include <cstddef>
#include <cstdint>

using namespace std::string_view_literals;

namespace viewer
{

namespace
{

const auto kSquircle = "squircle"sv;
constexpr uint32_t kUniformBufferSet = 0;
const std::string kUniformBufferName = "uniformBuffer";  // clazy:exclude=non-pod-global-static

constexpr vk::DeviceSize alignedSize(vk::DeviceSize size, vk::DeviceSize alignment)
{
    INVARIANT(std::bitset<std::numeric_limits<vk::DeviceSize>::digits>{alignment}.count() == 1, "Expected power of two alignment, got {}", alignment);
    return (size + alignment - 1) & ~(alignment - 1);
}

}  // namespace

Resources::Descriptors::Descriptors(const engine::Engine & engine, uint32_t framesInFlight, const engine::ShaderStages & shaderStages, const std::vector<vk::DescriptorPoolSize> & descriptorPoolSizes)
{
    const auto & dispatcher = engine.getLibrary().dispatcher;
    const auto & device = engine.getDevice();
    const auto & vma = engine.getMemoryAllocator();

    const auto & physicalDeviceProperties = device.physicalDevice.physicalDeviceProperties2Chain.get<vk::PhysicalDeviceProperties2>().properties;
    uniformBufferSize = alignedSize(sizeof(UniformBuffer), physicalDeviceProperties.limits.minUniformBufferOffsetAlignment);

    vk::BufferCreateInfo uniformBufferCreateInfo;
    uniformBufferCreateInfo.size = uniformBufferSize;
    uniformBufferCreateInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;
    uniformBuffer.reserve(framesInFlight);
    for (uint32_t i = 0; i < framesInFlight; ++i) {
        auto uniformBufferName = fmt::format("Uniform buffer frmae #{}", i);
        uniformBuffer.push_back(vma.createStagingBuffer(uniformBufferCreateInfo, uniformBufferName));
        INVARIANT(uniformBuffer.back().getMemoryPropertyFlags() & vk::MemoryPropertyFlagBits::eDeviceLocal, "Failed to allocate uniform buffer (frame #{}) in DEVICE_LOCAL memory", i);
    }

    vk::BufferCreateInfo vertexBufferCreateInfo;
    vertexBufferCreateInfo.size = sizeof kVertices;
    vertexBufferCreateInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer;
    vertexBuffer = vma.createStagingBuffer(vertexBufferCreateInfo, "Vertices of square");
    INVARIANT(vertexBuffer.getMemoryPropertyFlags() & vk::MemoryPropertyFlagBits::eDeviceLocal, "Failed to allocate uniform buffer in DEVICE_LOCAL memory");

    uint32_t setCount = utils::autoCast(std::size(shaderStages.descriptorSetLayouts));
    descriptorPool.emplace(kSquircle, engine, framesInFlight, setCount, descriptorPoolSizes);
    for (uint32_t i = 0; i < framesInFlight; ++i) {
        descriptorSets.emplace_back(kSquircle, engine, shaderStages, *descriptorPool);
    }

    pushConstantRanges = shaderStages.getDisjointPushConstantRanges();

    std::vector<vk::DescriptorBufferInfo> descriptorBufferInfos;
    descriptorBufferInfos.resize(framesInFlight);
    for (uint32_t i = 0; i < framesInFlight; ++i) {
        descriptorBufferInfos.at(i) = {
            .buffer = uniformBuffer.at(i).getBuffer(),
            .offset = 0,
            .range = uniformBuffer.at(i).getSize(),
        };
    }

    auto setBindings = shaderStages.setBindings.find(kUniformBufferSet);
    INVARIANT(setBindings != std::end(shaderStages.setBindings), "Set {} is not found", kUniformBufferSet);
    uint32_t uniformBufferSetIndex = utils::autoCast(std::distance(std::begin(shaderStages.setBindings), setBindings));  // linear, but who cares?
    const auto & uniformBufferBinding = setBindings->second.getBinding(kUniformBufferName);

    std::vector<vk::WriteDescriptorSet> writeDescriptorSets(framesInFlight);
    for (uint32_t i = 0; i < framesInFlight; ++i) {
        auto & writeDescriptorSet = writeDescriptorSets.at(i);

        writeDescriptorSet.dstSet = descriptorSets.at(i).descriptorSets.at(uniformBufferSetIndex);
        writeDescriptorSet.dstBinding = uniformBufferBinding.binding;
        writeDescriptorSet.dstArrayElement = 0;  // not an array
        writeDescriptorSet.descriptorType = uniformBufferBinding.descriptorType;
        writeDescriptorSet.setBufferInfo(descriptorBufferInfos.at(i));
    }
    device.device.updateDescriptorSets(writeDescriptorSets, nullptr, dispatcher);
}

Resources::GraphicsPipeline::GraphicsPipeline(std::string_view name, const engine::Engine & engine, vk::PipelineCache pipelineCache, const engine::ShaderStages & shaderStages, vk::RenderPass renderPass)
    : pipelineLayout{name, engine, shaderStages, renderPass}, pipelines{engine, pipelineCache}
{
    pipelines.add(pipelineLayout);
    pipelines.create();
}

uint32_t Resources::getFramesInFlight() const
{
    return framesInFlight;
}

std::shared_ptr<Resources> Resources::make(const engine::Engine & engine, const FileIo & fileIo, std::shared_ptr<const engine::PipelineCache> && pipelineCache, uint32_t framesInFlight)
{
    return std::shared_ptr<Resources>{new Resources{engine, fileIo, std::move(pipelineCache), framesInFlight}};
}

std::unique_ptr<const Resources::Descriptors> Resources::makeDescriptors() const
{
    return std::make_unique<Descriptors>(engine, framesInFlight, shaderStages, descriptorPoolSizes);
}

auto Resources::createGraphicsPipeline(vk::RenderPass renderPass) const -> std::unique_ptr<const GraphicsPipeline>
{
    return std::make_unique<GraphicsPipeline>(kSquircle, engine, pipelineCache->pipelineCache, shaderStages, renderPass);
}

Resources::Resources(const engine::Engine & engine, const FileIo & fileIo, std::shared_ptr<const engine::PipelineCache> && pipelineCache, uint32_t framesInFlight)
    : engine{engine}
    , fileIo{fileIo}
    , pipelineCache{std::move(pipelineCache)}
    , framesInFlight{framesInFlight}
    , vertexShader{"squircle.vert", engine, fileIo}
    , vertexShaderReflection{vertexShader, "main"}
    , fragmentShader{"squircle.frag", engine, fileIo}
    , fragmentShaderReflection{fragmentShader, "main"}
    , shaderStages{engine, vertexBufferBinding}
{
    init();
}

void Resources::init()
{
    INVARIANT(vertexShader.shaderStage == vk::ShaderStageFlagBits::eVertex, "Vertex shader has mismatched stage flags {} in reflection", vertexShader.shaderStage);
    INVARIANT(fragmentShader.shaderStage == vk::ShaderStageFlagBits::eFragment, "Fragment shader has mismatched stage flags {} in reflection", fragmentShader.shaderStage);

    {
        INVARIANT(std::size(vertexShaderReflection.descriptorSetLayoutSetBindings) == 0, "");

        INVARIANT(std::size(vertexShaderReflection.pushConstantRanges) == 1, "");
        const auto & pushConstantRange = vertexShaderReflection.pushConstantRanges.at(0);
        INVARIANT(pushConstantRange.stageFlags == vk::ShaderStageFlagBits::eVertex, "");
        INVARIANT(pushConstantRange.offset == offsetof(PushConstants, viewTransform), "");
        INVARIANT(pushConstantRange.size == sizeof(PushConstants::viewTransform), "");
    }
    shaderStages.append(vertexShader, vertexShaderReflection, vertexShaderReflection.entryPoint);

    {
        INVARIANT(std::size(fragmentShaderReflection.descriptorSetLayoutSetBindings) == 1, "");
        INVARIANT(fragmentShaderReflection.descriptorSetLayoutSetBindings.contains(kUniformBufferSet), "");
        auto & descriptorSetLayoutBindings = fragmentShaderReflection.descriptorSetLayoutSetBindings.at(kUniformBufferSet);
        INVARIANT(std::size(descriptorSetLayoutBindings) == 1, "");
        auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(kUniformBufferName);
        INVARIANT(descriptorSetLayoutBindingReflection.binding == 0, "");
        INVARIANT(descriptorSetLayoutBindingReflection.descriptorType == vk::DescriptorType::eUniformBuffer, "");
        INVARIANT(descriptorSetLayoutBindingReflection.descriptorCount == 1, "");
        INVARIANT(descriptorSetLayoutBindingReflection.stageFlags == vk::ShaderStageFlagBits::eFragment, "");

        INVARIANT(std::empty(fragmentShaderReflection.pushConstantRanges), "");
    }
    shaderStages.append(fragmentShader, fragmentShaderReflection, fragmentShaderReflection.entryPoint);

    shaderStages.createDescriptorSetLayouts(kSquircle);
    descriptorPoolSizes = shaderStages.getDescriptorPoolSizes();
}

ResourceManager::ResourceManager(const engine::Engine & engine) : engine{engine}
{}

std::shared_ptr<const Resources> ResourceManager::getOrCreateResources(uint32_t framesInFlight) const
{
    std::lock_guard<std::mutex> lock{mutex};

    auto pipelineCacheHolder = pipelineCache.lock();
    if (!pipelineCacheHolder) {
        pipelineCacheHolder = std::make_shared<engine::PipelineCache>(kSquircle, engine, fileIo);
        pipelineCache = pipelineCacheHolder;
    }

    auto & w = resources[framesInFlight];
    auto p = w.lock();
    if (p) {
        SPDLOG_DEBUG("Old resources reused");
    } else {
        p = Resources::make(engine, fileIo, std::move(pipelineCacheHolder), framesInFlight);
        w = p;
        SPDLOG_DEBUG("New resources created");
    }
    return p;
}

}  // namespace viewer
