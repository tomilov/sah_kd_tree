#include <engine/device.hpp>
#include <engine/engine.hpp>
#include <engine/format.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <utils/assert.hpp>
#include <viewer/resource_manager.hpp>

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
const std::string kUniformBufferName = "uniformBuffer";

constexpr vk::DeviceSize alignedSize(vk::DeviceSize size, vk::DeviceSize alignment)
{
    INVARIANT(std::bitset<std::numeric_limits<vk::DeviceSize>::digits>{alignment}.count() == 1, "Expected power of two alignment");
    return (size + alignment - 1) & ~(alignment - 1);
}

}  // namespace

Resources::Descriptors::Descriptors(const engine::Engine & engine, uint32_t framesInFlight, const engine::ShaderStages & shaderStages, const std::vector<vk::DescriptorPoolSize> & descriptorPoolSizes)
{
    const auto & physicalDeviceProperties = engine.getDevice().physicalDevice.physicalDeviceProperties2Chain.get<vk::PhysicalDeviceProperties2>().properties;

    vk::DeviceSize minUniformBufferOffsetAlignment = physicalDeviceProperties.limits.minUniformBufferOffsetAlignment;
    uniformBufferPerFrameSize = alignedSize(sizeof(UniformBuffer), minUniformBufferOffsetAlignment);
    vk::BufferCreateInfo uniformBufferCreateInfo;
    uniformBufferCreateInfo.size = uniformBufferPerFrameSize * framesInFlight;
    uniformBufferCreateInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;
    uniformBuffer = engine.getMemoryAllocator().createStagingBuffer(uniformBufferCreateInfo, "Uniform buffer consists of float t");
    INVARIANT(uniformBuffer.getMemoryPropertyFlags() & vk::MemoryPropertyFlagBits::eDeviceLocal, "Failed to allocate uniform buffer in DEVICE_LOCAL memory");

    vk::BufferCreateInfo vertexBufferCreateInfo;
    vertexBufferCreateInfo.size = sizeof kVertices;
    vertexBufferCreateInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer;
    vertexBuffer = engine.getMemoryAllocator().createStagingBuffer(vertexBufferCreateInfo, "Vertices of square");
    INVARIANT(vertexBuffer.getMemoryPropertyFlags() & vk::MemoryPropertyFlagBits::eDeviceLocal, "Failed to allocate uniform buffer in DEVICE_LOCAL memory");

    uint32_t setCount = utils::autoCast(std::size(shaderStages.descriptorSetLayouts));
    descriptorPool.emplace(kSquircle, engine, setCount, descriptorPoolSizes);
    descriptorSets.emplace(kSquircle, engine, shaderStages, *descriptorPool);

    pushConstantRanges = shaderStages.pushConstantRanges;

    auto device = engine.getDevice().device;
    const auto & dispatcher = engine.getLibrary().dispatcher;

    std::vector<vk::DescriptorBufferInfo> descriptorBufferInfos = {
        {
            .buffer = uniformBuffer.getBuffer(),
            .offset = 0,  // dynamic offset is used so this is ignored
            .range = sizeof(UniformBuffer),
        },
    };

    auto setBindings = shaderStages.setBindings.find(kUniformBufferSet);
    INVARIANT(setBindings != std::end(shaderStages.setBindings), "Set {} is not found", kUniformBufferSet);
    uint32_t uniformBufferSetIndex = utils::autoCast(std::distance(std::begin(shaderStages.setBindings), setBindings));
    const auto & uniformBufferBinding = setBindings->second.bindings.at(setBindings->second.bindingIndices.at(kUniformBufferName));

    std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
    auto & writeDescriptorSet = writeDescriptorSets.emplace_back();
    writeDescriptorSet.dstSet = descriptorSets.value().descriptorSets.at(uniformBufferSetIndex);
    writeDescriptorSet.dstBinding = uniformBufferBinding.binding;
    writeDescriptorSet.dstArrayElement = 0;  // not an array
    writeDescriptorSet.descriptorType = vk::DescriptorType::eUniformBufferDynamic;
    writeDescriptorSet.setBufferInfo(descriptorBufferInfos);
    device.updateDescriptorSets(writeDescriptorSets, nullptr, dispatcher);
}

Resources::GraphicsPipeline::GraphicsPipeline(std::string_view name, const engine::Engine & engine, vk::PipelineCache pipelineCache, const engine::ShaderStages & shaderStages, vk::RenderPass renderPass)
    : pipelineLayout{name, engine, shaderStages, renderPass}, pipelines{engine, pipelineCache}
{
    pipelines.add(pipelineLayout);
    pipelines.create();
}

std::unique_ptr<const Resources::Descriptors> Resources::getDescriptors() const
{
    return std::make_unique<Descriptors>(engine, framesInFlight, shaderStages, descriptorPoolSizes);
}

auto Resources::createGraphicsPipeline(vk::RenderPass renderPass) const -> std::unique_ptr<const Resources::GraphicsPipeline>
{
    return std::make_unique<GraphicsPipeline>(kSquircle, engine, pipelineCache->pipelineCache, shaderStages, renderPass);
}

Resources::Resources(const engine::Engine & engine, const FileIo & fileIo, uint32_t framesInFlight)
    : engine{engine}
    , fileIo{fileIo}
    , framesInFlight{framesInFlight}
    , vertexShader{"squircle.vert"sv, engine, fileIo}
    , vertexShaderReflection{vertexShader, "main"}
    , fragmentShader{"squircle.frag"sv, engine, fileIo}
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

        // patching VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER to VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
        descriptorSetLayoutBindingReflection.descriptorType = vk::DescriptorType::eUniformBufferDynamic;

        INVARIANT(std::empty(fragmentShaderReflection.pushConstantRanges), "");
    }
    shaderStages.append(fragmentShader, fragmentShaderReflection, fragmentShaderReflection.entryPoint);

    shaderStages.createDescriptorSetLayouts(kSquircle);
    descriptorPoolSizes = shaderStages.getDescriptorPoolSizes();

    pipelineCache = std::make_unique<engine::PipelineCache>(kSquircle, engine, fileIo);
}

ResourceManager::ResourceManager(engine::Engine & engine) : engine{engine}
{}

std::shared_ptr<const Resources> ResourceManager::getOrCreateResources(uint32_t framesInFlight) const
{
    std::lock_guard<std::mutex> lock{mutex};
    auto & w = resources[framesInFlight];
    auto p = w.lock();
    if (p) {
        SPDLOG_DEBUG("Old resources reused");
    } else {
        p = Resources::make(engine, fileIo, framesInFlight);
        w = p;
        SPDLOG_DEBUG("New resources created");
    }
    return p;
}

}  // namespace viewer
