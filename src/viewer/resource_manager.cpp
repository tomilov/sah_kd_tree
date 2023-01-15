#include <engine/device.hpp>
#include <engine/engine.hpp>
#include <engine/format.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <utils/assert.hpp>
#include <viewer/resource_manager.hpp>

#include <bitset>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>
#include <utility>

#include <cstdint>

using namespace std::string_view_literals;

namespace viewer
{

namespace
{

const auto kSquircle = "squircle"sv;

constexpr vk::DeviceSize alignedSize(vk::DeviceSize size, vk::DeviceSize alignment)
{
    INVARIANT(std::bitset<std::numeric_limits<vk::DeviceSize>::digits>{alignment}.count() == 1, "Expected power of two alignment");
    return (size + alignment - 1) & ~(alignment - 1);
}

}  // namespace

Resources::GraphicsPipeline::GraphicsPipeline(std::string_view name, const engine::Engine & engine, vk::PipelineCache pipelineCache, const engine::ShaderStages & shaderStages, vk::RenderPass renderPass,
                                              const std::vector<vk::PushConstantRange> & pushConstantRanges)
    : pipelineLayout{name, engine, shaderStages, renderPass, pushConstantRanges}, pipelines{engine, pipelineCache}
{
    pipelines.add(pipelineLayout);
    pipelines.create();
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

auto Resources::createGraphicsPipeline(vk::RenderPass renderPass) const -> std::unique_ptr<const Resources::GraphicsPipeline>
{
    return std::make_unique<GraphicsPipeline>(kSquircle, engine, pipelineCache->pipelineCache, shaderStages, renderPass, pushConstantRanges);
}

void Resources::init()
{
    INVARIANT(vertexShader.shaderStage == vk::ShaderStageFlagBits::eVertex, "Vertex shader has mismatched stage flags {} in reflection", vertexShader.shaderStage);
    INVARIANT(fragmentShader.shaderStage == vk::ShaderStageFlagBits::eFragment, "Fragment shader has mismatched stage flags {} in reflection", fragmentShader.shaderStage);

    vk::DeviceSize minUniformBufferOffsetAlignment = engine.getDevice().physicalDevice.physicalDeviceProperties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits.minUniformBufferOffsetAlignment;
    uniformBufferPerFrameSize = alignedSize(sizeof(UniformBuffer), minUniformBufferOffsetAlignment);
    vk::BufferCreateInfo uniformBufferCreateInfo;
    uniformBufferCreateInfo.size = uniformBufferPerFrameSize * framesInFlight;
    uniformBufferCreateInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;
    uniformBuffer = engine.getMemoryAllocator().createStagingBuffer(uniformBufferCreateInfo, "Uniform buffer consists of float t");

    vk::BufferCreateInfo vertexBufferCreateInfo;
    vertexBufferCreateInfo.size = sizeof kVertices;
    vertexBufferCreateInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer;
    vertexBuffer = engine.getMemoryAllocator().createStagingBuffer(vertexBufferCreateInfo, "Vertices of square");

    {
        INVARIANT(std::empty(vertexShaderReflection.descriptorSetLayoutSetBindings), "");
        INVARIANT(std::empty(vertexShaderReflection.pushConstantRanges), "");
    }

    constexpr uint32_t set = 0;
    const std::string kUniformBufferName = "uniformBuffer";

    {
        INVARIANT(std::size(fragmentShaderReflection.descriptorSetLayoutSetBindings) == 1, "");
        INVARIANT(fragmentShaderReflection.descriptorSetLayoutSetBindings.contains(set), "");
        auto & descriptorSetLayoutBindings = fragmentShaderReflection.descriptorSetLayoutSetBindings.at(set);
        INVARIANT(std::size(descriptorSetLayoutBindings) == 1, "");
        auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(kUniformBufferName);
        INVARIANT(descriptorSetLayoutBindingReflection.binding == 0, "");
        INVARIANT(descriptorSetLayoutBindingReflection.descriptorType == vk::DescriptorType::eUniformBuffer, "");
        INVARIANT(descriptorSetLayoutBindingReflection.descriptorCount == 1, "");
        INVARIANT(descriptorSetLayoutBindingReflection.stageFlags == vk::ShaderStageFlagBits::eFragment, "");
        descriptorSetLayoutBindingReflection.descriptorType = vk::DescriptorType::eUniformBufferDynamic;

        INVARIANT(std::empty(vertexShaderReflection.pushConstantRanges), "");
    }

    shaderStages.append(vertexShader, vertexShaderReflection, vertexShaderReflection.entryPoint);
    shaderStages.append(fragmentShader, fragmentShaderReflection, fragmentShaderReflection.entryPoint);

    shaderStages.createDescriptorSetLayouts(kSquircle);

    descriptorPoolSizes = shaderStages.getDescriptorPoolSizes();

    uint32_t setCount = utils::autoCast(std::size(shaderStages.descriptorSetLayouts));
    descriptorPool.emplace(kSquircle, engine, setCount, descriptorPoolSizes);
    descriptorSets.emplace(kSquircle, engine, shaderStages, *descriptorPool);

    auto device = engine.getDevice().device;
    const auto & dispatcher = engine.getLibrary().dispatcher;

    std::vector<vk::DescriptorBufferInfo> descriptorBufferInfos = {
        {
            .buffer = uniformBuffer.getBuffer(),
            .offset = 0,  // dynamic offset is used so this is ignored
            .range = sizeof(UniformBuffer),
        },
    };

    auto setBindings = shaderStages.setBindings.find(set);
    INVARIANT(setBindings != std::end(shaderStages.setBindings), "Set {} is not found", set);
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

    pipelineCache = std::make_unique<engine::PipelineCache>(kSquircle, engine, fileIo);
}

ResourceManager::ResourceManager(engine::Engine & engine) : engine{engine}
{}

std::shared_ptr<const Resources> ResourceManager::getOrCreateResources(uint32_t framesInFlight) const
{
    std::lock_guard<std::mutex> lock{mutex};
    auto & w = resources[framesInFlight];
    auto p = w.lock();
    if (!p) {
        p = Resources::make(engine, fileIo, framesInFlight);
        w = p;
    }
    return p;
}

}  // namespace viewer
