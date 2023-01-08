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

const float kVertices[] = {-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};

constexpr vk::DeviceSize alignedSize(vk::DeviceSize size, vk::DeviceSize alignment)
{
    INVARIANT(std::bitset<std::numeric_limits<vk::DeviceSize>::digits>{alignment}.count() == 1, "Expected power of two alignment");
    return (size + alignment - 1) & ~(alignment - 1);
}

}  // namespace

Resources::GraphicsPipeline::GraphicsPipeline(std::string_view name, const engine::Engine & engine, vk::PipelineCache pipelineCache, const engine::PipelineVertexInputState & pipelineVertexInputState, const engine::ShaderStages & shaderStages,
                                              vk::RenderPass renderPass, const std::vector<vk::DescriptorSetLayout> & descriptorSetLayouts, const std::vector<vk::PushConstantRange> & pushConstantRanges)
    : pipelineLayout{name, engine, pipelineVertexInputState, shaderStages, renderPass, descriptorSetLayouts, pushConstantRanges}, pipelines{engine, pipelineCache}
{
    pipelines.add(pipelineLayout);
    pipelines.create();
}

Resources::Resources(const engine::Engine & engine, const FileIo & fileIo, uint32_t framesInFlight)
    : engine{engine}
    , fileIo{fileIo}
    , framesInFlight{framesInFlight}
    , shaderStages{engine}
    , vertexShader{"fullscreen_triangle.vert"sv, engine, fileIo}
    , vertexShaderReflection{vertexShader, "main"}
    , fragmentShader{"fullscreen_triangle.frag"sv, engine, fileIo}
    , fragmentShaderReflection{fragmentShader, "main"}
{
    init();
}

auto Resources::createGraphicsPipeline(vk::RenderPass renderPass) const -> std::unique_ptr<const Resources::GraphicsPipeline>
{
    return std::make_unique<GraphicsPipeline>("rasterization", engine, pipelineCache->pipelineCache, pipelineVertexInputState, shaderStages, renderPass, descriptorSetLayouts, pushConstantRanges);
}

void Resources::init()
{
    INVARIANT(vertexShader.shaderStage == vk::ShaderStageFlagBits::eVertex, "Vertex shader has mismatched stage flags {} in reflection", vertexShader.shaderStage);
    INVARIANT(fragmentShader.shaderStage == vk::ShaderStageFlagBits::eFragment, "Fragment shader has mismatched stage flags {} in reflection", fragmentShader.shaderStage);

    shaderStages.append(vertexShader, vertexShaderReflection.entryPoint);
    shaderStages.append(fragmentShader, fragmentShaderReflection.entryPoint);

    vk::DeviceSize minUniformBufferOffsetAlignment = engine.device->physicalDevice.physicalDeviceProperties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits.minUniformBufferOffsetAlignment;
    uniformBufferPerFrameSize = alignedSize(sizeof(UniformBuffer), minUniformBufferOffsetAlignment);
    vk::BufferCreateInfo uniformBufferCreateInfo;
    uniformBufferCreateInfo.size = uniformBufferPerFrameSize * framesInFlight;
    uniformBufferCreateInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst;
    uniformBuffer = engine.vma->createBuffer(uniformBufferCreateInfo, "Uniform buffer");

    vk::BufferCreateInfo vertexBufferCreateInfo;
    vertexBufferCreateInfo.size = sizeof kVertices;
    vertexBufferCreateInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst;
    vertexBuffer = engine.vma->createBuffer(uniformBufferCreateInfo, "Uniform buffer");

    constexpr uint32_t vertexBufferBinding = 0;
    pipelineVertexInputState = vertexShaderReflection.getPipelineVertexInputState(vertexBufferBinding);

    {
        INVARIANT(std::size(vertexShaderReflection.descriptorSetLayouts) == 0, "");
    }

    {
        INVARIANT(std::size(fragmentShaderReflection.descriptorSetLayouts) == 1, "");
        auto & descriptorSetLayoutReflection = fragmentShaderReflection.descriptorSetLayouts.back();
        INVARIANT(descriptorSetLayoutReflection.set == 0, "");
        INVARIANT(std::size(descriptorSetLayoutReflection.bindings) == 1, "");
        auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutReflection.bindings.back();
        INVARIANT(descriptorSetLayoutBindingReflection.binding == vertexBufferBinding, "");
        INVARIANT(descriptorSetLayoutBindingReflection.descriptorType == vk::DescriptorType::eUniformBuffer, "");
        INVARIANT(descriptorSetLayoutBindingReflection.descriptorCount == 1, "");
        INVARIANT(descriptorSetLayoutBindingReflection.stageFlags == vk::ShaderStageFlagBits::eFragment, "");

        descriptorSetLayoutBindingReflection.descriptorType = vk::DescriptorType::eUniformBufferDynamic;
    }

    auto device = engine.device->device;
    const auto & allocationCallbacks = engine.library->allocationCallbacks;
    const auto & dispatcher = engine.library->dispatcher;

    // TODO: merge?

    size_t vertexShaderDescriptorCount = std::size(vertexShaderReflection.descriptorSetLayoutCreateInfos);
    descriptorSetLayoutHolders.reserve(std::size(descriptorSetLayoutHolders) + vertexShaderDescriptorCount);
    descriptorSetLayouts.reserve(std::size(descriptorSetLayouts) + vertexShaderDescriptorCount);
    for (size_t d = 0; d < vertexShaderDescriptorCount; ++d) {
        descriptorSetLayoutHolders.push_back(device.createDescriptorSetLayoutUnique(vertexShaderReflection.descriptorSetLayoutCreateInfos.at(d), allocationCallbacks, dispatcher));
        descriptorSetLayouts.push_back(*descriptorSetLayoutHolders.back());
        engine.device->setDebugUtilsObjectName(descriptorSetLayouts.back(), vertexShaderReflection.descriptorNames.at(d));
    }

    size_t fragmentShaderDescriptorCount = std::size(fragmentShaderReflection.descriptorSetLayoutCreateInfos);
    descriptorSetLayoutHolders.reserve(std::size(descriptorSetLayoutHolders) + fragmentShaderDescriptorCount);
    descriptorSetLayouts.reserve(std::size(descriptorSetLayouts) + fragmentShaderDescriptorCount);
    for (size_t d = 0; d < fragmentShaderDescriptorCount; ++d) {
        descriptorSetLayoutHolders.push_back(device.createDescriptorSetLayoutUnique(fragmentShaderReflection.descriptorSetLayoutCreateInfos.at(d), allocationCallbacks, dispatcher));
        descriptorSetLayouts.push_back(*descriptorSetLayoutHolders.back());
        engine.device->setDebugUtilsObjectName(descriptorSetLayouts.back(), fragmentShaderReflection.descriptorNames.at(d));
    }

    pushConstantRanges.insert(std::cend(pushConstantRanges), std::cbegin(vertexShaderReflection.pushConstantRanges), std::cend(vertexShaderReflection.pushConstantRanges));
    pushConstantRanges.insert(std::cend(pushConstantRanges), std::cbegin(fragmentShaderReflection.pushConstantRanges), std::cend(fragmentShaderReflection.pushConstantRanges));

    std::vector<vk::DescriptorPoolSize> descriptorPoolSize = {
        {vk::DescriptorType::eUniformBufferDynamic, 1},
    };
    vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo;
    // descriptorPoolCreateInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    descriptorPoolCreateInfo.setPoolSizes(descriptorPoolSize);
    descriptorPoolHolder = device.createDescriptorPoolUnique(descriptorPoolCreateInfo, allocationCallbacks, dispatcher);
    vk::DescriptorPool descriptorPool = *descriptorPoolHolder;
    // name

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.setSetLayouts(descriptorSetLayouts);
    descriptorSetHolders = device.allocateDescriptorSetsUnique(descriptorSetAllocateInfo, dispatcher);
    descriptorSets.reserve(descriptorSetHolders.size());
    for (const auto & descriptorSetHolder : descriptorSetHolders) {
        descriptorSets.push_back(*descriptorSetHolder);
        // name em
    }

    std::vector<vk::DescriptorBufferInfo> descriptorBufferInfos = {
        {
            .buffer = uniformBuffer.getBuffer(),
            .offset = 0,  // dynamic offset is used so this is ignored
            .range = uniformBufferPerFrameSize,
        },
    };

    std::vector<vk::WriteDescriptorSet> writeDescriptorSets;  // array?
    auto & writeDescriptorSet = writeDescriptorSets.back();
    writeDescriptorSet.dstSet = descriptorSets.at(0);
    writeDescriptorSet.dstBinding = 0;
    writeDescriptorSet.dstArrayElement = 0;
    writeDescriptorSet.descriptorType = vk::DescriptorType::eUniformBufferDynamic;
    writeDescriptorSet.setBufferInfo(descriptorBufferInfos);
    device.updateDescriptorSets(writeDescriptorSets, nullptr, dispatcher);

    pipelineCache = std::make_unique<engine::PipelineCache>("rasterization", engine, fileIo);
}

ResourceManager::ResourceManager(engine::Engine & engine) : engine{engine}
{}

std::shared_ptr<const Resources> ResourceManager::getOrCreateResources(uint32_t framesInFlight)
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
