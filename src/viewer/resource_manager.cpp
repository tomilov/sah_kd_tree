#include <engine/device.hpp>
#include <engine/engine.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <utils/assert.hpp>
#include <viewer/resource_manager.hpp>

#include <bitset>
#include <limits>
#include <string_view>

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

Resources::Resources(const engine::Engine & engine, const FileIo & fileIo, uint32_t framesInFlight)
    : engine{engine}
    , fileIo{fileIo}
    , framesInFlight{framesInFlight}
    , shaderStages{engine}
    , vertexShader{"fullscreen_triangle.vert"sv, engine, &fileIo}
    , vertexShaderReflection{vertexShader}
    , fragmentShader{"fullscreen_triangle.frag"sv, engine, &fileIo}
    , fragmentShaderReflection{fragmentShader}
{
    init();
}

auto Resources::createGraphicsPipelines(vk::RenderPass renderPass, vk::Extent2D extent) const -> std::unique_ptr<engine::GraphicsPipelines>
{
    return std::make_unique<engine::GraphicsPipelines>("squircle", engine, shaderStages, renderPass, pipelineCache->pipelineCache, descriptorSetLayouts, pushConstantRange, extent);
}

void Resources::init()
{
    shaderStages.append(vertexShader, "main");
    shaderStages.append(fragmentShader, "main");

    vk::DeviceSize minUniformBufferOffsetAlignment = engine.device->physicalDevice.physicalDeviceProperties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits.minUniformBufferOffsetAlignment;
    vk::DeviceSize uniformBufferPerFrameSize = alignedSize(sizeof(UniformBuffer), minUniformBufferOffsetAlignment);
    vk::BufferCreateInfo uniformBufferCreateInfo;
    uniformBufferCreateInfo.size = uniformBufferPerFrameSize * framesInFlight;
    uniformBufferCreateInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst;
    uniformBuffer = engine.vma->createBuffer(uniformBufferCreateInfo, "Uniform buffer");

    vk::BufferCreateInfo vertexBufferCreateInfo;
    vertexBufferCreateInfo.size = sizeof kVertices;
    vertexBufferCreateInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst;
    vertexBuffer = engine.vma->createBuffer(uniformBufferCreateInfo, "Uniform buffer");

    INVARIANT(std::size(fragmentShaderReflection.descriptorSetLayouts) == 1, "");
    auto & descriptorSetLayoutReflection = fragmentShaderReflection.descriptorSetLayouts.back();
    INVARIANT(descriptorSetLayoutReflection.set == 0, "");
    INVARIANT(std::size(descriptorSetLayoutReflection.bindings) == 1, "");
    auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutReflection.bindings.back();
    INVARIANT(descriptorSetLayoutBindingReflection.binding == 0, "");
    INVARIANT(descriptorSetLayoutBindingReflection.descriptorType == vk::DescriptorType::eUniformBuffer, "");
    INVARIANT(descriptorSetLayoutBindingReflection.descriptorCount == 1, "");
    INVARIANT(descriptorSetLayoutBindingReflection.stageFlags == vk::ShaderStageFlagBits::eFragment, "");
    descriptorSetLayoutBindingReflection.descriptorType = vk::DescriptorType::eUniformBufferDynamic;

    auto device = engine.device->device;
    const auto & allocationCallbacks = engine.library->allocationCallbacks;
    const auto & dispatcher = engine.library->dispatcher;

    descriptorSetLayoutHolder = device.createDescriptorSetLayoutUnique(fragmentShaderReflection.descriptorSetLayoutCreateInfos.back(), allocationCallbacks, dispatcher);
    descriptorSetLayouts.push_back(*descriptorSetLayoutHolder);

    pipelineCache = std::make_unique<engine::PipelineCache>("squircle", engine, &fileIo);
}

std::shared_ptr<const Resources> ResourceManager::getOrCreateResources(uint32_t framesInFlight)
{
    std::lock_guard<std::mutex> lock{mutex};
    auto & w = resources[framesInFlight];
    auto p = w.lock();
    if (!p) {
        p = std::make_shared<const Resources>(engine, fileIo, framesInFlight);
        w = p;
    }
    return p;
}

}  // namespace viewer
