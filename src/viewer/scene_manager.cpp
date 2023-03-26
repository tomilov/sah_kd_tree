#include <engine/device.hpp>
#include <engine/engine.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/pipeline_cache.hpp>
#include <engine/push_constant_ranges.hpp>
#include <engine/vma.hpp>
#include <format/vulkan.hpp>
#include <utils/assert.hpp>
#include <viewer/scene_manager.hpp>

#include <fmt/format.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include <bit>
#include <iterator>
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

[[nodiscard]] vk::DeviceSize alignedSize(vk::DeviceSize size, vk::DeviceSize alignment)
{
    INVARIANT(std::has_single_bit(alignment), "Expected power of two alignment, got {}", alignment);
    --alignment;
    return (size + alignment) & ~alignment;
}

}  // namespace

bool SceneDesignator::operator ==(const SceneDesignator & rhs) const noexcept
{
    return std::tie(token, path, framesInFlight) == std::tie(rhs.token, rhs.path, rhs.framesInFlight);
}

bool SceneDesignator::isValid() const noexcept
{
    if (std::empty(token)) {
        SPDLOG_WARN("token is empty");
        return false;
    }
    if (std::empty(path)) {
        SPDLOG_WARN("path is empty");
        return false;
    }
    if (framesInFlight < 1) {
        SPDLOG_WARN("framesInFlight is 0");
        return false;
    }
    return true;
}

Scene::Descriptors::Descriptors(const engine::Engine & engine, uint32_t framesInFlight, const engine::ShaderStages & shaderStages) : engine{engine}, framesInFlight{framesInFlight}, shaderStages{shaderStages}
{
    init();
}

size_t Scene::Descriptors::getDescriptorSize(vk::DescriptorType descriptorType) const
{
    const auto & device = engine.getDevice();
    const auto robustBufferAccess = device.physicalDevice.physicalDeviceFeatures2Chain.get<vk::PhysicalDeviceFeatures2>().features.robustBufferAccess;
    const auto & physicalDeviceDescriptorBufferProperties = device.physicalDevice.physicalDeviceProperties2Chain.get<vk::PhysicalDeviceDescriptorBufferPropertiesEXT>();
    switch (descriptorType) {
    case vk::DescriptorType::eSampler: {
        return physicalDeviceDescriptorBufferProperties.samplerDescriptorSize;
    }
    case vk::DescriptorType::eCombinedImageSampler: {
        return physicalDeviceDescriptorBufferProperties.combinedImageSamplerDescriptorSize;
    }
    case vk::DescriptorType::eSampledImage: {
        return physicalDeviceDescriptorBufferProperties.sampledImageDescriptorSize;
    }
    case vk::DescriptorType::eStorageImage: {
        return physicalDeviceDescriptorBufferProperties.storageImageDescriptorSize;
    }
    case vk::DescriptorType::eUniformTexelBuffer: {
        if (robustBufferAccess == VK_FALSE) {
            return physicalDeviceDescriptorBufferProperties.uniformTexelBufferDescriptorSize;
        } else {
            return physicalDeviceDescriptorBufferProperties.robustUniformTexelBufferDescriptorSize;
        }
    }
    case vk::DescriptorType::eStorageTexelBuffer: {
        if (robustBufferAccess == VK_FALSE) {
            return physicalDeviceDescriptorBufferProperties.storageTexelBufferDescriptorSize;
        } else {
            return physicalDeviceDescriptorBufferProperties.robustStorageTexelBufferDescriptorSize;
        }
    }
    case vk::DescriptorType::eUniformBuffer: {
        if (robustBufferAccess == VK_FALSE) {
            return physicalDeviceDescriptorBufferProperties.uniformBufferDescriptorSize;
        } else {
            return physicalDeviceDescriptorBufferProperties.robustUniformBufferDescriptorSize;
        }
    }
    case vk::DescriptorType::eStorageBuffer: {
        if (robustBufferAccess == VK_FALSE) {
            return physicalDeviceDescriptorBufferProperties.storageBufferDescriptorSize;
        } else {
            return physicalDeviceDescriptorBufferProperties.robustStorageBufferDescriptorSize;
        }
    }
    case vk::DescriptorType::eUniformBufferDynamic: {
        INVARIANT(false, "Dynamic uniform buffer descriptor cannot be stored in descriptor buffer");
    }
    case vk::DescriptorType::eStorageBufferDynamic: {
        INVARIANT(false, "Dynamic storage buffer descriptor cannot be stored in descriptor buffer");
    }
    case vk::DescriptorType::eInputAttachment: {
        return physicalDeviceDescriptorBufferProperties.inputAttachmentDescriptorSize;
    }
    case vk::DescriptorType::eInlineUniformBlock: {
        INVARIANT(false, "Inline uniform block descriptor cannot be stored in descriptor buffer");
    }
    case vk::DescriptorType::eAccelerationStructureKHR: {
        return physicalDeviceDescriptorBufferProperties.accelerationStructureDescriptorSize;
    }
    case vk::DescriptorType::eAccelerationStructureNV: {
        return physicalDeviceDescriptorBufferProperties.accelerationStructureDescriptorSize;
    }
    case vk::DescriptorType::eSampleWeightImageQCOM: {
        INVARIANT(false, "Sample weight image descriptor cannot be stored in descriptor buffer");
    }
    case vk::DescriptorType::eBlockMatchImageQCOM: {
        INVARIANT(false, "Block match image descriptor cannot be stored in descriptor buffer");
    }
    case vk::DescriptorType::eMutableEXT: {
        INVARIANT(false, "Mutable type descriptor cannot be stored in descriptor buffer");
    }
    }
    INVARIANT(false, "Unknown descriptor type {}", fmt::underlying(descriptorType));
}

void Scene::Descriptors::init()
{
    INVARIANT(framesInFlight > 0, "");

    const auto & dispatcher = engine.getLibrary().dispatcher;
    const auto & device = engine.getDevice();
    const auto & vma = engine.getMemoryAllocator();

    const auto & physicalDeviceLimits = device.physicalDevice.physicalDeviceProperties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits;
    INVARIANT(!kUseDescriptorBuffer || device.physicalDevice.enabledExtensionSet.contains(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME), VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME " should be enabled if kUseDescriptorBuffer");
    auto minAlignment = physicalDeviceLimits.nonCoherentAtomSize;

    {
        vk::BufferCreateInfo uniformBufferCreateInfo;
        uniformBufferCreateInfo.size = sizeof(UniformBuffer);
        uniformBufferCreateInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;
        if (kUseDescriptorBuffer) {
            uniformBufferCreateInfo.usage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
        }
        uniformBuffer.reserve(framesInFlight);
        constexpr vk::MemoryPropertyFlags kMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
        for (uint32_t i = 0; i < framesInFlight; ++i) {
            auto uniformBufferName = fmt::format("Uniform buffer (frame #{})", i);
            uniformBuffer.push_back(vma.createStagingBuffer(uniformBufferCreateInfo, minAlignment, uniformBufferName));
            auto memoryPropertyFlags = uniformBuffer.back().getMemoryPropertyFlags();
            INVARIANT(memoryPropertyFlags & kMemoryPropertyFlags, "Failed to allocate uniform buffer (frame #{}) in {} memory, got {} memory", i, kMemoryPropertyFlags, memoryPropertyFlags);
        }
    }

    {
        vk::BufferCreateInfo vertexBufferCreateInfo;
        vertexBufferCreateInfo.size = sizeof kVertices;
        vertexBufferCreateInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer;
        vertexBuffer = vma.createStagingBuffer(vertexBufferCreateInfo, minAlignment, "Vertices of square");
        constexpr vk::MemoryPropertyFlags kMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
        auto memoryPropertyFlags = vertexBuffer.getMemoryPropertyFlags();
        INVARIANT(memoryPropertyFlags & kMemoryPropertyFlags, "Failed to allocate uniform buffer in {} memory, got {} memory", kMemoryPropertyFlags, memoryPropertyFlags);
    }

    for (const auto & [set, bindings] : shaderStages.setBindings) {
        INVARIANT(set == bindings.setIndex, "Descriptor sets ids are not sequential non-negative numbers: {}, {}", set, bindings.setIndex);
    }

    if (kUseDescriptorBuffer) {
        constexpr vk::MemoryPropertyFlags kRequiredMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached;
        const auto descriptorBufferOffsetAlignment = device.physicalDevice.physicalDeviceProperties2Chain.get<vk::PhysicalDeviceDescriptorBufferPropertiesEXT>().descriptorBufferOffsetAlignment;
        descriptorSetBuffers.reserve(std::size(shaderStages.descriptorSetLayouts));
        auto set = std::cbegin(shaderStages.setBindings);
        for (const auto & descriptorSetLayout : shaderStages.descriptorSetLayouts) {
            vk::BufferCreateInfo descriptorBufferCreateInfo;
            descriptorBufferCreateInfo.usage = vk::BufferUsageFlagBits::eShaderDeviceAddress;
            descriptorBufferCreateInfo.size = alignedSize(device.device.getDescriptorSetLayoutSizeEXT(descriptorSetLayout, dispatcher), descriptorBufferOffsetAlignment) * framesInFlight;
            INVARIANT(set != std::cend(shaderStages.setBindings), "");
            for (const auto & binding : set->second.bindings) {
                switch (binding.descriptorType) {
                case vk::DescriptorType::eSampler: {
                    descriptorBufferCreateInfo.usage |= vk::BufferUsageFlagBits::eSamplerDescriptorBufferEXT;
                    break;
                }
                case vk::DescriptorType::eCombinedImageSampler: {
                    descriptorBufferCreateInfo.usage |= vk::BufferUsageFlagBits::eSamplerDescriptorBufferEXT;
                    descriptorBufferCreateInfo.usage |= vk::BufferUsageFlagBits::eResourceDescriptorBufferEXT;
                    break;
                }
                default: {
                    descriptorBufferCreateInfo.usage |= vk::BufferUsageFlagBits::eResourceDescriptorBufferEXT;
                    break;
                }
                }
            }
            auto descriptorBufferName = fmt::format("Descriptor buffer for set #{}", set->first);
            descriptorSetBuffers.push_back(vma.createDescriptorBuffer(descriptorBufferCreateInfo, minAlignment, descriptorBufferName));
            const auto & descriptorSetBuffer = descriptorSetBuffers.back();
            auto memoryPropertyFlags = descriptorSetBuffer.getMemoryPropertyFlags();
            INVARIANT(memoryPropertyFlags & kRequiredMemoryPropertyFlags, "Failed to allocate descriptor buffer in {} memory, got {} memory", kRequiredMemoryPropertyFlags, memoryPropertyFlags);

            descriptorBufferBindingInfos.emplace_back() = {
                .address = descriptorSetBuffer.getDeviceAddress(),
                .usage = descriptorBufferCreateInfo.usage,
            };

            ++set;
        }
    } else {
        descriptorPool.emplace(kSquircle, engine, framesInFlight, shaderStages);
        for (uint32_t i = 0; i < framesInFlight; ++i) {
            descriptorSets.emplace_back(kSquircle, engine, shaderStages, *descriptorPool);
        }
    }

    pushConstantRanges = engine::getDisjointPushConstantRanges(shaderStages.pushConstantRanges);

    if (kUseDescriptorBuffer) {
        for (const auto & [set, bindings] : shaderStages.setBindings) {
            const auto setIndex = bindings.setIndex;
            const auto & descriptorSetLayout = shaderStages.descriptorSetLayouts.at(setIndex);
            const auto & descriptorSetBuffer = descriptorSetBuffers.at(setIndex);
            auto mappedDescriptorSetBuffer = descriptorSetBuffer.map<std::byte>();
            auto setDescriptors = mappedDescriptorSetBuffer.get();
            const vk::DeviceSize descriptorSetBufferPerFrameSize = descriptorSetBuffer.getSize() / framesInFlight;
            for (uint32_t currentFrameSlot = 0; currentFrameSlot < framesInFlight; ++currentFrameSlot) {
                uint32_t b = 0;
                for (const auto & binding : bindings.bindings) {
                    vk::DescriptorGetInfoEXT descriptorGetInfo;
                    descriptorGetInfo.type = binding.descriptorType;
                    vk::DescriptorAddressInfoEXT descriptorAddressInfo;
                    const auto & bindingName = bindings.bindingNames.at(b);
                    if (bindingName == kUniformBufferName) {
                        ASSERT(binding.descriptorType == vk::DescriptorType::eUniformBuffer);
                        const auto & u = uniformBuffer.at(currentFrameSlot);
                        descriptorAddressInfo = {
                            .address = u.getDeviceAddress(),
                            .range = u.getSize(),
                            .format = vk::Format::eUndefined,
                        };
                        descriptorGetInfo.data.pUniformBuffer = &descriptorAddressInfo;
                    } else {
                        INVARIANT(false, "Cannot find descriptor for binding '{}'", bindingName);
                    }
                    vk::DeviceSize descriptorSize = getDescriptorSize(binding.descriptorType);
                    vk::DeviceSize bindingOffset = device.device.getDescriptorSetLayoutBindingOffsetEXT(descriptorSetLayout, binding.binding, dispatcher);
                    device.device.getDescriptorEXT(&descriptorGetInfo, descriptorSize, setDescriptors + bindingOffset, dispatcher);
                    ++b;
                }
                setDescriptors += descriptorSetBufferPerFrameSize;
            }
        }
    } else {
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
        uint32_t uniformBufferSetIndex = setBindings->second.setIndex;
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
}

Scene::GraphicsPipeline::GraphicsPipeline(std::string_view name, const engine::Engine & engine, vk::PipelineCache pipelineCache, const engine::ShaderStages & shaderStages, vk::RenderPass renderPass)
    : pipelineLayout{name, engine, shaderStages, renderPass}, pipelines{engine, pipelineCache}
{
    pipelines.add(pipelineLayout, kUseDescriptorBuffer);
    pipelines.create();
}

const std::shared_ptr<const SceneDesignator> & Scene::getSceneDesignator() const
{
    ASSERT(sceneDesignator);
    return sceneDesignator;
}

std::shared_ptr<Scene> Scene::make(const engine::Engine & engine, const FileIo & fileIo, std::shared_ptr<const engine::PipelineCache> && pipelineCache, SceneDesignatorPtr sceneDesignator)
{
    return std::shared_ptr<Scene>{new Scene{engine, fileIo, std::move(pipelineCache), std::move(sceneDesignator)}};
}

std::unique_ptr<const Scene::Descriptors> Scene::makeDescriptors() const
{
    return std::make_unique<Descriptors>(engine, getSceneDesignator()->framesInFlight, shaderStages);
}

auto Scene::createGraphicsPipeline(vk::RenderPass renderPass) const -> std::unique_ptr<const GraphicsPipeline>
{
    return std::make_unique<GraphicsPipeline>(kSquircle, engine, pipelineCache->pipelineCache, shaderStages, renderPass);
}

Scene::Scene(const engine::Engine & engine, const FileIo & fileIo, std::shared_ptr<const engine::PipelineCache> && pipelineCache, std::shared_ptr<const SceneDesignator> sceneDesignator)
    : engine{engine}
    , fileIo{fileIo}
    , pipelineCache{std::move(pipelineCache)}
    , sceneDesignator{std::move(sceneDesignator)}
    , vertexShader{"squircle.vert", engine, fileIo}
    , vertexShaderReflection{vertexShader, "main"}
    , fragmentShader{"squircle.frag", engine, fileIo}
    , fragmentShaderReflection{fragmentShader, "main"}
    , shaderStages{engine, vertexBufferBinding}
{
    init();
}

void Scene::init()
{
    ASSERT(sceneDesignator);
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

    vk::DescriptorSetLayoutCreateFlags descriptorSetLayoutCreateFlags = {};
    if (kUseDescriptorBuffer) {
        descriptorSetLayoutCreateFlags |= vk::DescriptorSetLayoutCreateFlagBits::eDescriptorBufferEXT;
    }
    shaderStages.createDescriptorSetLayouts(kSquircle, descriptorSetLayoutCreateFlags);
}

SceneManager::SceneManager(const engine::Engine & engine) : engine{engine}
{}

std::shared_ptr<const Scene> SceneManager::getOrCreateScene(SceneDesignator && sceneDesignator) const
{
    ASSERT(sceneDesignator.isValid());
    std::lock_guard<std::mutex> lock{mutex};

    auto pipelineCacheHolder = pipelineCache.lock();
    if (!pipelineCacheHolder) {
        pipelineCacheHolder = std::make_shared<engine::PipelineCache>(kSquircle, engine, fileIo);
        pipelineCache = pipelineCacheHolder;
    }

    auto & sceneDataWeakPtr = sceneData[sceneDesignator.path];
    auto sceneDataPtr = sceneDataWeakPtr.lock();
    if (sceneDataPtr) {
        SPDLOG_DEBUG("Old scene data {} reused", sceneDesignator.path);
    } else {
        auto sceneDataPtrMutable = std::make_shared<scene::Scene>();
        if (!sceneLoader.load(*sceneDataPtrMutable, QFileInfo{sceneDesignator.path})) {
            return {};
        }
        sceneDataPtr = std::move(sceneDataPtrMutable);
        sceneDataWeakPtr = sceneDataPtr;
        SPDLOG_DEBUG("Old scene data {} created", sceneDesignator.path);
    }

    auto sceneDesignatorPtr = std::make_shared<SceneDesignator>(std::move(sceneDesignator));
    auto & sceneWeakPtr = scenes[sceneDesignatorPtr];
    auto scenePtr = sceneWeakPtr.lock();
    if (scenePtr) {
        SPDLOG_DEBUG("Old scene {} reused", sceneDesignator);
    } else {
        scenePtr = Scene::make(engine, fileIo, std::move(pipelineCacheHolder), std::move(sceneDesignatorPtr));
        sceneWeakPtr = scenePtr;
        SPDLOG_DEBUG("New scene {} created", *scenePtr->getSceneDesignator());
    }
    return scenePtr;
}

}  // namespace viewer

size_t std::hash<viewer::SceneDesignatorPtr>::operator ()(const viewer::SceneDesignatorPtr & sceneDesignator) const noexcept
{
    ASSERT(sceneDesignator);
    size_t hash = 0;
    hash ^= std::hash<std::string>{}(sceneDesignator->token);
    hash ^= std::hash<std::filesystem::path>{}(sceneDesignator->path);
    hash ^= std::hash<uint32_t>{}(sceneDesignator->framesInFlight);
    return hash;
}
