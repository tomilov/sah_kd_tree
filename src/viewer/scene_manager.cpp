#include <engine/buffer.hpp>
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/pipeline_cache.hpp>
#include <engine/utils.hpp>
#include <engine/vma.hpp>
#include <format/vulkan.hpp>
#include <scene/scene.hpp>
#include <utils/assert.hpp>
#include <viewer/scene_manager.hpp>

#include <fmt/format.h>
#include <fmt/std.h>
#include <glm/ext/matrix_transform.hpp>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_format_traits.hpp>

#include <QFileInfo>
#include <QStandardPaths>

#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <cstddef>
#include <cstdint>

using namespace std::string_literals;
using namespace std::string_view_literals;

namespace viewer
{

namespace
{

template<vk::IndexType indexType>
using IndexCppType = typename vk::CppType<vk::IndexType, indexType>::Type;

const auto kRasterization = "rasterization"sv;
constexpr uint32_t kUniformBufferSet = 0;
constexpr uint32_t kTransformBuferSet = 0;
constexpr uint32_t kDisplaySamplerSet = 1;
const auto kUniformBufferName = "uniformBuffer"s;     // clazy:exclude=non-pod-global-static
const auto kTransformBuferName = "transformBuffer"s;  // clazy:exclude=non-pod-global-static
const auto kDisplaySamplerName = "display"s;          // clazy:exclude=non-pod-global-static

vk::Format indexTypeToFormat(vk::IndexType indexType)
{
    switch (indexType) {
    case vk::IndexType::eUint16: {
        return vk::Format::eR16Uint;
    }
    case vk::IndexType::eUint32: {
        return vk::Format::eR32Uint;
    }
    case vk::IndexType::eNoneKHR: {
        INVARIANT(false, "{} is not supported", indexType);
    }
    case vk::IndexType::eUint8EXT: {
        return vk::Format::eR8Uint;
    }
    }
    INVARIANT(false, "Unknown index type {}", indexType);
}

uint32_t indexTypeRank(vk::IndexType indexType)
{
    switch (indexType) {
    case vk::IndexType::eNoneKHR: {
        return 0;
    }
    case vk::IndexType::eUint8EXT: {
        return 1;
    }
    case vk::IndexType::eUint16: {
        return 2;
    }
    case vk::IndexType::eUint32: {
        return 3;
    }
    }
    INVARIANT(false, "{}", fmt::underlying(indexType));
}

bool indexTypeLess(vk::IndexType lhs, vk::IndexType rhs)
{
    return indexTypeRank(lhs) < indexTypeRank(rhs);
}

}  // namespace

Scene::GraphicsPipeline::GraphicsPipeline(std::string_view name, const engine::Context & context, vk::PipelineCache pipelineCache, const engine::ShaderStages & shaderStages, vk::RenderPass renderPass, bool useDescriptorBuffer)
    : pipelineLayout{name, context, shaderStages, renderPass}, pipelines{context, pipelineCache}
{
    pipelines.add(pipelineLayout, useDescriptorBuffer);
    pipelines.create();
}

std::shared_ptr<Scene> Scene::make(const engine::Context & context, const FileIo & fileIo, std::shared_ptr<const engine::PipelineCache> && pipelineCache, std::filesystem::path scenePath, scene::Scene && scene)
{
    return std::shared_ptr<Scene>{new Scene{context, fileIo, std::move(pipelineCache), std::move(scenePath), std::move(scene)}};
}

const std::filesystem::path & Scene::getScenePath() const
{
    return scenePath;
}

const scene::Scene & Scene::getScene() const
{
    return scene;
}

auto Scene::makeDescriptors(uint32_t framesInFlight) const -> Descriptors
{
    Descriptors descriptors;

    createInstances(descriptors);
    createVertexBuffer(descriptors);
    createUniformBuffers(framesInFlight, descriptors);

    for (const auto & [set, bindings] : shaderStages.setBindings) {
        INVARIANT(set == bindings.setIndex, "Descriptor sets ids are not sequential non-negative numbers: {}, {}", set, bindings.setIndex);
    }

    if (descriptorBufferEnabled) {
        createDescriptorBuffers(framesInFlight, descriptors);
    } else {
        createDescriptorSets(framesInFlight, descriptors);
    }

    descriptors.pushConstantRanges = shaderStages.pushConstantRanges;

    if (descriptorBufferEnabled) {
        fillDescriptorBuffers(framesInFlight, descriptors);
    } else {
        fillDescriptorSets(framesInFlight, descriptors);
    }

    return descriptors;
}

auto Scene::createGraphicsPipeline(vk::RenderPass renderPass) const -> std::unique_ptr<const GraphicsPipeline>
{
    return std::make_unique<GraphicsPipeline>(kRasterization, context, *pipelineCache, shaderStages, renderPass, descriptorBufferEnabled);
}

Scene::Scene(const engine::Context & context, const FileIo & fileIo, std::shared_ptr<const engine::PipelineCache> && pipelineCache, std::filesystem::path scenePath, scene::Scene && scene)
    : context{context}, fileIo{fileIo}, pipelineCache{std::move(pipelineCache)}, scenePath{std::move(scenePath)}, scene{std::move(scene)}, shaderStages{context, kVertexBufferBinding}, offscreenShaderStages{context, kVertexBufferBinding}
{
    init();
}

void Scene::init()
{
    uint32_t maxPushConstantsSize = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits.maxPushConstantsSize;
    INVARIANT(sizeof(PushConstants) <= maxPushConstantsSize, "{} ^ {}", sizeof(PushConstants), maxPushConstantsSize);

    const auto & device = context.getDevice();
    if (indexTypeUint8Enabled) {
        if (device.createInfoChain.get<vk::PhysicalDeviceIndexTypeUint8FeaturesEXT>().indexTypeUint8 == VK_FALSE) {
            INVARIANT(false, "");
        }
    }
    if (descriptorBufferEnabled) {
        if (device.createInfoChain.get<vk::PhysicalDeviceDescriptorBufferFeaturesEXT>().descriptorBuffer == VK_FALSE) {
            INVARIANT(false, "");
        }
    }
    if (multiDrawIndirectEnabled) {
        if (device.createInfoChain.get<vk::PhysicalDeviceFeatures2>().features.multiDrawIndirect == VK_FALSE) {
            INVARIANT(false, "");
        }
    }
    if (drawIndirectCountEnabled) {
        if (device.createInfoChain.get<vk::PhysicalDeviceVulkan12Features>().drawIndirectCount == VK_FALSE) {
            INVARIANT(false, "");
        }
    }

    ASSERT(!std::empty(scenePath));

    vk::DescriptorSetLayoutCreateFlags descriptorSetLayoutCreateFlags = {};
    if (descriptorBufferEnabled) {
        descriptorSetLayoutCreateFlags |= vk::DescriptorSetLayoutCreateFlagBits::eDescriptorBufferEXT;
    }

    const auto addShader = [this](std::string_view shaderName, std::string_view entryPoint = "main") -> const Shader &
    {
        auto [it, inserted] = shaders.emplace(std::piecewise_construct, std::tie(shaderName), std::tie(context, fileIo, shaderName, entryPoint));
        INVARIANT(inserted, "");
        return it->second;
    };

    {
        const auto & [vertexShader, vertexShaderReflection] = addShader("identity.vert");
        {
            INVARIANT(std::size(vertexShaderReflection.descriptorSetLayoutSetBindings) == 1, "");
            INVARIANT(vertexShaderReflection.descriptorSetLayoutSetBindings.contains(kTransformBuferSet), "");
            auto & descriptorSetLayoutBindings = vertexShaderReflection.descriptorSetLayoutSetBindings.at(kTransformBuferSet);
            INVARIANT(std::size(descriptorSetLayoutBindings) == 1, "{}", std::size(descriptorSetLayoutBindings));
            {
                auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(kTransformBuferName);
                INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 1, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eStorageBuffer, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eVertex, "");
                // INVARIANT(descriptorSetLayoutBindingReflection.size == sizeof(glm::mat4), "{} ^ {}", descriptorSetLayoutBindingReflection.size, sizeof(glm::mat4));
            }
            if ((false)) {
                auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(kUniformBufferName);
                INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 0, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eUniformBuffer, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eVertex, "");
                // INVARIANT(descriptorSetLayoutBindingReflection.size == sizeof(UniformBuffer), "{} ^ {}", descriptorSetLayoutBindingReflection.size, sizeof(UniformBuffer));
            }

            INVARIANT(vertexShaderReflection.pushConstantRange, "");
            {
                const auto & pushConstantRange = vertexShaderReflection.pushConstantRange.value();
                INVARIANT(pushConstantRange.stageFlags == vk::ShaderStageFlagBits::eVertex, "");
                INVARIANT(pushConstantRange.offset == offsetof(PushConstants, mvp), "");
                INVARIANT(pushConstantRange.size == sizeof(PushConstants::mvp), "");
            }
        }
        shaderStages.append(vertexShader, vertexShaderReflection);

        const auto & [fragmentShader, fragmentShaderReflection] = addShader("barycentric_color.frag");
        {
            INVARIANT(std::size(fragmentShaderReflection.descriptorSetLayoutSetBindings) == 1, "");
            INVARIANT(fragmentShaderReflection.descriptorSetLayoutSetBindings.contains(kUniformBufferSet), "");
            auto & descriptorSetLayoutBindings = fragmentShaderReflection.descriptorSetLayoutSetBindings.at(kUniformBufferSet);
            INVARIANT(std::size(descriptorSetLayoutBindings) == 1, "{}", std::size(descriptorSetLayoutBindings));
            auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(kUniformBufferName);
            INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 0, "");
            INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eUniformBuffer, "");
            INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
            INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eFragment, "");
            // INVARIANT(descriptorSetLayoutBindingReflection.size == sizeof(UniformBuffer), "{} ^ {}", descriptorSetLayoutBindingReflection.size, sizeof(UniformBuffer));

            INVARIANT(!fragmentShaderReflection.pushConstantRange, "");
        }
        shaderStages.append(fragmentShader, fragmentShaderReflection);
    }
    shaderStages.createDescriptorSetLayouts(kRasterization, descriptorSetLayoutCreateFlags);

    {
        const auto & [vertexShader, vertexShaderReflection] = addShader("fullscreen_rect.vert");
        {
            INVARIANT(std::size(vertexShaderReflection.descriptorSetLayoutSetBindings) == 1, "");
            INVARIANT(vertexShaderReflection.descriptorSetLayoutSetBindings.contains(kUniformBufferSet), "");
            auto & descriptorSetLayoutBindings = vertexShaderReflection.descriptorSetLayoutSetBindings.at(kUniformBufferSet);
            INVARIANT(std::size(descriptorSetLayoutBindings) == 1, "{}", std::size(descriptorSetLayoutBindings));
            {
                auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(kUniformBufferName);
                INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 0, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eUniformBuffer, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eVertex, "");
                // INVARIANT(descriptorSetLayoutBindingReflection.size == sizeof(UniformBuffer), "{} ^ {}", descriptorSetLayoutBindingReflection.size, sizeof(UniformBuffer));
            }

            INVARIANT(!vertexShaderReflection.pushConstantRange, "");
            if ((false)) {
                const auto & pushConstantRange = vertexShaderReflection.pushConstantRange.value();
                INVARIANT(pushConstantRange.stageFlags == vk::ShaderStageFlagBits::eVertex, "");
                INVARIANT(pushConstantRange.offset == offsetof(PushConstants, mvp), "");
                INVARIANT(pushConstantRange.size == sizeof(PushConstants::mvp), "");
            }
        }
        offscreenShaderStages.append(vertexShader, vertexShaderReflection);

        const auto & [fragmentShader, fragmentShaderReflection] = addShader("offscreen.frag");
        {
            INVARIANT(std::size(fragmentShaderReflection.descriptorSetLayoutSetBindings) == 2, "");
            {
                INVARIANT(fragmentShaderReflection.descriptorSetLayoutSetBindings.contains(kUniformBufferSet), "");
                auto & descriptorSetLayoutBindings = fragmentShaderReflection.descriptorSetLayoutSetBindings.at(kUniformBufferSet);
                INVARIANT(std::size(descriptorSetLayoutBindings) == 1, "{}", std::size(descriptorSetLayoutBindings));
                {
                    auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(kUniformBufferName);
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 0, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eUniformBuffer, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eFragment, "");
                    // INVARIANT(descriptorSetLayoutBindingReflection.size == sizeof(UniformBuffer), "{} ^ {}", descriptorSetLayoutBindingReflection.size, sizeof(UniformBuffer));
                }
            }
            {
                INVARIANT(fragmentShaderReflection.descriptorSetLayoutSetBindings.contains(kDisplaySamplerSet), "");
                auto & descriptorSetLayoutBindings = fragmentShaderReflection.descriptorSetLayoutSetBindings.at(kDisplaySamplerSet);
                INVARIANT(std::size(descriptorSetLayoutBindings) == 1, "{}", std::size(descriptorSetLayoutBindings));
                {
                    auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(kDisplaySamplerName);
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 0, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eCombinedImageSampler, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eFragment, "");
                }
            }

            INVARIANT(!fragmentShaderReflection.pushConstantRange, "");
            if (fragmentShaderReflection.pushConstantRange) {
                const auto & pushConstantRange = fragmentShaderReflection.pushConstantRange.value();
                INVARIANT(pushConstantRange.stageFlags == vk::ShaderStageFlagBits::eFragment, "");
                INVARIANT(pushConstantRange.offset == offsetof(PushConstants, x), "");
                INVARIANT(pushConstantRange.size == sizeof(PushConstants::x), "");
            }
        }
        offscreenShaderStages.append(fragmentShader, fragmentShaderReflection);
    }
    offscreenShaderStages.createDescriptorSetLayouts(kRasterization, descriptorSetLayoutCreateFlags);
}

size_t Scene::getDescriptorSize(vk::DescriptorType descriptorType) const
{
    const auto & physicalDevice = context.getPhysicalDevice();
    const vk::Bool32 robustBufferAccess = physicalDevice.features2Chain.get<vk::PhysicalDeviceFeatures2>().features.robustBufferAccess;
    const auto & physicalDeviceDescriptorBufferProperties = physicalDevice.properties2Chain.get<vk::PhysicalDeviceDescriptorBufferPropertiesEXT>();
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

vk::DeviceSize Scene::getMinAlignment() const
{
    const auto & physicalDeviceLimits = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits;
    return physicalDeviceLimits.nonCoherentAtomSize;
}

void Scene::createVertexBuffer(Descriptors & descriptors) const
{
    const auto minAlignment = getMinAlignment();
    const auto & vma = context.getMemoryAllocator();

    if (!shaderStages.vertexInputState) {
        return;
    }
    vk::DeviceSize vertexSize = 0;
    for (const auto & vertexInputAttributeDescription : shaderStages.vertexInputState.value().vertexInputAttributeDescriptions) {
        vertexSize += vk::blockSize(vertexInputAttributeDescription.format);
    }

    if (vertexSize == 0) {
        return;
    }
    INVARIANT(sizeof(scene::VertexAttributes) == vertexSize, "{} != {}", sizeof(scene::VertexAttributes), vertexSize);

    vk::BufferCreateInfo vertexBufferCreateInfo;
    vertexBufferCreateInfo.size = scene.vertices.getCount() * vertexSize;
    vertexBufferCreateInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer;
    descriptors.vertexBuffer.emplace(vma.createStagingBuffer("Vertices"s, vertexBufferCreateInfo, minAlignment));

    constexpr vk::MemoryPropertyFlags kMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    auto memoryPropertyFlags = descriptors.vertexBuffer.value().base().getMemoryPropertyFlags();
    INVARIANT((memoryPropertyFlags & kMemoryPropertyFlags) == kMemoryPropertyFlags, "Failed to allocate vertex buffer in {} memory, got {} memory", kMemoryPropertyFlags, memoryPropertyFlags);

    auto mappedVertexBuffer = descriptors.vertexBuffer.value().map();
    ASSERT(scene.vertices.getCount() == mappedVertexBuffer.getCount());
    if (std::copy_n(scene.vertices.begin(), scene.vertices.getCount(), mappedVertexBuffer.begin()) != mappedVertexBuffer.end()) {
        ASSERT(false);
    }
}

void Scene::createInstances(Descriptors & descriptors) const
{
    const auto minAlignment = getMinAlignment();
    const auto & vma = context.getMemoryAllocator();

    size_t sceneMeshCount = std::size(scene.meshes);

    std::vector<std::vector<glm::mat4>> transforms;  // [Scene::meshes index][instance index]
    transforms.resize(sceneMeshCount);
    descriptors.instances.resize(sceneMeshCount);

    const auto collectNodeInfos = [this, &descriptors, &transforms](const auto & collectNodeInfos, const scene::Node & sceneNode, glm::mat4 transform) -> void
    {
        transform *= sceneNode.transform;
        for (size_t m : sceneNode.meshes) {
            ++descriptors.instances.at(m).instanceCount;
            transforms.at(m).push_back(transform);
        }
        for (size_t sceneNodeChild : sceneNode.children) {
            collectNodeInfos(collectNodeInfos, scene.nodes.at(sceneNodeChild), transform);
        }
    };
    collectNodeInfos(collectNodeInfos, scene.nodes.front(), glm::identity<glm::mat4>());

    auto sceneIndices = scene.indices.begin();

    vk::DeviceSize indexBufferSize = 0;
    {
        vk::IndexType maxIndexType = vk::IndexType::eNoneKHR;
        for (size_t m = 0; m < sceneMeshCount; ++m) {
            const scene::Mesh & sceneMesh = scene.meshes.at(m);
            auto & instance = descriptors.instances.at(m);

            instance.vertexOffset = utils::autoCast(sceneMesh.vertexOffset);

            auto & indexType = descriptors.indexTypes.emplace_back();
            if (sceneMesh.indexCount == 0) {
                indexType = vk::IndexType::eNoneKHR;
                continue;
            }

            instance.indexCount = utils::autoCast(sceneMesh.indexCount);

            auto firstIndex = std::next(sceneIndices, sceneMesh.indexOffset);
            uint32_t maxIndex = *std::max_element(firstIndex, std::next(firstIndex, sceneMesh.indexCount));
            if (indexTypeUint8Enabled && (maxIndex <= std::numeric_limits<IndexCppType<vk::IndexType::eUint8EXT>>::max())) {
                indexType = vk::IndexType::eUint8EXT;
            } else if (maxIndex <= std::numeric_limits<IndexCppType<vk::IndexType::eUint16>>::max()) {
                indexType = vk::IndexType::eUint16;
            } else {
                indexType = vk::IndexType::eUint32;
            }
            if (indexTypeLess(maxIndexType, indexType)) {
                maxIndexType = indexType;
            }
        }

        if (maxIndexType != vk::IndexType::eNoneKHR) {
            for (size_t m = 0; m < sceneMeshCount; ++m) {
                auto & indexType = descriptors.indexTypes.at(m);
                if (indexType == vk::IndexType::eNoneKHR) {
                    continue;
                }
                if (multiDrawIndirectEnabled) {
                    indexType = maxIndexType;
                }
                vk::DeviceSize formatSize = vk::blockSize(indexTypeToFormat(indexType));
                indexBufferSize = engine::alignedSize(indexBufferSize, formatSize);
                auto & instance = descriptors.instances.at(m);
                instance.firstIndex = utils::autoCast(indexBufferSize / formatSize);
                indexBufferSize += instance.indexCount * formatSize;
            }
        }
    }

    if (indexBufferSize > 0) {
        {
            vk::BufferCreateInfo indexBufferCreateInfo;
            indexBufferCreateInfo.size = indexBufferSize;
            indexBufferCreateInfo.usage = vk::BufferUsageFlagBits::eIndexBuffer;
            descriptors.indexBuffer.emplace(vma.createStagingBuffer("Indices"s, indexBufferCreateInfo, minAlignment));

            constexpr vk::MemoryPropertyFlags kMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
            auto memoryPropertyFlags = descriptors.indexBuffer.value().getMemoryPropertyFlags();
            INVARIANT((memoryPropertyFlags & kMemoryPropertyFlags) == kMemoryPropertyFlags, "Failed to allocate index buffer in {} memory, got {} memory", kMemoryPropertyFlags, memoryPropertyFlags);
        }

        {
            auto mappedIndexBuffer = descriptors.indexBuffer.value().map();
            auto indices = mappedIndexBuffer.data();
            for (size_t m = 0; m < sceneMeshCount; ++m) {
                const auto & instance = descriptors.instances.at(m);

                ASSERT(std::size(transforms.at(m)) == instance.instanceCount);

                uint32_t sceneIndexOffset = scene.meshes.at(m).indexOffset;
                const auto convertCopy = [&instance, sceneIndices, sceneIndexOffset](auto indices)
                {
                    auto indexIn = std::next(sceneIndices, sceneIndexOffset);
                    auto indexOut = std::next(indices, instance.firstIndex);
                    for (uint32_t i = 0; i < instance.indexCount; ++i) {
                        *indexOut++ = utils::autoCast(*indexIn++);
                    }
                };
                switch (descriptors.indexTypes.at(m)) {
                case vk::IndexType::eNoneKHR: {
                    // no indices have to be copied
                    break;
                }
                case vk::IndexType::eUint8EXT: {
                    convertCopy(utils::safeCast<IndexCppType<vk::IndexType::eUint8EXT> *>(indices));
                    break;
                }
                case vk::IndexType::eUint16: {
                    convertCopy(utils::safeCast<IndexCppType<vk::IndexType::eUint16> *>(indices));
                    break;
                }
                case vk::IndexType::eUint32: {
                    convertCopy(utils::safeCast<IndexCppType<vk::IndexType::eUint32> *>(indices));
                    break;
                }
                }
            }
        }
    }

    uint32_t totalInstanceCount = 0;
    for (auto & instance : descriptors.instances) {
        instance.firstInstance = totalInstanceCount;
        totalInstanceCount += instance.instanceCount;
    }

    if (multiDrawIndirectEnabled) {
        descriptors.drawCount = std::size(descriptors.instances);

        if (drawIndirectCountEnabled) {
            vk::BufferCreateInfo drawCountBufferCreateInfo;
            drawCountBufferCreateInfo.size = sizeof(uint32_t);
            drawCountBufferCreateInfo.usage = vk::BufferUsageFlagBits::eIndirectBuffer;
            descriptors.drawCountBuffer.emplace(vma.createStagingBuffer("DrawCount"s, drawCountBufferCreateInfo, minAlignment));

            auto mappedDrawCountBuffer = descriptors.drawCountBuffer.value().map();
            mappedDrawCountBuffer.at(0) = descriptors.drawCount;
        }

        {
            vk::BufferCreateInfo instanceBufferCreateInfo;
            constexpr uint32_t kSize = sizeof(vk::DrawIndexedIndirectCommand);
            instanceBufferCreateInfo.size = descriptors.drawCount * kSize;
            instanceBufferCreateInfo.usage = vk::BufferUsageFlagBits::eIndirectBuffer;
            descriptors.instanceBuffer.emplace(vma.createStagingBuffer("Instances"s, instanceBufferCreateInfo, minAlignment));

            auto mappedInstanceBuffer = descriptors.instanceBuffer.value().map();
            auto end = std::copy(std::cbegin(descriptors.instances), std::cend(descriptors.instances), mappedInstanceBuffer.begin());
            INVARIANT(end == mappedInstanceBuffer.end(), "");
        }
    }

    {
        vk::BufferCreateInfo transformBufferCreateInfo;
        transformBufferCreateInfo.size = totalInstanceCount * sizeof(glm::mat4);
        transformBufferCreateInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer;
        if (descriptorBufferEnabled) {
            transformBufferCreateInfo.usage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
        }
        descriptors.transformBuffer.emplace(vma.createStagingBuffer("Transformations"s, transformBufferCreateInfo, minAlignment));

        constexpr vk::MemoryPropertyFlags kMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
        auto memoryPropertyFlags = descriptors.transformBuffer.value().base().getMemoryPropertyFlags();
        INVARIANT((memoryPropertyFlags & kMemoryPropertyFlags) == kMemoryPropertyFlags, "Failed to allocate transformation buffer in {} memory, got {} memory", kMemoryPropertyFlags, memoryPropertyFlags);
    }

    {
        auto mappedTransformBuffer = descriptors.transformBuffer.value().map();
        auto t = mappedTransformBuffer.begin();
        for (const auto & instanceTransforms : transforms) {
            ASSERT(mappedTransformBuffer.end() != t);
            t = std::copy(std::cbegin(instanceTransforms), std::cend(instanceTransforms), t);
        }
        ASSERT(t == mappedTransformBuffer.end());
    }
}

void Scene::createUniformBuffers(uint32_t framesInFlight, Descriptors & descriptors) const
{
    const auto minAlignment = getMinAlignment();
    const auto & vma = context.getMemoryAllocator();

    vk::BufferCreateInfo uniformBufferCreateInfo;
    auto alignment = std::max(minAlignment, utils::safeCast<vk::DeviceSize>(alignof(UniformBuffer)));
    auto alignedSize = engine::alignedSize(sizeof(UniformBuffer), alignment);
    uniformBufferCreateInfo.size = alignedSize * framesInFlight;
    uniformBufferCreateInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;
    if (descriptorBufferEnabled) {
        uniformBufferCreateInfo.usage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
    }
    constexpr vk::MemoryPropertyFlags kMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    auto uniformBufferName = fmt::format("Uniform buffer");
    descriptors.uniformBuffer.emplace(vma.createStagingBuffer(uniformBufferName, uniformBufferCreateInfo, alignment), framesInFlight);

    auto memoryPropertyFlags = descriptors.uniformBuffer.value().base().getMemoryPropertyFlags();
    INVARIANT((memoryPropertyFlags & kMemoryPropertyFlags) == kMemoryPropertyFlags, "Failed to allocate uniform buffer in {} memory, got {} memory", kMemoryPropertyFlags, memoryPropertyFlags);
}

void Scene::createDescriptorSets(uint32_t framesInFlight, Descriptors & descriptors) const
{
    descriptors.descriptorPool = std::make_unique<engine::DescriptorPool>(kRasterization, context, framesInFlight, shaderStages);
    for (uint32_t i = 0; i < framesInFlight; ++i) {
        descriptors.descriptorSets.emplace_back(kRasterization, context, shaderStages, *descriptors.descriptorPool);
    }
}

void Scene::createDescriptorBuffers(uint32_t framesInFlight, Descriptors & descriptors) const
{
    const auto minAlignment = getMinAlignment();

    constexpr vk::MemoryPropertyFlags kRequiredMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    const auto descriptorBufferOffsetAlignment = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceDescriptorBufferPropertiesEXT>().descriptorBufferOffsetAlignment;
    descriptors.descriptorSetBuffers.reserve(std::size(shaderStages.descriptorSetLayouts));
    auto set = std::cbegin(shaderStages.setBindings);
    for (const auto & descriptorSetLayout : shaderStages.descriptorSetLayouts) {
        vk::BufferCreateInfo descriptorBufferCreateInfo;
        descriptorBufferCreateInfo.usage = vk::BufferUsageFlagBits::eShaderDeviceAddress;
        auto alignment = std::max(descriptorBufferOffsetAlignment, minAlignment);
        descriptorBufferCreateInfo.size = engine::alignedSize(context.getDevice().getDevice().getDescriptorSetLayoutSizeEXT(descriptorSetLayout, context.getDispatcher()), alignment) * framesInFlight;
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
        auto descriptorSetBuffer = context.getMemoryAllocator().createStagingBuffer(descriptorBufferName, descriptorBufferCreateInfo, alignment);

        auto memoryPropertyFlags = descriptorSetBuffer.getMemoryPropertyFlags();
        INVARIANT((memoryPropertyFlags & kRequiredMemoryPropertyFlags) == kRequiredMemoryPropertyFlags, "Failed to allocate descriptor buffer in {} memory, got {} memory", kRequiredMemoryPropertyFlags, memoryPropertyFlags);

        descriptors.descriptorBufferBindingInfos.push_back(descriptorSetBuffer.getDescriptorBufferBindingInfo());

        descriptors.descriptorSetBuffers.emplace_back(std::move(descriptorSetBuffer), framesInFlight);

        ++set;
    }
}

void Scene::fillDescriptorSets(uint32_t framesInFlight, Descriptors & descriptors) const
{
    const auto & dispatcher = context.getDispatcher();
    const auto & device = context.getDevice();

    std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
    writeDescriptorSets.reserve(framesInFlight * 2);
    std::vector<vk::DescriptorBufferInfo> descriptorBufferInfos;
    descriptorBufferInfos.reserve(framesInFlight + 1);

    {
        auto setBindings = shaderStages.setBindings.find(kUniformBufferSet);
        INVARIANT(setBindings != std::end(shaderStages.setBindings), "Set {} for buffer {} is not found", kUniformBufferSet, kUniformBufferName);
        if (const auto * uniformBufferBinding = setBindings->second.getBinding(kUniformBufferName)) {
            uint32_t uniformBufferSetIndex = setBindings->second.setIndex;

            for (uint32_t i = 0; i < framesInFlight; ++i) {
                ASSERT(std::size(descriptorBufferInfos) < descriptorBufferInfos.capacity());
                descriptorBufferInfos.push_back(descriptors.uniformBuffer.value().getDescriptorBufferInfo(i));
            }

            for (uint32_t i = 0; i < framesInFlight; ++i) {
                ASSERT(std::size(writeDescriptorSets) < writeDescriptorSets.capacity());
                auto & writeDescriptorSet = writeDescriptorSets.emplace_back();

                writeDescriptorSet.dstSet = descriptors.descriptorSets.at(i).getDescriptorSets().at(uniformBufferSetIndex);
                writeDescriptorSet.dstBinding = uniformBufferBinding->binding;
                writeDescriptorSet.dstArrayElement = 0;  // not an array
                writeDescriptorSet.descriptorType = uniformBufferBinding->descriptorType;
                writeDescriptorSet.setBufferInfo(descriptorBufferInfos.at(i));
            }
        }
    }

    {
        ASSERT(descriptors.transformBuffer);
        auto setBindings = shaderStages.setBindings.find(kTransformBuferSet);
        INVARIANT(setBindings != std::end(shaderStages.setBindings), "Set {} for buffer {} is not found", kTransformBuferSet, kTransformBuferName);
        if (const auto * transformBufferBinding = setBindings->second.getBinding(kTransformBuferName)) {
            uint32_t transformBufferSetIndex = setBindings->second.setIndex;

            ASSERT(std::size(descriptorBufferInfos) < descriptorBufferInfos.capacity());
            auto & descriptorBufferInfo = descriptorBufferInfos.emplace_back(descriptors.transformBuffer.value().base().getDescriptorBufferInfo());

            for (uint32_t i = 0; i < framesInFlight; ++i) {
                ASSERT(std::size(writeDescriptorSets) < writeDescriptorSets.capacity());
                auto & writeDescriptorSet = writeDescriptorSets.emplace_back();

                writeDescriptorSet.dstSet = descriptors.descriptorSets.at(i).getDescriptorSets().at(transformBufferSetIndex);
                writeDescriptorSet.dstBinding = transformBufferBinding->binding;
                writeDescriptorSet.dstArrayElement = 0;  // not an array
                writeDescriptorSet.descriptorType = transformBufferBinding->descriptorType;
                writeDescriptorSet.setBufferInfo(descriptorBufferInfo);
            }
        }
    }

    device.getDevice().updateDescriptorSets(writeDescriptorSets, nullptr, dispatcher);
}

void Scene::fillDescriptorBuffers(uint32_t framesInFlight, Descriptors & descriptors) const
{
    ASSERT(descriptors.transformBuffer);

    const auto & dispatcher = context.getDispatcher();
    const auto & device = context.getDevice();

    for (const auto & [set, bindings] : shaderStages.setBindings) {
        const auto setIndex = bindings.setIndex;
        const auto & descriptorSetLayout = shaderStages.descriptorSetLayouts.at(setIndex);
        const auto & descriptorSetBuffer = descriptors.descriptorSetBuffers.at(setIndex);
        auto mappedDescriptorSetBuffer = descriptorSetBuffer.map();
        for (uint32_t currentFrameSlot = 0; currentFrameSlot < framesInFlight; ++currentFrameSlot) {
            for (uint32_t b = 0; b < std::size(bindings.bindings); ++b) {
                const auto & binding = bindings.bindings.at(b);
                vk::DescriptorGetInfoEXT descriptorGetInfo;
                descriptorGetInfo.type = binding.descriptorType;
                vk::DescriptorAddressInfoEXT descriptorAddressInfo;
                const auto & bindingName = bindings.bindingNames.at(b);
                if (bindingName == kUniformBufferName) {
                    ASSERT(binding.descriptorType == vk::DescriptorType::eUniformBuffer);
                    const auto & u = descriptors.uniformBuffer.value();
                    descriptorAddressInfo = u.base().getDescriptorAddressInfo();
                    descriptorAddressInfo.range /= framesInFlight;
                    descriptorAddressInfo.address += descriptorAddressInfo.range * currentFrameSlot;
                    descriptorGetInfo.data.pUniformBuffer = &descriptorAddressInfo;
                } else if (bindingName == kTransformBuferName) {
                    ASSERT(binding.descriptorType == vk::DescriptorType::eStorageBuffer);
                    descriptorAddressInfo = descriptors.transformBuffer.value().base().getDescriptorAddressInfo();
                    descriptorGetInfo.data.pStorageBuffer = &descriptorAddressInfo;
                } else {
                    INVARIANT(false, "Cannot find descriptor for binding '{}'", bindingName);
                }
                vk::DeviceSize descriptorSize = getDescriptorSize(binding.descriptorType);
                vk::DeviceSize bindingOffset = device.getDevice().getDescriptorSetLayoutBindingOffsetEXT(descriptorSetLayout, binding.binding, dispatcher);
                ASSERT(bindingOffset + descriptorSize <= descriptorSetBuffer.getElementSize());
                device.getDevice().getDescriptorEXT(&descriptorGetInfo, descriptorSize, &mappedDescriptorSetBuffer.at(currentFrameSlot) + bindingOffset, dispatcher);
            }
        }
    }
}

SceneManager::SceneManager(const engine::Context & context) : context{context}
{}

std::shared_ptr<const Scene> SceneManager::getOrCreateScene(std::filesystem::path scenePath) const
{
    ASSERT(!std::empty(scenePath));
    auto & w = scenes[scenePath];
    auto p = w.lock();
    if (p) {
        SPDLOG_DEBUG("Old scene {} reused", scenePath);
    } else {
        scene::Scene scene;
        if ((true)) {
            auto cacheLocation = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
            if (!scene_loader::cachingLoad(scene, QFileInfo{scenePath}, cacheLocation)) {
                return nullptr;
            }
        } else {
            if (!scene_loader::load(scene, QFileInfo{scenePath})) {
                return nullptr;
            }
        }
        p = Scene::make(context, fileIo, getOrCreatePipelineCache(), std::move(scenePath), std::move(scene));
        w = p;
        SPDLOG_DEBUG("New scene {} created", p->getScenePath());
    }
    return p;
}

std::shared_ptr<const engine::PipelineCache> SceneManager::getOrCreatePipelineCache() const
{
    auto p = pipelineCache.lock();
    if (p) {
        SPDLOG_DEBUG("Old pipeline cache reused");
    } else {
        p = std::make_shared<engine::PipelineCache>(kRasterization, context, fileIo);
        pipelineCache = p;
        SPDLOG_DEBUG("New pipeline cache created");
    }
    return p;
}

}  // namespace viewer
