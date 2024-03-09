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
#include <scene_data/scene_data.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
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
const auto kDisplaySamplerName = "display"s;  // clazy:exclude=non-pod-global-static

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

auto Scene::SceneDescriptors::Resources::getDescriptorSetInfos() const -> DescriptorSetInfos
{
    return {
        {kTransformBuferName, vk::DescriptorType::eStorageBuffer, transformBuffer.base().getDescriptorBufferInfo()},
    };
}

auto Scene::SceneDescriptors::Resources::getDescriptorBufferInfos() const -> DescriptorBufferInfos
{
    return {
        {kTransformBuferName, vk::DescriptorType::eStorageBuffer, transformBuffer.base().getDescriptorAddressInfo()},
    };
}

auto Scene::FrameDescriptors::Resources::getDescriptorSetInfos() const -> DescriptorSetInfos
{
    return {
        {kUniformBufferName, vk::DescriptorType::eUniformBuffer, uniformBuffer.base().getDescriptorBufferInfo()},
    };
}

auto Scene::FrameDescriptors::Resources::getDescriptorBufferInfos() const -> DescriptorBufferInfos
{
    return {
        {kUniformBufferName, vk::DescriptorType::eUniformBuffer, uniformBuffer.base().getDescriptorAddressInfo()},
    };
}

Scene::GraphicsPipeline::GraphicsPipeline(std::string_view name, const engine::Context & context, vk::PipelineCache pipelineCache, const engine::ShaderStages & shaderStages, vk::RenderPass renderPass, bool useDescriptorBuffer)
    : pipelineLayout{name, context, shaderStages, renderPass}, pipelines{context, pipelineCache}
{
    pipelines.add(pipelineLayout, useDescriptorBuffer);
    pipelines.create();
}

std::unique_ptr<Scene> Scene::make(const engine::Context & context, const FileIo & fileIo, std::shared_ptr<const engine::PipelineCache> pipelineCache, std::filesystem::path scenePath, scene_data::SceneData && sceneData)
{
    return std::unique_ptr<Scene>{new Scene{context, fileIo, std::move(pipelineCache), std::move(scenePath), std::move(sceneData)}};
}

const std::filesystem::path & Scene::getScenePath() const
{
    return scenePath;
}

const scene_data::SceneData & Scene::getScenedData() const
{
    return sceneData;
}

auto Scene::makeSceneDescriptors() const -> SceneDescriptors
{
    std::vector<std::vector<glm::mat4>> transforms(std::size(sceneData.meshes));  // [Scene::meshes index][instance index]
    std::vector<vk::DrawIndexedIndirectCommand> instances(std::size(sceneData.meshes));
    {
        const auto collectNodeInfos = [this, &transforms, &instances](const auto & collectNodeInfos, const scene_data::Node & sceneNode, glm::mat4 transform) -> void
        {
            transform *= sceneNode.transform;
            for (size_t m : sceneNode.meshes) {
                transforms.at(m).push_back(transform);
                ++instances.at(m).instanceCount;
            }
            for (size_t sceneNodeChild : sceneNode.children) {
                collectNodeInfos(collectNodeInfos, sceneData.nodes.at(sceneNodeChild), transform);
            }
        };
        collectNodeInfos(collectNodeInfos, sceneData.nodes.front(), glm::identity<glm::mat4>());
    }

    std::vector<vk::IndexType> indexTypes;
    vk::DeviceSize indexBufferSize = 0;
    uint32_t totalInstanceCount = 0;
    {
        vk::IndexType maxIndexType = vk::IndexType::eNoneKHR;
        for (size_t m = 0; m < std::size(sceneData.meshes); ++m) {
            const scene_data::Mesh & sceneMesh = sceneData.meshes.at(m);
            auto & instance = instances.at(m);

            instance.vertexOffset = utils::autoCast(sceneMesh.vertexOffset);

            auto & indexType = indexTypes.emplace_back();
            if (sceneMesh.indexCount == 0) {
                indexType = vk::IndexType::eNoneKHR;
                continue;
            }

            instance.indexCount = utils::autoCast(sceneMesh.indexCount);

            auto firstIndex = std::next(sceneData.indices.begin(), sceneMesh.indexOffset);
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
            for (size_t m = 0; m < std::size(sceneData.meshes); ++m) {
                auto & indexType = indexTypes.at(m);
                if (indexType == vk::IndexType::eNoneKHR) {
                    continue;
                }
                if (multiDrawIndirectEnabled) {
                    indexType = maxIndexType;
                }
                vk::DeviceSize formatSize = vk::blockSize(indexTypeToFormat(indexType));
                indexBufferSize = engine::alignedSize(indexBufferSize, formatSize);
                auto & instance = instances.at(m);
                instance.firstIndex = utils::autoCast(indexBufferSize / formatSize);
                indexBufferSize += instance.indexCount * formatSize;
            }
        }

        for (auto & instance : instances) {
            instance.firstInstance = totalInstanceCount;
            totalInstanceCount += instance.instanceCount;
        }
    }

    std::optional<engine::Buffer<void>> indexBuffer;
    if (indexBufferSize > 0) {
        {
            vk::BufferCreateInfo indexBufferCreateInfo;
            indexBufferCreateInfo.size = indexBufferSize;
            indexBufferCreateInfo.usage = vk::BufferUsageFlagBits::eIndexBuffer;
            indexBuffer.emplace(context.getMemoryAllocator().createStagingBuffer("Indices"s, indexBufferCreateInfo, getMinAlignment()));

            constexpr vk::MemoryPropertyFlags kMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
            auto memoryPropertyFlags = indexBuffer.value().getMemoryPropertyFlags();
            INVARIANT((memoryPropertyFlags & kMemoryPropertyFlags) == kMemoryPropertyFlags, "Failed to allocate index buffer in {} memory, got {} memory", kMemoryPropertyFlags, memoryPropertyFlags);
        }

        {
            auto mappedIndexBuffer = indexBuffer.value().map();
            auto indices = mappedIndexBuffer.data();
            for (size_t m = 0; m < std::size(sceneData.meshes); ++m) {
                const auto & instance = instances.at(m);

                ASSERT(std::size(transforms.at(m)) == instance.instanceCount);

                uint32_t sceneIndexOffset = sceneData.meshes.at(m).indexOffset;
                const auto convertCopy = [this, &instance, sceneIndexOffset](auto indices)
                {
                    auto indexIn = std::next(sceneData.indices.begin(), sceneIndexOffset);
                    auto indexOut = std::next(indices, instance.firstIndex);
                    for (uint32_t i = 0; i < instance.indexCount; ++i) {
                        *indexOut++ = utils::autoCast(*indexIn++);
                    }
                };
                switch (indexTypes.at(m)) {
                case vk::IndexType::eNoneKHR: {
                    // no indices have to be copied
                    break;
                }
                case vk::IndexType::eUint8EXT: {
                    convertCopy(static_cast<IndexCppType<vk::IndexType::eUint8EXT> *>(indices));
                    break;
                }
                case vk::IndexType::eUint16: {
                    convertCopy(static_cast<IndexCppType<vk::IndexType::eUint16> *>(indices));
                    break;
                }
                case vk::IndexType::eUint32: {
                    convertCopy(static_cast<IndexCppType<vk::IndexType::eUint32> *>(indices));
                    break;
                }
                }
            }
        }
    }

    uint32_t drawCount = utils::autoCast(std::size(instances));

    std::optional<engine::Buffer<uint32_t>> drawCountBuffer;
    std::optional<engine::Buffer<vk::DrawIndexedIndirectCommand>> instanceBuffer;
    if (multiDrawIndirectEnabled) {
        if (drawIndirectCountEnabled) {
            vk::BufferCreateInfo drawCountBufferCreateInfo;
            drawCountBufferCreateInfo.size = sizeof(uint32_t);
            drawCountBufferCreateInfo.usage = vk::BufferUsageFlagBits::eIndirectBuffer;
            drawCountBuffer.emplace(context.getMemoryAllocator().createStagingBuffer("DrawCount"s, drawCountBufferCreateInfo, getMinAlignment()));

            auto mappedDrawCountBuffer = drawCountBuffer.value().map();
            mappedDrawCountBuffer.at(0) = drawCount;
        }

        {
            vk::BufferCreateInfo instanceBufferCreateInfo;
            constexpr uint32_t kSize = sizeof(vk::DrawIndexedIndirectCommand);
            instanceBufferCreateInfo.size = drawCount * kSize;
            instanceBufferCreateInfo.usage = vk::BufferUsageFlagBits::eIndirectBuffer;
            instanceBuffer.emplace(context.getMemoryAllocator().createStagingBuffer("Instances"s, instanceBufferCreateInfo, getMinAlignment()));

            auto mappedInstanceBuffer = instanceBuffer.value().map();
            auto end = std::copy(std::cbegin(instances), std::cend(instances), mappedInstanceBuffer.begin());
            INVARIANT(end == mappedInstanceBuffer.end(), "");
        }
    }

    auto transformBuffer = createTransformBuffer(totalInstanceCount, transforms);

    SceneDescriptors::Resources resources = {
        .transforms = std::move(transforms),
        .instances = std::move(instances),
        .indexTypes = std::move(indexTypes),
        .indexBuffer = std::move(indexBuffer),
        .drawCount = drawCount,
        .drawCountBuffer = std::move(drawCountBuffer),
        .instanceBuffer = std::move(instanceBuffer),
        .transformBuffer = std::move(transformBuffer),
        .vertexBuffer = createVertexBuffer(),
    };
    if (descriptorBufferEnabled) {
        auto descriptorBuffer = createDescriptorBuffer(sceneShaderStages, SceneDescriptors::kSet);
        fillDescriptorBuffer(descriptorBuffer, sceneShaderStages, SceneDescriptors::kSet, resources.getDescriptorBufferInfos());
        return {
            .resources = std::move(resources),
            .descriptors = std::move(descriptorBuffer),
        };
    } else {
        auto descriptorSet = createDescriptorSet(sceneShaderStages, SceneDescriptors::kSet);
        fillDescriptorSet(descriptorSet, sceneShaderStages, SceneDescriptors::kSet, resources.getDescriptorSetInfos());
        return {
            .resources = std::move(resources),
            .descriptors = std::move(descriptorSet),
        };
    }
}

auto Scene::makeFrameDescriptors() const -> FrameDescriptors
{
    for (const auto & [set, bindings] : sceneShaderStages.setBindings) {
        INVARIANT(set == bindings.setIndex, "Descriptor set ids are not sequential non-negative numbers: {}, {}", set, bindings.setIndex);
    }
    FrameDescriptors::Resources resources = {
        .uniformBuffer = createUniformBuffer(),
    };
    if (descriptorBufferEnabled) {
        auto descriptorBuffer = createDescriptorBuffer(sceneShaderStages, FrameDescriptors::kSet);
        fillDescriptorBuffer(descriptorBuffer, sceneShaderStages, FrameDescriptors::kSet, resources.getDescriptorBufferInfos());
        return {
            .resources = std::move(resources),
            .descriptors = std::move(descriptorBuffer),
        };
    } else {
        auto descriptorSet = createDescriptorSet(sceneShaderStages, FrameDescriptors::kSet);
        fillDescriptorSet(descriptorSet, sceneShaderStages, FrameDescriptors::kSet, resources.getDescriptorSetInfos());
        return {
            .resources = std::move(resources),
            .descriptors = std::move(descriptorSet),
        };
    }
}

const std::vector<vk::PushConstantRange> & Scene::getPushConstantRanges() const
{
    return sceneShaderStages.pushConstantRanges;
}

auto Scene::createGraphicsPipeline(vk::RenderPass renderPass, PipelineKind pipelineKind) const -> std::unique_ptr<GraphicsPipeline>
{
    const engine::ShaderStages * shaderStages = nullptr;
    switch (pipelineKind) {
    case PipelineKind::kScenePipeline: {
        shaderStages = &sceneShaderStages;
        break;
    }
    case PipelineKind::kDisplayPipeline: {
        shaderStages = &offscreenShaderStages;
        break;
    }
    }
    ASSERT(shaderStages);
    return std::make_unique<GraphicsPipeline>(kRasterization, context, *pipelineCache, *shaderStages, renderPass, descriptorBufferEnabled);
}

void Scene::check()
{
    ASSERT(!std::empty(scenePath));

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
}

auto Scene::addShader(std::string_view shaderName, std::string_view entryPoint) -> const Shader &
{
    auto [it, inserted] = shaders.emplace(std::piecewise_construct, std::tie(shaderName), std::tie(context, fileIo, shaderName, entryPoint));
    ASSERT_MSG(inserted, "");
    return it->second;
}

void Scene::addShaders()
{
    vk::DescriptorSetLayoutCreateFlags descriptorSetLayoutCreateFlags;
    if (descriptorBufferEnabled) {
        descriptorSetLayoutCreateFlags |= vk::DescriptorSetLayoutCreateFlagBits::eDescriptorBufferEXT;
    }

    {
        const auto & [vertexShader, vertexShaderReflection] = addShader("identity.vert");
        {
            INVARIANT(std::size(vertexShaderReflection.descriptorSetLayoutSetBindings) == 1, "");
            INVARIANT(vertexShaderReflection.descriptorSetLayoutSetBindings.contains(SceneDescriptors::kSet), "");
            auto & descriptorSetLayoutBindings = vertexShaderReflection.descriptorSetLayoutSetBindings.at(SceneDescriptors::kSet);
            INVARIANT(std::size(descriptorSetLayoutBindings) == 1, "{}", std::size(descriptorSetLayoutBindings));
            {
                auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(SceneDescriptors::kTransformBuferName);
                INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 0, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eStorageBuffer, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eVertex, "");
                // INVARIANT(descriptorSetLayoutBindingReflection.size == sizeof(glm::mat4), "{} ^ {}", descriptorSetLayoutBindingReflection.size, sizeof(glm::mat4));
            }
            if ((false)) {
                auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(FrameDescriptors::kUniformBufferName);
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
        sceneShaderStages.append(vertexShader, vertexShaderReflection);

        const auto & [fragmentShader, fragmentShaderReflection] = addShader("barycentric_color.frag");
        {
            INVARIANT(std::size(fragmentShaderReflection.descriptorSetLayoutSetBindings) == 1, "");
            INVARIANT(fragmentShaderReflection.descriptorSetLayoutSetBindings.contains(FrameDescriptors::kSet), "");
            auto & descriptorSetLayoutBindings = fragmentShaderReflection.descriptorSetLayoutSetBindings.at(FrameDescriptors::kSet);
            INVARIANT(std::size(descriptorSetLayoutBindings) == 1, "{}", std::size(descriptorSetLayoutBindings));
            auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(FrameDescriptors::kUniformBufferName);
            INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 0, "");
            INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eUniformBuffer, "");
            INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
            INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eFragment, "");
            // INVARIANT(descriptorSetLayoutBindingReflection.size == sizeof(UniformBuffer), "{} ^ {}", descriptorSetLayoutBindingReflection.size, sizeof(UniformBuffer));

            INVARIANT(!fragmentShaderReflection.pushConstantRange, "");
        }
        sceneShaderStages.append(fragmentShader, fragmentShaderReflection);
    }
    sceneShaderStages.createDescriptorSetLayouts(kRasterization, descriptorSetLayoutCreateFlags);

    {
        const auto & [vertexShader, vertexShaderReflection] = addShader("fullscreen_rect.vert");
        {
            INVARIANT(std::size(vertexShaderReflection.descriptorSetLayoutSetBindings) == 1, "");
            INVARIANT(vertexShaderReflection.descriptorSetLayoutSetBindings.contains(FrameDescriptors::kSet), "");
            auto & descriptorSetLayoutBindings = vertexShaderReflection.descriptorSetLayoutSetBindings.at(FrameDescriptors::kSet);
            INVARIANT(std::size(descriptorSetLayoutBindings) == 1, "{}", std::size(descriptorSetLayoutBindings));
            {
                auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(FrameDescriptors::kUniformBufferName);
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
            INVARIANT(std::size(fragmentShaderReflection.descriptorSetLayoutSetBindings) == 1, "");
            {
                INVARIANT(fragmentShaderReflection.descriptorSetLayoutSetBindings.contains(FrameDescriptors::kSet), "");
                auto & descriptorSetLayoutBindings = fragmentShaderReflection.descriptorSetLayoutSetBindings.at(FrameDescriptors::kSet);
                INVARIANT(std::size(descriptorSetLayoutBindings) == 2, "{}", std::size(descriptorSetLayoutBindings));
                {
                    auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(FrameDescriptors::kUniformBufferName);
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 0, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eUniformBuffer, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eFragment, "");
                    // INVARIANT(descriptorSetLayoutBindingReflection.size == sizeof(UniformBuffer), "{} ^ {}", descriptorSetLayoutBindingReflection.size, sizeof(UniformBuffer));
                }
                {
                    auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(kDisplaySamplerName);
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 1, "");
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

Scene::Scene(const engine::Context & context, const FileIo & fileIo, std::shared_ptr<const engine::PipelineCache> pipelineCache, std::filesystem::path scenePath, scene_data::SceneData && sceneData)
    : context{context}, fileIo{fileIo}, pipelineCache{std::move(pipelineCache)}, scenePath{std::move(scenePath)}, sceneData{std::move(sceneData)}, sceneShaderStages{context, kVertexBufferBinding}, offscreenShaderStages{context, kVertexBufferBinding}
{
    check();
    addShaders();
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

engine::Buffer<glm::mat4> Scene::createTransformBuffer(uint32_t totalInstanceCount, const std::vector<std::vector<glm::mat4>> & transforms) const
{
    vk::BufferCreateInfo transformBufferCreateInfo;
    transformBufferCreateInfo.size = totalInstanceCount * sizeof(glm::mat4);
    transformBufferCreateInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer;
    if (descriptorBufferEnabled) {
        transformBufferCreateInfo.usage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
    }
    engine::Buffer<glm::mat4> transformBuffer{context.getMemoryAllocator().createStagingBuffer("Transformations"s, transformBufferCreateInfo, getMinAlignment())};

    constexpr vk::MemoryPropertyFlags kMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    auto memoryPropertyFlags = transformBuffer.base().getMemoryPropertyFlags();
    INVARIANT((memoryPropertyFlags & kMemoryPropertyFlags) == kMemoryPropertyFlags, "Failed to allocate transformation buffer in {} memory, got {} memory", kMemoryPropertyFlags, memoryPropertyFlags);

    {
        auto mappedTransformBuffer = transformBuffer.map();
        auto t = mappedTransformBuffer.begin();
        for (const auto & instanceTransforms : transforms) {
            ASSERT(mappedTransformBuffer.end() != t);
            t = std::copy(std::cbegin(instanceTransforms), std::cend(instanceTransforms), t);
        }
        ASSERT(t == mappedTransformBuffer.end());
    }

    return transformBuffer;
}

std::optional<engine::Buffer<scene_data::VertexAttributes>> Scene::createVertexBuffer() const
{
    if (!sceneShaderStages.vertexInputState) {
        return std::nullopt;
    }
    vk::DeviceSize vertexSize = 0;
    for (const auto & vertexInputAttributeDescription : sceneShaderStages.vertexInputState.value().vertexInputAttributeDescriptions) {
        vertexSize += vk::blockSize(vertexInputAttributeDescription.format);
    }

    if (vertexSize == 0) {
        return std::nullopt;
    }
    INVARIANT(sizeof(scene_data::VertexAttributes) == vertexSize, "{} != {}", sizeof(scene_data::VertexAttributes), vertexSize);

    vk::BufferCreateInfo vertexBufferCreateInfo;
    vertexBufferCreateInfo.size = sceneData.vertices.getCount() * vertexSize;
    vertexBufferCreateInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer;
    engine::Buffer<scene_data::VertexAttributes> vertexBuffer{context.getMemoryAllocator().createStagingBuffer("Vertices"s, vertexBufferCreateInfo, getMinAlignment())};

    constexpr vk::MemoryPropertyFlags kMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    auto memoryPropertyFlags = vertexBuffer.base().getMemoryPropertyFlags();
    INVARIANT((memoryPropertyFlags & kMemoryPropertyFlags) == kMemoryPropertyFlags, "Failed to allocate vertex buffer in {} memory, got {} memory", kMemoryPropertyFlags, memoryPropertyFlags);

    {
        auto mappedVertexBuffer = vertexBuffer.map();
        ASSERT(sceneData.vertices.getCount() == mappedVertexBuffer.getCount());
        if (std::copy_n(sceneData.vertices.begin(), sceneData.vertices.getCount(), mappedVertexBuffer.begin()) != mappedVertexBuffer.end()) {
            ASSERT(false);
        }
    }

    return vertexBuffer;
}

auto Scene::createUniformBuffer() const -> engine::Buffer<UniformBuffer>
{
    vk::BufferCreateInfo uniformBufferCreateInfo;
    uniformBufferCreateInfo.size = sizeof(UniformBuffer);
    uniformBufferCreateInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;
    if (descriptorBufferEnabled) {
        uniformBufferCreateInfo.usage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
    }
    constexpr vk::MemoryPropertyFlags kMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    auto uniformBufferName = fmt::format("Uniform buffer");
    engine::Buffer<UniformBuffer> uniformBuffer{context.getMemoryAllocator().createStagingBuffer(uniformBufferName, uniformBufferCreateInfo, getMinAlignment())};

    auto memoryPropertyFlags = uniformBuffer.base().getMemoryPropertyFlags();
    INVARIANT((memoryPropertyFlags & kMemoryPropertyFlags) == kMemoryPropertyFlags, "Failed to allocate uniform buffer in {} memory, got {} memory", kMemoryPropertyFlags, memoryPropertyFlags);

    return uniformBuffer;
}

engine::DescriptorSet Scene::createDescriptorSet(const engine::ShaderStages & shaderStages, uint32_t set) const
{
    constexpr uint32_t kFramesInFlight = 1;
    return {kRasterization, context, kFramesInFlight, set, shaderStages};
}

engine::Buffer<std::byte> Scene::createDescriptorBuffer(const engine::ShaderStages & shaderStages, uint32_t set) const
{
    constexpr vk::MemoryPropertyFlags kRequiredMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    const auto descriptorBufferOffsetAlignment = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceDescriptorBufferPropertiesEXT>().descriptorBufferOffsetAlignment;
    auto alignment = std::max(getMinAlignment(), descriptorBufferOffsetAlignment);
    const auto & setBindings = shaderStages.setBindings.at(set);
    const auto & descriptorSetLayout = shaderStages.descriptorSetLayouts.at(setBindings.setIndex);
    vk::BufferCreateInfo descriptorBufferCreateInfo;
    descriptorBufferCreateInfo.usage = vk::BufferUsageFlagBits::eShaderDeviceAddress;
    descriptorBufferCreateInfo.size = context.getDevice().getDevice().getDescriptorSetLayoutSizeEXT(descriptorSetLayout, context.getDispatcher());
    for (const auto & binding : setBindings.bindings) {
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
    auto descriptorBufferName = fmt::format("Descriptor buffer for set #{}", set);
    auto descriptorBuffer = context.getMemoryAllocator().createStagingBuffer(descriptorBufferName, descriptorBufferCreateInfo, alignment);

    auto memoryPropertyFlags = descriptorBuffer.getMemoryPropertyFlags();
    INVARIANT((memoryPropertyFlags & kRequiredMemoryPropertyFlags) == kRequiredMemoryPropertyFlags, "Failed to allocate descriptor buffer in {} memory, got {} memory", kRequiredMemoryPropertyFlags, memoryPropertyFlags);

    return std::move(descriptorBuffer);
}

void Scene::fillDescriptorSet(engine::DescriptorSet & descriptorSet, const engine::ShaderStages & shaderStages, uint32_t set, const DescriptorSetInfos & descriptorSetInfos) const
{
    std::vector<vk::DescriptorBufferInfo> descriptorBufferInfos;
    descriptorBufferInfos.reserve(std::size(descriptorSetInfos));
    std::vector<vk::WriteDescriptorSet> descriptorWrites;
    descriptorWrites.reserve(std::size(descriptorSetInfos));
    const auto & setBindings = shaderStages.setBindings.at(set);
    INVARIANT(std::size(setBindings.bindingIndices) >= std::size(descriptorSetInfos), "{} ^ {}", std::size(setBindings.bindingIndices), std::size(descriptorSetInfos));
    for (const auto & [symbol, descriptorType, descriptorBufferInfo] : descriptorSetInfos) {
        const auto * binding = setBindings.getBinding(symbol);
        ASSERT_MSG(binding, "Binding for symbol {} is not found", symbol);
        ASSERT_MSG(descriptorType == binding->descriptorType, "{} ^ {}", descriptorType, binding->descriptorType);
        descriptorBufferInfos.push_back(descriptorBufferInfo);
        auto & writeDescriptorSet = descriptorWrites.emplace_back();
        writeDescriptorSet = {
            .dstSet = descriptorSet,
            .dstBinding = binding->binding,
            .dstArrayElement = 0,  // not an array
            .descriptorType = descriptorType,
        };
        writeDescriptorSet.setBufferInfo(descriptorBufferInfo);
    }

    constexpr auto kDescriptorCopies = nullptr;
    context.getDevice().getDevice().updateDescriptorSets(descriptorWrites, kDescriptorCopies, context.getDispatcher());
}

void Scene::fillDescriptorBuffer(engine::Buffer<std::byte> & descriptorBuffer, const engine::ShaderStages & shaderStages, uint32_t set, const DescriptorBufferInfos & descriptorBufferInfos) const
{
    const auto & dispatcher = context.getDispatcher();
    const auto & device = context.getDevice();

    const auto & setBindings = shaderStages.setBindings.at(set);
    INVARIANT(std::size(setBindings.bindingIndices) >= std::size(descriptorBufferInfos), "{} ^ {}", std::size(setBindings.bindingIndices), std::size(descriptorBufferInfos));
    const auto & descriptorSetLayout = shaderStages.descriptorSetLayouts.at(setBindings.setIndex);
    auto mappedDescriptorSetBuffer = descriptorBuffer.map();
    for (const auto & [symbol, descriptorType, descriptorData] : descriptorBufferInfos) {
        const auto * binding = setBindings.getBinding(symbol);
        ASSERT_MSG(binding, "Binding for symbol {} is not found", symbol);
        ASSERT_MSG(descriptorType == binding->descriptorType, "{} ^ {}", descriptorType, binding->descriptorType);
        vk::DescriptorGetInfoEXT descriptorGetInfo = {
            .type = descriptorType,
        };
        const auto setDescriptorInfo = [descriptorType = descriptorType, &data = descriptorGetInfo.data]<typename T>(const T & descriptorData)
        {
            if constexpr (std::is_same_v<T, vk::Sampler>) {
                switch (descriptorType) {
                case vk::DescriptorType::eSampler: {
                    data.setPSampler(&descriptorData);
                    break;
                }
                default: {
                    ASSERT_MSG(false, "{}", descriptorType);
                }
                }
            } else if constexpr (std::is_same_v<T, vk::DescriptorImageInfo>) {
                switch (descriptorType) {
                case vk::DescriptorType::eCombinedImageSampler: {
                    data.setPCombinedImageSampler(&descriptorData);
                    break;
                }
                case vk::DescriptorType::eInputAttachment: {
                    data.setPInputAttachmentImage(&descriptorData);
                    break;
                }
                case vk::DescriptorType::eSampledImage: {
                    data.setPSampledImage(&descriptorData);
                    break;
                }
                case vk::DescriptorType::eStorageImage: {
                    data.setPStorageImage(&descriptorData);
                    break;
                }
                default: {
                    ASSERT_MSG(false, "{}", descriptorType);
                }
                }
            } else if constexpr (std::is_same_v<T, vk::DeviceAddress>) {
                switch (descriptorType) {
                case vk::DescriptorType::eAccelerationStructureKHR: {
                    data.setAccelerationStructure(descriptorData);
                    break;
                }
                default: {
                    ASSERT_MSG(false, "{}", descriptorType);
                }
                }
            } else if constexpr (std::is_same_v<T, vk::DescriptorAddressInfoEXT>) {
                switch (descriptorType) {
                case vk::DescriptorType::eUniformTexelBuffer: {
                    data.setPUniformTexelBuffer(&descriptorData);
                    break;
                }
                case vk::DescriptorType::eStorageTexelBuffer: {
                    data.setPStorageTexelBuffer(&descriptorData);
                    break;
                }
                case vk::DescriptorType::eUniformBuffer: {
                    data.setPUniformBuffer(&descriptorData);
                    break;
                }
                case vk::DescriptorType::eStorageBuffer: {
                    data.setPStorageBuffer(&descriptorData);
                    break;
                }
                default: {
                    ASSERT_MSG(false, "{}", descriptorType);
                }
                }
            } else {
                static_assert(sizeof(T) == 0);
            }
        };
        std::visit(setDescriptorInfo, descriptorData);
        vk::DeviceSize bindingOffset = device.getDevice().getDescriptorSetLayoutBindingOffsetEXT(descriptorSetLayout, binding->binding, dispatcher);
        vk::DeviceSize descriptorSize = getDescriptorSize(descriptorType);
        ASSERT(bindingOffset + descriptorSize <= descriptorBuffer.base().getSize());
        device.getDevice().getDescriptorEXT(&descriptorGetInfo, descriptorSize, mappedDescriptorSetBuffer.data() + bindingOffset, dispatcher);
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
        scene_data::SceneData sceneData;
        if ((true)) {
            auto cacheLocation = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
            if (!scene_loader::cachingLoad(sceneData, QFileInfo{scenePath}, cacheLocation)) {
                return nullptr;
            }
        } else {
            if (!scene_loader::load(sceneData, QFileInfo{scenePath})) {
                return nullptr;
            }
        }
        p = Scene::make(context, fileIo, getOrCreatePipelineCache(), std::move(scenePath), std::move(sceneData));
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
