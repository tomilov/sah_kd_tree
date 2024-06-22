#include <engine/buffer.hpp>
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/utils.hpp>
#include <engine/vma.hpp>
#include <format/vulkan.hpp>
#include <scene_data/scene_data.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <viewer/scenes.hpp>

#include <fmt/std.h>
#include <glm/ext/matrix_transform.hpp>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_format_traits.hpp>

#include <QFileInfo>
#include <QStandardPaths>

#include <algorithm>
#include <array>
#include <iterator>
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

OffscreenRenderPass OffscreenRenderPass::make(const engine::Context & context)
{
    vk::Format depthFormat = context.getPhysicalDevice().findDepthImageFormat(vk::ImageTiling::eOptimal);
    INVARIANT(depthFormat != vk::Format::eUndefined, "");
    vk::ImageLayout depthImageLayout = vk::ImageLayout::eUndefined;
    if (context.getDevice().createInfoChain.get<vk::PhysicalDeviceVulkan12Features>().separateDepthStencilLayouts == VK_FALSE) {
        depthImageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    } else {
        depthImageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
    }

    const vk::AttachmentDescription2 attachmentDecriptions[] = {
        {
            .format = OffscreenRenderPass::kColorFormat,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = kExternalColorImageLayout,
        },
        {
            .format = depthFormat,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eDontCare,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = depthImageLayout,
        },
    };

    const vk::AttachmentReference2 colorAttachmentReferences[] = {
        {
            .attachment = 0,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        },
    };

    const vk::AttachmentReference2 depthAttachmentReference = {
        .attachment = 1,
        .layout = depthImageLayout,
    };

    vk::SubpassDescription2 subpassDescriptions[] = {
        {
            .flags = {},
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .pDepthStencilAttachment = &depthAttachmentReference,
        },
    };
    subpassDescriptions[0].setColorAttachments(colorAttachmentReferences);

    constexpr auto kInternalColorStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    constexpr auto kInternalColorAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
    const vk::StructureChain<vk::SubpassDependency2, vk::MemoryBarrier2> subpassDependencyChain[] = {
        {
            {
                .srcSubpass = VK_SUBPASS_EXTERNAL,
                .dstSubpass = 0,
                .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                .viewOffset = 0,
            },
            {
                .srcStageMask = kExternalColorStageMask,
                .srcAccessMask = kExternalColorAccessMask,
                .dstStageMask = kInternalColorStageMask,
                .dstAccessMask = kInternalColorAccessMask,
            },
        },
        {
            {
                .srcSubpass = 0,
                .dstSubpass = VK_SUBPASS_EXTERNAL,
                .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                .viewOffset = 0,
            },
            {
                .srcStageMask = kInternalColorStageMask,
                .srcAccessMask = kInternalColorAccessMask,
                .dstStageMask = kExternalColorStageMask,
                .dstAccessMask = kExternalColorAccessMask,
            },
        },
        {
            {
                .srcSubpass = VK_SUBPASS_EXTERNAL,
                .dstSubpass = 0,
                .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                .viewOffset = 0,
            },
            {
                .srcStageMask = kDepthStageMask,
                .srcAccessMask = kDepthAccessMask,
                .dstStageMask = kDepthStageMask,
                .dstAccessMask = kDepthAccessMask,
            },
        },
        {
            {
                .srcSubpass = 0,
                .dstSubpass = VK_SUBPASS_EXTERNAL,
                .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                .viewOffset = 0,
            },
            {
                .srcStageMask = kDepthStageMask,
                .srcAccessMask = kDepthAccessMask,
                .dstStageMask = kDepthStageMask,
                .dstAccessMask = kDepthAccessMask,
            },
        },
    };

    auto subpassDependencies = engine::getHeads(subpassDependencyChain);

    vk::RenderPassCreateInfo2 renderPassCreateInfo = {
        .flags = {},
    };
    renderPassCreateInfo.setAttachments(attachmentDecriptions);
    renderPassCreateInfo.setSubpasses(subpassDescriptions);
    renderPassCreateInfo.setDependencies(subpassDependencies);
    vk::UniqueRenderPass renderPass = context.getDevice().getDevice().createRenderPass2Unique(renderPassCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());
    context.getDevice().setDebugUtilsObjectName(*renderPass, "Offscreen renderpass"s);

    return {
        .depthFormat = depthFormat,
        .depthImageLayout = depthImageLayout,
        .renderPass = std::move(renderPass),
    };
}

Framebuffer Framebuffer::make(const engine::Context & context, const vk::Extent2D & framebufferSize, const OffscreenRenderPass & offscreenRenderPass)
{
    vk::ImageAspectFlags depthImageAspectMask = vk::ImageAspectFlagBits::eDepth;
    if (context.getDevice().createInfoChain.get<vk::PhysicalDeviceVulkan12Features>().separateDepthStencilLayouts == VK_FALSE) {
        depthImageAspectMask |= vk::ImageAspectFlagBits::eStencil;
    }

    constexpr auto colorImageName = "offscreen framebuffer color image"sv;
    constexpr vk::ImageUsageFlags kColorImageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
    constexpr vk::ImageAspectFlags kColorImageAspectMask = vk::ImageAspectFlagBits::eColor;
    auto colorImage = context.getMemoryAllocator().createImage2D(colorImageName, OffscreenRenderPass::kColorFormat, framebufferSize, kColorImageUsage, kColorImageAspectMask);
    auto colorImageView = colorImage.createImageView(vk::ImageViewType::e2D, kColorImageAspectMask);

    constexpr auto depthImageName = "offscreen framebuffer depth image"sv;
    constexpr vk::ImageUsageFlags kDepthImageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
    auto depthImage = context.getMemoryAllocator().createImage2D(depthImageName, offscreenRenderPass.depthFormat, framebufferSize, kDepthImageUsage, depthImageAspectMask);
    auto depthImageView = depthImage.createImageView(vk::ImageViewType::e2D, depthImageAspectMask);

    const vk::ImageView attachments[] = {
        *colorImageView,
        *depthImageView,
    };
    vk::FramebufferCreateInfo framebufferCreateInfo = {
        .flags = {},
        .renderPass = *offscreenRenderPass.renderPass,
        .width = framebufferSize.width,
        .height = framebufferSize.height,
        .layers = 1,
    };
    framebufferCreateInfo.setAttachments(attachments);
    auto framebuffer = context.getDevice().getDevice().createFramebufferUnique(framebufferCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());
    context.getDevice().setDebugUtilsObjectName(*framebuffer, "Offscreen framebuffer"s);

    return {
        .size = framebufferSize,
        .depthImageAspectMask = depthImageAspectMask,
        .colorImage = std::move(colorImage),
        .colorImageView = std::move(colorImageView),
        .depthImage = std::move(depthImage),
        .depthImageView = std::move(depthImageView),
        .framebuffer = std::move(framebuffer),
    };
}

std::string SceneResources::getName()
{
    return "transformBuffer"s;
}

[[nodiscard]] DescriptorInfos SceneResources::getDescriptorInfos(bool descriptorBufferEnabled) const
{
    const auto getDescriptorData = [this, descriptorBufferEnabled]() -> DescriptorData
    {
        if (descriptorBufferEnabled) {
            return transformBuffer.base().getDescriptorAddressInfo();
        } else {
            return transformBuffer.base().getDescriptorBufferInfo();
        }
    };
    return {
        {getName(), vk::DescriptorType::eStorageBuffer, getDescriptorData()},
    };
}

std::string DisplayResources::getName()
{
    return "display"s;
}

[[nodiscard]] DescriptorInfos DisplayResources::getDescriptorInfos(bool descriptorBufferEnabled) const
{
    vk::DescriptorImageInfo descriptorImageInfo = {
        .sampler = **sampler,
        .imageView = *framebuffer.colorImageView,
        .imageLayout = OffscreenRenderPass::kExternalColorImageLayout,
    };
    const auto getDescriptorData = [descriptorBufferEnabled, &descriptorImageInfo]() -> DescriptorData
    {
        if (descriptorBufferEnabled) {
            return DescriptorBufferData{std::move(descriptorImageInfo)};
        } else {
            return DescriptorSetData{std::move(descriptorImageInfo)};
        }
    };
    return {
        {getName(), vk::DescriptorType::eCombinedImageSampler, getDescriptorData()},
    };
}

Scene::Scene(const engine::Context & context, const Settings & settings, const std::filesystem::path & scenePath, scene_data::SceneData && sceneData)
    : context{context}
    , settings{settings}
    , scenePath{std::move(scenePath)}
    , sceneData{std::move(sceneData)}
{
    checkSettings();
}

auto Scene::getSettings() const & -> const Settings &
{
    return settings;
}

const std::filesystem::path & Scene::getScenePath() const &
{
    return scenePath;
}

const scene_data::SceneData & Scene::getScenedData() const &
{
    return sceneData;
}

SceneResources Scene::makeSceneResources() const
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
            if (settings.indexTypeUint8Enabled && (maxIndex <= std::numeric_limits<engine::IndexCppType<vk::IndexType::eUint8EXT>>::max())) {
                indexType = vk::IndexType::eUint8KHR;
            } else if (maxIndex <= std::numeric_limits<engine::IndexCppType<vk::IndexType::eUint16>>::max()) {
                indexType = vk::IndexType::eUint16;
            } else {
                indexType = vk::IndexType::eUint32;
            }
            if (engine::indexTypeLess(maxIndexType, indexType)) {
                maxIndexType = indexType;
            }
        }

        if (maxIndexType != vk::IndexType::eNoneKHR) {
            for (size_t m = 0; m < std::size(sceneData.meshes); ++m) {
                auto & indexType = indexTypes.at(m);
                if (indexType == vk::IndexType::eNoneKHR) {
                    continue;
                }
                if (settings.multiDrawIndirectEnabled) {
                    indexType = maxIndexType;
                }
                vk::DeviceSize formatSize = vk::blockSize(engine::indexTypeToFormat(indexType));
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
            indexBuffer.emplace(context.getMemoryAllocator().createStagingBuffer("Indices"sv, indexBufferCreateInfo, context.getPhysicalDevice().getMinAlignment()));

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
                case vk::IndexType::eUint8KHR: {
                    convertCopy(static_cast<engine::IndexCppType<vk::IndexType::eUint8KHR> *>(indices));
                    break;
                }
                case vk::IndexType::eUint16: {
                    convertCopy(static_cast<engine::IndexCppType<vk::IndexType::eUint16> *>(indices));
                    break;
                }
                case vk::IndexType::eUint32: {
                    convertCopy(static_cast<engine::IndexCppType<vk::IndexType::eUint32> *>(indices));
                    break;
                }
                }
            }
        }
    }

    uint32_t drawCount = utils::autoCast(std::size(instances));

    std::optional<engine::Buffer<uint32_t>> drawCountBuffer;
    std::optional<engine::Buffer<vk::DrawIndexedIndirectCommand>> instanceBuffer;
    if (settings.multiDrawIndirectEnabled) {
        if (settings.drawIndirectCountEnabled) {
            vk::BufferCreateInfo drawCountBufferCreateInfo;
            drawCountBufferCreateInfo.size = sizeof(uint32_t);
            drawCountBufferCreateInfo.usage = vk::BufferUsageFlagBits::eIndirectBuffer;
            drawCountBuffer.emplace(context.getMemoryAllocator().createStagingBuffer("DrawCount"sv, drawCountBufferCreateInfo, context.getPhysicalDevice().getMinAlignment()));

            auto mappedDrawCountBuffer = drawCountBuffer.value().map();
            mappedDrawCountBuffer.at(0) = drawCount;
        }

        {
            vk::BufferCreateInfo instanceBufferCreateInfo;
            constexpr uint32_t kSize = sizeof(vk::DrawIndexedIndirectCommand);
            instanceBufferCreateInfo.size = drawCount * kSize;
            instanceBufferCreateInfo.usage = vk::BufferUsageFlagBits::eIndirectBuffer;
            instanceBuffer.emplace(context.getMemoryAllocator().createStagingBuffer("Instances"sv, instanceBufferCreateInfo, context.getPhysicalDevice().getMinAlignment()));

            auto mappedInstanceBuffer = instanceBuffer.value().map();
            auto end = std::copy(std::cbegin(instances), std::cend(instances), mappedInstanceBuffer.begin());
            INVARIANT(end == mappedInstanceBuffer.end(), "");
        }
    }

    auto transformBuffer = createTransformBuffer(totalInstanceCount, transforms);

    std::optional<engine::Buffer<scene_data::VertexAttributes>> vertexBuffer;
    if (!sceneData.vertices.isEmpty()) {
        vk::BufferCreateInfo vertexBufferCreateInfo;
        vertexBufferCreateInfo.size = sceneData.vertices.getCount() * sizeof(scene_data::VertexAttributes);
        vertexBufferCreateInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer;
        vertexBuffer.emplace(context.getMemoryAllocator().createStagingBuffer("Vertices"sv, vertexBufferCreateInfo, context.getPhysicalDevice().getMinAlignment()));

        constexpr vk::MemoryPropertyFlags kMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
        auto memoryPropertyFlags = vertexBuffer.value().base().getMemoryPropertyFlags();
        INVARIANT((memoryPropertyFlags & kMemoryPropertyFlags) == kMemoryPropertyFlags, "Failed to allocate vertex buffer in {} memory, got {} memory", kMemoryPropertyFlags, memoryPropertyFlags);

        {
            auto mappedVertexBuffer = vertexBuffer.value().map();
            ASSERT(sceneData.vertices.getCount() == mappedVertexBuffer.getCount());
            if (std::copy_n(sceneData.vertices.begin(), sceneData.vertices.getCount(), mappedVertexBuffer.begin()) != mappedVertexBuffer.end()) {
                ASSERT(false);
            }
        }
    }

    return {
        .transforms = std::move(transforms),
        .instances = std::move(instances),
        .indexTypes = std::move(indexTypes),
        .indexBuffer = std::move(indexBuffer),
        .drawCount = drawCount,
        .drawCountBuffer = std::move(drawCountBuffer),
        .instanceBuffer = std::move(instanceBuffer),
        .transformBuffer = std::move(transformBuffer),
        .vertexBuffer = std::move(vertexBuffer),
    };
}

DisplayResources Scene::makeDisplayResources(const vk::Extent2D & framebufferSize, const OffscreenRenderPass & offscreenRenderPass, std::shared_ptr<const vk::UniqueSampler> sampler) const
{
    auto framebuffer = Framebuffer::make(context, framebufferSize, offscreenRenderPass);
    return {
        .framebuffer = std::move(framebuffer),
        .sampler = std::move(sampler),
    };
}

DescriptorSet Scene::makeDescriptors(const SceneResources & sceneResources) const
{
    auto descriptors = makeDescriptors("scene"sv, sceneShaders->getShaderStagesPtr(), SceneResources::kSet);
    fillDescriptors(sceneResources, descriptors);
    return descriptors;
}

DescriptorSet Scene::makeDescriptors(const UniformBufferResource & uniformBufferResource, bool display) const
{
    std::string_view name;
    std::shared_ptr<const engine::ShaderStages> shaderStages;
    if (display) {
        name = "frame offscreen"sv;
        shaderStages = displayShaders->getShaderStagesPtr();
    } else {
        name = "frame direct"sv;
        shaderStages = sceneShaders->getShaderStagesPtr();
    }
    auto descriptors = makeDescriptors(name, std::move(shaderStages), FrameResources::kSet);
    fillDescriptors(frameResources, descriptors);
    return descriptors;
}

DescriptorSet Scene::makeDescriptors(const DisplayResources & displayResources) const
{
    auto descriptors = makeDescriptors("display"sv, displayShaders->getShaderStagesPtr(), DisplayResources::kSet);
    fillDescriptors(displayResources, descriptors);
    return descriptors;
}

void Scene::checkSettings() const
{
    ASSERT(!std::empty(scenePath));

    uint32_t maxPushConstantsSize = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits.maxPushConstantsSize;
    INVARIANT(sizeof(ScenePushConstants) <= maxPushConstantsSize, "{} ^ {}", sizeof(ScenePushConstants), maxPushConstantsSize);

    const auto & device = context.getDevice();
    if (settings.indexTypeUint8Enabled) {
        if (device.createInfoChain.get<vk::PhysicalDeviceIndexTypeUint8FeaturesKHR>().indexTypeUint8 == VK_FALSE) {
            INVARIANT(false, "");
        }
    }
    if (settings.descriptorBufferEnabled) {
        if (device.createInfoChain.get<vk::PhysicalDeviceDescriptorBufferFeaturesEXT>().descriptorBuffer == VK_FALSE) {
            INVARIANT(false, "");
        }
    }
    if (settings.multiDrawIndirectEnabled) {
        if (device.createInfoChain.get<vk::PhysicalDeviceFeatures2>().features.multiDrawIndirect == VK_FALSE) {
            INVARIANT(false, "");
        }
    }
    if (settings.drawIndirectCountEnabled) {
        if (device.createInfoChain.get<vk::PhysicalDeviceVulkan12Features>().drawIndirectCount == VK_FALSE) {
            INVARIANT(false, "");
        }
    }
}

engine::Buffer<glm::mat4> Scene::createTransformBuffer(uint32_t totalInstanceCount, const std::vector<std::vector<glm::mat4>> & transforms) const
{
    vk::BufferCreateInfo transformBufferCreateInfo;
    transformBufferCreateInfo.size = totalInstanceCount * sizeof(glm::mat4);
    transformBufferCreateInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer;
    if (settings.descriptorBufferEnabled) {
        transformBufferCreateInfo.usage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
    }
    engine::Buffer<glm::mat4> transformBuffer{context.getMemoryAllocator().createStagingBuffer("Transformations"sv, transformBufferCreateInfo, context.getPhysicalDevice().getMinAlignment())};

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

Scenes::Scenes(const engine::Context & context)
    : context{context}
{}

std::shared_ptr<const Scene> Scenes::getOrCreateScene(const std::filesystem::path & scenePath) const
{
    ASSERT(!std::empty(scenePath));
    std::lock_guard<std::mutex> lockGuard{mutex};
    auto & w = scenes[scenePath];
    auto p = w.lock();
    if (p) {
        SPDLOG_TRACE("Old scene {} reused", scenePath);
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
        Scene::Settings settings;
        p = std::make_shared<Scene>(context, settings, std::move(scenePath), std::move(sceneData));
        w = p;
        SPDLOG_TRACE("New scene {} created", p->getScenePath());
    }
    return p;
}

#if 0
void Pipelines::verifyShaders() const
{
    {
        const auto & shaderResources = sceneShaders->getShaderModules();
        INVARIANT(std::size(shaderResources) == 2, "{}", std::size(shaderResources));

        const auto & vertexShaderReflection = shaderResources.at(0).shaderReflection;
        {
            INVARIANT(std::size(vertexShaderReflection.descriptorSetLayoutSetBindings) == 1, "");
            INVARIANT(vertexShaderReflection.descriptorSetLayoutSetBindings.contains(SceneResources::kSet), "");
            auto & descriptorSetLayoutBindings = vertexShaderReflection.descriptorSetLayoutSetBindings.at(SceneResources::kSet);
            INVARIANT(std::size(descriptorSetLayoutBindings) == 1, "{}", std::size(descriptorSetLayoutBindings));
            {
                auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(SceneResources::kTransformBuferName);
                INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 0, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eStorageBuffer, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eVertex, "");
                // INVARIANT(descriptorSetLayoutBindingReflection.size == sizeof(glm::mat4), "{} ^ {}", descriptorSetLayoutBindingReflection.size, sizeof(glm::mat4));
            }
            if ((false)) {
                auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(FrameResources::kUniformBufferName);
                INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 0, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eUniformBuffer, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eVertex, "");
                // INVARIANT(descriptorSetLayoutBindingReflection.size == sizeof(UniformBuffer), "{} ^ {}", descriptorSetLayoutBindingReflection.size, sizeof(UniformBuffer));
            }

            INVARIANT(vertexShaderReflection.pushConstantRange, "used");
            if (vertexShaderReflection.pushConstantRange) {
                const auto & pushConstantRange = vertexShaderReflection.pushConstantRange.value();
                INVARIANT(pushConstantRange.stageFlags == vk::ShaderStageFlagBits::eVertex, "");
                INVARIANT(pushConstantRange.offset == offsetof(ScenePushConstants, mvp), "");
                INVARIANT(pushConstantRange.size == sizeof(ScenePushConstants::mvp), "");
            }
        }

        const auto & fragmentShaderReflection = shaderResources.at(1).shaderReflection;
        {
            INVARIANT(std::size(fragmentShaderReflection.descriptorSetLayoutSetBindings) == 1, "");
            INVARIANT(fragmentShaderReflection.descriptorSetLayoutSetBindings.contains(FrameResources::kSet), "");
            auto & descriptorSetLayoutBindings = fragmentShaderReflection.descriptorSetLayoutSetBindings.at(FrameResources::kSet);
            INVARIANT(std::size(descriptorSetLayoutBindings) == 1, "{}", std::size(descriptorSetLayoutBindings));
            auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(FrameResources::kUniformBufferName);
            INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 0, "");
            INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eUniformBuffer, "");
            INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
            INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eFragment, "");
            // INVARIANT(descriptorSetLayoutBindingReflection.size == sizeof(UniformBuffer), "{} ^ {}", descriptorSetLayoutBindingReflection.size, sizeof(UniformBuffer));

            INVARIANT(!fragmentShaderReflection.pushConstantRange, "not used");
        }
    }

    {
        const auto & shaderResources = displayShaders->getShaderResources();
        INVARIANT(std::size(shaderResources) == 2, "{}", std::size(shaderResources));

        const auto & vertexShaderReflection = shaderResources.at(0).shaderReflection;
        {
            INVARIANT(std::size(vertexShaderReflection.descriptorSetLayoutSetBindings) == 1, "");
            INVARIANT(vertexShaderReflection.descriptorSetLayoutSetBindings.contains(FrameResources::kSet), "");
            auto & descriptorSetLayoutBindings = vertexShaderReflection.descriptorSetLayoutSetBindings.at(FrameResources::kSet);
            INVARIANT(std::size(descriptorSetLayoutBindings) == 1, "{}", std::size(descriptorSetLayoutBindings));
            {
                auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(FrameResources::kUniformBufferName);
                INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 0, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eUniformBuffer, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
                INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eVertex, "");
                // INVARIANT(descriptorSetLayoutBindingReflection.size == sizeof(UniformBuffer), "{} ^ {}", descriptorSetLayoutBindingReflection.size, sizeof(UniformBuffer));
            }

            INVARIANT(!vertexShaderReflection.pushConstantRange, "not used");
            if ((false)) {
                const auto & pushConstantRange = vertexShaderReflection.pushConstantRange.value();
                INVARIANT(pushConstantRange.stageFlags == vk::ShaderStageFlagBits::eVertex, "");
                INVARIANT(pushConstantRange.offset == offsetof(ScenePushConstants, mvp), "");
                INVARIANT(pushConstantRange.size == sizeof(ScenePushConstants::mvp), "");
            }
        }

        const auto & fragmentShaderReflection = shaderResources.at(1).shaderReflection;
        {
            INVARIANT(std::size(fragmentShaderReflection.descriptorSetLayoutSetBindings) == 2, "");
            {
                INVARIANT(fragmentShaderReflection.descriptorSetLayoutSetBindings.contains(FrameResources::kSet), "");
                auto & descriptorSetLayoutBindings = fragmentShaderReflection.descriptorSetLayoutSetBindings.at(FrameResources::kSet);
                INVARIANT(std::size(descriptorSetLayoutBindings) == 1, "{}", std::size(descriptorSetLayoutBindings));
                {
                    auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(FrameResources::kUniformBufferName);
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 0, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eUniformBuffer, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eFragment, "");
                    // INVARIANT(descriptorSetLayoutBindingReflection.size == sizeof(UniformBuffer), "{} ^ {}", descriptorSetLayoutBindingReflection.size, sizeof(UniformBuffer));
                }
            }
            {
                INVARIANT(fragmentShaderReflection.descriptorSetLayoutSetBindings.contains(DisplayResources::kSet), "");
                auto & descriptorSetLayoutBindings = fragmentShaderReflection.descriptorSetLayoutSetBindings.at(DisplayResources::kSet);
                INVARIANT(std::size(descriptorSetLayoutBindings) == 1, "{}", std::size(descriptorSetLayoutBindings));
                {
                    auto & descriptorSetLayoutBindingReflection = descriptorSetLayoutBindings.at(DisplayResources::kDisplaySampler);
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.binding == 0, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorType == vk::DescriptorType::eCombinedImageSampler, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.descriptorCount == 1, "");
                    INVARIANT(descriptorSetLayoutBindingReflection.binding.stageFlags == vk::ShaderStageFlagBits::eFragment, "");
                }
            }

            INVARIANT(!fragmentShaderReflection.pushConstantRange, "not used");
            if (fragmentShaderReflection.pushConstantRange) {
                const auto & pushConstantRange = fragmentShaderReflection.pushConstantRange.value();
                INVARIANT(pushConstantRange.stageFlags == vk::ShaderStageFlagBits::eFragment, "");
                // INVARIANT(pushConstantRange.offset == offsetof(ScenePushConstants, x), "");
                // INVARIANT(pushConstantRange.size == sizeof(ScenePushConstants::x), "");
            }
        }
    }
}

void Pipelines::checkSceneVertexFormat() const
{
    auto shaderStages = sceneShaders->getShaderStages();
    ASSERT(shaderStages->vertexInputState);
    vk::DeviceSize vertexSize = 0;
    for (const auto & vertexInputAttributeDescription : shaderStages->vertexInputState.value().vertexInputAttributeDescriptions) {
        vertexSize += vk::blockSize(vertexInputAttributeDescription.format);
    }
    ASSERT(vertexSize != 0);
    INVARIANT(sizeof(scene_data::VertexAttributes) == vertexSize, "{} != {}", sizeof(scene_data::VertexAttributes), vertexSize);
}
#endif

}  // namespace viewer
