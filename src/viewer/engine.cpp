#include <engine/device.hpp>
#include <engine/physical_device.hpp>
#include <engine/vma.hpp>
#include <format/vulkan.hpp>
#include <utils/auto_cast.hpp>
#include <viewer/engine.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtx/matrix_operation.hpp>
#include <glm/mat4x4.hpp>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_format_traits.hpp>

#include <algorithm>
#include <iterator>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

using namespace std::string_literals;
using namespace std::string_view_literals;

namespace viewer
{

engine::DescriptorBindingNameAndType SceneResources::getBindingName()
{
    return {"transformBuffer"s, vk::DescriptorType::eStorageBuffer};
}

[[nodiscard]] DescriptorInfo SceneResources::getDescriptorInfo(bool descriptorBufferEnabled) const
{
    const auto getDescriptorData = [this, descriptorBufferEnabled]() -> DescriptorData
    {
        if (!transformBuffer) {
            if (descriptorBufferEnabled) {  // requires nullDescriptor
                return DescriptorBufferData{};
            } else {
                return DescriptorSetData{vk::DescriptorBufferInfo{}};
            }
        }
        const auto & t = transformBuffer.value().base();
        if (descriptorBufferEnabled) {
            return DescriptorBufferData{t.getDescriptorAddressInfo()};
        } else {
            return DescriptorSetData{t.getDescriptorBufferInfo()};
        }
    };
    return {getBindingName(), getDescriptorData()};
}

OffscreenRenderPass OffscreenRenderPass::make(const engine::Context & context)
{
    vk::Format depthFormat = context.getPhysicalDevice().findDepthImageFormat(vk::ImageTiling::eOptimal);
    INVARIANT(depthFormat != vk::Format::eUndefined, "");
    vk::ImageLayout depthImageLayout = vk::ImageLayout::eUndefined;
    if (context.getDevice().createInfoChain.get<vk::PhysicalDeviceVulkan12Features>().separateDepthStencilLayouts == vk::False) {
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
                .srcSubpass = vk::SubpassExternal,
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
                .dstSubpass = vk::SubpassExternal,
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
                .srcSubpass = vk::SubpassExternal,
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
                .dstSubpass = vk::SubpassExternal,
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
    if (context.getDevice().createInfoChain.get<vk::PhysicalDeviceVulkan12Features>().separateDepthStencilLayouts == vk::False) {
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

engine::DescriptorBindingNameAndType DrawOffscreenResources::getBindingName()
{
    return {"display"s, vk::DescriptorType::eCombinedImageSampler};
}

[[nodiscard]] DescriptorInfo DrawOffscreenResources::getDescriptorInfo(bool descriptorBufferEnabled) const
{
    ASSERT(sampler);
    ASSERT(*sampler);
    ASSERT(framebuffer.colorImageView);
    vk::DescriptorImageInfo descriptorImageInfo = {
        .sampler = **sampler,
        .imageView = *framebuffer.colorImageView,
        .imageLayout = OffscreenRenderPass::kExternalColorImageLayout,
    };
    const auto getDescriptorData = [descriptorBufferEnabled, &descriptorImageInfo]() -> DescriptorData
    {
        if (descriptorBufferEnabled) {
            return DescriptorBufferData{descriptorImageInfo};
        } else {
            return DescriptorSetData{descriptorImageInfo};
        }
    };
    return {getBindingName(), getDescriptorData()};
}

TraceFrameResources::TraceFrameResources(const engine::Context & context, const vk::Extent2D & imageSize, std::shared_ptr<const vk::UniqueSampler> sampler)
    : image{makeImage(context, imageSize)}
    , imageView{image.createImageView(vk::ImageViewType::e2D, kImageAspectMask)}
    , sampler{std::move(sampler)}
{}

engine::Image TraceFrameResources::makeImage(const engine::Context & context, const vk::Extent2D & imageSize)
{
    constexpr auto imageName = "tree render target"sv;
    const uint32_t queueFamilyIndex = context.getPhysicalDevice().computeQueueCreateInfo.familyIndex;
    return context.getMemoryAllocator().createImage2D(imageName, kFormat, imageSize, kImageUsage, kImageAspectMask, queueFamilyIndex);
}

engine::DescriptorBindingNameAndType TraceFrameResources::getBindingName(bool target)
{
    if (target) {
        return {"target"s, vk::DescriptorType::eStorageImage};
    } else {
        return {"display"s, vk::DescriptorType::eCombinedImageSampler};
    }
}

DescriptorInfo TraceFrameResources::getDescriptorInfo(bool descriptorBufferEnabled, bool target) const
{
    ASSERT(sampler);
    ASSERT(*sampler);
    vk::DescriptorImageInfo descriptorImageInfo = {
        .sampler = **sampler,
        .imageView = *imageView,
        .imageLayout = kExternalImageLayout,
    };
    const auto getDescriptorData = [descriptorBufferEnabled, &descriptorImageInfo]() -> DescriptorData
    {
        if (descriptorBufferEnabled) {
            return DescriptorBufferData{descriptorImageInfo};
        } else {
            return DescriptorSetData{descriptorImageInfo};
        }
    };
    return {getBindingName(target), getDescriptorData()};
}

Engine::Engine(const engine::Context & context, const Settings & settings)
    : context{context}
    , settings{settings}
    , pipelines{context, settings.descriptorBufferEnabled}
{
    const auto & device = context.getDevice();
    if (settings.indexTypeUint8Enabled) {
        if (device.createInfoChain.get<vk::PhysicalDeviceIndexTypeUint8FeaturesKHR>().indexTypeUint8 == vk::False) {
            INVARIANT(false, "");
        }
    }
    if (settings.descriptorBufferEnabled) {
        if (device.createInfoChain.get<vk::PhysicalDeviceDescriptorBufferFeaturesEXT>().descriptorBuffer == vk::False) {
            INVARIANT(false, "");
        }
    }
    if (settings.multiDrawIndirectEnabled) {
        if (device.createInfoChain.get<vk::PhysicalDeviceFeatures2>().features.multiDrawIndirect == vk::False) {
            INVARIANT(false, "");
        }
    }
    if (settings.drawIndirectCountEnabled) {
        if (device.createInfoChain.get<vk::PhysicalDeviceVulkan12Features>().drawIndirectCount == vk::False) {
            INVARIANT(false, "");
        }
    }
    {
        const auto & properties2Chain = context.getPhysicalDevice().properties2Chain;
        const auto & vkDeviceUuid = properties2Chain.get<vk::PhysicalDeviceIDProperties>().deviceUUID;
        builder::DeviceUuidType deviceUuid;
        ASSERT(std::size(vkDeviceUuid) == std::size(deviceUuid));
        std::memcpy(std::data(deviceUuid), std::data(vkDeviceUuid), std::size(vkDeviceUuid));
        builder.emplace(deviceUuid);
    }
}

auto Engine::createUniformBuffer(size_t uniformBufferSize) const -> engine::Buffer<void>
{
    vk::BufferCreateInfo uniformBufferCreateInfo;
    uniformBufferCreateInfo.size = uniformBufferSize;
    uniformBufferCreateInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;
    if (settings.descriptorBufferEnabled) {
        uniformBufferCreateInfo.usage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
    }
    constexpr vk::MemoryPropertyFlags kMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    auto uniformBufferName = fmt::format("Uniform buffer");
    auto uniformBuffer = context.getMemoryAllocator().createStagingBuffer(uniformBufferName, uniformBufferCreateInfo, context.getPhysicalDevice().getMinAlignment());

    auto memoryPropertyFlags = uniformBuffer.getMemoryPropertyFlags();
    INVARIANT((memoryPropertyFlags & kMemoryPropertyFlags) == kMemoryPropertyFlags, "Failed to allocate uniform buffer in {} memory, got {} memory", kMemoryPropertyFlags & ~memoryPropertyFlags, ~kMemoryPropertyFlags & memoryPropertyFlags);

    return uniformBuffer;
}

SceneResources Engine::makeResources(const scene_data::SceneData & sceneData) const
{
    std::vector<std::vector<glm::mat4>> transforms(std::size(sceneData.meshes));  // [Scene::meshes index][instance index]
    std::vector<vk::DrawIndexedIndirectCommand> instances(std::size(sceneData.meshes));
    {
        const auto collectNodeInfos = [&sceneData, &transforms, &instances](const auto & collectNodeInfos, const scene_data::Node & sceneNode, glm::mat4 transform) -> void
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
        auto transform = glm::identity<glm::mat4>();
        collectNodeInfos(collectNodeInfos, sceneData.nodes.front(), std::move(transform));
    }

    vk::IndexType maxIndexType = vk::IndexType::eNoneKHR;
    std::vector<vk::IndexType> indexTypes;
    vk::DeviceSize indexBufferSize = 0;
    uint32_t totalInstanceCount = 0;
    {
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
    if (indexBufferSize != 0) {
        {
            vk::BufferCreateInfo indexBufferCreateInfo;
            indexBufferCreateInfo.size = indexBufferSize;
            indexBufferCreateInfo.usage = vk::BufferUsageFlagBits::eIndexBuffer;
            indexBuffer.emplace(context.getMemoryAllocator().createStagingBuffer("Indices"sv, indexBufferCreateInfo, context.getPhysicalDevice().getMinAlignment()));

            constexpr vk::MemoryPropertyFlags kMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
            auto memoryPropertyFlags = indexBuffer.value().getMemoryPropertyFlags();
            INVARIANT((memoryPropertyFlags & kMemoryPropertyFlags) == kMemoryPropertyFlags, "Failed to allocate index buffer in {} memory, got {} memory", kMemoryPropertyFlags & ~memoryPropertyFlags, ~kMemoryPropertyFlags & memoryPropertyFlags);
        }

        {
            auto mappedIndexBuffer = indexBuffer.value().map();
            auto indices = mappedIndexBuffer.data();
            for (size_t m = 0; m < std::size(sceneData.meshes); ++m) {
                const auto & instance = instances.at(m);

                ASSERT(std::size(transforms.at(m)) == instance.instanceCount);

                uint32_t sceneIndexOffset = sceneData.meshes.at(m).indexOffset;
                const auto convertCopy = [&sceneData, &instance, sceneIndexOffset](auto indices)
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

        instances.clear();
        indexTypes.clear();
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
        INVARIANT((memoryPropertyFlags & kMemoryPropertyFlags) == kMemoryPropertyFlags, "Failed to allocate vertex buffer in {} memory, got {} memory", kMemoryPropertyFlags & ~memoryPropertyFlags, ~kMemoryPropertyFlags & memoryPropertyFlags);

        {
            auto mappedVertexBuffer = vertexBuffer.value().map();
            ASSERT(sceneData.vertices.getCount() == mappedVertexBuffer.getCount());
            if (std::copy_n(sceneData.vertices.begin(), sceneData.vertices.getCount(), mappedVertexBuffer.begin()) != mappedVertexBuffer.end()) {
                ASSERT(false);
            }
        }
    }

    return {
        .instances = std::move(instances),
        .instanceBuffer = std::move(instanceBuffer),
        .indexTypes = std::move(indexTypes),
        .maxIndexType = maxIndexType,
        .drawCount = drawCount,
        .drawCountBuffer = std::move(drawCountBuffer),
        .transformBuffer = std::move(transformBuffer),
        .vertexBuffer = std::move(vertexBuffer),
        .indexBuffer = std::move(indexBuffer),
    };
}

Descriptors Engine::makeDescriptors(std::string_view name, std::shared_ptr<const engine::ShaderStages> shaderStages, const DescriptorInfos & descriptorInfos) const
{
    const uint32_t set = utils::autoCast(shaderStages->findSetByBindingName(std::get<0>(descriptorInfos.at(0))));
    auto shaderBindingName = std::cbegin(shaderStages->setBindings.at(set).bindingNames);
    for (const auto & [name, data] : descriptorInfos) {
        if (*shaderBindingName != name) {
            INVARIANT(false, "{} ^ {}", *shaderBindingName, name);
        }
        ++shaderBindingName;
    }
    Descriptors descriptors{name, context, settings.descriptorBufferEnabled, std::move(shaderStages), set};
    descriptors.fill(descriptorInfos);
    return descriptors;
}

auto Engine::createTransformBuffer(uint32_t instanceCount, const std::vector<std::vector<glm::mat4>> & transforms) const -> std::optional<engine::Buffer<glm::mat4>>
{
    if (instanceCount == 0) {
        return {};
    }

    vk::BufferCreateInfo transformBufferCreateInfo;
    transformBufferCreateInfo.size = instanceCount * sizeof(glm::mat4);
    transformBufferCreateInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer;
    if (settings.descriptorBufferEnabled) {
        transformBufferCreateInfo.usage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
    }
    engine::Buffer<glm::mat4> transformBuffer{context.getMemoryAllocator().createStagingBuffer("transforms"sv, transformBufferCreateInfo, context.getPhysicalDevice().getMinAlignment())};

    constexpr vk::MemoryPropertyFlags kMemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    auto memoryPropertyFlags = transformBuffer.base().getMemoryPropertyFlags();
    INVARIANT((memoryPropertyFlags & kMemoryPropertyFlags) == kMemoryPropertyFlags, "Failed to allocate transformation buffer in {} memory, got {} memory", kMemoryPropertyFlags & ~memoryPropertyFlags, ~kMemoryPropertyFlags & memoryPropertyFlags);

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

}  // namespace viewer
