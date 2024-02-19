#include <codegen/vulkan_utils.hpp>
#include <common/version.hpp>
#include <engine/context.hpp>
#include <engine/debug_utils.hpp>
#include <engine/device.hpp>
#include <engine/framebuffer.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/image.hpp>
#include <engine/instance.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/vma.hpp>
#include <format/vulkan.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/checked_ptr.hpp>
#include <viewer/renderer.hpp>
#include <viewer/scene_manager.hpp>

#include <fmt/std.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_format_traits.hpp>

#include <queue>

#include <algorithm>
#include <filesystem>
#include <initializer_list>
#include <iterator>
#include <list>
#include <memory>
#include <stack>
#include <string_view>
#include <vector>

#include <cstddef>
#include <cstdint>

using namespace Qt::StringLiterals;
using namespace std::string_literals;
using namespace std::string_view_literals;

namespace viewer
{
namespace
{

template<typename T>
class ResourceStack final : std::stack<T, std::vector<T>>
{
    using base = std::stack<T, std::vector<T>>;

public:
    using base::empty;
    using base::pop;
    using base::push;
    using base::size;
    using base::top;

    void clear()
    {
        base::c.clear();
    }
};

template<typename T>
class ResourceQueue : std::queue<T, std::list<T>>
{
    using base = std::queue<T, std::list<T>>;

public:
    using base::back;
    using base::empty;
    using base::front;
    using base::pop;
    using base::push;
    using base::size;

    void clear()
    {
        base::c.clear();
    }
};

constexpr std::initializer_list<uint32_t> kUnmutedMessageIdNumbers = {
    0x5C0EC5D6,
    0xE4D96472,
};

void fillUniformBuffer(const FrameSettings & frameSettings, UniformBuffer & uniformBuffer)
{
    uniformBuffer = {
        .transform2D = frameSettings.transform2D,
        .alpha = frameSettings.alpha,
        .zNear = frameSettings.zNear,
        .zFar = frameSettings.zFar,
        .pos = frameSettings.position,
        .t = frameSettings.t,
    };
}

[[nodiscard]] PushConstants getPushConstants(const FrameSettings & frameSettings)
{
    auto view = glm::translate(glm::toMat4(glm::conjugate(frameSettings.orientation)), -frameSettings.position);
    auto projection = glm::perspectiveFovLH(frameSettings.fov, frameSettings.width, frameSettings.height, frameSettings.zNear, frameSettings.zFar);
    glm::mat4 transform2D{frameSettings.transform2D};  // 2D to 4D unit matrix extension
    auto mvp = transform2D * projection * view;
    return {
        .mvp = mvp,
        .x = 0.0f,
    };
}

}  // namespace

static_assert(utils::kIsOneTime<Renderer>);

struct Renderer::Impl
{
    struct OffscreenRenderPass
    {
        vk::Format depthFormat = vk::Format::eUndefined;
        vk::ImageAspectFlags depthImageAspect = vk::ImageAspectFlagBits::eNone;
        vk::ImageLayout depthImageLayout = vk::ImageLayout::eUndefined;

        vk::UniqueRenderPass renderPass;

        static OffscreenRenderPass make(const engine::Context & context);
    };

    struct Framebuffer
    {
        static constexpr vk::Format kFormat = vk::Format::eR8G8B8A8Unorm;

        vk::Extent2D size;
        vk::RenderPass renderPass;

        engine::Image colorImage;
        vk::UniqueImageView colorImageView;

        engine::Image depthImage;
        vk::UniqueImageView depthImageView;

        vk::UniqueFramebuffer framebuffer;

        static Framebuffer make(const engine::Context & context, const vk::Extent2D & size, const OffscreenRenderPass & offscreenRenderPass);
    };

    struct Frame
    {
        std::shared_ptr<const Scene::FrameDescriptors> descriptors;
    };

    const engine::Context & context;
    const uint32_t framesInFlight;

    std::shared_ptr<const Scene> scene;
    std::shared_ptr<const Scene::SceneDescriptors> sceneDescriptors;
    std::shared_ptr<const Scene::GraphicsPipeline> sceneGraphicsPipeline;
    ResourceQueue<Frame> frames;

    std::optional<OffscreenRenderPass> offscreenRenderPass;
    ResourceStack<Framebuffer> framebuffers;

    Impl(const engine::Context & context, uint32_t framesInFlight);

    [[nodiscard]] static vk::Format findDepthFormat(const engine::Context & context, vk::ImageTiling imageTiling);

    void setScene(std::shared_ptr<const Scene> scene);
    void advance(uint32_t currentFrameSlot, const FrameSettings & frameSettings);
    void render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, uint32_t currentFrameSlot, const FrameSettings & frameSettings);

    Framebuffer getFramebuffer(const vk::Extent2D & size);
    void putFramebuffer(Framebuffer && framebuffer);

    Frame getFrame();
    void putFramebuffer(Frame && frame);
};

Renderer::Renderer(const engine::Context & context, uint32_t framesInFlight) : impl_{context, framesInFlight}
{}

Renderer::Renderer(Renderer &&) noexcept = default;
Renderer::~Renderer() = default;

void Renderer::setScene(std::shared_ptr<const Scene> scene)
{
    return impl_->setScene(std::move(scene));
}

void Renderer::advance(uint32_t currentFrameSlot, const FrameSettings & frameSettings)
{
    return impl_->advance(currentFrameSlot, frameSettings);
}

void Renderer::render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, uint32_t currentFrameSlot, const FrameSettings & frameSettings)
{
    return impl_->render(commandBuffer, renderPass, currentFrameSlot, frameSettings);
}

std::shared_ptr<const Scene> Renderer::getScene() const
{
    return impl_->scene;
}

auto Renderer::Impl::OffscreenRenderPass::make(const engine::Context & context) -> OffscreenRenderPass
{
    vk::Format depthFormat = findDepthFormat(context, vk::ImageTiling::eOptimal);
    INVARIANT(depthFormat != vk::Format::eUndefined, "");
    vk::ImageAspectFlags depthImageAspect = vk::ImageAspectFlagBits::eDepth;
    vk::ImageLayout depthImageLayout = vk::ImageLayout::eUndefined;
    if (context.getDevice().createInfoChain.get<vk::PhysicalDeviceVulkan12Features>().separateDepthStencilLayouts == VK_FALSE) {
        depthImageAspect |= vk::ImageAspectFlagBits::eStencil;
        depthImageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    } else {
        depthImageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
    }

    const vk::AttachmentDescription attachmentDecriptions[] = {
        {
            .format = Framebuffer::kFormat,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        },
        {
            .format = depthFormat,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eDontCare,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = depthImageLayout,
            .finalLayout = depthImageLayout,
        },
    };

    const vk::AttachmentReference colorAttachmentReferences[] = {
        {
            .attachment = 0,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        },
    };

    const vk::AttachmentReference depthAttachmentReference = {
        .attachment = 1,
        .layout = depthImageLayout,
    };

    vk::SubpassDescription subpassDescriptions[] = {
        {
            .flags = {},
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .pDepthStencilAttachment = &depthAttachmentReference,
        },
    };
    subpassDescriptions[0].setColorAttachments(colorAttachmentReferences);

    const vk::SubpassDependency subpassDependencies[] = {
        {
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = vk::PipelineStageFlagBits::eFragmentShader,
            .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .srcAccessMask = vk::AccessFlagBits::eShaderRead,
            .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        },
        {
            .srcSubpass = 0,
            .dstSubpass = VK_SUBPASS_EXTERNAL,
            .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .dstStageMask = vk::PipelineStageFlagBits::eFragmentShader,
            .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        },
        {
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = vk::PipelineStageFlagBits::eLateFragmentTests,
            .dstStageMask = vk::PipelineStageFlagBits::eEarlyFragmentTests,
            .srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentRead,
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        },
        {
            .srcSubpass = 0,
            .dstSubpass = VK_SUBPASS_EXTERNAL,
            .srcStageMask = vk::PipelineStageFlagBits::eLateFragmentTests,
            .dstStageMask = vk::PipelineStageFlagBits::eEarlyFragmentTests,
            .srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentRead,
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        },
    };

    vk::RenderPassCreateInfo renderPassCreateInfo = {
        .flags = {},
    };
    renderPassCreateInfo.setAttachments(attachmentDecriptions);
    renderPassCreateInfo.setSubpasses(subpassDescriptions);
    renderPassCreateInfo.setDependencies(subpassDependencies);

    vk::UniqueRenderPass renderPass = context.getDevice().getDevice().createRenderPassUnique(renderPassCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());
    context.getDevice().setDebugUtilsObjectName(*renderPass, "Offscreen renderpass"s);
    return {
        .depthFormat = depthFormat,
        .depthImageAspect = depthImageAspect,
        .depthImageLayout = depthImageLayout,
        .renderPass = std::move(renderPass),
    };
}

auto Renderer::Impl::Framebuffer::make(const engine::Context & context, const vk::Extent2D & size, const OffscreenRenderPass & offscreenRenderPass) -> Framebuffer
{
    ASSERT(offscreenRenderPass.renderPass);
    auto renderPass = *offscreenRenderPass.renderPass;

    auto colorImageName = "offscreen framebuffer color image"s;
    constexpr vk::ImageUsageFlags kColorImageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
    auto colorImage = context.getMemoryAllocator().createImage2D(colorImageName, Framebuffer::kFormat, size, kColorImageUsage);
    auto colorImageView = colorImage.createImageView(vk::ImageViewType::e2D, vk::ImageAspectFlagBits::eColor);

    auto depthImageName = "offscreen framebuffer depth image"s;
    constexpr vk::ImageUsageFlags kDepthImageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
    auto depthImage = context.getMemoryAllocator().createImage2D(depthImageName, offscreenRenderPass.depthFormat, size, kDepthImageUsage);
    auto depthImageView = depthImage.createImageView(vk::ImageViewType::e2D, vk::ImageAspectFlagBits::eColor);

    const vk::ImageView attachments[] = {
        *colorImageView,
        *depthImageView,
    };
    vk::FramebufferCreateInfo framebufferCreateInfo = {
        .flags = {},
        .renderPass = renderPass,
        .width = size.width,
        .height = size.height,
        .layers = 1,
    };
    framebufferCreateInfo.setAttachments(attachments);
    auto framebuffer = context.getDevice().getDevice().createFramebufferUnique(framebufferCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());

    return {
        .size = size,
        .renderPass = renderPass,
        .colorImage = std::move(colorImage),
        .colorImageView = std::move(colorImageView),
        .depthImage = std::move(depthImage),
        .depthImageView = std::move(depthImageView),
        .framebuffer = std::move(framebuffer),
    };
}

Renderer::Impl::Impl(const engine::Context & context, uint32_t framesInFlight) : context{context}, framesInFlight{framesInFlight}
{}

vk::Format Renderer::Impl::findDepthFormat(const engine::Context & context, vk::ImageTiling imageTiling)
{
    const vk::FormatFeatureFlags2 vk::FormatProperties3::*p = nullptr;
    if (imageTiling == vk::ImageTiling::eLinear) {
        p = &vk::FormatProperties3::linearTilingFeatures;
    } else if (imageTiling == vk::ImageTiling::eOptimal) {
        p = &vk::FormatProperties3::optimalTilingFeatures;
    } else {
        INVARIANT(false, "{}", imageTiling);
    }

    constexpr vk::FormatFeatureFlags2 kFormatFeatureFlags = vk::FormatFeatureFlagBits2::eDepthStencilAttachment;
    auto physicalDevice = context.getPhysicalDevice().getPhysicalDevice();
    vk::Format depthFormat = vk::Format::eUndefined;
    for (vk::Format format : codegen::vulkan::kAllFormats) {
        auto formatProperties2Chain = physicalDevice.getFormatProperties2<vk::FormatProperties2, vk::FormatProperties3>(format, context.getDispatcher());
        if ((formatProperties2Chain.get<vk::FormatProperties3>().*p & kFormatFeatureFlags) != kFormatFeatureFlags) {
            continue;
        }
        const auto & formatDescription = codegen::vulkan::kFormatDescriptions.at(format);
        const auto * depthComponent = formatDescription.findComponent(codegen::vulkan::ComponentType::eD);
        if (!depthComponent) {
            continue;
        }
        if (depthFormat == vk::Format::eUndefined) {
            depthFormat = format;
            continue;
        }
        const auto & bestFormatDescription = codegen::vulkan::kFormatDescriptions.at(depthFormat);
        const auto * bestDepthComponent = bestFormatDescription.findComponent(codegen::vulkan::ComponentType::eD);
        ASSERT(bestDepthComponent);
        if (depthComponent->bitsize < bestDepthComponent->bitsize) {
            continue;
        }
        if (depthComponent->bitsize == bestDepthComponent->bitsize) {
            if (formatDescription.componentCount() > bestFormatDescription.componentCount()) {
                continue;
            }
            if (formatDescription.componentCount() == bestFormatDescription.componentCount()) {
                SPDLOG_WARN("{} is equivalent to {}", depthFormat, format);
            }
        }
        depthFormat = format;
    }
    return depthFormat;
}

void Renderer::Impl::setScene(std::shared_ptr<const Scene> scene)
{
    ASSERT(this->scene != scene);

    auto unmuteMessageGuard = context.getInstance().unmuteDebugUtilsMessages(kUnmutedMessageIdNumbers);

    sceneGraphicsPipeline.reset();

    frames.clear();
    sceneDescriptors.reset();
    if (scene) {
        sceneDescriptors = std::make_shared<const Scene::SceneDescriptors>(scene->makeSceneDescriptors());
    }

    this->scene = std::move(scene);
}

void Renderer::Impl::advance(uint32_t currentFrameSlot, const FrameSettings & frameSettings)
{
    ASSERT_MSG(currentFrameSlot < framesInFlight, "{} ^ {}", currentFrameSlot, framesInFlight);
    if (!scene) {
        return;
    }

    if (frameSettings.useOffscreenTexture) {
        if (!offscreenRenderPass) {
            offscreenRenderPass.emplace(OffscreenRenderPass::make(context));
        }
    } else {
        if (offscreenRenderPass) {
            offscreenRenderPass.reset();
        }
        if (!std::empty(framebuffers)) {
            framebuffers.clear();
        }
    }

    auto unmuteMessageGuard = context.getInstance().unmuteDebugUtilsMessages(kUnmutedMessageIdNumbers);

    // Frame & frame = frames.

    ASSERT(sceneDescriptors);
    {
        auto mappedUniformBuffer = frame.descriptors->uniformBuffer.value().map();
        fillUniformBuffer(frameSettings, *mappedUniformBuffer.data());
    }
}

void Renderer::Impl::render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, uint32_t currentFrameSlot, const FrameSettings & frameSettings)
{
    ASSERT(currentFrameSlot < framesInFlight);
    if (!scene) {
        return;
    }

    auto unmuteMessageGuard = context.getInstance().unmuteDebugUtilsMessages(kUnmutedMessageIdNumbers);

    constexpr engine::LabelColor kGreenColor = {0.0f, 1.0f, 0.0f, 1.0f};
    auto commandBufferLabel = engine::ScopedCommandBufferLabel::create(context.getDispatcher(), commandBuffer, "Renderer::render"sv, kGreenColor);

    ASSERT(sceneDescriptors);

    if (!sceneGraphicsPipeline || (sceneGraphicsPipeline->pipelineLayout.getAssociatedRenderPass() != renderPass)) {
        sceneGraphicsPipeline.reset();
        sceneGraphicsPipeline = scene->createGraphicsPipeline(renderPass, Scene::PipelineKind::kScenePipeline);
    }
    auto scenePipeline = sceneGraphicsPipeline->pipelines.getPipelines().at(0);
    auto scenePipelineLayout = sceneGraphicsPipeline->pipelineLayout.getPipelineLayout();

    constexpr engine::LabelColor kMagentaColor = {1.0f, 0.0f, 1.0f, 1.0f};
    auto rasterizationLabel = engine::ScopedCommandBufferLabel::create(context.getDispatcher(), commandBuffer, "Rasterization"sv, kMagentaColor);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, scenePipeline, context.getDispatcher());

    constexpr uint32_t kFirstSet = 0;
    if (scene->isDescriptorBufferEnabled()) {
        const auto & descriptorSetBuffers = frame.frameDescriptors->descriptorSetBuffers;
        if (!std::empty(descriptorSetBuffers)) {
            commandBuffer.bindDescriptorBuffersEXT(frame.frameDescriptors->descriptorBufferBindingInfos, context.getDispatcher());
            std::vector<uint32_t> bufferIndices(std::size(descriptorSetBuffers));
            std::iota(std::begin(bufferIndices), std::end(bufferIndices), bufferIndices.front());
            std::vector<vk::DeviceSize> offsets(std::size(descriptorSetBuffers), 0);
            // std::fill(std::begin(offsets), std::end(offsets), 0);
            commandBuffer.setDescriptorBufferOffsetsEXT(vk::PipelineBindPoint::eGraphics, scenePipelineLayout, kFirstSet, bufferIndices, offsets, context.getDispatcher());
        }
    } else {
        constexpr auto kDynamicOffsets = nullptr;
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, scenePipelineLayout, kFirstSet, frameDescriptors->descriptorSets.at(currentFrameSlot).getDescriptorSets(), kDynamicOffsets, context.getDispatcher());
    }

    {
        PushConstants pushConstants = getPushConstants(frameSettings);
        for (const auto & pushConstantRange : sceneDescriptors->pushConstantRanges) {
            const void * p = utils::safeCast<const std::byte *>(&pushConstants) + pushConstantRange.offset;
            commandBuffer.pushConstants(scenePipelineLayout, pushConstantRange.stageFlags, pushConstantRange.offset, pushConstantRange.size, p, context.getDispatcher());
        }
    }

    constexpr uint32_t kFirstViewport = 0;
    std::initializer_list<vk::Viewport> viewports = {
        frameSettings.viewport,
    };
    commandBuffer.setViewport(kFirstViewport, viewports, context.getDispatcher());

    constexpr uint32_t kFirstScissor = 0;
    std::initializer_list<vk::Rect2D> scissors = {
        frameSettings.scissor,
    };
    commandBuffer.setScissor(kFirstScissor, scissors, context.getDispatcher());

    constexpr uint32_t kFirstBinding = 0;
    const auto bufferOrNull = [this](const auto & wrapper) -> vk::Buffer
    {
        if (wrapper) {
            return wrapper.value();
        } else {
            if (sah_kd_tree::kIsDebugBuild) {
                INVARIANT(context.getPhysicalDevice().features2Chain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>().nullDescriptor == VK_TRUE, "");
            }
            return VK_NULL_HANDLE;
        }
    };
    std::initializer_list<vk::Buffer> vertexBuffers = {
        bufferOrNull(sceneDescriptors->vertexBuffer),
    };
    std::vector<vk::DeviceSize> vertexBufferOffsets(std::size(vertexBuffers), 0);
    commandBuffer.bindVertexBuffers(kFirstBinding, vertexBuffers, vertexBufferOffsets, context.getDispatcher());

    vk::Buffer indexBuffer;
    vk::DeviceSize indexBufferSize = 0;
    if (sceneDescriptors->indexBuffer) {
        indexBuffer = sceneDescriptors->indexBuffer.value();
        indexBufferSize = sceneDescriptors->indexBuffer.value().getSize();
    }
    constexpr vk::DeviceSize kIndexBufferDeviceOffset = 0;
    if (scene->isMultiDrawIndirectEnabled()) {
        const auto indexType = sceneDescriptors->indexTypes.at(0);
        commandBuffer.bindIndexBuffer2KHR(indexBuffer, kIndexBufferDeviceOffset, indexBufferSize, indexType, context.getDispatcher());
        constexpr vk::DeviceSize kInstanceBufferOffset = 0;
        constexpr uint32_t kStride = sizeof(vk::DrawIndexedIndirectCommand);
        uint32_t drawCount = sceneDescriptors->drawCount;
        const auto & physicalDeviceLimits = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits;
        INVARIANT(drawCount <= physicalDeviceLimits.maxDrawIndirectCount, "{} ^ {}", drawCount, physicalDeviceLimits.maxDrawIndirectCount);
        if (scene->isDrawIndirectCountEnabled()) {
            constexpr vk::DeviceSize kDrawCountBufferOffset = 0;
            uint32_t maxDrawCount = drawCount;
            commandBuffer.drawIndexedIndirectCount(sceneDescriptors->instanceBuffer.value(), kInstanceBufferOffset, sceneDescriptors->drawCountBuffer.value(), kDrawCountBufferOffset, maxDrawCount, kStride, context.getDispatcher());
        } else {
            commandBuffer.drawIndexedIndirect(sceneDescriptors->instanceBuffer.value(), kInstanceBufferOffset, drawCount, kStride, context.getDispatcher());
        }
    } else {
        auto indexType = std::cbegin(sceneDescriptors->indexTypes);
        for (const auto & [indexCount, instanceCount, firstIndex, vertexOffset, firstInstance] : sceneDescriptors->instances) {
            ASSERT(indexType != std::cend(sceneDescriptors->indexTypes));
            commandBuffer.bindIndexBuffer2KHR(indexBuffer, kIndexBufferDeviceOffset, indexBufferSize, *indexType++, context.getDispatcher());
            commandBuffer.drawIndexed(indexCount, instanceCount, firstIndex, vertexOffset, firstInstance, context.getDispatcher());
            // SPDLOG_TRACE("{{.indexCount = {}, .instanceCount = {}, .firstIndex = {}, .vertexOffset = {}, .firstInstance = {})}}", indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
        }
    }
}

auto Renderer::Impl::getFramebuffer(const vk::Extent2D & size) -> Framebuffer
{
    ASSERT(offscreenRenderPass);
    while (!std::empty(framebuffers)) {
        auto framebuffer = std::move(framebuffers.top());
        framebuffers.pop();
        if (framebuffer.size == size) {
            ASSERT(offscreenRenderPass.value().renderPass);
            ASSERT(framebuffer.renderPass == *offscreenRenderPass.value().renderPass);
            return framebuffer;
        }
    }
    return Framebuffer::make(context, size, offscreenRenderPass.value());
}

void Renderer::Impl::putFramebuffer(Framebuffer && framebuffer)
{
    framebuffers.push(std::move(framebuffer));
}

auto Renderer::Impl::getFrame() -> Frame
{
    if (!std::empty(frames)) {
        auto frame = std::move(frames.front());
        frames.pop();
        return frame;
    }
    return {
        .descriptors = std::make_shared<const Scene::FrameDescriptors>(scene->makeFrameDescriptors(*sceneDescriptors)),
    };
}

void Renderer::Impl::putFramebuffer(Frame && frame)
{
    frames.push(std::move(frame));
}

}  // namespace viewer
