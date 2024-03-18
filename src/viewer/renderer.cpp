#include <codegen/vulkan_utils.hpp>
#include <common/version.hpp>
#include <engine/context.hpp>
#include <engine/debug_utils.hpp>
#include <engine/device.hpp>
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
#include <utils/math.hpp>
#include <utils/noncopyable.hpp>
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
#include <tuple>
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

using Resource = std::shared_ptr<const void>;

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

class Recycler final : utils::OneTime<Recycler>
{
public:
    template<typename F, typename... Args>
    Recycler(F && f, Args &&... args) : holder{makeHolder<F, Args...>(f, args..., std::index_sequence_for<Args...>{})}  // NOLINT: google-explicit-constructor
    {}

    [[nodiscard]] operator Resource() && noexcept  // NOLINT: google-explicit-constructor
    {
        return std::move(holder);
    }

private:
    using Holder = std::unique_ptr<void, void (*)(void * p)>;

    Holder holder;

    template<typename F, typename... Args, size_t... Indices>
    [[nodiscard]] static Holder makeHolder(F & f, Args &... args, std::index_sequence<Indices...>)
    {
        static_assert(std::is_invocable_v<F &&, Args &&...>);
        using Storage = std::tuple<std::decay_t<F>, std::decay_t<Args>...>;
        constexpr auto recycle = [](void * p)
        {
            std::unique_ptr<Storage> storage{static_cast<Storage *>(p)};
            std::invoke(std::forward<F>(std::get<0>(*storage)), std::forward<Args>(std::get<1 + Indices>(*storage))...);
        };
        return {new Storage{std::forward<F>(f), std::forward<Args>(args)...}, recycle};
    }

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

template<typename T, typename U>
[[nodiscard]] std::vector<T> concatenate(const std::vector<U> & first, const std::vector<U> & second)
{
    std::vector<T> result;
    result.reserve(std::size(first) + std::size(second));
    result.insert(std::cend(result), std::cbegin(first), std::cend(first));
    result.insert(std::cend(result), std::cbegin(second), std::cend(second));
    return result;
}

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
        .position = frameSettings.position,
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

struct Renderer::Impl : utils::NonCopyable
{
    struct OffscreenRenderPass
    {
        vk::Format depthFormat = vk::Format::eUndefined;
        vk::ImageLayout depthImageLayout = vk::ImageLayout::eUndefined;

        vk::UniqueRenderPass renderPass;

        [[nodiscard]] static OffscreenRenderPass make(const engine::Context & context);
    };

    struct Framebuffer
    {
        static constexpr vk::Format kFormat = vk::Format::eR8G8B8A8Unorm;

        std::shared_ptr<const OffscreenRenderPass> offscreenRenderPass;
        vk::Extent2D size;

        vk::ImageAspectFlags depthImageAspectMask = vk::ImageAspectFlagBits::eNone;

        engine::Image colorImage;
        vk::UniqueImageView colorImageView;

        engine::Image depthImage;
        vk::UniqueImageView depthImageView;

        vk::UniqueFramebuffer framebuffer;

        [[nodiscard]] static Framebuffer make(const engine::Context & context, const vk::Extent2D & size, std::shared_ptr<const OffscreenRenderPass> offscreenRenderPass);
    };

    const engine::Context & context;
    const uint32_t framesInFlight;

    std::shared_ptr<const Scene> scene;
    utils::CheckedPtr<const std::vector<vk::PushConstantRange>> pushConstantRanges = nullptr;
    std::shared_ptr<const Scene::SceneDescriptors> sceneDescriptors;
    std::shared_ptr<const Scene::GraphicsPipeline> sceneGraphicsPipeline;
    ResourceStack<std::shared_ptr<const Scene::FrameDescriptors>> frameDescriptorsPool;
    std::shared_ptr<const Scene::FrameDescriptors> frameDescriptors;

    std::shared_ptr<const OffscreenRenderPass> offscreenRenderPass;
    ResourceStack<std::shared_ptr<const Framebuffer>> framebufferPool;

    // revocation lists should be the last members
    std::vector<std::vector<Resource>> deferredDeletionSlots{framesInFlight};

    Impl(const engine::Context & context, uint32_t framesInFlight);

    [[nodiscard]] static vk::Format findDepthImageFormat(const engine::Context & context, vk::ImageTiling imageTiling);

    void setScene(std::shared_ptr<const Scene> scene);
    void advance(uint32_t currentFrameSlot, const FrameSettings & frameSettings);
    [[nodiscard]] bool updateRenderPass(vk::RenderPass renderPass);
    void render(vk::CommandBuffer commandBuffer, uint32_t currentFrameSlot, const FrameSettings & frameSettings) const;

    std::shared_ptr<const Framebuffer> getFramebuffer(const vk::Extent2D & size);
    void putFramebuffer(std::shared_ptr<const Framebuffer> && framebuffer);

    std::shared_ptr<const Scene::FrameDescriptors> getFrameDescriptors();
    void putFrameDescriptors(std::shared_ptr<const Scene::FrameDescriptors> && frameDescriptors);

    template<typename... Resources>
    void deferDeletion(uint32_t previousFrameSlot, Resources &&... resources)
    {
        auto & slotResources = deferredDeletionSlots.at(previousFrameSlot);
        (slotResources.emplace_back(std::forward<Resources>(resources)), ...);
    }

    void deleteDeferred(uint32_t currentFrameSlot)
    {
        deferredDeletionSlots.at(currentFrameSlot).clear();
    }
};

Renderer::Renderer(const engine::Context & context, uint32_t framesInFlight) : impl_{context, framesInFlight}
{}

Renderer::~Renderer() = default;

void Renderer::setScene(std::shared_ptr<const Scene> scene)
{
    return impl_->setScene(std::move(scene));
}

void Renderer::advance(uint32_t currentFrameSlot, const FrameSettings & frameSettings)
{
    return impl_->advance(currentFrameSlot, frameSettings);
}

bool Renderer::updateRenderPass(vk::RenderPass renderPass)
{
    return impl_->updateRenderPass(renderPass);
}

void Renderer::render(vk::CommandBuffer commandBuffer, uint32_t currentFrameSlot, const FrameSettings & frameSettings) const
{
    return impl_->render(commandBuffer, currentFrameSlot, frameSettings);
}

auto Renderer::Impl::OffscreenRenderPass::make(const engine::Context & context) -> OffscreenRenderPass
{
    vk::Format depthFormat = findDepthImageFormat(context, vk::ImageTiling::eOptimal);
    INVARIANT(depthFormat != vk::Format::eUndefined, "");
    vk::ImageLayout depthImageLayout = vk::ImageLayout::eUndefined;
    if (context.getDevice().createInfoChain.get<vk::PhysicalDeviceVulkan12Features>().separateDepthStencilLayouts == VK_FALSE) {
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
        .depthImageLayout = depthImageLayout,
        .renderPass = std::move(renderPass),
    };
}

auto Renderer::Impl::Framebuffer::make(const engine::Context & context, const vk::Extent2D & size, std::shared_ptr<const OffscreenRenderPass> offscreenRenderPass) -> Framebuffer
{
    ASSERT(offscreenRenderPass);
    ASSERT(offscreenRenderPass->renderPass);
    auto renderPass = *offscreenRenderPass->renderPass;

    vk::ImageAspectFlags depthImageAspectMask = vk::ImageAspectFlagBits::eDepth;
    if (context.getDevice().createInfoChain.get<vk::PhysicalDeviceVulkan12Features>().separateDepthStencilLayouts == VK_FALSE) {
        depthImageAspectMask |= vk::ImageAspectFlagBits::eStencil;
    }

    auto colorImageName = "offscreen framebuffer color image"s;
    constexpr vk::ImageUsageFlags kColorImageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
    constexpr vk::ImageAspectFlags kColorImageAspectMask = vk::ImageAspectFlagBits::eColor;
    auto colorImage = context.getMemoryAllocator().createImage2D(colorImageName, Framebuffer::kFormat, size, kColorImageUsage, kColorImageAspectMask);
    auto colorImageView = colorImage.createImageView(vk::ImageViewType::e2D, kColorImageAspectMask);

    auto depthImageName = "offscreen framebuffer depth image"s;
    constexpr vk::ImageUsageFlags kDepthImageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
    auto depthImage = context.getMemoryAllocator().createImage2D(depthImageName, offscreenRenderPass->depthFormat, size, kDepthImageUsage, depthImageAspectMask);
    auto depthImageView = depthImage.createImageView(vk::ImageViewType::e2D, depthImageAspectMask);

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

    return {
        .offscreenRenderPass = std::move(offscreenRenderPass),
        .size = size,
        .depthImageAspectMask = depthImageAspectMask,
        .colorImage = std::move(colorImage),
        .colorImageView = std::move(colorImageView),
        .depthImage = std::move(depthImage),
        .depthImageView = std::move(depthImageView),
        .framebuffer = context.getDevice().getDevice().createFramebufferUnique(framebufferCreateInfo, context.getAllocationCallbacks(), context.getDispatcher()),
    };
}

Renderer::Impl::Impl(const engine::Context & context, uint32_t framesInFlight) : context{context}, framesInFlight{framesInFlight}
{}

vk::Format Renderer::Impl::findDepthImageFormat(const engine::Context & context, vk::ImageTiling imageTiling)
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
    framebufferPool.clear();
    offscreenRenderPass.reset();
    frameDescriptors.reset();
    frameDescriptorsPool.clear();
    sceneDescriptors.reset();
    pushConstantRanges = nullptr;
    this->scene = std::move(scene);
}

void Renderer::Impl::advance(uint32_t currentFrameSlot, const FrameSettings & frameSettings)
{
    ASSERT_MSG(currentFrameSlot < framesInFlight, "{} ^ {}", currentFrameSlot, framesInFlight);
    if (!scene) {
        return;
    }

    auto unmuteMessageGuard = context.getInstance().unmuteDebugUtilsMessages(kUnmutedMessageIdNumbers);

    deleteDeferred(currentFrameSlot);

    if (!pushConstantRanges) {
        pushConstantRanges = &scene->getPushConstantRanges();
    }
    if (!sceneDescriptors) {
        sceneDescriptors = std::make_shared<const Scene::SceneDescriptors>(scene->makeSceneDescriptors());
    }
    if (frameDescriptors) {
        uint32_t previousFrame = utils::modDown(currentFrameSlot, framesInFlight);
        Recycler recycler{&Impl::putFrameDescriptors, this, std::move(frameDescriptors)};  // 'this' captured, Renderer::Impl should be NonCopyable
        deferDeletion(previousFrame, std::move(recycler));
    }
    frameDescriptors = getFrameDescriptors();

    if (frameSettings.useOffscreenTexture) {
        if (!offscreenRenderPass) {
            offscreenRenderPass = std::make_shared<OffscreenRenderPass>(OffscreenRenderPass::make(context));
        }
    } else {
        framebufferPool.clear();
        offscreenRenderPass.reset();
    }

    fillUniformBuffer(frameSettings, frameDescriptors->resources.uniformBuffer.map().at(0));
}

bool Renderer::Impl::updateRenderPass(vk::RenderPass renderPass)
{
    if (sceneGraphicsPipeline) {
        if (sceneGraphicsPipeline->pipelineLayout.getAssociatedRenderPass() == renderPass) {
            return false;
        }
        sceneGraphicsPipeline.reset();
    }
    sceneGraphicsPipeline = scene->createGraphicsPipeline(renderPass, Scene::PipelineKind::kScenePipeline);
    return true;
}

void Renderer::Impl::render(vk::CommandBuffer commandBuffer, uint32_t currentFrameSlot, const FrameSettings & frameSettings) const
{
    ASSERT(currentFrameSlot < framesInFlight);
    if (!scene) {
        return;
    }

    auto unmuteMessageGuard = context.getInstance().unmuteDebugUtilsMessages(kUnmutedMessageIdNumbers);

    constexpr engine::LabelColor kGreenColor = {0.0f, 1.0f, 0.0f, 1.0f};
    auto commandBufferLabel = engine::ScopedCommandBufferLabel::create(context.getDispatcher(), commandBuffer, "Renderer::render"sv, kGreenColor);

    ASSERT(sceneDescriptors);

    auto scenePipelineLayout = sceneGraphicsPipeline->pipelineLayout.getPipelineLayout();
    auto scenePipeline = sceneGraphicsPipeline->pipelines.getPipelines().at(0);

    constexpr engine::LabelColor kMagentaColor = {1.0f, 0.0f, 1.0f, 1.0f};
    auto rasterizationLabel = engine::ScopedCommandBufferLabel::create(context.getDispatcher(), commandBuffer, "Rasterization"sv, kMagentaColor);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, scenePipeline, context.getDispatcher());

    constexpr uint32_t kFirstSet = 0;
    if (scene->isDescriptorBufferEnabled()) {
        vk::DescriptorBufferBindingInfoEXT descriptorBufferBindingInfos[] = {
            sceneDescriptors->getDescriptorBuffer().getDescriptorBufferBindingInfo(),
            frameDescriptors->getDescriptorBuffer().getDescriptorBufferBindingInfo(),
        };
        commandBuffer.bindDescriptorBuffersEXT(descriptorBufferBindingInfos, context.getDispatcher());

        uint32_t bufferIndices[std::size(descriptorBufferBindingInfos)];
        std::iota(std::begin(bufferIndices), std::end(bufferIndices), uint32_t{0});
        vk::DeviceSize offsets[std::size(descriptorBufferBindingInfos)];
        std::fill(std::begin(offsets), std::end(offsets), vk::DeviceSize{0});
        commandBuffer.setDescriptorBufferOffsetsEXT(vk::PipelineBindPoint::eGraphics, scenePipelineLayout, kFirstSet, bufferIndices, offsets, context.getDispatcher());
    } else {
        vk::DescriptorSet descriptorSets[] = {
            sceneDescriptors->getDescriptorSet(),
            frameDescriptors->getDescriptorSet(),
        };
        constexpr auto kDynamicOffsets = nullptr;
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, scenePipelineLayout, kFirstSet, descriptorSets, kDynamicOffsets, context.getDispatcher());
    }

    {
        PushConstants pushConstants = getPushConstants(frameSettings);
        ASSERT(pushConstantRanges);
        for (const auto & pushConstantRange : *pushConstantRanges) {
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

    const auto & sceneResources = sceneDescriptors->resources;

    constexpr uint32_t kFirstBinding = 0;
    const auto bufferOrNull = [this](const auto & wrapper) -> vk::Buffer
    {
        if (wrapper) {
            return wrapper.value();
        } else {
            ASSERT(context.getPhysicalDevice().features2Chain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>().nullDescriptor == VK_TRUE);
            return VK_NULL_HANDLE;
        }
    };
    vk::Buffer vertexBuffer = bufferOrNull(sceneResources.vertexBuffer);
    vk::DeviceSize vertexBufferOffset = 0;
    commandBuffer.bindVertexBuffers(kFirstBinding, vertexBuffer, vertexBufferOffset, context.getDispatcher());

    vk::Buffer indexBuffer;
    vk::DeviceSize indexBufferSize = 0;
    if (sceneResources.indexBuffer) {
        indexBuffer = sceneResources.indexBuffer.value();
        indexBufferSize = sceneResources.indexBuffer.value().getSize();
    } else {
        ASSERT(context.getPhysicalDevice().features2Chain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>().nullDescriptor == VK_TRUE);
        ASSERT(context.getPhysicalDevice().features2Chain.get<vk::PhysicalDeviceMaintenance6FeaturesKHR>().maintenance6 == VK_TRUE);
    }
    constexpr vk::DeviceSize kIndexBufferDeviceOffset = 0;
    if (scene->isMultiDrawIndirectEnabled()) {
        auto indexType = sceneResources.indexTypes.at(0);
        commandBuffer.bindIndexBuffer2KHR(indexBuffer, kIndexBufferDeviceOffset, indexBufferSize, indexType, context.getDispatcher());
        constexpr vk::DeviceSize kInstanceBufferOffset = 0;
        constexpr uint32_t kStride = sizeof(vk::DrawIndexedIndirectCommand);
        uint32_t drawCount = sceneResources.drawCount;
        const auto & physicalDeviceLimits = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits;
        INVARIANT(drawCount <= physicalDeviceLimits.maxDrawIndirectCount, "{} ^ {}", drawCount, physicalDeviceLimits.maxDrawIndirectCount);
        if (scene->isDrawIndirectCountEnabled()) {
            constexpr vk::DeviceSize kDrawCountBufferOffset = 0;
            uint32_t maxDrawCount = drawCount;
            commandBuffer.drawIndexedIndirectCount(sceneResources.instanceBuffer.value(), kInstanceBufferOffset, sceneResources.drawCountBuffer.value(), kDrawCountBufferOffset, maxDrawCount, kStride, context.getDispatcher());
        } else {
            commandBuffer.drawIndexedIndirect(sceneResources.instanceBuffer.value(), kInstanceBufferOffset, drawCount, kStride, context.getDispatcher());
        }
    } else {
        auto indexType = std::cbegin(sceneResources.indexTypes);
        for (const auto & [indexCount, instanceCount, firstIndex, vertexOffset, firstInstance] : sceneResources.instances) {
            ASSERT(indexType != std::cend(sceneResources.indexTypes));
            commandBuffer.bindIndexBuffer2KHR(indexBuffer, kIndexBufferDeviceOffset, indexBufferSize, *indexType++, context.getDispatcher());
            commandBuffer.drawIndexed(indexCount, instanceCount, firstIndex, vertexOffset, firstInstance, context.getDispatcher());
            // SPDLOG_TRACE("{{.indexCount = {}, .instanceCount = {}, .firstIndex = {}, .vertexOffset = {}, .firstInstance = {})}}", indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
        }
    }
}

auto Renderer::Impl::getFramebuffer(const vk::Extent2D & size) -> std::shared_ptr<const Framebuffer>
{
    ASSERT(offscreenRenderPass);
    while (!std::empty(framebufferPool)) {
        auto framebuffer = std::move(framebufferPool.top());
        framebufferPool.pop();
        if ((framebuffer->offscreenRenderPass == offscreenRenderPass) && (framebuffer->size == size)) {
            ASSERT(offscreenRenderPass->renderPass);
            return framebuffer;
        }
    }
    return std::make_shared<Framebuffer>(Framebuffer::make(context, size, offscreenRenderPass));
}

void Renderer::Impl::putFramebuffer(std::shared_ptr<const Framebuffer> && framebuffer)
{
    ASSERT_MSG(framebuffer.use_count() == 1, "Non-unique use in single-threaded context: {}", framebuffer.use_count());
    framebufferPool.push(std::move(framebuffer));
}

auto Renderer::Impl::getFrameDescriptors() -> std::shared_ptr<const Scene::FrameDescriptors>
{
    while (!std::empty(frameDescriptorsPool)) {
        auto frameDescriptors = std::move(frameDescriptorsPool.top());
        frameDescriptorsPool.pop();
        if (true) {
            return frameDescriptors;
        }
    }
    return std::make_shared<Scene::FrameDescriptors>(scene->makeFrameDescriptors());
}

void Renderer::Impl::putFrameDescriptors(std::shared_ptr<const Scene::FrameDescriptors> && frameDescriptors)
{
    ASSERT_MSG(frameDescriptors.use_count() == 1, "Non-unique use in single-threaded context: {}", frameDescriptors.use_count());
    frameDescriptorsPool.push(std::move(frameDescriptors));
}

}  // namespace viewer
