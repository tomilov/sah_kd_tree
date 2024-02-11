#include <codegen/vulkan_utils.hpp>
#include <common/version.hpp>
#include <engine/context.hpp>
#include <engine/debug_utils.hpp>
#include <engine/device.hpp>
#include <engine/framebuffer.hpp>
#include <engine/graphics_pipeline.hpp>
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

#include <algorithm>
#include <filesystem>
#include <initializer_list>
#include <iterator>
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
class Stack : std::stack<T, std::vector<T>>
{
    using base = std::stack<T, std::vector<T>>;

public:
    using base::base;
    using base::empty;
    using base::pop;
    using base::push;
    using base::top;

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
    struct Framebuffer
    {
        static constexpr vk::Format kFormat = vk::Format::eR8G8B8A8Unorm;

        vk::Extent2D size;

        engine::Image colorImage;
        vk::UniqueImageView colorImageView;

        engine::Image depthImage;
        vk::UniqueImageView depthImageView;

        vk::UniqueFramebuffer framebuffer;
    };

    const engine::Context & context;
    const uint32_t framesInFlight;

    std::shared_ptr<const Scene> scene;
    std::shared_ptr<const Scene::Descriptors> descriptors;
    std::shared_ptr<const Scene::GraphicsPipeline> graphicsPipeline;

    vk::Format depthFormat = vk::Format::eUndefined;
    vk::ImageAspectFlags depthImageAspect = vk::ImageAspectFlagBits::eNone;
    vk::ImageLayout depthImageLayout = vk::ImageLayout::eUndefined;

    vk::UniqueRenderPass offscreenRenderPass;
    Stack<Framebuffer> framebuffers;

    Impl(const engine::Context & context, uint32_t framesInFlight) : context{context}, framesInFlight{framesInFlight}
    {}

    void createOffscreenRenderPass();

    void setScene(std::shared_ptr<const Scene> scene);
    void advance(uint32_t currentFrameSlot, const FrameSettings & frameSettings);
    void render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, uint32_t currentFrameSlot, const FrameSettings & frameSettings);

    [[nodiscard]] std::shared_ptr<const Scene> getScene() const
    {
        return scene;
    }

    [[nodiscard]] vk::Format findSupportedDepthFormat(vk::ImageTiling imageTiling) const
    {
        const vk::FormatFeatureFlags2 vk::FormatProperties3::*p = nullptr;
        if (imageTiling == vk::ImageTiling::eLinear) {
            p = &vk::FormatProperties3::linearTilingFeatures;
        } else if (imageTiling == vk::ImageTiling::eOptimal) {
            p = &vk::FormatProperties3::optimalTilingFeatures;
        } else {
            INVARIANT(false, "{}", imageTiling);
        }

        const auto findDepthComponent = [](const codegen::vulkan::FormatComponent(&components)[4]) -> const codegen::vulkan::FormatComponent *
        {
            for (const auto & component : components) {
                if (component.componentType == codegen::vulkan::ComponentType::eD) {
                    return &component;
                }
            }
            return nullptr;
        };

        constexpr vk::FormatFeatureFlags2 kFormatFeatureFlags = vk::FormatFeatureFlagBits2::eDepthStencilAttachment;
        const auto physicalDevice = context.getPhysicalDevice().getPhysicalDevice();
        vk::Format depthFormat = vk::Format::eUndefined;
        const codegen::vulkan::Format * f = nullptr;
        for (const auto & [vkFormat, format] : codegen::vulkan::kFormats) {
            auto formatProperties2Chain = physicalDevice.getFormatProperties2<vk::FormatProperties2, vk::FormatProperties3>(vkFormat, context.getDispatcher());
            if ((formatProperties2Chain.get<vk::FormatProperties3>().*p & kFormatFeatureFlags) == kFormatFeatureFlags) {
                const auto * depthComponent = findDepthComponent(format.components);
                INVARIANT(depthComponent, "{}", vkFormat);
                auto bitsize = depthComponent->bitsize;
                auto currentBitsize = findDepthComponent(f->components)->bitsize;
                if (!f || (bitsize > currentBitsize) || ((bitsize == currentBitsize) && (format.componentCount() < f->componentCount()))) {
                    depthFormat = vkFormat;
                    f = &format;
                }
            }
        }
        INVARIANT(depthFormat != vk::Format::eUndefined, "");
        return depthFormat;
    }

    Framebuffer getFramebuffer(const vk::Extent2D & size)
    {
        ASSERT(offscreenRenderPass);

        while (!std::empty(framebuffers)) {
            auto framebuffer = std::move(framebuffers.top());
            framebuffers.pop();
            if (framebuffer.size == size) {
                return framebuffer;
            }
        }

        engine::AllocationCreateInfo colorAllocationCreateInfo = {
            .name = "offscreen framebuffer color image",
            .type = engine::AllocationCreateInfo::AllocationType::kAuto,
        };
        auto colorImage = engine::Image::createImage2D(context.getMemoryAllocator(), colorAllocationCreateInfo, Framebuffer::kFormat, size, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);
        auto colorImageView = colorImage.createImageView2D();

        engine::AllocationCreateInfo depthAllocationCreateInfo = {
            .name = "offscreen framebuffer depth image",
            .type = engine::AllocationCreateInfo::AllocationType::kAuto,
        };
        auto depthImage = engine::Image::createImage2D(context.getMemoryAllocator(), depthAllocationCreateInfo, depthFormat, size, vk::ImageUsageFlagBits::eDepthStencilAttachment);
        auto depthImageView = depthImage.createImageView2D();

        const vk::ImageView attachments[] = {
            *colorImageView,
            *depthImageView,
        };
        vk::FramebufferCreateInfo framebufferCreateInfo = {
            .flags = {},
            .renderPass = *offscreenRenderPass,
            .width = size.width,
            .height = size.height,
            .layers = 1,
        };
        framebufferCreateInfo.setAttachments(attachments);
        auto framebuffer = context.getDevice().getDevice().createFramebufferUnique(framebufferCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());

        return {
            .size = size,
            .colorImage = std::move(colorImage),
            .colorImageView = std::move(colorImageView),
            .depthImage = std::move(depthImage),
            .depthImageView = std::move(depthImageView),
            .framebuffer = std::move(framebuffer),
        };
    }
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
    return impl_->getScene();
}

void Renderer::Impl::createOffscreenRenderPass()
{
    ASSERT(offscreenRenderPass);
    if (context.getDevice().createInfoChain.get<vk::PhysicalDeviceSeparateDepthStencilLayoutsFeatures>().separateDepthStencilLayouts == VK_FALSE) {
        INVARIANT(false, "");
    }

    depthFormat = findSupportedDepthFormat(vk::ImageTiling::eOptimal);
    depthImageAspect = vk::ImageAspectFlagBits::eDepth;
    depthImageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
    {
        for (uint8_t i = 0; i < vk::componentCount(depthFormat); ++i) {
            if (vk::componentName(depthFormat, i) == "S"sv) {
                depthImageAspect |= vk::ImageAspectFlagBits::eStencil;
                depthImageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
            }
        }
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

    offscreenRenderPass = context.getDevice().getDevice().createRenderPassUnique(renderPassCreateInfo, context.getAllocationCallbacks(), context.getDispatcher());
    context.getDevice().setDebugUtilsObjectName(*offscreenRenderPass, "Offscreen renderpass");
}

void Renderer::Impl::setScene(std::shared_ptr<const Scene> scene)
{
    ASSERT(this->scene != scene);

    auto unmuteMessageGuard = context.getInstance().unmuteDebugUtilsMessages(kUnmutedMessageIdNumbers);

    graphicsPipeline.reset();

    descriptors.reset();
    if (scene) {
        descriptors = std::make_shared<Scene::Descriptors>(scene->makeDescriptors(framesInFlight));
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
            createOffscreenRenderPass();
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

    ASSERT(descriptors);
    {
        auto mappedUniformBuffer = descriptors->uniformBuffers.at(currentFrameSlot).map();
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
    auto commandBufferLabel = engine::ScopedCommandBufferLabel::create(context.getDispatcher(), commandBuffer, "Renderer::render", kGreenColor);

    ASSERT(descriptors);

    if (!graphicsPipeline || (graphicsPipeline->pipelineLayout.getAssociatedRenderPass() != renderPass)) {
        graphicsPipeline.reset();
        graphicsPipeline = scene->createGraphicsPipeline(renderPass);
    }
    auto pipeline = graphicsPipeline->pipelines.getPipelines().at(0);
    auto pipelineLayout = graphicsPipeline->pipelineLayout.getPipelineLayout();

    constexpr engine::LabelColor kMagentaColor = {1.0f, 0.0f, 1.0f, 1.0f};
    auto rasterizationLabel = engine::ScopedCommandBufferLabel::create(context.getDispatcher(), commandBuffer, "Rasterization", kMagentaColor);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline, context.getDispatcher());

    constexpr uint32_t kFirstSet = 0;
    if (scene->isDescriptorBufferEnabled()) {
        const auto & descriptorSetBuffers = descriptors->descriptorSetBuffers;
        if (!std::empty(descriptorSetBuffers)) {
            commandBuffer.bindDescriptorBuffersEXT(descriptors->descriptorBufferBindingInfos, context.getDispatcher());
            std::vector<uint32_t> bufferIndices(std::size(descriptorSetBuffers));
            std::iota(std::begin(bufferIndices), std::end(bufferIndices), bufferIndices.front());
            std::vector<vk::DeviceSize> offsets;
            offsets.reserve(std::size(descriptorSetBuffers));
            ASSERT(framesInFlight > 0);
            for (const auto & descriptorSetBuffer : descriptorSetBuffers) {
                offsets.push_back((descriptorSetBuffer.getSize() / framesInFlight) * currentFrameSlot);
            }
            commandBuffer.setDescriptorBufferOffsetsEXT(vk::PipelineBindPoint::eGraphics, pipelineLayout, kFirstSet, bufferIndices, offsets, context.getDispatcher());
        }
    } else {
        constexpr auto kDynamicOffsets = nullptr;
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, kFirstSet, descriptors->descriptorSets.at(currentFrameSlot).getDescriptorSets(), kDynamicOffsets, context.getDispatcher());
    }

    {
        PushConstants pushConstants = getPushConstants(frameSettings);
        for (const auto & pushConstantRange : descriptors->pushConstantRanges) {
            const void * p = utils::safeCast<const std::byte *>(&pushConstants) + pushConstantRange.offset;
            commandBuffer.pushConstants(pipelineLayout, pushConstantRange.stageFlags, pushConstantRange.offset, pushConstantRange.size, p, context.getDispatcher());
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
    constexpr auto bufferOrNull = [](const auto & wrapper) -> vk::Buffer
    {
        if (wrapper) {
            return wrapper.value();
        } else {
            return VK_NULL_HANDLE;
        }
    };
    std::initializer_list<vk::Buffer> vertexBuffers = {
        bufferOrNull(descriptors->vertexBuffer),
    };
    if (sah_kd_tree::kIsDebugBuild) {
        for (const auto & vertexBuffer : vertexBuffers) {
            if (!vertexBuffer) {
                INVARIANT(context.getPhysicalDevice().features2Chain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>().nullDescriptor == VK_TRUE, "");
            }
        }
    }
    std::vector<vk::DeviceSize> vertexBufferOffsets(std::size(vertexBuffers), 0);
    commandBuffer.bindVertexBuffers(kFirstBinding, vertexBuffers, vertexBufferOffsets, context.getDispatcher());

    vk::Buffer indexBuffer;
    vk::DeviceSize indexBufferSize = 0;
    if (descriptors->indexBuffer) {
        indexBuffer = descriptors->indexBuffer.value();
        indexBufferSize = descriptors->indexBuffer.value().getSize();
    }
    constexpr vk::DeviceSize kIndexBufferDeviceOffset = 0;
    if (scene->isMultiDrawIndirectEnabled()) {
        commandBuffer.bindIndexBuffer2KHR(indexBuffer, kIndexBufferDeviceOffset, indexBufferSize, descriptors->indexTypes.at(0), context.getDispatcher());
        constexpr vk::DeviceSize kInstanceBufferOffset = 0;
        constexpr uint32_t kStride = sizeof(vk::DrawIndexedIndirectCommand);
        uint32_t drawCount = descriptors->drawCount;
        const auto & physicalDeviceLimits = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits;
        INVARIANT(drawCount <= physicalDeviceLimits.maxDrawIndirectCount, "{} ^ {}", drawCount, physicalDeviceLimits.maxDrawIndirectCount);
        if (scene->isDrawIndirectCountEnabled()) {
            constexpr vk::DeviceSize kDrawCountBufferOffset = 0;
            uint32_t maxDrawCount = drawCount;
            commandBuffer.drawIndexedIndirectCount(descriptors->instanceBuffer.value(), kInstanceBufferOffset, descriptors->drawCountBuffer.value(), kDrawCountBufferOffset, maxDrawCount, kStride, context.getDispatcher());
        } else {
            commandBuffer.drawIndexedIndirect(descriptors->instanceBuffer.value(), kInstanceBufferOffset, drawCount, kStride, context.getDispatcher());
        }
    } else {
        auto indexType = std::cbegin(descriptors->indexTypes);
        for (const auto & [indexCount, instanceCount, firstIndex, vertexOffset, firstInstance] : descriptors->instances) {
            ASSERT(indexType != std::cend(descriptors->indexTypes));
            commandBuffer.bindIndexBuffer2KHR(indexBuffer, kIndexBufferDeviceOffset, indexBufferSize, *indexType++, context.getDispatcher());
            commandBuffer.drawIndexed(indexCount, instanceCount, firstIndex, vertexOffset, firstInstance, context.getDispatcher());
            // SPDLOG_TRACE("{{.indexCount = {}, .instanceCount = {}, .firstIndex = {}, .vertexOffset = {}, .firstInstance = {})}}", indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
        }
    }
}

}  // namespace viewer
