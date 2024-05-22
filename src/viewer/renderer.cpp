#include <codegen/vulkan_utils.hpp>
#include <common/version.hpp>
#include <engine/command_buffer.hpp>
#include <engine/command_pool.hpp>
#include <engine/context.hpp>
#include <engine/debug_utils.hpp>
#include <engine/device.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/image.hpp>
#include <engine/instance.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/queue.hpp>
#include <engine/vma.hpp>
#include <format/vulkan.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/checked_ptr.hpp>
#include <utils/math.hpp>
#include <utils/noncopyable.hpp>
#include <viewer/renderer.hpp>
#include <viewer/scene_manager.hpp>

#include <fmt/format.h>
#include <fmt/std.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_format_traits.hpp>

#include <queue>

#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <limits>
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
class ResourceQueue final : std::queue<T, std::list<T>>
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
    Recycler(F && f, Args &&... args)  // NOLINT: google-explicit-constructor
        : holder{makeHolder<F, Args...>(f, args..., std::index_sequence_for<Args...>{})}
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

class DisplayDescriptorPool final : utils::NonCopyable
{
public:
    explicit DisplayDescriptorPool(const engine::Context & context, std::shared_ptr<const Scene> scene)
        : context{context}
        , offscreenRenderPass{std::make_shared<OffscreenRenderPass>(OffscreenRenderPass::make(context))}
        , scene{std::move(scene)}
        , offscreenGraphicsPipeline{scene->createGraphicsPipeline(*offscreenRenderPass->renderPass, Scene::PipelineKind::kScenePipeline)}
    {
        float maxSamplerAnisotropy = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits.maxSamplerAnisotropy;
        vk::SamplerCreateInfo samplerCreateInfo = {
            .flags = {},
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eRepeat,
            .addressModeV = vk::SamplerAddressMode::eRepeat,
            .addressModeW = vk::SamplerAddressMode::eRepeat,
            .mipLodBias = 0.0f,
            .anisotropyEnable = VK_FALSE,
            .maxAnisotropy = maxSamplerAnisotropy,
            .compareEnable = VK_FALSE,
            .compareOp = vk::CompareOp::eNever,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = vk::BorderColor::eFloatTransparentBlack,
            .unnormalizedCoordinates = VK_FALSE,
        };
        sampler = std::make_shared<vk::UniqueSampler>(context.getDevice().getDevice().createSamplerUnique(samplerCreateInfo, context.getAllocationCallbacks(), context.getDispatcher()));
    }

    [[nodiscard]] const GraphicsPipeline & getOffscreenGraphicsPipeline() const &
    {
        return *offscreenGraphicsPipeline;
    }

    [[nodiscard]] std::shared_ptr<const ResourcesAndDescriptors<DisplayResources>> getDisplayDescriptors(const Scene & scene, const vk::Extent2D & size) &
    {
        while (!displayDescriptorPool.empty()) {
            auto displayDescriptors = std::move(displayDescriptorPool.top());
            displayDescriptorPool.pop();
            const auto & framebuffer = displayDescriptors->resources.framebuffer;
            if ((framebuffer.associatedRenderPass == offscreenRenderPass) && (framebuffer.size == size)) {
                return displayDescriptors;
            } else {
                // mb reuse framebuffer?
            }
        }
        return std::make_shared<ResourcesAndDescriptors<DisplayResources>>(scene.makeDisplayDescriptors(context, sampler, size, *offscreenRenderPass));
    }

    void putDisplayDescriptors(std::shared_ptr<const ResourcesAndDescriptors<DisplayResources>> displayDescriptors) &
    {
        ASSERT_MSG(displayDescriptors.use_count() == 1, "Non-unique use in single-threaded context: {}", displayDescriptors.use_count());
        displayDescriptorPool.push(std::move(displayDescriptors));
    }

private:
    const engine::Context & context;
    std::shared_ptr<const OffscreenRenderPass> offscreenRenderPass;
    const std::shared_ptr<const Scene> scene;
    std::shared_ptr<const GraphicsPipeline> offscreenGraphicsPipeline;
    std::shared_ptr<const vk::UniqueSampler> sampler;
    ResourceStack<std::shared_ptr<const ResourcesAndDescriptors<DisplayResources>>> displayDescriptorPool;
};

class RenderGraphNode final : utils::OneTime<RenderGraphNode>
{
public:
    explicit RenderGraphNode(std::string_view name, const engine::Context & context, const engine::Queue & queue)
        : name{name}
        , context{context}
        , queue{queue}
        , commandBuffers{std::make_shared<engine::CommandBuffers>(queue.allocateCommandBuffers(name))}
    {}

    RenderGraphNode(RenderGraphNode && rhs) noexcept = default;

    ~RenderGraphNode()
    {
        if (!commandBuffers) {
            return;
        }

        auto commandBuffer = commandBuffers->getCommandBuffer();
        commandBuffer.end(context.getDispatcher());

        vk::SubmitInfo submitInfo;
        submitInfo.setWaitSemaphores(waitSemaphores);
        submitInfo.setWaitDstStageMask(waitDstStageMasks);
        submitInfo.setSignalSemaphores(signalSemaphores);
        submitInfo.setCommandBuffers(commandBuffer);
        queue.submit(submitInfo, completionFence);

        if (waitIdle) {
            if (completionFence) {
                auto result = context.getDevice().getDevice().waitForFences(completionFence, VK_TRUE, std::numeric_limits<uint64_t>::max(), context.getDispatcher());
                INVARIANT(result == vk::Result::eSuccess, "{}: {}", name, result);
            } else {
                queue.waitIdle();
            }
        }
    }

    void setCompletionFence(vk::Fence completionFence)
    {
        this->completionFence = completionFence;
    }

    void setWaitCompletion(bool waitIdle = true)
    {
        this->waitIdle = waitIdle;
    }

    void setWaitCompletion(vk::Fence completionFence)
    {
        setCompletionFence(completionFence);
        setWaitCompletion();
    }

    const std::shared_ptr<const engine::CommandBuffers> & getCommandBuffers() const &
    {
        return commandBuffers;
    }

private:
    std::string name;
    const engine::Context & context;
    const engine::Queue & queue;

    bool waitIdle = false;
    vk::Fence completionFence;
    std::shared_ptr<const engine::CommandBuffers> commandBuffers;
    std::vector<vk::Semaphore> waitSemaphores;
    std::vector<vk::PipelineStageFlags> waitDstStageMasks;
    std::vector<vk::Semaphore> signalSemaphores;

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
    const engine::Context & context;
    const uint32_t framesInFlight;

    const engine::Queue graphisQueue{"renderer", context, context.getPhysicalDevice().graphicsQueueCreateInfo};

    std::shared_ptr<const Scene> scene;
    utils::CheckedPtr<const std::vector<vk::PushConstantRange>> pushConstantRanges = nullptr;
    std::shared_ptr<const ResourcesAndDescriptors<SceneResources>> sceneResourcesAndDescriptors;
    std::shared_ptr<const GraphicsPipeline> outputPipeline;
    mutable std::optional<DisplayDescriptorPool> displayDescriptorPool;
    ResourceStack<std::shared_ptr<const ResourcesAndDescriptors<FrameResources>>> frameDescriptorsPool;
    std::shared_ptr<const ResourcesAndDescriptors<FrameResources>> frameResourcesAndDescriptors;

    // revocation lists should be the last members
    std::vector<std::vector<Resource>> deferredDeletionSlots{framesInFlight};

    Impl(const engine::Context & context, uint32_t framesInFlight);

    void setScene(std::shared_ptr<const Scene> scene);
    void bindGraphicsPipeline(vk::CommandBuffer commandBuffer, const GraphicsPipeline & scenePipeline, std::initializer_list<const Descriptors *> descriptors, const FrameSettings & frameSettings) const;
    void drawScene(vk::CommandBuffer commandBuffer, const FrameSettings & frameSettings) const;
    [[nodiscard]] std::shared_ptr<const ResourcesAndDescriptors<DisplayResources>> offscreenPass(const FrameSettings & frameSettings) const;
    void advance(uint32_t currentFrameSlot, const FrameSettings & frameSettings);
    [[nodiscard]] bool updateRenderPass(vk::RenderPass renderPass, const FrameSettings & frameSettings);
    void drawDisplay(vk::CommandBuffer commandBuffer, const FrameSettings & frameSettings) const;
    void render(vk::CommandBuffer commandBuffer, uint32_t currentFrameSlot, const FrameSettings & frameSettings) const;

    std::shared_ptr<const ResourcesAndDescriptors<FrameResources>> getFrameDescriptors();
    void putFrameDescriptors(std::shared_ptr<const ResourcesAndDescriptors<FrameResources>> && frameDescriptors);

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

Renderer::Renderer(const engine::Context & context, uint32_t framesInFlight)
    : impl_{context, framesInFlight}
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

bool Renderer::updateRenderPass(vk::RenderPass renderPass, const FrameSettings & frameSettings)
{
    return impl_->updateRenderPass(renderPass, frameSettings);
}

void Renderer::render(vk::CommandBuffer commandBuffer, uint32_t currentFrameSlot, const FrameSettings & frameSettings) const
{
    return impl_->render(commandBuffer, currentFrameSlot, frameSettings);
}

Renderer::Impl::Impl(const engine::Context & context, uint32_t framesInFlight)
    : context{context}
    , framesInFlight{framesInFlight}
{}

void Renderer::Impl::setScene(std::shared_ptr<const Scene> scene)
{
    ASSERT(this->scene != scene);

    auto unmuteMessageGuard = context.getInstance().unmuteDebugUtilsMessages(kUnmutedMessageIdNumbers);

    outputPipeline.reset();
    displayDescriptorPool.reset();
    frameResourcesAndDescriptors.reset();
    frameDescriptorsPool.clear();
    sceneResourcesAndDescriptors.reset();
    pushConstantRanges = nullptr;
    this->scene = std::move(scene);
}

void Renderer::Impl::bindGraphicsPipeline(vk::CommandBuffer commandBuffer, const GraphicsPipeline & scenePipeline, std::initializer_list<const Descriptors *> descriptors, const FrameSettings & frameSettings) const
{
    vk::Pipeline pipeline = scenePipeline.pipelines.getPipelines().at(0);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline, context.getDispatcher());

    constexpr uint32_t kFirstSet = 0;
    vk::PipelineLayout pipelineLayout = scenePipeline.pipelineLayout.getPipelineLayout();
    if (scene->isDescriptorBufferEnabled()) {
        std::vector<vk::DescriptorBufferBindingInfoEXT> descriptorBufferBindingInfos;
        descriptorBufferBindingInfos.reserve(std::size(descriptors));
        for (const Descriptors * d : descriptors) {
            descriptorBufferBindingInfos.push_back(d->getDescriptorBuffer().getDescriptorBufferBindingInfo());
        }
        commandBuffer.bindDescriptorBuffersEXT(descriptorBufferBindingInfos, context.getDispatcher());

        std::vector<uint32_t> bufferIndices(std::size(descriptorBufferBindingInfos));
        std::iota(std::begin(bufferIndices), std::end(bufferIndices), uint32_t{0});

        std::vector<vk::DeviceSize> offsets(std::size(descriptorBufferBindingInfos));
        std::fill(std::begin(offsets), std::end(offsets), vk::DeviceSize{0});

        commandBuffer.setDescriptorBufferOffsetsEXT(vk::PipelineBindPoint::eGraphics, pipelineLayout, kFirstSet, bufferIndices, offsets, context.getDispatcher());
    } else {
        std::vector<vk::DescriptorSet> descriptorSets;
        descriptorSets.reserve(std::size(descriptors));
        for (const Descriptors * d : descriptors) {
            descriptorSets.push_back(d->getDescriptorSet());
        }
        constexpr auto kDynamicOffsets = nullptr;
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, kFirstSet, descriptorSets, kDynamicOffsets, context.getDispatcher());
    }

    {
        PushConstants pushConstants = getPushConstants(frameSettings);
        ASSERT(pushConstantRanges);
        for (const auto & pushConstantRange : *pushConstantRanges) {
            const void * p = utils::safeCast<const std::byte *>(&pushConstants) + pushConstantRange.offset;
            commandBuffer.pushConstants(pipelineLayout, pushConstantRange.stageFlags, pushConstantRange.offset, pushConstantRange.size, p, context.getDispatcher());
        }
    }
}

void Renderer::Impl::drawScene(vk::CommandBuffer commandBuffer, const FrameSettings & frameSettings) const
{
    constexpr engine::LabelColor kMagentaColor = {1.0f, 0.0f, 1.0f, 1.0f};
    auto drawSceneLabel = engine::ScopedCommandBufferLabel::create(context.getDispatcher(), commandBuffer, "Draw scene"sv, kMagentaColor);

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

    ASSERT(sceneResourcesAndDescriptors);
    const auto & sceneResources = sceneResourcesAndDescriptors->resources;

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

std::shared_ptr<const ResourcesAndDescriptors<DisplayResources>> Renderer::Impl::offscreenPass(const FrameSettings & frameSettings) const
{
    RenderGraphNode renderGraphNode{"Offscreen scene draw"sv, context, graphisQueue};
    auto commandBuffers = renderGraphNode.getCommandBuffers();
    auto commandBuffer = commandBuffers->getCommandBuffer();

    constexpr engine::LabelColor kGreenColor = {0.0f, 1.0f, 0.0f, 1.0f};
    auto offscreenPassLabel = engine::ScopedCommandBufferLabel::create(context.getDispatcher(), commandBuffer, "Offscreen pass"sv, kGreenColor);

    vk::Extent2D frameSize = frameSettings.scissor.extent;  // TODO: ¡do invent something better

    auto displayResourcesAndDescriptors = displayDescriptorPool->getDisplayDescriptors(*scene, frameSettings.scissor.extent);  // TODO: use another size instead
    const auto & framebuffer = displayResourcesAndDescriptors->resources.framebuffer;
    if (!framebuffer.colorImage.barrier(commandBuffer, OffscreenRenderPass::kExternalColorStageMask, OffscreenRenderPass::kExternalColorAccessMask, OffscreenRenderPass::kExternalColorImageLayout)) {
        //
    }
    const auto & offscreenRenderPass = *framebuffer.associatedRenderPass;
    const auto depthImageLayout = offscreenRenderPass.depthImageLayout;
    if (!framebuffer.depthImage.barrier(commandBuffer, OffscreenRenderPass::kDepthStageMask, OffscreenRenderPass::kDepthAccessMask, depthImageLayout)) {
        //
    }

    vk::RenderPassBeginInfo renderPassBeginInfo = {
        .renderPass = *offscreenRenderPass.renderPass,
        .framebuffer = *framebuffer.framebuffer,
        .renderArea = {
            .offset = {
                .x = 0,
                .y = 0,
            },
            .extent = frameSize,
        },
    };
    std::initializer_list<vk::ClearValue> clearValues = {
        {
            .color = {
                .float32 = {{
                    0.0f,
                    0.0f,
                    0.0f,
                    1.0f,
                }},
            },
        },
        {
            .depthStencil = {
                .depth = 1.0f,
                .stencil = 0,
            },
        },
    };
    renderPassBeginInfo.setClearValues(clearValues);
    vk::SubpassBeginInfo subpassBeginInfo = {
        .contents = vk::SubpassContents::eInline,
    };
    commandBuffer.beginRenderPass2(renderPassBeginInfo, subpassBeginInfo, context.getDispatcher());
    auto descriptors = {
        &sceneResourcesAndDescriptors->descriptors,
        &frameResourcesAndDescriptors->descriptors,
    };
    bindGraphicsPipeline(commandBuffer, displayDescriptorPool.value().getOffscreenGraphicsPipeline(), descriptors, frameSettings);
    drawScene(commandBuffer, frameSettings);
    vk::SubpassEndInfo subpassEndInfo;
    commandBuffer.endRenderPass2(subpassEndInfo, context.getDispatcher());

    // framebuffer barrier is not needed after wait fence
    // extend life time of displayResourcesAndDescriptors and other resources
    return displayResourcesAndDescriptors;
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
    if (!sceneResourcesAndDescriptors) {
        sceneResourcesAndDescriptors = std::make_shared<const ResourcesAndDescriptors<SceneResources>>(scene->makeSceneDescriptors());
    }
    if (frameResourcesAndDescriptors) {
        uint32_t previousFrame = utils::modDown(currentFrameSlot, framesInFlight);
        Recycler recycler{&Impl::putFrameDescriptors, this, std::move(frameResourcesAndDescriptors)};  // 'this' captured, Renderer::Impl should be NonCopyable
        deferDeletion(previousFrame, std::move(recycler));
    }
    frameResourcesAndDescriptors = getFrameDescriptors();
    fillUniformBuffer(frameSettings, frameResourcesAndDescriptors->resources.uniformBuffer.map().at(0));

    if (frameSettings.useOffscreenTexture) {
        if (!displayDescriptorPool) {
            displayDescriptorPool.emplace(context, scene);
        }
    } else {
        displayDescriptorPool.reset();
    }
}

bool Renderer::Impl::updateRenderPass(vk::RenderPass renderPass, const FrameSettings & frameSettings)
{
    if (outputPipeline) {
        if (outputPipeline->pipelineLayout.getAssociatedRenderPass() == renderPass) {
            return false;
        }
        outputPipeline.reset();
    }
    auto outputPipelineKind = frameSettings.useOffscreenTexture ? Scene::PipelineKind::kDisplayPipeline : Scene::PipelineKind::kScenePipeline;  // what if pipelineKind is changed?
    outputPipeline = scene->createGraphicsPipeline(renderPass, outputPipelineKind);
    return true;
}

void Renderer::Impl::drawDisplay(vk::CommandBuffer commandBuffer, const FrameSettings & frameSettings) const
{
    // bind sampler image
    (void)frameSettings;
    commandBuffer.draw(4, 1, 0, 0, context.getDispatcher());
}

void Renderer::Impl::render(vk::CommandBuffer commandBuffer, uint32_t currentFrameSlot, const FrameSettings & frameSettings) const
{
    ASSERT(currentFrameSlot < framesInFlight);
    if (!scene) {
        return;
    }
    auto unmuteMessageGuard = context.getInstance().unmuteDebugUtilsMessages(kUnmutedMessageIdNumbers);
    if (frameSettings.useOffscreenTexture) {
        auto displayResourcesAndDescriptors = offscreenPass(frameSettings);
        auto descriptors = {
            &displayResourcesAndDescriptors->descriptors,
        };
        bindGraphicsPipeline(commandBuffer, *outputPipeline, descriptors, frameSettings);
        drawDisplay(commandBuffer, frameSettings);
        // sink sampled image at currentFrameSlot
    } else {
        auto descriptors = {
            &sceneResourcesAndDescriptors->descriptors,
            &frameResourcesAndDescriptors->descriptors,
        };
        bindGraphicsPipeline(commandBuffer, *outputPipeline, descriptors, frameSettings);
        drawScene(commandBuffer, frameSettings);
    }
}

auto Renderer::Impl::getFrameDescriptors() -> std::shared_ptr<const ResourcesAndDescriptors<FrameResources>>
{
    while (!std::empty(frameDescriptorsPool)) {
        auto frameDescriptors = std::move(frameDescriptorsPool.top());
        frameDescriptorsPool.pop();
        if (true) {
            return frameDescriptors;
        }
    }
    return std::make_shared<ResourcesAndDescriptors<FrameResources>>(scene->makeFrameDescriptors());
}

void Renderer::Impl::putFrameDescriptors(std::shared_ptr<const ResourcesAndDescriptors<FrameResources>> && frameDescriptors)
{
    ASSERT_MSG(frameDescriptors.use_count() == 1, "Non-unique use in single-threaded context: {}", frameDescriptors.use_count());
    frameDescriptorsPool.push(std::move(frameDescriptors));
}

}  // namespace viewer
