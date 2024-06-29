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
#include <viewer/descriptor_set.hpp>
#include <viewer/engine.hpp>
#include <viewer/pipelines.hpp>
#include <viewer/renderer.hpp>
#include <viewer/scenes.hpp>

#include <fmt/format.h>
#include <fmt/std.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_format_traits.hpp>

#include <queue>

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <numeric>
#include <set>
#include <stack>
#include <string_view>
#include <tuple>
#include <vector>

#include <cstddef>
#include <cstdint>

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

using Fence = std::shared_ptr<vk::UniqueFence>;

[[nodiscard]] inline Fence makeFence(const engine::Context & context, vk::FenceCreateFlags flags = {})
{
    auto device = context.getDevice().getDevice();
    vk::FenceCreateInfo fenceCreateInfo = {
        .flags = flags,
    };
    return std::make_shared<vk::UniqueFence>(device.createFenceUnique(fenceCreateInfo, context.getAllocationCallbacks(), context.getDispatcher()));
}

inline void waitFence(const engine::Context & context, const Fence & fence)
{
    ASSERT(fence);
    ASSERT(*fence);
    auto device = context.getDevice().getDevice();
    auto result = device.waitForFences(**fence, VK_TRUE, std::numeric_limits<uint64_t>::max(), context.getDispatcher());
    INVARIANT(result == vk::Result::eSuccess, "Display fence: {}", result);
    device.resetFences(**fence, context.getDispatcher());
}

inline void checkFenceUnique(const Fence & fence)
{
    ASSERT_MSG(fence.use_count() == 1, "Non-unique use in single-threaded context: {}", fence.use_count());
}

class FencePool final : utils::NonCopyable
{
public:
    explicit FencePool(const engine::Context & context)
        : context{context}
    {}

    [[nodiscard]] Fence get() &
    {
        if (!fencePool.empty()) {
            auto fence = std::move(fencePool.top());
            fencePool.pop();
            return fence;
        }
        return makeFence(context);
    }

    void waitAndPut(Fence && fence)
    {
        checkFenceUnique(fence);
        waitFence(context, fence);
        fencePool.push(std::move(fence));
    }

private:
    const engine::Context & context;
    ResourceStack<Fence> fencePool;
};

#pragma pack(push, 1)
struct UniformBuffer
{
    glm::mat2 transform2D{1.0f};
    float alpha = 0.0f;
    float zNear = 1E-2f;
    float zFar = 1E4;
    vk::Bool32 useOffscreenTexture = VK_FALSE;
    glm::vec3 position{0.0f};
};
#pragma pack(pop)
static_assert(std::is_standard_layout_v<UniformBuffer>);

#pragma pack(push, 1)
struct ScenePushConstants
{
    glm::mat4 mvp{1.0f};
};
#pragma pack(pop)
static_assert(std::is_standard_layout_v<ScenePushConstants>);

#pragma pack(push, 1)
struct DisplayPushConstants
{
    float x = 1E-5f;
};
#pragma pack(pop)
static_assert(std::is_standard_layout_v<DisplayPushConstants>);

struct UniformBufferResource final : utils::OneTime<UniformBufferResource>
{
    engine::Buffer<UniformBuffer> uniformBuffer;

    [[nodiscard]] static std::string getBindingName()
    {
        return "uniformBuffer"s;
    }

    [[nodiscard]] DescriptorInfo getDescriptorInfo(bool descriptorBufferEnabled) const
    {
        const auto getDescriptorData = [this, descriptorBufferEnabled]() -> DescriptorData
        {
            if (descriptorBufferEnabled) {
                return DescriptorBufferData{uniformBuffer.getDescriptorAddressInfo()};
            } else {
                return DescriptorSetData{uniformBuffer.getDescriptorBufferInfo()};
            }
        };
        return {getBindingName(), vk::DescriptorType::eUniformBuffer, getDescriptorData()};
    }

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

struct FrameResourcesAndDescriptors
{
    UniformBufferResource resources;
    DescriptorSet directDescriptors;
    std::optional<DescriptorSet> displayDescriptors;

    FrameResourcesAndDescriptors(UniformBufferResource && resources, DescriptorSet && sceneDescriptors, std::optional<DescriptorSet> && displayDescriptors)
        : resources{std::move(resources)}
        , directDescriptors{std::move(sceneDescriptors)}
        , displayDescriptors{std::move(displayDescriptors)}
    {}
};

struct SceneResourcesAndDescriptors
{
    SceneResources resources;
    DescriptorSet descriptors;

    SceneResourcesAndDescriptors(SceneResources && resources, DescriptorSet && descriptors)
        : resources{std::move(resources)}
        , descriptors{std::move(descriptors)}
    {}
};

struct DisplayResourcesAndDescriptors
{
    DisplayResources resources;
    DescriptorSet descriptors;

    DisplayResourcesAndDescriptors(DisplayResources && resources, DescriptorSet && descriptors)
        : resources{std::move(resources)}
        , descriptors{std::move(descriptors)}
    {}
};

class DisplayResourcesAndDescriptorsPool final
    : utils::NonCopyable
    , public std::enable_shared_from_this<DisplayResourcesAndDescriptorsPool>
{
    struct Private
    {
        explicit Private() = default;
    };

public:
    DisplayResourcesAndDescriptorsPool(Private, const engine::Context & context, const Engine & engine)
        : context{context}
        , engine{engine}
        , displayRenderPass{OffscreenRenderPass::make(context)}
        , displayGraphicsPipeline{makeGraphicsPipeline()}
        , sampler{makeSampler()}
    {}

    [[nodiscard]] static std::shared_ptr<DisplayResourcesAndDescriptorsPool> make(const engine::Context & context, const Engine & engine)
    {
        return std::make_shared<DisplayResourcesAndDescriptorsPool>(Private{}, context, engine);
    }

    [[nodiscard]] const OffscreenRenderPass & getOffscreenRenderPass() const &
    {
        return displayRenderPass;
    }

    [[nodiscard]] const GraphicsPipeline & getGraphicsPipeline() const &
    {
        return displayGraphicsPipeline;
    }

    [[nodiscard]] std::shared_ptr<DisplayResourcesAndDescriptors> get(const vk::Extent2D & framebufferSize, std::shared_ptr<const engine::ShaderStages> shaderStages) &
    {
        std::shared_ptr<DisplayResourcesAndDescriptors> resourcesAndDescriptors;
        while (!pool.empty()) {
            resourcesAndDescriptors = std::move(pool.top());
            pool.pop();
            const auto & framebuffer = resourcesAndDescriptors->resources.framebuffer;
            constexpr auto isIncludes = [](const vk::Extent2D & lhs, const vk::Extent2D & rhs) -> bool
            {
                return lhs.width <= rhs.width && lhs.height <= rhs.height;
            };
            if (!isIncludes(framebufferSize, framebuffer.size)) {
                break;
            }
            constexpr auto tooLess = [](const vk::Extent2D & lhs, const vk::Extent2D & rhs) -> bool
            {
                return (lhs.width <= rhs.width / 2) || (lhs.height <= rhs.height / 2);
            };
            if (tooLess(framebufferSize, framebuffer.size)) {
                break;
            }
            return resourcesAndDescriptors;
        }
        if (resourcesAndDescriptors) {
            DisplayResources resources{context, framebufferSize, displayRenderPass, std::move(resourcesAndDescriptors->resources.sampler)};
            auto descriptors = std::move(resourcesAndDescriptors->descriptors);
            auto descriptorInfos = {resources.getDescriptorInfo(engine.getSettings().descriptorBufferEnabled)};
            descriptors.fill(descriptorInfos);
            return std::make_shared<DisplayResourcesAndDescriptors>(std::move(resources), std::move(descriptors));
        } else {
            DisplayResources resources{context, framebufferSize, displayRenderPass, sampler};
            auto descriptors = engine.makeDescriptors(std::move(shaderStages), resources);
            return std::make_shared<DisplayResourcesAndDescriptors>(std::move(resources), std::move(descriptors));
        }
    }

    void put(std::shared_ptr<DisplayResourcesAndDescriptors> resourcesAndDescriptors) &
    {
        ASSERT_MSG(resourcesAndDescriptors.use_count() == 1, "Non-unique use in single-threaded context: {}", resourcesAndDescriptors.use_count());
        pool.push(std::move(resourcesAndDescriptors));
    }

private:
    const engine::Context & context;
    const Engine & engine;

    OffscreenRenderPass displayRenderPass;
    GraphicsPipeline displayGraphicsPipeline;
    std::shared_ptr<const vk::UniqueSampler> sampler;
    ResourceStack<std::shared_ptr<DisplayResourcesAndDescriptors>> pool;

    [[nodiscard]] std::shared_ptr<const vk::UniqueSampler> makeSampler() const
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
        return std::make_shared<vk::UniqueSampler>(context.getDevice().getDevice().createSamplerUnique(samplerCreateInfo, context.getAllocationCallbacks(), context.getDispatcher()));
    }

    [[nodiscard]] GraphicsPipeline makeGraphicsPipeline() const
    {
        GraphicsPipeline graphicsPipeline{engine.getPipelines().getSceneShaders()};
        graphicsPipeline.initPipeline("offscreen scene"sv, context, engine.getPipelines().getPipelineCache(), engine.getSettings().descriptorBufferEnabled, displayRenderPass);
        return graphicsPipeline;
    }
};

class ScopedCommandBuffer final : utils::OneTime<ScopedCommandBuffer>
{
public:
    explicit ScopedCommandBuffer(std::string_view name, const engine::Context & context, const engine::Queue & queue)
        : name{name}
        , context{context}
        , queue{queue}
        , commandBuffers{std::make_shared<engine::CommandBuffers>(queue.allocateCommandBuffers(name))}
    {
        auto commandBuffer = commandBuffers->getCommandBuffer();
        vk::CommandBufferBeginInfo commandBufferBeginInfo = {
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        };
        commandBuffer.begin(commandBufferBeginInfo, context.getDispatcher());
    }

    ScopedCommandBuffer(ScopedCommandBuffer && rhs) noexcept = default;

    ~ScopedCommandBuffer()
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

    [[nodiscard]] std::shared_ptr<const engine::CommandBuffers> getCommandBuffers() const
    {
        return commandBuffers;
    }

    [[nodiscard]] const vk::CommandBuffer & getCommandBuffer() const &
    {
        return commandBuffers->getCommandBuffer();
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

constexpr std::initializer_list<uint32_t> kUnmutedMessageIdNumbers = {
    0x5C0EC5D6,
    0xE4D96472,
    0x6d0c146d,
    0xb302c33b,
    0x2f637ff,
    // 0xa96ad8,  // TODO: implement vkBindBufferMemory wrapper in VMA?
    0xc714b932,
};

void fillUniformBuffer(const FrameSettings & frameSettings, UniformBuffer & uniformBuffer)
{
    uniformBuffer = {
        .transform2D = frameSettings.transform2D,
        .alpha = frameSettings.alpha,
        .zNear = frameSettings.zNear,
        .zFar = frameSettings.zFar,
        .useOffscreenTexture = frameSettings.useOffscreenTexture ? VK_TRUE : VK_FALSE,
        .position = frameSettings.position,
    };
}

[[nodiscard]] ScenePushConstants getScenePushConstants(const FrameSettings & frameSettings)
{
    auto view = glm::translate(glm::toMat4(glm::conjugate(frameSettings.orientation)), -frameSettings.position);
    auto projection = glm::perspectiveFovLH(frameSettings.fov, frameSettings.width, frameSettings.height, frameSettings.zNear, frameSettings.zFar);
    auto mvp = projection * view;
    if (!frameSettings.useOffscreenTexture) {
        mvp = glm::mat4{frameSettings.transform2D} * mvp;
    }
    return {
        .mvp = mvp,
    };
}

[[nodiscard]] DisplayPushConstants getDisplayPushConstants([[maybe_unused]] const FrameSettings & frameSettings)
{
    return {
        .x = 0.0f,
    };
}

}  // namespace

struct Renderer::Impl : utils::NonCopyable
{
    const engine::Context & context;
    const Engine & engine;
    const uint32_t framesInFlight;

    const engine::Queue graphicsQueue{"renderer"sv, context, context.getPhysicalDevice().graphicsQueueCreateInfo};

    FrameSettings frameSettings;
    std::shared_ptr<const Scene> scene;

    FencePool fencePool{context};

    std::shared_ptr<GraphicsPipeline> directGraphicsPipeline = std::make_shared<GraphicsPipeline>(engine.getPipelines().getSceneShaders());
    std::shared_ptr<GraphicsPipeline> offscreenGraphicsPipeline = std::make_shared<GraphicsPipeline>(engine.getPipelines().getDisplayShaders());

    std::shared_ptr<SceneResourcesAndDescriptors> sceneResourcesAndDescriptors;
    ResourceStack<std::shared_ptr<FrameResourcesAndDescriptors>> frameResourcesAndDescriptorsPool;
    std::shared_ptr<FrameResourcesAndDescriptors> frameResourcesAndDescriptors;

    std::shared_ptr<DisplayResourcesAndDescriptorsPool> displayPool;

    Fence displayFence;
    std::shared_ptr<const engine::CommandBuffers> displayCommandBuffers;
    std::shared_ptr<DisplayResourcesAndDescriptors> displayResourcesAndDescriptors;

    // revocation lists should be the last members
    std::vector<std::vector<Resource>> deferredDeletionSlots{framesInFlight};

    Impl(const engine::Context & context, const Engine & engine, uint32_t framesInFlight);

    void setFrameSettings(const FrameSettings & frameSettings);
    void setScene(std::shared_ptr<const Scene> scene);

    void bindGraphicsPipeline(vk::CommandBuffer commandBuffer, const GraphicsPipeline & pipeline, std::initializer_list<std::reference_wrapper<const DescriptorSet>> descriptors, const std::byte * pushConstants) const;
    void drawScene(vk::CommandBuffer commandBuffer, const GraphicsPipeline & pipeline) const;
    void offscreenPass(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const Framebuffer & framebuffer);
    void drawDisplay(vk::CommandBuffer commandBuffer, const GraphicsPipeline & pipeline) const;

    void advance(uint32_t currentFrameSlot);

    [[nodiscard]] bool updateRenderPass(vk::RenderPass renderPass);

    void render(vk::CommandBuffer commandBuffer, uint32_t currentFrameSlot);

    [[nodiscard]] std::shared_ptr<FrameResourcesAndDescriptors> getFrameDescriptors();
    void putFrameDescriptors(std::shared_ptr<FrameResourcesAndDescriptors> && frameDescriptors);

    template<typename... Resources>
    void deferDeletion(uint32_t frameSlot, Resources &&... resources)
    {
        auto & slotResources = deferredDeletionSlots.at(frameSlot);
        (slotResources.emplace_back(std::forward<Resources>(resources)), ...);
    }

    void deleteDeferred(uint32_t currentFrameSlot)
    {
        deferredDeletionSlots.at(currentFrameSlot).clear();
    }
};

Renderer::Renderer(const engine::Context & context, const Engine & engine, uint32_t framesInFlight)
    : impl_{context, engine, framesInFlight}
{}

Renderer::~Renderer() = default;

void Renderer::setFrameSettings(const FrameSettings & frameSettings)
{
    return impl_->setFrameSettings(frameSettings);
}

void Renderer::setScene(std::shared_ptr<const Scene> scene)
{
    return impl_->setScene(std::move(scene));
}

void Renderer::advance(uint32_t currentFrameSlot)
{
    return impl_->advance(currentFrameSlot);
}

bool Renderer::updateRenderPass(vk::RenderPass renderPass)
{
    return impl_->updateRenderPass(renderPass);
}

void Renderer::render(vk::CommandBuffer commandBuffer, uint32_t currentFrameSlot)
{
    return impl_->render(commandBuffer, currentFrameSlot);
}

Renderer::Impl::Impl(const engine::Context & context, const Engine & engine, uint32_t framesInFlight)
    : context{context}
    , engine{engine}
    , framesInFlight{framesInFlight}
{
    uint32_t maxPushConstantsSize = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits.maxPushConstantsSize;
    INVARIANT(sizeof(ScenePushConstants) <= maxPushConstantsSize, "{} ^ {}", sizeof(ScenePushConstants), maxPushConstantsSize);
}

void Renderer::Impl::setFrameSettings(const FrameSettings & frameSettings)
{
    this->frameSettings = frameSettings;
}

void Renderer::Impl::setScene(std::shared_ptr<const Scene> scene)
{
    ASSERT(this->scene != scene);

    auto unmuteMessageGuard = context.getInstance().unmuteDebugUtilsMessages(kUnmutedMessageIdNumbers);

    // displayGraphicsPipeline.reset();
    // const GraphicsPipeline & graphicsPipeline.reset();
    displayPool.reset();
    // frameResourcesAndDescriptors.reset();
    // frameResourcesAndDescriptorsPool.clear();
    sceneResourcesAndDescriptors.reset();
    this->scene = std::move(scene);
}

void Renderer::Impl::bindGraphicsPipeline(vk::CommandBuffer commandBuffer, const GraphicsPipeline & pipeline, std::initializer_list<std::reference_wrapper<const DescriptorSet>> descriptors, const std::byte * pushConstants) const
{
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline.value(), context.getDispatcher());

    constexpr uint32_t kFirstSet = 0;
    vk::PipelineLayout pipelineLayout = pipeline.shaders->getGraphicsPipelineLayout();
    if (engine.getSettings().descriptorBufferEnabled) {
        std::vector<vk::DescriptorBufferBindingInfoEXT> descriptorBufferBindingInfos;
        descriptorBufferBindingInfos.reserve(std::size(descriptors));
        for (const DescriptorSet & d : descriptors) {
            descriptorBufferBindingInfos.push_back(d.getDescriptorBuffer().getDescriptorBufferBindingInfo());
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
        for (const DescriptorSet & d : descriptors) {
            descriptorSets.push_back(d.getDescriptorSet());
        }
        constexpr auto kDynamicOffsets = nullptr;
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, kFirstSet, descriptorSets, kDynamicOffsets, context.getDispatcher());
    }

    for (const auto & pushConstantRange : pipeline.shaders->getShaderStages().pushConstantRanges) {
        commandBuffer.pushConstants(pipelineLayout, pushConstantRange.stageFlags, pushConstantRange.offset, pushConstantRange.size, std::next(pushConstants, pushConstantRange.offset), context.getDispatcher());
    }
}

void Renderer::Impl::drawScene(vk::CommandBuffer commandBuffer, const GraphicsPipeline & pipeline) const
{
    {
        std::initializer_list<std::reference_wrapper<const DescriptorSet>> descriptors = {
            std::cref(frameResourcesAndDescriptors->directDescriptors),
            std::cref(sceneResourcesAndDescriptors->descriptors),
        };
        ScenePushConstants scenePushConstants = getScenePushConstants(frameSettings);
        bindGraphicsPipeline(commandBuffer, pipeline, descriptors, utils::autoCast(&scenePushConstants));
    }

    constexpr engine::LabelColor kMagentaColor = {1.0f, 0.0f, 1.0f, 1.0f};
    auto drawSceneLabel = engine::ScopedCommandBufferLabel::create(context.getDispatcher(), commandBuffer, "Draw scene"sv, kMagentaColor);

    {
        vk::Viewport viewport;
        vk::Rect2D scissor;
        if (frameSettings.useOffscreenTexture) {
            vk::Extent2D extent = displayResourcesAndDescriptors->resources.framebuffer.size;
            viewport = {
                .x = 0.0f,
                .y = 0.0f,
                .width = utils::autoCast(extent.width),
                .height = utils::autoCast(extent.height),
                .minDepth = engine::kMinDepth,
                .maxDepth = 1.0f,
            };
            scissor = {
                .offset = {
                    .x = 0,
                    .y = 0,
                },
                .extent = extent,
            };
        } else {
            viewport = frameSettings.viewport;
            scissor = frameSettings.scissor;
        }
        if (viewport.width == 0.0f) {
            viewport.width = 1.0f;
        }
        if (viewport.height == 0.0f) {
            viewport.height = 1.0f;
        }

        constexpr uint32_t kFirstViewport = 0;
        commandBuffer.setViewport(kFirstViewport, viewport, context.getDispatcher());

        constexpr uint32_t kFirstScissor = 0;
        commandBuffer.setScissor(kFirstScissor, scissor, context.getDispatcher());
    }

    ASSERT(sceneResourcesAndDescriptors);
    const auto & sceneResources = sceneResourcesAndDescriptors->resources;

    {
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
    }

    vk::Buffer indexBuffer;
    vk::DeviceSize indexBufferSize = 0;
    if (sceneResources.indexBuffer) {
        indexBuffer = sceneResources.indexBuffer.value();
        indexBufferSize = sceneResources.indexBuffer.value().getSize();
    } else {
        const auto & features2Chain = context.getPhysicalDevice().features2Chain;
        ASSERT(features2Chain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>().nullDescriptor == VK_TRUE);
        ASSERT(features2Chain.get<vk::PhysicalDeviceMaintenance6FeaturesKHR>().maintenance6 == VK_TRUE);
        // TODO: draw
    }
    constexpr vk::DeviceSize kIndexBufferDeviceOffset = 0;
    if (engine.getSettings().multiDrawIndirectEnabled) {
        ASSERT(std::empty(sceneResources.indexTypes));
        auto indexType = sceneResources.maxIndexType;
        commandBuffer.bindIndexBuffer2KHR(indexBuffer, kIndexBufferDeviceOffset, indexBufferSize, indexType, context.getDispatcher());
        constexpr vk::DeviceSize kInstanceBufferOffset = 0;
        constexpr uint32_t kStride = sizeof(vk::DrawIndexedIndirectCommand);
        uint32_t drawCount = sceneResources.drawCount;
        const auto & physicalDeviceLimits = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits;
        INVARIANT(drawCount <= physicalDeviceLimits.maxDrawIndirectCount, "{} ^ {}", drawCount, physicalDeviceLimits.maxDrawIndirectCount);
        if (engine.getSettings().drawIndirectCountEnabled) {
            constexpr vk::DeviceSize kDrawCountBufferOffset = 0;
            uint32_t maxDrawCount = drawCount;
            commandBuffer.drawIndexedIndirectCount(sceneResources.instanceBuffer.value(), kInstanceBufferOffset, sceneResources.drawCountBuffer.value(), kDrawCountBufferOffset, maxDrawCount, kStride, context.getDispatcher());
        } else {
            commandBuffer.drawIndexedIndirect(sceneResources.instanceBuffer.value(), kInstanceBufferOffset, drawCount, kStride, context.getDispatcher());
        }
    } else {
        ASSERT(!std::empty(sceneResources.instances));
        ASSERT(std::size(sceneResources.indexTypes) == std::size(sceneResources.instances));
        auto indexType = std::cbegin(sceneResources.indexTypes);
        for (const auto & [indexCount, instanceCount, firstIndex, vertexOffset, firstInstance] : sceneResources.instances) {
            ASSERT(indexType != std::cend(sceneResources.indexTypes));
            commandBuffer.bindIndexBuffer2KHR(indexBuffer, kIndexBufferDeviceOffset, indexBufferSize, *indexType++, context.getDispatcher());
            commandBuffer.drawIndexed(indexCount, instanceCount, firstIndex, vertexOffset, firstInstance, context.getDispatcher());
            // SPDLOG_TRACE("{{.indexCount = {}, .instanceCount = {}, .firstIndex = {}, .vertexOffset = {}, .firstInstance = {})}}", indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
        }
    }
}

void Renderer::Impl::offscreenPass(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const Framebuffer & framebuffer)
{
    constexpr engine::LabelColor kGreenColor = {0.0f, 1.0f, 0.0f, 1.0f};
    auto offscreenPassLabel = engine::ScopedCommandBufferLabel::create(context.getDispatcher(), commandBuffer, "Offscreen pass"sv, kGreenColor);

    vk::RenderPassBeginInfo renderPassBeginInfo = {
        .renderPass = renderPass,
        .framebuffer = *framebuffer.framebuffer,
        .renderArea = {
            .offset = {
                .x = 0,
                .y = 0,
            },
            .extent = framebuffer.size,
        },
    };
    std::initializer_list<vk::ClearValue> clearValues = {
        {
            .color = {
                .float32 = {{
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
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
    drawScene(commandBuffer, displayPool->getGraphicsPipeline());
    vk::SubpassEndInfo subpassEndInfo;
    commandBuffer.endRenderPass2(subpassEndInfo, context.getDispatcher());
}

void Renderer::Impl::drawDisplay(vk::CommandBuffer commandBuffer, const GraphicsPipeline & pipeline) const
{
    {
        ASSERT(frameResourcesAndDescriptors->displayDescriptors);
        std::initializer_list<std::reference_wrapper<const DescriptorSet>> descriptors = {
            std::cref(frameResourcesAndDescriptors->displayDescriptors.value()),
            std::cref(displayResourcesAndDescriptors->descriptors),
        };
        DisplayPushConstants displayPushConstants = getDisplayPushConstants(frameSettings);
        bindGraphicsPipeline(commandBuffer, pipeline, descriptors, utils::autoCast(&displayPushConstants));
    }

    {
        constexpr uint32_t kFirstViewport = 0;
        commandBuffer.setViewport(kFirstViewport, frameSettings.viewport, context.getDispatcher());

        constexpr uint32_t kFirstScissor = 0;
        commandBuffer.setScissor(kFirstScissor, frameSettings.scissor, context.getDispatcher());
    }

    commandBuffer.draw(4, 1, 0, 0, context.getDispatcher());
}

void Renderer::Impl::advance(uint32_t currentFrameSlot)
{
    ASSERT_MSG(currentFrameSlot < framesInFlight, "{} ^ {}", currentFrameSlot, framesInFlight);
    if (!scene) {
        return;
    }

    auto unmuteMessageGuard = context.getInstance().unmuteDebugUtilsMessages(kUnmutedMessageIdNumbers);

    deleteDeferred(currentFrameSlot);

    uint32_t previousFrameSlot = utils::modDown(currentFrameSlot, framesInFlight);

    if (displayResourcesAndDescriptors) {
        ASSERT(displayPool);
        Recycler recycler = [this, displayPool = displayPool, displayResourcesAndDescriptors = std::move(displayResourcesAndDescriptors), displayCommandBuffers = std::move(displayCommandBuffers), displayFence = std::move(displayFence)]() mutable
        {
            if (displayFence) {
                fencePool.waitAndPut(std::move(displayFence));
            }
            displayPool->put(std::move(displayResourcesAndDescriptors));
            displayCommandBuffers.reset();
        };
        deferDeletion(previousFrameSlot, std::move(recycler));
    } else {
        INVARIANT(!displayFence, "");
    }
    if (frameSettings.useOffscreenTexture) {
        if (!displayPool) {
            displayPool = DisplayResourcesAndDescriptorsPool::make(context, engine);
        }
    }
    {
        if (frameResourcesAndDescriptors) {
            Recycler recycler{&Impl::putFrameDescriptors, this, std::move(frameResourcesAndDescriptors)};
            deferDeletion(previousFrameSlot, std::move(recycler));
        }
        frameResourcesAndDescriptors = getFrameDescriptors();
        fillUniformBuffer(frameSettings, frameResourcesAndDescriptors->resources.uniformBuffer.map().at(0));
    }
    if (!sceneResourcesAndDescriptors) {
        auto & graphicsPipeline = frameSettings.useOffscreenTexture ? displayPool->getGraphicsPipeline() : *directGraphicsPipeline;
        auto resources = engine.makeResources(*scene);
        auto descriptors = engine.makeDescriptors(graphicsPipeline.shaders->getShaderStagesPtr(), resources);
        sceneResourcesAndDescriptors = std::make_shared<SceneResourcesAndDescriptors>(std::move(resources), std::move(descriptors));
    }
    if (frameSettings.useOffscreenTexture) {
        float width = std::ceil(frameSettings.width);
        float height = std::ceil(frameSettings.height);
        vk::Extent2D framebufferSize = {
            .width = utils::autoCast(width),
            .height = utils::autoCast(height),
        };

        ASSERT(offscreenGraphicsPipeline->shaders);
        displayResourcesAndDescriptors = displayPool->get(framebufferSize, offscreenGraphicsPipeline->shaders->getShaderStagesPtr());
        {
            ScopedCommandBuffer displayCommandBuffer{"Offscreen scene draw"sv, context, graphicsQueue};
            const OffscreenRenderPass & offscreenRenderPass = displayPool->getOffscreenRenderPass();
            offscreenPass(displayCommandBuffer.getCommandBuffer(), offscreenRenderPass, displayResourcesAndDescriptors->resources.framebuffer);
            INVARIANT(!displayFence, "");
            displayFence = fencePool.get();
            displayCommandBuffer.setCompletionFence(**displayFence);
            displayCommandBuffers = displayCommandBuffer.getCommandBuffers();
        }
    } else {
        displayPool.reset();
    }
}

bool Renderer::Impl::updateRenderPass(vk::RenderPass renderPass)
{
    auto & graphicsPipeline = frameSettings.useOffscreenTexture ? *offscreenGraphicsPipeline : *directGraphicsPipeline;
    if (graphicsPipeline.pipeline) {
        if (graphicsPipeline.pipeline.value().getRenderPass() == renderPass) {
            return false;
        }
        graphicsPipeline.pipeline.reset();
    }
    std::string_view name;
    if (frameSettings.useOffscreenTexture) {
        name = "offscreen display"sv;
    } else {
        name = "direct scene"sv;
    }
    graphicsPipeline.initPipeline(name, context, engine.getPipelines().getPipelineCache(), engine.getSettings().descriptorBufferEnabled, renderPass);
    return true;
}

void Renderer::Impl::render(vk::CommandBuffer commandBuffer, uint32_t currentFrameSlot)
{
    ASSERT(currentFrameSlot < framesInFlight);
    if (!scene) {
        return;
    }
    auto unmuteMessageGuard = context.getInstance().unmuteDebugUtilsMessages(kUnmutedMessageIdNumbers);

    if (frameSettings.useOffscreenTexture) {
        if (displayFence) {
            fencePool.waitAndPut(std::move(displayFence));
        }
        ASSERT(offscreenGraphicsPipeline->pipeline);
        drawDisplay(commandBuffer, *offscreenGraphicsPipeline);
    } else {
        ASSERT(directGraphicsPipeline->pipeline);
        drawScene(commandBuffer, *directGraphicsPipeline);
    }
}

auto Renderer::Impl::getFrameDescriptors() -> std::shared_ptr<FrameResourcesAndDescriptors>
{
    std::shared_ptr<FrameResourcesAndDescriptors> resourcesAndDescriptors;
    while (!std::empty(frameResourcesAndDescriptorsPool)) {
        resourcesAndDescriptors = std::move(frameResourcesAndDescriptorsPool.top());
        frameResourcesAndDescriptorsPool.pop();
        if (frameSettings.useOffscreenTexture) {
            if (!resourcesAndDescriptors->displayDescriptors) {
                const auto & resources = resourcesAndDescriptors->resources;
                auto displayDescriptors = engine.makeDescriptors("scene"sv, offscreenGraphicsPipeline->shaders->getShaderStagesPtr(), resources);
                resourcesAndDescriptors->displayDescriptors.emplace(std::move(displayDescriptors));
            }
        }
        return resourcesAndDescriptors;
    }
    UniformBufferResource resources = {
        .uniformBuffer = engine.createUniformBuffer(sizeof(UniformBuffer)),
    };
    std::shared_ptr<const engine::ShaderStages> sceneShaderStages;
    if (frameSettings.useOffscreenTexture) {
        sceneShaderStages = displayPool->getGraphicsPipeline().shaders->getShaderStagesPtr();
    } else {
        sceneShaderStages = directGraphicsPipeline->shaders->getShaderStagesPtr();
    }
    auto directDescriptors = engine.makeDescriptors("scene"sv, std::move(sceneShaderStages), resources);
    std::optional<DescriptorSet> displayDescriptors;
    if (frameSettings.useOffscreenTexture) {
        displayDescriptors.emplace(engine.makeDescriptors("scene"sv, offscreenGraphicsPipeline->shaders->getShaderStagesPtr(), resources));
    }
    return std::make_shared<FrameResourcesAndDescriptors>(std::move(resources), std::move(directDescriptors), std::move(displayDescriptors));
}

void Renderer::Impl::putFrameDescriptors(std::shared_ptr<FrameResourcesAndDescriptors> && frameDescriptors)
{
    ASSERT_MSG(frameDescriptors.use_count() == 1, "Non-unique use in single-threaded context: {}", frameDescriptors.use_count());
    frameResourcesAndDescriptorsPool.push(std::move(frameDescriptors));
}

}  // namespace viewer
