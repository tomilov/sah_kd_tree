#include <codegen/vulkan_utils.hpp>
#include <common/version.hpp>
#include <engine/command_buffer.hpp>
#include <engine/command_pool.hpp>
#include <engine/context.hpp>
#include <engine/debug_utils.hpp>
#include <engine/device.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/instance.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <engine/pipeline_layout.hpp>
#include <engine/queue.hpp>
#include <engine/vma.hpp>
#include <format/vulkan.hpp>
#include <scene_data/scene_data.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/checked_ptr.hpp>
#include <utils/math.hpp>
#include <utils/noncopyable.hpp>
#include <viewer/descriptors.hpp>
#include <viewer/engine.hpp>
#include <viewer/pipelines.hpp>
#include <viewer/renderer.hpp>
#include <viewer/scenes.hpp>
#include <viewer/tree.hpp>

#include <fmt/format.h>
#include <fmt/std.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/quaternion_common.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_format_traits.hpp>

#include <algorithm>
#include <deque>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <stack>
#include <string>
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

constexpr glm::uint kSubgroupSizeX = 32;
constexpr glm::uint kSubgroupSizeY = 32;

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
class ResourceQueue final : std::queue<T, std::deque<T>>
{
    using base = std::queue<T, std::deque<T>>;

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

    T steal()
    {
        auto value = std::move(front());
        pop();
        return value;
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

    static constexpr void completeClassContext [[maybe_unused]] ()
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

inline void resetFence(const engine::Context & context, const Fence & fence)
{
    ASSERT(fence);
    ASSERT(*fence);
    context.getDevice().getDevice().resetFences(**fence, context.getDispatcher());
}

inline void waitFence(const engine::Context & context, const Fence & fence)
{
    ASSERT(fence);
    ASSERT(*fence);
    auto result = context.getDevice().getDevice().waitForFences(**fence, vk::True, std::numeric_limits<uint64_t>::max(), context.getDispatcher());
    INVARIANT(result == vk::Result::eSuccess, "Display fence: {}", result);
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
        resetFence(context, fence);
        fencePool.push(std::move(fence));
    }

    void put(Fence && fence)
    {
        checkFenceUnique(fence);
        resetFence(context, fence);
        fencePool.push(std::move(fence));
    }

private:
    const engine::Context & context;
    ResourceStack<Fence> fencePool;
};

#pragma pack(push, 1)

struct UniformBuffer
{
    vk::Bool32 useOffscreenTexture = vk::False;
    vk::Bool32 discardInvisible = vk::False;
    vk::Bool32 wireFrame = vk::False;
    glm::vec3 position{0.0f};
    glm::float32 width = 0.0f;
    glm::float32 height = 0.0f;
    glm::float32 zNear = 0.0f;
    glm::float32 zFar = 0.0f;
    glm::float32 alpha = 0.0f;
    glm::mat4 windowMvp{1.0f};
};
static_assert(std::is_standard_layout_v<UniformBuffer>);

struct ScenePushConstants
{
    glm::mat4 mvp{1.0f};
};
static_assert(std::is_standard_layout_v<ScenePushConstants>);

struct DisplayPushConstants
{
    float x = 1E-5f;
};
static_assert(std::is_standard_layout_v<DisplayPushConstants>);

struct TraceUniformBuffer
{
    glm::uint triangleCount;
    glm::uint treeDepthMax;
    glm::uint polygonCount;
    glm::uint nodeCount;
    vk::DeviceAddress triangles;
    vk::DeviceAddress polygons;
    vk::DeviceAddress nodes;
    vk::DeviceAddress nodeParents;
};
static_assert(std::is_standard_layout_v<TraceUniformBuffer>);

struct Frustum
{
    glm::vec3 leftTop;
    glm::vec3 rightTop;
    glm::vec3 leftBottom;
    glm::vec3 rightBottom;
};
static_assert(std::is_standard_layout_v<Frustum>);

struct TracePushConstants
{
    glm::vec4 clearColor;
    glm::vec3 pos;
    Frustum frustum;
    glm::uint nodeIndex;
};
static_assert(std::is_standard_layout_v<TracePushConstants>);

#pragma pack(pop)

struct UniformBufferResource final
{
    engine::Buffer<UniformBuffer> uniformBuffer;

    [[nodiscard]] static engine::DescriptorBindingNameAndType getBindingName()
    {
        return {"uniformBuffer"s, vk::DescriptorType::eUniformBuffer};
    }

    [[nodiscard]] DescriptorInfo getDescriptorInfo(bool descriptorBufferEnabled) const
    {
        const auto getDescriptorData = [this, descriptorBufferEnabled]() -> DescriptorData
        {
            if (descriptorBufferEnabled) {
                return DescriptorBufferData{uniformBuffer.getDescriptorAddressInfo()};
            } else {
                return viewer::DescriptorSetData{uniformBuffer.getDescriptorBufferInfo()};
            }
        };
        return {getBindingName(), getDescriptorData()};
    }

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        utils::OneTime<UniformBufferResource>::checkTraits();
    }
};

struct TraceSceneResources final
{
    Tree tree;
    engine::Buffer<TraceUniformBuffer> uniformBuffer;

    [[nodiscard]] static engine::DescriptorBindingNameAndType getBindingName()
    {
        return {""s, vk::DescriptorType::eUniformBuffer};
    }

    [[nodiscard]] DescriptorInfo getDescriptorInfo(bool descriptorBufferEnabled) const
    {
        const auto getDescriptorData = [this, descriptorBufferEnabled]() -> DescriptorData
        {
            if (descriptorBufferEnabled) {
                return DescriptorBufferData{uniformBuffer.getDescriptorAddressInfo()};
            } else {
                return viewer::DescriptorSetData{uniformBuffer.getDescriptorBufferInfo()};
            }
        };
        return {getBindingName(), getDescriptorData()};
    }

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        utils::OneTime<UniformBufferResource>::checkTraits();
    }
};

struct FrameResourcesAndDescriptors
{
    UniformBufferResource resources;
    Descriptors directDescriptors;
    std::optional<Descriptors> displayDescriptors;

    FrameResourcesAndDescriptors(UniformBufferResource && resources, Descriptors && sceneDescriptors, std::optional<Descriptors> && displayDescriptors)
        : resources{std::move(resources)}
        , directDescriptors{std::move(sceneDescriptors)}
        , displayDescriptors{std::move(displayDescriptors)}
    {}
};

struct TraceSceneResourcesAndDescriptors
{
    TraceSceneResources resources;
    Descriptors descriptors;

    TraceSceneResourcesAndDescriptors(TraceSceneResources && resources, Descriptors && descriptors)
        : resources{std::move(resources)}
        , descriptors{std::move(descriptors)}
    {}
};

struct TraceFrameResourcesAndDescriptors
{
    TraceFrameResources resources;
    Descriptors writeDescriptors;
    Descriptors readDescriptors;

    TraceFrameResourcesAndDescriptors(TraceFrameResources && resources, Descriptors && writeDescriptors, Descriptors && readDescriptors)
        : resources{std::move(resources)}
        , writeDescriptors{std::move(writeDescriptors)}
        , readDescriptors{std::move(readDescriptors)}
    {}
};

struct SceneResourcesAndDescriptors
{
    SceneResources resources;
    Descriptors descriptors;

    SceneResourcesAndDescriptors(SceneResources && resources, Descriptors && descriptors)
        : resources{std::move(resources)}
        , descriptors{std::move(descriptors)}
    {}
};

struct DrawOffscreenResourcesAndDescriptors
{
    DrawOffscreenResources resources;
    Descriptors descriptors;

    DrawOffscreenResourcesAndDescriptors(DrawOffscreenResources && resources, Descriptors && descriptors)
        : resources{std::move(resources)}
        , descriptors{std::move(descriptors)}
    {}
};

class DrawOffscreenPool final
    : utils::NonCopyable
    , public std::enable_shared_from_this<DrawOffscreenPool>
{
    struct Private
    {
        explicit Private() = default;
    };

public:
    DrawOffscreenPool(Private, const engine::Context & context, const Engine & engine, std::shared_ptr<const vk::UniqueSampler> && sampler)
        : context{context}
        , engine{engine}
        , displayRenderPass{OffscreenRenderPass::make(context)}
        , displayGraphicsPipeline{makeGraphicsPipeline()}
        , sampler{std::move(sampler)}
    {}

    [[nodiscard]] static std::shared_ptr<DrawOffscreenPool> make(const engine::Context & context, const Engine & engine, std::shared_ptr<const vk::UniqueSampler> sampler)
    {
        return std::make_shared<DrawOffscreenPool>(Private{}, context, engine, std::move(sampler));
    }

    [[nodiscard]] const OffscreenRenderPass & getOffscreenRenderPass() const &
    {
        return displayRenderPass;
    }

    [[nodiscard]] const GraphicsPipeline & getGraphicsPipeline() const &
    {
        return displayGraphicsPipeline;
    }

    [[nodiscard]] std::shared_ptr<DrawOffscreenResourcesAndDescriptors> get(const vk::Extent2D & framebufferSize, std::shared_ptr<const engine::ShaderStages> shaderStages) &
    {
        std::shared_ptr<DrawOffscreenResourcesAndDescriptors> resourcesAndDescriptors;
        while (!pool.empty()) {
            resourcesAndDescriptors = std::move(pool.top());
            pool.pop();
            const auto & framebuffer = resourcesAndDescriptors->resources.framebuffer;
            if ((false)) {  // TODO: rethink more thoroughly
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
            } else {
                if (framebufferSize != framebuffer.size) {
                    break;
                }
            }
            return resourcesAndDescriptors;
        }
        if (resourcesAndDescriptors) {
            DrawOffscreenResources resources{context, framebufferSize, displayRenderPass, std::move(resourcesAndDescriptors->resources.sampler)};
            auto descriptors = std::move(resourcesAndDescriptors->descriptors);
            auto descriptorInfos = {resources.getDescriptorInfo(engine.getSettings().descriptorBufferEnabled)};
            descriptors.fill(descriptorInfos);
            return std::make_shared<DrawOffscreenResourcesAndDescriptors>(std::move(resources), std::move(descriptors));
        } else {
            DrawOffscreenResources resources{context, framebufferSize, displayRenderPass, sampler};
            auto descriptors = engine.makeDescriptors("display"sv, std::move(shaderStages), resources);
            return std::make_shared<DrawOffscreenResourcesAndDescriptors>(std::move(resources), std::move(descriptors));
        }
    }

    void put(std::shared_ptr<DrawOffscreenResourcesAndDescriptors> resourcesAndDescriptors) &
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
    ResourceStack<std::shared_ptr<DrawOffscreenResourcesAndDescriptors>> pool;

    [[nodiscard]] GraphicsPipeline makeGraphicsPipeline() const
    {
        GraphicsPipeline graphicsPipeline{engine.getPipelines().getSceneShaders()};
        graphicsPipeline.initPipeline("offscreen scene"sv, context, engine.getPipelines().getPipelineCache(), engine.getSettings().descriptorBufferEnabled, displayRenderPass).create();
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
        queue.submit(submitInfo, completionFence ? **completionFence : VK_NULL_HANDLE);

        if (waitIdle) {
            if (completionFence) {
                ASSERT(*completionFence);
                auto result = context.getDevice().getDevice().waitForFences(**completionFence, vk::True, std::numeric_limits<uint64_t>::max(), context.getDispatcher());
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

    [[nodiscard]] operator vk::CommandBuffer() const &
    {
        return getCommandBuffer();
    }

    void setCompletionFence(Fence completionFence)
    {
        this->completionFence = std::move(completionFence);
    }

    void setWaitCompletion(bool waitIdle = true)
    {
        this->waitIdle = waitIdle;
    }

    void setWaitCompletion(Fence completionFence)
    {
        setCompletionFence(completionFence);
        setWaitCompletion();
    }

private:
    std::string name;
    const engine::Context & context;
    const engine::Queue & queue;

    bool waitIdle = false;
    Fence completionFence;
    std::shared_ptr<const engine::CommandBuffers> commandBuffers;
    std::vector<vk::Semaphore> waitSemaphores;
    std::vector<vk::PipelineStageFlags> waitDstStageMasks;
    std::vector<vk::Semaphore> signalSemaphores;

    static constexpr void completeClassContext [[maybe_unused]] ()
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

UniformBuffer getUniformBuffer(const FrameSettings & frameSettings)
{
    return {
        .useOffscreenTexture = frameSettings.useOffscreenTexture ? vk::True : vk::False,
        .discardInvisible = frameSettings.discardInvisible ? vk::True : vk::False,
        .wireFrame = frameSettings.wireFrame ? vk::True : vk::False,
        .position = frameSettings.position,
        .width = frameSettings.width,
        .height = frameSettings.height,
        .zNear = frameSettings.zNear,
        .zFar = frameSettings.zFar,
        .alpha = frameSettings.alpha,
        .windowMvp = frameSettings.windowMvp,
    };
}

TraceUniformBuffer getTraceUniformBuffer(const Tree & tree)
{
    return {
        .triangleCount = utils::autoCast(tree.getTriangleCount()),
        .treeDepthMax = utils::autoCast(std::size(tree.getLayerSizes())),
        .polygonCount = utils::autoCast(tree.getPolygonCount()),
        .nodeCount = utils::autoCast(tree.getNodeCount()),
        .triangles = tree.getTriangleAddress(),
        .polygons = tree.getPolygonAddress(),
        .nodes = tree.getNodeAddress(),
        .nodeParents = tree.getNodeParentAddress(),
    };
}

[[nodiscard]] ScenePushConstants getScenePushConstants(const FrameSettings & frameSettings)
{
    auto view = glm::translate(glm::toMat4(glm::conjugate(frameSettings.orientation)), -frameSettings.position);
    auto projection = glm::perspectiveFovLH(frameSettings.fov, frameSettings.width, frameSettings.height, frameSettings.zNear, frameSettings.zFar);
    auto mvp = projection * view;
    if (!frameSettings.useOffscreenTexture) {
        auto windowMvp = glm::scale(frameSettings.windowMvp, glm::vec3{1.0f, -1.0f, 1.0f});
        mvp = windowMvp * mvp;
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

[[nodiscard]] TracePushConstants getTracePushConstants(const FrameSettings & frameSettings)
{
    const glm::float32 dy = glm::tan(frameSettings.fov * 0.5f);
    const glm::float32 dx = dy * (frameSettings.width / frameSettings.height);
    const glm::vec3 leftTop = glm::rotate(frameSettings.orientation, glm::vec3{-dx, dy, 1.0f});
    const glm::vec3 rightTop = glm::rotate(frameSettings.orientation, glm::vec3{dx, dy, 1.0f});
    const glm::vec3 leftBottom = glm::rotate(frameSettings.orientation, glm::vec3{-dx, -dy, 1.0f});
    const glm::vec3 rightBottom = glm::rotate(frameSettings.orientation, glm::vec3{dx, -dy, 1.0f});
    return {
        .clearColor = frameSettings.clearColor,
        .pos = frameSettings.position,
        .frustum = {
            .leftTop = leftTop,
            .rightTop = rightTop,
            .leftBottom = leftBottom,
            .rightBottom = rightBottom,
        },
        .nodeIndex = 0,  // TODO: O(logN) -> O(1) on movies
    };
}

}  // namespace

vk::Extent2D FrameSettings::getFramebufferSize() const
{
    float w = std::ceil(width);
    float h = std::ceil(height);
    return {
        .width = utils::autoCast(w),
        .height = utils::autoCast(h),
    };
}

struct Renderer::Impl : utils::NonCopyable
{
    using DescriptorRefs = std::initializer_list<std::reference_wrapper<const Descriptors>>;

    std::string name;
    const engine::Context & context;
    const Engine & engine;
    const uint32_t framesInFlight;

    const engine::Queue graphicsQueue{"renderer"sv, context, context.getPhysicalDevice().graphicsQueueCreateInfo};
    const engine::Queue computeQueue{"compute"sv, context, context.getPhysicalDevice().computeQueueCreateInfo};

    FrameSettings frameSettings;
    scene_data::SceneDataPtr sceneData;

    FencePool fencePool{context};

    const std::shared_ptr<GraphicsPipeline> directGraphicsPipeline = std::make_shared<GraphicsPipeline>(engine.getPipelines().getSceneShaders());
    const std::shared_ptr<GraphicsPipeline> displayGraphicsPipeline = std::make_shared<GraphicsPipeline>(engine.getPipelines().getDisplayShaders());
    const std::shared_ptr<ComputePipeline> traceComputePipeline = std::make_shared<ComputePipeline>(makeTraceComputePipeline(engine.getPipelines().getTraceSahKdTreeShaders()));

    const std::shared_ptr<const vk::UniqueSampler> sampler = makeSampler();

    std::shared_ptr<SceneResourcesAndDescriptors> sceneResourcesAndDescriptors;
    ResourceStack<std::shared_ptr<FrameResourcesAndDescriptors>> frameResourcesAndDescriptorsPool;
    std::shared_ptr<FrameResourcesAndDescriptors> frameResourcesAndDescriptors;

    std::shared_ptr<TraceSceneResourcesAndDescriptors> traceSceneResourcesAndDescriptors;
    ResourceStack<std::shared_ptr<TraceFrameResourcesAndDescriptors>> traceFrameResourcesAndDescriptorsPool;
    std::shared_ptr<TraceFrameResourcesAndDescriptors> traceFrameResourcesAndDescriptors;

    std::shared_ptr<DrawOffscreenPool> drawOffscreenPool;
    Fence drawRasterOffscreenFinishedFence;
    std::shared_ptr<DrawOffscreenResourcesAndDescriptors> offscreenResourcesAndDescriptors;
    std::shared_ptr<const engine::CommandBuffers> offscreenRasterCommandBuffers;

    // revocation lists should be the last members
    std::vector<std::vector<Resource>> deferredDeletionSlots{framesInFlight};

    Impl(std::string_view name, const engine::Context & context, const Engine & engine, uint32_t framesInFlight);

    [[nodiscard]] std::shared_ptr<const vk::UniqueSampler> makeSampler() const;

    void setFrameSettings(const FrameSettings & frameSettings);

    void unsetScene();
    void setScene(scene_data::SceneDataPtr sceneData);

    void unsetTree();
    void setTree(builder::TreePtr builderTree);

    [[nodiscard]] ComputePipeline makeTraceComputePipeline(std::shared_ptr<const Shaders> shaders) const;

    void bindPipeline(vk::CommandBuffer commandBuffer, vk::PipelineBindPoint pipelineBindPoint, const Shaders & shaders, DescriptorRefs descriptors, const std::byte * pushConstants) const;

    template<typename Pipeline>
    void bindPipeline(vk::CommandBuffer commandBuffer, const Pipeline & pipeline, DescriptorRefs descriptors, const std::byte * pushConstants) const
    {
        commandBuffer.bindPipeline(Pipeline::kPipelineBindPoint, pipeline.pipeline.value(), context.getDispatcher());
        bindPipeline(commandBuffer, Pipeline::kPipelineBindPoint, *pipeline.shaders, descriptors, pushConstants);
    }

    void drawScene(vk::CommandBuffer commandBuffer, const GraphicsPipeline & pipeline) const;
    void offscreenPass(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass);
    void drawDisplay(vk::CommandBuffer commandBuffer, const GraphicsPipeline & pipeline);
    void traceScene(vk::CommandBuffer graphicsCommandBuffer, const ComputePipeline & pipeline);

    void advance(vk::CommandBuffer commandBuffer, uint32_t currentFrameSlot);

    void updateRenderPass(vk::RenderPass renderPass, bool isRenderPassFormatChanged, uint32_t currentFrameSlot);

    void render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, bool isRenderPassFormatChanged, uint32_t currentFrameSlot);

    [[nodiscard]] std::shared_ptr<FrameResourcesAndDescriptors> getFrameDescriptors();
    void putFrameDescriptors(std::shared_ptr<FrameResourcesAndDescriptors> && frameDescriptors);

    [[nodiscard]] std::shared_ptr<TraceFrameResourcesAndDescriptors> getTraceFrameDescriptors();
    void putTraceFrameDescriptors(std::shared_ptr<TraceFrameResourcesAndDescriptors> && frameDescriptors);

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

Renderer::Renderer(std::string_view name, const engine::Context & context, const Engine & engine, uint32_t framesInFlight)
    : impl_{std::make_unique<Impl>(name, context, engine, framesInFlight)}
{}

uint32_t Renderer::getFramesInFlight() const
{
    return impl_->framesInFlight;
}

Renderer::~Renderer() = default;

void Renderer::setFrameSettings(const FrameSettings & frameSettings)
{
    return impl_->setFrameSettings(frameSettings);
}

void Renderer::setScene(scene_data::SceneDataPtr sceneData)
{
    return impl_->setScene(std::move(sceneData));
}

void Renderer::unsetScene()
{
    impl_->unsetScene();
}

const scene_data::SceneDataPtr & Renderer::getScene() const &
{
    return impl_->sceneData;
}

void Renderer::setTree(builder::TreePtr builderTree)
{
    return impl_->setTree(std::move(builderTree));
}

void Renderer::unsetTree()
{
    impl_->unsetTree();
}

builder::TreePtr Renderer::getTree() const
{
    return impl_->traceSceneResourcesAndDescriptors ? impl_->traceSceneResourcesAndDescriptors->resources.tree.getBuilderTree() : nullptr;
}

void Renderer::advance(vk::CommandBuffer commandBuffer, uint32_t currentFrameSlot)
{
    return impl_->advance(commandBuffer, currentFrameSlot);
}

void Renderer::render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, bool isRenderPassFormatChanged, uint32_t currentFrameSlot)
{
    return impl_->render(commandBuffer, renderPass, isRenderPassFormatChanged, currentFrameSlot);
}

Renderer::Impl::Impl(std::string_view name, const engine::Context & context, const Engine & engine, uint32_t framesInFlight)
    : name{name}
    , context{context}
    , engine{engine}
    , framesInFlight{framesInFlight}
{
    uint32_t maxPushConstantsSize = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits.maxPushConstantsSize;
    INVARIANT(sizeof(ScenePushConstants) <= maxPushConstantsSize, "{} ^ {}", sizeof(ScenePushConstants), maxPushConstantsSize);
}

std::shared_ptr<const vk::UniqueSampler> Renderer::Impl::makeSampler() const
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
        .anisotropyEnable = vk::False,
        .maxAnisotropy = maxSamplerAnisotropy,
        .compareEnable = vk::False,
        .compareOp = vk::CompareOp::eNever,
        .minLod = 0.0f,
        .maxLod = 0.0f,
        .borderColor = vk::BorderColor::eFloatTransparentBlack,
        .unnormalizedCoordinates = vk::False,
    };
    return std::make_shared<vk::UniqueSampler>(context.getDevice().getDevice().createSamplerUnique(samplerCreateInfo, context.getAllocationCallbacks(), context.getDispatcher()));
}

void Renderer::Impl::setFrameSettings(const FrameSettings & frameSettings)
{
    this->frameSettings = frameSettings;
}

void Renderer::Impl::unsetScene()
{
    sceneData.reset();
}

void Renderer::Impl::setScene(scene_data::SceneDataPtr newSceneData)
{
    ASSERT(!sceneData);
    ASSERT(newSceneData);
    sceneData = std::move(newSceneData);
}

void Renderer::Impl::unsetTree()
{
    if (!traceSceneResourcesAndDescriptors) {
        return;
    }
    traceSceneResourcesAndDescriptors.reset();
    SPDLOG_INFO("{}: Tree is unset", name);
}

void Renderer::Impl::setTree(builder::TreePtr builderTree)
{
    ASSERT(builderTree);
    if (traceSceneResourcesAndDescriptors && (traceSceneResourcesAndDescriptors->resources.tree.getBuilderTree() == builderTree)) {
        return;
    }

    Tree tree{name, context, builderTree};

    engine::Buffer<TraceUniformBuffer> uniformBuffer{engine.createUniformBuffer(sizeof(TraceUniformBuffer))};
    uniformBuffer.map().at(0) = getTraceUniformBuffer(tree);

    auto shaders = engine.getPipelines().getTraceSahKdTreeShaders();

    TraceSceneResources traceSceneResources = {
        .tree = std::move(tree),
        .uniformBuffer = std::move(uniformBuffer),
    };
    auto descriptors = engine.makeDescriptors("trace"sv, shaders->getShaderStagesPtr(), traceSceneResources);
    traceSceneResourcesAndDescriptors = std::make_shared<TraceSceneResourcesAndDescriptors>(std::move(traceSceneResources), std::move(descriptors));

    SPDLOG_INFO("{}: Tree is set", name);
}

ComputePipeline Renderer::Impl::makeTraceComputePipeline(std::shared_ptr<const Shaders> shaders) const
{
    ComputePipeline computePipeline{std::move(shaders)};
    auto & pipeline = computePipeline.initPipeline("trace"sv, context, engine.getPipelines().getPipelineCache(), engine.getSettings().descriptorBufferEnabled);
    struct SpecializationData
    {
        const glm::uint kSubgroupSizeX;
        const glm::uint kSubgroupSizeY;
        const glm::float32 kEps = 1E-7f;
    };
    const SpecializationData specializationData = {
        .kSubgroupSizeX = kSubgroupSizeX,
        .kSubgroupSizeY = kSubgroupSizeY,
    };
    pipeline.specializationInfo.setData<SpecializationData>(specializationData);
    const std::initializer_list<vk::SpecializationMapEntry> specializationMapEntries = {
        {
            .constantID = 0,
            .offset = offsetof(SpecializationData, kSubgroupSizeX),
            .size = sizeof(SpecializationData::kSubgroupSizeX),
        },
        {
            .constantID = 1,
            .offset = offsetof(SpecializationData, kSubgroupSizeY),
            .size = sizeof(SpecializationData::kSubgroupSizeY),
        },
        {
            .constantID = 2,
            .offset = offsetof(SpecializationData, kEps),
            .size = sizeof(SpecializationData::kEps),
        },
    };
    pipeline.specializationInfo.setMapEntries(specializationMapEntries);
    pipeline.create();
    return computePipeline;
}

void Renderer::Impl::bindPipeline(vk::CommandBuffer commandBuffer, vk::PipelineBindPoint pipelineBindPoint, const Shaders & shaders, DescriptorRefs descriptors, const std::byte * pushConstants) const
{
    constexpr uint32_t kFirstSet = 0;
    vk::PipelineLayout pipelineLayout = shaders.getPipelineLayout();
    if (engine.getSettings().descriptorBufferEnabled) {
        std::vector<vk::DescriptorBufferBindingInfoEXT> descriptorBufferBindingInfos;
        descriptorBufferBindingInfos.reserve(std::size(descriptors));
        for (const Descriptors & d : descriptors) {
            descriptorBufferBindingInfos.push_back(d.getDescriptorBuffer().getDescriptorBufferBindingInfo());
        }
        commandBuffer.bindDescriptorBuffersEXT(descriptorBufferBindingInfos, context.getDispatcher());

        std::vector<uint32_t> bufferIndices(std::size(descriptorBufferBindingInfos));
        std::iota(std::begin(bufferIndices), std::end(bufferIndices), uint32_t{0});

        std::vector<vk::DeviceSize> offsets(std::size(descriptorBufferBindingInfos));
        std::fill(std::begin(offsets), std::end(offsets), vk::DeviceSize{0});

        commandBuffer.setDescriptorBufferOffsetsEXT(pipelineBindPoint, pipelineLayout, kFirstSet, bufferIndices, offsets, context.getDispatcher());
    } else {
        std::vector<vk::DescriptorSet> descriptorSets;
        descriptorSets.reserve(std::size(descriptors));
        for (const Descriptors & d : descriptors) {
            descriptorSets.push_back(d.getDescriptorSet());
        }
        constexpr auto kDynamicOffsets = nullptr;
        commandBuffer.bindDescriptorSets(pipelineBindPoint, pipelineLayout, kFirstSet, descriptorSets, kDynamicOffsets, context.getDispatcher());
    }

    for (const auto & pushConstantRange : shaders.getShaderStages().pushConstantRanges) {
        commandBuffer.pushConstants(pipelineLayout, pushConstantRange.stageFlags, pushConstantRange.offset, pushConstantRange.size, std::next(pushConstants, pushConstantRange.offset), context.getDispatcher());
    }
}

void Renderer::Impl::drawScene(vk::CommandBuffer commandBuffer, const GraphicsPipeline & pipeline) const
{
    {
        ASSERT(frameResourcesAndDescriptors);
        ASSERT(sceneResourcesAndDescriptors);
        DescriptorRefs descriptors = {
            std::cref(frameResourcesAndDescriptors->directDescriptors),
            std::cref(sceneResourcesAndDescriptors->descriptors),
        };
        ScenePushConstants pushConstants = getScenePushConstants(frameSettings);
        bindPipeline(commandBuffer, pipeline, descriptors, utils::autoCast(&pushConstants));
    }

    constexpr engine::LabelColor kMagentaColor = {1.0f, 0.0f, 1.0f, 1.0f};
    auto drawSceneLabel = engine::ScopedCommandBufferLabel::create(context.getDispatcher(), commandBuffer, "Draw scene"sv, kMagentaColor);

    vk::Viewport viewport;
    vk::Rect2D scissor;
    if (frameSettings.useOffscreenTexture) {
        ASSERT(offscreenResourcesAndDescriptors);
        viewport = vk::Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = frameSettings.width,
            .height = frameSettings.height,
            .minDepth = engine::kMinDepth,
            .maxDepth = 1.0f,
        };
        scissor = vk::Rect2D{
            .offset = {
                .x = 0,
                .y = 0,
            },
            .extent = offscreenResourcesAndDescriptors->resources.framebuffer.size,
        };
    } else {
        viewport = frameSettings.viewport;
        scissor = frameSettings.scissor;
    }
    constexpr uint32_t kFirstViewport = 0;
    commandBuffer.setViewport(kFirstViewport, viewport, context.getDispatcher());
    constexpr uint32_t kFirstScissor = 0;
    commandBuffer.setScissor(kFirstScissor, scissor, context.getDispatcher());

    ASSERT(sceneResourcesAndDescriptors);
    const auto & sceneResources = sceneResourcesAndDescriptors->resources;

    {
        constexpr uint32_t kFirstBinding = 0;
        const auto bufferOrNull = [this](const auto & wrapper) -> vk::Buffer
        {
            if (wrapper) {
                return wrapper.value();
            } else {
                ASSERT(context.getPhysicalDevice().features2Chain.get<vk::PhysicalDeviceMaintenance6FeaturesKHR>().maintenance6 == vk::True);
                ASSERT(context.getPhysicalDevice().features2Chain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>().nullDescriptor == vk::True);
                return VK_NULL_HANDLE;
            }
        };
        vk::Buffer vertexBuffer = bufferOrNull(sceneResources.vertexBuffer);
        constexpr vk::DeviceSize kVertexBufferOffset = 0;
        commandBuffer.bindVertexBuffers(kFirstBinding, vertexBuffer, kVertexBufferOffset, context.getDispatcher());  // bindVertexBuffers2?
    }

    const auto & features2Chain = context.getPhysicalDevice().features2Chain;
    vk::Buffer indexBuffer;
    if (sceneResources.indexBuffer) {
        indexBuffer = sceneResources.indexBuffer.value();
    } else {
        ASSERT(features2Chain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>().nullDescriptor == vk::True);
        ASSERT(features2Chain.get<vk::PhysicalDeviceMaintenance6FeaturesKHR>().maintenance6 == vk::True);
        // TODO: or draw non-indexed
    }
    constexpr vk::DeviceSize kIndexBufferDeviceOffset = 0;
    if (engine.getSettings().multiDrawIndirectEnabled) {
        ASSERT(std::empty(sceneResources.indexTypes));
        auto indexType = sceneResources.maxIndexType;
        commandBuffer.bindIndexBuffer(indexBuffer, kIndexBufferDeviceOffset, indexType, context.getDispatcher());  // vkCmdBindIndexBuffer2KHR is not supported by Renderdoc
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
            commandBuffer.bindIndexBuffer(indexBuffer, kIndexBufferDeviceOffset, *indexType++, context.getDispatcher());
            commandBuffer.drawIndexed(indexCount, instanceCount, firstIndex, vertexOffset, firstInstance, context.getDispatcher());
            // SPDLOG_TRACE("{{.indexCount = {}, .instanceCount = {}, .firstIndex = {}, .vertexOffset = {}, .firstInstance = {})}}", indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
        }
        ASSERT(indexType == std::cend(sceneResources.indexTypes));
    }
}

void Renderer::Impl::offscreenPass(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass)
{
    constexpr engine::LabelColor kGreenColor = {0.0f, 1.0f, 0.0f, 1.0f};
    auto offscreenPassLabel = engine::ScopedCommandBufferLabel::create(context.getDispatcher(), commandBuffer, "Offscreen pass"sv, kGreenColor);

    ASSERT(offscreenResourcesAndDescriptors);
    const Framebuffer & framebuffer = offscreenResourcesAndDescriptors->resources.framebuffer;
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
                    frameSettings.clearColor.r,
                    frameSettings.clearColor.g,
                    frameSettings.clearColor.b,
                    frameSettings.clearColor.a,
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
    drawScene(commandBuffer, drawOffscreenPool->getGraphicsPipeline());
    vk::SubpassEndInfo subpassEndInfo;
    commandBuffer.endRenderPass2(subpassEndInfo, context.getDispatcher());
}

void Renderer::Impl::drawDisplay(vk::CommandBuffer commandBuffer, const GraphicsPipeline & pipeline)
{
    {
        ASSERT(frameResourcesAndDescriptors->displayDescriptors);
        const Descriptors * secondBinding = nullptr;
        if (traceFrameResourcesAndDescriptors) {
            ASSERT(!offscreenResourcesAndDescriptors);
            secondBinding = &traceFrameResourcesAndDescriptors->readDescriptors;
        } else {
            ASSERT(offscreenResourcesAndDescriptors);
            secondBinding = &offscreenResourcesAndDescriptors->descriptors;
        }
        ASSERT(secondBinding);
        const DescriptorRefs descriptors = {
            std::cref(frameResourcesAndDescriptors->displayDescriptors.value()),
            std::cref(*secondBinding),
        };
        const DisplayPushConstants displayPushConstants = getDisplayPushConstants(frameSettings);
        bindPipeline(commandBuffer, pipeline, descriptors, utils::autoCast(&displayPushConstants));
    }

    {
        constexpr uint32_t kFirstBinding = 0;
        constexpr vk::Buffer kVertexBuffer = VK_NULL_HANDLE;
        constexpr vk::DeviceSize kVertexBufferOffset = 0;
        commandBuffer.bindVertexBuffers(kFirstBinding, kVertexBuffer, kVertexBufferOffset, context.getDispatcher());
    }

    {
        constexpr uint32_t kFirstViewport = 0;
        commandBuffer.setViewport(kFirstViewport, frameSettings.viewport, context.getDispatcher());

        constexpr uint32_t kFirstScissor = 0;
        commandBuffer.setScissor(kFirstScissor, frameSettings.scissor, context.getDispatcher());
    }

    commandBuffer.draw(4, 1, 0, 0, context.getDispatcher());
}

void Renderer::Impl::traceScene(vk::CommandBuffer /*graphicsCommandBuffer*/, const ComputePipeline & pipeline)
{
    traceFrameResourcesAndDescriptors = getTraceFrameDescriptors();

    auto & image = traceFrameResourcesAndDescriptors->resources.image;
    const uint32_t graphicsQueueFamilyIndex = graphicsQueue.getQueueCreateInfo().familyIndex;
    const uint32_t computeQueueFamilyIndex = computeQueue.getQueueCreateInfo().familyIndex;
    {
        auto fenceGraphics = fencePool.get();
        {
            ScopedCommandBuffer graphicsReleaseCommandBuffer{"Graphics release"sv, context, graphicsQueue};
            graphicsReleaseCommandBuffer.setWaitCompletion(fenceGraphics);
            image.release(graphicsReleaseCommandBuffer, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite, vk::ImageLayout::eGeneral, computeQueueFamilyIndex);
        }
        fencePool.put(std::move(fenceGraphics));
    }
    {
        auto fenceCompute = fencePool.get();
        {
            ScopedCommandBuffer computeCommandBuffer{"Offscreen scene trace"sv, context, computeQueue};
            computeCommandBuffer.setWaitCompletion(fenceCompute);

            ASSERT(traceSceneResourcesAndDescriptors);
            DescriptorRefs descriptors = {
                std::cref(traceSceneResourcesAndDescriptors->descriptors),
                std::cref(traceFrameResourcesAndDescriptors->writeDescriptors),
            };
            TracePushConstants pushConstants = getTracePushConstants(frameSettings);
            bindPipeline(computeCommandBuffer, pipeline, descriptors, utils::autoCast(&pushConstants));

            image.acquire(computeCommandBuffer, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite, vk::ImageLayout::eGeneral, computeQueueFamilyIndex);
            {
                auto [width, height] = image.getExtent2D();
                width = utils::divUp(width, kSubgroupSizeX) * kSubgroupSizeX;
                height = utils::divUp(height, kSubgroupSizeY) * kSubgroupSizeY;
                constexpr uint32_t kDepth = 1;
                computeCommandBuffer.getCommandBuffer().dispatch(width, height, kDepth, context.getDispatcher());
            }
            image.release(computeCommandBuffer, vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead, vk::ImageLayout::eGeneral, graphicsQueueFamilyIndex);
        }
        fencePool.put(std::move(fenceCompute));
    }
    {
        auto fenceGraphics = fencePool.get();
        {
            ScopedCommandBuffer graphicsAcquireCommandBuffer{"Graphics acquire"sv, context, graphicsQueue};
            graphicsAcquireCommandBuffer.setWaitCompletion(fenceGraphics);
            image.acquire(graphicsAcquireCommandBuffer, vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead, vk::ImageLayout::eGeneral, graphicsQueueFamilyIndex);
        }
        fencePool.put(std::move(fenceGraphics));
    }
}

void Renderer::Impl::advance(vk::CommandBuffer commandBuffer, uint32_t currentFrameSlot)
{
    ASSERT_MSG(currentFrameSlot < framesInFlight, "{} ^ {}", currentFrameSlot, framesInFlight);

    auto unmuteMessageGuard = context.getInstance().unmuteDebugUtilsMessages(kUnmutedMessageIdNumbers);

    deleteDeferred(currentFrameSlot);

    uint32_t previousFrameSlot = utils::modDown(currentFrameSlot, framesInFlight);

    if (offscreenResourcesAndDescriptors) {
        Recycler recycler = [this, drawFinishedFence = std::move(drawRasterOffscreenFinishedFence), resourcesAndDescriptors = std::move(offscreenResourcesAndDescriptors), displayCommandBuffers = std::move(offscreenRasterCommandBuffers),
                             drawOffscreenPool = drawOffscreenPool]() mutable
        {
            if (drawFinishedFence) {
                fencePool.waitAndPut(std::move(drawFinishedFence));
            }
            displayCommandBuffers.reset();
            if (drawOffscreenPool) {
                drawOffscreenPool->put(std::move(resourcesAndDescriptors));
            } else {
                resourcesAndDescriptors.reset();
            }
        };
        deferDeletion(previousFrameSlot, std::move(recycler));
    } else {
        INVARIANT(!drawRasterOffscreenFinishedFence, "");
    }
    if (traceFrameResourcesAndDescriptors) {
        Recycler recycler = [this, resourcesAndDescriptors = std::move(traceFrameResourcesAndDescriptors)]() mutable
        {
            putTraceFrameDescriptors(std::move(resourcesAndDescriptors));
        };
        deferDeletion(previousFrameSlot, std::move(recycler));
    }
    if (frameSettings.useOffscreenTexture) {
        if (sceneData) {
            if (!drawOffscreenPool) {
                drawOffscreenPool = DrawOffscreenPool::make(context, engine, sampler);
            }
        } else {
            deferDeletion(previousFrameSlot, std::move(drawOffscreenPool));
        }
    } else {
        deferDeletion(previousFrameSlot, std::move(drawOffscreenPool));
    }
    {
        if (frameResourcesAndDescriptors) {
            Recycler recycler{&Impl::putFrameDescriptors, this, std::move(frameResourcesAndDescriptors)};
            deferDeletion(previousFrameSlot, std::move(recycler));
        }
        if (sceneData) {
            frameResourcesAndDescriptors = getFrameDescriptors();
            frameResourcesAndDescriptors->resources.uniformBuffer.map().at(0) = getUniformBuffer(frameSettings);
        }
    }
    if (sceneData) {
        if (!sceneResourcesAndDescriptors) {
            auto & graphicsPipeline = frameSettings.useOffscreenTexture ? drawOffscreenPool->getGraphicsPipeline() : *directGraphicsPipeline;
            auto resources = engine.makeResources(*sceneData);
            auto descriptors = engine.makeDescriptors("scene"sv, graphicsPipeline.shaders->getShaderStagesPtr(), resources);
            sceneResourcesAndDescriptors = std::make_shared<SceneResourcesAndDescriptors>(std::move(resources), std::move(descriptors));
        }
    } else {
        deferDeletion(previousFrameSlot, std::move(sceneResourcesAndDescriptors));
    }
    if (frameSettings.useOffscreenTexture) {
        ASSERT(!offscreenResourcesAndDescriptors);
        ASSERT(!traceFrameResourcesAndDescriptors);
        if (traceSceneResourcesAndDescriptors) {
            traceScene(commandBuffer, *traceComputePipeline);
        } else if (sceneData) {
            offscreenResourcesAndDescriptors = drawOffscreenPool->get(frameSettings.getFramebufferSize(), displayGraphicsPipeline->shaders->getShaderStagesPtr());
            {
                ScopedCommandBuffer commandBuffer{"Offscreen scene draw"sv, context, graphicsQueue};
                const OffscreenRenderPass & offscreenRenderPass = drawOffscreenPool->getOffscreenRenderPass();
                offscreenPass(commandBuffer, offscreenRenderPass);
                INVARIANT(!drawRasterOffscreenFinishedFence, "");
                drawRasterOffscreenFinishedFence = fencePool.get();
                commandBuffer.setCompletionFence(drawRasterOffscreenFinishedFence);
                offscreenRasterCommandBuffers = commandBuffer.getCommandBuffers();
            }
        }
    }
}

void Renderer::Impl::updateRenderPass(vk::RenderPass renderPass, [[maybe_unused]] bool isRenderPassFormatChanged, uint32_t currentFrameSlot)
{
    ASSERT(directGraphicsPipeline);
    auto & graphicsPipeline = frameSettings.useOffscreenTexture ? *displayGraphicsPipeline : *directGraphicsPipeline;
    if (graphicsPipeline.pipeline) {
        if (graphicsPipeline.pipeline.value().getRenderPass() == renderPass) {
            return;
        }
        uint32_t previousFrameSlot = utils::modDown(currentFrameSlot, framesInFlight);
        deferDeletion(previousFrameSlot, std::make_shared<const engine::GraphicsPipeline>(std::move(graphicsPipeline.pipeline).value()));
        graphicsPipeline.pipeline.reset();
    }
    std::string_view name;
    if (frameSettings.useOffscreenTexture) {
        name = "offscreen display"sv;
    } else {
        name = "direct scene"sv;
    }
    auto & p = graphicsPipeline.initPipeline(name, context, engine.getPipelines().getPipelineCache(), engine.getSettings().descriptorBufferEnabled, renderPass);
    if (frameSettings.useOffscreenTexture) {
        p.pipelineInputAssemblyStateCreateInfo.setTopology(vk::PrimitiveTopology::eTriangleStrip);
    }
    p.create();
}

void Renderer::Impl::render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, bool isRenderPassFormatChanged, uint32_t currentFrameSlot)
{
    ASSERT(currentFrameSlot < framesInFlight);
    auto unmuteMessageGuard = context.getInstance().unmuteDebugUtilsMessages(kUnmutedMessageIdNumbers);
    updateRenderPass(renderPass, isRenderPassFormatChanged, currentFrameSlot);
    if (frameSettings.useOffscreenTexture) {
        if (drawRasterOffscreenFinishedFence) {
            fencePool.waitAndPut(std::move(drawRasterOffscreenFinishedFence));
        }
        if (frameResourcesAndDescriptors && frameResourcesAndDescriptors->displayDescriptors && (offscreenResourcesAndDescriptors || traceFrameResourcesAndDescriptors)) {
            drawDisplay(commandBuffer, *displayGraphicsPipeline);
        }
    } else {
        ASSERT(directGraphicsPipeline->pipeline);
        if (sceneResourcesAndDescriptors && frameResourcesAndDescriptors) {
            drawScene(commandBuffer, *directGraphicsPipeline);
        }
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
                auto displayDescriptors = engine.makeDescriptors("scene"sv, displayGraphicsPipeline->shaders->getShaderStagesPtr(), resources);
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
        sceneShaderStages = drawOffscreenPool->getGraphicsPipeline().shaders->getShaderStagesPtr();
    } else {
        sceneShaderStages = directGraphicsPipeline->shaders->getShaderStagesPtr();
    }
    auto directDescriptors = engine.makeDescriptors("scene"sv, std::move(sceneShaderStages), resources);
    std::optional<Descriptors> displayDescriptors;
    if (frameSettings.useOffscreenTexture) {
        displayDescriptors.emplace(engine.makeDescriptors("scene"sv, displayGraphicsPipeline->shaders->getShaderStagesPtr(), resources));
    }
    return std::make_shared<FrameResourcesAndDescriptors>(std::move(resources), std::move(directDescriptors), std::move(displayDescriptors));
}

void Renderer::Impl::putFrameDescriptors(std::shared_ptr<FrameResourcesAndDescriptors> && frameDescriptors)
{
    ASSERT_MSG(frameDescriptors.use_count() == 1, "Non-unique use in single-threaded context: {}", frameDescriptors.use_count());
    frameResourcesAndDescriptorsPool.push(std::move(frameDescriptors));
}

auto Renderer::Impl::getTraceFrameDescriptors() -> std::shared_ptr<TraceFrameResourcesAndDescriptors>
{
    std::shared_ptr<TraceFrameResourcesAndDescriptors> resourcesAndDescriptors;
    while (!std::empty(traceFrameResourcesAndDescriptorsPool)) {
        resourcesAndDescriptors = std::move(traceFrameResourcesAndDescriptorsPool.top());
        traceFrameResourcesAndDescriptorsPool.pop();
        return resourcesAndDescriptors;
    }
    TraceFrameResources resources{context, frameSettings.getFramebufferSize(), sampler};
    auto writeShaderStages = traceComputePipeline->shaders->getShaderStagesPtr();
    Descriptors writeDescriptors = engine.makeDescriptors("trace"sv, std::move(writeShaderStages), resources, true);
    auto readShaderStages = displayGraphicsPipeline->shaders->getShaderStagesPtr();
    Descriptors readDescriptors = engine.makeDescriptors("trace"sv, std::move(readShaderStages), resources, false);
    return std::make_shared<TraceFrameResourcesAndDescriptors>(std::move(resources), std::move(writeDescriptors), std::move(readDescriptors));
}

void Renderer::Impl::putTraceFrameDescriptors(std::shared_ptr<TraceFrameResourcesAndDescriptors> && frameDescriptors)
{
    ASSERT_MSG(frameDescriptors.use_count() == 1, "Non-unique use in single-threaded context: {}", frameDescriptors.use_count());
    traceFrameResourcesAndDescriptorsPool.push(std::move(frameDescriptors));
}

}  // namespace viewer
