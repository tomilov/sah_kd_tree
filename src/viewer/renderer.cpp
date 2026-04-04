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
#include <engine/specialization_info.hpp>
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

#include <cmath>
#include <cstddef>
#include <cstdint>

using namespace std::string_literals;
using namespace std::string_view_literals;

namespace viewer
{
namespace
{

constexpr glm::uint kGroupSizeX = 32;
constexpr glm::uint kGroupSizeY = 32;

constexpr glm::float32 kWireframeThickness = 1.0f;

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
    template<
        typename F,
        typename... Args>
    // NOLINTNEXTLINE: google-explicit-constructor
    Recycler(
        F && f,
        Args &&... args)
        : holder{makeHolder<
              F,
              Args...>(
              f,
              args...,
              std::index_sequence_for<Args...>{})}
    {}

    [[nodiscard]] operator Resource() && noexcept  // NOLINT: google-explicit-constructor
    {
        return std::move(holder);
    }

private:
    using Holder = std::unique_ptr<void, void (*)(void * p)>;

    Holder holder;  // NOLINT: readability-dentifier-naming

    template<
        typename F,
        typename... Args,
        size_t... Indices>
    [[nodiscard]] static Holder makeHolder(
        F & f,
        Args &... args,
        std::index_sequence<Indices...>)
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
};

using Fence = std::shared_ptr<vk::UniqueFence>;

[[nodiscard]] inline Fence makeFence(
    const engine::Context & context,
    vk::FenceCreateFlags flags = {})
{
    auto device = context.getDevice().getHandle();
    vk::FenceCreateInfo fenceCreateInfo = {
        .flags = flags,
    };
    return std::make_shared<vk::UniqueFence>(device.createFenceUnique(fenceCreateInfo, context.getAllocationCallbacks(), context.getDispatcher()));
}

inline void resetFence(
    const engine::Context & context,
    const Fence & fence)
{
    SKT_ASSERT(fence);
    SKT_ASSERT(*fence);
    context.getDevice().getHandle().resetFences(**fence, context.getDispatcher());
}

inline void waitFence(
    const engine::Context & context,
    const Fence & fence)
{
    SKT_ASSERT(fence);
    SKT_ASSERT(*fence);
    auto result = context.getDevice().getHandle().waitForFences(**fence, vk::True, std::numeric_limits<uint64_t>::max(), context.getDispatcher());
    SKT_INVARIANT(result == vk::Result::eSuccess, "Display fence: {}", result);
}

inline void checkFenceUnique(const Fence & fence)
{
    SKT_ASSERT_MSG(fence.use_count() == 1, "Non-unique use in single-threaded context: {}", fence.use_count());
}

class FencePool final : utils::NonCopyable
{
public:
    explicit FencePool(const engine::Context & contextIn)
        : context{contextIn}
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
    glm::float32 wireframeThickness = 0.0f;
    glm::vec3 position{0.0f};
    glm::float32 width = 0.0f;
    glm::float32 height = 0.0f;
    glm::float32 zNear = 0.0f;
    glm::float32 zFar = 0.0f;
    glm::float32 alpha = 0.0f;
    glm::mat4 windowMvp{1.0f};
};
static_assert(std::is_standard_layout_v<UniformBuffer>);
static_assert(std::is_trivially_copyable_v<UniformBuffer>);

struct ScenePushConstants
{
    glm::mat4 mvp{1.0f};
};
static_assert(std::is_standard_layout_v<ScenePushConstants>);
static_assert(std::is_trivially_copyable_v<ScenePushConstants>);

struct DisplayPushConstants
{
    float x = 1E-5f;
};
static_assert(std::is_standard_layout_v<DisplayPushConstants>);
static_assert(std::is_trivially_copyable_v<DisplayPushConstants>);

struct TreeUniformBuffer
{
    glm::uint triangleCount;
    glm::uint treeDepthMax;
    glm::uint polygonCount;
    glm::uint nodeCount;
    vk::DeviceAddress indices;
    vk::DeviceAddress vertices;
    vk::DeviceAddress polygons;
    vk::DeviceAddress nodes;
    vk::DeviceAddress nodeParents;
};
static_assert(std::is_standard_layout_v<TreeUniformBuffer>);
static_assert(std::is_trivially_copyable_v<TreeUniformBuffer>);

struct Frustum
{
    glm::vec3 lt;
    glm::vec3 rt;
    glm::vec3 lb;
    glm::vec3 rb;
};
static_assert(std::is_standard_layout_v<Frustum>);
static_assert(std::is_trivially_copyable_v<Frustum>);

struct TracePushConstants
{
    glm::vec4 clearColor;
    glm::float32 tNear;
    glm::vec3 pos;
    glm::uint nodeIndex;
    Frustum frustum;
    glm::vec2 viewportSize;
    glm::float32 wireframeThickness;
    glm::vec4 errorColor;
};
static_assert(std::is_standard_layout_v<TracePushConstants>);
static_assert(std::is_trivially_copyable_v<TracePushConstants>);

#pragma pack(pop)

struct UniformBufferResource final
{
    engine::Buffer<UniformBuffer> uniformBuffer;

    [[nodiscard]] static engine::DescriptorBindingNameAndType getBindingName()
    {
        return {"uniformBuffer"s, vk::DescriptorType::eUniformBuffer};
    }

    [[nodiscard]] DescriptorInfo getDescriptorInfo(engine::DescriptorManagementKind descriptorManagementKind) const
    {
        const auto getDescriptorData = [this, descriptorManagementKind]
        {
            switch (descriptorManagementKind) {
            case engine::DescriptorManagementKind::Sets: {
                return DescriptorData{std::in_place_type<DescriptorSetData>, uniformBuffer.getDescriptorBufferInfo()};
            }
            case engine::DescriptorManagementKind::Buffer: {
                return DescriptorData{std::in_place_type<DescriptorBufferData>, uniformBuffer.getDescriptorAddressInfo()};
            }
            case engine::DescriptorManagementKind::Heap: {
                return DescriptorData{std::in_place_type<DescriptorHeapData>};
            }
            }
        };
        return {getBindingName(), getDescriptorData()};
    }
};

struct TraceSceneResources final
{
    Tree tree;
    engine::Buffer<TreeUniformBuffer> treeUniformBuffer;

    [[nodiscard]] static engine::DescriptorBindingNameAndType getBindingName()
    {
        return {""s, vk::DescriptorType::eUniformBuffer};
    }

    [[nodiscard]] DescriptorInfo getDescriptorInfo(engine::DescriptorManagementKind descriptorManagementKind) const
    {
        const auto getDescriptorData = [this, descriptorManagementKind]
        {
            switch (descriptorManagementKind) {
            case engine::DescriptorManagementKind::Sets: {
                return DescriptorData{std::in_place_type<DescriptorSetData>, treeUniformBuffer.getDescriptorBufferInfo()};
            }
            case engine::DescriptorManagementKind::Buffer: {
                return DescriptorData{std::in_place_type<DescriptorBufferData>, treeUniformBuffer.getDescriptorAddressInfo()};
            }
            case engine::DescriptorManagementKind::Heap: {
                return DescriptorData{std::in_place_type<DescriptorHeapData>};
            }
            }
        };
        return {getBindingName(), getDescriptorData()};
    }
};

struct FrameResourcesAndDescriptors
{
    UniformBufferResource resources;
    Descriptors directDescriptors;
    std::optional<Descriptors> displayDescriptors;

    FrameResourcesAndDescriptors(
        UniformBufferResource && resourcesIn,
        Descriptors && sceneDescriptorsIn,
        std::optional<Descriptors> && displayDescriptorsIn)
        : resources{std::move(resourcesIn)}
        , directDescriptors{std::move(sceneDescriptorsIn)}
        , displayDescriptors{std::move(displayDescriptorsIn)}
    {}
};

struct TraceSceneResourcesAndDescriptors
{
    TraceSceneResources resources;
    Descriptors descriptors;

    TraceSceneResourcesAndDescriptors(
        TraceSceneResources && resourcesIn,
        Descriptors && descriptorsIn)
        : resources{std::move(resourcesIn)}
        , descriptors{std::move(descriptorsIn)}
    {}
};

struct TraceFrameResourcesAndDescriptors
{
    TraceFrameResources resources;
    Descriptors writeDescriptors;
    Descriptors readDescriptors;

    TraceFrameResourcesAndDescriptors(
        TraceFrameResources && resourcesIn,
        Descriptors && writeDescriptorsIn,
        Descriptors && readDescriptorsIn)
        : resources{std::move(resourcesIn)}
        , writeDescriptors{std::move(writeDescriptorsIn)}
        , readDescriptors{std::move(readDescriptorsIn)}
    {}
};

struct SceneResourcesAndDescriptors
{
    SceneResources resources;
    Descriptors descriptors;

    SceneResourcesAndDescriptors(
        SceneResources && resourcesIn,
        Descriptors && descriptorsIn)
        : resources{std::move(resourcesIn)}
        , descriptors{std::move(descriptorsIn)}
    {}
};

struct DrawOffscreenResourcesAndDescriptors
{
    DrawOffscreenResources resources;
    Descriptors descriptors;

    Fence fence;
    std::shared_ptr<const engine::CommandBuffers> commandBuffers;

    DrawOffscreenResourcesAndDescriptors(
        DrawOffscreenResources && resourcesIn,
        Descriptors && descriptorsIn)
        : resources{std::move(resourcesIn)}
        , descriptors{std::move(descriptorsIn)}
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
    DrawOffscreenPool(
        Private,
        const engine::Context & contextIn,
        const Engine & engineIn,
        std::shared_ptr<const vk::UniqueSampler> && samplerIn)
        : context{contextIn}
        , engine{engineIn}
        , displayRenderPass{OffscreenRenderPass::make(context)}
        , displayGraphicsPipeline{makeGraphicsPipeline()}
        , sampler{std::move(samplerIn)}
    {}

    [[nodiscard]] static std::shared_ptr<DrawOffscreenPool> make(
        const engine::Context & context,
        const Engine & engine,
        std::shared_ptr<const vk::UniqueSampler> sampler)
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

    [[nodiscard]] std::shared_ptr<DrawOffscreenResourcesAndDescriptors> get(
        const vk::Extent2D & framebufferSize,
        std::shared_ptr<const engine::ShaderStages> shaderStages) &
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
            auto descriptorInfos = {resources.getDescriptorInfo(engine.getSettings().descriptorManagementKind)};
            descriptors.fill(descriptorInfos);
            return std::make_shared<DrawOffscreenResourcesAndDescriptors>(std::move(resources), std::move(descriptors));
        }
        DrawOffscreenResources resources{context, framebufferSize, displayRenderPass, sampler};
        auto descriptors = engine.makeDescriptors("display"sv, std::move(shaderStages), resources);
        return std::make_shared<DrawOffscreenResourcesAndDescriptors>(std::move(resources), std::move(descriptors));
    }

    void put(std::shared_ptr<DrawOffscreenResourcesAndDescriptors> resourcesAndDescriptors) &
    {
        SKT_ASSERT_MSG(resourcesAndDescriptors.use_count() == 1, "Non-unique use in single-threaded context: {}", resourcesAndDescriptors.use_count());
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
        graphicsPipeline.initPipeline("offscreen scene"sv, context, engine.getPipelines().getPipelineCache(), engine.getSettings().descriptorManagementKind, displayRenderPass, {}).create();
        return graphicsPipeline;
    }
};

class ScopedCommandBuffer final : utils::OneTime<ScopedCommandBuffer>
{
public:
    explicit ScopedCommandBuffer(
        std::string_view nameIn,
        const engine::Context & contextIn,
        const engine::Queue & queueIn)
        : name{nameIn}
        , context{contextIn}
        , queue{queueIn}
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
        queue.submit(submitInfo, completionFence ? **completionFence : nullptr);

        if (waitIdle) {
            if (completionFence) {
                SKT_ASSERT(*completionFence);
                auto result = context.getDevice().getHandle().waitForFences(**completionFence, vk::True, std::numeric_limits<uint64_t>::max(), context.getDispatcher());
                SKT_INVARIANT(result == vk::Result::eSuccess, "{}: {}", name, result);
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

    [[nodiscard]] operator vk::CommandBuffer() const &  // NOLINT: google-explicit-constructor
    {
        return getCommandBuffer();
    }

    void setCompletionFence(Fence completionFenceIn)
    {
        completionFence = std::move(completionFenceIn);
    }

    void setWaitCompletion(bool waitIdleIn = true)
    {
        waitIdle = waitIdleIn;
    }

    void setWaitCompletion(Fence completionFenceIn)
    {
        setCompletionFence(completionFenceIn);
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
        .wireframeThickness = frameSettings.wireframe ? kWireframeThickness : 0.0f,
        .position = frameSettings.position,
        .width = frameSettings.width,
        .height = frameSettings.height,
        .zNear = frameSettings.zNear,
        .zFar = frameSettings.zFar,
        .alpha = frameSettings.alpha,
        .windowMvp = frameSettings.windowMvp,
    };
}

TreeUniformBuffer getTreeUniformBuffer(const Tree & tree)
{
    return {
        .triangleCount = utils::autoCast(tree.getTriangleCount()),
        .treeDepthMax = utils::autoCast(std::size(tree.getLayerSizes())),
        .polygonCount = utils::autoCast(tree.getPolygonCount()),
        .nodeCount = utils::autoCast(tree.getNodeCount()),
        .indices = tree.getIndexAddress(),
        .vertices = tree.getVertexAddress(),
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
    const glm::float32 width = utils::autoCast(frameSettings.width);
    const glm::float32 height = utils::autoCast(frameSettings.height);
    return {
        .clearColor = frameSettings.clearColor,
        .tNear = 0.0f,
        .pos = frameSettings.position,
        .nodeIndex = 0,  // TODO: O(logN) -> O(1) on movies
        .frustum = {
            .lt = leftTop,
            .rt = rightTop,
            .lb = leftBottom,
            .rb = rightBottom,
        },
        .viewportSize = glm::vec2{width, height},
        .wireframeThickness = frameSettings.wireframe ? kWireframeThickness : 0.0f,
        .errorColor = glm::vec4{1.0f, 0.0f, 0.0f, 1.0f},
    };
}

}  // namespace

vk::Extent2D FrameSettings::getFramebufferSize() const
{
    const float w = std::ceil(width);
    const float h = std::ceil(height);
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
    std::shared_ptr<DrawOffscreenResourcesAndDescriptors> offscreenResourcesAndDescriptors;

    // revocation lists should be the last members
    std::vector<std::vector<Resource>> deferredDeletionSlots{framesInFlight};

    Impl(
        std::string_view name,
        const engine::Context & context,
        const Engine & engine,
        uint32_t framesInFlight);

    [[nodiscard]] std::shared_ptr<const vk::UniqueSampler> makeSampler() const;

    void setFrameSettings(const FrameSettings & frameSettings);

    void unsetScene();
    void setScene(scene_data::SceneDataPtr sceneData);

    void setTree(builder::Tree && builderTree);
    void unsetTree();

    [[nodiscard]] ComputePipeline makeTraceComputePipeline(std::shared_ptr<const Shaders> shaders) const;

    void bindPipeline(
        vk::CommandBuffer commandBuffer,
        vk::PipelineBindPoint pipelineBindPoint,
        const Shaders & shaders,
        DescriptorRefs descriptors,
        const std::byte * pushConstants) const;

    template<typename Pipeline>
    void bindPipeline(
        vk::CommandBuffer commandBuffer,
        const Pipeline & pipeline,
        DescriptorRefs descriptors,
        const std::byte * pushConstants) const
    {
        commandBuffer.bindPipeline(Pipeline::kPipelineBindPoint, *pipeline.pipeline, context.getDispatcher());
        bindPipeline(commandBuffer, Pipeline::kPipelineBindPoint, *pipeline.shaders, descriptors, pushConstants);
    }

    void drawScene(
        vk::CommandBuffer commandBuffer,
        const GraphicsPipeline & pipeline) const;
    void offscreenPass(
        vk::CommandBuffer commandBuffer,
        vk::RenderPass renderPass);
    void drawDisplay(
        vk::CommandBuffer commandBuffer,
        const GraphicsPipeline & pipeline);
    void traceScene(
        vk::CommandBuffer graphicsCommandBuffer,
        const ComputePipeline & pipeline);

    void advance(
        vk::CommandBuffer commandBuffer,
        uint32_t currentFrameSlot);

    void updateRenderPass(
        vk::RenderPass renderPass,
        bool isRenderPassFormatChanged,
        uint32_t currentFrameSlot);

    void render(
        vk::CommandBuffer commandBuffer,
        vk::RenderPass renderPass,
        bool isRenderPassFormatChanged,
        uint32_t currentFrameSlot);

    [[nodiscard]] std::shared_ptr<FrameResourcesAndDescriptors> getFrameDescriptors();
    void putFrameDescriptors(std::shared_ptr<FrameResourcesAndDescriptors> && frameDescriptors);

    [[nodiscard]] std::shared_ptr<TraceFrameResourcesAndDescriptors> getTraceFrameDescriptors();
    void putTraceFrameDescriptors(std::shared_ptr<TraceFrameResourcesAndDescriptors> && frameDescriptors);

    template<typename... Resources>
    void deferDeletion(
        uint32_t frameSlot,
        Resources &&... resources)
    {
        auto & slotResources = deferredDeletionSlots.at(frameSlot);
        (slotResources.emplace_back(std::forward<Resources>(resources)), ...);
    }

    void deleteDeferred(uint32_t currentFrameSlot)
    {
        deferredDeletionSlots.at(currentFrameSlot).clear();
    }
};

Renderer::Renderer(
    std::string_view name,
    const engine::Context & context,
    const Engine & engine,
    uint32_t framesInFlight)
    : impl_{std::make_unique<Impl>(
          name,
          context,
          engine,
          framesInFlight)}
{}

uint32_t Renderer::getFramesInFlight() const
{
    return impl_->framesInFlight;
}

Renderer::~Renderer() = default;

void Renderer::setFrameSettings(const FrameSettings & frameSettings)
{
    impl_->setFrameSettings(frameSettings);
}

void Renderer::setScene(scene_data::SceneDataPtr sceneData)
{
    impl_->setScene(std::move(sceneData));
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
    if (builderTree) {
        impl_->setTree(std::move(*builderTree));
        return;
    }
    impl_->unsetTree();
}

void Renderer::advance(
    vk::CommandBuffer commandBuffer,
    uint32_t currentFrameSlot)
{
    impl_->advance(commandBuffer, currentFrameSlot);
}

void Renderer::render(
    vk::CommandBuffer commandBuffer,
    vk::RenderPass renderPass,
    bool isRenderPassFormatChanged,
    uint32_t currentFrameSlot)
{
    impl_->render(commandBuffer, renderPass, isRenderPassFormatChanged, currentFrameSlot);
}

Renderer::Impl::Impl(
    std::string_view nameIn,
    const engine::Context & contextIn,
    const Engine & engineIn,
    uint32_t framesInFlightIn)
    : name{nameIn}
    , context{contextIn}
    , engine{engineIn}
    , framesInFlight{framesInFlightIn}
{
    uint32_t maxPushConstantsSize = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits.maxPushConstantsSize;
    SKT_INVARIANT(sizeof(ScenePushConstants) <= maxPushConstantsSize, "{} ^ {}", sizeof(ScenePushConstants), maxPushConstantsSize);
}

std::shared_ptr<const vk::UniqueSampler> Renderer::Impl::makeSampler() const
{
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
        .maxAnisotropy = 1.0f,  // 1.0f..maxSamplerAnisotropy
        .compareEnable = vk::False,
        .compareOp = vk::CompareOp::eNever,
        .minLod = 0.0f,
        .maxLod = 0.0f,
        .borderColor = vk::BorderColor::eFloatTransparentBlack,
        .unnormalizedCoordinates = vk::False,
    };
    return std::make_shared<vk::UniqueSampler>(context.getDevice().getHandle().createSamplerUnique(samplerCreateInfo, context.getAllocationCallbacks(), context.getDispatcher()));
}

void Renderer::Impl::setFrameSettings(const FrameSettings & frameSettingsIn)
{
    frameSettings = frameSettingsIn;
}

void Renderer::Impl::unsetScene()
{
    sceneData.reset();
}

void Renderer::Impl::setScene(scene_data::SceneDataPtr newSceneData)
{
    SKT_ASSERT(!sceneData);
    SKT_ASSERT(newSceneData);
    sceneData = std::move(newSceneData);
}

void Renderer::Impl::setTree(builder::Tree && builderTree)
{
    unsetTree();

    Tree tree{name, context, std::move(builderTree)};

    engine::Buffer<TreeUniformBuffer> treeUniformBuffer{engine.createUniformBuffer(sizeof(TreeUniformBuffer))};
    treeUniformBuffer.map().at(0) = getTreeUniformBuffer(tree);

    auto shaders = engine.getPipelines().getTraceSahKdTreeShaders();

    TraceSceneResources traceSceneResources = {
        .tree = std::move(tree),
        .treeUniformBuffer = std::move(treeUniformBuffer),
    };
    auto descriptors = engine.makeDescriptors("trace"sv, shaders->getShaderStagesPtr(), traceSceneResources);
    traceSceneResourcesAndDescriptors = std::make_shared<TraceSceneResourcesAndDescriptors>(std::move(traceSceneResources), std::move(descriptors));
    SPDLOG_INFO("{}: Tree is set", name);
}

void Renderer::Impl::unsetTree()
{
    if (!traceSceneResourcesAndDescriptors) {
        return;
    }
    traceSceneResourcesAndDescriptors.reset();
    SPDLOG_INFO("{}: Tree is unset", name);
}

ComputePipeline Renderer::Impl::makeTraceComputePipeline(std::shared_ptr<const Shaders> shaders) const
{
    ComputePipeline computePipeline{std::move(shaders)};
    struct SpecializationData
    {
        const glm::uint kGroupSizeX;
        const glm::uint kGroupSizeY;
        const glm::float32 kUlp = std::nextafter(0.0f, 1.0f);
        const glm::float32 kEps = std::numeric_limits<glm::float32>::epsilon();
        const glm::float32 kInf = std::numeric_limits<glm::float32>::infinity();
    };
    const SpecializationData specializationData = {
        .kGroupSizeX = kGroupSizeX,
        .kGroupSizeY = kGroupSizeY,
    };
    const std::initializer_list<vk::SpecializationMapEntry> specializationMapEntries = {
        {
            .constantID = 0,
            .offset = offsetof(SpecializationData, kGroupSizeX),
            .size = sizeof(SpecializationData::kGroupSizeX),
        },
        {
            .constantID = 1,
            .offset = offsetof(SpecializationData, kGroupSizeY),
            .size = sizeof(SpecializationData::kGroupSizeY),
        },
        {
            .constantID = 2,
            .offset = offsetof(SpecializationData, kUlp),
            .size = sizeof(SpecializationData::kUlp),
        },
        {
            .constantID = 3,
            .offset = offsetof(SpecializationData, kEps),
            .size = sizeof(SpecializationData::kEps),
        },
        {
            .constantID = 4,
            .offset = offsetof(SpecializationData, kInf),
            .size = sizeof(SpecializationData::kInf),
        },
    };
    engine::SpecializationInfos specializationInfos;
    specializationInfos.try_emplace(vk::ShaderStageFlagBits::eCompute, std::make_unique<SpecializationData>(specializationData), specializationMapEntries);
    auto & pipeline = computePipeline.initPipeline("trace"sv, context, engine.getPipelines().getPipelineCache(), engine.getSettings().descriptorManagementKind, std::move(specializationInfos));
    pipeline.create();
    return computePipeline;
}

void Renderer::Impl::bindPipeline(
    vk::CommandBuffer commandBuffer,
    vk::PipelineBindPoint pipelineBindPoint,
    const Shaders & shaders,
    DescriptorRefs descriptors,
    const std::byte * pushConstants) const
{
    constexpr uint32_t kFirstSet = 0;
    vk::PipelineLayout pipelineLayout = shaders.getPipelineLayout();
    switch (engine.getSettings().descriptorManagementKind) {
    case engine::DescriptorManagementKind::Sets: {
        std::vector<vk::DescriptorSet> descriptorSets;
        descriptorSets.reserve(std::size(descriptors));
        for (const Descriptors & d : descriptors) {
            descriptorSets.push_back(d.getDescriptorSet());
        }
        constexpr auto kDynamicOffsets = nullptr;
        commandBuffer.bindDescriptorSets(pipelineBindPoint, pipelineLayout, kFirstSet, descriptorSets, kDynamicOffsets, context.getDispatcher());
        break;
    }
    case engine::DescriptorManagementKind::Buffer: {
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
        break;
    }
    case engine::DescriptorManagementKind::Heap: {
        // TODO:
        break;
    }
    }

    for (const auto & pushConstantRange : shaders.getShaderStages().pushConstantRanges) {
        if (engine.getSettings().descriptorManagementKind == engine::DescriptorManagementKind::Heap) {
            [[maybe_unused]] const auto & descriptorHeapProperties = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceDescriptorHeapPropertiesEXT>();
            SKT_ASSERT(pushConstantRange.offset + pushConstantRange.size <= descriptorHeapProperties.maxPushDataSize);
            vk::HostAddressRangeConstEXT data = {
                .address = pushConstants,
                .size = pushConstantRange.size,
            };
            vk::PushDataInfoEXT pushDataInfo = {
                .offset = pushConstantRange.offset,
                .data = data,
            };
            commandBuffer.pushDataEXT(pushDataInfo, context.getDispatcher());
        } else {
            commandBuffer.pushConstants(pipelineLayout, pushConstantRange.stageFlags, pushConstantRange.offset, pushConstantRange.size, std::next(pushConstants, pushConstantRange.offset), context.getDispatcher());
        }
    }
}

void Renderer::Impl::drawScene(
    vk::CommandBuffer commandBuffer,
    const GraphicsPipeline & pipeline) const
{
    {
        SKT_ASSERT(frameResourcesAndDescriptors);
        SKT_ASSERT(sceneResourcesAndDescriptors);
        const DescriptorRefs descriptors = {
            std::cref(frameResourcesAndDescriptors->directDescriptors),
            std::cref(sceneResourcesAndDescriptors->descriptors),
        };
        const ScenePushConstants pushConstants = getScenePushConstants(frameSettings);
        bindPipeline(commandBuffer, pipeline, descriptors, utils::autoCast(&pushConstants));
    }

    constexpr engine::LabelColor kMagentaColor = {1.0f, 0.0f, 1.0f, 1.0f};
    auto drawSceneLabel = engine::ScopedCommandBufferLabel::create(context.getDispatcher(), commandBuffer, "Draw scene"sv, kMagentaColor);

    {
        vk::Viewport viewport;
        vk::Rect2D scissor;
        if (frameSettings.useOffscreenTexture) {
            SKT_ASSERT(offscreenResourcesAndDescriptors);
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
    }

    SKT_ASSERT(sceneResourcesAndDescriptors);
    const auto & sceneResources = sceneResourcesAndDescriptors->resources;

    {
        constexpr uint32_t kFirstBinding = 0;
        const auto bufferOrNull = [this](const auto & wrapper) -> vk::Buffer
        {
            if (wrapper) {
                return wrapper.value();
            }
            SKT_ASSERT(context.getPhysicalDevice().features2Chain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>().nullDescriptor != vk::False);
            SKT_ASSERT(context.getPhysicalDevice().features2Chain.get<vk::PhysicalDeviceVulkan14Features>().maintenance6 != vk::False);
            return nullptr;
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
        SKT_ASSERT(features2Chain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>().nullDescriptor != vk::False);
        SKT_ASSERT(features2Chain.get<vk::PhysicalDeviceVulkan14Features>().maintenance6 != vk::False);
        // TODO: or draw non-indexed
    }
    constexpr vk::DeviceSize kIndexBufferDeviceOffset = 0;
    if (engine.getSettings().multiDrawIndirectEnabled) {
        SKT_ASSERT(std::empty(sceneResources.indexTypes));
        auto indexType = sceneResources.maxIndexType;
        commandBuffer.bindIndexBuffer(indexBuffer, kIndexBufferDeviceOffset, indexType, context.getDispatcher());  // vkCmdBindIndexBuffer2KHR is not supported by Renderdoc
        constexpr vk::DeviceSize kInstanceBufferOffset = 0;
        constexpr uint32_t kStride = sizeof(vk::DrawIndexedIndirectCommand);
        uint32_t drawCount = sceneResources.drawCount;
        const auto & physicalDeviceLimits = context.getPhysicalDevice().properties2Chain.get<vk::PhysicalDeviceProperties2>().properties.limits;
        SKT_INVARIANT(drawCount <= physicalDeviceLimits.maxDrawIndirectCount, "{} ^ {}", drawCount, physicalDeviceLimits.maxDrawIndirectCount);
        if (engine.getSettings().drawIndirectCountEnabled) {
            constexpr vk::DeviceSize kDrawCountBufferOffset = 0;
            uint32_t maxDrawCount = drawCount;
            commandBuffer.drawIndexedIndirectCount(sceneResources.instanceBuffer.value(), kInstanceBufferOffset, sceneResources.drawCountBuffer.value(), kDrawCountBufferOffset, maxDrawCount, kStride, context.getDispatcher());
        } else {
            commandBuffer.drawIndexedIndirect(sceneResources.instanceBuffer.value(), kInstanceBufferOffset, drawCount, kStride, context.getDispatcher());
        }
    } else {
        SKT_ASSERT(!std::empty(sceneResources.instances));
        SKT_ASSERT(std::size(sceneResources.indexTypes) == std::size(sceneResources.instances));
        auto indexType = std::cbegin(sceneResources.indexTypes);
        for (const auto & [indexCount, instanceCount, firstIndex, vertexOffset, firstInstance] : sceneResources.instances) {
            SKT_ASSERT(indexType != std::cend(sceneResources.indexTypes));
            commandBuffer.bindIndexBuffer(indexBuffer, kIndexBufferDeviceOffset, *indexType++, context.getDispatcher());
            commandBuffer.drawIndexed(indexCount, instanceCount, firstIndex, vertexOffset, firstInstance, context.getDispatcher());
            // SPDLOG_TRACE("{{.indexCount = {}, .instanceCount = {}, .firstIndex = {}, .vertexOffset = {}, .firstInstance = {})}}", indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
        }
        SKT_ASSERT(indexType == std::cend(sceneResources.indexTypes));
    }
}

void Renderer::Impl::offscreenPass(
    vk::CommandBuffer commandBuffer,
    vk::RenderPass renderPass)
{
    constexpr engine::LabelColor kGreenColor = {0.0f, 1.0f, 0.0f, 1.0f};
    auto offscreenPassLabel = engine::ScopedCommandBufferLabel::create(context.getDispatcher(), commandBuffer, "Offscreen pass"sv, kGreenColor);

    SKT_ASSERT(offscreenResourcesAndDescriptors);
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

void Renderer::Impl::drawDisplay(
    vk::CommandBuffer commandBuffer,
    const GraphicsPipeline & pipeline)
{
    {
        SKT_ASSERT(frameResourcesAndDescriptors->displayDescriptors);
        const Descriptors * secondBinding = nullptr;
        if (traceFrameResourcesAndDescriptors) {
            SKT_ASSERT(!offscreenResourcesAndDescriptors);
            secondBinding = &traceFrameResourcesAndDescriptors->readDescriptors;
        } else {
            SKT_ASSERT(offscreenResourcesAndDescriptors);
            secondBinding = &offscreenResourcesAndDescriptors->descriptors;
        }
        SKT_ASSERT(secondBinding);
        const DescriptorRefs descriptors = {
            std::cref(frameResourcesAndDescriptors->displayDescriptors.value()),
            std::cref(*secondBinding),
        };
        const DisplayPushConstants displayPushConstants = getDisplayPushConstants(frameSettings);
        bindPipeline(commandBuffer, pipeline, descriptors, utils::autoCast(&displayPushConstants));
    }

    {
        constexpr uint32_t kFirstBinding = 0;
        constexpr vk::Buffer kVertexBuffer = nullptr;
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

void Renderer::Impl::traceScene(
    vk::CommandBuffer graphicsCommandBuffer,
    const ComputePipeline & pipeline)
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
            image.release(graphicsReleaseCommandBuffer, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite, TraceFrameResources::kInternalImageLayout, computeQueueFamilyIndex);
        }
        fencePool.put(std::move(fenceGraphics));
    }
    {
        auto fenceCompute = fencePool.get();
        {
            ScopedCommandBuffer computeCommandBuffer{"Offscreen scene trace"sv, context, computeQueue};
            computeCommandBuffer.setWaitCompletion(fenceCompute);

            {
                SKT_ASSERT(traceSceneResourcesAndDescriptors);
                const DescriptorRefs descriptors = {
                    std::cref(traceSceneResourcesAndDescriptors->descriptors),
                    std::cref(traceFrameResourcesAndDescriptors->writeDescriptors),
                };
                const TracePushConstants pushConstants = getTracePushConstants(frameSettings);
                bindPipeline(computeCommandBuffer, pipeline, descriptors, utils::autoCast(&pushConstants));
            }

            image.acquire(computeCommandBuffer, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite, TraceFrameResources::kInternalImageLayout, computeQueueFamilyIndex);
            {
                auto [width, height] = image.getExtent2D();
                width = utils::alignUp(width, kGroupSizeX);
                height = utils::alignUp(height, kGroupSizeY);
                constexpr uint32_t kDepth = 1;
                computeCommandBuffer.getCommandBuffer().dispatch(width, height, kDepth, context.getDispatcher());
            }
            image.release(computeCommandBuffer, vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead, TraceFrameResources::kExternalImageLayout, graphicsQueueFamilyIndex);
        }
        fencePool.put(std::move(fenceCompute));
    }
    image.acquire(graphicsCommandBuffer, vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead, TraceFrameResources::kExternalImageLayout, graphicsQueueFamilyIndex);
}

void Renderer::Impl::advance(
    vk::CommandBuffer commandBuffer,
    uint32_t currentFrameSlot)
{
    SKT_ASSERT_MSG(currentFrameSlot < framesInFlight, "{} ^ {}", currentFrameSlot, framesInFlight);

    auto unmuteMessageGuard = context.getInstance().unmuteDebugUtilsMessages(kUnmutedMessageIdNumbers);

    deleteDeferred(currentFrameSlot);

    uint32_t previousFrameSlot = utils::modDown(currentFrameSlot, framesInFlight);

    if (offscreenResourcesAndDescriptors) {
        Recycler recycler = [this, resourcesAndDescriptors = std::move(offscreenResourcesAndDescriptors), drawOffscreenPoolOld = drawOffscreenPool]() mutable
        {
            if (resourcesAndDescriptors->fence) {
                fencePool.waitAndPut(std::move(resourcesAndDescriptors->fence));
            }
            resourcesAndDescriptors->commandBuffers.reset();
            if (drawOffscreenPoolOld) {
                drawOffscreenPoolOld->put(std::move(resourcesAndDescriptors));
            } else {
                resourcesAndDescriptors.reset();
            }
        };
        deferDeletion(previousFrameSlot, std::move(recycler));
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
            const auto & graphicsPipeline = frameSettings.useOffscreenTexture ? drawOffscreenPool->getGraphicsPipeline() : *directGraphicsPipeline;
            auto resources = engine.makeResources(*sceneData);
            auto descriptors = engine.makeDescriptors("scene"sv, graphicsPipeline.shaders->getShaderStagesPtr(), resources);
            sceneResourcesAndDescriptors = std::make_shared<SceneResourcesAndDescriptors>(std::move(resources), std::move(descriptors));
        }
    } else {
        deferDeletion(previousFrameSlot, std::move(sceneResourcesAndDescriptors));
    }
    if (frameSettings.useOffscreenTexture) {
        SKT_ASSERT(!offscreenResourcesAndDescriptors);
        SKT_ASSERT(!traceFrameResourcesAndDescriptors);
        if (traceSceneResourcesAndDescriptors) {
            traceScene(commandBuffer, *traceComputePipeline);
        } else if (sceneData) {
            offscreenResourcesAndDescriptors = drawOffscreenPool->get(frameSettings.getFramebufferSize(), displayGraphicsPipeline->shaders->getShaderStagesPtr());
            {
                ScopedCommandBuffer offscreenCommandBuffer{"Offscreen scene draw"sv, context, graphicsQueue};
                const OffscreenRenderPass & offscreenRenderPass = drawOffscreenPool->getOffscreenRenderPass();
                offscreenPass(offscreenCommandBuffer, offscreenRenderPass);
                SKT_ASSERT(!offscreenResourcesAndDescriptors->fence);
                offscreenResourcesAndDescriptors->fence = fencePool.get();
                offscreenCommandBuffer.setCompletionFence(offscreenResourcesAndDescriptors->fence);
                SKT_ASSERT(!offscreenResourcesAndDescriptors->commandBuffers);
                offscreenResourcesAndDescriptors->commandBuffers = offscreenCommandBuffer.getCommandBuffers();
            }
        }
    }
}

void Renderer::Impl::updateRenderPass(
    vk::RenderPass renderPass,
    [[maybe_unused]] bool isRenderPassFormatChanged,
    uint32_t currentFrameSlot)
{
    SKT_ASSERT(directGraphicsPipeline);
    auto & graphicsPipeline = frameSettings.useOffscreenTexture ? *displayGraphicsPipeline : *directGraphicsPipeline;
    if (graphicsPipeline.pipeline) {
        if (graphicsPipeline.pipeline->getRenderPass() == renderPass) {
            return;
        }
        uint32_t previousFrameSlot = utils::modDown(currentFrameSlot, framesInFlight);
        deferDeletion(previousFrameSlot, std::move(graphicsPipeline.pipeline));
    }
    std::string_view graphicsPipelineName;
    if (frameSettings.useOffscreenTexture) {
        graphicsPipelineName = "offscreen display"sv;
    } else {
        graphicsPipelineName = "direct scene"sv;
    }
    auto & p = graphicsPipeline.initPipeline(graphicsPipelineName, context, engine.getPipelines().getPipelineCache(), engine.getSettings().descriptorManagementKind, renderPass, {});
    if (frameSettings.useOffscreenTexture) {
        p.pipelineInputAssemblyStateCreateInfo.setTopology(vk::PrimitiveTopology::eTriangleStrip);
    }
    p.create();
}

void Renderer::Impl::render(
    vk::CommandBuffer commandBuffer,
    vk::RenderPass renderPass,
    bool isRenderPassFormatChanged,
    uint32_t currentFrameSlot)
{
    SKT_ASSERT(currentFrameSlot < framesInFlight);
    auto unmuteMessageGuard = context.getInstance().unmuteDebugUtilsMessages(kUnmutedMessageIdNumbers);
    updateRenderPass(renderPass, isRenderPassFormatChanged, currentFrameSlot);
    if (frameSettings.useOffscreenTexture) {
        if (offscreenResourcesAndDescriptors && offscreenResourcesAndDescriptors->fence) {
            fencePool.waitAndPut(std::move(offscreenResourcesAndDescriptors->fence));
        }
        if (frameResourcesAndDescriptors && frameResourcesAndDescriptors->displayDescriptors && (offscreenResourcesAndDescriptors || traceFrameResourcesAndDescriptors)) {
            drawDisplay(commandBuffer, *displayGraphicsPipeline);
        }
    } else {
        SKT_ASSERT(directGraphicsPipeline->pipeline);
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
    SKT_ASSERT_MSG(frameDescriptors.use_count() == 1, "Non-unique use in single-threaded context: {}", frameDescriptors.use_count());
    frameResourcesAndDescriptorsPool.push(std::move(frameDescriptors));
}

auto Renderer::Impl::getTraceFrameDescriptors() -> std::shared_ptr<TraceFrameResourcesAndDescriptors>
{
    std::shared_ptr<TraceFrameResourcesAndDescriptors> resourcesAndDescriptors;
    while (!std::empty(traceFrameResourcesAndDescriptorsPool)) {
        resourcesAndDescriptors = std::move(traceFrameResourcesAndDescriptorsPool.top());
        traceFrameResourcesAndDescriptorsPool.pop();
        if (resourcesAndDescriptors->resources.image.getExtent2D() == frameSettings.getFramebufferSize()) {
            return resourcesAndDescriptors;
        }
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
    SKT_ASSERT_MSG(frameDescriptors.use_count() == 1, "Non-unique use in single-threaded context: {}", frameDescriptors.use_count());
    traceFrameResourcesAndDescriptorsPool.push(std::move(frameDescriptors));
}

}  // namespace viewer

template struct utils::OneTime<viewer::Recycler>::CheckTraits;
template struct utils::OneTime<viewer::UniformBufferResource>::CheckTraits;
template struct utils::OneTime<viewer::TraceSceneResources>::CheckTraits;
template struct utils::OneTime<viewer::ScopedCommandBuffer>::CheckTraits;
