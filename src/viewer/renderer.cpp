#include <common/version.hpp>
#include <engine/context.hpp>
#include <engine/debug_utils.hpp>
#include <engine/device.hpp>
#include <engine/graphics_pipeline.hpp>
#include <engine/instance.hpp>
#include <engine/library.hpp>
#include <engine/physical_device.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/checked_ptr.hpp>
#include <viewer/renderer.hpp>
#include <viewer/scene_manager.hpp>

#include <fmt/std.h>
#include <glm/gtx/quaternion.hpp>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <QtCore/QByteArray>
#include <QtCore/QDebug>
#include <QtCore/QFile>
#include <QtCore/QIODevice>
#include <QtCore/QLoggingCategory>
#include <QtCore/QRunnable>
#include <QtCore/QString>
#include <QtGui/QVulkanInstance>
#include <QtQuick/QSGRendererInterface>

#include <algorithm>
#include <filesystem>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <string_view>
#include <vector>

#include <cstddef>
#include <cstdint>

using namespace Qt::StringLiterals;

namespace viewer
{
namespace
{
// clang-format off
[[maybe_unused]] Q_DECLARE_LOGGING_CATEGORY(viewerRendererCategory)
Q_LOGGING_CATEGORY(viewerRendererCategory, "viewer.renderer")
// clang-format on
}  // namespace

struct Renderer::Impl
{
    const std::string token;
    const std::filesystem::path scenePath;
    const engine::Context & context;
    const SceneManager & sceneManager;

    const engine::Library & library = context.getLibrary();
    const engine::Instance & instance = context.getInstance();
    const engine::Device & device = context.getDevice();

    std::shared_ptr<const Scene> scene;
    std::unique_ptr<const Scene::Descriptors> descriptors;
    std::unique_ptr<const Scene::GraphicsPipeline> graphicsPipeline;

    UniformBuffer uniformBuffer;
    vk::Viewport viewport;
    vk::Rect2D scissor;
    PushConstants pushConstants;

    Impl(std::string_view token, const std::filesystem::path & scenePath, const engine::Context & context, const SceneManager & sceneManager) : token{token}, scenePath{scenePath}, context{context}, sceneManager{sceneManager}
    {
        init();
    }

    void setOrientation(glm::quat orientation)
    {
        uniformBuffer.orientation = glm::toMat3(orientation);
    }

    void setT(float t)
    {
        uniformBuffer.t = t;
    }

    void setAlpha(qreal alpha)
    {
        uniformBuffer.alpha = utils::autoCast(alpha);
    }

    void setViewportRect(const QRectF & viewportRect)
    {
        auto x = std::ceil(viewportRect.x());
        auto y = std::ceil(viewportRect.y());
        auto width = std::floor(viewportRect.width());
        auto height = std::floor(viewportRect.height());

        viewport = vk::Viewport{
            .x = utils::autoCast(x),
            .y = utils::autoCast(y),
            .width = utils::autoCast(width),
            .height = utils::autoCast(height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        scissor = vk::Rect2D{
            .offset = {
                .x = utils::autoCast(x),
                .y = utils::autoCast(y),
            },
            .extent = {
                .width = utils::autoCast(width),
                .height = utils::autoCast(height),
            },
        };
    }

    void setViewTransform(const glm::dmat3 & viewTransform)
    {
        pushConstants = {
            .viewTransform = glm::mat3{viewTransform},
        };
    }

    [[nodiscard]] const std::filesystem::path & getScenePath() const
    {
        return scenePath;
    }

    void frameStart(const QQuickWindow::GraphicsStateInfo & graphicsStateInfo);
    void render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo);

    void init()
    {
        INVARIANT(!std::empty(token), "token should not be empty");
        INVARIANT(!std::empty(scenePath), "scenePath should not be empty");
    }
};

Renderer::Renderer(std::string_view token, const std::filesystem::path & scenePath, const engine::Context & context, const SceneManager & sceneManager) : impl_{token, scenePath, context, sceneManager}
{}

void Renderer::setOrientation(glm::quat orientation)
{
    return impl_->setOrientation(orientation);
}

Renderer::~Renderer() = default;

void Renderer::setT(float t)
{
    return impl_->setT(t);
}

void Renderer::setAlpha(qreal t)
{
    return impl_->setAlpha(t);
}

void Renderer::setViewportRect(const QRectF & viewportRect)
{
    return impl_->setViewportRect(viewportRect);
}

void Renderer::setViewTransform(const glm::dmat3 & viewTransform)
{
    return impl_->setViewTransform(viewTransform);
}

const std::filesystem::path & Renderer::getScenePath() const
{
    return impl_->getScenePath();
}

void Renderer::frameStart(const QQuickWindow::GraphicsStateInfo & graphicsStateInfo)
{
    return impl_->frameStart(graphicsStateInfo);
}

void Renderer::render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo)
{
    return impl_->render(commandBuffer, renderPass, graphicsStateInfo);
}

void Renderer::Impl::frameStart(const QQuickWindow::GraphicsStateInfo & graphicsStateInfo)
{
    auto unmuteMessageGuard = context.unmuteDebugUtilsMessages({0x5C0EC5D6, 0xE4D96472});

    uint32_t framesInFlight = utils::autoCast(graphicsStateInfo.framesInFlight);
    bool dirty = false;
    if (scene) {
        const auto & old = *scene->getSceneDesignator();
        if (old.framesInFlight != framesInFlight) {
            dirty = true;
            SPDLOG_INFO("framesInFlight changed: {} -> {}", old.framesInFlight, framesInFlight);
        }
    }
    if (dirty) {
        graphicsPipeline.reset();
        descriptors.reset();
        scene.reset();
    }
    if (!scene) {
        SceneDesignator sceneDesignator = {
            .token = token,
            .path = scenePath,
            .framesInFlight = framesInFlight,
        };
        scene = sceneManager.getOrCreateScene(std::move(sceneDesignator));
        if (!scene) {
            return;
        }

        descriptors = scene->makeDescriptors();
    }

    uint32_t currentFrameSlot = utils::autoCast(graphicsStateInfo.currentFrameSlot);
    *descriptors->uniformBuffers.at(currentFrameSlot).map<UniformBuffer>().data() = uniformBuffer;
}

void Renderer::Impl::render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo)
{
    std::initializer_list<uint32_t> unmutedMessageIdNumbers = {
        0x5C0EC5D6,
        0xE4D96472,
    };
    auto unmuteMessageGuard = context.unmuteDebugUtilsMessages(unmutedMessageIdNumbers);

    engine::LabelColor labelColor = {0.0f, 1.0f, 0.0f, 1.0f};
    auto commandBufferLabel = engine::ScopedCommandBufferLabel::create(library.dispatcher, commandBuffer, "Renderer::render", labelColor);

    if (!scene) {
        return;
    }
    ASSERT(descriptors);
    if (!graphicsPipeline || (graphicsPipeline->pipelineLayout.renderPass != renderPass)) {
        graphicsPipeline.reset();
        graphicsPipeline = scene->createGraphicsPipeline(renderPass);
    }
    auto pipeline = graphicsPipeline->pipelines.pipelines.at(0);
    auto pipelineLayout = graphicsPipeline->pipelineLayout.pipelineLayout;

    constexpr engine::LabelColor kMagentaColor = {1.0f, 0.0f, 1.0f, 1.0f};
    auto rasterizationLabel = engine::ScopedCommandBufferLabel::create(library.dispatcher, commandBuffer, "Rasterization", kMagentaColor);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline, library.dispatcher);

    constexpr uint32_t kFirstSet = 0;
    uint32_t currentFrameSlot = utils::autoCast(graphicsStateInfo.currentFrameSlot);
    if (scene->isDescriptorBufferUsed()) {
        const auto & descriptorSetBuffers = descriptors->descriptorSetBuffers;
        if (!std::empty(descriptorSetBuffers)) {
            commandBuffer.bindDescriptorBuffersEXT(descriptors->descriptorBufferBindingInfos, library.dispatcher);
            std::vector<uint32_t> bufferIndices(std::size(descriptorSetBuffers));
            std::iota(std::begin(bufferIndices), std::end(bufferIndices), bufferIndices.front());
            std::vector<vk::DeviceSize> offsets;
            offsets.reserve(std::size(descriptorSetBuffers));
            uint32_t framesInFlight = utils::autoCast(graphicsStateInfo.framesInFlight);
            INVARIANT(framesInFlight > 0, "");
            for (const auto & descriptorSetBuffer : descriptorSetBuffers) {
                offsets.push_back((descriptorSetBuffer.getSize() / framesInFlight) * currentFrameSlot);
            }
            commandBuffer.setDescriptorBufferOffsetsEXT(vk::PipelineBindPoint::eGraphics, pipelineLayout, kFirstSet, bufferIndices, offsets, library.dispatcher);
        }
    } else {
        constexpr auto kDynamicOffsets = nullptr;
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, kFirstSet, descriptors->descriptorSets.at(currentFrameSlot).descriptorSets, kDynamicOffsets, library.dispatcher);
    }

    {
        for (const auto & pushConstantRange : descriptors->pushConstantRanges) {
            const void * p = utils::safeCast<const std::byte *>(&pushConstants) + pushConstantRange.offset;
            commandBuffer.pushConstants(pipelineLayout, pushConstantRange.stageFlags, pushConstantRange.offset, pushConstantRange.size, p, library.dispatcher);
        }
    }

    constexpr uint32_t kFirstViewport = 0;
    std::initializer_list<vk::Viewport> viewports = {
        viewport,
    };
    commandBuffer.setViewport(kFirstViewport, viewports, library.dispatcher);

    constexpr uint32_t kFirstScissor = 0;
    std::initializer_list<vk::Rect2D> scissors = {
        scissor,
    };
    commandBuffer.setScissor(kFirstScissor, scissors, library.dispatcher);

    constexpr uint32_t kFirstBinding = 0;
    std::initializer_list<vk::Buffer> vertexBuffers = {
        descriptors->vertexBuffer,
    };
    if (sah_kd_tree::kIsDebugBuild) {
        for (const auto & vertexBuffer : vertexBuffers) {
            if (!vertexBuffer) {
                ASSERT(device.physicalDevice.physicalDeviceFeatures2Chain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>().nullDescriptor == VK_TRUE);
            }
        }
    }
    std::vector<vk::DeviceSize> vertexBufferOffsets(std::size(vertexBuffers), 0);
    commandBuffer.bindVertexBuffers(kFirstBinding, vertexBuffers, vertexBufferOffsets, library.dispatcher);

    // TODO: Scene::useDrawIndexedIndirect
    vk::Buffer indexBuffer = descriptors->indexBuffer;
    constexpr vk::DeviceSize kBufferDeviceOffset = 0;
    auto indexType = std::cbegin(descriptors->indexTypes);
    for (const auto & [indexCount, instanceCount, firstIndex, vertexOffset, firstInstance] : descriptors->instances) {
        INVARIANT(indexType != std::cend(descriptors->indexTypes), "");
        commandBuffer.bindIndexBuffer(indexBuffer, kBufferDeviceOffset, *indexType++, library.dispatcher);
        commandBuffer.drawIndexed(indexCount, instanceCount, firstIndex, vertexOffset, firstInstance, library.dispatcher);
        // SPDLOG_TRACE("{{.indexCount = {}, .instanceCount = {}, .firstIndex = {}, .vertexOffset = {}, .firstInstance = {})}}", indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
    }
}

}  // namespace viewer
