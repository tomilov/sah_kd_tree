#include <common/version.hpp>
#include <engine/debug_utils.hpp>
#include <engine/device.hpp>
#include <engine/engine.hpp>
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
    const engine::Engine & engine;
    const SceneManager & sceneManager;

    const engine::Library & library = engine.getLibrary();
    const engine::Instance & instance = engine.getInstance();
    const engine::Device & device = engine.getDevice();

    std::shared_ptr<const Scene> scene;
    std::unique_ptr<const Scene::Descriptors> descriptors;
    std::unique_ptr<const Scene::GraphicsPipeline> graphicsPipeline;

    // TODO: descriptor allocator + descriptor layout cache

    UniformBuffer uniformBuffer;
    vk::Viewport viewport;
    vk::Rect2D scissor;
    PushConstants pushConstants;

    Impl(std::string_view token, const std::filesystem::path & scenePath, const engine::Engine & engine, const SceneManager & sceneManager) : token{token}, scenePath{scenePath}, engine{engine}, sceneManager{sceneManager}
    {
        init();
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

        viewport = {
            .x = utils::autoCast(x),
            .y = utils::autoCast(y),
            .width = utils::autoCast(width),
            .height = utils::autoCast(height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        scissor = {
            .offset = {.x = utils::autoCast(x), .y = utils::autoCast(y)},
            .extent = {.width = utils::autoCast(width), .height = utils::autoCast(height)},
        };
    }

    void setViewTransform(const glm::dmat3 & viewTransform)
    {
        pushConstants = {
            .viewTransform{glm::mat3(viewTransform)},  // double to float conversion and mat3x3 to mat3x4 conversion
        };
    }

    const std::filesystem::path & getScenePath() const
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

Renderer::Renderer(std::string_view token, const std::filesystem::path & scenePath, const engine::Engine & engine, const SceneManager & sceneManager) : impl_{token, scenePath, engine, sceneManager}
{}

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
    auto unmuteMessageGuard = engine.unmuteDebugUtilsMessages({0x5C0EC5D6, 0xE4D96472});

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
        graphicsPipeline = nullptr;
        descriptors = nullptr;
        scene = nullptr;
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
    *descriptors->uniformBuffers.at(currentFrameSlot).map<UniformBuffer>().get() = uniformBuffer;
}

void Renderer::Impl::render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo)
{
    auto unmuteMessageGuard = engine.unmuteDebugUtilsMessages({0x5C0EC5D6, 0xE4D96472});

    if (!scene) {
        return;
    }
    ASSERT(descriptors);
    if (!graphicsPipeline || (graphicsPipeline->pipelineLayout.renderPass != renderPass)) {
        graphicsPipeline = scene->createGraphicsPipeline(renderPass);
    }
    auto pipeline = graphicsPipeline->pipelines.pipelines.at(0);
    auto pipelineLayout = graphicsPipeline->pipelineLayout.pipelineLayout;

    constexpr engine::LabelColor kMagentaColor = {1.0f, 0.0f, 1.0f, 1.0f};
    auto rasterizationLabel = engine::ScopedCommandBufferLabel::create(library.dispatcher, commandBuffer, "Rasterization", kMagentaColor);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline, library.dispatcher);

    constexpr uint32_t firstSet = 0;
    uint32_t currentFrameSlot = utils::autoCast(graphicsStateInfo.currentFrameSlot);
    if (Scene::kUseDescriptorBuffer) {
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
            commandBuffer.setDescriptorBufferOffsetsEXT(vk::PipelineBindPoint::eGraphics, pipelineLayout, firstSet, bufferIndices, offsets, library.dispatcher);
        }
    } else {
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, firstSet, descriptors->descriptorSets.at(currentFrameSlot).descriptorSets, nullptr, library.dispatcher);
    }

    for (const auto & pushConstantRange : descriptors->pushConstantRanges) {
        commandBuffer.pushConstants(pipelineLayout, pushConstantRange.stageFlags, pushConstantRange.offset, pushConstantRange.size, &pushConstants, library.dispatcher);
    }

    constexpr uint32_t firstViewport = 0;
    std::initializer_list<vk::Viewport> viewports = {
        viewport,
    };
    commandBuffer.setViewport(firstViewport, viewports, library.dispatcher);

    constexpr uint32_t firstScissor = 0;
    std::initializer_list<vk::Rect2D> scissors = {
        scissor,
    };
    commandBuffer.setScissor(firstScissor, scissors, library.dispatcher);

    constexpr uint32_t firstBinding = 0;
    std::initializer_list<vk::Buffer> vertexBuffers = {
        descriptors->vertexBuffer.getBuffer(),
    };
    if (sah_kd_tree::kIsDebugBuild) {
        for (const auto & vertexBuffer : vertexBuffers) {
            if (vertexBuffer == vk::Buffer{}) {
                ASSERT(device.physicalDevice.physicalDeviceFeatures2Chain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>().nullDescriptor == VK_TRUE);
            }
        }
    }
    std::vector<vk::DeviceSize> vertexBufferOffsets(std::size(vertexBuffers), 0);
    commandBuffer.bindVertexBuffers(firstBinding, vertexBuffers, vertexBufferOffsets, library.dispatcher);

    // TODO: Scene::kUseDrawIndexedIndirect
    auto indexBuffer = descriptors->indexBuffer.getBuffer();
    constexpr vk::DeviceSize bufferDeviceOffset = 0;
    auto indexType = std::cbegin(descriptors->indexTypes);
    for (const auto & [indexCount, instanceCount, firstIndex, vertexOffset, firstInstance] : descriptors->instances) {
        INVARIANT(indexType != std::cend(descriptors->indexTypes), "");
        commandBuffer.bindIndexBuffer(indexBuffer, bufferDeviceOffset, *indexType++, library.dispatcher);
        commandBuffer.drawIndexed(indexCount, instanceCount, firstIndex, vertexOffset, firstInstance, library.dispatcher);
    }
}

}  // namespace viewer
