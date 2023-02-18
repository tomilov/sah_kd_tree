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
#include <viewer/resource_manager.hpp>

#include <vulkan/vulkan.hpp>
#include <spdlog/spdlog.h>

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
#include <initializer_list>
#include <iterator>
#include <memory>
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
    enum class Stage
    {
        Vertex,
        Fragment,
    };

    const engine::Engine & engine;
    const ResourceManager & resourceManager;

    const engine::Library & library = engine.getLibrary();
    const engine::Instance & instance = engine.getInstance();
    const engine::Device & device = engine.getDevice();

    std::shared_ptr<const Resources> resources;
    std::unique_ptr<const Resources::Descriptors> descriptors;
    std::unique_ptr<const Resources::GraphicsPipeline> graphicsPipeline;

    // TODO: descriptor allocator + descriptor layout cache

    UniformBuffer uniformBuffer;
    vk::Viewport viewport;
    vk::Rect2D scissor;
    PushConstants pushConstants;

    Impl(const engine::Engine & engine, const ResourceManager & resourceManager) : engine{engine}, resourceManager{resourceManager}
    {}

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

    void setViewTransform(const glm::dmat4x4 & viewTransform)
    {
        pushConstants = {
            .viewTransform{viewTransform},  // double to float conversion
        };
    }

    void frameStart(const QQuickWindow::GraphicsStateInfo & graphicsStateInfo);
    void render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo);
};

Renderer::Renderer(const engine::Engine & engine, const ResourceManager & resourceManager) : impl_{engine, resourceManager}
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

void Renderer::setViewTransform(const glm::dmat4x4 & viewTransform)
{
    return impl_->setViewTransform(viewTransform);
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
    if (!resources || (resources->getFramesInFlight() != framesInFlight)) {
        graphicsPipeline = nullptr;
        descriptors = nullptr;
        resources = resourceManager.getOrCreateResources(framesInFlight);

        descriptors = resources->makeDescriptors();

        std::copy_n(std::data(kVertices), std::size(kVertices), descriptors->vertexBuffer.map<VertexType>().get());
    }

    uint32_t currentFrameSlot = utils::autoCast(graphicsStateInfo.currentFrameSlot);
    *descriptors->uniformBuffer.at(currentFrameSlot).map<UniformBuffer>().get() = uniformBuffer;
}

void Renderer::Impl::render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo)
{
    auto unmuteMessageGuard = engine.unmuteDebugUtilsMessages({0x5C0EC5D6, 0xE4D96472});

    if (!resources) {
        return;
    }
    ASSERT(descriptors);
    if (!graphicsPipeline || (graphicsPipeline->pipelineLayout.renderPass != renderPass)) {
        graphicsPipeline = resources->createGraphicsPipeline(renderPass);
    }
    auto pipeline = graphicsPipeline->pipelines.pipelines.at(0);
    auto pipelineLayout = graphicsPipeline->pipelineLayout.pipelineLayout;

    constexpr engine::LabelColor kMagentaColor = {1.0f, 0.0f, 1.0f, 1.0f};
    auto rasterizationLabel = engine::ScopedCommandBufferLabel::create(library.dispatcher, commandBuffer, "Rasterization", kMagentaColor);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline, library.dispatcher);

    constexpr uint32_t firstBinding = 0;
    std::initializer_list<vk::Buffer> vertexBuffers = {
        descriptors->vertexBuffer.getBuffer(),
    };
    std::vector<vk::DeviceSize> vertexBufferOffsets(std::size(vertexBuffers), 0);
    commandBuffer.bindVertexBuffers(firstBinding, vertexBuffers, vertexBufferOffsets, library.dispatcher);

    constexpr uint32_t firstSet = 0;
    uint32_t currentFrameSlot = utils::autoCast(graphicsStateInfo.currentFrameSlot);
    if (Resources::kUseDescriptorBuffer) {
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

    uint32_t vertexCount = utils::autoCast(std::size(kVertices));
    constexpr uint32_t instanceCount = 1;
    constexpr uint32_t firstVertex = 0;
    constexpr uint32_t firstInstance = 0;
    commandBuffer.draw(vertexCount, instanceCount, firstVertex, firstInstance, library.dispatcher);
}

}  // namespace viewer
