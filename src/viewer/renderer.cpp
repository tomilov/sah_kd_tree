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

#include <cstdint>

using namespace Qt::StringLiterals;

namespace viewer
{
namespace
{
[[maybe_unused]] Q_DECLARE_LOGGING_CATEGORY(viewerRendererCategory) Q_LOGGING_CATEGORY(viewerRendererCategory, "viewer.renderer")
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

    Impl(const engine::Engine & engine, const ResourceManager & resourceManager) : engine{engine}, resourceManager{resourceManager}
    {}

    void setAlpha(qreal alpha)
    {
        uniformBuffer.alpha = utils::autoCast(alpha);
    }

    void setT(float t)
    {
        uniformBuffer.t = t;
    }

    void frameStart(const QQuickWindow::GraphicsStateInfo & graphicsStateInfo);
    void render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo, const QRectF & viewportRect, const glm::dmat4x4 & viewTransform);
};

Renderer::Renderer(const engine::Engine & engine, const ResourceManager & resourceManager) : impl_{engine, resourceManager}
{}

Renderer::~Renderer() = default;

void Renderer::setAlpha(qreal t)
{
    return impl_->setAlpha(t);
}

void Renderer::setT(float t)
{
    return impl_->setT(t);
}

void Renderer::frameStart(const QQuickWindow::GraphicsStateInfo & graphicsStateInfo)
{
    return impl_->frameStart(graphicsStateInfo);
}

void Renderer::render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo, const QRectF & viewportRect, const glm::dmat4x4 & viewTransform)
{
    return impl_->render(commandBuffer, renderPass, graphicsStateInfo, viewportRect, viewTransform);
}

void Renderer::Impl::frameStart(const QQuickWindow::GraphicsStateInfo & graphicsStateInfo)
{
    uint32_t framesInFlight = utils::autoCast(graphicsStateInfo.framesInFlight);
    if (!resources || (resources->getFramesInFlight() != framesInFlight)) {
        graphicsPipeline = nullptr;
        descriptors = nullptr;
        resources = resourceManager.getOrCreateResources(framesInFlight);

        descriptors = resources->makeDescriptors();

        std::copy_n(std::data(kVertices), std::size(kVertices), descriptors->vertexBuffer.map<VertexType>().get());
    }

    uint32_t uniformBufferIndex = utils::autoCast(graphicsStateInfo.currentFrameSlot);
    *descriptors->uniformBuffer.map<UniformBuffer>(descriptors->uniformBufferPerFrameSize * uniformBufferIndex, sizeof uniformBuffer).get() = uniformBuffer;
}

void Renderer::Impl::render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo, const QRectF & viewportRect, const glm::dmat4x4 & viewTransform)
{
    if (!resources) {
        return;
    }
    if (!graphicsPipeline || (graphicsPipeline->pipelineLayout.renderPass != renderPass)) {
        graphicsPipeline = resources->createGraphicsPipeline(renderPass);
    }
    auto pipeline = graphicsPipeline->pipelines.pipelines.at(0);
    auto pipelineLayout = graphicsPipeline->pipelineLayout.pipelineLayout;

    constexpr engine::LabelColor kRedColor = {1.0f, 0.0f, 0.0f, 1.0f};
    auto rasterizationLabel = engine::ScopedCommandBufferLabel::create(library.dispatcher, commandBuffer, "Rasterization", kRedColor);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline, library.dispatcher);

    constexpr uint32_t firstBinding = 0;
    std::initializer_list<vk::Buffer> vertexBuffers = {
        descriptors->vertexBuffer.getBuffer(),
    };
    std::vector<vk::DeviceSize> vertexBufferOffsets(std::size(vertexBuffers), 0);
    commandBuffer.bindVertexBuffers(firstBinding, vertexBuffers, vertexBufferOffsets, library.dispatcher);

    uint32_t uniformBufferIndex = utils::autoCast(graphicsStateInfo.currentFrameSlot);
    std::initializer_list<uint32_t> dinamicOffsets = {
        uint32_t(utils::autoCast(descriptors->uniformBufferPerFrameSize * uniformBufferIndex)),
    };
    constexpr uint32_t firstSet = 0;
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, firstSet, descriptors->descriptorSets.value().descriptorSets, dinamicOffsets, library.dispatcher);

    PushConstants pushConstants = {
        .viewTransform{viewTransform},  // double to float conversion
    };
    for (const auto & pushConstantRange : descriptors->pushConstantRanges) {
        commandBuffer.pushConstants(pipelineLayout, pushConstantRange.stageFlags, pushConstantRange.offset, pushConstantRange.size, &pushConstants, library.dispatcher);
    }

    auto x = std::ceil(viewportRect.x());
    auto y = std::ceil(viewportRect.y());
    auto width = std::floor(viewportRect.width());
    auto height = std::floor(viewportRect.height());

    constexpr uint32_t firstViewport = 0;
    std::initializer_list<vk::Viewport> viewports = {
        {
            .x = utils::autoCast(x),
            .y = utils::autoCast(y),
            .width = utils::autoCast(width),
            .height = utils::autoCast(height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        },
    };
    commandBuffer.setViewport(firstViewport, viewports, library.dispatcher);

    constexpr uint32_t firstScissor = 0;
    std::initializer_list<vk::Rect2D> scissors = {
        {
            .offset = {.x = utils::autoCast(x), .y = utils::autoCast(y)},
            .extent = {.width = utils::autoCast(width), .height = utils::autoCast(height)},
        },
    };
    commandBuffer.setScissor(firstScissor, scissors, library.dispatcher);

    uint32_t vertexCount = utils::autoCast(std::size(kVertices));
    constexpr uint32_t instanceCount = 1;
    constexpr uint32_t firstVertex = 0;
    constexpr uint32_t firstInstance = 0;
    commandBuffer.draw(vertexCount, instanceCount, firstVertex, firstInstance, library.dispatcher);
}

}  // namespace viewer
