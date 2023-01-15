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
// Q_DECLARE_LOGGING_CATEGORY(viewerRendererCategory)
// Q_LOGGING_CATEGORY(viewerRendererCategory, "viewer.renderer")
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
    std::unique_ptr<const Resources::GraphicsPipeline> graphicsPipeline;

    // TODO: descriptor allocator + descriptor layout cache

    vk::DeviceSize uniformBufferPerFrameSize = 0;
    UniformBuffer uniformBuffer;

    Impl(const engine::Engine & engine, const ResourceManager & resourceManager) : engine{engine}, resourceManager{resourceManager}
    {}

    void setT(float t)
    {
        uniformBuffer.t = t;
    }

    void frameStart(const QQuickWindow::GraphicsStateInfo & graphicsStateInfo);
    void render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo, const QSizeF & size);
};

Renderer::Renderer(const engine::Engine & engine, const ResourceManager & resourceManager) : impl_{engine, resourceManager}
{}

Renderer::~Renderer() = default;

void Renderer::setT(float t)
{
    return impl_->setT(t);
}

void Renderer::frameStart(const QQuickWindow::GraphicsStateInfo & graphicsStateInfo)
{
    return impl_->frameStart(graphicsStateInfo);
}

void Renderer::render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo, const QSizeF & size)
{
    return impl_->render(commandBuffer, renderPass, graphicsStateInfo, size);
}

void Renderer::Impl::frameStart(const QQuickWindow::GraphicsStateInfo & graphicsStateInfo)
{
    uint32_t framesInFlight = utils::autoCast(graphicsStateInfo.framesInFlight);
    if (!resources || (resources->getFramesInFlight() != framesInFlight)) {
        graphicsPipeline = nullptr;
        resources = resourceManager.getOrCreateResources(framesInFlight);

        std::copy_n(std::data(kVertices), std::size(kVertices), resources->getVertexBuffer().map<VertexType>().get());
    }

    auto uniformBufferPerFrameSize = resources->getUniformBufferPerFrameSize();
    uint32_t uniformBufferIndex = utils::autoCast(graphicsStateInfo.currentFrameSlot);
    *resources->getUniformBuffer().map<UniformBuffer>(uniformBufferPerFrameSize * uniformBufferIndex, uniformBufferPerFrameSize).get() = uniformBuffer;
}

void Renderer::Impl::render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo, const QSizeF & size)
{
    if (!resources) {
        return;
    }
    if (!graphicsPipeline || (graphicsPipeline->pipelineLayout.renderPass != renderPass)) {
        graphicsPipeline = resources->createGraphicsPipeline(renderPass);
    }

    constexpr engine::LabelColor redColor = {1.0f, 0.0f, 0.0f, 1.0f};
    auto rasterizationLabel = engine::ScopedCommandBufferLabel::create(library.dispatcher, commandBuffer, "Rasterization", redColor);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline->pipelines.pipelines.at(0), library.dispatcher);

    constexpr uint32_t firstBinding = 0;
    std::initializer_list<vk::Buffer> vertexBuffers = {
        resources->getVertexBuffer().getBuffer(),
    };
    std::vector<vk::DeviceSize> vertexBufferOffsets(std::size(vertexBuffers), 0);
    commandBuffer.bindVertexBuffers(firstBinding, vertexBuffers, vertexBufferOffsets, library.dispatcher);

    std::initializer_list<uint32_t> dinamicOffsets = {
        uint32_t(utils::autoCast(uniformBufferPerFrameSize * uint32_t(utils::autoCast(graphicsStateInfo.currentFrameSlot)))),
    };
    constexpr uint32_t firstSet = 0;
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphicsPipeline->pipelineLayout.pipelineLayout, firstSet, resources->getDescriptorSets(), dinamicOffsets, library.dispatcher);

    constexpr uint32_t firstViewport = 0;
    std::initializer_list<vk::Viewport> viewports = {
        {
            .x = 0,
            .y = 0,
            .width = utils::autoCast(size.width()),
            .height = utils::autoCast(size.height()),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        },
    };
    commandBuffer.setViewport(firstViewport, viewports, library.dispatcher);

    constexpr uint32_t firstScissor = 0;
    std::initializer_list<vk::Rect2D> scissors = {
        {
            .offset = {.x = 0, .y = 0},
            .extent = {.width = utils::autoCast(size.width()), .height = utils::autoCast(size.height())},
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
