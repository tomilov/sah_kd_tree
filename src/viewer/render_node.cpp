#include <builder/builder.hpp>
#include <debug_utils/renderdoc.hpp>  //
#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/instance.hpp>
#include <engine/physical_device.hpp>
#include <format/glm.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <viewer/engine.hpp>
#include <viewer/engine_wrapper.hpp>
#include <viewer/render_node.hpp>
#include <viewer/renderer.hpp>
#include <viewer/scenes.hpp>
#include <viewer/utils.hpp>

#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/quaternion_float.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/trigonometric.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <QtCore/QFileInfo>
#include <QtCore/QLoggingCategory>
#include <QtCore/QPromise>
#include <QtCore/QRectF>
#include <QtCore/QSizeF>
#include <QtCore/QVector>
#include <QtCore/QtAssert>
#include <QtCore/QtLogging>
#include <QtCore/QtNumeric>
#include <QtGui/QGuiApplication>
#include <QtGui/QMatrix4x4>
#include <QtGui/QVulkanInstance>
#include <QtGui/rhi/qrhi.h>
#include <QtQuick/QQuickWindow>
#include <QtQuick/QSGNode>
#include <QtQuick/QSGRendererInterface>

#include <optional>
#include <utility>

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdint>

using namespace Qt::StringLiterals;
using namespace std::string_view_literals;

namespace viewer
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(viewerRenderNodeCategory)
Q_LOGGING_CATEGORY(
    viewerRenderNodeCategory,
    "viewer.render_node")

void checkContext(
    QQuickWindow * window,
    const engine::Context & context)
{
    Q_CHECK_PTR(window);

    auto * ri = window->rendererInterface();

    QVulkanInstance * instance = utils::autoCast(ri->getResource(window, QSGRendererInterface::Resource::VulkanInstanceResource));
    Q_CHECK_PTR(instance);

    vk::PhysicalDevice * physicalDevice = utils::autoCast(ri->getResource(window, QSGRendererInterface::Resource::PhysicalDeviceResource));
    Q_CHECK_PTR(physicalDevice);

    vk::Device * device = utils::autoCast(ri->getResource(window, QSGRendererInterface::Resource::DeviceResource));
    Q_CHECK_PTR(device);

    uint32_t * queueFamilyIndex = utils::autoCast(ri->getResource(window, QSGRendererInterface::Resource::GraphicsQueueFamilyIndexResource));
    Q_CHECK_PTR(queueFamilyIndex);

    uint32_t * queueIndex = utils::autoCast(ri->getResource(window, QSGRendererInterface::Resource::GraphicsQueueIndexResource));
    Q_CHECK_PTR(queueIndex);

    vk::Queue * queue = utils::autoCast(ri->getResource(window, QSGRendererInterface::Resource::CommandQueueResource));
    Q_CHECK_PTR(queue);

#define GET_INSTANCE_PROC_ADDR(name) PFN_##name name = utils::autoCast(instance->getInstanceProcAddr(#name))
    // GET_INSTANCE_PROC_ADDR(vkGetInstanceProcAddr);
    GET_INSTANCE_PROC_ADDR(vkGetDeviceProcAddr);
#undef GET_INSTANCE_PROC_ADDR
    PFN_vkGetDeviceQueue vkGetDeviceQueue = utils::autoCast(vkGetDeviceProcAddr(*device, "vkGetDeviceQueue"));

    INVARIANT(vk::Instance(instance->vkInstance()) == context.getInstance().getHandle(), "Should match");
    INVARIANT(*physicalDevice == context.getPhysicalDevice().getHandle(), "Should match");
    INVARIANT(*device == context.getDevice().getHandle(), "Should match");
    const auto & queueCreateInfo = context.getPhysicalDevice().externalGraphicsQueueCreateInfo;
    INVARIANT(*queueFamilyIndex == queueCreateInfo.familyIndex, "Should match");
    INVARIANT(*queueIndex == queueCreateInfo.index, "Should match");
    {
        VkQueue q = nullptr;
        vkGetDeviceQueue(*device, *queueFamilyIndex, *queueIndex, &q);
        INVARIANT(*queue == vk::Queue(q), "Should match");
    }

    context.getDevice().setDebugUtilsObjectName(*queue, "Qt graphical queue");
}
}  // namespace

struct RenderNode::Impl
{
    QString name;
    QQuickWindow * const window;
    const engine::Context & context;
    const Engine & engine;

    scene_data::SceneDataPtr sceneData;
    std::optional<Renderer> renderer;
    builder::TreePtr builderTree;
    bool treeIsDirty = false;

    bool isDirty = false;

    QRectF rect;
    FrameSettings frameSettings;

    int renderdocCaptureFrameCounter = 0;
    int renderdocCaptureFrameCount = 0;
    std::optional<debug_utils::Renderdoc::FrameCapture> frameCapture;

    QVector<quint32> renderPassFormat;

    bool update = false;

    Impl(
        QString nameIn,
        QQuickWindow * windowIn,
        const EngineWrapper & engineWrapper)
        : name{nameIn}
        , window{windowIn}
        , context{engineWrapper.getContext()}
        , engine{engineWrapper.getEngine()}
    {
        Q_ASSERT(window);
        checkContext(window, context);
    }

    template<
        typename Dst,
        typename Src>
    [[maybe_unused]] bool updateState(
        Dst & lhs,
        Src && rhs,
        [[maybe_unused]] const char * stateName)
    {
        if (lhs == rhs) {
            return false;
        }
        lhs = std::forward<Src>(rhs);
        isDirty = true;
        return true;
    }

#define UPDATE_STATE(lhs, rhs) updateState(lhs, rhs, #rhs)
    void unsetScene()
    {
        if (sceneData) {
            sceneData.reset();
            isDirty = true;
        }
    }

    void updateScene(const scene_data::SceneDataPtr & sceneDataIn)
    {
        UPDATE_STATE(sceneData, sceneDataIn);
    }

    void setTree(builder::TreePtr builderTreeIn)
    {
        builderTree = std::move(builderTreeIn);
        treeIsDirty = true;
        isDirty = true;
    }

    void updateRect(const QRectF & rectIn)
    {
        UPDATE_STATE(rect, rectIn);
    }

    void updateMode(
        bool traceSahKdTree,
        bool useOffscreenTexture,
        bool discardInvisible,
        bool wireframe)
    {
        UPDATE_STATE(frameSettings.traceSahKdTree, traceSahKdTree);
        UPDATE_STATE(frameSettings.useOffscreenTexture, useOffscreenTexture);
        UPDATE_STATE(frameSettings.discardInvisible, discardInvisible);
        UPDATE_STATE(frameSettings.wireframe, wireframe);
    }

    void updateCamera(
        const glm::vec3 & position,
        const glm::quat & orientation,
        float fov,
        float zNear,
        float zFar)
    {
        UPDATE_STATE(frameSettings.position, position);
        UPDATE_STATE(frameSettings.orientation, orientation);
        UPDATE_STATE(frameSettings.fov, fov);
        UPDATE_STATE(frameSettings.zNear, zNear);
        UPDATE_STATE(frameSettings.zFar, zFar);
    }

    void updateClearColor(const glm::vec4 & clearColor)
    {
        UPDATE_STATE(frameSettings.clearColor, clearColor);
    }

    void updateRenderdocCaptureFrameCounter(int renderdocCaptureFrameCounterIn)
    {
        UPDATE_STATE(renderdocCaptureFrameCounter, renderdocCaptureFrameCounterIn);
    }
#undef UPDATE_STATE

    bool resetDirty()
    {
        if (!isDirty) {
            return false;
        }
        isDirty = false;
        update = true;
        return true;
    }

    [[nodiscard]] QRectF getScissorRect(
        int width,
        int height,
        const QMatrix4x4 & mvp) const
    {
        QRectF scissorRect = mvp.mapRect(rect);
        scissorRect.translate(1.0, 1.0);
        scissorRect.setTopLeft(scissorRect.topLeft() * 0.5);
        scissorRect.setBottomRight(scissorRect.bottomRight() * 0.5);
        scissorRect &= QRectF{0.0, 0.0, 1.0, 1.0};

        auto [x, y] = scissorRect.topLeft();
        x *= width;
        y *= height;

        auto [w, h] = scissorRect.bottomRight();
        w *= width;
        h *= height;

        return {x, y, w, h};
    }

    void advance(vk::CommandBuffer commandBuffer)
    {
        const QQuickWindow::GraphicsStateInfo & graphicsStateInfo = window->graphicsStateInfo();
        uint32_t framesInFlight = utils::autoCast(graphicsStateInfo.framesInFlight);
        if (renderer) {
            ASSERT(renderer.value().getFramesInFlight() == framesInFlight);
        } else {
            renderer.emplace(name.toStdString(), context, engine, framesInFlight);
        }
        renderer.value().setFrameSettings(frameSettings);
        if (renderer.value().getScene() != sceneData) {
            if (renderer.value().getScene()) {
                renderer.value().unsetScene();
            }
            if (sceneData) {
                renderer.value().setScene(sceneData);
            }
        }
        if (treeIsDirty) {
            treeIsDirty = false;
            renderer.value().setTree(std::move(builderTree));
        }
        if (renderdocCaptureFrameCount < renderdocCaptureFrameCounter) {
            ++renderdocCaptureFrameCount;
            frameCapture.emplace(debug_utils::Renderdoc::makeFrameCapture(context.getInstance().getHandle(), utils::autoCast(window->winId())));
        }
        renderer.value().advance(commandBuffer, utils::autoCast(graphicsStateInfo.currentFrameSlot));
    }

    void prepare(
        vk::CommandBuffer commandBuffer,
        float alpha,
        const QSize & renderTargetSize,
        const QMatrix4x4 & mvp,
        bool isAxisAligned)
    {
        if (!rect.isValid()) {
            return;
        }

        frameSettings.alpha = alpha;

        frameSettings.width = utils::autoCast(std::ceil(rect.width()));
        frameSettings.height = utils::autoCast(std::ceil(rect.height()));

        frameSettings.viewport = vk::Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = utils::autoCast(renderTargetSize.width()),
            .height = utils::autoCast(renderTargetSize.height()),
            .minDepth = engine::kMinDepth,
            .maxDepth = 1.0f,
        };

        const QRectF scissorRect = getScissorRect(renderTargetSize.width(), renderTargetSize.height(), mvp);
        frameSettings.scissor = vk::Rect2D{
            .offset = {
                .x = utils::autoCast(std::floor(scissorRect.x())),
                .y = utils::autoCast(std::floor(scissorRect.y())),
            },
            .extent = {
                .width = utils::autoCast(std::ceil(scissorRect.width())),
                .height = utils::autoCast(std::ceil(scissorRect.height())),
            },
        };
        glm::mat4 & windowMvp = frameSettings.windowMvp;
        windowMvp = glm::make_mat4x4(mvp.constData());
        windowMvp = glm::scale(windowMvp, glm::vec3{frameSettings.width * 0.5f, frameSettings.height * 0.5f, 1.0f});
        windowMvp = glm::translate(windowMvp, glm::vec3{1.0f, 1.0f, 0.0f});

        if (isAxisAligned) {
            frameSettings.useOffscreenTexture = false;
        }

        advance(commandBuffer);
    }

    void render(
        vk::CommandBuffer commandBuffer,
        vk::RenderPass renderPass,
        bool isRenderPassFormatChanged)
    {
        if (!rect.isValid()) {
            return;
        }

        const auto & device = context.getDevice();
        device.setDebugUtilsObjectName(commandBuffer, "Qt command buffer");

        ASSERT(renderer);

        const QQuickWindow::GraphicsStateInfo & graphicsStateInfo = window->graphicsStateInfo();
        ASSERT(renderer.value().getFramesInFlight() == utils::safeCast<uint32_t>(graphicsStateInfo.framesInFlight));
        renderer.value().render(commandBuffer, renderPass, isRenderPassFormatChanged, utils::autoCast(graphicsStateInfo.currentFrameSlot));

        frameCapture.reset();
    }

    void releaseResources()
    {
        renderer.reset();
    }

    void flags(QSGRenderNode::RenderingFlags & renderingFlags) const
    {
        if (frameSettings.useOffscreenTexture) {
            renderingFlags |= RenderingFlag::DepthAwareRendering;
            renderingFlags |= RenderingFlag::BoundedRectRendering;
            if (!frameSettings.discardInvisible && (frameSettings.alpha == 1.0f)) {
                renderingFlags |= RenderingFlag::OpaqueRendering;
            }
        }
    }
};

RenderNode::RenderNode(
    QString name,
    QQuickWindow * window,
    const EngineWrapper & engineWrapper)
    : impl_{std::make_unique<Impl>(
          name,
          window,
          engineWrapper)}
{}

void RenderNode::unsetScene()
{
    impl_->unsetScene();
}

void RenderNode::updateScene(const scene_data::SceneDataPtr & sceneData)
{
    impl_->updateScene(sceneData);
}

auto RenderNode::getScene() const & -> const scene_data::SceneDataPtr &
{
    return impl_->sceneData;
}

void RenderNode::setTree(builder::TreePtr && tree)
{
    impl_->setTree(std::move(tree));
}

void RenderNode::updateRect(const QRectF & rect)
{
    impl_->updateRect(rect);
}

void RenderNode::updateMode(
    bool traceSahKdTree,
    bool useOffscreenTexture,
    bool discardInvisible,
    bool wireframe)
{
    impl_->updateMode(traceSahKdTree, useOffscreenTexture, discardInvisible, wireframe);
}

void RenderNode::updateCamera(
    const QVector3D & cameraPosition,
    const QQuaternion & cameraOrientation,
    float cameraFov,
    float zNear,
    float zFar)
{
    glm::vec3 position{cameraPosition.x(), cameraPosition.y(), cameraPosition.z()};
    glm::quat orientation{cameraOrientation.scalar(), cameraOrientation.x(), cameraOrientation.y(), cameraOrientation.z()};
    float fov = glm::radians(cameraFov);
    impl_->updateCamera(position, orientation, fov, zNear, zFar);
}

void RenderNode::updateClearColor(const QColor & clearColor)
{
    float r, g, b, a;
    clearColor.getRgbF(&r, &g, &b, &a);
    impl_->updateClearColor({r, g, b, a});
}

void RenderNode::updateRenderdocCaptureFrameCounter(int renderdocCaptureFrameCounter)
{
    impl_->updateRenderdocCaptureFrameCounter(renderdocCaptureFrameCounter);
}

void RenderNode::updateDirty()
{
    if (!impl_->resetDirty()) {
        return;
    }
    QSGNode::markDirty(QSGNode::DirtyStateBit::DirtyForceUpdate);
}

void RenderNode::prepare()
{
    if (!impl_->update) {
        return;
    }
    impl_->update = false;
    float alpha = utils::autoCast(inheritedOpacity());
    const QSize renderTargetSize = renderTarget()->pixelSize();
    const QMatrix4x4 mvp = *projectionMatrix() * *matrix();
    bool isAxisAligned = false;
    if ((false)) {  // sadly,  does not reset automatically w/o extra update()
        // optimization for axis aligned transform case
        const QMatrix4x4::Flags modelViewMatrixFlags = matrix()->flags();
        if ((modelViewMatrixFlags < QMatrix4x4::Flag::Rotation) && (modelViewMatrixFlags & QMatrix4x4::Flag::Rotation2D)) {
            isAxisAligned = true;
        }
    }

    const auto * const commandBufferNativeHandles = commandBuffer()->nativeHandles();
    Q_CHECK_PTR(commandBufferNativeHandles);
    vk::CommandBuffer commandBuffer = static_cast<const QRhiVulkanCommandBufferNativeHandles *>(commandBufferNativeHandles)->commandBuffer;
    impl_->prepare(commandBuffer, alpha, renderTargetSize, mvp, isAxisAligned);
}

void RenderNode::render(const RenderState * renderState)
{
    if ((false)) {
        QStringList clipRegions;
        if (const auto * clipRegion = renderState->clipRegion()) {
            for (const QRect & rect : *clipRegion) {
                clipRegions << toString(rect);
            }
        }
        qCInfo(viewerRenderNodeCategory)                                                                   //
            << u"scissorRect(%1) scissorEnabled(%2) stencilValue(%3) stencilEnabled(%4) clipRegion(%5)"_s  //
                   .arg(toString(renderState->scissorRect()))                                              //
                   .arg(renderState->scissorEnabled())                                                     //
                   .arg(renderState->stencilValue())                                                       //
                   .arg(renderState->stencilEnabled())                                                     //
                   .arg(clipRegions.join(u"|"_s));                                                         //
    }

    const auto * commandBufferNativeHandles = commandBuffer()->nativeHandles();
    Q_CHECK_PTR(commandBufferNativeHandles);
    vk::CommandBuffer commandBuffer = static_cast<const QRhiVulkanCommandBufferNativeHandles *>(commandBufferNativeHandles)->commandBuffer;

    auto * renderPassDescriptor = renderTarget()->renderPassDescriptor();
    auto newRenderPassFormat = renderPassDescriptor->serializedFormat();
    const bool isRenderPassFormatChanged = impl_->renderPassFormat != newRenderPassFormat;
    if (isRenderPassFormatChanged) {
        impl_->renderPassFormat = std::move(newRenderPassFormat);
        qCDebug(viewerRenderNodeCategory) << u"Render pass format changed"_s;
    }
    const auto * renderPassNativeHandles = renderPassDescriptor->nativeHandles();
    Q_CHECK_PTR(renderPassNativeHandles);
    vk::RenderPass renderPass = static_cast<const QRhiVulkanRenderPassNativeHandles *>(renderPassNativeHandles)->renderPass;

    impl_->render(commandBuffer, renderPass, isRenderPassFormatChanged);
}

void RenderNode::releaseResources()
{
    impl_->releaseResources();
}

auto RenderNode::flags() const -> RenderingFlags
{
    auto renderingFlags = QSGRenderNode::flags();
    impl_->flags(renderingFlags);
    return renderingFlags;
}

QRectF RenderNode::rect() const
{
    if (flags() & RenderingFlag::BoundedRectRendering) {
        return impl_->rect;
    }
    return QSGRenderNode::rect();
}

QSGRenderNode::StateFlags RenderNode::changedStates() const
{
    return StateFlag::ViewportState | StateFlag::ScissorState;
}

}  // namespace viewer
