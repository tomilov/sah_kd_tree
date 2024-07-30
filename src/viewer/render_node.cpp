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
#include <builder/builder.hpp>
#include <viewer/utils.hpp>

#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/quaternion_float.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_transform_2d.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/trigonometric.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <vulkan/vulkan.hpp>

#include <QtCore/QLoggingCategory>
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

#include <memory>
#include <optional>
#include <utility>

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdint>

using namespace Qt::StringLiterals;

namespace viewer
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(viewerRenderNodeCategory)
Q_LOGGING_CATEGORY(viewerRenderNodeCategory, "viewer.render_node")

void checkContext(QQuickWindow * window, const engine::Context & context)
{
    Q_CHECK_PTR(window);

    auto ri = window->rendererInterface();

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

    INVARIANT(vk::Instance(instance->vkInstance()) == context.getInstance().getInstance(), "Should match");
    INVARIANT(*physicalDevice == context.getPhysicalDevice().getPhysicalDevice(), "Should match");
    INVARIANT(*device == context.getDevice().getDevice(), "Should match");
    const auto & queueCreateInfo = context.getPhysicalDevice().externalGraphicsQueueCreateInfo;
    INVARIANT(*queueFamilyIndex == queueCreateInfo.familyIndex, "Should match");
    INVARIANT(*queueIndex == queueCreateInfo.index, "Should match");
    {
        VkQueue q = VK_NULL_HANDLE;
        vkGetDeviceQueue(*device, *queueFamilyIndex, *queueIndex, &q);
        INVARIANT(*queue == vk::Queue(q), "Should match");
    }

    context.getDevice().setDebugUtilsObjectName(*queue, "Qt graphical queue");
}
}  // namespace

struct RenderNode::Impl
{
    QQuickWindow * const window;
    const engine::Context & context;
    const Engine & engine;

    std::optional<Renderer> renderer;
    std::shared_ptr<const Scene> scene;
    std::shared_ptr<const builder::Tree> tree;

    bool isDirty = false;

    QRectF rect;
    FrameSettings frameSettings;

    int renderdocCaptureFrameCounter = 0;
    int renderdocCaptureFrameCount = 0;
    std::optional<debug_utils::Renderdoc::FrameCapture> frameCapture;

    QVector<quint32> renderPassFormat;

    Impl(QQuickWindow * window, const EngineWrapper & engineWrapper)
        : window{window}
        , context{engineWrapper.getContext()}
        , engine{engineWrapper.getEngine()}
    {
        Q_ASSERT(window);
        checkContext(window, context);
    }

    void unsetScene()
    {
        if (renderer) {
            ASSERT(renderer.value().getScene() == scene);
            renderer.value().unsetScene();
        }
        scene.reset();
        isDirty = true;
    }

    bool setScene(const std::filesystem::path & scenePath)
    {
        ASSERT(!scene);
        ASSERT(!renderer || !renderer.value().getScene());
        scene = engine.getScenes().getScene(scenePath);
        if (!scene) {
            return false;
        }
        isDirty = true;
        return true;
    }

    void unsetTree()
    {
        if (!tree) {
            return;
        }
        tree.reset();
        isDirty = true;
    }

    bool updateTree(float emptinessFactor, float traversalCost, float intersectionCost, uint32_t maxDepth)
    {
        if (!scene) {
            unsetTree();
            return true;
        }
        const builder::Tree::Settings treeSettings = {
            .emptinessFactor = emptinessFactor,
            .traversalCost = traversalCost,
            .intersectionCost = intersectionCost,
            .maxDepth = utils::autoCast(maxDepth),
        };
        if (!tree || (tree->getSettings() != treeSettings)) {
            unsetTree();
            const auto getNewTree = [this, &treeSettings]
            {
                ElapsedTimer elapsedTimer{viewerRenderNodeCategory, u"Build SAH kd-tree for '%1'"_s.arg(QString::fromStdString(scene->scenePath))};
                return engine.getBuilder().build(treeSettings, scene->sceneData);
            };
            if (auto newTree = getNewTree()) {
                tree = std::make_shared<builder::Tree>(std::move(newTree).value());
                isDirty = true;
            } else {
                return false;
            }
        }
        return true;
    }

    template<typename T>
    void updateState(T & lhs, const T & rhs, [[maybe_unused]] const char * name)
    {
        if (lhs == rhs) {
            return;
        }
        lhs = rhs;
        isDirty = true;
    }

#define UPDATE_STATE(lhs, rhs) updateState(lhs, rhs, #rhs)
    void updateRect(const QRectF & rect)
    {
        UPDATE_STATE(this->rect, rect);
    }

    void updateMode(bool useOffscreenTexture, bool discardInvisible, bool wireFrame)
    {
        UPDATE_STATE(frameSettings.useOffscreenTexture, useOffscreenTexture);
        UPDATE_STATE(frameSettings.discardInvisible, discardInvisible);
        UPDATE_STATE(frameSettings.wireFrame, wireFrame);
    }

    void updateCamera(const glm::vec3 & position, const glm::quat & orientation, float fov, float zNear, float zFar)
    {
        UPDATE_STATE(frameSettings.position, position);
        UPDATE_STATE(frameSettings.orientation, orientation);
        UPDATE_STATE(frameSettings.fov, fov);
        UPDATE_STATE(frameSettings.zNear, zNear);
        UPDATE_STATE(frameSettings.zFar, zFar);
    }

    void setClearColor(const glm::vec4 & clearColor)
    {
        UPDATE_STATE(frameSettings.clearColor, clearColor);
    }

    void setRenderdocCaptureFrameCounter(int renderdocCaptureFrameCounter)
    {
        UPDATE_STATE(this->renderdocCaptureFrameCounter, renderdocCaptureFrameCounter);
    }
#undef UPDATE_STATE

    bool markDirty()
    {
        if (!isDirty) {
            return false;
        }
        isDirty = false;
        return true;
    }

    [[nodiscard]] QRectF getScissorRect(const QSize & renderTargetSize, const QMatrix4x4 & mvp)
    {
        QRectF scissorRect = mvp.mapRect(rect);
        scissorRect.translate(1.0, 1.0);
        scissorRect.setTopLeft(scissorRect.topLeft() * 0.5);
        scissorRect.setBottomRight(scissorRect.bottomRight() * 0.5);
        scissorRect &= QRectF{0.0, 0.0, 1.0, 1.0};

        auto [x, y] = scissorRect.topLeft();
        x *= renderTargetSize.width();
        y *= renderTargetSize.height();

        auto [w, h] = scissorRect.size();
        w *= renderTargetSize.width();
        h *= renderTargetSize.height();

        return {x, y, w, h};
    }

    void advance()
    {
        const QQuickWindow::GraphicsStateInfo & graphicsStateInfo = window->graphicsStateInfo();
        uint32_t framesInFlight = utils::autoCast(graphicsStateInfo.framesInFlight);
        if (renderer) {
            ASSERT(renderer.value().getFramesInFlight() == framesInFlight);
        } else {
            renderer.emplace(context, engine, framesInFlight);
        }
        renderer.value().setFrameSettings(frameSettings);
        if (scene && !renderer.value().getScene()) {
            renderer.value().setScene(scene);
        }
        if (renderdocCaptureFrameCount < renderdocCaptureFrameCounter) {
            ++renderdocCaptureFrameCount;
            frameCapture.emplace(debug_utils::Renderdoc::makeFrameCapture(context.getInstance().getInstance(), utils::autoCast(window->winId())));
        }
        renderer.value().advance(utils::autoCast(graphicsStateInfo.currentFrameSlot));
    }

    void prepare(float alpha, const QSize & renderTargetSize, const QMatrix4x4 & mvp, bool isAxisAligned)
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

        const QRectF scissorRect = getScissorRect(renderTargetSize, mvp);
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

        advance();
    }

    void render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, bool isRenderPassFormatChanged)
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

RenderNode::RenderNode(QQuickWindow * window, const EngineWrapper & engineWrapper)
    : impl_{window, engineWrapper}
{}

void RenderNode::unsetScene()
{
    return impl_->unsetScene();
}

bool RenderNode::setScene(const std::filesystem::path & scenePath)
{
    return impl_->setScene(scenePath);
}

const std::shared_ptr<const Scene> & RenderNode::getScene() const &
{
    return impl_->scene;
}

void RenderNode::unsetTree()
{
    return impl_->unsetTree();
}

const std::shared_ptr<const builder::Tree> &RenderNode::getTree() const &
{
    return impl_->tree;
}

bool RenderNode::updateTree(float emptinessFactor, float traversalCost, float intersectionCost, uint32_t maxDepth)
{
    return impl_->updateTree(emptinessFactor, traversalCost, intersectionCost, maxDepth);
}

void RenderNode::updateRect(const QRectF & rect)
{
    return impl_->updateRect(rect);
}

void RenderNode::updateMode(bool useOffscreenTexture, bool discardInvisible, bool wireFrame)
{
    return impl_->updateMode(useOffscreenTexture, discardInvisible, wireFrame);
}

void RenderNode::updateCamera(const QVector3D & cameraPosition, const QQuaternion & cameraOrientation, float cameraFov, float zNear, float zFar)
{
    glm::vec3 position{cameraPosition.x(), cameraPosition.y(), cameraPosition.z()};
    glm::quat orientation{cameraOrientation.scalar(), cameraOrientation.x(), cameraOrientation.y(), cameraOrientation.z()};
    float fov = glm::radians(cameraFov);
    return impl_->updateCamera(position, orientation, fov, zNear, zFar);
}

void RenderNode::setClearColor(const QColor & clearColor)
{
    float r, g, b, a;
    clearColor.getRgbF(&r, &g, &b, &a);
    return impl_->setClearColor({r, g, b, a});
}

void RenderNode::setRenderdocCaptureFrameCounter(int renderdocCaptureFrameCounter)
{
    return impl_->setRenderdocCaptureFrameCounter(renderdocCaptureFrameCounter);
}

void RenderNode::markDirty()
{
    if (!impl_->markDirty()) {
        return;
    }
    return QSGNode::markDirty(QSGNode::DirtyStateBit::DirtyForceUpdate);
}

void RenderNode::prepare()
{
    float alpha = utils::autoCast(inheritedOpacity());
    const QSize renderTargetSize = renderTarget()->pixelSize();
    const QMatrix4x4 mvp = *projectionMatrix() * *matrix();
    bool isAxisAligned = false;
    if ((false)) {  // sadly,  does not reset automatically w/o extra update()
        // optimization for axis aligned transform case
        const QMatrix4x4 & modelView = *matrix();
        if (modelView.flags() < QMatrix4x4::Flag::Rotation) {
            if ((qFuzzyIsNull(modelView(0, 1)) && qFuzzyIsNull(modelView(1, 0))) || (qFuzzyIsNull(modelView(0, 0)) && qFuzzyIsNull(modelView(1, 1)))) {
                isAxisAligned = true;
            }
        }
    }
    return impl_->prepare(alpha, renderTargetSize, mvp, isAxisAligned);
}

void RenderNode::render(const RenderState * renderState)
{
    if ((false)) {
        QStringList clipRegions;
        if (auto clipRegion = renderState->clipRegion()) {
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

    auto commandBufferNativeHandles = commandBuffer()->nativeHandles();
    Q_CHECK_PTR(commandBufferNativeHandles);
    vk::CommandBuffer commandBuffer = static_cast<const QRhiVulkanCommandBufferNativeHandles *>(commandBufferNativeHandles)->commandBuffer;

    auto renderPassDescriptor = renderTarget()->renderPassDescriptor();
    auto newRenderPassFormat = renderPassDescriptor->serializedFormat();
    const bool isRenderPassFormatChanged = impl_->renderPassFormat != newRenderPassFormat;
    if (isRenderPassFormatChanged) {
        impl_->renderPassFormat = std::move(newRenderPassFormat);
        qCDebug(viewerRenderNodeCategory) << u"Render pass format changed"_s;
    }
    auto renderPassNativeHandles = renderPassDescriptor->nativeHandles();
    Q_CHECK_PTR(renderPassNativeHandles);
    vk::RenderPass renderPass = static_cast<const QRhiVulkanRenderPassNativeHandles *>(renderPassNativeHandles)->renderPass;

    return impl_->render(commandBuffer, renderPass, isRenderPassFormatChanged);
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
