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

#include <memory>
#include <new>
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
    using ScenePtr = std::shared_ptr<const Scene>;
    using SceneFutureWatcher = QFutureWatcher<ScenePtr>;
    using TreePtr = std::shared_ptr<const builder::Tree>;
    using FutureWatcher = QFutureWatcher<TreePtr>;

    QQuickWindow * const window;
    const engine::Context & context;
    const Engine & engine;
    TaskQueue * const taskQueue;
    QSharedPointer<QFutureWatcherBase> & sceneFutureWatcher;
    QSharedPointer<QFutureWatcherBase> & treeFutureWatcher;

    ScenePtr scene;
    std::optional<Renderer> renderer;
    TreePtr tree;

    bool isDirty = false;

    QRectF rect;
    FrameSettings frameSettings;

    int renderdocCaptureFrameCounter = 0;
    int renderdocCaptureFrameCount = 0;
    std::optional<debug_utils::Renderdoc::FrameCapture> frameCapture;

    QVector<quint32> renderPassFormat;

    Impl(QQuickWindow * window, const EngineWrapper & engineWrapper, TaskQueue * taskQueue, QSharedPointer<QFutureWatcherBase> & sceneFutureWatcher, QSharedPointer<QFutureWatcherBase> & treeFutureWatcher)
        : window{window}
        , context{engineWrapper.getContext()}
        , engine{engineWrapper.getEngine()}
        , taskQueue{taskQueue}
        , sceneFutureWatcher{sceneFutureWatcher}
        , treeFutureWatcher{treeFutureWatcher}
    {
        Q_ASSERT(window);
        Q_ASSERT(taskQueue);
        checkContext(window, context);
    }

    void unsetScene(bool * isUpdated)
    {
        if (scene) {
            if (renderer) {
                ASSERT(renderer.value().getScene() == scene);
                renderer.value().unsetScene();
            }
            scene.reset();
            if (isUpdated) {
                *isUpdated = true;
            }
            isDirty = true;
        }
        if (sceneFutureWatcher) {
            sceneFutureWatcher->cancel();
            sceneFutureWatcher.clear();
        }
    }

    QString updateScene(const std::filesystem::path & scenePath, bool * isUpdated)
    {
        ASSERT(!std::empty(scenePath));

        if (sceneFutureWatcher) {
            auto s = sceneFutureWatcher.dynamicCast<SceneFutureWatcher>();
            Q_ASSERT(s);
            auto future = s->future();
            if (future.isCanceled()) {
                sceneFutureWatcher.clear();
                if (future.isValid()) {
                    try {
                        (void)future.result();
                    } catch (const std::exception & e) {
                        return u"Exception: %1"_s.arg(QString::fromUtf8(e.what()));
                    }
                }
                return u"Cancelled"_s;
            }
            if (future.isFinished()) {
                auto newScene = future.result();
                if (newScene != scene) {
                    scene = std::move(newScene);
                    if (isUpdated) {
                        *isUpdated = true;
                    }
                    isDirty = true;
                    return {};
                }
            }
        }
        if (!scene || (scene->scenePath != scenePath)) {
            unsetScene(isUpdated);
            const auto buidScene = [&engine = engine, scenePath](QPromise<ScenePtr> & promise)
            {
                ElapsedTimer elapsedTimer{viewerRenderNodeCategory, u"Build scene '%1'"_s.arg(QString::fromStdString(scenePath.native()))};
                if (promise.isCanceled()) {
                    return;
                }
                if (auto scene = engine.getScenes().getScene(scenePath)) {
                    promise.addResult(std::move(scene));
                }
            };
            QFileInfo sceneFileInfo{scenePath};
            sceneFutureWatcher = taskQueue->runTask(sceneFileInfo.fileName(), sceneFileInfo.filePath(), std::move(buidScene));
        }
        return {};
    }

    void unsetTree(bool * isUpdated)
    {
        if (tree) {
            tree.reset();
            if (isUpdated) {
                *isUpdated = true;
            }
            isDirty = true;
        }
        if (treeFutureWatcher) {
            treeFutureWatcher->cancel();
            treeFutureWatcher.clear();
        }
    }

    QString updateTree(float emptinessFactor, float traversalCost, float intersectionCost, uint32_t maxDepth, bool * isUpdated)
    {
        if (!scene) {
            unsetTree(isUpdated);
            return {};
        }
        if (treeFutureWatcher) {
            auto t = treeFutureWatcher.dynamicCast<FutureWatcher>();
            Q_ASSERT(t);
            auto future = t->future();
            if (future.isCanceled()) {
                treeFutureWatcher.clear();
                if (future.isValid()) {
                    try {
                        (void)future.result();
                    } catch (const std::exception & e) {
                        return u"Exception: %1"_s.arg(QString::fromUtf8(e.what()));
                    }
                }
                return u"Cancelled"_s;
            }
            if (!future.isResultReadyAt(0) || !future.isValid()) {
                return {};
            }
            try {
                auto newTree = future.result();
                if (newTree != tree) {
                    tree = std::move(newTree);
                    if (isUpdated) {
                        *isUpdated = true;
                    }
                    isDirty = true;
                    return {};
                }
            } catch (const std::bad_alloc & e) {
                return QString::fromUtf8(e.what());
            }
        }
        const builder::Tree::Settings treeSettings = {
            .emptinessFactor = emptinessFactor,
            .traversalCost = traversalCost,
            .intersectionCost = intersectionCost,
            .maxDepth = utils::autoCast(maxDepth),
        };
        if (!tree || (tree->getSettings() != treeSettings)) {
            unsetTree(isUpdated);
            auto scenePath = QString::fromStdString(scene->scenePath.native());
            const auto buildTree = [&engine = engine, scenePath, scene = scene, treeSettings](QPromise<TreePtr> & promise) mutable
            {
                ElapsedTimer elapsedTimer{viewerRenderNodeCategory, u"Build SAH kd-tree for '%1'"_s.arg(scenePath)};
                if (promise.isCanceled()) {
                    return;
                }
                const auto cancel = [&promise]
                {
                    promise.suspendIfRequested();
                    return promise.isCanceled();
                };
                if (auto tree = engine.getBuilder().build(treeSettings, scene->sceneData, cancel)) {
                    promise.addResult(std::make_shared<builder::Tree>(std::move(tree).value()));
                }
            };
            QFileInfo sceneFileInfo{scenePath};
            auto description = u"%1: emptinessFactor %2, traversalCost: %3, intersectionCost: %4, maxDepth: %5"_s  //
                                   .arg(sceneFileInfo.filePath())                                                  //
                                   .arg(utils::safeCast<double>(treeSettings.emptinessFactor))                     //
                                   .arg(utils::safeCast<double>(treeSettings.traversalCost))                       //
                                   .arg(utils::safeCast<double>(treeSettings.intersectionCost))                    //
                                   .arg(treeSettings.maxDepth);                                                    //
            treeFutureWatcher = taskQueue->runTask(sceneFileInfo.fileName(), qMove(description), std::move(buildTree));
        }
        return {};
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

    void updateClearColor(const glm::vec4 & clearColor)
    {
        UPDATE_STATE(frameSettings.clearColor, clearColor);
    }

    void updateRenderdocCaptureFrameCounter(int renderdocCaptureFrameCounter)
    {
        UPDATE_STATE(this->renderdocCaptureFrameCounter, renderdocCaptureFrameCounter);
    }
#undef UPDATE_STATE

    bool resetDirty()
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

        auto [w, h] = scissorRect.bottomRight();
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

RenderNode::RenderNode(QQuickWindow * window, const EngineWrapper & engineWrapper, TaskQueue * taskQueue, QSharedPointer<QFutureWatcherBase> & sceneFutureWatcher, QSharedPointer<QFutureWatcherBase> & treeFutureWatcher)
    : impl_{window, engineWrapper, taskQueue, sceneFutureWatcher, treeFutureWatcher}
{}

void RenderNode::unsetScene(bool * isUpdated)
{
    return impl_->unsetScene(isUpdated);
}

QString RenderNode::updateScene(const std::filesystem::path & scenePath, bool * isUpdated)
{
    return impl_->updateScene(scenePath, isUpdated);
}

const std::shared_ptr<const Scene> & RenderNode::getScene() const &
{
    return impl_->scene;
}

void RenderNode::unsetTree(bool * isUpdated)
{
    return impl_->unsetTree(isUpdated);
}

QString RenderNode::updateTree(float emptinessFactor, float traversalCost, float intersectionCost, uint32_t maxDepth, bool * isUpdated)
{
    return impl_->updateTree(emptinessFactor, traversalCost, intersectionCost, maxDepth, isUpdated);
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

void RenderNode::updateClearColor(const QColor & clearColor)
{
    float r, g, b, a;
    clearColor.getRgbF(&r, &g, &b, &a);
    return impl_->updateClearColor({r, g, b, a});
}

void RenderNode::updateRenderdocCaptureFrameCounter(int renderdocCaptureFrameCounter)
{
    return impl_->updateRenderdocCaptureFrameCounter(renderdocCaptureFrameCounter);
}

void RenderNode::updateDirty()
{
    if (!impl_->resetDirty()) {
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
        const QMatrix4x4::Flags modelViewMatrixFlags = matrix()->flags();
        if ((modelViewMatrixFlags < QMatrix4x4::Flag::Rotation) && (modelViewMatrixFlags & QMatrix4x4::Flag::Rotation2D)) {
            isAxisAligned = true;
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
