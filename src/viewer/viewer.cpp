#include <engine/device.hpp>
#include <engine/engine.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <viewer/engine_wrapper.hpp>
#include <viewer/renderer.hpp>
#include <viewer/viewer.hpp>

#include <fmt/std.h>
#include <glm/gtx/matrix_operation.hpp>
#include <glm/gtx/matrix_transform_2d.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/mat3x3.hpp>
#include <glm/trigonometric.hpp>
#include <glm/vec3.hpp>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <QtCore/QDebug>
#include <QtCore/QFileInfo>
#include <QtCore/QLoggingCategory>
#include <QtCore/QObject>
#include <QtCore/QRunnable>
#include <QtGui/QVulkanInstance>
#include <QtQuick/QQuickItem>
#include <QtQuick/QQuickWindow>
#include <QtQuick/QSGRenderNode>
#include <QtQuick/QSGRendererInterface>
#include <QtWidgets/QApplication>

#include <chrono>
#include <memory>

#include <cstdint>

using namespace Qt::StringLiterals;

namespace viewer
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(viewerCategory)
Q_LOGGING_CATEGORY(viewerCategory, "viewer.viewer")

class CleanupJob : public QRunnable
{
public:
    CleanupJob(std::unique_ptr<Renderer> && renderer) : renderer{std::move(renderer)}
    {}

    void run() override
    {
        renderer.reset();
    }

private:
    std::unique_ptr<Renderer> renderer;
};

}  // namespace

Viewer::Viewer()
{
    setFlag(QQuickItem::Flag::ItemHasContents);

    connect(this, &QQuickItem::windowChanged, this, &Viewer::onWindowChanged);

    setAcceptedMouseButtons(Qt::MouseButton::LeftButton);
    // setFocus(true);

    Q_CHECK_PTR(doubleClickTimer);
    doubleClickTimer->setInterval(qApp->doubleClickInterval());
    doubleClickTimer->setSingleShot(true);
    connect(doubleClickTimer, &QTimer::timeout, this, [this] { setCursor(Qt::CursorShape::BlankCursor); });
}

void Viewer::rotate(QVector3D tiltPanRoll)
{
    setEulerAngles(eulerAngles + tiltPanRoll);
}

void Viewer::rotate(QVector2D tiltPan)
{
    setEulerAngles(QVector3D(eulerAngles.toVector2D() + tiltPan));
}

void Viewer::rotate(qreal tilt, qreal pan, qreal roll)
{
    setEulerAngles(QVector3D{utils::autoCast(tilt), utils::autoCast(pan), utils::autoCast(roll)});
}

Viewer::~Viewer() = default;

void Viewer::onWindowChanged(QQuickWindow * w)
{
    if (!w) {
        qCDebug(viewerCategory) << "Window is lost";
        return;
    }

    INVARIANT(w->graphicsApi() == QSGRendererInterface::GraphicsApi::Vulkan, "Expected Vulkan backend");

    connect(w, &QQuickWindow::beforeSynchronizing, this, &Viewer::sync, Qt::DirectConnection);
    connect(w, &QQuickWindow::sceneGraphInvalidated, this, &Viewer::cleanup, Qt::DirectConnection);
    connect(w, &QQuickWindow::beforeRendering, this, &Viewer::frameStart, Qt::DirectConnection);
    connect(w, &QQuickWindow::beforeRenderPassRecording, this, &Viewer::beforeRenderPassRecording, Qt::DirectConnection);
}

void Viewer::sync()
{
    if (!renderer) {
        return;
    }

    if (boundingRect().isEmpty()) {
        return;
    }

    renderer->setT(t);

    auto alpha = opacity();
    auto scaleFactor = scale();
    auto angle = rotation();

    for (auto p = parentItem(); p; p = p->parentItem()) {
        alpha *= p->opacity();
        scaleFactor *= p->scale();
        angle += p->rotation();
    }

    renderer->setAlpha(alpha);

    auto viewportRect = mapRectToScene(boundingRect());
    auto devicePixelRatio = window()->effectiveDevicePixelRatio();
    renderer->setViewportRect({viewportRect.topLeft() * devicePixelRatio, viewportRect.size() * devicePixelRatio});

    glm::dmat3 viewTransform = glm::diagonal3x3(glm::dvec3{scaleFactor});
    viewTransform = glm::scale(viewTransform, glm::dvec2{viewportRect.height() / viewportRect.width(), 1.0});
    viewTransform = glm::rotate(viewTransform, glm::radians(angle));
    viewTransform = glm::scale(viewTransform, glm::dvec2{width() / viewportRect.height(), height() / viewportRect.height()});
    // qCDebug(viewerCategory) << u"view transform matrix: %1"_s.arg(QString::fromStdString(glm::to_string(viewTransform)));
    renderer->setViewTransform(viewTransform);
}

void Viewer::cleanup()
{
    renderer.reset();
}

void Viewer::frameStart()
{
    if (!engine) {
        return;
    }

    if (scenePath.isEmpty()) {
        renderer.reset();
        return;
    }

    if (!scenePath.isLocalFile()) {
        renderer.reset();
        SPDLOG_WARN("scenePath URL is not local file", scenePath.toString().toStdString());
        return;
    }

    QFileInfo sceneFileInfo{scenePath.toLocalFile()};
    if (renderer && (renderer->getScenePath() != sceneFileInfo.filesystemFilePath())) {
        renderer.reset();
    }

    if (!renderer) {
        checkEngine();
        auto token = objectName();
        INVARIANT(!token.isEmpty(), "Viewer objectName should not be empty");
        renderer = std::make_unique<Renderer>(token.toStdString(), sceneFileInfo.filesystemFilePath(), engine->getEngine(), engine->getSceneManager());
    }

    if (boundingRect().isEmpty()) {
        return;
    }

    renderer->frameStart(window()->graphicsStateInfo());
}

void Viewer::beforeRenderPassRecording()
{
    if (!renderer) {
        return;
    }

    if (boundingRect().isEmpty()) {
        return;
    }

    if (!isVisible()) {
        return;
    }

    auto w = window();

    w->beginExternalCommands();
    {
        INVARIANT(engine, "");
        auto ri = w->rendererInterface();
        auto commandBuffer = static_cast<vk::CommandBuffer *>(ri->getResource(w, QSGRendererInterface::Resource::CommandListResource));
        Q_CHECK_PTR(commandBuffer);
        auto renderPass = static_cast<vk::RenderPass *>(ri->getResource(w, QSGRendererInterface::Resource::RenderPassResource));
        Q_CHECK_PTR(renderPass);

        auto & device = engine->getEngine().getDevice();
        device.setDebugUtilsObjectName(*commandBuffer, "Qt command buffer");
        device.setDebugUtilsObjectName(*renderPass, "Qt render pass");

        renderer->render(*commandBuffer, *renderPass, w->graphicsStateInfo());
    }
    w->endExternalCommands();
}

void Viewer::setEulerAngles(QVector3D newEulerAngles)
{
    float & roll = newEulerAngles[2];
    if (roll > 180.0f) {
        roll -= 360.0f;
    } else if (roll < -180.0f) {
        roll += 360.0f;
    }

    float & pitch = newEulerAngles[0];
    constexpr float kPitchMax = 89.9f;
    if (pitch > kPitchMax) {
        pitch = kPitchMax;
    } else if (pitch < -kPitchMax) {
        pitch = -kPitchMax;
    }

    float & yaw = newEulerAngles[1];
    if (yaw > 180.0f) {
        yaw -= 360.0f;
    } else if (yaw < -180.0f) {
        yaw += 360.0f;
    }

    if (qFuzzyCompare(newEulerAngles, eulerAngles)) {
        return;
    }

    eulerAngles = newEulerAngles;
    Q_EMIT eulerAnglesChanged(newEulerAngles);
}

void Viewer::checkEngine() const
{
    Q_CHECK_PTR(engine);

    auto w = window();
    auto ri = w->rendererInterface();

    auto vulkanInstance = static_cast<QVulkanInstance *>(ri->getResource(w, QSGRendererInterface::Resource::VulkanInstanceResource));
    Q_CHECK_PTR(vulkanInstance);

    auto vulkanPhysicalDevice = static_cast<vk::PhysicalDevice *>(ri->getResource(w, QSGRendererInterface::Resource::PhysicalDeviceResource));
    Q_CHECK_PTR(vulkanPhysicalDevice);

    auto vulkanDevice = static_cast<vk::Device *>(ri->getResource(w, QSGRendererInterface::Resource::DeviceResource));
    Q_CHECK_PTR(vulkanInstance);

    uint32_t queueFamilyIndex = 0;  // chosen by smart heuristics

    auto vulkanQueue = static_cast<vk::Queue *>(ri->getResource(w, QSGRendererInterface::Resource::CommandQueueResource));
    Q_CHECK_PTR(vulkanInstance);

#define GET_INSTANCE_PROC_ADDR(name) PFN_##name name = utils::autoCast(vulkanInstance->getInstanceProcAddr(#name))
    // GET_INSTANCE_PROC_ADDR(vkGetInstanceProcAddr);
    GET_INSTANCE_PROC_ADDR(vkGetDeviceProcAddr);
#undef GET_INSTANCE_PROC_ADDR
    PFN_vkGetDeviceQueue vkGetDeviceQueue = utils::autoCast(vkGetDeviceProcAddr(*vulkanDevice, "vkGetDeviceQueue"));

    const auto & e = engine->getEngine();
    INVARIANT(vk::Instance(vulkanInstance->vkInstance()) == e.getVulkanInstance(), "Should match");
    INVARIANT(*vulkanPhysicalDevice == e.getVulkanPhysicalDevice(), "Should match");
    INVARIANT(*vulkanDevice == e.getVulkanDevice(), "Should match");
    INVARIANT(queueFamilyIndex == e.getVulkanGraphicsQueueFamilyIndex(), "Should match");
    {
        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(*vulkanDevice, queueFamilyIndex, e.getVulkanGraphicsQueueIndex(), &queue);
        INVARIANT(*vulkanQueue == vk::Queue(queue), "Should match");
    }

    e.getDevice().setDebugUtilsObjectName(*vulkanQueue, "Qt graphical queue");
}

void Viewer::handleKeyEvent(QKeyEvent * event, bool isPressed)
{
    auto key = Qt::Key(event->key());
    switch (key) {
    case Qt::Key_W:
    case Qt::Key_A:
    case Qt::Key_S:
    case Qt::Key_D: {
        if (event->isAutoRepeat()) {
            break;
        }
        if (isPressed) {
            if (pressedKeys.contains(key)) {
                qCWarning(viewerCategory) << u"Key is already pressed:"_s << key;
            } else {
                pressedKeys.insert(key);
            }
        } else {
            if (!pressedKeys.remove(key)) {
                qCWarning(viewerCategory) << u"Key is not pressed:"_s << key;
            }
        }
        event->accept();
        return;
    }
    default: {
        break;
    }
    }
    event->ignore();
}

void Viewer::releaseResources()
{
    window()->scheduleRenderJob(new CleanupJob{std::move(renderer)}, QQuickWindow::BeforeSynchronizingStage);
}

void Viewer::wheelEvent(QWheelEvent * event)
{
    constexpr qreal kUnitsPerDegree = 8.0;
    auto numDegrees = QPointF(event->angleDelta()) / kUnitsPerDegree;
    if ((keyboardModifiers & Qt::KeyboardModifier::ShiftModifier)) {
        rotate(0.0f, numDegrees.x(), numDegrees.y());
    } else {
        auto newFieldOfView = qBound<qreal>(5.0, fieldOfView * numDegrees.y() / 3.0, 175.0);
        if (!setProperty("fieldOfView", newFieldOfView)) {
            qFatal("unreachable");
        }
    }
    event->accept();
    update();
}

void Viewer::mouseUngrabEvent()
{
    unsetCursor();
    update();
    return QQuickItem::mouseUngrabEvent();
}

void Viewer::mousePressEvent(QMouseEvent * event)
{
    switch (event->button()) {
    case Qt::MouseButton::LeftButton: {
        doubleClickTimer->start();
        startPos = QCursor::pos();
        event->accept();
        break;
    }
    default: {
        break;
    }
    }
    if (event->isAccepted()) {
        update();
    } else {
        return QQuickItem::mousePressEvent(event);
    }
}

void Viewer::mouseMoveEvent(QMouseEvent * event)
{
    if (event->buttons() & Qt::MouseButton::LeftButton) {
        auto posDelta = QCursor::pos() - startPos;
        if (!posDelta.isNull()) {
            doubleClickTimer->stop();
            setCursor(Qt::CursorShape::BlankCursor);
            if (!size().isEmpty()) {
                auto roll = eulerAngles.z();
                qreal angularSpeed = mouseLookSpeed / qMax(1.0, qMin(width(), height()));
                QPointF tiltPan = posDelta;
                tiltPan = -QTransform{}.rotate(-roll).map(angularSpeed * tiltPan);
                rotate(tiltPan.y(), tiltPan.x());
            }
        }
        QCursor::setPos(startPos);
        startPos = QCursor::pos();
        event->accept();
    }
    if (event->isAccepted()) {
        update();
    } else {
        return QQuickItem::mouseMoveEvent(event);
    }
}

void Viewer::mouseReleaseEvent(QMouseEvent * event)
{
    switch (event->button()) {
    case Qt::MouseButton::LeftButton: {
        if (doubleClickTimer->isActive()) {
            doubleClickTimer->stop();
        } else {
            unsetCursor();
        }
        event->accept();
        break;
    }
    default: {
        break;
    }
    }
    if (event->isAccepted()) {
        update();
    } else {
        return QQuickItem::mouseReleaseEvent(event);
    }
}

void Viewer::mouseDoubleClickEvent(QMouseEvent * event)
{
    switch (event->button()) {
    case Qt::MouseButton::LeftButton: {
        doubleClickTimer->stop();
        event->accept();
        break;
    }
    default: {
        break;
    }
    }
    if (event->isAccepted()) {
        update();
    } else {
        return QQuickItem::mouseReleaseEvent(event);
    }
}

void Viewer::focusInEvent(QFocusEvent * event)
{
    qCDebug(viewerCategory) << event;
}

void Viewer::focusOutEvent(QFocusEvent * event)
{
    qCDebug(viewerCategory) << event;
}

void Viewer::keyPressEvent(QKeyEvent * event)
{
    qDebug() << Q_FUNC_INFO << event << objectName();
    handleKeyEvent(event, true);
    if (event->isAccepted()) {
        update();
    } else {
        return QQuickItem::keyPressEvent(event);
    }
}

void Viewer::keyReleaseEvent(QKeyEvent * event)
{
    handleKeyEvent(event, false);
    if (event->isAccepted()) {
        update();
    } else {
        return QQuickItem::keyReleaseEvent(event);
    }
}

}  // namespace viewer
