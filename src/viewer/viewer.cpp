#include <engine/context.hpp>
#include <engine/device.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <viewer/engine_wrapper.hpp>
#include <viewer/renderer.hpp>
#include <viewer/viewer.hpp>

#include <fmt/std.h>
#include <glm/ext/quaternion_float.hpp>
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
    explicit CleanupJob(std::unique_ptr<Renderer> && renderer) : renderer{std::move(renderer)}
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

    Q_CHECK_PTR(handleInputTimer);
    connect(handleInputTimer, &QTimer::timeout, this, &Viewer::handleInput);
    if (auto primaryScreen = qApp->primaryScreen()) {
        const auto onRefrashRateChaned = [this](qreal refreshRate)
        {
            Q_ASSERT(refreshRate > 0.0);
            dt = 1.0 / refreshRate;
            Q_EMIT dtChanged(dt);

            constexpr qreal kMsPerS = 1000.0;
            handleInputTimer->start(utils::safeCast<int>(kMsPerS * dt));
        };
        onRefrashRateChaned(primaryScreen->refreshRate());
        connect(primaryScreen, &QScreen::refreshRateChanged, handleInputTimer, onRefrashRateChaned);
    }

    connect(this, &Viewer::eulerAnglesChanged, this, &QQuickItem::update);
    connect(this, &Viewer::cameraPositionChanged, this, &QQuickItem::update);
    connect(this, &Viewer::fieldOfViewChanged, this, &QQuickItem::update);
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
    QVector3D tiltPanRoll{utils::autoCast(tilt), utils::autoCast(pan), utils::autoCast(roll)};
    setEulerAngles(eulerAngles + tiltPanRoll);
}

Viewer::~Viewer() = default;

void Viewer::onWindowChanged(QQuickWindow * w)
{
    if (!w) {
        qCDebug(viewerCategory) << "Window is lost";
        return;
    }

    INVARIANT(w->graphicsApi() == QSGRendererInterface::GraphicsApi::Vulkan, "Expected Vulkan backend");

    connect(w, &QQuickWindow::beforeSynchronizing, this, &Viewer::sync, Qt::ConnectionType::DirectConnection);
    connect(w, &QQuickWindow::sceneGraphInvalidated, this, &Viewer::cleanup, Qt::ConnectionType::DirectConnection);
    connect(w, &QQuickWindow::beforeRendering, this, &Viewer::beforeRendering, Qt::ConnectionType::DirectConnection);
    connect(w, &QQuickWindow::beforeRenderPassRecording, this, &Viewer::beforeRenderPassRecording, Qt::ConnectionType::DirectConnection);
}

void Viewer::sync()
{
    if (!frameSettings) {
        frameSettings = std::make_unique<FrameSettings>();
    }

    frameSettings->position = glm::vec3{cameraPosition.x(), cameraPosition.y(), cameraPosition.z()};
    auto orientation = QQuaternion::fromEulerAngles(eulerAngles);
    frameSettings->orientation = glm::quat{orientation.scalar(), orientation.x(), orientation.y(), orientation.z()};
    frameSettings->t = t;

    qreal alpha = opacity();
    for (auto p = parentItem(); p; p = p->parentItem()) {
        alpha *= p->opacity();
    }
    frameSettings->alpha = utils::autoCast(alpha);

    if (!boundingRect().isEmpty()) {
        qreal scaleFactor = scale();
        qreal angle = rotation();
        for (auto p = parentItem(); p; p = p->parentItem()) {
            scaleFactor *= p->scale();
            angle += p->rotation();
        }

        auto mappedBoundingRect = mapRectToScene(boundingRect());
        qreal devicePixelRatio = window()->effectiveDevicePixelRatio();
        QRectF viewportRect{mappedBoundingRect.topLeft() * devicePixelRatio, mappedBoundingRect.size() * devicePixelRatio};
        {
            qreal x = std::ceil(viewportRect.x());
            qreal y = std::ceil(viewportRect.y());
            qreal width = std::floor(viewportRect.width());
            qreal height = std::floor(viewportRect.height());

            frameSettings->viewport = vk::Viewport{
                .x = utils::autoCast(x),
                .y = utils::autoCast(y),
                .width = utils::autoCast(width),
                .height = utils::autoCast(height),
                .minDepth = 0.0f,
                .maxDepth = 1.0f,
            };
            frameSettings->scissor = vk::Rect2D{
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

        qreal aspectRatio = viewportRect.height() / viewportRect.width();
        glm::dmat3 transform2D = glm::diagonal3x3(glm::dvec3{aspectRatio * scaleFactor, scaleFactor, 0.0});
        transform2D = glm::rotate(transform2D, glm::radians(angle));
        transform2D = glm::scale(transform2D, glm::dvec2{width(), height()} / viewportRect.height());
        // qCDebug(viewerCategory) << u"view transform matrix: %1"_s.arg(QString::fromStdString(glm::to_string(transform2D)));
        frameSettings->transform2D = glm::mat3{transform2D};
    }
}

void Viewer::cleanup()
{
    renderer.reset();
}

void Viewer::beforeRendering()
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

    auto sceneFileInfo = QFileInfo{scenePath.toLocalFile()}.filesystemFilePath();
    if (renderer && (renderer->getScenePath() != sceneFileInfo)) {
        renderer.reset();
    }

    if (!renderer) {
        checkEngine();
        auto token = objectName();
        INVARIANT(!token.isEmpty(), "Viewer objectName should not be empty");
        renderer = std::make_unique<Renderer>(token.toStdString(), sceneFileInfo, engine->getContext(), engine->getSceneManager());
    }

    if (boundingRect().isEmpty()) {
        return;
    }

    auto graphicsStateInfo = window()->graphicsStateInfo();
    renderer->advance(utils::autoCast(graphicsStateInfo.framesInFlight));
}

void Viewer::beforeRenderPassRecording()
{
    if (!frameSettings) {
        return;
    }
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
        vk::CommandBuffer * commandBuffer = utils::autoCast(ri->getResource(w, QSGRendererInterface::Resource::CommandListResource));
        Q_CHECK_PTR(commandBuffer);
        vk::RenderPass * renderPass = utils::autoCast(ri->getResource(w, QSGRendererInterface::Resource::RenderPassResource));
        Q_CHECK_PTR(renderPass);

        auto & device = engine->getContext().getDevice();
        device.setDebugUtilsObjectName(*commandBuffer, "Qt command buffer");
        device.setDebugUtilsObjectName(*renderPass, "Qt render pass");

        auto [currentFrameSlot, framesInFlight] = w->graphicsStateInfo();
        renderer->render(*commandBuffer, *renderPass, utils::autoCast(currentFrameSlot), utils::autoCast(framesInFlight), *frameSettings);
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
    Q_EMIT eulerAnglesChanged(eulerAngles);
}

void Viewer::checkEngine() const
{
    Q_CHECK_PTR(engine);

    auto w = window();
    auto ri = w->rendererInterface();

    QVulkanInstance * vulkanInstance = utils::autoCast(ri->getResource(w, QSGRendererInterface::Resource::VulkanInstanceResource));
    Q_CHECK_PTR(vulkanInstance);

    vk::PhysicalDevice * vulkanPhysicalDevice = utils::autoCast(ri->getResource(w, QSGRendererInterface::Resource::PhysicalDeviceResource));
    Q_CHECK_PTR(vulkanPhysicalDevice);

    vk::Device * vulkanDevice = utils::autoCast(ri->getResource(w, QSGRendererInterface::Resource::DeviceResource));
    Q_CHECK_PTR(vulkanDevice);

    uint32_t * queueFamilyIndex = utils::autoCast(ri->getResource(w, QSGRendererInterface::Resource::GraphicsQueueFamilyIndexResource));
    Q_CHECK_PTR(queueFamilyIndex);

    uint32_t * queueIndex = utils::autoCast(ri->getResource(w, QSGRendererInterface::Resource::GraphicsQueueIndexResource));
    Q_CHECK_PTR(queueIndex);

    vk::Queue * vulkanQueue = utils::autoCast(ri->getResource(w, QSGRendererInterface::Resource::CommandQueueResource));
    Q_CHECK_PTR(vulkanQueue);

#define GET_INSTANCE_PROC_ADDR(name) PFN_##name name = utils::autoCast(vulkanInstance->getInstanceProcAddr(#name))
    // GET_INSTANCE_PROC_ADDR(vkGetInstanceProcAddr);
    GET_INSTANCE_PROC_ADDR(vkGetDeviceProcAddr);
#undef GET_INSTANCE_PROC_ADDR
    PFN_vkGetDeviceQueue vkGetDeviceQueue = utils::autoCast(vkGetDeviceProcAddr(*vulkanDevice, "vkGetDeviceQueue"));

    const auto & context = engine->getContext();
    INVARIANT(vk::Instance(vulkanInstance->vkInstance()) == context.getVulkanInstance(), "Should match");
    INVARIANT(*vulkanPhysicalDevice == context.getVulkanPhysicalDevice(), "Should match");
    INVARIANT(*vulkanDevice == context.getVulkanDevice(), "Should match");
    INVARIANT(*queueFamilyIndex == context.getVulkanGraphicsQueueFamilyIndex(), "Should match");
    INVARIANT(*queueIndex == context.getVulkanGraphicsQueueIndex(), "Should match");
    {
        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(*vulkanDevice, *queueFamilyIndex, *queueIndex, &queue);
        INVARIANT(*vulkanQueue == vk::Queue(queue), "Should match");
    }

    context.getDevice().setDebugUtilsObjectName(*vulkanQueue, "Qt graphical queue");
}

void Viewer::handleKeyEvent(QKeyEvent * event, bool isPressed)
{
    Qt::Key key = utils::autoCast(event->key());
    switch (key) {
    case Qt::Key_W:
    case Qt::Key_A:
    case Qt::Key_S:
    case Qt::Key_D:
    case Qt::Key_Q:
    case Qt::Key_E:
    case Qt::Key_Left:
    case Qt::Key_Right:
    case Qt::Key_Up:
    case Qt::Key_Down:
    case Qt::Key_Space: {
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

void Viewer::handleInput()
{
    if (pressedKeys.isEmpty()) {
        return;
    }
    if (pressedKeys.contains(Qt::Key_Space)) {
        eulerAngles = QVector3D{0.0f, 0.0f, 0.0f};
        Q_EMIT eulerAnglesChanged(eulerAngles);

        cameraPosition = QVector3D{0.0f, 0.0f, 0.0f};
        Q_EMIT cameraPositionChanged(cameraPosition);

        fieldOfView = kDefaultFov;
        Q_EMIT fieldOfViewChanged(fieldOfView);
        return;
    }
    QVector3D direction;
    float pan = 0.0f;
    float tilt = 0.0f;
    for (Qt::Key key : std::as_const(pressedKeys)) {
        switch (key) {
        case Qt::Key_W:
        case Qt::Key_A:
        case Qt::Key_S:
        case Qt::Key_D:
        case Qt::Key_Q:
        case Qt::Key_E: {
            switch (key) {
            case Qt::Key_W:
                direction[2] += 1.0f;
                break;
            case Qt::Key_A:
                direction[0] -= 1.0f;
                break;
            case Qt::Key_S:
                direction[2] -= 1.0f;
                break;
            case Qt::Key_D:
                direction[0] += 1.0f;
                break;
            case Qt::Key_Q:
                direction[1] -= 1.0f;
                break;
            case Qt::Key_E:
                direction[1] += 1.0f;
                break;
            default:
                Q_ASSERT(false);
            }
            break;
        }
        case Qt::Key_Left:
            pan -= 1.0f;
            break;
        case Qt::Key_Right:
            pan += 1.0f;
            break;
        case Qt::Key_Down:
            tilt -= 1.0f;
            break;
        case Qt::Key_Up:
            tilt += 1.0f;
            break;
        default: {
            Q_ASSERT(false);
        }
        }
    }
    auto rotation = QQuaternion::fromEulerAngles(eulerAngles);
    qreal speedModifier = (keyboardModifiers & Qt::ShiftModifier) ? 0.05 : 1.0;
    {
        direction.normalize();
        auto velocity = speedModifier * linearSpeed;
        cameraPosition += rotation.rotatedVector(direction) * (velocity * dt);
        Q_EMIT cameraPositionChanged(cameraPosition);
    }
    {
        qreal angularSpeed = speedModifier * keyboardLookSpeed;
        qreal roll = qDegreesToRadians(eulerAngles.z());
        rotate(angularSpeed * (tilt * qSin(roll) + pan * qCos(roll)) * dt, angularSpeed * (tilt * qCos(roll) - pan * qSin(roll)) * dt);
    }
}

void Viewer::releaseResources()
{
    window()->scheduleRenderJob(new CleanupJob{std::move(renderer)}, QQuickWindow::RenderStage::BeforeSynchronizingStage);
}

void Viewer::wheelEvent(QWheelEvent * event)
{
    constexpr qreal kUnitsPerDegree = 8.0;
    auto numDegrees = QPointF(event->angleDelta()) / kUnitsPerDegree;
    if ((keyboardModifiers & Qt::KeyboardModifier::ShiftModifier)) {
        rotate(0.0f, numDegrees.x(), numDegrees.y());
    } else {
        auto newFieldOfView = qBound<qreal>(5.0, fieldOfView * numDegrees.y() / 3.0, 175.0);
        fieldOfView = newFieldOfView;
        Q_EMIT fieldOfViewChanged(fieldOfView);
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
                // tiltPan = QTransform{}.rotate(-roll).map(angularSpeed * tiltPan);
                rotate(-tiltPan.y(), tiltPan.x());
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
