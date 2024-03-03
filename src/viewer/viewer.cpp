﻿#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/instance.hpp>
#include <engine/physical_device.hpp>
#include <scene/scene.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <viewer/engine_wrapper.hpp>
#include <viewer/renderer.hpp>
#include <viewer/scene_manager.hpp>
#include <viewer/utils.hpp>
#include <viewer/viewer.hpp>

#include <fmt/std.h>
#include <glm/ext/quaternion_float.hpp>
#include <glm/gtx/matrix_operation.hpp>
#include <glm/gtx/matrix_transform_2d.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/mat2x2.hpp>
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
#include <QtCore/QtMath>
#include <QtCore/QtMinMax>
#include <QtCore/QtNumeric>
#include <QtGui/QGuiApplication>
#include <QtGui/QStyleHints>
#include <QtGui/QVulkanInstance>
#include <QtQuick/QQuickItem>
#include <QtQuick/QQuickWindow>
#include <QtQuick/QSGRenderNode>
#include <QtQuick/QSGRendererInterface>

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

// https://bugreports.qt.io/browse/QTBUG-121137
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

    Q_CHECK_PTR(mousePressAndHoldTimer);
    mousePressAndHoldTimer->setInterval(qApp->styleHints()->mousePressAndHoldInterval());
    mousePressAndHoldTimer->setSingleShot(true);
    connect(mousePressAndHoldTimer, &QTimer::timeout, this, [this] { setCursor(Qt::CursorShape::BlankCursor); });

    Q_CHECK_PTR(handleInputTimer);
    if (auto primaryScreen = qApp->primaryScreen()) {
        const auto onRefrashRateChaned = [this](qreal refreshRate)
        {
            Q_ASSERT(refreshRate > 0.0);
            setDt(1.0 / refreshRate);

            constexpr qreal kMsPerS = 1000.0;
            handleInputTimer->setInterval(utils::safeCast<int>(kMsPerS * dt));
        };
        onRefrashRateChaned(primaryScreen->refreshRate());
        connect(primaryScreen, &QScreen::refreshRateChanged, this, onRefrashRateChaned);

        connect(handleInputTimer, &QTimer::timeout, this, &Viewer::handleInput);
        handleInputTimer->start();
    }

    connect(this, &Viewer::eulerAnglesChanged, this, &QQuickItem::update);
    connect(this, &Viewer::cameraPositionChanged, this, &QQuickItem::update);
    connect(this, &Viewer::fieldOfViewChanged, this, &QQuickItem::update);
}

Viewer::~Viewer() = default;

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

void Viewer::setEulerAngles(QVector3D eulerAngles)
{
    float & pitch = eulerAngles[0];
    float & yaw = eulerAngles[1];
    float & roll = eulerAngles[2];

    while (pitch > 180.0f) {
        pitch -= 360.0f;
    }
    while (pitch < -180.0f) {
        pitch += 360.0f;
    }
    if (pitch > 90.0f) {
        pitch = 180.0f - pitch;
        yaw += 180.0;
        roll += 180.0;
    } else if (pitch < -90.0f) {
        pitch = -180.0f - pitch;
        yaw -= 180.0;
        roll -= 180.0;
    }

    while (roll > 180.0f) {
        roll -= 360.0f;
    }
    while (roll < -180.0f) {
        roll += 360.0f;
    }

    while (yaw > 180.0f) {
        yaw -= 360.0f;
    }
    while (yaw < -180.0f) {
        yaw += 360.0f;
    }

    if (qFuzzyCompare(this->eulerAngles, eulerAngles)) {
        return;
    }
    this->eulerAngles = eulerAngles;
    Q_EMIT eulerAnglesChanged(eulerAngles);
}

void Viewer::setCameraPosition(QVector3D cameraPosition)
{
    if (qFuzzyCompare(this->cameraPosition, cameraPosition)) {
        return;
    }
    this->cameraPosition = cameraPosition;
    Q_EMIT cameraPositionChanged(cameraPosition);
}

void Viewer::setFieldOfView(qreal fieldOfView)
{
    fieldOfView = qBound<qreal>(5.0, fieldOfView, 175.0);

    if (qFuzzyCompare(this->fieldOfView, fieldOfView)) {
        return;
    }
    this->fieldOfView = fieldOfView;
    Q_EMIT fieldOfViewChanged(fieldOfView);
}

void Viewer::setDt(qreal dt)
{
    if (qFuzzyCompare(this->dt, dt)) {
        return;
    }
    this->dt = dt;
    Q_EMIT dtChanged(dt);
}

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
    frameSettings->useOffscreenTexture = useOffscreenTexture;

    if (currentScenePath != scenePath) {
        currentScenePath = scenePath;

        scene.reset();

        if (scenePath.isEmpty()) {
            return;
        }
        if (!scenePath.isLocalFile()) {
            SPDLOG_WARN("scenePath URL is not local file", scenePath.toString().toStdString());
            return;
        }
        auto filesystemScenePath = QFileInfo{scenePath.toLocalFile()}.filesystemFilePath();

        const auto & sceneManager = engine->getSceneManager();
        scene = sceneManager.getOrCreateScene(filesystemScenePath);
        if (!scene) {
            return;
        }

        const auto & [aabbMin, aabbMax] = scene->getScene().aabb;
        characteristicSize = glm::length(aabbMax - aabbMin);
        if (!setProperty("linearSpeed", utils::safeCast<qreal>(characteristicSize / 5.0f))) {
            qFatal("unreachable");
        }
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

    frameSettings->width = utils::autoCast(width());
    frameSettings->height = utils::autoCast(height());

    frameSettings->zNear = std::sqrt(std::numeric_limits<float>::epsilon()) * characteristicSize;
    frameSettings->zFar = characteristicSize;

    if (!boundingRect().isEmpty()) {
        auto mappedBoundingRect = mapRectToScene(boundingRect());
        qreal devicePixelRatio = window()->effectiveDevicePixelRatio();  // QT_SCALE_FACTOR
        QRectF viewportRect{mappedBoundingRect.topLeft() * devicePixelRatio, mappedBoundingRect.size() * devicePixelRatio};
        {
            qreal x = std::ceil(viewportRect.x());
            qreal y = std::ceil(viewportRect.y());
            qreal w = std::floor(viewportRect.width());
            qreal h = std::floor(viewportRect.height());

            frameSettings->scissor = {
                .offset = {
                    .x = utils::autoCast(x),
                    .y = utils::autoCast(y),
                },
                .extent = {
                    .width = utils::autoCast(w),
                    .height = utils::autoCast(h),
                },
            };

            y += h;
            h = -h;

#if GLM_FORCE_DEPTH_ZERO_TO_ONE
            constexpr float kMinDepth = 0.0f;
#else
            constexpr float kMinDepth = -1.0f;
#endif
            frameSettings->viewport = {
                .x = utils::autoCast(x),
                .y = utils::autoCast(y),
                .width = utils::autoCast(w),
                .height = utils::autoCast(h),
                .minDepth = kMinDepth,
                .maxDepth = 1.0f,
            };
        }

        qreal scaleFactor = scale();
        qreal rotationAngle = rotation();
        for (auto p = parentItem(); p; p = p->parentItem()) {
            scaleFactor *= p->scale();
            rotationAngle += p->rotation();
        }

        qreal aspectRatio = viewportRect.height() / viewportRect.width();
        glm::dmat3 equilized = glm::diagonal3x3(glm::dvec3{aspectRatio * scaleFactor, scaleFactor, 1.0});
        glm::dmat3 rotatedEquilized = glm::rotate(equilized, glm::radians(-rotationAngle));
        glm::dmat3 scaledRotatedEquilized = glm::scale(rotatedEquilized, glm::dvec2{width(), height()} / viewportRect.height());
        // qCDebug(viewerCategory) << u"view transform matrix: %1"_s.arg(QString::fromStdString(glm::to_string(scaledRotatedEquilized)));
        frameSettings->transform2D = glm::mat2{glm::mat3{scaledRotatedEquilized}};
    }

    frameSettings->fov = utils::autoCast(qDegreesToRadians(fieldOfView));
}

void Viewer::beforeRendering()
{
    if (!engine) {
        return;
    }

    auto graphicsStateInfo = window()->graphicsStateInfo();

    if (!renderer) {
        checkEngine();
        renderer = std::make_unique<Renderer>(engine->getContext(), utils::autoCast(graphicsStateInfo.framesInFlight));
    }

    if (renderer->getScene() != scene) {
        renderer->setScene(scene);
    }

    if (boundingRect().isEmpty()) {
        return;
    }
    renderer->advance(utils::autoCast(graphicsStateInfo.currentFrameSlot), *frameSettings);
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
        ASSERT(engine);
        auto ri = w->rendererInterface();
        vk::CommandBuffer * commandBuffer = utils::autoCast(ri->getResource(w, QSGRendererInterface::Resource::CommandListResource));
        Q_CHECK_PTR(commandBuffer);
        vk::RenderPass * renderPass = utils::autoCast(ri->getResource(w, QSGRendererInterface::Resource::RenderPassResource));
        Q_CHECK_PTR(renderPass);

        auto & device = engine->getContext().getDevice();
        device.setDebugUtilsObjectName(*commandBuffer, "Qt command buffer");
        device.setDebugUtilsObjectName(*renderPass, "Qt render pass");

        auto graphicsStateInfo = w->graphicsStateInfo();
        renderer->render(*commandBuffer, *renderPass, utils::autoCast(graphicsStateInfo.currentFrameSlot), *frameSettings);
    }
    w->endExternalCommands();
}

void Viewer::cleanup()
{
    releaseResources();
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
    INVARIANT(vk::Instance(vulkanInstance->vkInstance()) == context.getInstance().getInstance(), "Should match");
    INVARIANT(*vulkanPhysicalDevice == context.getPhysicalDevice().getPhysicalDevice(), "Should match");
    INVARIANT(*vulkanDevice == context.getDevice().getDevice(), "Should match");
    const auto & queueCreateInfo = context.getPhysicalDevice().externalGraphicsQueueCreateInfo;
    INVARIANT(*queueFamilyIndex == queueCreateInfo.familyIndex, "Should match");
    INVARIANT(*queueIndex == queueCreateInfo.index, "Should match");
    {
        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(*vulkanDevice, *queueFamilyIndex, *queueIndex, &queue);
        INVARIANT(*vulkanQueue == vk::Queue(queue), "Should match");
    }

    context.getDevice().setDebugUtilsObjectName(*vulkanQueue, "Qt graphical queue");
}

void Viewer::onKeyEvent(QKeyEvent * event, bool isPressed)
{
    keyboardModifiers = event->modifiers();
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
    case Qt::Key_Space:
    case Qt::Key_Z:
    case Qt::Key_R:
    case Qt::Key_X: {
        if (event->isAutoRepeat()) {
            break;
        }
        if (isPressed) {
            if (pressedKeys.contains(key)) {
                qCWarning(viewerCategory) << u"Key is already pressed:"_s << key;
            } else {
                pressedKeys.insert(key, 0);
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
        setEulerAngles({});
        setCameraPosition({});
        setFieldOfView(kDefaultFov);
        return;
    }
    QVector3D direction;
    float pan = 0.0f;
    float tilt = 0.0f;
    QMutableHashIterator<Qt::Key, int> it{pressedKeys};
    while (it.hasNext()) {
        auto curr = it.next();
        Qt::Key key = curr.key();
        switch (key) {
        case Qt::Key_W:
        case Qt::Key_A:
        case Qt::Key_S:
        case Qt::Key_D:
        case Qt::Key_Q:
        case Qt::Key_E:
        case Qt::Key_Z: {
            if (pressedKeys.contains(Qt::Key_Z)) {
                break;
            }
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
                ASSERT_MSG(false, "{}", key);
            }
            break;
        }
        case Qt::Key_Left:
        case Qt::Key_Right:
        case Qt::Key_Down:
        case Qt::Key_Up:
        case Qt::Key_R:
        case Qt::Key_X: {
            if (pressedKeys.contains(Qt::Key_R)) {
                break;
            }
            if (pressedKeys.contains(Qt::Key_X)) {
                break;
            }
            switch (key) {
            case Qt::Key_Left:
                tilt -= 1.0f;
                break;
            case Qt::Key_Right:
                tilt += 1.0f;
                break;
            case Qt::Key_Down:
                pan -= 1.0f;
                break;
            case Qt::Key_Up:
                pan += 1.0f;
                break;
            default:
                ASSERT_MSG(false, "{}", key);
            }
            break;
        }
        default:
            ASSERT_MSG(false, "{}", key);
        }
    }
    qreal speedModifier = 1.0;
    if (keyboardModifiers == Qt::KeyboardModifier::ShiftModifier) {
        speedModifier = 0.05;
    } else if (keyboardModifiers == Qt::KeyboardModifier::ControlModifier) {
        speedModifier = 5.0;
    }
    if (pressedKeys.contains(Qt::Key_Z)) {
        if (0 == pressedKeys[Qt::Key_Z]++) {
            setCameraPosition({});
        }
    } else {
        direction.normalize();
        auto velocity = speedModifier * linearSpeed;
        auto rotation = QQuaternion::fromEulerAngles(eulerAngles);
        auto newCameraPosition = cameraPosition + rotation.rotatedVector(direction) * (velocity * dt);
        setCameraPosition(newCameraPosition);
    }
    if (pressedKeys.contains(Qt::Key_R)) {
        if (0 == pressedKeys[Qt::Key_R]++) {
            setEulerAngles({-eulerAngles.x(), eulerAngles.y() + 180.0f, -eulerAngles.z()});
        }
    } else if (pressedKeys.contains(Qt::Key_X)) {
        if (0 == pressedKeys[Qt::Key_X]++) {
            constexpr auto roundToStraightAngle = [](float angle) -> float
            {
                return qRound(angle / 90.0f) * 90.0f;
            };
            setEulerAngles({roundToStraightAngle(eulerAngles.x()), roundToStraightAngle(eulerAngles.y()), roundToStraightAngle(eulerAngles.z())});
        }
    } else {
        qreal angularSpeed = speedModifier * keyboardLookSpeed;
        qreal roll = -qDegreesToRadians(eulerAngles.z());
        rotate(angularSpeed * (tilt * qSin(roll) - pan * qCos(roll)) * dt, angularSpeed * (tilt * qCos(roll) + pan * qSin(roll)) * dt);
    }
}

void Viewer::releaseResources()
{
    scene.reset();
    if (renderer) {
        window()->scheduleRenderJob(new CleanupJob{std::move(renderer)}, QQuickWindow::RenderStage::BeforeSynchronizingStage);
    }
}

void Viewer::wheelEvent(QWheelEvent * event)
{
    constexpr qreal kUnitsPerDegree = 8.0;
    auto numDegrees = QPointF(event->angleDelta()) / kUnitsPerDegree;
    if (keyboardModifiers & Qt::KeyboardModifier::ShiftModifier) {
        rotate(0.0f, numDegrees.x(), numDegrees.y());
    } else {
        constexpr qreal kUnitsPerStep = 15.0;
        qreal degreesPerStep = 5.0;
        if (keyboardModifiers & Qt::KeyboardModifier::ControlModifier) {
            degreesPerStep = 1.0;
        }
        setFieldOfView(fieldOfView + numDegrees.y() / (kUnitsPerStep / degreesPerStep));
    }
    event->accept();
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
        mousePressAndHoldTimer->start();
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
            mousePressAndHoldTimer->stop();
            setCursor(Qt::CursorShape::BlankCursor);
            if (!size().isEmpty()) {
                qreal angularSpeed = mouseLookSpeed / qMax(1.0, qMin(width(), height()));
                QPointF tiltPan = posDelta;
                tiltPan *= angularSpeed;
                auto roll = -eulerAngles.z();
                tiltPan = QTransform{}.rotate(roll).map(tiltPan);
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
        if (mousePressAndHoldTimer->isActive()) {
            mousePressAndHoldTimer->stop();
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
        mousePressAndHoldTimer->stop();
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

void Viewer::keyPressEvent(QKeyEvent * event)
{
    onKeyEvent(event, true);
    if (event->isAccepted()) {
        update();
    } else {
        return QQuickItem::keyPressEvent(event);
    }
}

void Viewer::keyReleaseEvent(QKeyEvent * event)
{
    onKeyEvent(event, false);
    if (event->isAccepted()) {
        update();
    } else {
        return QQuickItem::keyReleaseEvent(event);
    }
}

}  // namespace viewer
