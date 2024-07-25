#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <viewer/engine.hpp>
#include <viewer/engine_wrapper.hpp>
#include <viewer/render_node.hpp>
#include <viewer/scenes.hpp>
#include <viewer/utils.hpp>
#include <viewer/viewer.hpp>

#include <QtCore/QDebug>
#include <QtCore/QFileInfo>
#include <QtCore/QLoggingCategory>
#include <QtCore/QPointF>
#include <QtCore/QStringList>
#include <QtCore/QtAssert>
#include <QtCore/QtLogging>
#include <QtCore/QtMath>
#include <QtCore/QtMinMax>
#include <QtCore/QtNumeric>
#include <QtCore/QtTypes>
#include <QtGui/QCursor>
#include <QtGui/QGuiApplication>
#include <QtGui/QKeyEvent>
#include <QtGui/QMouseEvent>
#include <QtGui/QQuaternion>
#include <QtGui/QScreen>
#include <QtGui/QStyleHints>
#include <QtGui/QTransform>
#include <QtGui/QVector3D>
#include <QtGui/QWheelEvent>
#include <QtQuick/QQuickItem>
#include <QtQuick/QQuickWindow>
#include <QtQuick/QSGNode>
#include <QtQuick/QSGRendererInterface>

#include <limits>
#include <utility>

#include <cmath>

using namespace Qt::StringLiterals;

namespace viewer
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(viewerCategory)
Q_LOGGING_CATEGORY(viewerCategory, "viewer.viewer")

}  // namespace

void SceneSettings::setUrl(const QUrl & newUrl)
{
    if (url == newUrl) {
        return;
    }
    isUrlChanged = true;
    url = newUrl;
    Q_EMIT urlChanged();
}

void SceneSettings::unsetUrl()
{
    if (url.isEmpty()) {
        return;
    }
    isUrlChanged = true;
    url.clear();
    Q_EMIT urlChanged();
}

void SceneSettings::setScene(EngineWrapper * engine, RenderNode & renderNode)
{
    if (url.isEmpty()) {
        return;
    }
    if (!url.isLocalFile()) {
        qCWarning(viewerCategory) << u"Scene URL is not local file:"_s << url;
        return;
    }
    QGuiApplication::setOverrideCursor(Qt::CursorShape::WaitCursor);
    const auto scenePath = QFileInfo{url.toLocalFile()}.filesystemCanonicalFilePath();
    auto scene = engine->getEngine().getScenes().getScene(scenePath);
    QGuiApplication::restoreOverrideCursor();
    if (!scene) {
        return;
    }
    {
        const auto & [aabbMin, aabbMax] = scene->sceneData.aabb;
        sceneAabbMin = {aabbMin.x, aabbMin.y, aabbMin.z};
        sceneAabbMax = {aabbMax.x, aabbMax.y, aabbMax.z};
        Q_EMIT settingsChanged();
    }
    renderNode.setScene(std::move(scene));
}

void SceneSettings::updateScene(EngineWrapper * engine, RenderNode & renderNode)
{
    if (!isUrlChanged) {
        return;
    }
    isUrlChanged = false;
    {
        sceneAabbMin = {};
        sceneAabbMax = {};
        Q_EMIT settingsChanged();
    }
    renderNode.unsetScene();
    setScene(engine, renderNode);
}

void RendererSettings::renderdocCaptureFrame()
{
    ++renderdocCaptureFrameCounter;
}

QString RendererSettings::getModeDescription() const
{
    QStringList mode;
    if (useOffscreenTexture) {
        mode << addRichTextColor(u"O"_s, u"fuchsia"_s);
    }
    if (discardInvisible) {
        mode << addRichTextColor(u"D"_s, u"blue"_s);
    }
    if (wireFrame) {
        mode << addRichTextColor(u"W"_s, u"green"_s);
    }
    return uR"xml(<b>%1</b>)xml"_s.arg(mode.join(QChar(u'|')));
}

QString RendererSettings::getModeDescriptionVerbose() const
{
    QStringList mode;
    if (useOffscreenTexture) {
        mode << addRichTextColor(u"Use offscreen texture"_s, u"fuchsia"_s);
    }
    if (discardInvisible) {
        mode << addRichTextColor(u"Discard invisible pixels"_s, u"blue"_s);
    }
    mode << addRichTextColor(u"%1 mode"_s.arg(wireFrame ? u"Wireframe"_s : u"Barycentric Color"_s), u"green"_s);
    return u"<b>%1</b>"_s.arg(mode.join(u" AND "_s));
}

void CameraView::shift(const QVector3D & direction)
{
    auto newPosition = position + orientation.rotatedVector(direction);

    const Viewer * viewer = qobject_cast<const Viewer *>(parent());
    Q_CHECK_PTR(viewer);
    const QVector3D & sceneAabbMin = viewer->scene->sceneAabbMin;
    const QVector3D & sceneAabbMax = viewer->scene->sceneAabbMax;
    float worldScale = viewer->scene->worldScale;
    if (!qFuzzyCompare(sceneAabbMin, sceneAabbMax)) {
        const QVector3D direction = position - 0.5f * (sceneAabbMin + sceneAabbMax);
        const float c = direction.length() - 0.5f * (sceneAabbMax - sceneAabbMin).length() * worldScale;
        if (c > 0.0f) {
            newPosition -= c * direction.normalized();
        }
    }

    setPosition(newPosition);
}

void CameraView::rotate(float pan, float tilt)
{
    if ((false)) {
        auto tiltRotation = QQuaternion::fromAxisAndAngle(1.0f, 0.0f, 0.0f, tilt);
        auto panRotation = QQuaternion::fromAxisAndAngle(0.0f, 1.0f, 0.0f, pan);
        setOrientation(orientation * panRotation * tiltRotation);
    } else {
        float pitch, yaw, roll;
        orientation.getEulerAngles(&pitch, &yaw, &roll);

        float screenRoll = qDegreesToRadians(roll);
        pitch += tilt * qCos(screenRoll) - pan * qSin(screenRoll);
        yaw += tilt * qSin(screenRoll) + pan * qCos(screenRoll);

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

        setOrientation(QQuaternion::fromEulerAngles(pitch, yaw, roll));
    }
}

void CameraView::roll(float angle)
{
    if ((false)) {
        auto rollRotation = QQuaternion::fromAxisAndAngle(0.0f, 0.0f, 1.0f, angle);
        setOrientation(orientation * rollRotation);
    } else {
        float pitch, yaw, roll;
        orientation.getEulerAngles(&pitch, &yaw, &roll);
        roll += angle;

        while (roll > 180.0f) {
            roll -= 360.0f;
        }
        while (roll < -180.0f) {
            roll += 360.0f;
        }

        setOrientation(QQuaternion::fromEulerAngles(pitch, yaw, roll));
    }
}

void CameraView::addFov(float angle)
{
    auto newFov = qBound<float>(5.0f, fov + angle, 175.0f);
    setFov(newFov);
}

float CameraView::getFovRatio() const
{
    return fov / kDefaultFov;
}

void CameraView::setPosition(const QVector3D & newPosition)
{
    if (qFuzzyCompare(position, newPosition)) {
        return;
    }
    position = newPosition;
    Q_EMIT viewChanged();
}

void CameraView::setOrientation(const QQuaternion & newOrientation)
{
    if (qFuzzyCompare(orientation, newOrientation)) {
        return;
    }
    orientation = newOrientation;
    Q_EMIT viewChanged();
}

void CameraView::setFov(float newFov)
{
    if (qFuzzyCompare(fov, newFov)) {
        return;
    }
    fov = newFov;
    Q_EMIT viewChanged();
}

void CameraView::resetPosition()
{
    setPosition({});
}

void CameraView::resetOrientation()
{
    setOrientation({});
}

void CameraView::resetFov()
{
    setFov(kDefaultFov);
}

void CameraView::alignOrientation()
{
    float pitch, yaw, roll;
    orientation.getEulerAngles(&pitch, &yaw, &roll);
    constexpr auto roundAngle = [](float angle) -> float
    {
        return qRound(angle / 90.0f) * 90.0f;
    };
    setOrientation(QQuaternion::fromEulerAngles(roundAngle(pitch), roundAngle(yaw), roundAngle(roll)));
}

void CameraView::reflectOrientation()
{
    float pitch, yaw, roll;
    orientation.getEulerAngles(&pitch, &yaw, &roll);
    setOrientation(QQuaternion::fromEulerAngles(-pitch, yaw + 180.0f, -roll));
}

QString CameraView::getDescription() const
{
    float pitch, yaw, roll;
    orientation.getEulerAngles(&pitch, &yaw, &roll);
    return u"xyz(%1, %2, %3) \x3C6\x3B8\x3C8(%4, %5, %6) fov(%7)"_s  //
        .arg(position.x(), 5, 'f', 3)                                //
        .arg(position.y(), 5, 'f', 3)                                //
        .arg(position.z(), 5, 'f', 3)                                //
        .arg(pitch, 5, 'f', 1)                                       //
        .arg(yaw, 5, 'f', 1)                                         //
        .arg(roll, 5, 'f', 1)                                        //
        .arg(fov, 5, 'f', 1);                                        //
}

void CameraController::resetSensitivity()
{
    sensitivity = kDefaultSensitivity;
}

void CameraController::resetSpeed()
{
    speed = kDefaultSpeed;
}

QString CameraController::getDescription() const
{
    return u"sens(%1) speed(%2)"_s    //
        .arg(sensitivity, 5, 'f', 4)  //
        .arg(speed, 5, 'f', 2);       //
}

Viewer::Viewer(QQuickItem * parent)
    : QQuickItem{parent}
{
    setFlag(QQuickItem::Flag::ItemHasContents);
    setFocus(true);
    setFocusPolicy(Qt::FocusPolicy::WheelFocus);
    setAcceptedMouseButtons(Qt::MouseButton::LeftButton);
    // setAcceptHoverEvents(true);

    Q_CHECK_PTR(mousePressAndHoldTimer);
    mousePressAndHoldTimer->setInterval(QGuiApplication::styleHints()->mousePressAndHoldInterval());
    mousePressAndHoldTimer->setSingleShot(true);
    connect(mousePressAndHoldTimer, &QTimer::timeout, this, [this] { setCursor(Qt::CursorShape::BlankCursor); });

    Q_CHECK_PTR(handleKeyboardInputTimer);
    const auto onPrimaryScreenChanged = [this](QScreen * primaryScreen)
    {
        disconnect(refreshRateConnection);
        if (!primaryScreen) {
            qCDebug(viewerCategory) << "primaryScreen is lost";
            handleKeyboardInputTimer->stop();
            return;
        }
        const auto onRefreshRateChanged = [this](qreal refreshRate)
        {
            constexpr qreal kMsPerS = 1000.0;
            Q_ASSERT(!qFuzzyIsNull(refreshRate));
            handleKeyboardInputTimer->start(qFloor(kMsPerS / refreshRate));
        };
        onRefreshRateChanged(primaryScreen->refreshRate());
        refreshRateConnection = connect(primaryScreen, &QScreen::refreshRateChanged, this, onRefreshRateChanged);
    };
    onPrimaryScreenChanged(qApp->primaryScreen());
    connect(qApp, &QGuiApplication::primaryScreenChanged, this, onPrimaryScreenChanged);
    connect(handleKeyboardInputTimer, &QTimer::timeout, this, &Viewer::handleKeyboardInput);
    const auto onActiveFocusChanged = [this](bool activeFocus)
    {
        if (!activeFocus) {
            pressedKeys.clear();
        }
    };
    connect(this, &QQuickItem::activeFocusChanged, this, onActiveFocusChanged);
    const auto onVisibleChanged = [this]
    {
        if (!isVisible()) {
            pressedKeys.clear();
        }
    };
    connect(this, &QQuickItem::visibleChanged, this, onVisibleChanged);

    connect(scene, &SceneSettings::urlChanged, this, &QQuickItem::update);
    connect(scene, &SceneSettings::settingsChanged, this, &QQuickItem::update);
    connect(cameraView, &CameraView::viewChanged, this, &QQuickItem::update);
    connect(cameraController, &CameraController::controllerChanged, this, &QQuickItem::update);
    connect(renderer, &RendererSettings::settingsChanged, this, &QQuickItem::update);

    const auto onWindowChanged = [this](QQuickWindow * window)
    {
        disconnect(sceneGraphInvalidatedConnection);
        if (!window) {
            qCDebug(viewerCategory) << "window is lost";
            return;
        }
        INVARIANT(window->graphicsApi() == QSGRendererInterface::GraphicsApi::Vulkan, "Expected Vulkan backend");
        const auto onSceneGraphInvalidated = [this]
        {
            releaseResources();
        };
        sceneGraphInvalidatedConnection = connect(window, &QQuickWindow::sceneGraphInvalidated, this, onSceneGraphInvalidated, Qt::ConnectionType::DirectConnection);
    };
    connect(this, &QQuickItem::windowChanged, this, onWindowChanged);
}

Viewer::~Viewer() = default;

void Viewer::handleKeyboardInput()
{
    if (pressedKeys.isEmpty()) {
        return;
    }
    if (pressedKeys.contains(Qt::Key_Space)) {
        cameraView->resetOrientation();
        cameraView->resetPosition();
        cameraView->resetFov();
        return;
    }
    QVector3D direction;
    float tilt = 0.0f;
    float pan = 0.0f;
    QHashIterator<Qt::Key, int> pressedKey{pressedKeys};
    while (pressedKey.hasNext()) {
        auto curr = pressedKey.next();
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
        case Qt::Key_Up:
        case Qt::Key_Down:
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
                pan -= 1.0f;
                break;
            case Qt::Key_Right:
                pan += 1.0f;
                break;
            case Qt::Key_Down:
                tilt += 1.0f;
                break;
            case Qt::Key_Up:
                tilt -= 1.0f;
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
    float speedModifier = 1.0f;
    if (keyboardModifiers == Qt::KeyboardModifier::ShiftModifier) {
        speedModifier = 0.05f;
    } else if (keyboardModifiers == Qt::KeyboardModifier::ControlModifier) {
        speedModifier = 5.0f;
    }
    QPointF ortX = mapToGlobal(1.0, 0.0) - mapToGlobal(0.0, 0.0);
    QTransform transform;
    transform.rotateRadians(qAtan2(ortX.y(), ortX.x()));
    if (pressedKeys.contains(Qt::Key_Z)) {
        if (0 == pressedKeys[Qt::Key_Z]++) {
            cameraView->resetPosition();
        }
    } else {
        QPointF planeDirection{direction.x(), direction.y()};
        planeDirection = transform.map(planeDirection);
        direction.setX(utils::autoCast(planeDirection.x()));
        direction.setY(utils::autoCast(planeDirection.y()));
        float refreshRate = utils::autoCast(window()->screen()->refreshRate());
        float step = speedModifier * cameraController->speed / refreshRate;
        cameraView->shift(direction.normalized() * step);
    }
    if (pressedKeys.contains(Qt::Key_R)) {
        if (0 == pressedKeys[Qt::Key_R]++) {
            cameraView->reflectOrientation();
        }
    } else if (pressedKeys.contains(Qt::Key_X)) {
        if (0 == pressedKeys[Qt::Key_X]++) {
            cameraView->alignOrientation();
        }
    } else {
        float angularSpeed = speedModifier;
        QPointF planeDirection{tilt, pan};
        planeDirection = transform.map(planeDirection);
        cameraView->rotate(planeDirection.y() * angularSpeed, planeDirection.x() * angularSpeed);
    }
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

void Viewer::wheelEvent(QWheelEvent * event)
{
    constexpr qreal kUnitsPerDegree = 8.0f;
    float angle = utils::safeCast<float>(event->angleDelta().y()) / kUnitsPerDegree;
    if (keyboardModifiers == Qt::KeyboardModifier::ShiftModifier) {
        cameraView->roll(angle);
    } else {
        constexpr float kUnitsPerStep = 15.0f;
        constexpr float kDegreesPerStep = 5.0f;
        cameraView->addFov(angle / (kUnitsPerStep / kDegreesPerStep));
    }
    event->accept();
}

void Viewer::mouseUngrabEvent()
{
    unsetCursor();
    return QQuickItem::mouseUngrabEvent();
}

void Viewer::mousePressEvent(QMouseEvent * event)
{
    switch (event->button()) {
    case Qt::MouseButton::LeftButton: {
        mousePressAndHoldTimer->start();
        startDragPos = mapFromGlobal(QCursor::pos().toPointF());
        event->accept();
        break;
    }
    default: {
        break;
    }
    }
    if (event->isAccepted()) {
        setKeepMouseGrab(true);
        update();
    } else {
        return QQuickItem::mousePressEvent(event);
    }
}

void Viewer::mouseMoveEvent(QMouseEvent * event)
{
    if (event->buttons() & Qt::MouseButton::LeftButton) {
        QPointF dragPosDelta = mapFromGlobal(QCursor::pos().toPointF()) - startDragPos;
        if (!dragPosDelta.isNull()) {
            mousePressAndHoldTimer->stop();
            setCursor(Qt::CursorShape::BlankCursor);
            if (!size().isEmpty()) {
                auto screen = window()->screen();
                float screenDensityX = utils::autoCast(screen->physicalDotsPerInchX());
                float screenDensityY = utils::autoCast(screen->physicalDotsPerInchY());
                float pixelRatio = window()->effectiveDevicePixelRatio();
                float fovRatio = cameraView->getFovRatio();
                float angularSpeed = cameraController->sensitivity * fovRatio * pixelRatio;
                float pan = utils::autoCast(dragPosDelta.x());
                float tilt = utils::autoCast(dragPosDelta.y());
                cameraView->rotate(pan * screenDensityX * angularSpeed, tilt * screenDensityY * angularSpeed);
            }
        }
        QCursor::setPos(mapToGlobal(startDragPos).toPoint());
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
        setKeepMouseGrab(false);
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

QSGNode * Viewer::updatePaintNode(QSGNode * old, UpdatePaintNodeData * updatePaintNodeData)
{
    if (!window() || !engine) {
        return QQuickItem::updatePaintNode(old, updatePaintNodeData);
    }
    auto node = static_cast<RenderNode *>(old);
    if (old) {
        Q_ASSERT(dynamic_cast<RenderNode *>(old));
        scene->updateScene(engine, *node);
    } else {
        node = new RenderNode{window(), engine};
        scene->setScene(engine, *node);
    }
    node->updateRect(boundingRect());
    node->updateMode(renderer->useOffscreenTexture, renderer->discardInvisible, renderer->wireFrame);
    {
        float zFar = (scene->sceneAabbMax - scene->sceneAabbMin).length() * scene->worldScale;
        float zNear = 2.0f * std::sqrt(std::numeric_limits<float>::epsilon()) * zFar;
        node->updateCamera(cameraView->position, cameraView->orientation, cameraView->fov, zNear, zFar);
    }
    node->setClearColor(renderer->clearColor);
    node->setRenderdocCaptureFrameCounter(renderer->renderdocCaptureFrameCounter);
    node->markDirty();
    return node;
}

void Viewer::releaseResources()
{
    // all resources owned by RenderNode
}

}  // namespace viewer
