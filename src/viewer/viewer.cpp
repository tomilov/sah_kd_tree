#include <builder/builder.hpp>
#include <compute/compute.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <viewer/engine.hpp>
#include <viewer/engine_wrapper.hpp>
#include <viewer/render_node.hpp>
#include <viewer/scenes.hpp>
#include <viewer/task_queue.hpp>
#include <viewer/utils.hpp>
#include <viewer/viewer.hpp>

#include <fmt/std.h>

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

#include <algorithm>
#include <chrono>
#include <iterator>
#include <limits>
#include <random>
#include <thread>

#include <cmath>

using namespace Qt::StringLiterals;
using namespace std::chrono_literals;

namespace viewer
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(viewerCategory)
Q_LOGGING_CATEGORY(viewerCategory, "viewer.viewer")

}  // namespace

SceneSettings::SceneSettings(QObject * parent)
    : QObject{parent}
{
    // onUrlChanged have to be postponed to a moment after QQmlApplicationEngine::objectCreated,
    // because VkDevice is not created at the ApplicationWindow construction time.
    // The moment is when event handling loop is started.
    if (!connect(this, &SceneSettings::urlChanged, this, &SceneSettings::onUrlChanged, Qt::ConnectionType::QueuedConnection)) {
        qFatal("unreachable");
    }
    if (!connect(this, &SceneSettings::sceneChanged, this, &SceneSettings::onTreeSettingsChanged, Qt::ConnectionType::QueuedConnection)) {
        qFatal("unreachable");
    }
    if (!connect(this, &SceneSettings::treeSettingsChanged, this, &SceneSettings::onTreeSettingsChanged, Qt::ConnectionType::QueuedConnection)) {
        qFatal("unreachable");
    }

    if ((false)) {
        const auto addTasks = [this]
        {
            if (!taskQueue) {
                return;
            }
            for (int64_t i = 0; i < 12; ++i) {
                auto taskWithPromise = [i = std::make_unique<int>(i)](QPromise<int> & promise) mutable  // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
                {
                    promise.suspendIfRequested();
                    if (promise.isCanceled()) {
                        return;
                    }
                    constexpr int kProgressRangeStart = 0;
                    constexpr int kProgressRangeStop = 100;
                    promise.setProgressRange(kProgressRangeStart, kProgressRangeStop);
                    promise.setProgressValueAndText(kProgressRangeStart, u"0%"_s);
                    std::mt19937 rng{utils::safeCast<std::mt19937::result_type>(*i)};
                    QList<int> indices(kProgressRangeStop - kProgressRangeStart);
                    std::iota(std::begin(indices), std::end(indices), 0);
                    std::shuffle(std::begin(indices), std::end(indices), rng);
                    auto index = std::begin(indices);
                    int progress = kProgressRangeStart;
                    while (++progress <= kProgressRangeStop) {
                        promise.suspendIfRequested();
                        if (promise.isCanceled()) {
                            return;
                        }
                        {  // work hard
                            std::this_thread::sleep_for(100ms);
                        }
                        const float numerator = utils::autoCast(progress - kProgressRangeStart);
                        const float denominator = utils::autoCast(kProgressRangeStop - kProgressRangeStart);
                        promise.setProgressValueAndText(progress, u"%1%"_s.arg(qRound(100.0f * numerator / denominator)));
                        Q_ASSERT(index != std::end(indices));
                        if (!promise.emplaceResultAt(*index++, progress)) {
                            qCWarning(viewerCategory).noquote() << u"Cannot add result %1 at index %2"_s.arg(progress).arg(*std::prev(index));
                        }
                    }
                };
                Q_ASSERT(taskQueue);
                tasks.append(taskQueue->runTask(u"(w/ promise) name %1"_s.arg(i), u"(w/ promise) description %1"_s.arg(i), std::move(taskWithPromise)));

                const auto task = []
                {
                    std::this_thread::sleep_for(10000ms);
                    return 0;
                };
                Q_ASSERT(taskQueue);
                tasks.append(taskQueue->runTask(u"(w/o promise) name %1"_s.arg(i), u"(w/o promise) description %1"_s.arg(i), task));
            }
        };
        QTimer::singleShot(1000, this, addTasks);
    }
}

QVector3D SceneSettings::getSceneAabbMin() const
{
    if (!sceneData) {
        return {};
    }
    const auto & aabbMin = sceneData->aabb.min;
    return {aabbMin.x, aabbMin.y, aabbMin.z};
}

QVector3D SceneSettings::getSceneAabbMax() const
{
    if (!sceneData) {
        return {};
    }
    const auto & aabbMax = sceneData->aabb.max;
    return {aabbMax.x, aabbMax.y, aabbMax.z};
}

int SceneSettings::getDepth() const &
{
    if (!tree) {
        return -1;
    }
    return utils::autoCast(tree->getLayerSizes().size());
}

void SceneSettings::resetUrl()
{
    if (url.isEmpty()) {
        return;
    }
    url.clear();
    Q_EMIT urlChanged();
}

void SceneSettings::updateScene()
{
    Q_CHECK_PTR(sceneFutureWatcher);
    Q_ASSERT(sceneFutureWatcher->isFinished());
    auto future = sceneFutureWatcher->future();
    Q_ASSERT(future.isValid());
    if (future.isCanceled()) {
        /*
        try {
            (void)future.result();
        } catch (const std::exception & e) {
            sceneStatus = u"Exception: %1"_s.arg(QString::fromUtf8(e.what()));
        }
        */
        {
            sceneData.reset();
            Q_EMIT sceneChanged();
        }
        sceneStatus = u"Cancelled"_s;
    } else {
        INVARIANT(future.isFinished(), "");
        if (future.isResultReadyAt(0)) {
            scene_data::SceneDataPtr newSceneData = future.takeResult();
            if (sceneData != newSceneData) {
                sceneData = std::move(newSceneData);
                Q_EMIT sceneChanged();
            }
        } else {
            sceneData.reset();
            Q_EMIT sceneChanged();
        }
        sceneStatus.clear();
    }
    Q_EMIT sceneStatusChanged();
};

void SceneSettings::onUrlChanged()
{
    if (sceneFutureWatcher) {
        if (!sceneFutureWatcher->disconnect(this)) {
            qFatal("unreachable");
        }
        sceneFutureWatcher->cancel();
        sceneFutureWatcher.clear();
    }
    if (url.isEmpty() || !url.isLocalFile()) {
        if (sceneData) {
            sceneData.reset();
            Q_EMIT sceneChanged();
        }
        if (url.isEmpty()) {
            sceneStatus.clear();
        } else {
            Q_ASSERT(!url.isLocalFile());
            sceneStatus = u"Scene URL is not local file: %1"_s.arg(url.toString());
        }
        Q_EMIT sceneStatusChanged();
        return;
    }
    QFileInfo sceneFileInfo{url.toLocalFile()};
    std::filesystem::path scenePath = sceneFileInfo.filesystemCanonicalFilePath();
    if (sceneData && (sceneData->name == scenePath)) {
        return;
    }
    Q_CHECK_PTR(engineWrapper);
    const auto buidSceneData = [this, sceneFileInfo, scenePath = std::move(scenePath)](QPromise<scene_data::SceneDataPtr> & promise)
    {
        ElapsedTimer elapsedTimer{viewerCategory, u"Build scene data '%1'"_s.arg(sceneFileInfo.canonicalFilePath())};
        promise.suspendIfRequested();
        if (promise.isCanceled()) {
            return;
        }
        if (auto sceneData = engineWrapper->getEngine().getScenes().getScene(scenePath)) {
            promise.addResult(std::move(sceneData));
        }
    };
    Q_ASSERT(taskQueue);
    sceneFutureWatcher = taskQueue->runTask(sceneFileInfo.fileName(), sceneFileInfo.filePath(), std::move(buidSceneData));
    if (!connect(sceneFutureWatcher.get(), &QFutureWatcherBase::finished, this, &SceneSettings::updateScene)) {
        qFatal("unreachable");
    }
}

void SceneSettings::updateTree()
{
    Q_CHECK_PTR(treeFutureWatcher);
    Q_ASSERT(treeFutureWatcher->isFinished());
    auto future = treeFutureWatcher->future();
    Q_ASSERT(future.isValid());
    if (future.isCanceled()) {
        treeStatus = u"Cancelled"_s;
    } else {
        INVARIANT(future.isFinished(), "");
        if (tree) {
            tree.reset();
            isTreeChanged = true;
        }
        if (future.isResultReadyAt(0)) {
            tree = future.takeResult();
            isTreeChanged = true;
            treeStatus.clear();
        } else {
            treeStatus = u"Reset"_s;
        }
        Q_EMIT treeChanged();
    }
    Q_EMIT treeStatusChanged();
}

void SceneSettings::onTreeSettingsChanged()
{
    if (treeFutureWatcher) {
        if (!treeFutureWatcher->disconnect(this)) {
            qFatal("unreachable");
        }
        treeFutureWatcher->cancel();
        treeFutureWatcher.clear();
    }
    if (!traceTree) {
        tree.reset();
        isTreeChanged = true;
        Q_EMIT treeChanged();
        return;
    }
    if (!sceneData) {
        if (tree) {
            tree.reset();
            isTreeChanged = true;
            Q_EMIT treeChanged();
        }
        if (!treeStatus.isEmpty()) {
            treeStatus.clear();
            Q_EMIT treeStatusChanged();
        }
        return;
    }
    const builder::Tree::Settings builderTreeSettings = {
        .emptinessFactor = emptinessFactor,
        .traversalCost = traversalCost,
        .intersectionCost = intersectionCost,
        .maxDepth = utils::autoCast(maxDepth),
    };
    if (tree && (tree->getSettings() == builderTreeSettings) && (tree->getCudaDevice() == *engineWrapper->getEngine().getCudaDevice()) && (tree->getSceneData() == sceneData)) {
        return;
    }
    auto scenePath = QString::fromStdString(sceneData->name);
    Q_CHECK_PTR(engineWrapper);
    const auto buildTree = [this, scenePath, sceneData = sceneData, builderTreeSettings](QPromise<builder::TreePtr> & promise)
    {
        ElapsedTimer elapsedTimer{viewerCategory, u"Build SAH kd-tree for '%1'"_s.arg(scenePath)};
        const int progressRange = utils::autoCast(builderTreeSettings.maxDepth);
        promise.setProgressRange(0, progressRange);
        const auto progress = [&promise, progressRange](size_t p)
        {
            const int progressValue = utils::autoCast(p);
            promise.setProgressValueAndText(qBound(0, progressValue, progressRange), u"%1/%2"_s.arg(progressValue).arg(progressRange));
            promise.suspendIfRequested();
            return promise.isCanceled();
        };
        progress(0);
        try {
            if (auto cudaDevice = engineWrapper->getEngine().getCudaDevice()) {
                builder::Tree tree{builderTreeSettings, *cudaDevice, sceneData, progress};
                if (!tree.isEmpty()) {
                    promise.addResult(builder::makeTreePtr(std::move(tree)));
                }
            }
        } catch (const std::exception & e) {
            qCCritical(viewerCategory).noquote() << QString::fromStdString(fmt::to_string(e));
            throw;
        }
    };
    QFileInfo sceneFileInfo{scenePath};
    auto name = u"%1 tree"_s.arg(sceneFileInfo.fileName());
    auto description = u"%1:\n\temptinessFactor %2,\n\ttraversalCost: %3,\n\tintersectionCost: %4,\n\tmaxDepth: %5"_s  //
                           .arg(sceneFileInfo.filePath())                                                              //
                           .arg(utils::safeCast<double>(builderTreeSettings.emptinessFactor))                          //
                           .arg(utils::safeCast<double>(builderTreeSettings.traversalCost))                            //
                           .arg(utils::safeCast<double>(builderTreeSettings.intersectionCost))                         //
                           .arg(builderTreeSettings.maxDepth);                                                         //
    Q_ASSERT(taskQueue);
    treeFutureWatcher = taskQueue->runTask(qMove(name), qMove(description), std::move(buildTree));
    if (!connect(treeFutureWatcher.get(), &QFutureWatcherBase::finished, this, &SceneSettings::updateTree)) {
        qFatal("unreachable");
    }
}

auto RendererSettings::getRenderMode() const -> RenderModeFlags
{
    return renderMode;
}

void RendererSettings::setRenderMode(RenderModeFlags newRenderMode)
{
    if (renderMode == newRenderMode) {
        return;
    }
    renderMode = newRenderMode;
    Q_EMIT settingsChanged();
}

void RendererSettings::renderdocCaptureFrame()
{
    ++renderdocCaptureFrameCounter;
}

void CameraView::shift(const QVector3D & direction)
{
    auto newPosition = position + orientation.rotatedVector(direction);
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
            yaw += 180.0f;
            roll += 180.0f;
        } else if (pitch < -90.0f) {
            pitch = -180.0f - pitch;
            yaw -= 180.0f;
            roll -= 180.0f;
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

void CameraView::widen(float angle)
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
        return utils::safeCast<float>(qRound(angle / 90.0f)) * 90.0f;
    };
    setOrientation(QQuaternion::fromEulerAngles(roundAngle(pitch), roundAngle(yaw), roundAngle(roll)));
}

void CameraView::reflectOrientation()
{
    float pitch, yaw, roll;
    orientation.getEulerAngles(&pitch, &yaw, &roll);
    setOrientation(QQuaternion::fromEulerAngles(-pitch, yaw + 180.0f, -roll));
}

void CameraController::resetSensitivity()
{
    sensitivity = kDefaultSensitivity;
}

void CameraController::resetSpeed()
{
    speed = kDefaultSpeed;
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
    connect(this, &QQuickItem::activeFocusChanged, onActiveFocusChanged);
    const auto onVisibleChanged = [this]
    {
        if (!isVisible()) {
            pressedKeys.clear();
        }
    };
    connect(this, &QQuickItem::visibleChanged, onVisibleChanged);

    const auto onSceneSettingsChanged = [this]
    {
        disconnect(sceneUrlChangedConnection);
        disconnect(sceneChangedConnection);
        disconnect(sceneStatusChangedConnection);
        disconnect(sceneSettingsTreeChangedConnection);
        disconnect(sceneSettingsTreeStatusChangedConnection);
        disconnect(sceneSettingsBuildSettingsChangedConnection);
        if (!sceneSettings) {
            return;
        }
        sceneUrlChangedConnection = connect(sceneSettings, &SceneSettings::urlChanged, this, &QQuickItem::update);
        sceneChangedConnection = connect(sceneSettings, &SceneSettings::sceneChanged, this, &QQuickItem::update);
        sceneStatusChangedConnection = connect(sceneSettings, &SceneSettings::sceneStatusChanged, this, &QQuickItem::update);
        sceneSettingsBuildSettingsChangedConnection = connect(sceneSettings, &SceneSettings::treeSettingsChanged, this, &QQuickItem::update);
        sceneSettingsTreeChangedConnection = connect(sceneSettings, &SceneSettings::treeChanged, this, &QQuickItem::update);
        sceneSettingsTreeStatusChangedConnection = connect(sceneSettings, &SceneSettings::treeStatusChanged, this, &QQuickItem::update);
    };
    connect(this, &Viewer::sceneSettingsChanged, onSceneSettingsChanged);
    connect(cameraView, &CameraView::viewChanged, this, &QQuickItem::update);
    connect(cameraController, &CameraController::controllerChanged, this, &QQuickItem::update);
    connect(rendererSettings, &RendererSettings::settingsChanged, this, &QQuickItem::update);

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
    connect(this, &QQuickItem::windowChanged, onWindowChanged);
}

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
    qreal tilt = 0.0;
    qreal pan = 0.0;
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
                pan -= 1.0;
                break;
            case Qt::Key_Right:
                pan += 1.0;
                break;
            case Qt::Key_Down:
                tilt += 1.0;
                break;
            case Qt::Key_Up:
                tilt -= 1.0;
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
        QPointF planeDirection{utils::safeCast<qreal>(direction.x()), utils::safeCast<qreal>(direction.y())};
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
        float dx = utils::autoCast(planeDirection.x());
        float dy = utils::autoCast(planeDirection.y());
        cameraView->rotate(dy * angularSpeed, dx * angularSpeed);
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
    constexpr qreal kUnitsPerDegree = 8.0;
    float angle = utils::safeCast<float>(event->angleDelta().y() / kUnitsPerDegree);
    if (keyboardModifiers == Qt::KeyboardModifier::ShiftModifier) {
        cameraView->roll(angle);
    } else {
        constexpr float kUnitsPerStep = 15.0f;
        constexpr float kDegreesPerStep = 5.0f;
        cameraView->widen(angle / (kUnitsPerStep / kDegreesPerStep));
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
                const auto screen = window()->screen();
                const qreal screenDensityX = screen->physicalDotsPerInchX();
                const qreal screenDensityY = screen->physicalDotsPerInchY();
                const float pixelRatio = utils::autoCast(window()->effectiveDevicePixelRatio());
                const float fovRatio = cameraView->getFovRatio();
                const float angularSpeed = cameraController->sensitivity * fovRatio * pixelRatio;
                const qreal pan = dragPosDelta.x();
                const qreal tilt = dragPosDelta.y();
                cameraView->rotate(utils::safeCast<float>(pan * screenDensityX) * angularSpeed, utils::safeCast<float>(tilt * screenDensityY) * angularSpeed);
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
    if (!window() || !engineWrapper) {
        return QQuickItem::updatePaintNode(old, updatePaintNodeData);
    }
    auto renderNode = static_cast<RenderNode *>(old);
    if (old) {
        Q_ASSERT(dynamic_cast<RenderNode *>(old));
    } else {
        renderNode = new RenderNode{sceneSettings->url.toString(), window(), *engineWrapper};
    }
    renderNode->updateScene(sceneSettings->sceneData);
    if (sceneSettings->isTreeChanged) {
        sceneSettings->isTreeChanged = false;
        renderNode->setTree(std::move(sceneSettings->tree));
    }
    renderNode->updateRect(boundingRect());
    const bool traceSahKdTree = sceneSettings->traceTree;
    const bool useOffscreenTexture = rendererSettings->renderMode & RendererSettings::RenderModeFlag::UseOffscreenTexture;
    const bool discardInvisible = rendererSettings->renderMode & RendererSettings::RenderModeFlag::DiscardInvisibleFragments;
    const bool wireFrame = rendererSettings->texturingMode == RendererSettings::TexturingMode::WireFrame;
    renderNode->updateMode(traceSahKdTree, useOffscreenTexture, discardInvisible, wireFrame);
    {
        const QVector3D sceneCenter = (sceneSettings->getSceneAabbMin() + sceneSettings->getSceneAabbMax()) / 2.0f;
        const float zFar = (sceneSettings->getSceneAabbMax() - sceneSettings->getSceneAabbMin()).length() + (cameraView->position - sceneCenter).length();
        const float zNear = zFar * std::numeric_limits<float>::epsilon() * 1000.0f;
        renderNode->updateCamera(cameraView->position, cameraView->orientation, cameraView->fov, zNear, zFar);
    }
    renderNode->updateClearColor(rendererSettings->clearColor);
    renderNode->updateRenderdocCaptureFrameCounter(rendererSettings->renderdocCaptureFrameCounter);
    renderNode->updateDirty();
    return renderNode;
}

void Viewer::releaseResources()
{
    // all graphical resources owned by RenderNode
}

}  // namespace viewer
