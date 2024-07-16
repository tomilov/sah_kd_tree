#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/instance.hpp>
#include <engine/physical_device.hpp>
#include <scene_data/scene_data.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <viewer/engine.hpp>
#include <viewer/engine_wrapper.hpp>
#include <viewer/renderer.hpp>
#include <viewer/scenes.hpp>
#include <viewer/utils.hpp>
#include <viewer/viewer.hpp>

#include <fmt/std.h>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/quaternion_float.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_operation.hpp>
#include <glm/gtx/matrix_transform_2d.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/mat2x2.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/trigonometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <QtCore/QDebug>
#include <QtCore/QFileInfo>
#include <QtCore/QFlags>
#include <QtCore/QList>
#include <QtCore/QLoggingCategory>
#include <QtCore/QMetaType>
#include <QtCore/QObject>
#include <QtCore/QRectF>
#include <QtCore/QRunnable>
#include <QtCore/QSize>
#include <QtCore/QString>
#include <QtCore/QTypeInfo>
#include <QtCore/QVariant>
#include <QtCore/QtAssert>
#include <QtCore/QtLogging>
#include <QtCore/QtMath>
#include <QtCore/QtMinMax>
#include <QtCore/QtNumeric>
#include <QtGui/QCursor>
#include <QtGui/QGuiApplication>
#include <QtGui/QMatrix4x4>
#include <QtGui/QQuaternion>
#include <QtGui/QScreen>
#include <QtGui/QStyleHints>
#include <QtGui/QTransform>
#include <QtGui/QVector2D>
#include <QtGui/QVector3D>
#include <QtGui/QVulkanInstance>
#include <QtGui/rhi/qrhi.h>
#include <QtQml/QQmlProperty>
#include <QtQuick/QQuickItem>
#include <QtQuick/QQuickWindow>
#include <QtQuick/QSGNode>
#include <QtQuick/QSGRenderNode>
#include <QtQuick/QSGRendererInterface>
#include <QtQuick/QSGTextureProvider>
#include <QtQuick/QSGTransformNode>

#include <limits>
#include <memory>
#include <utility>

#include <cmath>
#include <cstdint>

using namespace Qt::StringLiterals;

namespace viewer
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(viewerCategory)
Q_LOGGING_CATEGORY(viewerCategory, "viewer.viewer")

void checkEngine(QQuickWindow * w, const engine::Context & context)
{
    Q_CHECK_PTR(w);

    auto ri = w->rendererInterface();

    QVulkanInstance * instance = utils::autoCast(ri->getResource(w, QSGRendererInterface::Resource::VulkanInstanceResource));
    Q_CHECK_PTR(instance);

    vk::PhysicalDevice * physicalDevice = utils::autoCast(ri->getResource(w, QSGRendererInterface::Resource::PhysicalDeviceResource));
    Q_CHECK_PTR(physicalDevice);

    vk::Device * device = utils::autoCast(ri->getResource(w, QSGRendererInterface::Resource::DeviceResource));
    Q_CHECK_PTR(device);

    uint32_t * queueFamilyIndex = utils::autoCast(ri->getResource(w, QSGRendererInterface::Resource::GraphicsQueueFamilyIndexResource));
    Q_CHECK_PTR(queueFamilyIndex);

    uint32_t * queueIndex = utils::autoCast(ri->getResource(w, QSGRendererInterface::Resource::GraphicsQueueIndexResource));
    Q_CHECK_PTR(queueIndex);

    vk::Queue * queue = utils::autoCast(ri->getResource(w, QSGRendererInterface::Resource::CommandQueueResource));
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

// https://bugreports.qt.io/browse/QTBUG-121137
class CleanupJob : public QRunnable
{
public:
    explicit CleanupJob(std::unique_ptr<Renderer> && renderer)
        : renderer{std::move(renderer)}
    {}

    void run() override
    {
        renderer.reset();
    }

private:
    std::unique_ptr<Renderer> renderer;
};

class RenderNode final : public QSGRenderNode
{
public:
    explicit RenderNode(QQuickWindow * window, EngineWrapper * engine, Renderer * renderer)
        : window{window}
        , engine{engine}
        , renderer{renderer}
    {
        ASSERT(window);
        ASSERT(engine);
        ASSERT(renderer);
    }

    void set(bool useOffscreenTexture, bool wireFrame)
    {
        frameSettings.useOffscreenTexture = useOffscreenTexture;
        frameSettings.wireFrame = wireFrame;
    }

    void setCamera(const glm::vec3 & position, const glm::quat & orientation, float fov, float zNear, float zFar)
    {
        frameSettings.position = position;
        frameSettings.orientation = orientation;
        frameSettings.fov = fov;
        frameSettings.zNear = zNear;
        frameSettings.zFar = zFar;
    }

    void setSize(QSizeF size)
    {
        this->size = size;
    }

private:
    QQuickWindow * const window;
    EngineWrapper * const engine;
    Renderer * const renderer;

    QSizeF size;
    FrameSettings frameSettings;

    QVector<quint32> renderPassFormat;

    [[nodiscard]] QRectF getScissorRect(const QSizeF & renderTargetSize, const QMatrix4x4 & mvp) const
    {
        QRectF scissorRect = mvp.mapRect({{}, size});  // in NDC, turn back to window coordinates
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

    void prepare() override
    {
        // renderTarget()->resourceType() == QRhiResource::TextureRenderTarget, vk::DynamicState::eViewport

        frameSettings.alpha = inheritedOpacity();

        frameSettings.width = utils::autoCast(size.width());
        frameSettings.height = utils::autoCast(size.height());

        const QSizeF renderTargetSize = renderTarget()->pixelSize();  // renderTarget()->devicePixelRatio() == 1.0f
        frameSettings.viewport = vk::Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = utils::autoCast(renderTargetSize.width()),
            .height = utils::autoCast(renderTargetSize.height()),
            .minDepth = engine::kMinDepth,
            .maxDepth = 1.0f,
        };

        const QMatrix4x4 mvp = *projectionMatrix() * *matrix();
        {
            const QRectF scissorRect = getScissorRect(renderTargetSize, mvp);
            frameSettings.scissor = vk::Rect2D{
                .offset = {
                    .x = utils::autoCast(scissorRect.x()),
                    .y = utils::autoCast(scissorRect.y()),
                },
                .extent = {
                    .width = utils::autoCast(scissorRect.width()),
                    .height = utils::autoCast(scissorRect.height()),
                },
            };
        }
        {
            glm::mat4 & transform2D = frameSettings.transform2D;
            transform2D = glm::make_mat4x4(mvp.constData());
            transform2D = glm::scale(transform2D, glm::vec3{frameSettings.width * 0.5f, frameSettings.height * 0.5f, 1.0f});
            transform2D = glm::translate(transform2D, glm::vec3{1.0f, 1.0f, 0.0f});
        }

        if (!matrix()->toTransform().isRotating()) {  // optimization for axis aligned case
            frameSettings.useOffscreenTexture = false;
        }
        renderer->setFrameSettings(frameSettings);

        int currentFrameSlot = window->graphicsStateInfo().currentFrameSlot;
        renderer->advance(utils::autoCast(currentFrameSlot));
    }

    void render([[maybe_unused]] const RenderState * renderState) override
    {
        auto commandBufferNativeHandles = commandBuffer()->nativeHandles();
        Q_CHECK_PTR(commandBufferNativeHandles);
        vk::CommandBuffer cb = static_cast<const QRhiVulkanCommandBufferNativeHandles *>(commandBufferNativeHandles)->commandBuffer;

        const auto & device = engine->getContext().getDevice();
        device.setDebugUtilsObjectName(cb, "Qt command buffer");

        auto renderPassDescriptor = renderTarget()->renderPassDescriptor();
        {
            auto newRenderPassFormat = renderPassDescriptor->serializedFormat();
            if (renderPassFormat != newRenderPassFormat) {
                renderPassFormat = std::move(newRenderPassFormat);
                qCInfo(viewerCategory) << u"Render pass format changed"_s;
            }
        }
        auto renderPassNativeHandles = renderPassDescriptor->nativeHandles();
        Q_CHECK_PTR(renderPassNativeHandles);
        vk::RenderPass renderPass = static_cast<const QRhiVulkanRenderPassNativeHandles *>(renderPassNativeHandles)->renderPass;

        int currentFrameSlot = window->graphicsStateInfo().currentFrameSlot;
        renderer->render(cb, renderPass, utils::autoCast(currentFrameSlot));
    }

    void releaseResources() override
    {
        // there is no resources
    }

    [[nodiscard]] RenderingFlags flags() const override
    {
        auto renderingFlags = QSGRenderNode::flags();
        if (frameSettings.useOffscreenTexture) {
            renderingFlags |= RenderingFlag::DepthAwareRendering;
            renderingFlags |= RenderingFlag::BoundedRectRendering;
        } else {
            renderingFlags |= RenderingFlag::OpaqueRendering;
        }
        // renderingFlags |= RenderingFlag::OpaqueRendering;
        return renderingFlags;
    }

    [[nodiscard]] QRectF rect() const override
    {
        if (flags() & RenderingFlag::BoundedRectRendering) {
            return {{}, size};
        }
        return QSGRenderNode::rect();
    }

    [[nodiscard]] StateFlags changedStates() const override
    {
        return StateFlag::ViewportState | StateFlag::ScissorState;
    }
};

}  // namespace

Viewer::Viewer(QQuickItem * parent)
    : QQuickItem{parent}
{
    setFlag(QQuickItem::Flag::ItemHasContents);

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

    connect(this, &Viewer::scenePathChanged, this, &QQuickItem::update);

    connect(this, &Viewer::useOffscreenTextureChanged, this, &QQuickItem::update);
    connect(this, &Viewer::wireFrameChanged, this, &QQuickItem::update);

    connect(this, &QQuickItem::windowChanged, this, &Viewer::onWindowChanged);
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

void Viewer::setScenePath(QUrl scenePath)
{
    if (this->scenePath == scenePath) {
        return;
    }
    isScenePathChanged = true;
    this->scenePath = scenePath;
    Q_EMIT scenePathChanged(scenePath);
}

void Viewer::cleanup()
{
    releaseResources();
}

void Viewer::onWindowChanged(QQuickWindow * w)
{
    if (!w) {
        qCDebug(viewerCategory) << "Window is lost";
        return;
    }

    INVARIANT(w->graphicsApi() == QSGRendererInterface::GraphicsApi::Vulkan, "Expected Vulkan backend");

    connect(w, &QQuickWindow::sceneGraphInvalidated, this, &Viewer::cleanup, Qt::ConnectionType::DirectConnection);
}

void Viewer::setScene()
{
    if (!isScenePathChanged) {
        return;
    }
    isScenePathChanged = false;
    ASSERT(renderer);
    renderer->unsetScene();
    if (scenePath.isEmpty()) {
        return;
    }
    if (!scenePath.isLocalFile()) {
        qCWarning(viewerCategory) << u"scenePath URL is not local file:"_s << scenePath;
        return;
    }
    auto newScene = engine->getEngine().getScenes().getScene(QFileInfo{scenePath.toLocalFile()}.filesystemCanonicalFilePath());
    if (!newScene) {
        return;
    }
    const auto & aabb = newScene->sceneData.aabb;
    characteristicSize = glm::distance(aabb.min, aabb.max);
    if (!setProperty("linearSpeed", utils::safeCast<qreal>(characteristicSize / 10.0f))) {
        qFatal("unreachable");
    }
    renderer->setScene(std::move(newScene));
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

QSGNode * Viewer::updatePaintNode(QSGNode * old, UpdatePaintNodeData * updatePaintNodeData)
{
    if (engine) {
        if (auto w = window()) {
            uint32_t framesInFlight = utils::autoCast(w->graphicsStateInfo().framesInFlight);
            if (renderer) {
                ASSERT(renderer->getFramesInFlight() == framesInFlight);
            } else {
                checkEngine(w, engine->getContext());
                renderer = std::make_unique<Renderer>(engine->getContext(), engine->getEngine(), framesInFlight);
            }
        }
        if (renderer) {
            setScene();
            auto node = static_cast<RenderNode *>(old);
            if (old) {
                ASSERT(dynamic_cast<RenderNode *>(old));
            } else {
                ASSERT(renderer);
                node = new RenderNode{window(), engine, renderer.get()};
            }
            node->set(useOffscreenTexture, wireFrame);
            {
                glm::vec3 position{cameraPosition.x(), cameraPosition.y(), cameraPosition.z()};
                auto cameraOrientation = QQuaternion::fromEulerAngles(eulerAngles);
                glm::quat orientation{cameraOrientation.scalar(), cameraOrientation.x(), cameraOrientation.y(), cameraOrientation.z()};
                float fov = utils::autoCast(qDegreesToRadians(fieldOfView));
                float zNear = std::sqrt(std::numeric_limits<float>::epsilon()) * characteristicSize;
                float zFar = characteristicSize;
                node->setCamera(position, orientation, fov, zNear, zFar);
            }
            node->setSize(size());
            return node;
        }
    }
    return QQuickItem::updatePaintNode(old, updatePaintNodeData);
}

}  // namespace viewer
