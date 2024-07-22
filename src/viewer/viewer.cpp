#include <engine/context.hpp>
#include <engine/device.hpp>
#include <engine/graphics_pipeline.hpp>
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
#include <optional>
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

// https://bugreports.qt.io/browse/QTBUG-121137
class CleanupJob : public QRunnable
{
public:
    static void scheduleRenderJob(QQuickWindow * window, std::unique_ptr<Renderer> && renderer)
    {
        window->scheduleRenderJob(new CleanupJob{std::move(renderer)}, QQuickWindow::RenderStage::NoStage);
    }

private:
    std::unique_ptr<Renderer> renderer;

    explicit CleanupJob(std::unique_ptr<Renderer> && renderer)
        : renderer{std::move(renderer)}
    {}

    void run() override
    {
        renderer.reset();
    }
};

}  // namespace

void Camera::shift(QVector3D direction)
{
    auto newPosition = position + orientation.rotatedVector(direction);

    const Viewer * viewer = qobject_cast<const Viewer *>(parent());
    Q_CHECK_PTR(viewer);
    const QVector3D & sceneAabbMin = viewer->sceneAabbMin;
    const QVector3D & sceneAabbMax = viewer->sceneAabbMax;
    const float & worldScale = viewer->worldScale;
    if (!qFuzzyCompare(sceneAabbMin, sceneAabbMax)) {
        const auto direction = position - 0.5f * (sceneAabbMin + sceneAabbMax);
        const float c = direction.length() - 0.5f * (sceneAabbMax - sceneAabbMin).length() * worldScale;
        if (c > 0.0f) {
            newPosition -= c * direction.normalized();
        }
    }

    setPosition(newPosition);
}

void Camera::rotate(float pan, float tilt)
{
    if ((false)) {
        auto tiltRotation = QQuaternion::fromAxisAndAngle(1.0f, 0.0f, 0.0f, tilt);
        auto panRotation = QQuaternion::fromAxisAndAngle(0.0f, 1.0f, 0.0f, pan);
        setOrientation(orientation * panRotation * tiltRotation);
    } else {
        float pitch, yaw, roll;
        orientation.getEulerAngles(&pitch, &yaw, &roll);

        float rollRadians = qDegreesToRadians(roll);
        pitch += tilt * qCos(rollRadians) - pan * qSin(rollRadians);
        yaw += pan * qCos(rollRadians) + tilt * qSin(rollRadians);

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

void Camera::roll(float angle)
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

void Camera::addFov(float angle)
{
    auto newFieldOfView = qBound<float>(5.0f, fieldOfView + angle, 175.0f);
    setFieldOfView(newFieldOfView);
}

float Camera::getFovRatio() const
{
    return fieldOfView / kDefaultFieldOfView;
}

void Camera::setPosition(QVector3D position)
{
    if (qFuzzyCompare(this->position, position)) {
        return;
    }
    this->position = position;
    Q_EMIT viewChanged();
}

void Camera::setOrientation(QQuaternion orientation)
{
    if (qFuzzyCompare(this->orientation, orientation)) {
        return;
    }
    this->orientation = orientation;
    Q_EMIT viewChanged();
}

void Camera::setFieldOfView(float fieldOfView)
{
    if (qFuzzyCompare(this->fieldOfView, fieldOfView)) {
        return;
    }
    this->fieldOfView = fieldOfView;
    Q_EMIT viewChanged();
}

void Camera::resetView()
{
    setPosition({});
    setOrientation({});
    setFieldOfView(kDefaultFieldOfView);
}

void Camera::alignOrientation()
{
    float pitch, yaw, roll;
    orientation.getEulerAngles(&pitch, &yaw, &roll);
    constexpr auto round = [](float angle) -> float
    {
        return qRound(angle / 90.0f) * 90.0f;
    };
    setOrientation(QQuaternion::fromEulerAngles(round(pitch), round(yaw), round(roll)));
}

void Camera::reflectOrientation()
{
    float pitch, yaw, roll;
    orientation.getEulerAngles(&pitch, &yaw, &roll);
    setOrientation(QQuaternion::fromEulerAngles(-pitch, yaw + 180.0f, -roll));
}

QString Camera::getDescription() const
{
    float pitch, yaw, roll;
    orientation.getEulerAngles(&pitch, &yaw, &roll);
    return u"xyz(%1, %2, %3) \x3C6\x3B8\x3C8(%4, %5, %6) fov(%7)"_s.arg(position.x(), 5, 'f', 3).arg(position.y(), 5, 'f', 3).arg(position.z(), 5, 'f', 3).arg(pitch, 5, 'f', 1).arg(yaw, 5, 'f', 1).arg(roll, 5, 'f', 1).arg(fieldOfView, 5, 'f', 1);
}

class Viewer::RenderNode final : public QSGRenderNode
{
public:
    explicit RenderNode(QQuickWindow * window, const EngineWrapper * engineWrapper)
        : window{window}
        , engineWrapper{engineWrapper}
    {
        Q_ASSERT(window);
        ASSERT(engineWrapper);
        checkContext(window, engineWrapper->getContext());
    }

    void unsetScene()
    {
        scene.reset();
        isDirty = true;
    }

    void setScene(std::shared_ptr<const Scene> newScene)
    {
        unsetScene();
        scene = std::move(newScene);
    }

    void updateSize(QSizeF newSize)
    {
        updateState(size, newSize);
    }

    void updateMode(bool useOffscreenTexture, bool discardInvisible, bool wireFrame)
    {
        updateState(frameSettings.useOffscreenTexture, useOffscreenTexture);
        updateState(frameSettings.discardInvisible, discardInvisible);
        updateState(frameSettings.wireFrame, wireFrame);
    }

    void updateCamera(const glm::vec3 & position, const glm::quat & orientation, float fov, float zNear, float zFar)
    {
        updateState(frameSettings.position, position);
        updateState(frameSettings.orientation, orientation);
        updateState(frameSettings.fov, fov);
        updateState(frameSettings.zNear, zNear);
        updateState(frameSettings.zFar, zFar);
    }

    void setClearColor(const glm::vec4 & clearColor)
    {
        updateState(frameSettings.clearColor, clearColor);
    }

    void markDirty()
    {
        if (!isDirty) {
            return;
        }
        isDirty = false;
        return QSGNode::markDirty(QSGNode::DirtyStateBit::DirtyForceUpdate);
    }

private:
    QQuickWindow * const window;
    const EngineWrapper * const engineWrapper;

    std::optional<Renderer> renderer;
    std::shared_ptr<const Scene> scene;

    bool isDirty = false;

    QSizeF size;
    FrameSettings frameSettings;

    QVector<quint32> renderPassFormat;

    template<typename T>
    void updateState(T & lhs, const T & rhs)
    {
        if (lhs == rhs) {
            return;
        }
        lhs = rhs;
        isDirty = true;
    }

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

    void advance()
    {
        const QQuickWindow::GraphicsStateInfo & graphicsStateInfo = window->graphicsStateInfo();
        uint32_t framesInFlight = utils::autoCast(graphicsStateInfo.framesInFlight);
        if (renderer) {
            ASSERT(renderer.value().getFramesInFlight() == framesInFlight);
        } else {
            renderer.emplace(engineWrapper->getContext(), engineWrapper->getEngine(), framesInFlight);
        }
        renderer.value().setFrameSettings(frameSettings);
        if (!scene) {
            renderer.value().unsetScene();
        } else if (scene != renderer.value().getScene()) {
            renderer.value().setScene(scene);
        }
        renderer.value().advance(utils::autoCast(graphicsStateInfo.currentFrameSlot));
    }

    void prepare() override
    {
        frameSettings.alpha = inheritedOpacity();

        frameSettings.width = utils::safeCast<float>(std::ceil(size.width()));
        frameSettings.height = utils::safeCast<float>(std::ceil(size.height()));

        const QSizeF renderTargetSize = renderTarget()->pixelSize();
        frameSettings.viewport = vk::Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = utils::autoCast(renderTargetSize.width()),
            .height = utils::autoCast(renderTargetSize.height()),
            .minDepth = engine::kMinDepth,
            .maxDepth = 1.0f,
        };

        const QMatrix4x4 mvp = *projectionMatrix() * *matrix();
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
        glm::mat4 & windowViewPorjection = frameSettings.windowViewPorjection;
        windowViewPorjection = glm::make_mat4x4(mvp.constData());
        windowViewPorjection = glm::scale(windowViewPorjection, glm::vec3{frameSettings.width * 0.5f, frameSettings.height * 0.5f, 1.0f});
        windowViewPorjection = glm::translate(windowViewPorjection, glm::vec3{1.0f, 1.0f, 0.0f});

        if ((false)) {                                // does not reset automatically w/o update()
            if (frameSettings.useOffscreenTexture) {  // optimization for axis aligned transform case
                const QMatrix4x4 & m = *matrix();
                if (m.flags() < QMatrix4x4::Flag::Rotation) {
                    if ((qFuzzyIsNull(m(0, 1)) && qFuzzyIsNull(m(1, 0))) || (qFuzzyIsNull(m(0, 0)) && qFuzzyIsNull(m(1, 1)))) {
                        frameSettings.useOffscreenTexture = false;
                    }
                }
            }
        }

        advance();
    }

    void render(const RenderState * renderState) override
    {
        if (!renderer) {  // recover after releaseResources()
            advance();
        }

        if ((false)) {
            QStringList clipRegions;
            if (auto clipRegion = renderState->clipRegion()) {
                for (const QRect & rect : *clipRegion) {
                    clipRegions << toString(rect);
                }
            }
            qCInfo(viewerCategory) << u"scissorRect(%1) scissorEnabled(%2) stencilValue(%3) stencilEnabled(%4) clipRegion(%5)"_s.arg(toString(renderState->scissorRect()))
                                          .arg(renderState->scissorEnabled())
                                          .arg(renderState->stencilValue())
                                          .arg(renderState->stencilEnabled())
                                          .arg(clipRegions.join(u"|"_s));
        }

        auto commandBufferNativeHandles = commandBuffer()->nativeHandles();
        Q_CHECK_PTR(commandBufferNativeHandles);
        vk::CommandBuffer commandBuffer = static_cast<const QRhiVulkanCommandBufferNativeHandles *>(commandBufferNativeHandles)->commandBuffer;

        const auto & device = engineWrapper->getContext().getDevice();
        device.setDebugUtilsObjectName(commandBuffer, "Qt command buffer");

        auto renderPassDescriptor = renderTarget()->renderPassDescriptor();
        auto newRenderPassFormat = renderPassDescriptor->serializedFormat();
        const bool isRenderPassFormatChanged = renderPassFormat != newRenderPassFormat;
        if (isRenderPassFormatChanged) {
            renderPassFormat = std::move(newRenderPassFormat);
            qCDebug(viewerCategory) << u"Render pass format changed"_s;
        }
        auto renderPassNativeHandles = renderPassDescriptor->nativeHandles();
        Q_CHECK_PTR(renderPassNativeHandles);
        vk::RenderPass renderPass = static_cast<const QRhiVulkanRenderPassNativeHandles *>(renderPassNativeHandles)->renderPass;

        const QQuickWindow::GraphicsStateInfo & graphicsStateInfo = window->graphicsStateInfo();
        ASSERT(renderer.value().getFramesInFlight() == utils::safeCast<uint32_t>(graphicsStateInfo.framesInFlight));
        renderer.value().render(commandBuffer, renderPass, isRenderPassFormatChanged, utils::autoCast(graphicsStateInfo.currentFrameSlot));
    }

    void releaseResources() override
    {
        renderer.reset();
    }

    [[nodiscard]] RenderingFlags flags() const override
    {
        auto renderingFlags = QSGRenderNode::flags();
        if (frameSettings.useOffscreenTexture) {
            renderingFlags |= RenderingFlag::DepthAwareRendering;
            renderingFlags |= RenderingFlag::BoundedRectRendering;
            if (!frameSettings.discardInvisible && (frameSettings.alpha == 1.0f)) {
                renderingFlags |= RenderingFlag::OpaqueRendering;
            }
        }
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

Viewer::Viewer(QQuickItem * parent)
    : QQuickItem{parent}
{
    qRegisterMetaType<Camera *>("Camera*");

    setFlag(QQuickItem::Flag::ItemHasContents);
    setFocusPolicy(Qt::FocusPolicy::WheelFocus);
    setAcceptedMouseButtons(Qt::MouseButton::LeftButton);

    Q_CHECK_PTR(mousePressAndHoldTimer);
    mousePressAndHoldTimer->setInterval(QGuiApplication::styleHints()->mousePressAndHoldInterval());
    mousePressAndHoldTimer->setSingleShot(true);
    connect(mousePressAndHoldTimer, &QTimer::timeout, this, [this] { setCursor(Qt::CursorShape::BlankCursor); });

    Q_CHECK_PTR(handleInputTimer);
    const auto onPrimaryScreenChanged = [this](QScreen * primaryScreen)
    {
        disconnect(refreshRateConnection);
        if (!primaryScreen) {
            qCDebug(viewerCategory) << "primaryScreen is lost";
            handleInputTimer->stop();
            return;
        }
        const auto onRefrashRateChanged = [this](qreal refreshRate)
        {
            constexpr qreal kMsPerS = 1000.0;
            Q_ASSERT(!qFuzzyIsNull(refreshRate));
            handleInputTimer->start(qFloor(kMsPerS / refreshRate));
        };
        onRefrashRateChanged(primaryScreen->refreshRate());
        refreshRateConnection = connect(primaryScreen, &QScreen::refreshRateChanged, this, onRefrashRateChanged);
    };
    onPrimaryScreenChanged(qApp->primaryScreen());
    connect(qApp, &QGuiApplication::primaryScreenChanged, this, onPrimaryScreenChanged);
    connect(handleInputTimer, &QTimer::timeout, this, &Viewer::handleInput);

    connect(this, &Viewer::sceneChanged, this, &QQuickItem::update);
    connect(camera, &Camera::viewChanged, this, &QQuickItem::update);
    connect(this, &Viewer::renderModeChanged, this, &QQuickItem::update);
    connect(this, &Viewer::clearColorChanged, this, &QQuickItem::update);

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

void Viewer::setSceneUrl(QUrl sceneUrl)
{
    if (this->sceneUrl == sceneUrl) {
        return;
    }
    isSceneUrlChanged = true;
    this->sceneUrl = sceneUrl;
    Q_EMIT sceneUrlChanged();
}

void Viewer::unsetSceneUrl()
{
    if (sceneUrl.isEmpty()) {
        return;
    }
    isSceneUrlChanged = true;
    sceneUrl.clear();
    Q_EMIT sceneUrlChanged();
}

void Viewer::handleInput()
{
    if (pressedKeys.isEmpty()) {
        return;
    }
    if (pressedKeys.contains(Qt::Key_Space)) {
        camera->resetView();
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
    if (pressedKeys.contains(Qt::Key_Z)) {
        if (0 == pressedKeys[Qt::Key_Z]++) {
            camera->setPosition({});
        }
    } else {
        float step = speedModifier * speed / utils::safeCast<float>(qApp->primaryScreen()->refreshRate());
        camera->shift(direction.normalized() * step);
    }
    if (pressedKeys.contains(Qt::Key_R)) {
        if (0 == pressedKeys[Qt::Key_R]++) {
            camera->reflectOrientation();
        }
    } else if (pressedKeys.contains(Qt::Key_X)) {
        if (0 == pressedKeys[Qt::Key_X]++) {
            camera->alignOrientation();
        }
    } else {
        float angularSpeed = speedModifier;
        camera->rotate(pan * angularSpeed, tilt * angularSpeed);
    }
}

QString Viewer::getCameraControllerDescription() const
{
    return u"sens(%1) speed(%2)"_s.arg(sensitivity, 5, 'f', 4).arg(speed, 5, 'f', 2);
}

QString Viewer::getModeDescription() const
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

QString Viewer::getModeDescriptionVerbose() const
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

void Viewer::setScene(RenderNode & renderNode)
{
    if (!isSceneUrlChanged) {
        return;
    }
    isSceneUrlChanged = false;
    renderNode.unsetScene();
    {
        sceneAabbMin = {};
        sceneAabbMax = {};
    }
    Q_EMIT sceneChanged();
    if (sceneUrl.isEmpty()) {
        return;
    }
    if (!sceneUrl.isLocalFile()) {
        qCWarning(viewerCategory) << u"sceneUrl URL is not local file:"_s << sceneUrl;
        return;
    }
    const auto scenePath = QFileInfo{sceneUrl.toLocalFile()}.filesystemCanonicalFilePath();
    QGuiApplication::setOverrideCursor(Qt::CursorShape::WaitCursor);
    auto newScene = engine->getEngine().getScenes().getScene(scenePath);
    QGuiApplication::restoreOverrideCursor();
    if (!newScene) {
        return;
    }
    {
        const auto & [aabbMin, aabbMax] = newScene->sceneData.aabb;
        sceneAabbMin = {aabbMin.x, aabbMin.y, aabbMin.z};
        sceneAabbMax = {aabbMax.x, aabbMax.y, aabbMax.z};
        Q_EMIT sceneChanged();
    }
    renderNode.setScene(std::move(newScene));
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

void Viewer::releaseResources()
{
    // all resources owned by RenderNode
}

void Viewer::wheelEvent(QWheelEvent * event)
{
    constexpr qreal kUnitsPerDegree = 8.0f;
    float angle = utils::safeCast<float>(event->angleDelta().y()) / kUnitsPerDegree;
    if (keyboardModifiers == Qt::KeyboardModifier::ShiftModifier) {
        camera->roll(angle);
    } else {
        constexpr float kUnitsPerStep = 15.0f;
        constexpr float kDegreesPerStep = 5.0f;
        camera->addFov(angle / (kUnitsPerStep / kDegreesPerStep));
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
        setKeepMouseGrab(true);
        mousePressAndHoldTimer->start();
        startDragPos = QCursor::pos();
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
        auto dragPosDelta = QCursor::pos() - startDragPos;
        if (!dragPosDelta.isNull()) {
            mousePressAndHoldTimer->stop();
            setCursor(Qt::CursorShape::BlankCursor);
            if (!size().isEmpty()) {
                float angularSpeed = sensitivity * qApp->primaryScreen()->physicalDotsPerInch() * camera->getFovRatio();
                float pan = utils::autoCast(dragPosDelta.x());
                float tilt = utils::autoCast(dragPosDelta.y());
                camera->rotate(pan * angularSpeed, tilt * angularSpeed);
            }
        }
        QCursor::setPos(startDragPos);
        startDragPos = QCursor::pos();
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
        setKeepMouseGrab(false);
        if (mousePressAndHoldTimer->isActive()) {
            mousePressAndHoldTimer->stop();
            setKeepMouseGrab(true);
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
    if (window() && engine) {
        auto node = static_cast<RenderNode *>(old);
        if (old) {
            ASSERT(dynamic_cast<RenderNode *>(old));
        } else {
            node = new RenderNode{window(), engine};
        }
        setScene(*node);
        node->updateSize(size());
        node->updateMode(useOffscreenTexture, discardInvisible, wireFrame);
        {
            const auto getCameraProperty = [this](const char * propertyName) -> QVariant
            {
                auto property = camera->property(propertyName);
                Q_ASSERT(property.isValid());
                return property;
            };
            auto cameraPosition = getCameraProperty("position").value<QVector3D>();
            auto cameraOrientation = getCameraProperty("orientation").value<QQuaternion>();
            auto cameraFieldOfView = getCameraProperty("fieldOfView").value<float>();

            glm::vec3 position{cameraPosition.x(), cameraPosition.y(), cameraPosition.z()};
            glm::quat orientation{cameraOrientation.scalar(), cameraOrientation.x(), cameraOrientation.y(), cameraOrientation.z()};
            float fov = utils::autoCast(qDegreesToRadians(cameraFieldOfView));
            float zFar = (sceneAabbMax - sceneAabbMin).length() * worldScale;
            float zNear = 2.0f * std::sqrt(std::numeric_limits<float>::epsilon()) * zFar;
            node->updateCamera(position, orientation, fov, zNear, zFar);
        }
        {
            float r, g, b, a;
            clearColor.getRgbF(&r, &g, &b, &a);
            node->setClearColor({r, g, b, a});
        }
        node->markDirty();
        return node;
    }
    return QQuickItem::updatePaintNode(old, updatePaintNodeData);
}

}  // namespace viewer
