#pragma once

#include <QtCore/QHash>
#include <QtCore/QMetaObject>
#include <QtCore/QObject>
#include <QtCore/QPoint>
#include <QtCore/QPointF>
#include <QtCore/QString>
#include <QtCore/QTimer>
#include <QtCore/QUrl>
#include <QtGui/QColor>
#include <QtGui/QQuaternion>
#include <QtGui/QVector3D>
#include <QtQuick/QQuickItem>
#include <QtQuick/QQuickWindow>

namespace viewer
{
class EngineWrapper;
class RenderNode;
class Viewer;

class SceneSettings : public QObject
{
    Q_OBJECT

    Q_PROPERTY(QUrl url MEMBER url WRITE setUrl NOTIFY urlChanged RESET unsetUrl)
    Q_PROPERTY(float worldScale MEMBER worldScale NOTIFY settingsChanged)
    Q_PROPERTY(QVector3D sceneAabbMin MEMBER sceneAabbMin NOTIFY settingsChanged)
    Q_PROPERTY(QVector3D sceneAabbMax MEMBER sceneAabbMax NOTIFY settingsChanged)
public:
    QUrl url;
    bool isUrlChanged = false;
    float worldScale = 1.0f;
    QVector3D sceneAabbMin;
    QVector3D sceneAabbMax;

    using QObject::QObject;

public Q_SLOTS:
    void setUrl(const QUrl & newUrl);
    void unsetUrl();

Q_SIGNALS:
    void urlChanged();
    void settingsChanged();

private:
    friend Viewer;

    void setScene(EngineWrapper * engine, RenderNode & renderNode);
    void updateScene(EngineWrapper * engine, RenderNode & renderNode);
};

class RendererSettings : public QObject
{
    Q_OBJECT
    QML_ELEMENT

    Q_PROPERTY(RenderModeFlags renderMode READ getRenderMode WRITE setRenderMode NOTIFY settingsChanged)
    Q_PROPERTY(TexturingMode texturingMode MEMBER texturingMode NOTIFY settingsChanged)

    Q_PROPERTY(QColor clearColor MEMBER clearColor NOTIFY settingsChanged)

public:
    enum class RenderModeFlag
    {
        Default = 0x0000,
        UseOffscreenTexture = 0x0001,  // TODO: QQuickRhiItem instead?
        DiscardInvisibleFragments = 0x0002,
    };
    Q_DECLARE_FLAGS(RenderModeFlags, RenderModeFlag)
    Q_FLAG(RenderModeFlags)

    enum class TexturingMode
    {
        BarycentricColor,
        WireFrame,
    };
    Q_ENUM(TexturingMode)

    RenderModeFlags renderMode;
    TexturingMode texturingMode = TexturingMode::BarycentricColor;

    QColor clearColor;

    int renderdocCaptureFrameCounter = 0;

    using QObject::QObject;

    [[nodiscard]] Q_INVOKABLE RenderModeFlags getRenderMode() const;

public Q_SLOTS:
    void setRenderMode(viewer::RendererSettings::RenderModeFlags renderMode);

    void renderdocCaptureFrame();

Q_SIGNALS:
    void settingsChanged();
};

class CameraView : public QObject
{
    Q_OBJECT

    Q_PROPERTY(QVector3D position MEMBER position WRITE setPosition NOTIFY viewChanged RESET resetPosition)
    Q_PROPERTY(QQuaternion orientation MEMBER orientation WRITE setOrientation NOTIFY viewChanged RESET resetOrientation)
    Q_PROPERTY(float fov MEMBER fov WRITE setFov NOTIFY viewChanged RESET resetFov)

public:
    QVector3D position;
    QQuaternion orientation;
    float fov = kDefaultFov;

    using QObject::QObject;

    Q_INVOKABLE void shift(const QVector3D & direction);
    Q_INVOKABLE void rotate(float pan, float tilt);
    Q_INVOKABLE void roll(float angle);
    Q_INVOKABLE void addFov(float angle);

    [[nodiscard]] Q_INVOKABLE float getFovRatio() const;

public Q_SLOTS:
    void setPosition(const QVector3D & position);
    void setOrientation(const QQuaternion & orientation);
    void setFov(float fov);

    void resetPosition();
    void resetOrientation();
    void resetFov();

    void alignOrientation();
    void reflectOrientation();

Q_SIGNALS:
    void viewChanged();

private:
    static constexpr float kDefaultFov = 90.0f;
};

class CameraController : public QObject
{
    Q_OBJECT

    Q_PROPERTY(float sensitivity MEMBER sensitivity NOTIFY controllerChanged RESET resetSensitivity)
    Q_PROPERTY(float speed MEMBER speed NOTIFY controllerChanged RESET resetSpeed)

public:
    float sensitivity = kDefaultSensitivity;
    float speed = kDefaultSpeed;

    using QObject::QObject;

public Q_SLOTS:
    void resetSensitivity();
    void resetSpeed();

Q_SIGNALS:
    void controllerChanged();

private:
    static constexpr float kDefaultSensitivity = 0.0012f;
    static constexpr float kDefaultSpeed = 1.0;
};

class Viewer : public QQuickItem
{
    Q_OBJECT
    QML_ELEMENT

    Q_PROPERTY(EngineWrapper * engine MEMBER engine NOTIFY engineChanged REQUIRED)
    Q_PROPERTY(SceneSettings * scene MEMBER scene CONSTANT)
    Q_PROPERTY(RendererSettings * renderer MEMBER renderer CONSTANT)
    Q_PROPERTY(CameraView * cameraView MEMBER cameraView CONSTANT)
    Q_PROPERTY(CameraController * cameraController MEMBER cameraController CONSTANT)

public:
    explicit Viewer(QQuickItem * parent = nullptr);
    ~Viewer() override;

Q_SIGNALS:
    void engineChanged();

private Q_SLOTS:
    void handleKeyboardInput();

private:
    friend CameraView;

    EngineWrapper * engine = nullptr;
    SceneSettings * const scene = new SceneSettings{this};
    RendererSettings * const renderer = new RendererSettings{this};
    CameraView * const cameraView = new CameraView{this};
    CameraController * const cameraController = new CameraController{this};

    QPointF startDragPos;
    QTimer * const mousePressAndHoldTimer = new QTimer{this};

    Qt::KeyboardModifiers keyboardModifiers = Qt::KeyboardModifier::NoModifier;
    QHash<Qt::Key, int> pressedKeys;
    QTimer * const handleKeyboardInputTimer = new QTimer{this};

    QMetaObject::Connection refreshRateConnection;
    QMetaObject::Connection sceneGraphInvalidatedConnection;

    void onKeyEvent(QKeyEvent * event, bool isPressed);

    void wheelEvent(QWheelEvent * event) override;
    void mouseUngrabEvent() override;
    void mousePressEvent(QMouseEvent * event) override;
    void mouseMoveEvent(QMouseEvent * event) override;
    void mouseReleaseEvent(QMouseEvent * event) override;
    void mouseDoubleClickEvent(QMouseEvent * event) override;

    void keyPressEvent(QKeyEvent * event) override;
    void keyReleaseEvent(QKeyEvent * event) override;

    [[nodiscard]] QSGNode * updatePaintNode(QSGNode * old, UpdatePaintNodeData * updatePaintNodeData) override;
    void releaseResources() override;
};

}  // namespace viewer
