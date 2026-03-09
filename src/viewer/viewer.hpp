#pragma once

#include <builder/fwd.hpp>
#include <scene_data/fwd.hpp>

#include <QtCore/QFutureWatcher>
#include <QtCore/QHash>
#include <QtCore/QMetaObject>
#include <QtCore/QObject>
#include <QtCore/QPoint>
#include <QtCore/QPointF>
#include <QtCore/QSharedPointer>
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
struct Scene;
class Viewer;
class RenderNode;
class TaskQueue;

class SceneSettings : public QObject
{
    Q_OBJECT
    QML_ELEMENT

    Q_PROPERTY(EngineWrapper * engine MEMBER engineWrapper NOTIFY engineChanged REQUIRED)
    Q_PROPERTY(TaskQueue * taskQueue MEMBER taskQueue NOTIFY taskQueueChanged REQUIRED)

    Q_PROPERTY(QUrl url MEMBER url NOTIFY urlChanged RESET resetUrl)

    Q_PROPERTY(QVector3D sceneAabbMin READ getSceneAabbMin NOTIFY sceneChanged STORED false)
    Q_PROPERTY(QVector3D sceneAabbMax READ getSceneAabbMax NOTIFY sceneChanged STORED false)
    Q_PROPERTY(QString sceneStatus READ getSceneStatus NOTIFY sceneStatusChanged)

    Q_PROPERTY(QVariantList thrustDeviceSystems READ getThrustDeviceSystems CONSTANT)
    Q_PROPERTY(ThrustDeviceSystem thrustDeviceSystem MEMBER thrustDeviceSystem NOTIFY thrustDeviceSystemChanged)

    Q_PROPERTY(bool traceTree MEMBER traceTree NOTIFY treeSettingsChanged)

    Q_PROPERTY(float emptinessFactor MEMBER emptinessFactor NOTIFY treeSettingsChanged)
    Q_PROPERTY(float traversalCost MEMBER traversalCost NOTIFY treeSettingsChanged)
    Q_PROPERTY(float intersectionCost MEMBER intersectionCost NOTIFY treeSettingsChanged)
    Q_PROPERTY(int maxTreeDepth MEMBER maxTreeDepth NOTIFY treeSettingsChanged)

    Q_PROPERTY(int depth READ getDepth NOTIFY treeChanged STORED false)
    Q_PROPERTY(QString treeStatus READ getTreeStatus NOTIFY treeStatusChanged)

public:
    enum class ThrustDeviceSystem
    {
        Default,
        CPP,
        OMP,
        TBB,
        CUDA,
    };
    Q_ENUM(ThrustDeviceSystem);

    EngineWrapper * engineWrapper = nullptr;
    TaskQueue * taskQueue = nullptr;

    QUrl url;

    ThrustDeviceSystem thrustDeviceSystem = ThrustDeviceSystem::Default;

    bool traceTree = false;

    float emptinessFactor = 0.8f;
    float traversalCost = 2.0f;
    float intersectionCost = 1.0f;
    int maxTreeDepth = 1000;

    QList<QSharedPointer<QFutureWatcher<int>>> tasks;

    explicit SceneSettings(QObject * parent = nullptr);

    [[nodiscard]] QVector3D getSceneAabbMin() const;
    [[nodiscard]] QVector3D getSceneAabbMax() const;

    [[nodiscard]] const QString & getSceneStatus() const &
    {
        return sceneStatus;
    }

    [[nodiscard]] static QVariantList getThrustDeviceSystems();

    [[nodiscard]] int getDepth() const &;

    [[nodiscard]] const QString & getTreeStatus() const &
    {
        return treeStatus;
    }

Q_SIGNALS:
    void engineChanged();
    void taskQueueChanged();

    void urlChanged();
    void sceneChanged();
    void sceneStatusChanged();

    void thrustDeviceSystemChanged();

    void treeSettingsChanged();
    void treeChanged();
    void treeStatusChanged();

public Q_SLOTS:
    void resetUrl();

private Q_SLOTS:
    void updateScene();
    void onUrlChanged();

    void updateTree();
    void onTreeSettingsChanged();

private:
    friend Viewer;

    using SceneFutureWatcher = QFutureWatcher<scene_data::SceneDataPtr>;
    using TreeFutureWatcher = QFutureWatcher<builder::TreePtr>;

    QString sceneStatus;
    QSharedPointer<SceneFutureWatcher> sceneFutureWatcher;
    scene_data::SceneDataPtr sceneData;

    QString treeStatus;
    QSharedPointer<TreeFutureWatcher> treeFutureWatcher;
    builder::TreePtr tree;
    bool isTreeChanged = false;
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
    Q_DECLARE_FLAGS(
        RenderModeFlags,
        RenderModeFlag)
    Q_FLAG(RenderModeFlags)

    enum class TexturingMode
    {
        BarycentricColor,
        Wireframe,
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
    QML_ELEMENT

    Q_PROPERTY(QVector3D position MEMBER position WRITE setPosition NOTIFY viewChanged RESET resetPosition)
    Q_PROPERTY(QQuaternion orientation MEMBER orientation WRITE setOrientation NOTIFY viewChanged RESET resetOrientation)
    Q_PROPERTY(float fov MEMBER fov WRITE setFov NOTIFY viewChanged RESET resetFov)

public:
    QVector3D position;
    QQuaternion orientation;
    float fov = kDefaultFov;

    using QObject::QObject;

    Q_INVOKABLE void shift(const QVector3D & direction);
    Q_INVOKABLE void rotate(
        float pan,
        float tilt);
    Q_INVOKABLE void roll(float angle);
    Q_INVOKABLE void widen(float angle);

    [[nodiscard]] static Q_INVOKABLE float getDefaultFov();
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
    QML_ELEMENT

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

    Q_PROPERTY(EngineWrapper * engine MEMBER engineWrapper NOTIFY engineChanged REQUIRED)
    Q_PROPERTY(SceneSettings * scene MEMBER sceneSettings NOTIFY sceneSettingsChanged REQUIRED)
    Q_PROPERTY(RendererSettings * renderer MEMBER rendererSettings CONSTANT)
    Q_PROPERTY(CameraView * cameraView MEMBER cameraView CONSTANT)
    Q_PROPERTY(CameraController * cameraController MEMBER cameraController CONSTANT)

public:
    explicit Viewer(QQuickItem * parent = nullptr);

Q_SIGNALS:
    void engineChanged();
    void sceneSettingsChanged();

private Q_SLOTS:
    void handleKeyboardInput();

private:
    friend CameraView;

    EngineWrapper * engineWrapper = nullptr;
    SceneSettings * sceneSettings = nullptr;
    RendererSettings * const rendererSettings = new RendererSettings{this};
    CameraView * const cameraView = new CameraView{this};
    CameraController * const cameraController = new CameraController{this};

    QPointF startDragPos;
    QTimer * const mousePressAndHoldTimer = new QTimer{this};

    Qt::KeyboardModifiers keyboardModifiers = Qt::KeyboardModifier::NoModifier;
    QHash<Qt::Key, int> pressedKeys;
    QTimer * const handleKeyboardInputTimer = new QTimer{this};

    QMetaObject::Connection refreshRateConnection;
    QMetaObject::Connection sceneGraphInvalidatedConnection;

    QMetaObject::Connection sceneUrlChangedConnection;
    QMetaObject::Connection sceneChangedConnection;
    QMetaObject::Connection sceneStatusChangedConnection;
    QMetaObject::Connection sceneSettingsTreeChangedConnection;
    QMetaObject::Connection sceneSettingsTreeStatusChangedConnection;
    QMetaObject::Connection sceneSettingsBuildSettingsChangedConnection;

    void onKeyEvent(
        QKeyEvent * event,
        bool isPressed);

    void wheelEvent(QWheelEvent * event) override;
    void mouseUngrabEvent() override;
    void mousePressEvent(QMouseEvent * event) override;
    void mouseMoveEvent(QMouseEvent * event) override;
    void mouseReleaseEvent(QMouseEvent * event) override;
    void mouseDoubleClickEvent(QMouseEvent * event) override;

    void keyPressEvent(QKeyEvent * event) override;
    void keyReleaseEvent(QKeyEvent * event) override;

    [[nodiscard]] QSGNode * updatePaintNode(
        QSGNode * old,
        UpdatePaintNodeData * updatePaintNodeData) override;
    void releaseResources() override;
};

}  // namespace viewer
