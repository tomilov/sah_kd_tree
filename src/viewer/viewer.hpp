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
    void setUrl(const QUrl & url);
    void unsetUrl();

Q_SIGNALS:
    void urlChanged();
    void settingsChanged();

private:
    friend Viewer;

    void setScene(EngineWrapper * engine, RenderNode & renderNode);
};

class RendererSettings : public QObject
{
    Q_OBJECT

    Q_PROPERTY(bool useOffscreenTexture MEMBER useOffscreenTexture NOTIFY settingsChanged)  // TODO: QQuickRhiItem instead?
    Q_PROPERTY(bool discardInvisible MEMBER discardInvisible NOTIFY settingsChanged)
    Q_PROPERTY(bool wireFrame MEMBER wireFrame NOTIFY settingsChanged)
    Q_PROPERTY(QString modeDescription READ getModeDescription NOTIFY settingsChanged STORED false)
    Q_PROPERTY(QString modeDescriptionVerbose READ getModeDescriptionVerbose NOTIFY settingsChanged STORED false)
    Q_PROPERTY(QColor clearColor MEMBER clearColor NOTIFY settingsChanged)

public:
    bool useOffscreenTexture = true;
    bool discardInvisible = true;
    bool wireFrame = false;
    QColor clearColor;

    using QObject::QObject;

Q_SIGNALS:
    void settingsChanged();

private:
    [[nodiscard]] QString getModeDescription() const;
    [[nodiscard]] QString getModeDescriptionVerbose() const;
};

class CameraView : public QObject
{
    Q_OBJECT

    Q_PROPERTY(QVector3D position MEMBER position WRITE setPosition NOTIFY viewChanged RESET resetPosition)
    Q_PROPERTY(QQuaternion orientation MEMBER orientation WRITE setOrientation NOTIFY viewChanged RESET resetOrientation)
    Q_PROPERTY(float fov MEMBER fov WRITE setFov NOTIFY viewChanged RESET resetFov)
    Q_PROPERTY(QString description READ getDescription NOTIFY viewChanged STORED false)

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

    [[nodiscard]] QString getDescription() const;
};

class CameraController : public QObject
{
    Q_OBJECT

    Q_PROPERTY(float sensitivity MEMBER sensitivity NOTIFY controllerChanged RESET resetSensitivity)
    Q_PROPERTY(float speed MEMBER speed NOTIFY controllerChanged RESET resetSpeed)
    Q_PROPERTY(QString description READ getDescription NOTIFY controllerChanged STORED false)

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

    [[nodiscard]] QString getDescription() const;
};

class Viewer : public QQuickItem
{
    Q_OBJECT
    QML_NAMED_ELEMENT(SahKdTreeViewer)

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
