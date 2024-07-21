#pragma once

#include <QtCore/QHash>
#include <QtCore/QMetaObject>
#include <QtCore/QObject>
#include <QtCore/QPoint>
#include <QtCore/QTimer>
#include <QtCore/QUrl>
#include <QtCore/QtTypes>
#include <QtGui/QKeyEvent>
#include <QtGui/QMouseEvent>
#include <QtGui/QVector2D>
#include <QtGui/QVector3D>
#include <QtGui/QWheelEvent>
#include <QtQuick/QQuickItem>
#include <QtQuick/QQuickWindow>
#include <QtQuick/QSGNode>

namespace viewer
{
class EngineWrapper;
struct Scene;
struct FrameSettings;
class Renderer;

class Viewer : public QQuickItem
{
    Q_OBJECT
    QML_NAMED_ELEMENT(SahKdTreeViewer)

    Q_PROPERTY(EngineWrapper * engine MEMBER engine NOTIFY engineChanged REQUIRED)

    Q_PROPERTY(QUrl sceneUrl MEMBER sceneUrl WRITE setSceneUrl NOTIFY sceneUrlChanged RESET unsetSceneUrl)
    Q_PROPERTY(float worldScale MEMBER worldScale NOTIFY sceneChanged)
    Q_PROPERTY(QVector3D sceneAabbMin MEMBER sceneAabbMin NOTIFY sceneChanged)
    Q_PROPERTY(QVector3D sceneAabbMax MEMBER sceneAabbMax NOTIFY sceneChanged)

    Q_PROPERTY(QVector3D cameraPosition MEMBER cameraPosition WRITE setCameraPosition NOTIFY cameraViewChanged)
    Q_PROPERTY(QQuaternion cameraOrientation MEMBER cameraOrientation WRITE setCameraOrientation NOTIFY cameraViewChanged)
    Q_PROPERTY(float cameraFieldOfView MEMBER cameraFieldOfView WRITE setCameraFieldOfView NOTIFY cameraViewChanged)
    Q_PROPERTY(QString cameraDescription READ getCameraDescription NOTIFY cameraViewChanged STORED false)

    Q_PROPERTY(float sensitivity MEMBER sensitivity NOTIFY cameraControllerChanged)
    Q_PROPERTY(float speed MEMBER speed NOTIFY cameraControllerChanged)
    Q_PROPERTY(QString cameraControllerDescription READ getCameraControllerDescription NOTIFY cameraControllerChanged STORED false)

    Q_PROPERTY(bool useOffscreenTexture MEMBER useOffscreenTexture NOTIFY renderModeChanged)
    Q_PROPERTY(bool wireFrame MEMBER wireFrame NOTIFY renderModeChanged)
    Q_PROPERTY(QString modeDescription READ getModeDescription NOTIFY renderModeChanged STORED false)
    Q_PROPERTY(QString modeDescriptionVerbose READ getModeDescriptionVerbose NOTIFY renderModeChanged STORED false)

public:
    explicit Viewer(QQuickItem * parent = nullptr);
    ~Viewer() override;

    [[nodiscard]] QString getCameraDescription() const;
    [[nodiscard]] QString getCameraControllerDescription() const;
    [[nodiscard]] QString getModeDescription() const;
    [[nodiscard]] QString getModeDescriptionVerbose() const;

public Q_SLOTS:
    void setSceneUrl(QUrl sceneUrl);
    void unsetSceneUrl();

    void setCameraPosition(QVector3D cameraPosition);
    void setCameraOrientation(QQuaternion cameraOrientation);
    void setCameraFieldOfView(float cameraFieldOfView);
    void resetCameraView();
    void alignCameraOrientation();
    void reflectCameraOrientation();

Q_SIGNALS:
    void engineChanged();
    void sceneUrlChanged();
    void sceneChanged();
    void cameraViewChanged();
    void cameraControllerChanged();
    void renderModeChanged();

private:
    static constexpr float kDefaultCameraFieldOfView = 90.0f;

    class RenderNode;

    EngineWrapper * engine = nullptr;

    QUrl sceneUrl;
    bool isSceneUrlChanged = false;
    float worldScale = 1.0f;
    QVector3D sceneAabbMin;
    QVector3D sceneAabbMax;

    QVector3D cameraPosition;
    QQuaternion cameraOrientation;
    float cameraFieldOfView = kDefaultCameraFieldOfView;

    float sensitivity = 0.0012f;
    float speed = 1.0f;

    bool useOffscreenTexture = true;
    bool wireFrame = false;

    QTimer * const mousePressAndHoldTimer = new QTimer{this};
    QPoint startDragPos;
    Qt::KeyboardModifiers keyboardModifiers = Qt::KeyboardModifier::NoModifier;
    QHash<Qt::Key, int> pressedKeys;
    QTimer * const handleInputTimer = new QTimer{this};
    QMetaObject::Connection refreshRateConnection;
    QMetaObject::Connection sceneGraphInvalidatedConnection;

    void setScene(RenderNode & renderNode);

    void onKeyEvent(QKeyEvent * event, bool isPressed);
    void handleInput();

    void releaseResources() override;

    void wheelEvent(QWheelEvent * event) override;

    void mouseUngrabEvent() override;

    void mousePressEvent(QMouseEvent * event) override;
    void mouseMoveEvent(QMouseEvent * event) override;
    void mouseReleaseEvent(QMouseEvent * event) override;

    void mouseDoubleClickEvent(QMouseEvent * event) override;

    void keyPressEvent(QKeyEvent * event) override;
    void keyReleaseEvent(QKeyEvent * event) override;

    [[nodiscard]] QSGNode * updatePaintNode(QSGNode * old, UpdatePaintNodeData * updatePaintNodeData) override;
};

}  // namespace viewer
