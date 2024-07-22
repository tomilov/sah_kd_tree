#pragma once

#include <QtCore/QDataStream>
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
#include <QtQml/QQmlEngine>
#include <QtQuick/QQuickItem>
#include <QtQuick/QQuickWindow>
#include <QtQuick/QSGNode>

namespace viewer
{
class EngineWrapper;
struct Scene;
struct FrameSettings;
class Renderer;

class Viewer;
class Camera : public QObject
{
    Q_OBJECT

    Q_PROPERTY(QVector3D position MEMBER position WRITE setPosition NOTIFY viewChanged)
    Q_PROPERTY(QQuaternion orientation MEMBER orientation WRITE setOrientation NOTIFY viewChanged)
    Q_PROPERTY(float fieldOfView MEMBER fieldOfView WRITE setFieldOfView NOTIFY viewChanged)
    Q_PROPERTY(QString description READ getDescription NOTIFY viewChanged STORED false)
    QML_ELEMENT

public:
    using QObject::QObject;

    Q_INVOKABLE void shift(QVector3D direction);
    Q_INVOKABLE void rotate(float pan, float tilt);
    Q_INVOKABLE void roll(float angle);
    Q_INVOKABLE void addFov(float angle);

    [[nodiscard]] Q_INVOKABLE float getFovRatio() const;

    friend QDataStream & operator<<(QDataStream & dataStream, const Camera & camera)
    {
        return dataStream << camera.position << camera.orientation << camera.fieldOfView;
    }

    friend QDataStream & operator>>(QDataStream & dataStream, Camera & camera)
    {
        return dataStream >> camera.position >> camera.orientation >> camera.fieldOfView;
    }

public Q_SLOTS:
    void setPosition(QVector3D position);
    void setOrientation(QQuaternion orientation);
    void setFieldOfView(float fieldOfView);

    void resetView();
    void alignOrientation();
    void reflectOrientation();

Q_SIGNALS:
    void viewChanged();

private:
    static constexpr float kDefaultFieldOfView = 90.0f;

    QVector3D position;
    QQuaternion orientation;
    float fieldOfView = kDefaultFieldOfView;

    [[nodiscard]] QString getDescription() const;
};

class Viewer : public QQuickItem
{
    Q_OBJECT
    QML_NAMED_ELEMENT(SahKdTreeViewer)

    Q_PROPERTY(EngineWrapper * engine MEMBER engine NOTIFY engineChanged REQUIRED)

    Q_PROPERTY(QUrl sceneUrl MEMBER sceneUrl WRITE setSceneUrl NOTIFY sceneUrlChanged RESET unsetSceneUrl)
    Q_PROPERTY(float worldScale MEMBER worldScale NOTIFY sceneChanged)
    Q_PROPERTY(QVector3D sceneAabbMin MEMBER sceneAabbMin NOTIFY sceneChanged)
    Q_PROPERTY(QVector3D sceneAabbMax MEMBER sceneAabbMax NOTIFY sceneChanged)

    Q_PROPERTY(Camera * camera MEMBER camera CONSTANT)

    Q_PROPERTY(float sensitivity MEMBER sensitivity NOTIFY cameraControllerChanged)
    Q_PROPERTY(float speed MEMBER speed NOTIFY cameraControllerChanged)
    Q_PROPERTY(QString cameraControllerDescription READ getCameraControllerDescription NOTIFY cameraControllerChanged STORED false)

    Q_PROPERTY(bool useOffscreenTexture MEMBER useOffscreenTexture NOTIFY renderModeChanged)
    Q_PROPERTY(bool discardInvisible MEMBER discardInvisible NOTIFY renderModeChanged)
    Q_PROPERTY(bool wireFrame MEMBER wireFrame NOTIFY renderModeChanged)
    Q_PROPERTY(QString modeDescription READ getModeDescription NOTIFY renderModeChanged STORED false)
    Q_PROPERTY(QString modeDescriptionVerbose READ getModeDescriptionVerbose NOTIFY renderModeChanged STORED false)

    Q_PROPERTY(QColor clearColor MEMBER clearColor NOTIFY clearColorChanged)

public:
    explicit Viewer(QQuickItem * parent = nullptr);
    ~Viewer() override;

public Q_SLOTS:
    void setSceneUrl(QUrl sceneUrl);
    void unsetSceneUrl();

Q_SIGNALS:
    void engineChanged();
    void sceneUrlChanged();
    void sceneChanged();
    void cameraControllerChanged();
    void renderModeChanged();
    void clearColorChanged();

private Q_SLOTS:
    void handleInput();

private:
    friend Camera;
    class RenderNode;

    EngineWrapper * engine = nullptr;

    QUrl sceneUrl;
    bool isSceneUrlChanged = false;
    float worldScale = 1.0f;
    QVector3D sceneAabbMin;
    QVector3D sceneAabbMax;

    Camera * const camera = new Camera{this};

    float sensitivity = 0.0012f;
    float speed = 1.0f;

    bool useOffscreenTexture = true;
    bool discardInvisible = true;
    bool wireFrame = false;

    QColor clearColor;

    QTimer * const mousePressAndHoldTimer = new QTimer{this};
    QPoint startDragPos;
    Qt::KeyboardModifiers keyboardModifiers = Qt::KeyboardModifier::NoModifier;
    QHash<Qt::Key, int> pressedKeys;
    QTimer * const handleInputTimer = new QTimer{this};
    QMetaObject::Connection refreshRateConnection;
    QMetaObject::Connection sceneGraphInvalidatedConnection;

    [[nodiscard]] QString getCameraControllerDescription() const;
    [[nodiscard]] QString getModeDescription() const;
    [[nodiscard]] QString getModeDescriptionVerbose() const;

    void setScene(RenderNode & renderNode);
    void onKeyEvent(QKeyEvent * event, bool isPressed);

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

Q_DECLARE_METATYPE(viewer::Camera)
