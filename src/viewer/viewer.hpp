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
#include <QtQmlIntegration/QtQmlIntegration>
#include <QtQuick/QQuickItem>
#include <QtQuick/QQuickWindow>
#include <QtQuick/QSGNode>

#include <memory>

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

    Q_PROPERTY(QVector3D eulerAngles MEMBER eulerAngles WRITE setEulerAngles NOTIFY eulerAnglesChanged)
    Q_PROPERTY(QVector3D cameraPosition MEMBER cameraPosition WRITE setCameraPosition NOTIFY cameraPositionChanged)
    Q_PROPERTY(qreal fieldOfView MEMBER fieldOfView WRITE setFieldOfView NOTIFY fieldOfViewChanged)

    Q_PROPERTY(qreal dt MEMBER dt WRITE setDt NOTIFY dtChanged)
    Q_PROPERTY(qreal mouseLookSpeed MEMBER mouseLookSpeed NOTIFY mouseLookSpeedChanged)
    Q_PROPERTY(qreal keyboardLookSpeed MEMBER keyboardLookSpeed NOTIFY keyboardLookSpeedChanged)
    Q_PROPERTY(qreal linearSpeed MEMBER linearSpeed NOTIFY linearSpeedChanged)

    Q_PROPERTY(QUrl scenePath MEMBER scenePath WRITE setScenePath NOTIFY scenePathChanged)

    Q_PROPERTY(bool useOffscreenTexture MEMBER useOffscreenTexture NOTIFY useOffscreenTextureChanged)
    Q_PROPERTY(bool wireFrame MEMBER wireFrame NOTIFY wireFrameChanged)

public:
    explicit Viewer(QQuickItem * parent = nullptr);
    ~Viewer() override;

    Q_INVOKABLE void rotate(QVector3D tiltPanRoll);
    Q_INVOKABLE void rotate(QVector2D tiltPan);
    Q_INVOKABLE void rotate(qreal tilt /*pitch*/, qreal pan /*yaw*/, qreal roll = 0.0);

Q_SIGNALS:
    void engineChanged(EngineWrapper * engine);

    void eulerAnglesChanged(QVector3D euelerAngles);
    void cameraPositionChanged(QVector3D cameraPosition);
    void fieldOfViewChanged(qreal fieldOfView);

    void dtChanged(qreal dt);
    void mouseLookSpeedChanged(qreal mouseLookSpeed);
    void keyboardLookSpeedChanged(qreal keyboardLookSpeed);
    void linearSpeedChanged(qreal linearSpeed);

    void scenePathChanged(QUrl scenePath);

    void useOffscreenTextureChanged(bool useOffscreenTexture);
    void wireFrameChanged(bool wireFrame);

public Q_SLOTS:
    void setEulerAngles(QVector3D newEulerAngles);
    void setCameraPosition(QVector3D cameraPosition);
    void setFieldOfView(qreal fieldOfView);

    void setDt(qreal dt);

    void setScenePath(QUrl scenePath);

private Q_SLOTS:
    void cleanup();
    void onWindowChanged(QQuickWindow * w);

private:
    static constexpr qreal kDefaultFov = 90.0f;

    EngineWrapper * engine = nullptr;

    QVector3D eulerAngles;
    QVector3D cameraPosition;
    qreal fieldOfView = kDefaultFov;

    qreal dt = 1.0 / 60.0;
    qreal mouseLookSpeed = 60.0;
    qreal keyboardLookSpeed = 20.0;
    qreal linearSpeed = 1.0;

    QTimer * const mousePressAndHoldTimer = new QTimer{this};
    QPoint startPos;
    Qt::KeyboardModifiers keyboardModifiers = Qt::NoModifier;
    QHash<Qt::Key, int> pressedKeys;
    QTimer * const handleInputTimer = new QTimer{this};

    QUrl scenePath;
    bool isScenePathChanged = false;

    bool useOffscreenTexture = false;
    bool wireFrame = false;

    float characteristicSize = 0.0f;

    std::unique_ptr<Renderer> renderer;

    void setScene();

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
