#pragma once

#include <QtCore/QObject>
#include <QtCore/QSet>
#include <QtCore/QTimer>
#include <QtGui/QKeyEvent>
#include <QtGui/QMouseEvent>
#include <QtGui/QVector2D>
#include <QtGui/QVector3D>
#include <QtGui/QWheelEvent>
#include <QtQuick/QQuickItem>
#include <QtQuick/QQuickWindow>

#include <memory>

namespace viewer
{
class Engine;
struct FrameSettings;
class Renderer;

class Viewer : public QQuickItem
{
    Q_OBJECT
    QML_NAMED_ELEMENT(SahKdTreeViewer)

    Q_PROPERTY(QVector3D eulerAngles MEMBER eulerAngles WRITE setEulerAngles NOTIFY eulerAnglesChanged)
    Q_PROPERTY(QVector3D cameraPosition MEMBER cameraPosition NOTIFY cameraPositionChanged)
    Q_PROPERTY(qreal fieldOfView MEMBER fieldOfView NOTIFY fieldOfViewChanged)

    Q_PROPERTY(qreal dt MEMBER dt NOTIFY dtChanged)
    Q_PROPERTY(qreal mouseLookSpeed MEMBER mouseLookSpeed NOTIFY mouseLookSpeedChanged)
    Q_PROPERTY(qreal keyboardLookSpeed MEMBER keyboardLookSpeed NOTIFY keyboardLookSpeedChanged)
    Q_PROPERTY(qreal linearSpeed MEMBER linearSpeed NOTIFY linearSpeedChanged)

    Q_PROPERTY(Engine * engine MEMBER engine NOTIFY engineChanged REQUIRED)

    Q_PROPERTY(float t MEMBER t NOTIFY tChanged)
    Q_PROPERTY(QUrl scenePath MEMBER scenePath NOTIFY scenePathChanged)

public:
    Viewer();
    ~Viewer() override;

    Q_INVOKABLE void rotate(QVector3D tiltPanRoll);
    Q_INVOKABLE void rotate(QVector2D tiltPan);
    Q_INVOKABLE void rotate(qreal tilt /*pitch*/, qreal pan /*yaw*/, qreal roll = 0.0);

Q_SIGNALS:
    void eulerAnglesChanged(QVector3D euelerAngles);
    void cameraPositionChanged(QVector3D cameraPosition);
    void fieldOfViewChanged(qreal fieldOfView);

    void dtChanged(qreal dt);
    void mouseLookSpeedChanged(qreal mouseLookSpeed);
    void keyboardLookSpeedChanged(qreal keyboardLookSpeed);
    void linearSpeedChanged(qreal linearSpeed);

    void engineChanged(Engine * engine);

    void scenePathChanged(QUrl scenePath);
    void tChanged(qreal t);

public Q_SLOTS:
    void setEulerAngles(QVector3D newEulerAngles);

private Q_SLOTS:
    void onWindowChanged(QQuickWindow * w);

    void sync();
    void cleanup();
    void beforeRendering();
    void beforeRenderPassRecording();

private:
    static constexpr qreal kDefaultFov = 90.0f;

    QVector3D eulerAngles;
    QVector3D cameraPosition;
    qreal fieldOfView = kDefaultFov;

    qreal dt = 1.0 / 60.0;
    qreal mouseLookSpeed = 60.0;
    qreal keyboardLookSpeed = 20.0;
    qreal linearSpeed = 1.0;

    QTimer * const doubleClickTimer = new QTimer{this};
    QPoint startPos;
    Qt::KeyboardModifiers keyboardModifiers = Qt::NoModifier;
    QSet<Qt::Key> pressedKeys;
    QTimer * const handleInputTimer = new QTimer{this};

    Engine * engine = nullptr;

    QUrl scenePath;
    float t = 0.0;

    std::unique_ptr<FrameSettings> frameSettings;
    std::unique_ptr<Renderer> renderer;

    void checkEngine() const;

    void handleKeyEvent(QKeyEvent * event, bool isPressed);
    void handleInput();

    void releaseResources() override;

    void wheelEvent(QWheelEvent * event) override;

    void mouseUngrabEvent() override;

    void mousePressEvent(QMouseEvent * event) override;
    void mouseMoveEvent(QMouseEvent * event) override;
    void mouseReleaseEvent(QMouseEvent * event) override;

    void mouseDoubleClickEvent(QMouseEvent * event) override;

    void focusInEvent(QFocusEvent * event) override;
    void focusOutEvent(QFocusEvent * event) override;

    void keyPressEvent(QKeyEvent * event) override;
    void keyReleaseEvent(QKeyEvent * event) override;
};

}  // namespace viewer
