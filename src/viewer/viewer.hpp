#pragma once

#include <QtCore/QObject>
#include <QtCore/QTimer>
#include <QtQuick/QQuickItem>
#include <QtQuick/QQuickWindow>

#include <memory>

namespace viewer
{
class Engine;
class ExampleRenderer;

class Viewer : public QQuickItem
{
    Q_OBJECT
    QML_NAMED_ELEMENT(SahKdTreeViewer)

    Q_PROPERTY(Engine * engine MEMBER engine NOTIFY engineChanged)
    Q_PROPERTY(int fps MEMBER fps NOTIFY fpsChanged)

    Q_PROPERTY(qreal t MEMBER t NOTIFY tChanged)

public:
    Viewer();
    ~Viewer();

Q_SIGNALS:
    void engineChanged(viewer::Engine * engine);
    void fpsChanged(int fps);

    void tChanged(qreal t);

private Q_SLOTS:
    void onWindowChanged(QQuickWindow * window);

    void sync();
    void cleanup();
    void frameStart();
    void renderPassRecordingStart();

private:
    class CleanupJob;

    Engine * engine = nullptr;
    int fps = 144;

    qreal t = 0.0;

    std::unique_ptr<ExampleRenderer> renderer;

    QTimer * const updateTimer = new QTimer{this};

    void releaseResources() override;
};

}  // namespace viewer
