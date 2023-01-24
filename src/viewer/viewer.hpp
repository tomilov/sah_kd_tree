#pragma once

#include <QtCore/QObject>
#include <QtCore/QTimer>
#include <QtQuick/QQuickItem>
#include <QtQuick/QQuickWindow>

#include <memory>

namespace viewer
{
class Engine;
class Renderer;

class Viewer : public QQuickItem
{
    Q_OBJECT
    QML_NAMED_ELEMENT(SahKdTreeViewer)

    Q_PROPERTY(Engine * engine MEMBER engine NOTIFY engineChanged REQUIRED)

    Q_PROPERTY(float t MEMBER t NOTIFY tChanged)

public:
    Viewer();
    ~Viewer();

Q_SIGNALS:
    void engineChanged(viewer::Engine * engine);

    void tChanged(qreal t);

private Q_SLOTS:
    void onWindowChanged(QQuickWindow * window);

    void sync();
    void cleanup();
    void frameStart();
    void beforeRenderPassRecording();

private:
    Engine * engine = nullptr;

    float t = 0.0;

    std::unique_ptr<Renderer> renderer;

    void checkEngine() const;

    void releaseResources() override;
};

}  // namespace viewer
