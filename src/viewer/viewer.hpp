#pragma once

#include <viewer/qml_engine_wrapper.hpp>

#include <QtCore/QObject>
#include <QtQuick/QQuickItem>
#include <QtQuick/QQuickWindow>

namespace viewer
{
class ExampleRenderer;

class Viewer : public QQuickItem
{
    Q_OBJECT
    QML_NAMED_ELEMENT(SahKdTreeViewer)

    Q_PROPERTY(Engine * engine MEMBER engine NOTIFY engineChanged)
    Q_PROPERTY(qreal t MEMBER t NOTIFY tChanged)

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

private:
    qreal t = 0.0;
    std::unique_ptr<ExampleRenderer> renderer;
    Engine * engine = nullptr;

    void releaseResources() override;
};

}  // namespace viewer
