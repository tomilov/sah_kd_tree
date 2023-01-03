#include <utils/assert.hpp>
#include <viewer/example_renderer.hpp>
#include <viewer/viewer.hpp>

#include <QtCore/QDebug>
#include <QtCore/QLoggingCategory>
#include <QtCore/QRunnable>
#include <QtQuick/QQuickItem>
#include <QtQuick/QQuickWindow>

#include <memory>
#include <new>

namespace viewer
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(viewerCategory)
Q_LOGGING_CATEGORY(viewerCategory, "viewer.viewer")
}  // namespace

class CleanupJob : public QRunnable
{
public:
    CleanupJob(std::unique_ptr<ExampleRenderer> && renderer) : renderer{std::move(renderer)}
    {}

    void run() override
    {
        renderer.reset();
    }

private:
    std::unique_ptr<ExampleRenderer> renderer;
};

void Viewer::sync()
{
    auto w = window();
    INVARIANT(w, "Window should exist");
    if (!renderer) {
        renderer = std::make_unique<ExampleRenderer>();

        // Initializing resources is done before starting to record the
        // renderpass, regardless of wanting an underlay or overlay.
        connect(w, &QQuickWindow::beforeRendering, renderer.get(), &ExampleRenderer::frameStart, Qt::DirectConnection);
        // Here we want an underlay and therefore connect to
        // beforeRenderPassRecording. Changing to afterRenderPassRecording
        // would render the squircle on top (overlay).
        connect(w, &QQuickWindow::beforeRenderPassRecording, renderer.get(), &ExampleRenderer::mainPassRecordingStart, Qt::DirectConnection);
    }
    renderer->setViewportSize(w->size() * w->devicePixelRatio());
    renderer->setT(t);
    renderer->setWindow(w);
}

void Viewer::cleanup()
{
    renderer.reset();
}

void Viewer::releaseResources()
{
    window()->scheduleRenderJob(new CleanupJob{std::move(renderer)}, QQuickWindow::BeforeSynchronizingStage);
}

Viewer::Viewer()
{
    connect(this, &QQuickItem::windowChanged, this, &Viewer::onWindowChanged);
}

Viewer::~Viewer() = default;

void Viewer::onWindowChanged(QQuickWindow * w)
{
    if (!w) {
        qCDebug(viewerCategory) << "Window lost";
        return;
    }

    INVARIANT(w->graphicsApi() == QSGRendererInterface::GraphicsApi::Vulkan, "Expected Vulkan backend");

    connect(w, &QQuickWindow::beforeSynchronizing, this, &Viewer::sync, Qt::DirectConnection);
    connect(w, &QQuickWindow::sceneGraphInvalidated, this, &Viewer::cleanup, Qt::DirectConnection);
    connect(this, &Viewer::tChanged, w, &QQuickWindow::update);

    w->setColor(Qt::GlobalColor::black);
}

}  // namespace viewer
