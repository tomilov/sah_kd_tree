#include <viewer/example_renderer.hpp>
#include <viewer/viewer.hpp>

#include <QtCore/QLoggingCategory>
#include <QtCore/QRunnable>
#include <QtQuick/QQuickItem>
#include <QtQuick/QQuickWindow>

namespace viewer
{
Q_DECLARE_LOGGING_CATEGORY(viewerCategory)
Q_LOGGING_CATEGORY(viewerCategory, "viewer")

Viewer::Viewer()
{
    connect(this, &QQuickItem::windowChanged, this, &Viewer::handleWindowChanged);
}

void Viewer::setT(qreal t)
{
    if (t == m_t) return;
    m_t = t;
    Q_EMIT tChanged();
    if (window()) window()->update();
}

void Viewer::handleWindowChanged(QQuickWindow * win)
{
    if (win) {
        connect(win, &QQuickWindow::beforeSynchronizing, this, &Viewer::sync, Qt::DirectConnection);
        connect(win, &QQuickWindow::sceneGraphInvalidated, this, &Viewer::cleanup, Qt::DirectConnection);

        // Ensure we start with cleared to black. The squircle's blend mode relies on this.
        win->setColor(Qt::black);
    }
}

// The safe way to release custom graphics resources is to both connect to
// sceneGraphInvalidated() and implement releaseResources(). To support
// threaded render loops the latter performs the ExampleRenderer destruction
// via scheduleRenderJob(). Note that the Viewer may be gone by the time
// the QRunnable is invoked.

void Viewer::cleanup()
{
    delete m_renderer;
    m_renderer = nullptr;
}

class CleanupJob : public QRunnable
{
public:
    CleanupJob(ExampleRenderer * renderer) : m_renderer(renderer)
    {}
    void run() override
    {
        delete m_renderer;
    }

private:
    ExampleRenderer * m_renderer;
};

void Viewer::releaseResources()
{
    window()->scheduleRenderJob(new CleanupJob(m_renderer), QQuickWindow::BeforeSynchronizingStage);
    m_renderer = nullptr;
}

void Viewer::sync()
{
    if (!m_renderer) {
        m_renderer = new ExampleRenderer;
        // Initializing resources is done before starting to record the
        // renderpass, regardless of wanting an underlay or overlay.
        connect(window(), &QQuickWindow::beforeRendering, m_renderer, &ExampleRenderer::frameStart, Qt::DirectConnection);
        // Here we want an underlay and therefore connect to
        // beforeRenderPassRecording. Changing to afterRenderPassRecording
        // would render the squircle on top (overlay).
        connect(window(), &QQuickWindow::beforeRenderPassRecording, m_renderer, &ExampleRenderer::mainPassRecordingStart, Qt::DirectConnection);
    }
    m_renderer->setViewportSize(window()->size() * window()->devicePixelRatio());
    m_renderer->setT(m_t);
    m_renderer->setWindow(window());
}

}  // namespace viewer
