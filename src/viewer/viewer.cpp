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

struct Viewer::Impl
{
    qreal m_t = 0;
    ExampleRenderer * m_renderer = nullptr;
};

Viewer::Viewer()
{
    connect(this, &QQuickItem::windowChanged, this, &Viewer::handleWindowChanged);
}

Viewer::~Viewer() = default;

qreal Viewer::t() const
{
    return impl_->m_t;
}

void Viewer::setT(qreal t)
{
    if (t == impl_->m_t) return;
    impl_->m_t = t;
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
    delete impl_->m_renderer;
    impl_->m_renderer = nullptr;
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
    window()->scheduleRenderJob(new CleanupJob(impl_->m_renderer), QQuickWindow::BeforeSynchronizingStage);
    impl_->m_renderer = nullptr;
}

void Viewer::sync()
{
    if (!impl_->m_renderer) {
        impl_->m_renderer = new ExampleRenderer;
        // Initializing resources is done before starting to record the
        // renderpass, regardless of wanting an underlay or overlay.
        connect(window(), &QQuickWindow::beforeRendering, impl_->m_renderer, &ExampleRenderer::frameStart, Qt::DirectConnection);
        // Here we want an underlay and therefore connect to
        // beforeRenderPassRecording. Changing to afterRenderPassRecording
        // would render the squircle on top (overlay).
        connect(window(), &QQuickWindow::beforeRenderPassRecording, impl_->m_renderer, &ExampleRenderer::mainPassRecordingStart, Qt::DirectConnection);
    }
    impl_->m_renderer->setViewportSize(window()->size() * window()->devicePixelRatio());
    impl_->m_renderer->setT(impl_->m_t);
    impl_->m_renderer->setWindow(window());
}

}  // namespace viewer
