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
Q_LOGGING_CATEGORY(viewerCategory, "viewer")
}  // namespace

struct Viewer::Impl
{
    qreal m_t = 0;

    void sync(QQuickWindow * window);

    void cleanup();
    void releaseResources(QQuickWindow * window);

private:
    class CleanupJob;

    std::unique_ptr<ExampleRenderer> m_renderer;
};

class Viewer::Impl::CleanupJob : public QRunnable
{
public:
    CleanupJob(std::unique_ptr<ExampleRenderer> && renderer) : m_renderer{std::move(renderer)}
    {}

    void run() override
    {
        m_renderer.reset();
    }

private:
    std::unique_ptr<ExampleRenderer> m_renderer;
};

void Viewer::Impl::sync(QQuickWindow * window)
{
    INVARIANT(window, "Window should exist");
    if (!m_renderer) {
        m_renderer = std::make_unique<ExampleRenderer>();

        // Initializing resources is done before starting to record the
        // renderpass, regardless of wanting an underlay or overlay.
        connect(window, &QQuickWindow::beforeRendering, m_renderer.get(), &ExampleRenderer::frameStart, Qt::DirectConnection);
        // Here we want an underlay and therefore connect to
        // beforeRenderPassRecording. Changing to afterRenderPassRecording
        // would render the squircle on top (overlay).
        connect(window, &QQuickWindow::beforeRenderPassRecording, m_renderer.get(), &ExampleRenderer::mainPassRecordingStart, Qt::DirectConnection);
    }
    m_renderer->setViewportSize(window->size() * window->devicePixelRatio());
    m_renderer->setT(m_t);
    m_renderer->setWindow(window);
}

void Viewer::Impl::cleanup()
{
    m_renderer.reset();
}

void Viewer::Impl::releaseResources(QQuickWindow * window)
{
    window->scheduleRenderJob(new CleanupJob{std::move(m_renderer)}, QQuickWindow::BeforeSynchronizingStage);
}

Viewer::Viewer()
{
    connect(this, &QQuickItem::windowChanged, this, &Viewer::onWindowChanged);
}

Viewer::~Viewer() = default;

qreal Viewer::t() const
{
    return impl_->m_t;
}

void Viewer::setT(qreal t)
{
    if (t == impl_->m_t) {
        return;
    }
    impl_->m_t = t;
    Q_EMIT tChanged(impl_->m_t);

    if (auto w = window()) {
        w->update();
    }
}

void Viewer::sync()
{
    impl_->sync(window());
}

// The safe way to release custom graphics resources is to both connect to
// sceneGraphInvalidated() and implement releaseResources(). To support
// threaded render loops the latter performs the ExampleRenderer destruction
// via scheduleRenderJob(). Note that the Viewer may be gone by the time
// the QRunnable is invoked.

void Viewer::cleanup()
{
    impl_->cleanup();
}

void Viewer::onWindowChanged(QQuickWindow * window)
{
    if (!window) {
        qCInfo(viewerCategory) << "Window lost";
        return;
    }

    INVARIANT(window->graphicsApi() == QSGRendererInterface::GraphicsApi::Vulkan, "Expected Vulkan backend");

    connect(window, &QQuickWindow::beforeSynchronizing, this, &Viewer::sync, Qt::DirectConnection);
    connect(window, &QQuickWindow::sceneGraphInvalidated, this, &Viewer::cleanup, Qt::DirectConnection);

    window->setColor(Qt::GlobalColor::black);
}

void Viewer::releaseResources()
{
    impl_->releaseResources(window());
}

}  // namespace viewer
