#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <viewer/engine_wrapper.hpp>
#include <viewer/renderer.hpp>
#include <viewer/viewer.hpp>

#include <engine/engine.hpp>

#include <vulkan/vulkan.hpp>

#include <QtCore/QDebug>
#include <QtCore/QLoggingCategory>
#include <QtCore/QRunnable>
#include <QtGui/QVulkanInstance>
#include <QtQuick/QQuickItem>
#include <QtQuick/QQuickWindow>
#include <QtQuick/QSGRendererInterface>

#include <chrono>
#include <memory>

namespace viewer
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(viewerCategory)
Q_LOGGING_CATEGORY(viewerCategory, "viewer.viewer")
}  // namespace

class Viewer::CleanupJob : public QRunnable
{
public:
    CleanupJob(std::unique_ptr<Renderer> && renderer) : renderer{std::move(renderer)}
    {}

    void run() override
    {
        renderer.reset();
    }

private:
    std::unique_ptr<Renderer> renderer;
};

Viewer::Viewer()
{
    setFlag(QQuickItem::Flag::ItemHasContents);

    connect(this, &QQuickItem::windowChanged, this, &Viewer::onWindowChanged);

    const auto restartTimer = [this]
    {
        bool ok = false;
        int fps = property("fps").toInt(&ok);
        Q_ASSERT(ok);
        if (fps > 0) {
            using namespace std::chrono_literals;
            updateTimer->start(std::chrono::milliseconds(1s) / fps);
        } else {
            updateTimer->stop();
        }
    };
    restartTimer();
    connect(this, &Viewer::fpsChanged, updateTimer, restartTimer);
}

Viewer::~Viewer() = default;

void Viewer::onWindowChanged(QQuickWindow * w)
{
    if (!w) {
        qCDebug(viewerCategory) << "Window is lost";
        return;
    }

    INVARIANT(w->graphicsApi() == QSGRendererInterface::GraphicsApi::Vulkan, "Expected Vulkan backend");

    connect(w, &QQuickWindow::beforeSynchronizing, this, &Viewer::sync, Qt::DirectConnection);
    connect(w, &QQuickWindow::sceneGraphInvalidated, this, &Viewer::cleanup, Qt::DirectConnection);
    connect(w, &QQuickWindow::beforeRendering, this, &Viewer::frameStart, Qt::DirectConnection);
    connect(w, &QQuickWindow::beforeRenderPassRecording, this, &Viewer::renderPassRecordingStart, Qt::DirectConnection);

    connect(updateTimer, &QTimer::timeout, this, &QQuickItem::update);
}

void Viewer::sync()
{
    if (renderer) {
        renderer->setT(t);
    }
}

void Viewer::cleanup()
{
    renderer.reset();
}

void Viewer::frameStart()
{
    auto w = window();
    auto ri = w->rendererInterface();

    if (!renderer) {
        auto vulkanInstance = static_cast<QVulkanInstance *>(ri->getResource(w, QSGRendererInterface::Resource::VulkanInstanceResource));
        Q_CHECK_PTR(vulkanInstance);

        auto vulkanPhysicalDevice = static_cast<vk::PhysicalDevice *>(ri->getResource(w, QSGRendererInterface::Resource::PhysicalDeviceResource));
        Q_CHECK_PTR(vulkanPhysicalDevice);

        auto vulkanDevice = static_cast<vk::Device *>(ri->getResource(w, QSGRendererInterface::Resource::DeviceResource));
        Q_CHECK_PTR(vulkanInstance);

        uint32_t queueFamilyIndex = 0;  // chosen by smart heuristics

        auto vulkanQueue = static_cast<vk::Queue *>(ri->getResource(w, QSGRendererInterface::Resource::CommandQueueResource));
        Q_CHECK_PTR(vulkanInstance);

        if (engine) {
            INVARIANT(vulkanInstance->vkInstance() == engine->get().getInstance(), "Should match");
            INVARIANT(*vulkanPhysicalDevice == engine->get().getPhysicalDevice(), "Should match");
            INVARIANT(*vulkanDevice == engine->get().getDevice(), "Should match");
            INVARIANT(queueFamilyIndex == engine->get().getGraphicsQueueFamilyIndex(), "Should match");
            VkQueue queue = VK_NULL_HANDLE;
            vulkanInstance->deviceFunctions(*vulkanDevice)->vkGetDeviceQueue(*vulkanDevice, queueFamilyIndex, engine->get().getGraphicsQueueIndex(), &queue);
            INVARIANT(*vulkanQueue == vk::Queue(queue), "Should match");
        }

#ifdef GET_INSTANCE_PROC_ADDR
#error "!"
#endif
#define GET_INSTANCE_PROC_ADDR(name) PFN_##name name = utils::autoCast(vulkanInstance->getInstanceProcAddr(#name))
        GET_INSTANCE_PROC_ADDR(vkGetInstanceProcAddr);
        GET_INSTANCE_PROC_ADDR(vkGetDeviceProcAddr);
#undef GET_INSTANCE_PROC_ADDR
        renderer = std::make_unique<Renderer>(vkGetInstanceProcAddr, vulkanInstance, *vulkanPhysicalDevice, vkGetDeviceProcAddr, *vulkanDevice, queueFamilyIndex, *vulkanQueue);
    }
    if (renderer) {
        const QQuickWindow::GraphicsStateInfo & graphicsStateInfo = w->graphicsStateInfo();
        renderer->frameStart(graphicsStateInfo);
    }
}

void Viewer::renderPassRecordingStart()
{
    if (!renderer) {
        return;
    }

    auto w = window();
    auto ri = w->rendererInterface();

    w->beginExternalCommands();
    {
        auto commandBuffer = static_cast<vk::CommandBuffer *>(ri->getResource(w, QSGRendererInterface::Resource::CommandListResource));
        Q_CHECK_PTR(commandBuffer);

        auto renderPass = static_cast<vk::RenderPass *>(ri->getResource(w, QSGRendererInterface::Resource::RenderPassResource));
        Q_CHECK_PTR(renderPass);

        const QQuickWindow::GraphicsStateInfo & graphicsStateInfo = w->graphicsStateInfo();

        auto viewportSize = size();
        viewportSize *= w->devicePixelRatio();
        renderer->render(*commandBuffer, *renderPass, graphicsStateInfo, viewportSize);
    }
    w->endExternalCommands();
}

void Viewer::releaseResources()
{
    window()->scheduleRenderJob(new CleanupJob{std::move(renderer)}, QQuickWindow::BeforeSynchronizingStage);
}

}  // namespace viewer
