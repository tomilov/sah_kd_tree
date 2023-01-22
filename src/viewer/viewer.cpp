﻿#include <engine/engine.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <viewer/engine_wrapper.hpp>
#include <viewer/renderer.hpp>
#include <viewer/viewer.hpp>

#include <glm/ext/matrix_transform.hpp>
#include <glm/gtx/matrix_operation.hpp>
#include <glm/mat4x4.hpp>
#include <glm/trigonometric.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <vulkan/vulkan.hpp>

#include <QtCore/QDebug>
#include <QtCore/QLoggingCategory>
#include <QtCore/QObject>
#include <QtCore/QRunnable>
#include <QtGui/QVulkanInstance>
#include <QtQuick/QQuickItem>
#include <QtQuick/QQuickWindow>
#include <QtQuick/QSGRendererInterface>

#include <chrono>
#include <memory>

#include <cstdint>

using namespace Qt::StringLiterals;

namespace viewer
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(viewerCategory)
Q_LOGGING_CATEGORY(viewerCategory, "viewer.viewer")

class CleanupJob : public QRunnable
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

}  // namespace

Viewer::Viewer()
{
    setFlag(QQuickItem::Flag::ItemHasContents);

    connect(this, &QQuickItem::windowChanged, this, &Viewer::onWindowChanged);
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
    connect(w, &QQuickWindow::beforeRenderPassRecording, this, &Viewer::beforeRenderPassRecording, Qt::DirectConnection);
}

void Viewer::sync()
{
    if (renderer) {
        auto alpha = opacity();
        auto p = parentItem();
        while (qobject_cast<const QQuickItem *>(p)) {
            alpha *= p->opacity();
            p = p->parentItem();
        }
        renderer->setAlpha(alpha);

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

    if (!renderer && engine) {
        {
            auto vulkanInstance = static_cast<QVulkanInstance *>(ri->getResource(w, QSGRendererInterface::Resource::VulkanInstanceResource));
            Q_CHECK_PTR(vulkanInstance);

            auto vulkanPhysicalDevice = static_cast<vk::PhysicalDevice *>(ri->getResource(w, QSGRendererInterface::Resource::PhysicalDeviceResource));
            Q_CHECK_PTR(vulkanPhysicalDevice);

            auto vulkanDevice = static_cast<vk::Device *>(ri->getResource(w, QSGRendererInterface::Resource::DeviceResource));
            Q_CHECK_PTR(vulkanInstance);

            uint32_t queueFamilyIndex = 0;  // chosen by smart heuristics

            auto vulkanQueue = static_cast<vk::Queue *>(ri->getResource(w, QSGRendererInterface::Resource::CommandQueueResource));
            Q_CHECK_PTR(vulkanInstance);

#ifdef GET_INSTANCE_PROC_ADDR
#error "!"
#endif
#define GET_INSTANCE_PROC_ADDR(name) PFN_##name name = utils::autoCast(vulkanInstance->getInstanceProcAddr(#name))
            // GET_INSTANCE_PROC_ADDR(vkGetInstanceProcAddr);
            GET_INSTANCE_PROC_ADDR(vkGetDeviceProcAddr);
#undef GET_INSTANCE_PROC_ADDR
            PFN_vkGetDeviceQueue vkGetDeviceQueue = utils::autoCast(vkGetDeviceProcAddr(*vulkanDevice, "vkGetDeviceQueue"));

            INVARIANT(vk::Instance(vulkanInstance->vkInstance()) == engine->getEngine().getVulkanInstance(), "Should match");
            INVARIANT(*vulkanPhysicalDevice == engine->getEngine().getVulkanPhysicalDevice(), "Should match");
            INVARIANT(*vulkanDevice == engine->getEngine().getVulkanDevice(), "Should match");
            INVARIANT(queueFamilyIndex == engine->getEngine().getVulkanGraphicsQueueFamilyIndex(), "Should match");
            {
                VkQueue queue = VK_NULL_HANDLE;
                vkGetDeviceQueue(*vulkanDevice, queueFamilyIndex, engine->getEngine().getVulkanGraphicsQueueIndex(), &queue);
                INVARIANT(*vulkanQueue == vk::Queue(queue), "Should match");
            }
        }

        renderer = std::make_unique<Renderer>(engine->getEngine(), engine->getResourceManager());
    }
    if (renderer) {
        renderer->frameStart(w->graphicsStateInfo());
    }
}

void Viewer::beforeRenderPassRecording()
{
    if (!renderer) {
        return;
    }

    if (!isVisible() || !boundingRect().isValid()) {
        return;
    }

    auto w = window();

    w->beginExternalCommands();
    {
        auto ri = w->rendererInterface();

        auto commandBuffer = static_cast<vk::CommandBuffer *>(ri->getResource(w, QSGRendererInterface::Resource::CommandListResource));
        Q_CHECK_PTR(commandBuffer);

        auto renderPass = static_cast<vk::RenderPass *>(ri->getResource(w, QSGRendererInterface::Resource::RenderPassResource));
        Q_CHECK_PTR(renderPass);

        auto angle = rotation();
        auto scaleFactor = scale();
        auto p = parentItem();
        while (qobject_cast<const QQuickItem *>(p)) {
            angle += p->rotation();
            scaleFactor *= p->scale();
            p = p->parentItem();
        }

        auto viewportRect = mapRectToScene(boundingRect());
        glm::dmat4x4 viewTransform = glm::diagonal4x4(glm::dvec4{glm::dvec3{scaleFactor}, 1.0});
        viewTransform = glm::scale(viewTransform, glm::dvec3{viewportRect.height() / viewportRect.width(), 1.0, 1.0});
        const glm::dvec3 zAxis = {0.0, 0.0, 1.0};
        viewTransform = glm::rotate(viewTransform, glm::radians(angle), zAxis);
        viewTransform = glm::scale(viewTransform, glm::dvec3{width() / viewportRect.height(), height() / viewportRect.height(), 1.0});

        auto devicePixelRatio = w->effectiveDevicePixelRatio();
        renderer->render(*commandBuffer, *renderPass, w->graphicsStateInfo(), {viewportRect.topLeft() * devicePixelRatio, viewportRect.size() * devicePixelRatio}, viewTransform);
    }
    w->endExternalCommands();
}

void Viewer::releaseResources()
{
    window()->scheduleRenderJob(new CleanupJob{std::move(renderer)}, QQuickWindow::BeforeSynchronizingStage);
}

}  // namespace viewer
