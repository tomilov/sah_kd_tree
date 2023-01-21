#include <engine/engine.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <viewer/engine_wrapper.hpp>
#include <viewer/renderer.hpp>
#include <viewer/viewer.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_operation.hpp>
#include <glm/mat4x4.hpp>
#include <glm/trigonometric.hpp>
#include <glm/vec3.hpp>
#include <vulkan/vulkan.hpp>

#include <QtCore/QDebug>
#include <QtCore/QLoggingCategory>
#include <QtCore/QObject>
#include <QtCore/QRectF>
#include <QtCore/QRunnable>
#include <QtGui/QMatrix4x4>
#include <QtGui/QVector2D>
#include <QtGui/QVector3D>
#include <QtGui/QVulkanInstance>
#include <QtQml/QQmlListReference>
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
        if (engine) {
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
    }
    if (renderer) {
        auto alpha = opacity();
        auto p = parentItem();
        while (qobject_cast<const QQuickItem *>(p)) {
            alpha *= p->opacity();
            p = p->parentItem();
        }
        renderer->frameStart(w->graphicsStateInfo(), alpha);
    }
}

void Viewer::renderPassRecordingStart()
{
    if (!isVisible() || qFuzzyIsNull(opacity())) {
        return;
    }

    if (!renderer) {
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

        auto viewportRect = mapRectToScene(boundingRect());

        {
            QMatrix4x4 matrix;
            QQmlListReference transforms{this, "transform"};
            ASSERT(transforms.isValid());
            ASSERT(transforms.canCount());
            qsizetype transformCount = transforms.count();
            ASSERT(transforms.canAt());
            for (qsizetype i = 0; i < transformCount; ++i) {
                auto t = qobject_cast<const QQuickTransform *>(transforms.at(i));
                Q_CHECK_PTR(t);
                t->applyTo(&matrix);
            }
            qDebug() << matrix;
        }

        if ((false)) {
            QMatrix4x4 matrix;
            matrix.scale(float(utils::autoCast(viewportRect.height() / viewportRect.width())), 1.0f);
            matrix.rotate(utils::autoCast(rotation()), 0.0f, 0.0f, 1.0f);
            matrix.scale(float(utils::autoCast(width() / viewportRect.height())), float(utils::autoCast(height() / viewportRect.height())));
            qDebug() << matrix;
        }

        auto angle = rotation();
        auto scaleFactor = scale();
        auto p = parentItem();
        while (qobject_cast<const QQuickItem *>(p)) {
            angle += p->rotation();
            scaleFactor *= p->scale();
            p = p->parentItem();
        }

        glm::dmat4x4 viewMatrix = glm::diagonal4x4(glm::dvec4{glm::dvec3{scaleFactor}, 1.0});
        viewMatrix = glm::scale(viewMatrix, glm::dvec3{viewportRect.height() / viewportRect.width(), 1.0, 1.0});
        viewMatrix = glm::rotate(viewMatrix, glm::radians(angle), glm::dvec3{0.0, 0.0, 1.0});
        viewMatrix = glm::scale(viewMatrix, glm::dvec3{width() / viewportRect.height(), height() / viewportRect.height(), 1.0});

        auto devicePixelRatio = w->effectiveDevicePixelRatio();
        renderer->render(*commandBuffer, *renderPass, w->graphicsStateInfo(), {viewportRect.topLeft() * devicePixelRatio, viewportRect.size() * devicePixelRatio}, viewMatrix);
    }
    w->endExternalCommands();
}

void Viewer::releaseResources()
{
    window()->scheduleRenderJob(new CleanupJob{std::move(renderer)}, QQuickWindow::BeforeSynchronizingStage);
}

}  // namespace viewer
