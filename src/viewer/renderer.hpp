#pragma once

#include <engine/fwd.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <glm/mat4x4.hpp>
#include <vulkan/vulkan.hpp>

#include <QtCore/QByteArray>
#include <QtCore/QRectF>
#include <QtQuick/QQuickWindow>

#include <cstdint>

namespace viewer
{
class SceneManager;

class Renderer : utils::NonCopyable
{
public:
    Renderer(const engine::Engine & engine, const SceneManager & sceneManager);
    ~Renderer();

    void setT(float t);
    void setAlpha(qreal alpha);

    void setViewportRect(const QRectF & viewportRect);
    void setViewTransform(const glm::dmat3 & viewTransform);

    void frameStart(const QQuickWindow::GraphicsStateInfo & graphicsStateInfo);
    void render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo);

private:
    struct Impl;

    static constexpr size_t kSize = 168;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;
};

}  // namespace viewer
