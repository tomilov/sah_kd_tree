#pragma once

#include <engine/fwd.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <vulkan/vulkan.hpp>
#include <glm/mat4x4.hpp>

#include <QtCore/QByteArray>
#include <QtCore/QRectF>
#include <QtQuick/QQuickWindow>

#include <cstdint>

namespace viewer
{
class ResourceManager;

class Renderer : utils::NonCopyable
{
public:
    Renderer(const engine::Engine & engine, const ResourceManager & resourceManager);
    ~Renderer();

    void setT(float t);

    void frameStart(const QQuickWindow::GraphicsStateInfo & graphicsStateInfo, qreal alpha);
    void render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo, const QRectF & viewportRect, const glm::dmat4x4 & viewMatrix);

private:
    struct Impl;

    static constexpr size_t kSize = 80;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;
};

}  // namespace viewer
