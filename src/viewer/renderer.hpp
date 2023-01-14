#pragma once

#include <engine/fwd.hpp>
#include <utils/fast_pimpl.hpp>

#include <vulkan/vulkan.hpp>

#include <QtCore/QByteArray>
#include <QtCore/QSizeF>
#include <QtQuick/QQuickWindow>

#include <cstdint>

namespace viewer
{
class ResourceManager;

class Renderer
{
public:
    Renderer(engine::Engine & engine, ResourceManager & resourceManager);
    ~Renderer();

    void setT(float t);

    void frameStart(const QQuickWindow::GraphicsStateInfo & graphicsStateInfo);
    void render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo, const QSizeF & size);

private:
    struct Impl;

    static constexpr size_t kSize = 88;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;
};

}  // namespace viewer
