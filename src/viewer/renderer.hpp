#pragma once

#include <engine/fwd.hpp>
#include <utils/assert.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <glm/ext/quaternion_float.hpp>
#include <glm/mat4x4.hpp>
#include <vulkan/vulkan.hpp>

#include <QtCore/QByteArray>
#include <QtCore/QRectF>
#include <QtQuick/QQuickWindow>

#include <filesystem>
#include <string_view>

#include <cstdint>

namespace viewer
{
class SceneManager;

class Renderer : utils::NonCopyable
{
public:
    Renderer(std::string_view token, const std::filesystem::path & scenePath, const engine::Context & context, const SceneManager & sceneManager);
    ~Renderer();

    void setPosition(const glm::vec3 & position);
    void setOrientation(const glm::quat & orientation);
    void setT(float t);
    void setAlpha(qreal alpha);

    void setViewportRect(const QRectF & viewportRect);
    void setViewTransform(const glm::dmat3 & viewTransform);

    [[nodiscard]] const std::filesystem::path & getScenePath() const;

    void frameStart(const QQuickWindow::GraphicsStateInfo & graphicsStateInfo);
    void render(vk::CommandBuffer commandBuffer, vk::RenderPass renderPass, const QQuickWindow::GraphicsStateInfo & graphicsStateInfo);

private:
    struct Impl;

    static constexpr size_t kSize = 280;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;
};

}  // namespace viewer
