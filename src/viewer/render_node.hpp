#pragma once

#include <utils/fast_pimpl.hpp>

#include <QtCore/QRectF>
#include <QtGui/QColor>
#include <QtGui/QQuaternion>
#include <QtGui/QVector3D>
#include <QtQuick/QQuickWindow>
#include <QtQuick/QSGRenderNode>

#include <memory>

#include <cstddef>

namespace viewer
{
class EngineWrapper;
class Scene;

class RenderNode final : public QSGRenderNode
{
public:
    explicit RenderNode(QQuickWindow * window, const EngineWrapper * engineWrapper);

    void unsetScene();
    void setScene(std::shared_ptr<const Scene> scene);

    void updateRect(const QRectF & rect);
    void updateMode(bool useOffscreenTexture, bool discardInvisible, bool wireFrame);
    void updateCamera(const QVector3D & cameraPosition, const QQuaternion & cameraOrientation, float cameraFov, float zNear, float zFar);
    void setClearColor(const QColor & clearColor);
    void markDirty();

private:
    struct Impl;

    static constexpr size_t kSize = 816;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;

    void prepare() override;
    void render(const RenderState * renderState) override;
    void releaseResources() override;
    [[nodiscard]] RenderingFlags flags() const override;
    [[nodiscard]] QRectF rect() const override;
    [[nodiscard]] StateFlags changedStates() const override;
};

}  // namespace viewer
