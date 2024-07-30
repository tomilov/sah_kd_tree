#pragma once

#include <utils/fast_pimpl.hpp>
#include <builder/fwd.hpp>
#include <QtCore/QRectF>
#include <QtGui/QColor>
#include <QtGui/QQuaternion>
#include <QtGui/QVector3D>
#include <QtQuick/QQuickWindow>
#include <QtQuick/QSGRenderNode>

#include <memory>
#include <filesystem>

#include <cstddef>

namespace viewer
{
class EngineWrapper;
class Scene;

class RenderNode final : public QSGRenderNode
{
public:
    explicit RenderNode(QQuickWindow * window, const EngineWrapper & engineWrapper);

    void unsetScene();
    bool setScene(const std::filesystem::path & scenePath);
    [[nodiscard]] const std::shared_ptr<const Scene> & getScene() const &;

    void unsetTree();
    [[nodiscard]] const std::shared_ptr<const builder::Tree> & getTree() const &;
    bool updateTree(float emptinessFactor, float traversalCost, float intersectionCost, uint32_t maxDepth);

    void updateRect(const QRectF & rect);
    void updateMode(bool useOffscreenTexture, bool discardInvisible, bool wireFrame);
    void updateCamera(const QVector3D & cameraPosition, const QQuaternion & cameraOrientation, float cameraFov, float zNear, float zFar);
    void setClearColor(const QColor & clearColor);
    void setRenderdocCaptureFrameCounter(int renderdocCaptureFrameCounter);
    void markDirty();

private:
    struct Impl;

    static constexpr size_t kSize = 896;
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
