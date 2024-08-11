#pragma once

#include <builder/fwd.hpp>
#include <utils/fast_pimpl.hpp>

#include <QtCore/QFutureWatcher>
#include <QtCore/QRectF>
#include <QtCore/QSharedPointer>
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
struct Scene;

class RenderNode final : public QSGRenderNode
{
public:
    using ScenePtr = std::shared_ptr<const Scene>;
    using TreePtr = std::shared_ptr<const builder::Tree>;

    explicit RenderNode(QQuickWindow * window, const EngineWrapper & engineWrapper);

    void unsetScene();
    void updateScene(const ScenePtr & scene);
    [[nodiscard]] const ScenePtr & getScene() const &;

    void unsetTree();
    void updateTree(const TreePtr & tree);
    [[nodiscard]] const TreePtr & getTree() const &;

    void updateRect(const QRectF & rect);
    void updateMode(bool useOffscreenTexture, bool discardInvisible, bool wireFrame);
    void updateCamera(const QVector3D & cameraPosition, const QQuaternion & cameraOrientation, float cameraFov, float zNear, float zFar);
    void updateClearColor(const QColor & clearColor);
    void updateRenderdocCaptureFrameCounter(int renderdocCaptureFrameCounter);
    void updateDirty();

private:
    struct Impl;

    static constexpr size_t kSize = 896;
    static constexpr size_t kAlignment = 8;
    utils::FastPimpl<Impl, kSize, kAlignment> impl_;

    void prepare() override;
    void render(const RenderState * renderState) override;
    void releaseResources() override;  // https://bugreports.qt.io/browse/QTBUG-121137
    [[nodiscard]] RenderingFlags flags() const override;
    [[nodiscard]] QRectF rect() const override;
    [[nodiscard]] StateFlags changedStates() const override;
};

}  // namespace viewer
