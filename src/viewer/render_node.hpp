#pragma once

#include <builder/fwd.hpp>
#include <scene_data/fwd.hpp>

#include <QtCore/QFutureWatcher>
#include <QtCore/QRectF>
#include <QtCore/QSharedPointer>
#include <QtCore/QString>
#include <QtGui/QColor>
#include <QtGui/QQuaternion>
#include <QtGui/QVector3D>
#include <QtQuick/QQuickWindow>
#include <QtQuick/QSGRenderNode>

#include <memory>

namespace viewer
{
class EngineWrapper;
struct Scene;

class RenderNode final : public QSGRenderNode
{
public:
    explicit RenderNode(QString name, QQuickWindow * window, const EngineWrapper & engineWrapper);

    void unsetScene();
    void updateScene(const scene_data::SceneDataPtr & sceneData);
    [[nodiscard]] const scene_data::SceneDataPtr & getScene() const &;

    void setTree(builder::TreePtr && tree);

    void updateRect(const QRectF & rect);
    void updateMode(bool traceSahKdTree, bool useOffscreenTexture, bool discardInvisible, bool wireframe);
    void updateCamera(const QVector3D & cameraPosition, const QQuaternion & cameraOrientation, float cameraFov, float zNear, float zFar);
    void updateClearColor(const QColor & clearColor);
    void updateRenderdocCaptureFrameCounter(int renderdocCaptureFrameCounter);
    void updateDirty();

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;

    void prepare() override;
    void render(const RenderState * renderState) override;
    void releaseResources() override;  // https://bugreports.qt.io/browse/QTBUG-121137
    [[nodiscard]] RenderingFlags flags() const override;
    [[nodiscard]] QRectF rect() const override;
    [[nodiscard]] StateFlags changedStates() const override;
};

}  // namespace viewer
