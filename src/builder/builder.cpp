#include <builder/build_from_triangles.hpp>
#include <builder/builder.hpp>
#include <scene_data/scene_data.hpp>
#include <scene_loader/scene_loader.hpp>

#include <thrust/device_vector.h>

#include <QtCore/QDebug>
#include <QtCore/QFileInfo>
#include <QtCore/QLoggingCategory>
#include <QtCore/QString>

#include <iterator>

using namespace Qt::StringLiterals;

namespace builder
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(builderLog)
Q_LOGGING_CATEGORY(builderLog, "builder")
}  // namespace

bool buildSceneFromFile(QString sceneFileName, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth)
{
    scene_data::SceneData scene;
    QFileInfo sceneFileInfo{sceneFileName};
    if (!scene_loader::load(scene, sceneFileInfo)) {
        qCDebug(builderLog).noquote() << u"Cannot load scene from file %1"_s.arg(sceneFileName);
        return false;
    }
    auto triangles = scene.makeTriangles();
    return buildSceneFromTriangles(triangles.begin(), triangles.end(), emptinessFactor, traversalCost, intersectionCost, maxDepth);
}

bool buildSceneFromFileOrCache(QString sceneFileName, QString cachePath, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth)
{
    scene_data::SceneData scene;
    QFileInfo sceneFileInfo{sceneFileName};
    if (!scene_loader::cachingLoad(scene, sceneFileInfo, cachePath.isEmpty() ? QDir::temp() : cachePath)) {
        qCDebug(builderLog).noquote() << u"Cannot load scene from file %1"_s.arg(sceneFileName);
        return false;
    }
    auto triangles = scene.makeTriangles();
    return buildSceneFromTriangles(triangles.begin(), triangles.end(), emptinessFactor, traversalCost, intersectionCost, maxDepth);
}
}  // namespace builder
