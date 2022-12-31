#include <builder/build_from_triangles.hpp>
#include <builder/builder.hpp>
#include <scene_loader/scene_loader.hpp>

#include <thrust/device_vector.h>

#include <QtCore/QDebug>
#include <QtCore/QFileInfo>
#include <QtCore/QLoggingCategory>
#include <QtCore/QString>
#include <QtCore/QStringLiteral>

#include <iterator>

namespace builder
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(builderLog)
Q_LOGGING_CATEGORY(builderLog, "builder")
}  // namespace

bool buildSceneFromFile(QString sceneFileName, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth)
{
    scene_loader::SceneLoader sceneLoader;
    QFileInfo sceneFileInfo{sceneFileName};
    if (!sceneLoader.load(sceneFileInfo)) {
        qCDebug(builderLog).noquote() << QStringLiteral("Cannot load scene from file %1").arg(sceneFileName);
        return false;
    }
    auto trianglesBegin = std::data(sceneLoader.scene.triangles);
    auto trianglesEnd = std::next(trianglesBegin, std::size(sceneLoader.scene.triangles));
    return buildSceneFromTriangles(trianglesBegin, trianglesEnd, emptinessFactor, traversalCost, intersectionCost, maxDepth);
}

bool buildSceneFromFileOrCache(QString sceneFileName, QString cachePath, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth)
{
    scene_loader::SceneLoader sceneLoader;
    QFileInfo sceneFileInfo{sceneFileName};
    if (!sceneLoader.cachingLoad(sceneFileInfo, cachePath.isEmpty() ? QDir::temp() : cachePath)) {
        qCDebug(builderLog).noquote() << QStringLiteral("Cannot load scene from file %1").arg(sceneFileName);
        return false;
    }
    auto trianglesBegin = std::data(sceneLoader.scene.triangles);
    auto trianglesEnd = std::next(trianglesBegin, std::size(sceneLoader.scene.triangles));
    return buildSceneFromTriangles(trianglesBegin, trianglesEnd, emptinessFactor, traversalCost, intersectionCost, maxDepth);
}
}  // namespace builder
