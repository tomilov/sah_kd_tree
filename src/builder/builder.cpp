#include <builder/build_from_triangles.hpp>
#include <builder/builder.hpp>
#include <scene_loader/scene_loader.hpp>

#include <thrust/device_vector.h>

#include <QtCore/QFileInfo>
#include <QtCore/QLoggingCategory>
#include <QtCore/QString>

#include <iterator>

namespace builder
{
Q_DECLARE_LOGGING_CATEGORY(builderLog)
Q_LOGGING_CATEGORY(builderLog, "builder")

bool buildSceneFromFile(QString sceneFileName, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth)
{
    scene_loader::SceneLoader sceneLoader;
    QFileInfo sceneFileInfo{sceneFileName};
    if (!sceneLoader.load(sceneFileInfo)) {
        return false;
    }
    auto trianglesBegin = std::data(sceneLoader.triangles);
    auto trianglesEnd = std::next(trianglesBegin, std::size(sceneLoader.triangles));
    return buildSceneFromTriangles(trianglesBegin, trianglesEnd, emptinessFactor, traversalCost, intersectionCost, maxDepth);
}

bool buildSceneFromFileOrCache(QString sceneFileName, QString cachePath, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth)
{
    scene_loader::SceneLoader sceneLoader;
    QFileInfo sceneFileInfo{sceneFileName};
    if (!sceneLoader.cachingLoad(sceneFileInfo, cachePath.isEmpty() ? QDir::temp() : cachePath)) {
        return false;
    }
    auto trianglesBegin = std::data(sceneLoader.triangles);
    auto trianglesEnd = std::next(trianglesBegin, std::size(sceneLoader.triangles));
    return buildSceneFromTriangles(trianglesBegin, trianglesEnd, emptinessFactor, traversalCost, intersectionCost, maxDepth);
}
}  // namespace builder
