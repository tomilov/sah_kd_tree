#include <sah_kd_tree/helpers/setup.cuh>
#include <sah_kd_tree/sah_kd_tree.cuh>
#include <scene_loader/scene_loader.hpp>

#include <builder/build_from_triangles.hpp>
#include <builder/builder.hpp>

#include <thrust/device_vector.h>

#include <QDebug>

namespace builder
{
Q_LOGGING_CATEGORY(builderLog, "builder")

bool buildSceneFromFile(QString sceneFileName, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth)
{
    scene_loader::SceneLoader sceneLoader;
    if (!sceneLoader.load(sceneFileName)) {
        return false;
    }
    auto trianglesBegin = std::data(sceneLoader.triangles);
    auto trianglesEnd = trianglesBegin + std::size(sceneLoader.triangles);
    return buildSceneFromTriangles(trianglesBegin, trianglesEnd, emptinessFactor, traversalCost, intersectionCost, maxDepth);
}

bool buildSceneFromFileOrCache(QString sceneFileName, QString cachePath, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth)
{
    scene_loader::SceneLoader sceneLoader;
    if (!sceneLoader.cachingLoad(sceneFileName, cachePath)) {
        return false;
    }
    auto trianglesBegin = std::data(sceneLoader.triangles);
    auto trianglesEnd = trianglesBegin + std::size(sceneLoader.triangles);
    return buildSceneFromTriangles(trianglesBegin, trianglesEnd, emptinessFactor, traversalCost, intersectionCost, maxDepth);
}
}  // namespace builder
