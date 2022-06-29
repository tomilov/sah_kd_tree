#include "builder/builder.hpp"

#include "sah_kd_tree/sah_kd_tree.hpp"
#include "scene_loader/scene_loader.hpp"

#include <thrust/device_vector.h>

#include <QDebug>

Q_LOGGING_CATEGORY(builderLog, "builder")

namespace {

bool buildSceneFromTriangles(const QVector<sah_kd_tree::Triangle> & triangles, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth)
{
    using namespace sah_kd_tree;
    Builder builder;
    {
        thrust::device_vector<Triangle> deviceTriangles{triangles.cbegin(), triangles.cend()};
        builder.setTriangle(deviceTriangles.data(), deviceTriangles.data() + deviceTriangles.size());
        // deviceTriangles.clear() cause link error
    }
    Params params;
    if (emptinessFactor > 0.0f) {
        params.emptinessFactor = F(emptinessFactor);
    }
    if (traversalCost > 0.0f) {
        params.traversalCost = F(traversalCost);
    }
    if (intersectionCost > 0.0f) {
        params.intersectionCost = F(intersectionCost);
    }
    if (maxDepth > 0) {
        params.maxDepth = U(maxDepth);
    }
    Tree sahKdTree = builder(params);
    if (sahKdTree.depth > U(triangles.size())) {
        return false;
    }
    qCDebug(builderLog) << QStringLiteral("sahKdTree.depth = %1").arg(sahKdTree.depth);
    return true;
}

}  // namespace

bool builder::buildSceneFromFile(QString sceneFileName, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth)
{
    SceneLoader sceneLoader;
    if (!sceneLoader.load(sceneFileName)) {
        return false;
    }
    return buildSceneFromTriangles(sceneLoader.triangle, emptinessFactor, traversalCost, intersectionCost, maxDepth);
}

bool builder::buildSceneFromFileOrCache(QString sceneFileName, QString cachePath, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth)
{
    SceneLoader sceneLoader;
    if (!sceneLoader.cachingLoad(sceneFileName, cachePath)) {
        return false;
    }
    return buildSceneFromTriangles(sceneLoader.triangle, emptinessFactor, traversalCost, intersectionCost, maxDepth);
}
