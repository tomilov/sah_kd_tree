#include "Builder.hpp"

#include <SahKdTree.hpp>
#include <SceneLoader.hpp>

#include <thrust/device_vector.h>

bool build(QString sceneFileName, bool useCache, QString cachePath, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth)
{
    SceneLoader sceneLoader;
    if (useCache) {
        if (!sceneLoader.cachingLoad(sceneFileName, cachePath)) {
            return false;
        }
    } else {
        if (!sceneLoader.load(sceneFileName)) {
            return false;
        }
    }
    const auto & triangles = sceneLoader.triangle;
    using namespace SahKdTree;
    Builder builder;
    {
        thrust::device_vector<Triangle> deviceTriangles{triangles.cbegin(), triangles.cend()};
        builder.setTriangle(deviceTriangles.data(), deviceTriangles.data() + deviceTriangles.size());
    }  // deviceTriangles.clear() cause link error
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
    Q_UNUSED(sahKdTree)  // TODO(tomilov): make use it somehow eventually!
    return true;
}
