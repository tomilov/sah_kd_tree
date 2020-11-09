#include "builder.hpp"

#include <sah_kd_tree/builder.hpp>
#include <sah_kd_tree/sah_kd_tree.hpp>
#include <sah_kd_tree/types.hpp>
#include <scene_loader/scene_loader.hpp>

#include <thrust/device_vector.h>

bool build(QString sceneFileName, bool useCache, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth)
{
    SceneLoader sceneLoader;
    if (useCache) {
        if (!sceneLoader.cachingLoad(sceneFileName)) {
            return false;
        }
    } else {
        if (!sceneLoader.load(sceneFileName)) {
            return false;
        }
    }
    const auto & triangles = sceneLoader.triangle;
    SahKdTree::Builder builder;
    {
        thrust::device_vector<SahKdTree::Triangle> deviceTriangles{triangles.cbegin(), triangles.cend()};
        builder.setTriangle(deviceTriangles.data(), deviceTriangles.data() + deviceTriangles.size());
    }  // deviceTriangles.clear() cause link error
    SahKdTree::Params params;
    if (emptinessFactor > 0.0f) {
        params.emptinessFactor = SahKdTree::F(emptinessFactor);
    }
    if (traversalCost > 0.0f) {
        params.traversalCost = SahKdTree::F(traversalCost);
    }
    if (intersectionCost > 0.0f) {
        params.intersectionCost = SahKdTree::F(intersectionCost);
    }
    if (maxDepth > 0) {
        params.maxDepth = SahKdTree::U(maxDepth);
    }
    SahKdTree::SahKdTree sahKdTree = builder(params);
    Q_UNUSED(sahKdTree)  // TODO(tomilov): make use it somehow eventually!
    return true;
}
