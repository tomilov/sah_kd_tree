#include "builder.hpp"

#include <sah_kd_tree/builder.hpp>
#include <sah_kd_tree/sah_kd_tree.hpp>
#include <sah_kd_tree/types.hpp>
#include <scene_loader/scene_loader.hpp>

#include <thrust/device_vector.h>

bool build(QString sceneFileName, bool useCache)
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
    params.emptinessFactor = 0.8f;
    params.traversalCost = 2.0;
    params.intersectionCost = 1.0f;
    SahKdTree::SahKdTree sahKdTree = builder(params);
    (void)sahKdTree;
    return true;
}
