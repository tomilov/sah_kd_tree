#include "builder.hpp"

#include <sah_kd_tree/sah_kd_tree.hpp>
#include <scene_loader/scene_loader.hpp>

#include <thrust/system/cuda/vector.h>

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
    const auto & triangles = sceneLoader.triangles;
    thrust::cuda::vector<SahKdTree::Triangle> deviceTriangles{triangles.cbegin(), triangles.cend()};
    SahKdTree::Params params;
    SahKdTree::build(params, deviceTriangles.data(), deviceTriangles.data() + deviceTriangles.size());
    return true;
}
