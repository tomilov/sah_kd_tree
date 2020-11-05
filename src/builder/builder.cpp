#include "builder.hpp"

#include <sah_kd_tree/sah_kd_tree.hpp>
#include <scene_loader/scene_loader.hpp>

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
    SahKdTree::build({}, triangles.data(), triangles.data() + triangles.size());
    return true;
}
