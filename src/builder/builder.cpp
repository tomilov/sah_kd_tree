#include "builder.hpp"

#include <sah_kd_tree/sah_kd_tree.hpp>
#include <scene_loader/scene_loader.hpp>

bool build(QString sceneFileName, bool useCache)
{
    SceneLoader sceneLoader;
    if (useCache) {
        return sceneLoader.cachingLoad(sceneFileName);
    } else {
        return sceneLoader.load(sceneFileName);
    }
}
