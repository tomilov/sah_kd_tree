#pragma once

#include <scene/scene.hpp>

#include <QtCore/QDir>
#include <QtCore/QFileInfo>

#include <tuple>

#include <scene_loader/scene_loader_export.h>

namespace scene_loader
{
struct SCENE_LOADER_EXPORT SceneLoader
{
    bool load(QFileInfo sceneFileInfo);

    QFileInfo getCacheFileInfo(QFileInfo sceneFileInfo, QDir cacheDir);

    bool loadFromCache(QFileInfo cacheFileInfo);
    bool storeToCache(QFileInfo cacheFileInfo);

    bool cachingLoad(QFileInfo sceneFileInfo, QDir cacheDir);

    scene::Scene scene;
};
}  // namespace scene_loader
