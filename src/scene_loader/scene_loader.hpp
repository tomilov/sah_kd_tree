#pragma once

#include <scene_loader/scene_loader_export.h>

#include <scene/scene.hpp>

#include <QtCore/QDir>
#include <QtCore/QFileInfo>

#include <tuple>

namespace scene_loader
{
struct SCENE_LOADER_EXPORT SceneLoader
{
    bool load(QFileInfo sceneFileInfo);

    QFileInfo getCacheEntryFileInfo(QFileInfo sceneFileInfo, QDir cacheDir);

    bool loadFromCache(QFileInfo cacheEntryFileInfo);
    bool storeToCache(QFileInfo cacheEntryFileInfo);

    bool cachingLoad(QFileInfo sceneFileInfo, QDir cacheDir);

    scene::Scene scene;
};
}  // namespace scene_loader
