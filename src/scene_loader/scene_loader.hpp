#pragma once

#include <scene/fwd.hpp>
#include <scene_loader/fwd.hpp>

#include <QtCore/QDir>
#include <QtCore/QFileInfo>
#include <QtCore/QStringList>

#include <tuple>

#include <scene_loader/scene_loader_export.h>

namespace scene_loader
{
struct SCENE_LOADER_EXPORT SceneLoader
{
    bool load(scene::Scene & scene, QFileInfo sceneFileInfo) const;

    QFileInfo getCacheFileInfo(QFileInfo sceneFileInfo, QDir cacheDir) const;

    bool loadFromCache(scene::Scene & scene, QFileInfo cacheFileInfo) const;
    bool storeToCache(scene::Scene & scene, QFileInfo cacheFileInfo) const;

    bool cachingLoad(scene::Scene & scene, QFileInfo sceneFileInfo, QDir cacheDir) const;

    static QStringList getSupportedExtensions();
};
}  // namespace scene_loader
