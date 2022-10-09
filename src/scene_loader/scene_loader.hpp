#pragma once

#include <scene_loader/geometry_types.hpp>
#include <scene_loader/scene_loader_export.h>

#include <QtCore/QVector>
#include <QtCore/QFileInfo>
#include <QtCore/QDir>

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

    QVector<Triangle> triangles;
};
}  // namespace scene_loader
