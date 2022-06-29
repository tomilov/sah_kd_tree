#pragma once

#include "sah_kd_tree/types.hpp"
#include "scene_loader/scene_loader_export.h"

#include <QtCore>

Q_DECLARE_LOGGING_CATEGORY(sceneLoaderLog)

struct SCENE_LOADER_EXPORT SceneLoader
{
    bool load(QFileInfo sceneFileInfo);

    QFileInfo getCacheEntryFileInfo(QFileInfo sceneFileInfo, QString cachePath);

    bool loadFromCache(QFileInfo cacheEntryFileInfo);
    bool storeToCache(QFileInfo cacheEntryFileInfo);

    bool cachingLoad(QString fileName, QString cachePath);

    QVector<sah_kd_tree::Triangle> triangle;
};
