#pragma once

#include <SahKdTree/types.hpp>

#include <QtCore>

Q_DECLARE_LOGGING_CATEGORY(sceneLoader)

struct SceneLoader
{
    bool load(QFileInfo sceneFileInfo);

    QFileInfo getCacheEntryFileInfo(QFileInfo sceneFileInfo);

    bool loadFromCache(QFileInfo cacheEntryFileInfo);
    bool storeToCache(QFileInfo cacheEntryFileInfo);

    bool cachingLoad(QString fileName);

    QVector<SahKdTree::Triangle> triangle;
};