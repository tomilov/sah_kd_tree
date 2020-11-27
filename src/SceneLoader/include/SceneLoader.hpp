#pragma once

#include <SahKdTree/Types.hpp>

#include <QtCore>

Q_DECLARE_LOGGING_CATEGORY(sceneLoader)

struct SceneLoader
{
    bool load(QFileInfo sceneFileInfo);

    QFileInfo getCacheEntryFileInfo(QFileInfo sceneFileInfo, QString cachePath);

    bool loadFromCache(QFileInfo cacheEntryFileInfo);
    bool storeToCache(QFileInfo cacheEntryFileInfo);

    bool cachingLoad(QString fileName, QString cachePath);

    QVector<SahKdTree::Triangle> triangle;
};
