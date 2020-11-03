#pragma once

#include <QtCore>

Q_DECLARE_LOGGING_CATEGORY(sceneLoader)

#include <vector_functions.hpp>

using Vertex = float3;

struct Triangle
{
    Vertex A, B, C;
};

struct SceneLoader
{
    bool load(QFileInfo sceneFileInfo);

    QFileInfo getCacheEntryFileInfo(QFileInfo sceneFileInfo);

    bool loadFromCache(QFileInfo cacheEntryFileInfo);
    bool saveToCache(QFileInfo cacheEntryFileInfo);

    bool cachingLoad(QString fileName);

    QVector<Triangle> triangles;
};
