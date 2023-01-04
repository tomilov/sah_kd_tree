#pragma once

#include <QtCore/QString>

#include <builder/builder_export.h>

namespace builder
{
bool buildSceneFromFile(QString sceneFileName, float emptinessFactor = 0.0f, float traversalCost = 0.0f, float intersectionCost = 0.0f, int maxDepth = 0) BUILDER_EXPORT;
bool buildSceneFromFileOrCache(QString sceneFileName, QString cachePath, float emptinessFactor = 0.0f, float traversalCost = 0.0f, float intersectionCost = 0.0f, int maxDepth = 0) BUILDER_EXPORT;
}  // namespace builder
