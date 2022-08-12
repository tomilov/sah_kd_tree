#pragma once

#include <builder/builder_export.h>

#include <QLoggingCategory>
#include <QString>

namespace builder
{
Q_DECLARE_LOGGING_CATEGORY(builderLog)

bool buildSceneFromFile(QString sceneFileName, float emptinessFactor = 0.0f, float traversalCost = 0.0f, float intersectionCost = 0.0f, int maxDepth = 0) BUILDER_EXPORT;
bool buildSceneFromFileOrCache(QString sceneFileName, QString cachePath, float emptinessFactor = 0.0f, float traversalCost = 0.0f, float intersectionCost = 0.0f, int maxDepth = 0) BUILDER_EXPORT;
}  // namespace builder
