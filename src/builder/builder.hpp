#pragma once

#include "builder/builder_export.h"

#include <QString>
#include <QLoggingCategory>

Q_DECLARE_LOGGING_CATEGORY(builderLog)

namespace builder BUILDER_EXPORT
{
bool buildSceneFromFile(QString sceneFileName, float emptinessFactor = 0.0f, float traversalCost = 0.0f, float intersectionCost = 0.0f, int maxDepth = 0);
bool buildSceneFromFileOrCache(QString sceneFileName, QString cachePath, float emptinessFactor = 0.0f, float traversalCost = 0.0f, float intersectionCost = 0.0f, int maxDepth = 0);
}
