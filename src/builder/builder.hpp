#pragma once

#include "builder/builder_export.h"

#include <QString>

namespace builder BUILDER_EXPORT
{
bool build(QString sceneFileName, bool useCache = false, QString cachePath = {}, float emptinessFactor = 0.0f, float traversalCost = 0.0f, float intersectionCost = 0.0f, int maxDepth = 0);
}
