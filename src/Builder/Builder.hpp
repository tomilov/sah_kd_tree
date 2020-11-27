#pragma once

#include <QString>

bool build(QString sceneFileName, bool useCache, QString cachePath, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth);
