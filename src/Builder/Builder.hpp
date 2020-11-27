#pragma once

#include <QString>

bool build(QString sceneFileName, bool useCache, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth);
