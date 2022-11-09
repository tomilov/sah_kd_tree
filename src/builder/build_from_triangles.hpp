#pragma once

#include <scene/scene.hpp>

namespace builder
{
bool buildSceneFromTriangles(const scene::Triangle * triangleBegin, const scene::Triangle * triangleEnd, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth);
}  // namespace builder
