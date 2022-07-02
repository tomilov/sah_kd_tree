#pragma once

#include <scene_loader/geometry_types.hpp>

namespace builder
{
bool buildSceneFromTriangles(const scene_loader::Triangle * triangleBegin, const scene_loader::Triangle * triangleEnd, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth);
}  // namespace builder
