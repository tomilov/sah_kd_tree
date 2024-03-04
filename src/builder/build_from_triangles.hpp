#pragma once

#include <scene_data/scene_data.hpp>

namespace builder
{
bool buildSceneFromTriangles(const scene_data::Triangle * triangleBegin, const scene_data::Triangle * triangleEnd, float emptinessFactor, float traversalCost, float intersectionCost, int maxDepth);
}  // namespace builder
