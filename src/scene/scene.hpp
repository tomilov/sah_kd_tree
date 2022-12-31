#pragma once

#include <glm/glm.hpp>

#include <vector>

#include <scene/scene_export.h>

namespace scene
{

using Vertex = glm::vec3;

struct Triangle
{
    Vertex a, b, c;
};

struct Scene
{
    std::vector<Triangle> triangles;
};

}  // namespace scene
