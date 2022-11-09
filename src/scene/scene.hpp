#pragma once

#include <scene/scene_export.h>

#include <glm/glm.hpp>

#include <vector>

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

}
