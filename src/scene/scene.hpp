#pragma once

#include <glm/glm.hpp>

#include <memory>

#include <cstddef>

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
    size_t triangleCount = 0;
    std::unique_ptr<Triangle[]> triangles;

    void resize(size_t newTraingleCount) SCENE_EXPORT;
};

}  // namespace scene
