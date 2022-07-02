#pragma once

#include <scene_loader/scene_loader_export.h>

#include <tuple>

namespace scene_loader
{

struct SCENE_LOADER_EXPORT Vertex
{
    float x, y, z;

    bool operator<(const Vertex & rhs) const
    {
        return std::tie(x, y, z) < std::tie(rhs.x, rhs.y, rhs.z);
    }
};

struct SCENE_LOADER_EXPORT Triangle
{
    Vertex a, b, c;

    bool operator<(const Triangle & rhs) const
    {
        return std::tie(a, b, c) < std::tie(rhs.a, rhs.b, rhs.c);
    }
};

}  // namespace scene_loader
