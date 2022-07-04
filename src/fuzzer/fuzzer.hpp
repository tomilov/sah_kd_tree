#pragma once

#include <sah_kd_tree/types.cuh>

#include <limits>
#include <tuple>
#include <vector>

#include <cstdlib>

#define INVARIANT(condition)            \
    do {                                \
        if (!(condition)) std::abort(); \
    } while (false)

namespace fuzzer
{
using sah_kd_tree::F;
using sah_kd_tree::I;
using sah_kd_tree::U;

struct Params
{
    F emptinessFactor;
    F traversalCost;
    F intersectionCost;
    U maxDepth;
};

struct Vertex
{
    F x, y, z;

    bool operator<(const Vertex & rhs) const
    {
        return std::tie(x, y, z) < std::tie(rhs.x, rhs.y, rhs.z);
    }

    bool operator==(const Vertex & rhs) const  // legal for floating-point values obtained by identity operations
    {
        return std::tie(x, y, z) == std::tie(rhs.x, rhs.y, rhs.z);
    }
};

struct Triangle
{
    Vertex a, b, c;

    bool operator<(const Triangle & rhs) const
    {
        return std::tie(a, b, c) < std::tie(rhs.a, rhs.b, rhs.c);
    }

    bool operator==(const Triangle & rhs) const
    {
        return std::tie(a, b, c) == std::tie(rhs.a, rhs.b, rhs.c);
    }
};

void testOneInput(const Params & p, const std::vector<Triangle> & t);
}  // namespace fuzzer
