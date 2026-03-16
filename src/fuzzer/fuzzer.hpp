#pragma once

#include <tuple>
#include <type_traits>
#include <vector>

namespace fuzzer
{

using U = unsigned int;
using F = float;

struct Params
{
    F emptinessFactor;
    F traversalCost;
    F intersectionCost;
    U maxTreeDepth;
};
static_assert(std::is_standard_layout_v<Params>);
static_assert(std::is_trivially_copyable_v<Params>);

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
static_assert(std::is_standard_layout_v<Vertex>);
static_assert(std::is_trivially_copyable_v<Vertex>);

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
static_assert(std::is_standard_layout_v<Triangle>);
static_assert(std::is_trivially_copyable_v<Triangle>);

void testOneInput(
    const Params & p,
    const std::vector<Triangle> & t);

}  // namespace fuzzer
