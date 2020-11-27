#pragma once

namespace SahKdTree
{
using I = int;
using U = unsigned int;
using F = float;

struct Vertex
{
    F x, y, z;
};

struct Triangle
{
    Vertex a, b, c;
};
}  // namespace SahKdTree
