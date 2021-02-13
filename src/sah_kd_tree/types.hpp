#pragma once

namespace sah_kd_tree
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
}  // namespace sah_kd_tree
