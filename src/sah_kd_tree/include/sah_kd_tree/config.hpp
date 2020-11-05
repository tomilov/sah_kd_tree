#pragma once

#include <limits>

namespace SahKdTree
{
using I = int;
using U = unsigned int;
using F = float;

struct Params
{
    F emptiness_factor = 0.8f;   // (0, 1]
    F traversal_cost = 2.0f;     // (0, inf)
    F intersection_cost = 1.0f;  // (0, inf)
    U max_depth = std::numeric_limits<U>::max();
};
}  // namespace SahKdTree
