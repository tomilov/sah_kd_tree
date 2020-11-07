#pragma once

#include <limits>

namespace SahKdTree
{
using I = int;
using U = unsigned int;
using F = float;

struct Params
{
    F emptinessFactor = 0.8f;   // (0, 1]
    F traversalCost = 2.0f;     // (0, inf)
    F intersectionCost = 1.0f;  // (0, inf)
    U maxDepth = std::numeric_limits<U>::max();
};
}  // namespace SahKdTree
