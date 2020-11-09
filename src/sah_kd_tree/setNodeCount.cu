#include "utility.cuh"

#include <sah_kd_tree/projection.hpp>

void SahKdTree::Projection::setNodeCount(U nodeCount)
{
    node.min.resize(nodeCount);
    node.max.resize(nodeCount);
}
