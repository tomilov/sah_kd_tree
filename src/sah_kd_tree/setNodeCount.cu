#include "utility.cuh"

#include <sah_kd_tree/sah_kd_tree.hpp>

void SahKdTree::Projection::setNodeCount(U nodeCount)
{
    node.min.resize(nodeCount);
    node.max.resize(nodeCount);
}
